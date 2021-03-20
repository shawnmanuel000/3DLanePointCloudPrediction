import os
import sys
import torch
import inspect
import numpy as np
import itertools as it
import pyquaternion as pyq
from collections import OrderedDict
from mlagents_envs import logging_util
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from src.envs.CarRacing.UnitySimulator.unity_gym import UnityToGymWrapper
from src.envs.CarRacing.dynamics import CarDynamics, CarState, DELTA_T, USE_DELTA
from src.envs.CarRacing.viewer import PathAnimator
from src.envs.CarRacing.track import Track, track_name
from src.envs.CarRacing.tires import all_tire_models
from src.envs.CarRacing.cost import CostModel, USE_TEMP
from src.envs.CarRacing.ref import RefDriver
from src.envs.Gym import gym
np.set_printoptions(precision=3, sign=" ")

with_tracks = {"sebring": False, "curve": True}
delta_ts = {"sebring": DELTA_T, "curve": 0.1}
use_temp = {"sebring": USE_TEMP, "curve": True}
use_delta = {"sebring": USE_DELTA, "curve": False}

tracks = {"curve": ["curve1","curve2","curve3","curve4","curve5"],"cubic": ["cubic1","cubic2","cubic3","cubic4","cubic5"]}

def extract_track_name(env_name):
	fields = env_name.split("-")
	names = fields[1:-1]
	return names[0] if names else track_name

def extract_tire_name(env_name):
	fields = env_name.split("-")
	names = fields[1:-1]
	return names[1] if len(names)>1 else None

def extract_pos(state): return state[...,[0,1]]

def rotate_path(state, refstates):
	refstates[...,4:-1] = 0
	s = CarRacingV1.observation_spec(state[...,None,:])
	rs = CarRacingV1.observation_spec(refstates)
	refrot = np.pi/2 - s.ψ
	X = np.cos(refrot)*(rs.X-s.X) - np.sin(refrot)*(rs.Y-s.Y)
	Y = np.sin(refrot)*(rs.X-s.X) + np.cos(refrot)*(rs.Y-s.Y)
	ψ = np.arctan2(np.sin(rs.ψ-s.ψ), np.cos(rs.ψ-s.ψ))
	Vx = rs.Vx - s.Vx
	timediff = rs.realtime - s.realtime
	relref = CarState(X=X, Y=Y, ψ=ψ, Vx=Vx).observation()
	relref = np.concatenate([relref, timediff[...,None]],-1)
	return np.concatenate([relref, refstates[...,relref.shape[-1]:]], -1)

class EnvMeta(type):
	def __new__(meta, name, bases, class_dict):
		cls = super().__new__(meta, name, bases, class_dict)
		gym.register(cls.name, entry_point=cls)
		for track in os.listdir(os.path.join(os.path.dirname(__file__), "CarRacing", "spec", "tracks")):
			track_name = os.path.splitext(track)[0]
			fields = cls.name.split("-")
			fields.insert(1, track_name)
			gym.register("-".join(fields), entry_point=lambda: cls("-".join(fields)))
			fields.insert(2, "")
			for tire_name in all_tire_models:
				fields[2] = tire_name
				gym.register("-".join(fields), entry_point=lambda: cls("-".join(fields)))
		return cls

class CarRacingV1(gym.Env, metaclass=EnvMeta):
	name = "CarRacing-v1"
	dynamics_norm = np.concatenate([CarDynamics.dynamics_norm, [100]])
	dynamics_somask = np.concatenate([CarDynamics.dynamics_somask, [0]])
	def __init__(self, env_name="CarRacing-v1", max_time=None, delta_t=DELTA_T, withtrack=True):
		self.track_name = extract_track_name(env_name)
		self.tire_name = extract_tire_name(env_name)
		self.dynamics = CarDynamics(self.tire_name)
		self.withtrack = with_tracks.get(self.track_name, withtrack)
		self.use_delta = use_delta.get(self.track_name, True)
		self.init_track(self.track_name, max_time, delta_t)
		self.action_space = self.dynamics.action_space
		self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.reset(sample_track=False).shape)
		self.spec = gym.envs.registration.EnvSpec(self.name, max_episode_steps=int(self.max_time/self.delta_t))

	def init_track(self, track_name, max_time=None, delta_t=DELTA_T):
		self.ref = RefDriver(track_name)
		self.track = Track(track_name)
		self.delta_t = delta_ts.get(track_name, delta_t)
		self.max_time = self.ref.max_time if max_time is None else max_time
		self.cost_model = CostModel(self.track, self.ref, self.max_time, self.delta_t)

	def reset(self, train=True, sample_track=False):
		if sample_track: self.init_track(np.random.choice(tracks.get(self.track_name,[self.track_name])))
		self.time = 0
		self.realtime = 0.0
		self.action = np.zeros(self.action_space.shape)
		self.dynamics.reset(self.ref.start_pos, self.ref.start_vel)
		self.state_spec, state = self.observation()
		self.info = {"ref":{}, "car":{}}
		self.done = False
		return state

	def step(self, action, device=None, info=True, temp=USE_TEMP):
		self.time += 1
		self.realtime = self.time * self.delta_t
		self.dynamics.step(action, dt=self.delta_t, use_delta=self.use_delta)
		next_state_spec, next_state = self.observation()
		pos = np.stack([next_state_spec.X, next_state_spec.Y], -1)
		temp = use_temp.get(self.track.track_name, temp)
		reward = -self.cost_model.get_cost(next_state_spec, self.state_spec, self.time, temp)
		ind, trackdist = self.track.get_nearest(pos)
		done = np.logical_or(trackdist > 40.0, self.done)
		if temp: 
			done = np.logical_or(self.ref.get_time(pos, next_state_spec.S) >= self.max_time, done)
			done = np.logical_or(next_state_spec.Vx < 2.0, done)
		self.done = np.logical_or(self.realtime >= self.max_time, done)
		self.info = self.get_info(reward, action) if info else {"ref":{}, "car":{}}
		self.state_spec = next_state_spec
		return next_state, reward, self.done, self.info

	def render(self, mode="human", **kwargs):
		if not hasattr(self, "viewer"): self.viewer = PathAnimator(interactive=mode!="video", dt=self.delta_t)
		ref_spec = self.ref.state(self.realtime)
		pos = np.stack([self.state_spec.X, self.state_spec.Y], -1)
		refpos = np.stack([ref_spec.X, ref_spec.Y], -1)
		car = np.stack([pos, pos + np.array([np.cos(self.state_spec.ψ ), np.sin(self.state_spec.ψ)])])
		ref = np.stack([refpos, refpos + np.array([np.cos(ref_spec.ψ), np.sin(ref_spec.ψ)])])
		if self.withtrack:
			state = self.observation()[1][...,:self.dynamics_size]*self.dynamics_norm
			reftime = self.ref.get_time(pos)/self.delta_t
			refstates = self.ref.get_sequence(reftime, 10, dt=self.delta_t)
			relrefs = rotate_path(state, refstates)
			kwargs["path"] = extract_pos(relrefs[0]) + pos
		return self.viewer.animate_path(self.track, pos, [car, ref], info=self.info, **kwargs)

	def observation(self, carstate=None):
		dyn_state = self.dynamics.observation(carstate)
		realtime = np.expand_dims(self.realtime, axis=-1)
		state = np.concatenate([dyn_state, realtime], axis=-1)
		self.dynamics_size = state.shape[-1]
		spec = self.observation_spec(state)
		if self.withtrack:
			pos = dyn_state[...,[0,1]]
			reftime = self.ref.get_time(pos)/self.delta_t
			refstates = self.ref.get_sequence(reftime, 10, dt=self.delta_t)
			refpath = rotate_path(state, refstates)[0]
			veldiff = refpath[...,3]
			yawdiff = refpath[...,2]
			timediff = refpath[...,-1]
			path = extract_pos(refpath)
			path = np.reshape(path, [*path.shape[:-2], -1]) #self.track.get_path(dyn_state[...,[0,1]], heading=spec.ψ)
			state = np.concatenate([state/self.dynamics_norm, path, veldiff, yawdiff, timediff], -1)
		return spec, state

	@staticmethod
	def observation_spec(observation):
		dyn_state = observation[...,:-1]
		dyn_spec = CarState.observation_spec(dyn_state)
		realtime = observation[...,-1]
		dyn_spec.realtime = realtime
		return dyn_spec

	def set_state(self, state, device=None, times=None):
		if isinstance(state, torch.Tensor): state = state.cpu().numpy()
		dyn_state = state[...,:-1]
		self.dynamics.set_state(dyn_state, device=device)
		self.realtime = state[...,-1] if times is None else times*self.delta_t 
		self.state_spec = self.observation()[0]
		self.time = self.realtime / self.delta_t

	def get_info(self, reward, action):
		dynspec = self.dynamics.state
		refspec = self.ref.state(self.realtime)
		refaction = self.ref.action(self.realtime, self.delta_t)
		reftime = self.ref.get_time(np.stack([dynspec.X, dynspec.Y], -1), dynspec.S)
		carinfo = info_stats(dynspec, reftime, reward, action)
		refinfo = info_stats(refspec, self.realtime, 0, refaction)
		info = {"ref": refinfo, "car": carinfo}
		return info

	def close(self, path=None):
		if hasattr(self, "viewer"): self.viewer.close(path)
		self.closed = True

def info_stats(state, realtime, reward, action):
	turn_rate = action[...,0]
	pedal_rate = action[...,1]
	info = {
		"Time": f"{realtime:7.2f}",
		"Pos": f"{{'X':{justify(state.X)}, 'Y':{justify(state.Y)}}}",
		"Vel": f"{{'X':{justify(state.Vx)}, 'Y':{justify(state.Vy)}}}",
		"Speed": np.round(state.info["V"], 4),
		"Dist": np.round(state.S, 4),
		"Yaw angle": np.round(state.ψ, 4),
		"Yaw vel": np.round(state.ψ̇, 4),
		"Beta": np.round(state.info["β"], 4),
		"Alpha": f"{{'F':{justify(state.info['αf'])}, 'R':{justify(state.info['αr'])}}}",
		"Fz": f"{{'F':{justify(state.info['FzF'])}, 'R':{justify(state.info['FzR'])}}}",
		"Fy": f"{{'F':{justify(state.info['FyF'])}, 'R':{justify(state.info['FyR'])}}}",
		"Fx": f"{{'F':{justify(state.info['FxF'])}, 'R':{justify(state.info['FxR'])}}}",
		"Steer angle": np.round(state.δ, 4),
		"Pedals": np.round(state.pedals, 4),
		"Reward": np.round(reward, 4),
		"Action": f"{{'Trn':{justify(turn_rate)}, 'ped':{justify(pedal_rate)}}}"
	}
	return info

def justify(num): return str(np.round(num, 3)).rjust(10,' ')

class CarRacingV2(gym.Env, metaclass=EnvMeta):
	name = "CarRacing-v2"
	def __new__(cls, **kwargs):
		cls.id = getattr(cls, "id", 0)+1
		return super().__new__(cls)

	def __init__(self, max_time=500, pixels=False):
		logging_util.set_log_level(logging_util.ERROR)
		root = os.path.dirname(os.path.abspath(__file__))
		sim_file = os.path.abspath(os.path.join(root, "UnitySimulator", sys.platform, "CircuitRacing"))
		self.channel = EngineConfigurationChannel()
		logging_util.set_log_level(logging_util.ERROR)
		unity_env = UnityEnvironment(file_name=sim_file, side_channels=[self.channel], worker_id=self.id + np.random.randint(10000, 20000))
		self.scale_sim = lambda s: self.channel.set_configuration_parameters(width=50*int(1+9*s), height=50*int(1+9*s), quality_level=int(1+3*s), time_scale=int(1+9*(1-s)))
		self.env = UnityToGymWrapper(unity_env, use_visual=pixels)
		self.cost_model = CostModel()
		self.action_space = self.env.action_space
		self.cost_queries = list(it.product(np.linspace(-2,2,5), [0], np.linspace(0,4,5)))
		self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.observation()[0].shape)
		self.src = '\t'.join([line for line in open(os.path.abspath(__file__), 'r')][47:58])
		self.max_time = max_time
		self.reset()

	def reset(self, idle_timeout=10, train=True):
		self.time = 0
		self.scale_sim(0)
		self.idle_timeout = idle_timeout if isinstance(idle_timeout, int) else np.Inf
		state, self.spec = self.observation()
		return state

	def step(self, action):
		self.time += 1
		next_state, reward, done, info = self.env.step(action)
		idle = next_state[29]
		done = done or idle>self.idle_timeout or self.time > self.max_time
		next_state, next_spec = self.observation(next_state)
		terminal = -(1-self.time/self.max_time)*int(done)
		reward = -self.cost_model.get_cost(next_spec, self.spec) + terminal
		self.spec = next_spec
		return next_state, reward, done, info

	def render(self, mode=None, **kwargs):
		self.scale_sim(1)
		return self.env.render(mode=mode, **kwargs)

	@staticmethod
	def dynamics_spec(state):
		pos = state[...,:3]
		vel = state[...,3:6]
		angvel = state[...,6:9]
		refrot = state[...,9:13]
		fl_drive = state[...,13:17] # steer angle, motor torque, brake torque, rpm
		fr_drive = state[...,17:21]
		rl_drive = state[...,21:25]
		rr_drive = state[...,25:29]
		idle = state[...,29:30]
		steer_angle = fl_drive[...,0:1]
		rpm = np.array([x[...,-1] for x in [fl_drive, fr_drive, rl_drive, rr_drive]])
		spec = {"pos":pos, "vel":vel, "angvel":angvel, "refrot":refrot, "steer_angle":steer_angle, "rpm":rpm, "idle":idle}
		return spec

	def track_spec(self, state):
		spec = self.dynamics_spec(state)
		quat = pyq.Quaternion(spec["refrot"])
		points = np.array([(x,z,y) for x,y,z in [quat.rotate(p) for p in self.cost_queries]])
		path = np.array(self.cost_model.track.get_path(spec["pos"]))
		# costs = self.cost_model.get_point_cost(spec["pos"]+points, transform=False)
		costs = np.min(np.sqrt(np.sum(np.power(points[:,None,:]-path[None,:,:],2),-1)),-1)/10
		spec.update({"costs":costs})
		return spec

	def observation(self, state_in=None):
		state = self.env.reset() if state_in is None else state_in
		spec = self.track_spec(state)
		dynamics_keys, _ = self.dynamics_keys()
		values = list(map(spec.get, dynamics_keys))
		dynamics_lens = list(map(len, values))
		self.dynamics_size = sum(dynamics_lens[:4])
		observation = np.concatenate(values, -1)
		obs_dot = observation - self.obs if state_in is not None else np.zeros_like(observation)
		self.obs = observation
		spec["dot"] = obs_dot
		return np.concatenate([observation, obs_dot],-1), spec

	@staticmethod
	def dynamics_keys():
		keys = ["pos", "vel", "angvel", "refrot", "steer_angle", "idle", "costs"]
		lens = [3, 3, 3, 4, 1, 1, 25]
		return keys, np.cumsum(lens)

	@staticmethod
	def observation_spec(observation):
		keys, splits = CarRacing.dynamics_keys()
		spec = {k:v for k,v in zip([*keys, "dot"], np.split(observation,splits,-1))}
		return spec

	def close(self):
		if not hasattr(self, "closed"): self.env.close()
		self.closed = True
