import os
import sys
import inspect
import numpy as np
import itertools as it
import pyquaternion as pyq
from types import SimpleNamespace
from collections import OrderedDict
# from src.utils.config import Config
try:
	from src.envs.CarRacing.tires import all_tire_models, BicycleTireModel
	from src.envs.Gym import gym
except:
	from tires import all_tire_models, BicycleTireModel
	import gym


USE_DELTA = False
DELTA_T = 0.02
TURN_SCALE = 0.05
PEDAL_SCALE = 1.0
TURN_LIMIT = 0.025
PEDAL_LIMIT = 1.0

constants = SimpleNamespace(
	m = 1370.0, 			# Mass (kg)
	I_zz = 4453.0, 			# Inertia (kg m^2)
	l_f = 1.293, 			# Distance from CG to front axle (m)
	l_r = 1.475, 			# Distance from CG to rear axle (m)
	δ_ratio = 17.85, 		# Steer ratio
	F_ZFStatic = 3577.0, 	# F Static tire normal force (N)
	F_ZRStatic = 3136.0, 	# R Static tire normal force (N)
	ρ = 1.205, 				# Air density (kg/m^3)
	SA = 2.229, 			# Surface area (m^2)
	C_LF = 0.392, 			# Coefficient of front down force
	C_LR = 0.918, 			# Coefficient of rear down force
	C_D = 0.6, 				# Coefficient of drag
	C_αf = 312631.0, 		# Front tire cornering stiffness (N/rad)
	C_αr = 219079.0, 		# Rear tire cornering stiffness (N/rad)
	μ_f = 1.612, 			# Front tire friction
	μ_r = 1.587, 			# Rear tire friction
	Mz = 0,					# Tire aligning torque (Nm)
	F_RR = 0,				# Tire rolling resistance force (N)
	F_YAero = 0,			# Aero side force (N)
	M_ZAero = 0,			# Aero yaw moment (Nm)
)

def clamp(x, r):
	return np.clip(x, -r, r)

class CarState():
	def __init__(self, *args, **kwargs):
		self.update(*args, **kwargs)

	def update(self,X=None,Y=None,ψ=None,Vx=None,Vy=None,S=None,ψ̇=None,δ=None,pedals=None,info={}):
		givens = [x for x in [X,Y,ψ,Vx,Vy,S,ψ̇,δ,pedals] if x is not None]
		default = lambda: givens[0]*0 if len(givens) > 0 else 0.0
		self.X = X if X is not None else default()
		self.Y = Y if Y is not None else default()
		self.ψ = ψ if ψ is not None else default()
		self.Vx = Vx if Vx is not None else default()
		self.Vy = Vy if Vy is not None else default()
		self.S = S if S is not None else default()
		self.ψ̇  = ψ̇  if ψ̇  is not None else default()
		self.δ = δ if δ is not None else default()
		self.pedals = pedals if pedals is not None else default()
		self.info = info
		self.shape = getattr(default(), "shape", ())
		return self

	def observation(self):
		pos_x = np.expand_dims(self.X, axis=-1)
		pos_y = np.expand_dims(self.Y, axis=-1)
		rot_f = np.expand_dims(self.ψ, axis=-1)
		vel_f = np.expand_dims(self.Vx, axis=-1)
		vel_s = np.expand_dims(self.Vy, axis=-1)
		dist = np.expand_dims(self.S, axis=-1)
		yaw_dot = np.expand_dims(self.ψ̇, axis=-1)
		steer = np.expand_dims(self.δ, axis=-1)
		pedals = np.expand_dims(self.pedals, axis=-1)
		return np.concatenate([pos_x, pos_y, rot_f, vel_f, vel_s, dist, yaw_dot, steer, pedals], axis=-1)

	@staticmethod
	def observation_spec(state):
		pos_x = state[...,0]
		pos_y = state[...,1]
		rot_f = state[...,2]
		vel_f = state[...,3]
		vel_s = state[...,4]
		dist = state[...,5]
		yaw_dot = state[...,6]
		steer = state[...,7]
		pedals = state[...,8]
		info = OrderedDict(V=0, β=0, αf=0, αr=0, FxF=0, FxR=0, FyF=0, FyR=0, FzF=0, FzR=0)
		state_spec = CarState(pos_x,pos_y,rot_f,vel_f,vel_s,dist,yaw_dot,steer,pedals,info)
		return state_spec

	def print(self):
		return f"X: {self.X:4.3f}, Y: {self.Y:4.3f}, ψ: {self.ψ:4.3f}, Vx: {self.Vx:4.3f}, Vy: {self.Vy:4.3f}, δ: {self.δ:4.3f}, pedals: {self.pedals:4.3f}"

class CarDynamics():
	dynamics_norm = np.array([100, 100, 2*np.pi, 50, 50, 5, 1, 0.1, 1])
	dynamics_somask = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1])
	def __init__(self, tire_name=None, *kwargs):
		self.action_space = gym.spaces.Box(-1.0, 1.0, (2,))
		self.tire_model = all_tire_models.get(tire_name, BicycleTireModel)()

	def reset(self, start_pos, start_vel):
		self.state = CarState(X=start_pos[0], Y=start_pos[1], ψ=start_pos[2], Vx=start_vel)
		self.turn_scale = TURN_SCALE
		self.pedal_scale = PEDAL_SCALE
		self.turn_limit = TURN_LIMIT
		self.pedal_limit = PEDAL_LIMIT

	def step(self, action, dt=DELTA_T, integration_steps=1, use_delta=USE_DELTA):
		turn_rate = action[...,0]
		pedal_rate = action[...,1]
		dt = dt/integration_steps
		state = self.state
		vel_avg = 50 if use_delta else 50
		turn_limit = self.turn_limit*(vel_avg/state.Vx)**2
		
		for i in range(integration_steps):
			F_ZF_Aero = self.tire_model.calc_F_Aero(state.Vx, state.Vy, constants.C_LF, constants.ρ, constants.SA)
			F_ZR_Aero = self.tire_model.calc_F_Aero(state.Vx, state.Vy, constants.C_LR, constants.ρ, constants.SA)
			F_X_Aero = self.tire_model.calc_F_Aero(state.Vx, state.Vy, constants.C_D, constants.ρ, constants.SA)
			Fy_scale = np.minimum(np.abs(state.Vx), 1)

			δ = clamp(state.δ + clamp(turn_rate*self.turn_scale-state.δ, turn_limit) * dt, turn_limit) if use_delta else turn_rate*turn_limit
			αf = -np.arctan2((state.Vy + constants.l_f * state.ψ̇),state.Vx) + δ
			αr = -np.arctan2((state.Vy - constants.l_r * state.ψ̇),state.Vx) + 0.0
			
			pedals = clamp(state.pedals + clamp(pedal_rate*self.pedal_scale-state.pedals, self.pedal_limit) * dt, PEDAL_LIMIT) if use_delta else pedal_rate*self.pedal_limit
			acc = np.maximum(pedals, 0)
			accel = np.maximum(-(acc**3)*10523.0 + (acc**2)*12394.0 + (acc)*1920.0, 0)
			brake = np.minimum(pedals, 0)*22500*(self.state.Vx > 0)

			FzF = self.tire_model.calc_Fz(constants.F_ZFStatic, F_ZF_Aero)
			FzR = self.tire_model.calc_Fz(constants.F_ZRStatic, F_ZR_Aero)
			FyF = self.tire_model.calc_Fy(αf, constants.μ_f, constants.C_αf, FzF, side="F") * Fy_scale
			FyR = self.tire_model.calc_Fy(αr, constants.μ_r, constants.C_αr, FzR, side="R") * Fy_scale
			FxF = clamp(brake*0.6, self.tire_model.calc_Fx(constants.μ_f, FyF, FzF))
			FxR = clamp(accel+brake*0.4, self.tire_model.calc_Fx(constants.μ_r, FyR, FzR))
			
			ψ̈ = (1/constants.I_zz) * ((2*FxF * np.sin(δ) + 2*FyF * np.cos(δ)) * constants.l_f - 2*FyR * constants.l_r)
			V̇x = (1/constants.m) * (2*FxF * np.cos(δ) - 2*FyF * np.sin(δ) + 2*FxR - F_X_Aero) + state.ψ̇ * state.Vy
			V̇y = (1/constants.m) * (2*FyF * np.cos(δ) + 2*FxF * np.sin(δ) + 2*FyR) - state.ψ̇ * state.Vx
			
			ψ̇ = state.ψ̇ + ψ̈  * dt
			Vx = state.Vx + V̇x * dt
			Vy = state.Vy + V̇y * dt
			V = np.sqrt(Vx**2 + Vy**2)
			
			β = np.arctan2(Vy,Vx)
			ψ = (state.ψ + ψ̇  * dt)
			X = (state.X + (Vx * np.cos(ψ) - Vy * np.sin(ψ)) * dt)
			Y = (state.Y + (Vy * np.cos(ψ) + Vx * np.sin(ψ)) * dt)
			S = (state.S*1000 + V * dt)/1000

			info = OrderedDict(F_ZF_Aero=F_ZF_Aero, F_ZR_Aero=F_ZR_Aero, F_X_Aero=F_X_Aero, yaw_acc=ψ̈ , vx_dot=V̇x, vy_dot=V̇y,
				V=V, β=β, αf=αf, αr=αr, FxF=FxF/1000, FxR=FxR/1000, FyF=FyF/1000, FyR=FyR/1000, FzF=FzF/1000, FzR=FzR/1000)
			state = state.update(X,Y,ψ,Vx,Vy,S,ψ̇,δ,pedals,info)

		self.state = state

	def observation(self, state):
		state = self.state if state == None else state
		return state.observation()

	@staticmethod
	def observation_spec(state, device=None):
		return CarState.observation_spec(state)

	def set_state(self, state, device=None):
		self.state = self.observation_spec(state, device=device)