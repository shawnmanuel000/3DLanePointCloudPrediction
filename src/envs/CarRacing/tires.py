import os
import numpy as np
from ctypes import cdll, c_wchar_p, c_int, c_double, POINTER

N_PER_KG = 9.81
LB_PER_KG = 2.20462
LB_PER_N = LB_PER_KG/N_PER_KG

class BicycleTireModel():
	name = "bicycle"
	def calc_F_Aero(self, Vx, Vy, C, ρ, SA): 
		return (0.5*ρ) * (Vx**2 + Vy**2) * (SA*C)

	def calc_Fz(self, F_ZStatic, F_Z_Aero): 
		return F_ZStatic + 0.5*F_Z_Aero

	def calc_Fy(self, α, μ, C_α, F_Z, tan=np.tan, **kwargs):
		return np.where(np.abs(α) < 3*μ*F_Z/C_α, C_α*tan(α) - C_α**2/(3*μ*F_Z)*np.abs(tan(α))*tan(α) + C_α**3/(27*(μ*F_Z)**2)*(tan(α)**3), μ*F_Z*np.sign(α))

	def calc_Fx(self, μ, F_Y, F_Z): 
		return F_Z * np.abs(np.sqrt(np.maximum(μ**2 - (F_Y / np.maximum(F_Z,1e-8))**2, 1e-8)))

class TRDTireModel(BicycleTireModel):
	name = "trdtire"
	def __init__(self):
		self.cmodel_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "cTireLinux"))
		self.cTire = cdll.LoadLibrary(os.path.join(self.cmodel_folder, "libctire.so"))
		cTireInit = self.cTire.TRD_cTireModel_Initialize
		cTireInit.arguments = [c_wchar_p, c_int]
		cTireInit(bytes(os.path.join(self.cmodel_folder, "TRD_M8000_20Seb1_LF_2018_f02_c01.ctir"), "utf-8"), 0)
		cTireInit(bytes(os.path.join(self.cmodel_folder, "TRD_M8000_20Seb1_RF_2018_f02_c01.ctir"), "utf-8"), 1)
		cTireInit(bytes(os.path.join(self.cmodel_folder, "TRD_M8000_20Seb1_LR_2018_f02_c01.ctir"), "utf-8"), 2)
		cTireInit(bytes(os.path.join(self.cmodel_folder, "TRD_M8000_20Seb1_RR_2018_f02_c01.ctir"), "utf-8"), 3)

	def batch_reset(self, num=1):
		if getattr(self, "num", None) == num: return
		ptr_type = c_double * num
		self.cTireFY = self.cTire.TRD_cTireModel_FY
		self.cTireFY.restype = c_double 
		self.cTireFY.argtypes = [c_int, POINTER(ptr_type), POINTER(ptr_type), POINTER(ptr_type), POINTER(ptr_type), POINTER(ptr_type), POINTER(ptr_type), POINTER(ptr_type), c_int, POINTER(ptr_type)]
		self.ptr_type = ptr_type
		self.num = num

	def calc_Fy_single(self, α, F_Z, index):
		self.batch_reset(int(np.prod(α.shape)))
		α_flat = α.flatten() * 180/np.pi
		P_flat = 30.0*np.ones_like(α_flat)
		Fz_flat = F_Z.flatten() * -LB_PER_N
		ones_flat = np.ones_like(α_flat)
		zeros_flat = np.zeros_like(α_flat)
		α_ptr = self.ptr_type(*α_flat)
		SR_ptr = self.ptr_type(*zeros_flat)
		IA_ptr = self.ptr_type(*zeros_flat)
		FZ_ptr = self.ptr_type(*Fz_flat)
		P_ptr = self.ptr_type(*P_flat)
		LTMY_ptr = self.ptr_type(*ones_flat) 
		LTKY_ptr = self.ptr_type(*ones_flat) 
		FY = self.ptr_type(*zeros_flat)
		self.cTireFY(index, α_ptr, SR_ptr, IA_ptr, FZ_ptr, P_ptr, LTMY_ptr, LTKY_ptr, len(α_ptr), FY)
		Fy = np.array([*FY]) / -LB_PER_N
		return Fy
		
	def calc_Fy(self, α, μ, C_α, F_Z, tan=np.tan, side="F"):
		index = 0 if side == "F" else 2 if side == "R" else None
		shape = α.shape
		num = int(np.prod(shape))
		self.batch_reset(int(np.prod(shape)))
		α_flat = α.flatten()
		Fz_flat = F_Z.flatten()
		FyL = self.calc_Fy_single(α_flat, Fz_flat, index)
		FyR = self.calc_Fy_single(α_flat, Fz_flat, index+1)
		Fy = 0.5*(FyL + FyR).reshape(shape)
		return Fy

all_tire_models = {model.name: model for model in [BicycleTireModel, TRDTireModel]}

if __name__ == "__main__":
	tiremodel = TRDTireModel()
	alphas = np.arange(-7.0, 7.1, 0.1) / 180*np.pi
	Fz = np.ones_like(alphas) / -LB_PER_N
	Fylf = tiremodel.calc_Fy_single(alphas, Fz * -804, 0) * -LB_PER_N
	Fyrf = tiremodel.calc_Fy_single(alphas, Fz * -804, 1) * -LB_PER_N
	Fylr = tiremodel.calc_Fy_single(alphas, Fz * -705, 2) * -LB_PER_N
	Fyrr = tiremodel.calc_Fy_single(alphas, Fz * -705, 3) * -LB_PER_N
	for fylf, fyrf, fylr, fyrr in zip(Fylf, Fyrf, Fylr, Fyrr):
		print(fylf, fyrf, fylr, fyrr)