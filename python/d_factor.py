import numpy as np
from scipy.integrate import quad

rs = 20 #kpc
rE = 8.5 #kpc, distance of Earth to GC
rho_local = 0.3 #GeV/cm^3, local DM density
rho_local_alt = 0.4 #GeV/cm^3, local DM density
rE_alt = 8.127 #kpc, distance of Earth to GC



max_int = 500 #kpc

def rho_UN(r):
	return 1./r/(1+r/rs)**2

rho_N = rho_UN(rE)

def rho(r):
	return rho_UN(r)/rho_N*rho_local

def dist(rfE,b,ell):
	return np.sqrt( (rfE*np.cos(b)*np.cos(ell)-rE)**2 + rfE**2*((np.cos(b)*np.sin(ell))**2 + np.sin(b)**2))

def rho_Earth_frame(rfE,b,ell):
	return rho(dist(rfE,b,ell))

def return_D_factor(ell,b):
	'''
	ell, b in degrees
	'''
	ell_rad = ell*np.pi/180.
	b_rad = b*np.pi/180.
	f = lambda rfE: rho_Earth_frame(rfE,ell_rad,b_rad)
	D_factor_raw = quad(f,0,max_int)[0] #units of kpc*GeV/cm^3

	D_factor_units = D_factor_raw*1e6*3e21 #units of keV/cm^2
	return D_factor_units


def return_sin_theta_lim(E_lines,fluxes,D_factor):
	'''
	everything keV,cm^2,s
	fluxes in units of counts/cm^2/s/sr
	'''
	masses = 2*E_lines
	res = 1e-10*(4*np.pi*masses/D_factor)/1.38e-32*(1/masses)**5*fluxes
	return res

class D_factor_alt:
	def __init__(self,rs=20,rho_local=0.4,r_GC = 8.127,ell_0=0.0):
		'''
		ell, b in degrees
		'''

		self.rs = rs
		self.ell_0= ell_0*np.pi/180.
		self.rho_local = rho_local
		self.r_GC = r_GC

		self._max_int = 500 #kpc

		self._rho_N = self._rho_UN(r_GC)

	def _rho_UN(self,r):
		return 1./r/(1+r/self.rs)**2

	def _rho(self,r):
		return self._rho_UN(r)/self._rho_N*self.rho_local

	def _dist(self,rfE,b,ell):
		rE = self.r_GC
		return np.sqrt( (rfE*np.cos(b)*np.cos(ell-self.ell_0)-rE)**2 + rfE**2*((np.cos(b)*np.sin(ell-self.ell_0))**2 + np.sin(b)**2))
		# return np.sqrt( (rfE*np.cos(b)*np.cos(ell)-rE)**2 + rfE**2*((np.cos(b)*np.sin(ell))**2 + np.sin(b)**2))

	def _rho_Earth_frame(self,rfE,b,ell):
		return self._rho(self._dist(rfE,b,ell))

	def return_D_factor(self,ell,b):
		'''
		ell, b in degrees
		'''
		ell_rad = ell*np.pi/180.
		b_rad = b*np.pi/180.
		f = lambda rfE: self._rho_Earth_frame(rfE,ell_rad,b_rad)
		D_factor_raw = quad(f,0,self._max_int)[0] #units of kpc*GeV/cm^3

		D_factor_units = D_factor_raw*1e6*3e21 #units of keV/cm^2
		return D_factor_units

class D_factor_core:
	def __init__(self,rs=20,rc=1,rho_local=0.4,r_GC = 8.127,ell_0=0.0):
		'''
		ell, b in degrees
		ell_0 in degrees, location of the GC
		'''

		self.rs = rs
		self.rc = rc
		self.rho_local = rho_local
		self.r_GC = r_GC
		self.ell_0 = ell_0*np.pi/180.

		self._max_int = 500 #kpc

		self._rho_N = self._rho_UN(r_GC)

	def _rho_UN(self,r):
		res = 1./r/(1+r/self.rs)**2
		if r<self.rc:
			res = 1./self.rc/(1+self.rc/self.rs)**2
		return res#1./r/(1+r/self.rs)**2

	def _rho(self,r):
		return self._rho_UN(r)/self._rho_N*self.rho_local

	def _dist(self,rfE,b,ell):
		rE = self.r_GC
		return np.sqrt( (rfE*np.cos(b)*np.cos(ell-self.ell_0)-rE)**2 + rfE**2*((np.cos(b)*np.sin(ell-self.ell_0))**2 + np.sin(b)**2))

	def _rho_Earth_frame(self,rfE,b,ell):
		return self._rho(self._dist(rfE,b,ell))

	def return_D_factor(self,ell,b):
		'''
		ell, b in degrees
		'''
		ell_rad = ell*np.pi/180.
		b_rad = b*np.pi/180.
		f = lambda rfE: self._rho_Earth_frame(rfE,ell_rad,b_rad)
		D_factor_raw = quad(f,0,self._max_int)[0] #units of kpc*GeV/cm^3

		D_factor_units = D_factor_raw*1e6*3e21 #units of keV/cm^2
		return D_factor_units

class D_factor_burk:
	def __init__(self,rc=9.0,rho_local=0.4,r_GC = 8.127):
		'''
		ell, b in degrees
		'''

		self.rc = rc
		self.rho_local = rho_local
		self.r_GC = r_GC

		self._max_int = 500 #kpc

		self._rho_N = self._rho_UN(r_GC)

	def _rho_UN(self,r):
		return 1./(1+r/self.rc)/(1.+(r/self.rc)**2)

	def _rho(self,r):
		return self._rho_UN(r)/self._rho_N*self.rho_local

	def _dist(self,rfE,b,ell):
		rE = self.r_GC
		return np.sqrt( (rfE*np.cos(b)*np.cos(ell)-rE)**2 + rfE**2*((np.cos(b)*np.sin(ell))**2 + np.sin(b)**2))

	def _rho_Earth_frame(self,rfE,b,ell):
		return self._rho(self._dist(rfE,b,ell))

	def return_D_factor(self,ell,b):
		'''
		ell, b in degrees
		'''
		ell_rad = ell*np.pi/180.
		b_rad = b*np.pi/180.
		f = lambda rfE: self._rho_Earth_frame(rfE,ell_rad,b_rad)
		D_factor_raw = quad(f,0,self._max_int)[0] #units of kpc*GeV/cm^3

		D_factor_units = D_factor_raw*1e6*3e21 #units of keV/cm^2
		return D_factor_units
