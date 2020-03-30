import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

mach_h = 30                             # 1D Mach number on scale h
p = 2                                   # negative turbulent velocity PS index
Q = 1                                   # Toomre parameter
kappa_tilde = np.sqrt(2)                # ratio of epicyclic and orbital freqs

h = 1.0
rho_0 = 1.0
beta = 0.5
v_A = np.sqrt(1/beta)
c_s = 1.0
M_SOL = 1.0
G = kappa_tilde * ( (mach_h**2 + 1)*c_s**2 + v_A**2 )/(np.sqrt(2)*np.pi*Q)

R_sonic = h*mach_h**(-2/(p-1))
M_sonic = 2/3* c_s**2 * R_sonic/G

# turbulent velocity dispersion smoothed on R (v_t)
def sigma_t(R) :
    if R == h :
        return mach_h * c_s
    else :
        if p != 1.0 :
            return sigma_t(h) * (R/h)**((p-1)/2)
        else :
            return sigma_t(h) * np.sqrt(np.log(R/r_small)/np.log(h/r_small))

# mach number at size R
def mach(R) :
    return sigma_t(R) / c_s

# total gas velocity dispersion at size R (sigma_g)
def sigma_gas(R) :
    return np.sqrt(sigma_t(R)**2 + c_s**2 + v_A**2)

# density dispersion at size R (sigma_k)
def sigma_delta(R) :
    mach_squared_correction = 1 + kappa_tilde**2*mach(h)**2/(h/R)**2
    sigma_squared = np.log(1 + 0.75*mach(R)**2/mach_squared_correction)
    return np.sqrt(sigma_squared)

def S(R) :
    integral = integrate.quad(lambda lnk: sigma_delta(1/np.exp(lnk))**2, -10, np.log(1/R))
    return integral[0]

# barrier function
def B(R) :
    # rho_crit / rho_0
    dens_ratio = (Q/(2*kappa_tilde)
                 *(1+h/R)
                 *(sigma_gas(R)**2/sigma_gas(h)**2 * h/R + kappa_tilde**2 * R/h)
                 )
    return np.log(dens_ratio) + S(R)/2

# mass of collapsing region with size R
def M(R) :
    ln_rho_crit = (B(R)-S(R)/2) + np.log(rho_0) - np.log(M_sonic)

    if (R/h > 5e-5) :
        result = 4*np.pi*np.exp(ln_rho_crit)*h**3
        result *= R**2/(2*h) + (1+R/h)*np.exp(-R/h) - 1
    else :
        result = 4*np.pi/3 * np.exp(ln_rho_crit + 3*np.log(R))
    return result

if __name__ == "__main__" :
    Rs = np.power(10,np.arange(2,-15,-1, dtype=float))
    for R in Rs :
        print((R, S(R), B(R), M(R)))
