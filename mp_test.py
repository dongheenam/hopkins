# dependencies
import datetime
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate # for definite integration

import mpltools
from constants import M_SOL, G

""" variables """
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
    #mach_squared_correction = 1 + kappa_tilde**2*mach(h)**2/(h/R)**2
    #sigma_squared = np.log(1 + 0.75*mach(R)**2/mach_squared_correction)
    kappa = kappa_tilde * sigma_gas(R) / h
    sigma_squared = np.log(1 + b**2*sigma_t(R)**2/(c_s**2 + kappa**2*R**2))
    return np.sqrt(sigma_squared)

# window function (of k)
def window(k, R) :
    inside_window = (k < 1/R)
    if inside_window :
        return 1
    else :
        return 0

# global dispersion of density smoothed at R
# here we introduce an arbitrary constant k_0 for numerical stability
def S(R) :
    integral = integrate.quad(lambda lnk: sigma_delta(1/np.exp(lnk))**2, np.log(1/(h*1e20)), np.log(1/R))
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
    ln_rho_crit = (B(R)-S(R)/2) + np.log(rho_0)

    if (R/h > 5e-5) :
        result = 4*np.pi*np.exp(ln_rho_crit)*h**3
        result *= R**2/(2*h**2) + (1+R/h)*np.exp(-R/h) - 1
    else :
        result = 4*np.pi/3 * np.exp(ln_rho_crit + 3*np.log(R))

    if result == 0.0 :
        sys.exit("zero mass encountered!")
    else :
        return result

"""
================================================================================
main
================================================================================
"""
if __name__ == "__main__" :
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    fig4 = plt.figure()
    ax4 = fig4.add_subplot()
    fig5 = plt.figure()
    ax5 = fig5.add_subplot()
    for p in [1.001, 4/3, 5/3, 2, 5/2] :

        """ constants """
        mach_h = 10                             # 1D Mach number on scale h
        h = 800 * 3.086e18                      # length scale h (pc > cm)
        r_small = 1.0
        c_s = 0.2e5                             # sonic speed (cm/s)
        n_0 = 1.0                               # mean number density (cm^-3)
        mu = 2.34e-24                           # mean mass per particle (g)
        rho_0 = n_0 * mu                        # mean mass density (g cm^-3)
        B_mag = 0                               # magnetic field strength (Gauss)
        v_A = B_mag / np.sqrt(4*np.pi*rho_0)    # Alfven speed (cm/s)

        b = 1.0                                 # turbulence driving parameter
        #p = 2                                   # negative turbulent velocity PS index
        Q = 1                                   # Toomre parameter
        kappa_tilde = np.sqrt(2)                # ratio of epicyclic and orbital freqs
        #kappa = Q*np.pi*G*2*rho_0*h / np.sqrt((mach_h*c_s)**2 + c_s**2 + v_A**2)

        # calculate M_sonic
        if p != 1.0 :
            R_sonic = h*mach_h**(-2/(p-1))
            #M_sonic = M(R_sonic)
            M_sonic = 2/3* c_s**2 * R_sonic/G
        else :
            #R_sonic = h*np.exp(1-mach_h**2)
            M_sonic = M_SOL

        """ calculation parameters """
        size_start = 1                          # starting scale (= 10^3 h)
        size_end = -4                           # finishing scale (= 10^-4 h)
        n_R = 500                               # resolution

        # calculate R
        print("calculating R...")
        logh = np.log10(h)
        Rs = np.power(10, np.linspace(logh+size_start, logh+size_end, n_R))

        # calculate S and B
        print("calculating S, B, and M...")
        Ss, Bs, Ms = np.zeros((3,n_R), dtype=np.float64)
        for i in range(n_R) :
            Ss[i] = S(Rs[i])
            Bs[i] = B(Rs[i])
            Ms[i] = M(Rs[i])

        label = f"p={p:.1f}"
        ax1.plot(Rs/h, Ms/(rho_0*h**3), label=label)
        ax2.plot(Rs/h, Ss, label=label)
        ax3.plot(Rs/h, np.exp((Bs-Ss/2) + np.log(rho_0))/rho_0, label=label)
        ax4.plot((Rs/h)[1:], np.diff(Bs)/np.diff(Ss), label=label)
        ax5.plot(Rs/h, Bs/np.sqrt(Ss), label=label)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(left=1e-4, right=1e1)
    ax1.set_ylim(bottom=1e-5, top=1e1)
    fig1.legend()

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(left=1e-4, right=1e1)
    ax2.set_ylim(bottom=1e-4, top=1e2)
    fig2.legend()


    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(left=1e-4, right=1e1)
    ax3.set_ylim(bottom=1, top=1e5)
    fig3.legend()

    ax4.set_xscale("log")
    ax4.set_xlim(left=1e-4, right=1e1)
    ax4.set_ylim(bottom=-2, top=6)
    fig4.legend()

    ax5.set_xscale("log")
    ax5.set_xlim(left=1e-4, right=1e1)
    ax5.set_ylim(bottom=1, top=10)
    fig5.legend()

    fig1.savefig("test/p_1logM.pdf")
    fig2.savefig("test/p_2S.pdf")
    fig3.savefig("test/p_3rho_crit.pdf")
    fig4.savefig("test/p_4dB_dS.pdf")
    fig5.savefig("test/p_5nu.pdf")
