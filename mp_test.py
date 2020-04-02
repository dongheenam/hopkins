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
        return sigma_t(h) * (R/h)**((p-1)/2)

# mach number at size R
def mach(R) :
    return sigma_t(R) / c_s

# total gas velocity dispersion at size R (sigma_g)
def sigma_gas(R) :
    return np.sqrt(sigma_t(R)**2 + c_s**2 + v_A**2)

# density dispersion squared at size R (sigma_k)
def sigma_dens_squared(R) :
    kappa = kappa_tilde * sigma_gas(R)*np.sqrt(3) / h
    sigma_squared = np.log(1 + b**2*sigma_t(R)**2/(c_s**2 + kappa**2*R**2))
    return sigma_squared

# global dispersion of density smoothed at R
def S(R) :
    integral = integrate.quad(lambda lnk: sigma_dens_squared(np.exp(-lnk)), np.log(1/h*1e-20), np.log(1/R))
    #integral = integrate.quad(lambda k: sigma_dens_squared(1/k)/k, 0, 1/R)
    return integral[0]

# barrier function
def B(R) :
    # rho_crit / rho_0
    dens_ratio = (Q/(2*kappa_tilde)
                 *(1+h/R)
                 *((sigma_gas(R)/sigma_gas(h))**2 * h/R + kappa_tilde**2 * R/h)
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
    #for mach_h, lc in [(0.1,'black'), (1.0, 'blueviolet'), (3.0,'dodgerblue'), (10.0,'skyblue'), (30.0,'springgreen')] :
    for p, lc in [(1.001,'black'), (1.3,'blueviolet'), (1.7,'dodgerblue'), (2.0,'skyblue'), (2.5,'springgreen')] :
    #for Q, lc in [(0.5,'black'), (1,'blueviolet'), (2,'dodgerblue'), (5,'skyblue'), (10,'springgreen')] :
    #for b, lc in [(1/3,'black'), (1/2,'blueviolet'), (1,'dodgerblue')] :

        """ constants """
        mach_h = 10                             # 1D Mach number on scale h
        h = 100 * 3.086e18                      # length scale h (pc > cm)
        c_s = 0.2e5                             # sonic speed (cm/s)
        n_0 = 10.0                              # mean number density (cm^-3)
        mu = 2.34e-24                           # mean mass per particle (g)
        rho_0 = n_0 * mu                        # mean mass density (g cm^-3)
        B_mag = 0                               # magnetic field strength (Gauss)
        v_A = B_mag / np.sqrt(4*np.pi*rho_0)    # Alfven speed (cm/s)

        b = 1.0                                 # turbulence driving parameter
        #p = 2                                   # negative turbulent velocity PS index
        Q = 1                                   # Toomre parameter
        kappa_tilde = np.sqrt(2)                # ratio of epicyclic and orbital freqs

        """ calculation parameters """
        size_start = 1                          # starting scale (= 10^3 h)
        size_end = -4                           # finishing scale (= 10^-4 h)
        n_R = 100                               # resolution

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

        #label= rf"$M={mach_h:.1f}$"
        label = rf"$p={p:.1f}$"
        ax1.plot(Rs/h, Ms/(rho_0*h**3), color=lc, label=label)
        ax2.plot(Rs/h, Ss, color=lc, label=label)
        ax3.plot(Rs/h, np.exp((Bs-Ss/2)), color=lc, label=label)
        ax4.plot((Rs/h)[1:], np.diff(Bs)/np.diff(Ss), color=lc, label=label)
        ax5.plot(Rs/h, Bs/np.sqrt(Ss), color=lc, label=label)

    test_name="p"
    filenames_csv = [f"test/test_{test_name}_{data}.csv" for data in ["logm", "s", "rhocrit", "dbds"]]
    for filename_csv, ax in zip(filenames_csv, [ax1, ax2, ax3, ax4]) :
        x1,y1, x2,y2, x3,y3, x4,y4, x5,y5 = np.genfromtxt(
            filename_csv, delimiter=',', skip_header=2, unpack=True)
        ax.plot(x1,y1, ls='--', color='springgreen')
        ax.plot(x2,y2, ls='--', color='skyblue')
        ax.plot(x3,y3, ls='--', color='dodgerblue')
        ax.plot(x4,y4, ls='--', color='blueviolet')
        ax.plot(x5,y5, ls='--', color='black', label='Hopkins(2013)')

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

    fig1.savefig(f"test/{test_name}_1logM.pdf")
    fig2.savefig(f"test/{test_name}_2S.pdf")
    fig3.savefig(f"test/{test_name}_3rho_crit.pdf")
    fig4.savefig(f"test/{test_name}_4dB_dS.pdf")
    fig5.savefig(f"test/{test_name}_5nu.pdf")
