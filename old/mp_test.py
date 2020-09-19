# dependencies
import datetime
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate # for definite integration
from scipy import interpolate

import mpltools
from constants import M_SOL, G

""" variables """
# turbulent velocity dispersion smoothed on R (v_t)
def sigma_t(R) :
    return mach_h*c_s * (R/h)**((p-1)/2) * (1+ (R/h)**2)**((1-p)/4)

# total gas velocity dispersion at size R (sigma_g)
def sigma_gas(R) :
    return np.sqrt(sigma_t(R)**2 + c_s**2 + v_A**2)

# density dispersion squared at size R (sigma_k)
def sigma_dens_squared(k) :
    sigma_squared = np.log(1 + b**2*sigma_t(1/k)**2/(c_s**2 + kappa**2/k**2))
    return sigma_squared

# global dispersion of density smoothed at R
def S(R) :
    integral = integrate.quad(lambda lnk: sigma_dens_squared(np.exp(lnk)), np.log(1/h*1e-10), np.log(1/R))
    #integral = integrate.quad(lambda k: sigma_dens_squared(k)/k, 0, 1/R)
    return integral[0]

# rho_crit / rho_0
def dens_ratio_at_crit(R) :
    k = h/R
    dens_ratio = Q/(2*kappa_tilde) * (1+k)
    dens_ratio*= (sigma_gas(R)/sigma_gas(h))**2*k + kappa_tilde**2/k
    return dens_ratio

# barrier function
def B(R) :
    return np.log(dens_ratio_at_crit(R)) + S(R)/2

# mass of collapsing region with size R
def M(R) :
    rho_crit = dens_ratio_at_crit(R) * rho_0

    result = 4*np.pi*rho_crit*h**3
    result *= R**2/(2*h**2) + (1+R/h)*np.exp(-R/h) - 1

    if result == 0.0 :
        sys.exit("zero mass encountered!")
    else :
        return result

def S_small(R) :
    A = b / kappa * sigma_t(100*h)
    #result = -0.5 * re(polylog(2, -A**2/R**2))
    result = A**2 / (2*R**2)
    #result = 3/4* (h/R)**2
    return result

def dens_crit_small(R) :
    return R/h * kappa**2/(4*np.pi*G)

def dens_crit_big(R) :
    return sigma_gas(R)**2 / (4*np.pi*G*R**2)

def M_small(R) :
    #return np.pi * 2*dens_crit_small(R)*h * R**2
    return np.pi*kappa_tilde*Q*rho_0*R**3

def M_big(R) :
    return 4*np.pi/3 * dens_crit_big(R) * R**3


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
    #for p, lc in [(1.001,'black'), (1.3,'blueviolet'), (1.7,'dodgerblue'), (2.0,'skyblue'), (2.5,'springgreen')] :
    #for Q, lc in [(0.5,'black'), (1,'blueviolet'), (2,'dodgerblue'), (5,'skyblue'), (10,'springgreen')] :
    for b, lc in [(1/3,'black'), (1/2,'blueviolet'), (1,'dodgerblue')] :

        """ constants """
        mach_h = 10                             # 1D Mach number on scale h
        h = 100 * 3.086e18                      # length scale h (pc > cm)
        c_s = 0.2e5                             # sonic speed (cm/s)
        n_0 = 1.0                               # mean number density (cm^-3)
        mu = 2.34e-24                           # mean mass per particle (g)
        #rho_0 = n_0 * mu                        # mean mass density (g cm^-3)
        #B_mag = 0                               # magnetic field strength (Gauss)
        #v_A = B_mag / np.sqrt(4*np.pi*rho_0)    # Alfven speed (cm/s)
        beta = np.inf
        v_A = c_s/np.sqrt(beta)

        #b = 1.0                                 # turbulence driving parameter
        p = 2                                   # negative turbulent velocity PS index
        Q = 1                                   # Toomre parameter
        kappa_tilde = np.sqrt(2)                # ratio of epicyclic and orbital freqs
        kappa = kappa_tilde * np.sqrt(c_s**2 + (mach_h*c_s)**2 + v_A**2) / (np.sqrt(2)*h)

        rho_0 = kappa*mach_h*c_s /(2*np.pi*G*Q*h)
        print(f"rho_0 = {rho_0:.6E}")
        print(f"v_t[h] = {sigma_t(h):.6E}")
        print(dens_ratio_at_crit(h))

        """ calculation parameters """
        size_start = 2                          # starting scale (= 10^3 h)
        size_end = -4                           # finishing scale (= 10^-4 h)
        n_R = 100                               # resolution

        # calculate R
        print("calculating R...")
        logh = np.log10(h)
        Rs = np.power(10, np.linspace(logh+size_start, logh+size_end, n_R))

        # calculate S and B
        print("calculating S, B, and M...")
        Ss, Bs, Ms, crits = np.zeros((4,n_R), dtype=np.float64)
        S_smalls, crit_smalls, crit_bigs, M_smalls, M_bigs = np.zeros((5,n_R))
        for i in range(n_R) :
            Ss[i] = S(Rs[i])
            Bs[i] = B(Rs[i])
            Ms[i] = M(Rs[i])
            crits[i] = dens_ratio_at_crit(Rs[i])

            S_smalls[i] = S_small(Rs[i])
            crit_smalls[i] = dens_crit_small(Rs[i])
            crit_bigs[i] = dens_crit_big(Rs[i])
            M_smalls[i] = M_small(Rs[i])
            M_bigs[i] = M_big(Rs[i])

        # calculate dB/dS
        B_tck = interpolate.splrep(Ss, Bs)
        dBdS = interpolate.splev(Ss, B_tck, der=1)

        #label = "asymp." if mach_h == 0.1 else None
        label = "asymp." if p == 1.001 else None
        ax1.plot(Rs/h, M_smalls/(rho_0*h**3), color=lc, ls=':', label=label)
        ax1.plot(Rs/h, M_bigs/(rho_0*h**3), color=lc, ls=':')
        ax2.plot(Rs/h, S_smalls, color=lc, ls=':', label=label)
        ax3.plot(Rs/h, crit_smalls/rho_0, color=lc, ls=':', label=label)
        ax3.plot(Rs/h, crit_bigs/rho_0, color=lc, ls=':')

        #label= rf"$M={mach_h:.1f}$"
        label = rf"$b={b:.1f}$"
        ax1.plot(Rs/h, Ms/(rho_0*h**3), color=lc, label=label)
        ax2.plot(Rs/h, Ss, color=lc, label=label)
        ax3.plot(Rs/h, crits, color=lc, label=label)
        ax4.plot(Rs/h, dBdS, color=lc, label=label)
        ax5.plot(Rs/h, Bs/np.sqrt(Ss), color=lc, label=label)

    test_name="b"
    #tests = ["logm", "s", "rhocrit", "dbds"]
    #axes = [ax1, ax2, ax3, ax4]
    tests = ["s", "dbds"]
    axes = [ax2, ax4]
    filenames_csv = [f"test/test_{test_name}_{data}.csv" for data in tests]
    for filename_csv, ax in zip(filenames_csv, axes) :
        x1,y1, x2,y2, x3,y3 = np.genfromtxt(
            filename_csv, delimiter=',', skip_header=2, unpack=True)
        ax.plot(x1,y1, ls='--', color='dodgerblue')
        ax.plot(x2,y2, ls='--', color='blueviolet')
        ax.plot(x3,y3, ls='--', color='black')
        #ax.plot(x1,y1, ls='--', color='springgreen')
        #ax.plot(x2,y2, ls='--', color='skyblue')
        #ax.plot(x3,y3, ls='--', color='dodgerblue')
        #ax.plot(x4,y4, ls='--', color='blueviolet')
        #ax.plot(x5,y5, ls='--', color='black', label='Hopkins(2013)')

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(left=1e-1, right=1e0)
    ax1.set_ylim(bottom=1e-1, top=1e1)
    ax1.set_xlabel("R[h]")
    ax1.set_ylabel(r"M(R)[$\rho_0 h^3$]")
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(left=1e-4, right=1e1)
    ax2.set_ylim(bottom=1e-4, top=1e2)
    ax2.set_xlabel("R[h]")
    ax2.set_ylabel("S(R)")
    ax2.legend()

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(left=1e-1, right=1e1)
    ax3.set_ylim(bottom=1, top=1e1)
    ax3.set_xlabel("R[h]")
    ax3.set_ylabel(r"$\rho_\mathrm{crit}[\rho_0]$")
    ax3.legend()

    ax4.set_xscale("log")
    ax4.set_xlim(left=1e-4, right=1e1)
    ax4.set_ylim(bottom=-2, top=6)
    ax4.legend()

    ax5.set_xscale("log")
    ax5.set_xlim(left=1e-4, right=1e1)
    ax5.set_ylim(bottom=1, top=10)
    ax5.legend()

    fig1.savefig(f"test/{test_name}_1logM.pdf")
    fig2.savefig(f"test/{test_name}_2S.pdf")
    fig3.savefig(f"test/{test_name}_3rho_crit.pdf")
    fig4.savefig(f"test/{test_name}_4dB_dS.pdf")
    fig5.savefig(f"test/{test_name}_5nu.pdf")
