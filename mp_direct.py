# dependencies
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.integrate import quad

import mpltools
from constants import M_SOL, G
from mp_hopkins import mach_h, h, rho_0, v_A, c_s, p, Q, kappa_tilde, \
                       size_start, size_end, use_beta
from mp_hopkins import S, dens_ratio_at_crit, B, M

""" calculation parameters """
n_S = 5000

""" probability functions """

def P_0(x, var=1) :
    if var <= 0 :
        print(f"{var} is not valid variance")
    return 1/np.sqrt(2*np.pi*var) * np.exp(x**2/(-2*var))

""" other functions """
def dB_dS(S) :
    return interpolate.splev(S, B_tck, der=1)

def g_1(S) :
    B = interpolate.splev(S, B_tck, der=0)
    return (2*dB_dS(S) - B/S)*P_0(B, var=S)

def g_2(S1,S2) :
    B1 = interpolate.splev(S1, B_tck, der=0)
    B2 = interpolate.splev(S2, B_tck, der=0)

    part_1 = (B1-B2)/(S1-S2) + B1/S1 - 2*dB_dS(S1)
    part_2 = P_0(B1 - B2*S1/S2, var=((S2-S1)*S1/S2))
    return part_1 * part_2

def calc_H(S1, S2, dS) :
    if S1 != S2 :
        return dS/2 * g_2(S1, S2+dS/2), 0
    else :
        # integration of Taylor expansion near singularity
        #dS1 = dS
        #B1 = interpolate.splev(S1, B_tck, der=0)
        #dB1 = interpolate.splev(S1, B_tck, der=1)
        #ddB1 = interpolate.splev(S1, B_tck, der=2)

        #int1 = np.sqrt(dS1) * (
        #    (-B1+S1*dB1) * (-S1*(dS1+6*S1)+dS1*(B1-S1*dB1)**2) + dS1*S1**3*ddB1
        #) / (3*np.sqrt(2*np.pi)*S1**3)

        int2, error =  quad(lambda Sp: g_2(S1, Sp), S1, S1+dS)
        return int2/2, error

def calc_IMF(Rs, Ss, Bs, Ms, locs_collapse) :
    """ calculate the IMF (dn/dM) based on the collapse prob. dist. """
    dn_dM = np.zeros(len(locs_collapse))
    for i in range(1, len(dn_dM)) :
        r = Rs[i]/h
        if (r > 5e-5) :
            vol = 4*np.pi*h**3 * (r**2/2+(1+r)*np.exp(-r)-1)
        else :
            vol = 4/3*np.pi*Rs[i]**3
        dS_dM = 1/interpolate.splev(Ss[i], M_tck, der=1)
        dn_dM[i] = 1/vol * locs_collapse[i] * np.abs(dS_dM)

    return dn_dM

if __name__ == "__main__" :
    # calculate R
    print("calculating R...")
    logh = np.log10(h)
    Rs = np.power(10, np.linspace(logh+size_start, logh+size_end, 1000))

    # normalisation
    R_sonic = h*mach_h**(-2/(p-1))
    #M_sonic = M(R_sonic)
    M_sonic = 2/3* c_s**2 * R_sonic/G

    # calculate S and B
    print("calculating S and B...")
    Ss, Bs, Ms = np.zeros((3,len(Rs)), dtype=np.float64)
    for i in range(len(Rs)) :
        Ss[i] = S(Rs[i])
        Bs[i] = B(Rs[i])
        Ms[i] = M(Rs[i])

    # truncate the distribution
    S_is_nonzero = (Ss!=0.0)
    Rs = Rs[S_is_nonzero]
    Bs = Bs[S_is_nonzero]
    Ms = Ms[S_is_nonzero]
    Ss = Ss[S_is_nonzero]

    plt.figure()
    plt.plot(Ss, Ms/(rho_0*h**3))
    plt.yscale("log")
    plt.axhline(y=5e-3, color='red')
    plt.savefig("test_SM.pdf")

    plt.figure()
    plt.plot(Ss, Bs)
    plt.plot(Ss, 0.7 + 2.5*Ss)
    plt.ylim(bottom=0, top=12)
    plt.savefig("test_SB.pdf")

    # create a mesh frame for S (from largest to smallest)
    print(f"Range of S: {Ss[0]} to {Ss[-1]}")
    print(f"Range of M: {Ms[0]:.6E} to {Ms[-1]:.6E}")
    print(f"M_sonic   : {M_sonic:.6E}")
    S_meshs = np.linspace(Ss[-1],Ss[0],n_S)
    #S_meshs = S_meshs[10:-10]
    dS = np.abs(S_meshs[0] - S_meshs[1])

    # reparameterise the measurements in terms of S
    B_tck = interpolate.splrep(Ss, Bs)
    B_meshs = interpolate.splev(S_meshs, B_tck, der=0)

    M_tck = interpolate.splrep(Ss, Ms)
    M_meshs = interpolate.splev(S_meshs, M_tck, der=0)

    R_tck = interpolate.splrep(Ss, Rs)
    R_meshs = interpolate.splev(S_meshs, R_tck, der=0)
    print("created the mesh for S and function B(S)!")

    plt.figure()
    dSdMs = 1/interpolate.splev(S_meshs, M_tck, der=1)
    plt.plot(M_meshs/(rho_0*h**3), dSdMs)
    plt.plot(M_meshs[1:]/(rho_0*h**3), np.diff(S_meshs)/np.diff(M_meshs), ls='--')
    plt.xscale("log")
    plt.savefig("test_dSdM.pdf")

    # evaluate the H matrix
    print("calculating H[n, m]...")
    H = np.zeros((n_S,n_S))
    for n in range(n_S) :
        for m in range(1, n) :
            H[n, m], _ = calc_H(S_meshs[n], S_meshs[m], dS)
        H[n, n], error = calc_H(S_meshs[n], S_meshs[n], dS)
        print(f"H[{n},{n}] = {H[n,n]:.10f}, rel_error={error:.6E}...", end='\r')

    outfile = open("last_crossing.dat", 'w+')
    # put everything together and calculate f_l(S_n)
    print("calculating last crossing distribution...")
    last_crossing = np.zeros(n_S)
    last_crossing[0] = g_1(S_meshs[0])
    last_crossing[1] = (g_1(S_meshs[1])+last_crossing[0]*H[1,1])/(1-H[1,1])
    for n in range(2, len(last_crossing)) :
        sum = np.sum([(last_crossing[m]+last_crossing[m-1]) * H[n,m] for m in range(1,n)])
        last_crossing[n] = (g_1(S_meshs[n]) + last_crossing[n-1]*H[n,n] + sum)/(1-H[n,n])
        outfile.write(f"LCD[{n}]: {last_crossing[n]:.3E}, M {M_meshs[n]:.3E}, S {S_meshs[n]:.3E}, sum {sum:.3E}, H_n {H[n,n]:.3E}, g_1 {g_1(S_meshs[n]):.3E}\n")

    IMF = calc_IMF(R_meshs, S_meshs, B_meshs, M_meshs, last_crossing)

    # export the results
    if use_beta :
        from mp_hopkins import beta
        bb = beta
    else :
        from mp_hopkins import B_mag
        bb = B_mag
    print("preparing to export the IMF...")
    filename_hdf5 = f"M{mach_h:.0f}p{p}B{bb}n{n_S}_dir.hdf5"

    # try whether the data already is there
    try :
        h5 = h5py.File(filename_hdf5, 'a')
        h5.create_dataset('M',data=M_meshs)
        h5.create_dataset('IMF',data=IMF)
    except RuntimeError:
        print("file already exists!")
        filename_hdf5 = "temp.hdf5"
        h5 = h5py.File(filename_hdf5, 'w')
        h5.create_dataset('M',data=M_meshs)
        h5.create_dataset('IMF',data=IMF)
    finally :
        print(f"data written successfully in {filename_hdf5}!")
        print(f"M_sonic={M_sonic:.6E}")
        print(f"M_gas={rho_0*h**3:.6E}")
        print(f"rho_0={rho_0:.6E}")
        h5.close()
