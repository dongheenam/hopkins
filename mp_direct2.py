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
from mp_hopkins import mach_h, h, rho_0, beta, v_A, c_s, p, Q, kappa_tilde
from mp_hopkins import S, B, M, calc_IMF

""" calculation parameters """
n_S = 5000

size_start = 2                          # starting scale (= 10^3 h)
size_end = -10                           # finishing scale (= 10^-4 h)

""" probability functions """

def P_0(x, var=1) :
    if var < 0 :
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
        return dS/2 * g_2(S1, S2+dS/2)
    else :
        # integration of Taylor expansion near singularity
        dS1 = dS
        B1 = interpolate.splev(S1, B_tck, der=0)
        dB1 = interpolate.splev(S1, B_tck, der=1)
        ddB1 = interpolate.splev(S1, B_tck, der=2)

        int1 = np.sqrt(dS1) * (
            (-B1+S1*dB1) * (-S1*(dS1+6*S1)+dS1*(B1-S1*dB1)**2) + dS1*S1**3*ddB1
        ) / (3*np.sqrt(2*np.pi)*S1**3)

        #int2, _ =  quad(lambda Sp: g_2(S1, Sp), S1+dS1, S1+dS)
        return int1/2

if __name__ == "__main__" :
    # calculate R
    print("calculating R...")
    logh = np.log10(h)
    Rs = np.power(10, np.linspace(logh+size_start, logh+size_end, 1000))

    G = kappa_tilde * ( (mach_h**2 + 1)*c_s**2 + v_A**2 )/(np.sqrt(2)*np.pi*Q)
    R_sonic = h*mach_h**(-2/(p-1))
    M_sonic = M(R_sonic)

    # calculate S and B
    print("calculating S and B...")
    Ss, Bs, Ms = np.zeros((3,len(Rs)), dtype=np.float64)
    for i in range(len(Rs)) :
        Ss[i] = S(Rs[i])
        Bs[i] = B(Rs[i])
        Ms[i] = M(Rs[i]) / M_sonic

    # truncate the distribution
    S_is_nonzero = (Ss!=0.0)
    Rs = Rs[S_is_nonzero]
    Bs = Bs[S_is_nonzero]
    Ms = Ms[S_is_nonzero]
    Ss = Ss[S_is_nonzero]

    # create a Spline interpolation for the measurements as a function of S
    print(f"Range of S: {Ss[0]} to {Ss[-1]}")
    print(f"Range of M: {Ms[0]/M_sonic:.4E} M_sonic to {Ms[-1]/M_sonic:.4E} M_sonic")
    B_tck = interpolate.splrep(Ss, Bs)
    M_tck = interpolate.splrep(Ss, Ms)
    R_tck = interpolate.splrep(Ss, Rs)

    # initial steps
    S_start = Ss[-1]
    S_end = Ss[0]
    dS = 1e-5
    print(f"Starting at S = {S_start:.3E} with dS = {dS:.3E}...")

    # arrays for the final outputs
    IMFs = np.array([], dtype=np.float64)
    Ms = np.array([], dtype=np.float64)

    # size of a single mesh grid
    n_mesh = 100

    # flag to exit the loop
    last_mesh = False

    while True :
        # initial points
        lc = np.zeros(n_mesh)
        lc[0] = g_1(S_start)
        H1 = calc_H(S_start+dS, S_start+dS, dS)
        lc[1] = (g_1(S_start+dS)+lc[0]*H1)/(1-H1)

        # create a mesh
        S_meshs = np.linspace(S_start, S_start-(n_mesh-1)*dS, n_mesh)
        print(f"mesh created from {S_meshs[0]:.3E} to {S_meshs[-1]:.3E} with dS = {S_meshs[0]-S_meshs[1]:.3E}")
        if S_meshs[-1] < S_end :
            break
            lc = lc[S_meshs>S_end]
            S_meshs = S_meshs[S_meshs>S_end]
            n_mesh = len(S_meshs)
            last_mesh = True

        # evaluate the H matrix
        print("calculating H[n, m]...")
        H = np.zeros((n_mesh, n_mesh))
        for n in range(n_mesh) :
            for m in range(1, n+1) :
                H[n, m] = calc_H(S_meshs[n], S_meshs[m], dS)
            print(f"H[{n},{n}] = {H[n,n]}...", end='\r')

        # put everything together and calculate f_l(S_n)
        print("calculating last crossing distribution...")
        for n in range(2, len(lc)) :
            sum = np.sum([(lc[m]+lc[m-1]) * H[n,m] for m in range(1,n)])
            lc[n] = (g_1(S_meshs[n]) + lc[n-1]*H[n,n] + sum)/(1-H[n,n])
            print(f"lc[n] = {lc[n]}...", end='\r')

        # calculate the IMF from f_l
        print("calculating the IMF...")
        R_meshs = interpolate.splev(S_meshs, R_tck, der=0)
        B_meshs = interpolate.splev(S_meshs, B_tck, der=0)
        M_meshs = interpolate.splev(S_meshs, M_tck, der=0)
        IMF = calc_IMF(R_meshs, S_meshs, B_meshs, M_meshs, lc)

        # append the results (except the initial conditions)
        Ms = np.append(Ms, M_meshs[2:])
        IMFs = np.append(IMFs, IMF[2:])

        # exit the loop here if S_end is reached
        if last_mesh :
            break

        # move to the next grid setup
        lc[...] = 0.0
        dS = np.abs(S_meshs[-1]-S_meshs[-1-n_mesh//10])
        print(f"dS = {dS} now, moving to the next iteration...")

    # export the results
    print("preparing to export the IMF...")
    filename_hdf5 = f"M{mach_h:.0f}p{p}B{beta}n_var.hdf5"
    h5 = h5py.File(filename_hdf5, 'a')

    # try whether the data already is there
    try :
        h5.create_dataset('M',data=Ms)
        h5.create_dataset('IMF',data=IMFs)

        print("data written successfully!")
        h5.close()
    except RuntimeError:
        print("file already exists!")
