# dependencies
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d # one-dimensional interpolation
from scipy.misc import derivative

import mpltools
from constants import M_SOL, G
from mp_hopkins import mach_h, h, rho_0, beta, v_A, c_s, p, Q, kappa_tilde, \
                       size_start, size_end, n_R
from mp_hopkins import S, B, M, calc_IMF

""" calculation parameters """
n_S = 300

""" probability functions """

def P_0(x, var=1) :
    if var < 0 :
        print(f"{var} is not valid variance")
    return 1/np.sqrt(2*np.pi*var) * np.exp(-0.5*x**2/var)

def P_01(S1, S2) :
    x = B_map(S1)-B_map(S2)*(S1/S2)
    var = (S2-S1)*(S1/S2)
    return P_0(x, var=var)

""" other functions """
def dB_dS(S) :
    #return derivative(B_map, S, dx=1e-6)
    return (np.diff(B_meshs)/np.diff(S_meshs))[S_meshs[1:]==S][0]

def g_1(S) :
    B = B_map(S)
    return (2*dB_dS(S) - B/S)*P_0(B, var=S)

def g_2(S1,S2) :
    B1 = B_map(S1)
    B2 = B_map(S2)

    return ( (B1-B2)/(S1-S2) + B1/S1 - 2*dB_dS(S1) ) * P_01(S1, S2)

if __name__ == "__main__" :
    # calculate R
    print("calculating R...")
    logh = np.log10(h)
    Rs = np.power(10, np.linspace(logh+size_start, logh+size_end, n_R))

    # calculate S and B
    print("calculating S and B...")
    Ss, Bs, Ms = np.zeros((3,n_R), dtype=np.float64)
    for i in range(n_R) :
        Ss[i] = S(Rs[i])
        Bs[i] = B(Rs[i])
        Ms[i] = M(Rs[i])

    R_sonic = h*mach_h**(-2/(p-1))
    M_sonic = 2/3* c_s**2 * R_sonic/G

    # truncate the distribution
    S_is_nonzero = (Ss!=0.0)
    Rs = Rs[S_is_nonzero]
    Bs = Bs[S_is_nonzero]
    Ms = Ms[S_is_nonzero]
    Ss = Ss[S_is_nonzero]

    # create a mesh frame for S
    S_meshs = np.linspace(Ss[-1],Ss[0],n_S)
    dS = S_meshs[0] - S_meshs[1]

    # function B(S)
    B_map = interp1d(np.flip(Ss), np.flip(Bs), kind='cubic')
    B_meshs = B_map(S_meshs)
    M_map = interp1d(np.flip(Ss), np.flip(Bs), kind='cubic')
    M_meshs = M_map(S_meshs)
    R_map = interp1d(np.flip(Ss), np.flip(Rs), kind='cubic')
    R_meshs = M_map(S_meshs)
    print(f"range of S: {(Ss[-1], Ss[0])}")
    print("created the mesh for S and function B(S)!")

    print("calcualting H[n, m]...")
    H = np.zeros((n_S,n_S))
    for n in range(2, n_S) :
        for m in range(1, n) :
            H[n, m] = dS/2 * g_2(S_meshs[n], S_meshs[m]+dS/2)
            print(f"now at H[{n}, {m}]...", end='\r')

    last_crossing = np.zeros(n_S)

    last_crossing[0] = 0.0
    last_crossing[1] = g_1(S_meshs[1])/(1-H[1,1])
    for n in range(2, len(last_crossing)) :
        sum = 0.0
        for m in range(1,n) :
            sum += (last_crossing[m] + last_crossing[m-1])*H[m,n]
        last_crossing[n] = (g_1(S_meshs[n]) + last_crossing[n-1]*H[n,n] + sum)/(1-H[n,n])

        print(f"LCD[{n}] = {last_crossing[n]}...")

    IMF = calc_IMF(R_meshs, S_meshs, B_meshs, M_meshs, last_crossing)

    # export the results
    print("preparing to export the IMF...")
    filename_hdf5 = f"M{mach_h:.0f}p{p}B{beta}n{n_R}_dir.hdf5"
    h5 = h5py.File(filename_hdf5, 'a')

    # try whether the data already is there
    try :
        h5.create_dataset('M',data=M_meshs/M_sonic)
        h5.create_dataset('IMF',data=IMF)
    except RuntimeError:
        print("file already exists! appending the data...")
        h5['IMF'][...] += IMF
    finally :
        print("data written successfully!")
        h5.close()
