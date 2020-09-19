import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.integrate import quad

""" calculation parameters """
n_S = 1000
B_0 = 0.7
B_slope = 2.5

def B(S) :
    return B_0 + S*B_slope

def f_l(S) :
    return B_slope/np.sqrt(2*np.pi*S) * np.exp(-B(S)**2/(2*S))

""" probability functions """

def P_0(x, var=1) :
    if var <= 0 :
        print(f"{var} is not valid variance")
    return 1/np.sqrt(2*np.pi*var) * np.exp(x**2/(-2*var))

""" other functions """
def dB_dS(S) :
    return B_slope

def g_1(S1) :
    B1 = B(S1)
    return (2*dB_dS(S1) - B1/S1)*P_0(B1, var=S1)

def g_2(S1,S2) :
    B1 = B(S1)
    B2 = B(S2)

    part_1 = (B1-B2)/(S1-S2) + B1/S1 - 2*dB_dS(S1)
    part_2 = P_0(B1 - B2*S1/S2, var=((S2-S1)*S1/S2))
    return part_1 * part_2

def calc_H(S1, S2, dS) :
    if S1 != S2 :
        return dS/2 * g_2(S1, S2+dS/2)
    else :
        #B1 = interpolate.splev(S1, B_tck, der=0)
        #dB1 = interpolate.splev(S1, B_tck, der=1)
        int2, _ = quad(lambda Sp: g_2(S1, Sp), S1, S1+dS)
        return (int1 + int2)/2

if __name__ == "__main__" :
    # calculate S
    S_meshs = np.linspace(10.0, 0.01, n_S)
    dS = np.abs(S_meshs[0] - S_meshs[1])

    # calculate B
    B_meshs = B(S_meshs)

    # evaluate H matrix
    print("calculating H[n, m]...")
    H = np.zeros((n_S,n_S))
    for n in range(n_S) :
        for m in range(1, n+1) :
            H[n, m] = calc_H(S_meshs[n], S_meshs[m], dS)
        print(f"H[{n},{n}] = {H[n,n]}...", end='\r')

    plt.figure()
    plt.plot([H[n,n] for n in range(n_S)])
    plt.yscale("log")
    plt.savefig("test_H.pdf")

    # put everything together and calculate f_l(S_n)
    print("calculating last crossing distribution...")
    last_crossing = np.zeros(n_S)
    last_crossing[0] = g_1(S_meshs[0])
    last_crossing[1] = (g_1(S_meshs[1])+last_crossing[0]*H[1,1])/(1-H[1,1])
    for n in range(2, len(last_crossing)) :
        sum = np.sum([(last_crossing[m]+last_crossing[m-1]) * H[n,m] for m in range(1,n)])
        last_crossing[n] = (g_1(S_meshs[n]) + last_crossing[n-1]*H[n,n] + sum)/(1-H[n,n])

    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim(right=1.5)
    ax.set_ylim(bottom=-0.01, top=0.01)

    ax.plot(S_meshs, S_meshs*0, color='black')
    ax.plot(S_meshs, (last_crossing-f_l(S_meshs))/np.max(last_crossing), color='aquamarine')

    fig.savefig("lastcrossing.pdf")
