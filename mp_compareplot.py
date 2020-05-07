# dependencies
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mpltools
from constants import G

def find_slope(x, y) :
    # a = where M = 0.001M_total
    i_a = np.abs(x - 0.001).argmin()
    a = x[i_a]
    fa = y[i_a]
    # b = where M = 0.01M_total
    i_b = np.abs(x - 0.01).argmin()
    b = x[i_b]
    fb = y[i_b]

    # k is the power-law slope
    k = np.log10(fb/fa) / np.log10(b/a)
    return k


def plot_imf(filename, ax, norm_x=1, norm_y=1, binned=True, **kwargs) :
    # read the IMF
    h5 = h5py.File(filename, 'r')

    IMF = np.array(h5['IMF'])
    M = np.array(h5['M'])

    # truncate the distribution
    if binned :
        bin_width = 4
        IMF_binned = np.zeros(IMF.size//bin_width)
        for i in range(IMF_binned.size) :
            IMF_binned[i] = np.average(IMF[i*bin_width:(i+1)*bin_width])
        M_binned = M[bin_width//2::bin_width]
    else :
        IMF_binned = IMF
        M_binned = M

    # remove zero entries
    IMF_nonzero = IMF_binned[IMF_binned!=0]
    M_nonzero = M_binned[IMF_binned!=0]

    # normalise
    x = M_nonzero/norm_x
    y = M_nonzero**2*IMF_nonzero/norm_y

    # plot the IMF
    ax.plot(x, y, **kwargs)

    # find the slope
    k = find_slope(x, y)
    print(f"slope for {filename} is {k:.3f}")

if __name__ == "__main__" :
    # initialise mpl
    mpltools.mpl_init()
    fig = plt.figure()
    ax = fig.add_subplot()

    # set up the axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1e-8, right=1e2)
    ax.set_ylim(bottom=1e-2, top=1e0)

    ax.set_xlabel(r"$M[\rho_0 h^3]$")
    ax.set_ylabel(r"$M \dfrac{dn}{d\log M}[\rho_0]$")

    # data from Hopkins (2013)
    b10x, b10y, b05x, b05y, b03x, b03y = np.genfromtxt(
        "test/hopkins2013_b.csv", delimiter=',', skip_header=2, unpack=True)
    ax.plot(b10x, b10y, color='blueviolet', ls='--')
    ax.plot(b05x, b05y, color='dodgerblue', ls='--')
    ax.plot(b03x, b03y, color='black', ls='--', label='Hopkins(2013)')

    # plot the IMFs
    M_gas = 9.3702e33
    rho_0 = 3.985e-23
    plot_imf("M10p2B0.0n1000_dir.hdf5", ax, norm_x=M_gas, norm_y=rho_0/2.4, binned=False, color='blueviolet', label=r'$p=2, \mathcal{M}_h=10, b=1 \quad (\times 2.4)$')
    plot_imf("M10p2B0.0b0.5n1000_dir.hdf5", ax, norm_x=M_gas, norm_y=rho_0/2.4, binned=False, color='dodgerblue', label=r'$p=2, \mathcal{M}_h=10, b=1/2 \quad (\times 2.4)$')
    plot_imf("M10p2B0.0b0.3n1000_dir.hdf5", ax, norm_x=M_gas, norm_y=rho_0/2.4, binned=False, color='black', label=r'$p=2, \mathcal{M}_h=10, b=1/3 \quad (\times 2.4)$')

    # save the plot
    plt.legend(prop={'size':12}, loc='upper left')
    plt.savefig("IMF.pdf")
