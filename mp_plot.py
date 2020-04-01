# dependencies
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mpltools
from constants import G

def plot_imf(filename, ax, shift_x=1, shift_y=1, binned=True, **kwargs) :
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

    # normalise the IMF
    M_nonzero = M_nonzero * shift_x

    # plot the IMF
    ax.plot(M_nonzero, M_nonzero*IMF_nonzero*shift_y, **kwargs)

if __name__ == "__main__" :
    # initialise mpl
    mpltools.mpl_init()
    fig = plt.figure()
    ax = fig.add_subplot()

    # set up the axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1e-8, right=1e2)
    ax.set_ylim(bottom=1e-4, top=1e8)

    ax.set_xlabel(r"$M[\rho_0 h^3]$")
    ax.set_ylabel(r"$\dfrac{dN}{d\log M}$")

    # Salpeter slope
    x_Sal = np.linspace(1e-4,1e0,10)
    ax.plot(x_Sal, x_Sal**(-1.35), color='orange', linewidth=3, ls='--', label='Salpeter')

    # plot the IMFs
    #plot_imf("M1p2B0.0n5000_dir.hdf5", ax, shift_x=1/3.521e40, shift_y=1e105, ls='--', binned=False, label=r'$\mathcal{M}_h=1,p=2,B=0$')
    #plot_imf("M3p2B0.0n5000_dir.hdf5", ax, shift_x=1/3.521e40, shift_y=1e105, ls='--', binned=False, label=r'$\mathcal{M}_h=3,p=2,B=0$')
    plot_imf("M10p1.0B0.0n5000_dir.hdf5", ax, shift_x=1/3.521e40, shift_y=1e105, ls='-', color='pink', lw=3, binned=False, label=r'$\mathcal{M}_h=10,p=1,B=0$')
    plot_imf("M10p1.3B0.0n5000_dir.hdf5", ax, shift_x=1/3.521e40, shift_y=1e105, ls='-', binned=False, label=r'$\mathcal{M}_h=10,p=4/3,B=0$')
    plot_imf("M10p1.7B0.0n5000_dir.hdf5", ax, shift_x=1/3.521e40, shift_y=1e105, ls='-', binned=False, label=r'$\mathcal{M}_h=10,p=5/3,B=0$')
    plot_imf("M10p2B0.0n5000_dir.hdf5", ax, shift_x=1/3.521e40, shift_y=1e105, ls='-', color='k', lw=3, binned=False, label=r'$\mathcal{M}_h=10,p=2,B=0$')
    plot_imf("M10p2.5B0.0n5000_dir.hdf5", ax, shift_x=1/3.521e40, shift_y=1e105, ls='-', binned=False, label=r'$\mathcal{M}_h=10,p=5/2,B=0$')
    #plot_imf("M30p2B0.0n5000_dir.hdf5", ax, shift_x=1/3.521e40, shift_y=1e105, ls='--', binned=False, label=r'$\mathcal{M}_h=30,p=2,B=0$')

    # data from Hopkins (2012)
    #m30x, m30y, m10x, m10y = np.power(10, np.genfromtxt(
    #    "Hopkins2012.csv", delimiter=',', skip_header=2, unpack=True))
    #ax.plot(m10x, m10y, color='k', linestyle='--')
    #ax.plot(m30x, m30y, color='k', linewidth=2, label='Hopkins(2012)')

    # save the plot
    plt.legend(prop={'size':12}, loc='lower left')
    plt.savefig("IMF.pdf")
