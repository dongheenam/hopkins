# dependencies
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mpltools
from constants import G

def plot_imf(filename, ax, **kwargs) :
    # read the IMF
    h5 = h5py.File(filename, 'r')

    IMF = np.array(h5['IMF'])
    M = np.array(h5['M'])

    # truncate the distribution
    bin_width = 4
    IMF_binned = np.zeros(IMF.size//bin_width)
    for i in range(IMF_binned.size) :
        IMF_binned[i] = np.average(IMF[i*bin_width:(i+1)*bin_width])
    M_binned = M[bin_width//2::bin_width]

    # remove zero entries
    IMF_nonzero = IMF_binned[IMF_binned!=0]
    M_nonzero = M_binned[IMF_binned!=0]

    # plot the IMF
    new_G = 903/(np.sqrt(2)*np.pi)
    M_nonzero = M_nonzero/G*new_G
    ax.plot(M_nonzero, M_nonzero*IMF_nonzero, **kwargs)

if __name__ == "__main__" :
    # initialise mpl
    mpltools.mpl_init()
    fig = plt.figure()
    ax = fig.add_subplot()

    # set up the axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1e-2, right=1e4)
    ax.set_ylim(bottom=1e13, top=1e16)

    ax.set_xlabel(r"$M[M_\mathrm{sonic}]$")
    ax.set_ylabel(r"$\dfrac{dN}{d\log M}$")
    #ax.set_title(r'$E_v\propto k^{-2}, h=1$')

    # Salpeter slope
    x_Sal = np.linspace(1e0,1e4,10)
    ax.plot(x_Sal, 3e15*(x_Sal/100)**(-1.35), color='b', linestyle=':', label='Salpeter')

    # plot the IMFs
    #plot_imf("result/M30p2B0n800.hdf5", ax, color='k', label=r'$\mathcal{M}_{h,1D}=30,\beta=\infty$')
    plot_imf("M30p2B0.5n800.hdf5", ax, color='r', label=r'$p=2, \mathcal{M}_{h,1D}=30,\beta=0.5$')
    #plot_imf("M20p2B0.5n800.hdf5", ax, color='b', label=r'$\mathcal{M}_{h,1D}=45,\beta=0.5$')

    # data from Hopkins (2012)
    m30x, m30y, m10x, m10y = np.power(10, np.genfromtxt(
        "Hopkins2012.csv", delimiter=',', skip_header=2, unpack=True))

    #ax.plot(m10x, m10y, color='k', linestyle='--')
    ax.plot(m30x, 3e11*m30y, color='k', linestyle='--', label='Hopkins(2012)')

    # save the plot
    plt.legend(prop={'size': 16}, loc='lower right')
    plt.savefig("IMF.pdf")
