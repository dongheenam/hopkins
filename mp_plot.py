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

    # plot the IMF
    M_nonzero = M_nonzero*shift_x
    ax.plot(M_nonzero, M_nonzero*IMF_nonzero*shift_y, **kwargs)

if __name__ == "__main__" :
    # initialise mpl
    mpltools.mpl_init()
    fig = plt.figure()
    ax = fig.add_subplot()

    # set up the axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    #ax.set_xlim(left=1e-4, right=1e5)
    #ax.set_ylim(bottom=1e1, top=1e6)

    ax.set_xlabel(r"$M[M_\mathrm{sonic}]$")
    ax.set_ylabel(r"$\dfrac{dN}{d\log M}$")
    #ax.set_title(r'$E_v\propto k^{-2}, h=1$')

    # Salpeter slope
    x_Sal = np.linspace(1e1,1e5,10)
    ax.plot(x_Sal, 1e4*(x_Sal/100)**(-1.35), color='orange', linewidth=5, label='Salpeter')

    # plot the IMFs
    plot_imf("result/M30p2B0.5n800.hdf5", ax, color='r', label=r'$p=2, \mathcal{M}_{h,1D}=30,\beta=0.5$')
    plot_imf("M30p2B0.0n5000_dir.hdf5", ax, shift_x=1/1.096026e34, shift_y=1e97, color='r', ls='--', binned=False)
    plot_imf("result/M30p1.7B0.5n800.hdf5", ax, color='g', label=r'$p=5/3, \mathcal{M}_{h,1D}=30,\beta=0.5$')
    #plot_imf("M30p1.7B0.0n5000_dir.hdf5", ax, shift_x=1/3.653419e32, shift_y=1e97, color='g', ls='--', binned=False)
    plot_imf("result/M10p2B0.5n800.hdf5", ax, color='b', label=r'$p=2, \mathcal{M}_{h,1D}=10,\beta=0.5$')
    #plot_imf("M10p2B0.0n5000_dir.hdf5", ax, shift_x=1/9.864231e34, shift_y=1e97, color='b', ls='--', binned=False)
    #plot_imf("M30p2B0.5n_var.hdf5", ax, shift_x=4e0, shift_y=1.2e16, label='direct, newmesh')

    # data from Hopkins (2012)
    m30x, m30y, m10x, m10y = np.power(10, np.genfromtxt(
        "Hopkins2012.csv", delimiter=',', skip_header=2, unpack=True))

    #ax.plot(m10x, m10y, color='k', linestyle='--')
    ax.plot(m30x, m30y, color='k', linewidth=2, label='Hopkins(2012)')

    # save the plot
    plt.legend(prop={'size':12}, loc='lower left')
    plt.savefig("IMF.pdf")
