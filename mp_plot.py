# dependencies
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mpltools
from constants import G

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

    # plot the IMF
    ax.plot(M_nonzero/norm_x, M_nonzero**2*IMF_nonzero/norm_y, **kwargs)

if __name__ == "__main__" :
    # initialise mpl
    mpltools.mpl_init()
    fig = plt.figure()
    ax = fig.add_subplot()

    # set up the axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1e-8, right=1e2)
    ax.set_ylim(bottom=1e-3, top=1e0)

    ax.set_xlabel(r"$M[\rho_0 h^3]$")
    ax.set_ylabel(r"$M \dfrac{dN}{d\log M}[\rho_0]$")

    # Salpeter slope
    #x_Sal = np.linspace(1e-4,1e0,10)
    #ax.plot(x_Sal, x_Sal**(-1.35), color='orange', linewidth=3, ls='--', label='Salpeter')

    # plot the IMFs
    plot_imf("M10p2B0.0n5000_dir.hdf5", ax, norm_x=1.171e39, norm_y=3.985e-23, binned=False, color='k', label=r'$h=100~\mathrm{pc}, n=10~\mathrm{cm}^{-3}, N=5000$')
    plot_imf("M10p2B0.0n1000_dir.hdf5", ax, norm_x=1.171e39, norm_y=3.985e-23, binned=False, label=r'$h=100~\mathrm{pc}, n=10~\mathrm{cm}^{-3}, N=1000$')
    plot_imf("M10p2B0.0n500_dir.hdf5", ax, norm_x=1.171e39, norm_y=3.985e-23, binned=False, label=r'$h=100~\mathrm{pc}, n=10~\mathrm{cm}^{-3}, N=500$')

    # data from Hopkins (2013)
    m30x, m30y, m10x, m10y, m3x, m3y, m1x, m1y = np.genfromtxt(
        "test/hopkins2013_mach.csv", delimiter=',', skip_header=2, unpack=True)
    #ax.plot(m30x, m30y, color='red', label=r'$\mathcal{M}_h=30$')
    ax.plot(m10x, m10y, color='gray', lw=3, label=r'Hopkins, $\mathcal{M}_h=10$')
    #ax.plot(m3x, m3y, color='blue', label=r'$\mathcal{M}_h=3$')
    #ax.plot(m1x, m1y, color='green', label=r'$\mathcal{M}_h=1$')

    # save the plot
    plt.legend(prop={'size':12}, loc='upper left')
    plt.savefig("IMF.pdf")
