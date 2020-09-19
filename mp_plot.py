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

    # normalise
    x = M_nonzero/norm_x
    y = M_nonzero*IMF_nonzero/norm_y

    # plot the IMF
    ax.plot(x, y, **kwargs)

if __name__ == "__main__" :
    # initialise mpl
    mpltools.mpl_init()
    fig = plt.figure()
    ax = fig.add_subplot()

    # set up the axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1e-3, right=2e1)
    ax.set_ylim(bottom=1e0, top=1e4)

    ax.set_xlabel(r"$M[\rho_0 h^3]$")
    ax.set_ylabel(r"$\dfrac{dn}{d\log M}[1/h^{3}]$")

    # Salpeter slope
    base_Sal = (1e0, 3e2)
    x_Sal = np.linspace(1e-1,1e1,10)
    ax.plot(x_Sal, base_Sal[1]*(x_Sal/base_Sal[0])**(-1.35), color='orange', linewidth=3, ls='--', label='Salpeter')

    # plot the IMFs
    rho_0 = 3.985e-23
    h = 6.152e18
    M_gas = rho_0*h**3
    plot_imf("M1.8p1.9B0.0n1000_dir.hdf5", ax, norm_x=M_gas, norm_y=1/h**3, binned=False, lw=2, color='black', ls='-', label=r'$n=1.9, \mathcal{M}_h=1.8$')
    plot_imf("M0.5p1.001B0.0n1000_dir.hdf5", ax, norm_x=M_gas, norm_y=1/h**3, binned=False, lw=2, color='blue', ls='-', label=r'$n=1, \mathcal{M}_h=0.5$')
    plot_imf("M2.9p1.9B0.0n1000_dir.hdf5", ax, norm_x=M_gas, norm_y=1/h**3, binned=False, lw=2, color='black', ls='--', label=r'$n=1.9, \mathcal{M}_h=2.9$')
    plot_imf("M1.3p1.001B0.0n1000_dir.hdf5", ax, norm_x=M_gas, norm_y=1/h**3, binned=False, lw=2, color='blue', ls='--', label=r'$n=1, \mathcal{M}=1.3$')
    plot_imf("M5.0p1.9B0.0n1000_dir.hdf5", ax, norm_x=M_gas, norm_y=1/h**3, binned=False, lw=2, color='black', ls=':', label=r'$n=1.9, \mathcal{M}_h=5$')
    plot_imf("M5.0p1.001B0.0n1000_dir.hdf5", ax, norm_x=M_gas, norm_y=1/h**3, binned=False, lw=2, color='blue', ls=':', label=r'$n=1, \mathcal{M}=5$')
    # save the plot
    plt.legend(prop={'size':14}, loc='upper right')
    plt.savefig("IMF.pdf")
