# dependencies
import datetime
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
import ray
import scipy.integrate as integrate # for definite integration

import mpltools
from constants import M_SOL, G

""" constants """
mach_h = 10                              # 1D Mach number on scale h
h = 2 * 3.086e18                      # length scale h (pc > cm)
c_s = 0.2e5                             # sonic speed (cm/s)
n_0 = 10.0                               # mean number density (cm^-3)
mu = 2.4 / 6.022e23                     # mean mass per particle (g)
rho_0 = n_0 * mu                        # mean mass density (g cm^-3)

use_beta = False
if use_beta :
    beta = 5.0                          # plasma beta
    v_A = c_s/np.sqrt(beta)             # Alfven speed
else :
    B_mag = 0.0                               # magnetic field strength (Gauss)
    v_A = B_mag / np.sqrt(4*np.pi*rho_0)      # Alfven speed (cm/s)
    beta = (c_s/v_A)**2 if v_A!=0 else np.inf # plasma beta

b = 1                                  # turbulence driving parameter
p = 2                                   # negative turbulent velocity PS index
Q = 1                                   # Toomre parameter
kappa_tilde = np.sqrt(2)                # ratio of epicyclic and orbital freqs
kappa = kappa_tilde * np.sqrt(c_s**2 + (mach_h*c_s)**2 + v_A**2) / (np.sqrt(2)*h)


# dimensionless setup
dimensionless = False
if dimensionless :
    h = 1.0
    rho_0 = 1.0
    c_s = 1.0
    M_SOL = 1.0

    use_beta = True
    beta = 0.5
    v_A = np.sqrt(1/beta)

    G = kappa_tilde * ( (mach_h**2 + 1)*c_s**2 + v_A**2 )/(np.sqrt(2)*np.pi*Q)

""" calculation parameters """
size_start = 2                          # starting scale (= 10^3 h)
size_end = -10                          # finishing scale (= 10^-4 h)
n_R = 800                               # resolution
n_path = int(1e8)                       # number of paths

threads = psutil.cpu_count()            # number of threads
work_size = threads*100

""" variables """
# turbulent velocity dispersion smoothed on R (v_t)
def sigma_t(R) :
    return mach_h*c_s * (R/h)**((p-1)/2) * (1+ (h/R)**(-2))**((1-p)/4)

# total gas velocity dispersion at size R (sigma_g)
def sigma_gas(R) :
    return np.sqrt(sigma_t(R)**2 + c_s**2 + v_A**2)

# density dispersion squared at size R (sigma_k)
def sigma_dens_squared(k) :
    sigma_squared = np.log(1 + b**2*sigma_t(1/k)**2/(c_s**2 + kappa**2/k**2))
    return sigma_squared

# global dispersion of density smoothed at R
def S(R) :
    integral = integrate.quad(lambda lnk: sigma_dens_squared(np.exp(lnk)), np.log(1/h*1e-20), np.log(1/R))
    #integral = integrate.quad(lambda k: sigma_dens_squared(1/k)/k, 0, 1/R)
    return integral[0]

# rho_crit / rho_0
def dens_ratio_at_crit(R) :
    k = h/R
    dens_ratio = (
         Q/(2*kappa_tilde) * (1+k)
        *( (sigma_gas(R)/sigma_gas(h))**2*k + kappa_tilde**2/k ) )
    return dens_ratio

# barrier function
def B(R) :
    return np.log(dens_ratio_at_crit(R)) + S(R)/2

# mass of collapsing region with size R
def M(R) :
    #ln_rho_crit = np.log(dens_ratio_at_crit(R)) + np.log(rho_0)
    #if (R/h > 5e-5) :
    #    mass = 4*np.pi*np.exp(ln_rho_crit)*h**3
    #    mass *= R**2/(2*h**2) + (1+R/h)*np.exp(-R/h) - 1
    #else :
    #    mass = 4/3*np.pi*np.exp(ln_rho_crit)*R**3

    rho_crit = dens_ratio_at_crit(R) * rho_0
    if (R/h > 5e-5) :
        exp_term = np.exp(-R/h)
    else :
        exp_term = 1
    mass = 4*np.pi*rho_crit*h**3 * (R**2/(2*h**2)+(1+R/h)*exp_term-1)

    if mass == 0.0 :
        sys.exit("zero mass encountered!")
    else :
        return mass

# calculate M_sonic
if p != 1.0 :
    R_sonic = h*mach_h**(-2/(p-1))
    #M_sonic = M(R_sonic)
    M_sonic = 2/3* c_s**2 * R_sonic/G
else :
    #R_sonic = h*np.exp(1-mach_h**2)
    M_sonic = M_SOL

""" subroutines """
def random_walk(Rs, Ss) :
    """ generate a single random walk path in density space """
    path = np.zeros(n_R, dtype=np.float128)

    # the beginning point
    path[0] = np.random.normal(scale=np.sqrt(Ss[0]))

    # the random walk iteration
    for i in range(1, n_R) :
        last_dens = path[i-1]
        S_diff = Ss[i] - Ss[i-1]
        path[i] = np.random.normal(loc=last_dens, scale=np.sqrt(S_diff))

    return path

def last_collapse(path, Bs) :
    """ compares the density random walk with the barrier
    searches from the last entry (SMALLEST length scale) """

    for i in range(n_R-1,-1,-1) :
        is_collapsing = (path[i] > Bs[i])
        if is_collapsing :
            return (True, i)
    return (False, 0)

def first_collapse(path, Bs) :
    """ compares the density random walk with the barrier
    searches from the first entry (LARGEST length scale) """

    for i in range(n_R) :
        is_collapsing = (path[i] > Bs[i])
        if is_collapsing :
            return (True, i)
    return (False, 0)

@ray.remote
def count_collapse(Rs, Ss, Bs, collapse_finder=last_collapse) :
    """
    creates single random walk path and determine its collapse scale
    """
    path = random_walk(Rs, Ss)
    is_collapsing, i_collapse = collapse_finder(path, Bs)
    if is_collapsing :
        return i_collapse, path[-1]
    else :
        return None, path[-1]

def calc_IMF(Rs, Ss, Bs, Ms, locs_collapse) :
    """ calculate the IMF (dn/dM) based on the collapse prob. dist. """
    dn_dM = np.zeros(len(locs_collapse))
    for i in range(1, len(dn_dM)) :
        rho_crit = np.exp(Bs[i]-Ss[i]/2)*rho_0
        dS_dM = (Ss[i-1]-Ss[i]) / (Ms[i-1]-Ms[i])
        dn_dM[i] = rho_crit/Ms[i] * locs_collapse[i] * np.abs(dS_dM)

    return dn_dM

"""
================================================================================
main
================================================================================
"""
if __name__ == "__main__" :
    # calculate R
    print("calculating R...")
    logh = np.log10(h)
    Rs = np.power(10, np.linspace(logh+size_start, logh+size_end, n_R))

    # calculate S and B
    print("calculating S, B, and M...")
    Ss, Bs, Ms = np.zeros((3,n_R), dtype=np.float64)
    for i in range(n_R) :
        Ss[i] = S(Rs[i])
        Bs[i] = B(Rs[i])
        Ms[i] = M(Rs[i])

    print(f"no of points: {n_R}")
    print(f"scale h     : {h}")
    print(f"Mach at h   : {mach_h}")
    print(f"Alfven speed: {v_A:.3E}")
    if v_A != 0.0 :
        print(f"Mach_A at h : {sigma_t(h)/v_A:.3f}")
        print(f"plasma beta : {beta:.3f}")
    print(f"S_smallest  : {Ss[-1]:.3f}")
    print(f"M_largest   : {Ms[0]/M_SOL:.3E} M_SOL ({Ms[0]/M_sonic:.3E} M_sonic)")
    print(f"M_h         : {M(h)/M_SOL:.3E} M_SOL ({M(h)/M_sonic:.3E} M_sonic)")
    print(f"M_smallest  : {Ms[-1]/M_SOL:.3E} M_SOL ({Ms[-1]/M_sonic:.3E} M_sonic)")

    # print first twenty random walks
    if True :
        print("plotting the first twenty Monte-Carlo paths...")
        mpltools.mpl_init()
        fig = plt.figure()
        ax = fig.add_subplot()

        # paths
        for i in range(20) :
            path = random_walk(Rs, Ss)
            ax.plot(Rs/h, path, linewidth=1, alpha=0.5)

        # S and B
        ax.plot(Rs/h, np.sqrt(Ss), label=r'$\sqrt{S(R)}$', color='k', linestyle='--', linewidth=3)
        ax.plot(Rs/h, -np.sqrt(Ss), color='k', linestyle='--', linewidth=3)
        ax.plot(Rs/h, Bs, label='$B(R)$', color='r', linewidth=3)

        ax.set_xscale("log")
        ax.set_xlim(left=1e-5, right=1e2)
        ax.set_ylim(bottom=-5, top=12)
        ax.set_xlabel("$R/h$")
        ax.set_ylabel("overdensity")
        plt.legend(loc="upper right")
        plt.savefig(f"barrier_M{mach_h:.0f}p{p}n{n_R}.pdf")

    sys.exit("finish here for now...")

    # begin the simulation
    print("initiating Monte Carlo random walking simulation...")
    ray.init(num_cpus=threads)

    t_start = datetime.datetime.now()
    times_history = []
    locs_collapse = np.zeros(n_R, dtype=int)
    i_path = 0
    while True :
        # clean the buffer and throw lots of indices in there first
        locs_buffer = []
        for _ in range(work_size) :
            locs_buffer.append(count_collapse.remote(Rs, Ss, Bs))

        # use the indices in the buffer to increase the count
        locs = ray.get(locs_buffer)
        for i, _ in locs :
            if i is not None :
                locs_collapse[i] += 1

        # calculate S(R)
        S = np.std([e for i, e in locs])**2

        # if the number of paths have reached the desired amount, exit the loop
        i_path += work_size
        if i_path >= n_path :
            break

        # estimate finish time
        time_now = datetime.datetime.now()
        times_history.append(time_now)
        if len(times_history) > 100:
            times_history.pop(0)
            estimate_time_remaining = \
                (times_history[-1]-times_history[0])*(n_path-i_path)/(work_size*100)
            estimate_finish = time_now + estimate_time_remaining
        else :
            estimate_finish = datetime.datetime.max

        print((
            f"[{time_now.isoformat(' ', 'milliseconds')}] S={S:5.2f}, "
            f"calculated {i_path} paths ({i_path/n_path*100:.2f}%), "
            f"{np.sum(locs_collapse)} collapses "
            f"(finish at {estimate_finish.isoformat(' ', 'seconds')})")
              )


    t_end = datetime.datetime.now()
    print("finished!")
    print(f"time spent: {(t_end-t_start).total_seconds()}s")

    # calculate the IMF
    print("calculating the IMF...")
    IMF = calc_IMF(Rs, Ss, Bs, Ms, locs_collapse)

    # export the results
    print("preparing to export the IMF...")
    if dimensionless :
        filename_hdf5 = f"M{mach_h:.0f}p{p}B{beta}n{n_R}.hdf5"
    else :
        filename_hdf5 = f"M{mach_h:.0f}p{p}B{int(B_mag*1e6)}n{n_R}.hdf5"
    h5 = h5py.File(filename_hdf5, 'a')

    # try whether the data already is there
    try :
        h5.create_dataset('M',data=Ms/M_sonic)
        h5.create_dataset('IMF',data=IMF)
    except RuntimeError:
        print("file already exists! appending the data...")
        h5['IMF'][...] += IMF
    finally :
        print("data written successfully!")
        h5.close()
