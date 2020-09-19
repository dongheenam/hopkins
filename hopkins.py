# dependencies
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate

""" options """
vary_Q = True    # If True, calculate Q from other values
                 # If False, acccept the value of Q below
rot_supp = False # If True, include the rotational support term

""" physical constants """
G = 6.674e-8         # gravitational constant
pc = 3.086e18        # parsec [cm]
c_s = 2e4            # isothermal sound speed [cm/s]

""" physical parameters """
h = 2*pc             # characteristic length [cm]
p = 2.0              # negative velocity PS index
mach_h = 5           # characteristic Mach number
b = 0.4              # turbulence driving parameter
B_mag = 0.0          # magnetic field [Gauss]
Q = 1.0              # Toomre parameter
kappa_t = np.sqrt(2) # ratio of epicyclic and orbital frequency
rho_0 = 1.31e-20     # mean mass density [g cm^-3]

# calculate Alfven speed, kappa and Toomre Q (if asked to)
v_A = B_mag / np.sqrt(4*np.pi*rho_0)
kappa = kappa_t * np.sqrt(c_s**2 + (mach_h*c_s)**2 + v_A**2) / (np.sqrt(2)*h)
if vary_Q :
    Q = ((mach_h*c_s)**2+c_s**2+v_A**2)/(np.sqrt(2)*np.pi*G*rho_0*h**2)


""" numerical parameters """
R_start = h*1e1      # starting length scale for the random walk
R_end = h*1e-8       # ending length scale for the random walk
n_S = 1500           # number of mesh points for S-parametrisation
small_k = 1/h*1e-20  # lower limit of k when integrating in k-space


""" functions for calculating the parameters appearing in Hopkins (2012) """

def sigma_t(R) :
    """ turbulent velocity dispersion smoothed on R (v_t) """
    return mach_h*c_s * (R/h)**((p-1)/2) * (1+ (h/R)**(-2))**((1-p)/4)

def sigma_gas(R) :
    """ total gas velocity dispersion at size R (sigma_g) """
    return np.sqrt(sigma_t(R)**2 + c_s**2 + v_A**2)

def sigma_dens_squared(k) :
    """ density dispersion squared at size R (sigma_k) """
    sigma_squared = np.log(1 + b**2*sigma_t(1/k)**2/(c_s**2 + kappa**2/k**2))
    return sigma_squared

def calc_S(R) :
    """ global dispersion of density smoothed at R """
    integral = integrate.quad(lambda lnk: sigma_dens_squared(np.exp(lnk)), np.log(small_k), np.log(1/R))
    return integral[0]

def dens_ratio_at_crit(R) :
    """ rho_crit / rho_0 """
    k = h/R
    if rot_supp :
        dens_ratio = (
            Q/(2*kappa_t) * (1+k)
            * ( (sigma_gas(R)/sigma_gas(h))**2*k ) 
        )
    else :
        dens_ratio = (
            Q/(2*kappa_t) * (1+k)
            * ( (sigma_gas(R)/sigma_gas(h))**2*k + kappa_t**2/k ) 
        )
    return dens_ratio

def calc_B(R) :
    """ barrier function """
    return np.log(dens_ratio_at_crit(R)) + calc_S(R)/2

def calc_M(R) :
    """ mass of collapsing region with size R """
    rho_crit = dens_ratio_at_crit(R) * rho_0
    r = R/h
    if (r > 5e-5) :
        mass = 4*np.pi*rho_crit*h**3 * (r**2/2+(1+r)*np.exp(-r)-1)
    else :
        mass = 4/3*np.pi*rho_crit*R**3

    if mass == 0.0 :
        sys.exit(f"zero mass encountered at: r={r}!")
    return mass

def calc_IMF(Rs, Ss, Bs, Ms, locs_collapse) :
    """ calculate the IMF (dn/dM) based on the collapse prob. dist. """
    dn_dM = np.zeros(len(locs_collapse))
    for i in range(1, len(dn_dM)) :
        rho_crit = dens_ratio_at_crit(Rs[i]) * rho_0
        dS_dM = 1/interpolate.splev(Ss[i], M_tck, der=1) # 1/(dM/dS(S))
        dn_dM[i] = rho_crit/Ms[i] * locs_collapse[i] * np.abs(dS_dM)

    return dn_dM


""" functions for the direct calculation of the last-crossing distribution """

def P_0(x, var=1.0) :
    """ returns Gaussian pdf for given variance """
    if var <= 0 : # basic sanity check
        sys.exit(f"{var} is not valid variance!")
    prob = 1/np.sqrt(2*np.pi*var) * np.exp(x**2/(-2*var))
    return prob

def dB_dS(S) :
    """ returns dB/dS evaluated at S """
    return interpolate.splev(S, B_tck, der=1)

def g_1(S) :
    """ returns g_1(S), as defined in Hopkins (2012) """
    B = interpolate.splev(S, B_tck, der=0) # find B(S)
    return (2*dB_dS(S) - B/S)*P_0(B, var=S)

def g_2(S1, S2) :
    """ returns g_2(S, S'), as defined in Hopkins (2012) """
    B1 = interpolate.splev(S1, B_tck, der=0) # find B(S1)
    B2 = interpolate.splev(S2, B_tck, der=0) # find B(S2)

    part_1 = (B1-B2)/(S1-S2) + B1/S1 - 2*dB_dS(S1)
    part_2 = P_0(B1 - B2*S1/S2, var=((S2-S1)*S1/S2))
    return part_1*part_2

def calc_H(S1, S2, dS) :
    """
    returns the triangular matrix H(S, S')
    the output is in the (value, rel_error) format
    """
    if S1 != S2 :
        return dS/2*g_2(S1, S2+dS/2), 0
    else :
        value, error = integrate.quad(lambda Sp: g_2(S1, Sp), S1, S1+dS)
        return value/2, error


""" MAIN """
if __name__ == "__main__" :

    # create a mesh in R
    print("calculating R...")
    Rs = np.logspace(np.log10(R_start), np.log10(R_end), n_S)

    # calculate S and B
    print("calculating S and B...")
    Ss, Bs, Ms = np.zeros( (3,len(Rs)) )
    for i in range(len(Rs)) :
        Ss[i] = calc_S(Rs[i])
        Bs[i] = calc_B(Rs[i])
        Ms[i] = calc_M(Rs[i])

    # truncate the values where S is zero
    S_is_nonzero = (Ss!=0.0)
    Rs = Rs[S_is_nonzero]
    Bs = Bs[S_is_nonzero]
    Ms = Ms[S_is_nonzero]
    Ss = Ss[S_is_nonzero]

    # create a uniform mesh in S (from largest to smallest)
    print(f"Range of S: {Ss[0]} to {Ss[-1]}")
    print(f"Range of M: {Ms[0]:.6E} to {Ms[-1]:.6E}")
    S_meshs = np.linspace(Ss[-1],Ss[0],n_S)
    dS = np.abs(S_meshs[0] - S_meshs[1])

    # reparameterise the measurements in terms of S
    B_tck = interpolate.splrep(Ss, Bs)
    B_meshs = interpolate.splev(S_meshs, B_tck, der=0)
    M_tck = interpolate.splrep(Ss, Ms)
    M_meshs = interpolate.splev(S_meshs, M_tck, der=0)
    R_tck = interpolate.splrep(Ss, Rs)
    R_meshs = interpolate.splev(S_meshs, R_tck, der=0)
    print("created the mesh for S!")

     # evaluate the H matrix
    print("calculating H[n, m]...")
    H = np.zeros((n_S,n_S))
    for n in range(n_S) :
        for m in range(1, n) :
            H[n, m], _ = calc_H(S_meshs[n], S_meshs[m], dS)
        H[n, n], error = calc_H(S_meshs[n], S_meshs[n], dS)
        print(f"H[{n},{n}] = {H[n,n]:.10f}, rel_error={error:.6E}...", end='\r')

    print("finished calculating H!")

    # put everything together and calculate f_l(S_n)
    print("calculating last crossing distribution...")
    last_crossing = np.zeros(n_S)
    last_crossing[0] = g_1(S_meshs[0])
    last_crossing[1] = (g_1(S_meshs[1])+last_crossing[0]*H[1,1])/(1-H[1,1])
    for n in range(2, len(last_crossing)) :
        sum = np.sum([(last_crossing[m]+last_crossing[m-1]) * H[n,m] for m in range(1,n)])
        last_crossing[n] = (g_1(S_meshs[n]) + last_crossing[n-1]*H[n,n] + sum)/(1-H[n,n])

    IMF = calc_IMF(R_meshs, S_meshs, B_meshs, M_meshs, last_crossing)
    print("calculation complete!")

    # export the IMF
    print("exporting the IMF...")
    filename_hdf5 = f"M{mach_h:.1f}p{p}B{bb}n{n_S}_dir.hdf5"
    try :
        h5 = h5py.File(filename_hdf5, 'a')
        h5.create_dataset('M',data=M_meshs)
        h5.create_dataset('IMF',data=IMF)
    except RuntimeError:
        # if the file is already occupied, save it to temp.hdf5
        print("file already exists!")
        filename_hdf5 = "temp.hdf5"
        h5 = h5py.File(filename_hdf5, 'w')
        h5.create_dataset('M',data=M_meshs)
        h5.create_dataset('IMF',data=IMF)
    finally :
        print(f"data written successfully in {filename_hdf5}!")
        print("normalisation parameters:")
        print(f"M_cloud={rho_0*h**3:.6E}")
        print(f"rho_0={rho_0:.6E}")
        h5.close()