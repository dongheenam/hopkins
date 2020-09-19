# dependencies
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate

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
rho = 1.31e-20       # mean mass density [g cm^-3]

""" numerical parameters """
R_start = h*1e2      # starting length scale for the random walk
R_end = h*1e-10      # ending length scale for the random walk
n_S = 5000           # number of mesh points for S-parametrisation

""" functions for calculating the important parameters """



""" functions for the direct calculation of the last-crossing distribution """
def P_0(x, var=1.0) :
    """ returns Gaussian pdf for given variance """
    if var <= 0 : # basic sanity check
        sys.exit(f"{var} is not valid variance!")
    prob = 1/np.sqrt(2*np.pi*var) * np.exp(x**2/(-2*var))
    return prob

def 

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

# calculate Alfven speed
v_A = B_mag/np.sqrt(4*np.pi*rho_0)