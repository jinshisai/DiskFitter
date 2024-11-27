import numpy as np
from astropy import constants, units
from numba import jit, njit, prange


# constants
clight = constants.c.cgs.value        # light speed (cm s^-1)
kb     = constants.k_B.cgs.value      # Boltzman coefficient
hp     = constants.h.cgs.value        # Planck constant


@jit(nopython=True)
def glnprof(t0: float, 
    v: np.ndarray, 
    v0: float, delv: float, fn: float):
    '''
    Gaussian line profile with the linewidth definition of the Doppler broadening.

    Parameters
    ----------
     t0 (float): Total optical depth or integrated intensity.
     v (ndarray): Velocity axis.
     delv (float): Doppler linewidth.
     fn (float): A normalizing factor. t0 will be in units of inverse of velocity if fn is not given.
    '''
    exponent =  - (v - v0) **2 / delv**2
    return t0 / (np.sqrt(np.pi) * delv) * np.exp(exponent) * fn


@jit(nopython=True)
def interp(x, xp, fp):
    """
    Perform linear interpolation or extrapolation.

    Parameters:
    -----------
    x : float
        The x-value for which to interpolate.
    xp : ndarray
        The x-coordinates of the data points, must be sorted.
    fp : ndarray
        The y-coordinates of the data points.

    Returns:
    --------
    float
        Interpolated or extrapolated value at x.
    """
    if x < xp[0]:  # Below the first point, extrapolate
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        return fp[0] + slope * (x - xp[0])
    elif x > xp[-1]:  # Above the last point, extrapolate
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return fp[-1] + slope * (x - xp[-1])
    else:  # Interpolate within bounds
        for i in range(len(xp) - 1):
            if xp[i] <= x <= xp[i + 1]:
                return fp[i] + (fp[i + 1] - fp[i]) * (x - xp[i]) / (xp[i + 1] - xp[i])


@njit(parallel=True) # fastmath=True
def Tndv_to_cube(T_g, n_gf, n_gr, vlos, dv, ve, ds,
    freq, Aul, Eu, gu, Qgrid):
    """
    Compute a 4D cube from temperature, density, and velocity information,
    with optical depth calculation and integration termination based on tau.

    Parameters:
        T_g: 3D numpy array for temperature.
        n_gf: 3D numpy array for density (foreground).
        n_gr: 3D numpy array for density (rear side).
        vlos: 3D numpy array for line-of-sight velocities.
        dv: 3D numpy array for velocity dispersions.
        ve: 1D numpy array for velocity edges.
        ds: Step size (float).

    Returns:
        Tncube: 4D numpy array of the resulting cube.
    """
    nx, ny, nz = T_g.shape
    nv = len(ve) - 1
    delv = ve[1] - ve[0] # channel width
    Tncube = np.zeros((4, nx, ny, nv))

    for i in prange(nx):
        for j in range(ny):
            for l in range(nv):
                vl = 0.5 * (ve[l] + ve[l + 1])
                T_gf_sum = 0.0
                T_gr_sum = 0.0
                n_gf_sum = 0.0
                n_gr_sum = 0.0
                tau_gf = 0.0
                tau_gr = 0.0

                for k in range(nz):
                    if (tau_gf >= 30.0) and (tau_gr >= 30.0):
                        #print('Went over 30!')
                        break

                    dv_cell = dv[i, j, k]
                    vlos_cell = vlos[i, j, k]
                    if (vl - vlos_cell) ** 2 < 25.0 * dv_cell * 0.5:
                        T_cell = T_g[i, j, k]

                        # Foreground side
                        if tau_gf < 30.0:
                            n_v_gf = glnprof(n_gf[i, j, k] * ds, vl, vlos_cell, dv_cell, 1.0)
                            n_gf_sum += n_v_gf
                            T_gf_sum += T_cell * n_v_gf

                            if n_gf_sum > 0.0:
                                T_gf_mn = T_gf_sum / n_gf_sum # density-weighted mean temperature
                                Qrot = interp(T_gf_mn, Qgrid[0], Qgrid[1])
                                tau_gf = NT_to_tau(n_gf_sum, T_gf_mn, 
                                    freq, Aul, Eu, gu, Qrot, delv)

                        # Rear side
                        if tau_gr < 30.0:
                            n_v_gr = glnprof(n_gr[i, j, k] * ds, vl, vlos_cell, dv_cell, 1.0)
                            n_gr_sum += n_v_gr
                            T_gr_sum += T_cell * n_v_gr

                            if n_gr_sum > 0.0:
                                T_gr_mn = T_gf_sum / n_gf_sum # density-weighted mean temperature
                                Qrot = interp(T_gr_mn, Qgrid[0], Qgrid[1])
                                tau_gr = NT_to_tau(n_gr_sum, T_gr_mn, 
                                    freq, Aul, Eu, gu, Qrot, delv)

                # Calculate mean temperature and store results
                if n_gf_sum > 0.0:
                    Tncube[0, i, j, l] = T_gf_mn
                    Tncube[2, i, j, l] = tau_gf

                if n_gr_sum > 0.0:
                    Tncube[1, i, j, l] = T_gr_mn
                    Tncube[3, i, j, l] = tau_gr

    return Tncube


@jit(nopython=True)
def NT_to_tau(Ntot, Tex, freq, Aul, Eu, gu, Qrot, delv):
    '''
    Calculate the line optical depth.

    Parameters
    ----------
     N (float or ndarray): Number column density of the molecule (cm^-2).
     T (float or ndarray): Excitation temperature for the line (K).
     freq (float): Frequency of the line (Hz).
     Eu (float): Energy of the upper energy state (K).
     gu (float): g of the upper energy state.
     Aul (float): Einstein A coeffient of the transition.
     Qrot (float or ndarray): Partition function.
     delv (float): Linewidth or channel width (cm s^-1).
    '''
    return (clight*clight*clight)/(8.*np.pi*freq*freq*freq)*(gu/Qrot)\
    *np.exp(-Eu/Tex)*Ntot*Aul*(np.exp(hp*freq/(kb*Tex)) - 1.) / delv