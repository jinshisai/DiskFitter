import numpy as np
cimport numpy as cnp
cimport cython # compile

# to use numpy array
cnp.import_array()

# Define data type
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

# laod functions from standard C library
cdef extern from "<math.h>": # from math
    DEF HUGE_VAL = 1e500     # define macro
    double exp(double x)     # function to use
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    double M_PI


# Constants (use inline values for Cython efficiency)
cdef double clight = 2.99792458e10  # cm/s
cdef double kb = 1.380649e-16       # erg/K
cdef double hp = 6.62607015e-27     # erg.s


# Temperature and tau to cube
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=4] Tt_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] T_g,
    cnp.ndarray[DTYPE_t, ndim=3] tau_rho_gf,
    cnp.ndarray[DTYPE_t, ndim=3] tau_rho_gr,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    double ds,
    ):
    cdef int nx = T_g.shape[0]
    cdef int ny = T_g.shape[1]
    cdef int nz = T_g.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=4] Ttcube = np.zeros((4, nv, nx, ny))
    cdef double T_gf_sum, T_gr_sum, tau_rho_gf_sum, tau_rho_gr_sum
    cdef int i, j, k, l

    for l in range(nv):
        for i in range(nx):
            for j in range(ny):
                T_gf_sum = 0.
                T_gr_sum = 0.
                tau_rho_gf_sum = 0.
                tau_rho_gr_sum = 0.
                for k in range(nz):
                    if (ve[l] <= vlos[i,j,k]) and ( ve[l+1] > vlos[i,j,k]):
                        T_gf_sum += T_g[i,j,k] * tau_rho_gf[i,j,k] * ds
                        T_gr_sum += T_g[i,j,k] * tau_rho_gr[i,j,k] * ds
                        tau_rho_gf_sum += tau_rho_gf[i,j,k] * ds
                        tau_rho_gr_sum += tau_rho_gr[i,j,k] * ds
                if tau_rho_gf_sum > 0.:
                    Ttcube[0,l,i,j] = T_gf_sum / tau_rho_gf_sum # weighted mean
                    Ttcube[2,l,i,j] = tau_rho_gf_sum # sum
                if tau_rho_gr_sum > 0.:
                    Ttcube[1,l,i,j] = T_gr_sum / tau_rho_gr_sum # weighted mean
                    Ttcube[3,l,i,j] = tau_rho_gr_sum # sum
    return Ttcube



# temperature, density and linewidth to cube
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=4] Tndv_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] T_g,
    cnp.ndarray[DTYPE_t, ndim=3] n_gf,
    cnp.ndarray[DTYPE_t, ndim=3] n_gr,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=3] dv,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    double ds,
    ):
    cdef int nx = T_g.shape[0]
    cdef int ny = T_g.shape[1]
    cdef int nz = T_g.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=4] Tncube = np.zeros((4, nx, ny, nv))
    cdef double n_v_gf, n_v_gr
    cdef double T_gf_sum, T_gr_sum, n_gf_sum, n_gr_sum
    cdef double vl
    cdef int i, j, k, l

    #for l in range(nv):
    for i in range(nx):
        for j in range(ny):
            for l in range(nv):
                vl = 0.5 * (ve[l] + ve[l+1])
                T_gf_sum = 0.
                T_gr_sum = 0.
                n_gf_sum = 0.
                n_gr_sum = 0.
                for k in range(nz):
                    # calculate smearing effect
                    # only when the velocity separation is less than 5 Gaussian sigma
                    if (vl - vlos[i,j,k]) * (vl - vlos[i,j,k]) < 25. * dv[i,j,k] * 0.5:
                        # fore side
                        # tau
                        n_v_gf = glnprof(
                            n_gf[i,j,k] * ds, vl, vlos[i,j,k],
                            dv[i,j,k], 1.
                            )
                        n_gf_sum += n_v_gf
                        # temperature
                        T_gf_sum += T_g[i,j,k] * n_v_gf

                        # rear side
                        # tau
                        n_v_gr = glnprof(
                            n_gr[i,j,k] * ds, vl, vlos[i,j,k],
                            dv[i,j,k], 1.
                            )
                        n_gr_sum += n_v_gr
                        # temperature
                        T_gr_sum += T_g[i,j,k] * n_v_gr

                if n_gf_sum > 0.:
                    Tncube[0,i,j,l] = T_gf_sum / n_gf_sum
                    Tncube[2,i,j,l] = n_gf_sum
                if n_gr_sum > 0.:
                    Tncube[1,i,j,l] = T_gr_sum / n_gr_sum 
                    Tncube[3,i,j,l] = n_gr_sum

    return Tncube



# temperature, density and linewidth to cube
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=4] Tndv_to_Ttaucube(
    cnp.ndarray[DTYPE_t, ndim=3] T_g,
    cnp.ndarray[DTYPE_t, ndim=3] n_gf,
    cnp.ndarray[DTYPE_t, ndim=3] n_gr,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=3] dv,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    double ds,
    double freq, double Aul, double Eu, double gu, double[:] T_Q, double[:] PF
    ):

    # variables
    cdef int nx = T_g.shape[0]
    cdef int ny = T_g.shape[1]
    cdef int nz = T_g.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=4] Ttau_cube = np.zeros((4, nx, ny, nv))
    cdef double n_v_gf, n_v_gr
    cdef double T_gf_sum, T_gr_sum, n_gf_sum, n_gr_sum, tau_gf, tau_gr
    cdef double vl, dv_cell, vlos_cell, T_cell
    cdef double delv = ve[1] - ve[0]
    cdef int i, j, k, l

    #for l in range(nv):
    for i in range(nx):
        for j in range(ny):
            for l in range(nv):
                vl = 0.5 * (ve[l] + ve[l+1])
                T_gf_sum = 0.
                T_gr_sum = 0.
                n_gf_sum = 0.
                n_gr_sum = 0.
                tau_gf = 0.0
                tau_gr = 0.0

                for k in range(nz):
                    if (tau_gf >= 30.0) and (tau_gr >= 30.0):
                        break

                    # calculate smearing effect
                    # only when the velocity separation is less than 5 Gaussian sigma
                    dv_cell = dv[i, j, k]
                    vlos_cell = vlos[i, j, k]
                    if (vl - vlos_cell ) * (vl - vlos_cell ) < 25. * dv_cell * 0.5:
                        T_cell = T_g[i, j, k]

                        # fore side
                        if tau_gf < 30.0:
                            n_v_gf = glnprof(
                                n_gf[i,j,k] * ds, vl, vlos_cell,
                                dv_cell, 1.
                                )
                            n_gf_sum += n_v_gf
                            # temperature
                            T_gf_sum += T_cell * n_v_gf

                            if n_gf_sum > 0.0:
                                T_gf_mn = T_gf_sum / n_gf_sum # density-weighted mean temperature
                                Qrot = interp(T_gf_mn, T_Q, PF)
                                tau_gf = NT_to_tau(n_gf_sum, T_gf_mn, 
                                    freq, Aul, Eu, gu, Qrot, delv)

                        # rear side
                        if tau_gr < 30.0:
                            n_v_gr = glnprof(
                                n_gr[i,j,k] * ds, vl, vlos_cell,
                                dv_cell, 1.
                                )
                            n_gr_sum += n_v_gr
                            # temperature
                            T_gr_sum += T_cell * n_v_gr

                            if n_gr_sum > 0.0:
                                T_gr_mn = T_gf_sum / n_gf_sum # density-weighted mean temperature
                                Qrot = interp(T_gr_mn, T_Q, PF)
                                tau_gr = NT_to_tau(n_gr_sum, T_gr_mn, 
                                    freq, Aul, Eu, gu, Qrot, delv)

                if n_gf_sum > 0.:
                    Ttau_cube[0,i,j,l] = T_gf_sum / n_gf_sum
                    Ttau_cube[2,i,j,l] = tau_gf
                if n_gr_sum > 0.:
                    Ttau_cube[1,i,j,l] = T_gr_sum / n_gr_sum 
                    Ttau_cube[3,i,j,l] = tau_gr

    return Ttau_cube


# Gaussian line broadening
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef inline double glnprof(
    double t0,
    double v,
    double v0,
    double delv,
    double fn):
    '''
    Gaussian line profile with the linewidth definition of the Doppler broadening.

    Parameters
    ----------
     t0 (ndarray): Total optical depth or integrated intensity.
     v (ndarray): Velocity axis.
     delv (float): Doppler linewidth.
     fn (float): A normalizing factor. t0 will be in units of velocity if fn is not given.
    '''
    return t0 / sqrt(M_PI) / delv * exp( - (v - v0) * (v - v0) / delv / delv) * fn


# linear interpolation
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef inline double interp(
    double x,
    double[:] xp,
    double[:] fp):
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
    cdef int i
    cdef int n_xp = len(xp)
    cdef double slope

    if x < xp[0]:  # Below the first point, extrapolate
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        return fp[0] + slope * (x - xp[0])
    elif x > xp[n_xp-1]:  # Above the last point, extrapolate
        slope = (fp[n_xp-1] - fp[n_xp-2]) / (xp[n_xp-1] - xp[n_xp-2])
        return fp[n_xp-1] + slope * (x - xp[n_xp-1])
    else:  # Interpolate within bounds
        for i in range(n_xp - 1):
            if xp[i] <= x <= xp[i + 1]:
                return fp[i] + (fp[i + 1] - fp[i]) * (x - xp[i]) / (xp[i + 1] - xp[i])


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef inline double NT_to_tau(
    double Ntot, 
    double Tex, 
    double freq, 
    double Aul, 
    double Eu, 
    double gu, 
    double Qrot, 
    double delv):
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
    return (clight*clight*clight)/(8.*M_PI*freq*freq*freq)*(gu/Qrot)\
    *exp(-Eu/Tex)*Ntot*Aul*(exp(hp*freq/(kb*Tex)) - 1.) / delv


# to cube
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] tocube(
    cnp.ndarray[DTYPE_t, ndim=2] tau, 
    cnp.ndarray[DTYPE_t, ndim=2] vlos, 
    cnp.ndarray[DTYPE_t, ndim=1] ve):
    cdef int ny = tau.shape[0]
    cdef int nx = tau.shape[1]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=3] tau_v = np.zeros([nv, ny, nx], dtype = DTYPE)
    cdef int i, j, k

    for k in range(nv):
        for j in range(ny):
            for i in range(nx):
                if (ve[k] <= vlos[j,i]) and ( ve[k+1] > vlos[j,i]):
                    tau_v[k, j, i] = tau[j,i]
    return tau_v


# solve radiative transfer for three-layer model
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] solve_3LRT(
    cnp.ndarray[DTYPE_t, ndim=2] Sv_gf, 
    cnp.ndarray[DTYPE_t, ndim=2] Sv_gr, 
    cnp.ndarray[DTYPE_t, ndim=2] Sv_d, 
    cnp.ndarray[DTYPE_t, ndim=3] tau_gf,
    cnp.ndarray[DTYPE_t, ndim=3] tau_gr,
    cnp.ndarray[DTYPE_t, ndim=2] tau_d,
    double Sv_bg,
    int nv):
    cdef int ny = Sv_d.shape[0]
    cdef int nx = Sv_d.shape[1]
    cdef double Iv_d
    cdef cnp.ndarray[DTYPE_t, ndim=3] Iv = np.zeros((nv, ny, nx))
    cdef int i, j, k

    for k in range(nv):
        for j in range(ny):
            for i in range(nx):
                Iv_d = (Sv_d[j,i] - Sv_bg) * (1. - exp(- tau_d[j,i]))
                Iv[k, j, i] = Sv_bg * (
                    exp(- tau_gf[k,j,i] - tau_d[j,i] - tau_gr[k,j,i]) - 1.) \
                + Sv_gr[j,i] * (1. - exp(- tau_gr[k,j,i])) \
                * exp(- tau_gf[k,j,i]  - tau_d[j,i] ) \
                + Sv_d[j,i] * (1. - exp(- tau_d[j,i])) * exp(- tau_gf[k,j,i]) \
                + Sv_gf[j,i] * (1. - exp(- tau_gf[k,j,i])) \
                - Iv_d
    return Iv


# solve radiative transfer for multi-layer model
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] solve_MLRT(
    cnp.ndarray[DTYPE_t, ndim=3] Sv_gf, 
    cnp.ndarray[DTYPE_t, ndim=3] Sv_gr, 
    cnp.ndarray[DTYPE_t, ndim=2] Sv_d, 
    cnp.ndarray[DTYPE_t, ndim=3] tau_gf,
    cnp.ndarray[DTYPE_t, ndim=3] tau_gr,
    cnp.ndarray[DTYPE_t, ndim=2] tau_d,
    double Sv_bg,
    int nv):
    cdef int ny = Sv_d.shape[0]
    cdef int nx = Sv_d.shape[1]
    cdef double Iv_d
    cdef cnp.ndarray[DTYPE_t, ndim=3] Iv = np.zeros((nv, ny, nx))
    cdef int i, j, k

    for k in range(nv):
        for j in range(ny):
            for i in range(nx):
                Iv_d = (Sv_d[j,i] - Sv_bg) * (1. - exp(- tau_d[j,i]))
                Iv[k, j, i] = Sv_bg * (
                    exp(- tau_gf[k,j,i] - tau_d[j,i] - tau_gr[k,j,i]) - 1.) \
                + Sv_gr[k,j,i] * (1. - exp(- tau_gr[k,j,i])) \
                * exp(- tau_gf[k,j,i]  - tau_d[j,i] ) \
                + Sv_d[j,i] * (1. - exp(- tau_d[j,i])) * exp(- tau_gf[k,j,i]) \
                + Sv_gf[k,j,i] * (1. - exp(- tau_gf[k,j,i])) \
                - Iv_d
    return Iv


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] collapse_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] w,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    ):
    cdef int nx = w.shape[0]
    cdef int ny = w.shape[1]
    cdef int nz = w.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=3] cube = np.zeros((nv, nx, ny))
    cdef double wsum
    cdef int i, j, k, l

    for l in range(nv):
        for i in range(nx):
            for j in range(ny):
                wsum = 0.
                for k in range(nz):
                    if (ve[l] <= vlos[i,j,k]) and ( ve[l+1] > vlos[i,j,k]):
                        wsum += w[i,j,k]
                cube[l,i,j] = wsum
    return cube


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] integrate_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] w,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    double ds
    ):
    cdef int nx = w.shape[0]
    cdef int ny = w.shape[1]
    cdef int nz = w.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=3] cube = np.zeros((nv, nx, ny))
    cdef double wsum
    cdef int i, j, k, l

    for l in range(nv):
        for i in range(nx):
            for j in range(ny):
                wsum = 0.
                for k in range(nz):
                    if (ve[l] <= vlos[i,j,k]) and ( ve[l+1] > vlos[i,j,k]):
                        wsum += w[i,j,k] * ds
                cube[l,i,j] = wsum
    return cube


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] average_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] w,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    ):
    cdef int nx = w.shape[0]
    cdef int ny = w.shape[1]
    cdef int nz = w.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=3] cube = np.zeros((nv, nx, ny))
    cdef double wsum, navg
    cdef int i, j, k, l

    for l in range(nv):
        for i in range(nx):
            for j in range(ny):
                wsum = 0.
                navg = 0.
                for k in range(nz):
                    if (ve[l] <= vlos[i,j,k]) and ( ve[l+1] > vlos[i,j,k]):
                        wsum += w[i,j,k]
                        navg =+ 1.
                if navg > 0.:
                    cube[l,i,j] = wsum / navg
    return cube


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=3] waverage_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] q,
    cnp.ndarray[DTYPE_t, ndim=3] w,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    ):
    '''
    Collapse 4D data into cube through weighted averaging.
    '''
    cdef int nx = w.shape[0]
    cdef int ny = w.shape[1]
    cdef int nz = w.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=3] cube = np.zeros((nv, nx, ny))
    cdef double wsum, navg
    cdef int i, j, k, l

    for l in range(nv):
        for i in range(nx):
            for j in range(ny):
                qwsum = 0.
                wsum = 0.
                for k in range(nz):
                    if (ve[l] <= vlos[i,j,k]) and ( ve[l+1] > vlos[i,j,k]):
                        qwsum += q[i,j,k] * w[i,j,k]
                        wsum += w[i,j,k]
                if wsum > 0.:
                    cube[l,i,j] = qwsum / wsum
    return cube


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=2] xrot(
    cnp.ndarray[DTYPE_t, ndim=1] x, 
    cnp.ndarray[DTYPE_t, ndim=1] y, 
    cnp.ndarray[DTYPE_t, ndim=1] z, 
    double inc):
    cdef int nsize = x.size
    cdef cnp.ndarray[DTYPE_t, ndim=2] xyzp = np.zeros((3, nsize))
    cdef double cosi = cos(inc)
    cdef double sini = sin(inc)
    for i in range(nsize):
        xyzp[0,i] = x[i]
        xyzp[1,i] = y[i] * cosi - z[i] * sini
        xyzp[2,i] = y[i] * sini + z[i] * cosi
    return xyzp


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=4] xrot3D(
    cnp.ndarray[DTYPE_t, ndim=3] x, 
    cnp.ndarray[DTYPE_t, ndim=3] y, 
    cnp.ndarray[DTYPE_t, ndim=3] z, 
    double inc):
    cdef int nx = x.shape[0]
    cdef int ny = x.shape[1]
    cdef int nz = x.shape[2]
    cdef cnp.ndarray[DTYPE_t, ndim=2] xyzp = np.zeros((3, nx,ny,nz))
    cdef double cosi = cos(inc)
    cdef double sini = sin(inc)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                xyzp[0,i,j,k] = x[i,j,k]
                xyzp[1,i,j,k] = y[i,j,k] * cosi - z[i,j,k] * sini
                xyzp[2,i,j,k] = y[i,j,k] * sini + z[i,j,k] * cosi
    return xyzp

