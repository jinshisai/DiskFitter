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


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[DTYPE_t, ndim=4] Ttdv_to_cube(
    cnp.ndarray[DTYPE_t, ndim=3] T_g,
    cnp.ndarray[DTYPE_t, ndim=3] tau_rho_gf,
    cnp.ndarray[DTYPE_t, ndim=3] tau_rho_gr,
    cnp.ndarray[DTYPE_t, ndim=3] vlos,
    cnp.ndarray[DTYPE_t, ndim=3] dv,
    cnp.ndarray[DTYPE_t, ndim=1] ve,
    double ds,
    ):
    cdef int nx = T_g.shape[0]
    cdef int ny = T_g.shape[1]
    cdef int nz = T_g.shape[2]
    cdef int nv = len(ve) - 1
    cdef cnp.ndarray[DTYPE_t, ndim=4] Ttcube = np.zeros((4, nx, ny, nv))
    cdef double tau_v_gf, tau_v_gr
    cdef double T_gf_sum, T_gr_sum, tau_rho_gf_sum, tau_rho_gr_sum
    cdef double vl
    cdef int i, j, k, l

    #for l in range(nv):
    for i in range(nx):
        for j in range(ny):
            for l in range(nv):
                vl = 0.5 * (ve[l] + ve[l+1])
                T_gf_sum = 0.
                T_gr_sum = 0.
                tau_rho_gf_sum = 0.
                tau_rho_gr_sum = 0.
                for k in range(nz):
                    # calculate smearing effect
                    # only when the velocity separation is less than 5 Gaussian sigma
                    if (vl - vlos[i,j,k]) * (vl - vlos[i,j,k]) < 25. * dv[i,j,k] * 0.5:
                        # fore side
                        # tau
                        tau_v_gf = glnprof(
                            tau_rho_gf[i,j,k] * ds, vl, vlos[i,j,k],
                            dv[i,j,k], 1.
                            )
                        tau_rho_gf_sum += tau_v_gf
                        # temperature
                        T_gf_sum += T_g[i,j,k] * tau_v_gf

                        # rear side
                        # tau
                        tau_v_gr = glnprof(
                            tau_rho_gr[i,j,k] * ds, vl, vlos[i,j,k],
                            dv[i,j,k], 1.
                            )
                        tau_rho_gr_sum += tau_v_gr
                        # temperature
                        T_gr_sum += T_g[i,j,k] * tau_v_gr

                if tau_rho_gf_sum > 0.:
                    Ttcube[0,i,j,l] = T_gf_sum / tau_rho_gf_sum
                    Ttcube[2,i,j,l] = tau_rho_gf_sum
                if tau_rho_gr_sum > 0.:
                    Ttcube[1,i,j,l] = T_gr_sum / tau_rho_gr_sum 
                    Ttcube[3,i,j,l] = tau_rho_gr_sum

    return Ttcube


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef double glnprof(
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