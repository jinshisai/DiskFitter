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