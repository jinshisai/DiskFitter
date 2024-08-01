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

#from libc.math cimport sin, cos


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
    cdef cnp.ndarray[DTYPE_t, ndim=3] spec = np.zeros((nv, nx, ny))
    cdef double wsum
    cdef int i, j, k, l

    for l in range(nv):
        for i in range(nx):
            for j in range(ny):
                wsum = 0.
                for k in range(nz):
                    if (ve[l] <= vlos[i,j,k]) and ( ve[l+1] > vlos[i,j,k]):
                        wsum += w[i,j,k]
                spec[l,i,j] = wsum
    return spec


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

