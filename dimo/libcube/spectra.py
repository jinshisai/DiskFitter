import numpy as np
from numba import njit, prange
from math import erf


@njit(parallel=True)
def glnprof_series(v, v0, delv):
    '''
    Generate series of normalized Gaussian line profiles.

    Parameters
    ----------
     v (1D array): velocity axis
     v0 (1D array): series of line centre
     delv (1D array): series of linewidths
    '''
    nd = v0.size
    nv = len(v)
    dv_cell = v[1] - v[0]
    v_min, v_max = v[0], v[-1]

    lnprof = np.zeros((nv, nd))
    for i in prange(nd):
        profi = np.exp( - (v - v0[i])**2. / delv[i]**2.)
        sampled_fraction = 0.5 * (erf((v_max - v0[i]) / (delv[i])) # np.sqrt(2.)
            - erf((v_min - v0[i]) / (delv[i])))
        #print(sampled_fraction)
        lnprof[:,i] = profi / np.sum(profi * dv_cell) * sampled_fraction # / f_norm

    return lnprof


@njit(parallel=True)
def Tn_to_cube(Tg, n_gf, n_gr, lnprof_series, dz):
    nv, nx, ny, nz = lnprof_series.shape

    # output
    Tv_gf = np.zeros((nv, nx, ny))
    Tv_gr = np.zeros((nv, nx, ny))
    Nv_gf = np.zeros((nv, nx, ny))
    Nv_gr = np.zeros((nv, nx, ny))

    # loop
    for i in prange(nv):
        for j in range(nx):
            for k in range(ny):
                _Nv_gf = 0.
                _Nv_gr = 0.
                _Tv_gf = 0.
                _Tv_gr = 0.

                for l in range(nz):
                    lnpf_i = lnprof_series[i,j,k,l]
                    _Tg = Tg[j,k,l]
                    nv_gf = lnpf_i * n_gf[j,k,l]
                    nv_gr = lnpf_i * n_gr[j,k,l]

                    # sum up
                    _Nv_gf += nv_gf
                    _Nv_gr += nv_gr
                    _Tv_gf += _Tg * nv_gf
                    _Tv_gr += _Tg * nv_gr

                if _Nv_gf > 0.:
                    Nv_gf[i,j,k] = _Nv_gf * dz * 1.e-5
                    Tv_gf[i,j,k] = _Tv_gf / _Nv_gf

                if _Nv_gr > 0.:
                    Nv_gr[i,j,k] = _Nv_gr * dz * 1.e-5
                    Tv_gr[i,j,k] = _Tv_gr / _Nv_gr

    return Tv_gf, Tv_gr, Nv_gf, Nv_gr