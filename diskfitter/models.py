# import modules
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import root, minimize
from scipy.signal import convolve
from astropy import constants, units
import dataclasses
from dataclasses import dataclass
import time

from .funcs import beam_convolution, gaussian2d, glnprof_conv
from .grid import Nested2DGrid, Nested1DGrid, SubGrid2D


### constants
Ggrav  = constants.G.cgs.value        # Gravitational constant
Msun   = constants.M_sun.cgs.value    # Solar mass (g)
Lsun   = constants.L_sun.cgs.value    # Solar luminosity (erg s^-1)
Rsun   = constants.R_sun.cgs.value    # Solar radius (cm)
clight = constants.c.cgs.value        # light speed (cm s^-1)
kb     = constants.k_B.cgs.value      # Boltzman coefficient
sigsb  = constants.sigma_sb.cgs.value # Stefan-Boltzmann constant (erg s^-1 cm^-2 K^-4)
mH     = constants.m_p.cgs.value      # Proton mass (g)
hp     = constants.h.cgs.value # Planck constant [erg s]

# unit
auTOcm = units.au.to('cm') # 1 au (cm)


@dataclass(slots=True)
class ThreeLayerDisk:

    # params for dust layer
    Td0: float = 400.
    qd: float = 0.5
    log_tau_dc: float = 0. # tau of dust at rc
    rc_d: float = 1.
    gamma_d: float = 1.
    # params for gas layer
    Tg0: float = 400. # Gas temperature at r0
    qg: float = 0.5 # power-law index of temperature distribution
    log_tau_gc: float = 0.
    rc_g: float = 1.
    gamma_g: float = 1.
    z0: float = 0.
    hp: float = 1.25
    # geometry & velocity
    inc: float = 0.
    pa: float = 0.
    ms: float = 1.
    vsys: float = 0.
    # positional offsets
    dx0: float = 0.
    dy0: float = 0.
    # reference radius
    r0: float = 1.
    # line width
    delv: float = 0.

    def set_params(self, 
        Td0 = 400., qd = 0.5, log_tau_dc = 0., rc_d = 100., gamma_d = 1., 
        Tg0 = 400., qg = 0.5, log_tau_gc = 0., rc_g = 100., gamma_g = 1., 
        z0 = 0., hp = 1.25, inc = 0., pa = 0., ms = 1., vsys = 0, 
        dx0=0., dy0=0., r0 = 1., delv = 0.):
        '''

        Parameters
        ----------
         Td0
         qd
         Tg0 (float): K
         qg (float):
         z0 (float): au
         hp (float):
         r0 (float): au
         tau_dc (float):
         rc_d (float): au
         gamma_d (float):
         tau_gc (float):
         rc_g (float): au
         gamma_g (float):
         inc (float): deg
         pa (float): deg
         ms (float): Msun
         vsys (float): km/s
        '''
        # initialize parameters
        # dust layer
        self.Td0 = Td0
        self.qd  = qd
        self.log_tau_dc = log_tau_dc
        self.rc_d = rc_d
        self.gamma_d = gamma_d
        # gas layer
        self.Tg0 = Tg0 # gas temperature
        self.qg = qg
        self.log_tau_gc = log_tau_gc
        self.rc_g = rc_g
        self.gamma_g = gamma_g
        self.z0 = z0
        self.hp = hp
        # geometry & velocity
        self.inc = inc
        self.pa = pa
        self.ms = ms
        self.vsys = vsys
        # positional offsets
        self.dx0 = dx0
        self.dy0 = dy0
        # reference radius
        self.r0 = r0
        # line width
        self.delv = delv


    def get_paramkeys(self):
        return list(self.__annotations__.keys())


    def print_params(self):
        fields = dataclasses.fields(self)
        for v in fields:
            print(f'{v.name}: ({v.type.__name__}) = {getattr(self, v.name)}')


    def build(self, xx_sky, yy_sky, rin = 0.1):
        '''
        Build a model given sky coordinates and return a info for making a image cube.
        '''
        # parameters
        _inc_rad = np.radians(self.inc)
        _pa_rad = np.radians(self.pa)
        _fz = lambda r, z0, r0, hp: z0*(r/r0)**hp
        _dfz = lambda x, y, z0, r0, hp: 2. * y * 0.5 / np.sqrt(x*x + y*y) \
        / r0 * z0 * hp * (np.sqrt(x*x + y*y)/r0)*(hp - 1.)

        # calculate temperature (T), velocity (v) and tau (t)
        def get_Tvt(xx_sky, yy_sky, 
            _fz, zargs, inc, pa, ms, T0, q, r0, tau_c, rc, gamma, _dfz=None):
            # deprojection
            depr = sky_to_local(xx_sky, yy_sky, 
                inc, pa + 0.5 * np.pi, _fz, 
                zargs, _dfz, zarg_lims = [[-0.3, 0.3], [0.1, 100.1], [0., 2.]])
            if type(depr) == int:
                T = np.full(xx_sky.shape, 1.) # 1 instead of zero to prevent numerical errors
                vlos  = np.zeros(xx_sky.shape)
                tau = np.zeros(xx_sky.shape)
                return T, vlos, tau
            else:
                xx, yy = depr

            # local coordinates
            rr = np.sqrt(xx * xx + yy * yy) # radius
            phph = np.arctan2(yy, xx) # azimuthal angle (rad)
            zz = _fz(rr, *zargs) # height
            # prevent r=0
            rr[rr == 0.] = np.nan

            # quantities
            # temperature
            T = T0 * (rr / r0)**(-q)
            T[np.isnan(T)] = 1. # to prevent computational errors
            T[T <= 1.] = 1. # safty net

            # line of sight velocity
            # take y-axis as the line of sight
            vlos = vkep(rr * auTOcm, ms * Msun, zz * auTOcm) \
            * np.cos(phph) * np.sin(_inc_rad) * 1.e-5 # cm/s --> km/s

            # tau
            tau = ssdisk(rr, tau_c, rc, gamma, beta = None)
            tau[np.isnan(tau)] = 0.  # to prevent computational errors
            tau[tau < 0.] = 0. # safty net

            T[rr < rin] = 0.
            vlos[rr < rin] = 0.
            tau[rr < rin] = 0.

            return T, vlos, tau


        # front gas layer
        T_gf, vlos_gf, tau_gf = get_Tvt(
            xx_sky - self.dx0, yy_sky - self.dy0, 
            _fz, [self.z0, self.r0, self.hp],
            _inc_rad, _pa_rad, self.ms,
            self.Tg0, self.qg, self.r0,
            10.**self.log_tau_gc, self.rc_g, self.gamma_g,)

        # dust layer
        T_d, _, tau_d = get_Tvt(
            xx_sky - self.dx0, yy_sky - self.dy0, 
            _fz, [0., self.r0, self.hp],
            _inc_rad, _pa_rad, self.ms,
            self.Td0, self.qd, self.r0,
            10.**self.log_tau_dc, self.rc_d, self.gamma_d,)

        # rear gas layer
        T_gr, vlos_gr, tau_gr = get_Tvt(
            xx_sky - self.dx0, yy_sky - self.dy0, 
            _fz, [-self.z0, self.r0, self.hp],
            _inc_rad, _pa_rad, self.ms,
            self.Tg0, self.qg, self.r0,
            10.**self.log_tau_gc, self.rc_g, self.gamma_g,)

        return T_gf, T_gr, T_d, vlos_gf, vlos_gr, tau_gf, tau_gr, tau_d


    def build_cube(self, xx, yy, v, 
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,):
        # get quantities
        T_gf, T_gr, T_d, vlos_gf, vlos_gr, tau_gf, tau_gr, tau_d = self.build(xx, yy)
        vlos_gf += self.vsys
        vlos_gr += self.vsys

        # velocity grid
        ny, nx = xx.shape
        dx = xx[0,1] - xx[0,0]
        dy = yy[1,0] - yy[0,0]
        nv = len(v)
        dv = np.mean(v[1:] - v[:-1])
        ve = np.hstack([v - dv * 0.5, v[-1] + 0.5 * dv])

        # nested velocity grid
        #nstg = Nested1DGrid(v)
        #nstg.nest(5)
        #v_nst = nstg.x_sub
        #ve_nst = nstg.xe_sub
        #nv_nst = nstg.nx_sub

        # making a cube
        _tau_gf = np.zeros((nv, ny, nx))
        _tau_gr = np.zeros((nv, ny, nx))
        _Bv = lambda T, v: Bvppx(T, v, dx, dy, dist = dist, au = True)
        #_Bv = lambda T, v: Bv(T, v)
        # calculate tau_v
        for i in range(nv):
            #print('vrange: %.2f-%.2f'%(ve[i],ve[i+1]))
            # front side
            _tau_gf[i,:,:] = np.where(
                (ve[i] <= vlos_gf) & (vlos_gf < ve[i+1]),
                tau_gf, 0.
                )
            # rear side
            _tau_gr[i,:,:] = np.where(
                (ve[i] <= vlos_gr) & (vlos_gr < ve[i+1]),
                tau_gr, 0.
                )
        # line width
        if self.delv > 0.:
            _tau_gf = glnprof_conv(_tau_gf, v, self.delv)
            _tau_gr = glnprof_conv(_tau_gr, v, self.delv)
        # get it back to original grid
        #_tau_gf = nstg.binning_onsubgrid(_tau_gf)
        #_tau_gr = nstg.binning_onsubgrid(_tau_gr)

        # radiative transfer
        _Bv_cmb = _Bv(Tcmb, f0)
        _Bv_gf  = _Bv(T_gf, f0)
        _Bv_gr  = _Bv(T_gr, f0)
        _Bv_d   = _Bv(T_d, f0)
        Idust = (_Bv_d - _Bv_cmb) * (1. - np.exp(- tau_d))
        I_cube = np.array([
            _Bv_cmb * (np.exp(- _tau_gf[i,:,:] - tau_d - _tau_gr[i,:,:]) - 1.) \
            + _Bv_gr * (1. - np.exp(- _tau_gr[i,:,:])) * np.exp(- _tau_gf[i,:,:] - tau_d) \
            + _Bv_d * (1. - np.exp(- tau_d)) * np.exp(- _tau_gf[i,:,:]) \
            + _Bv_gf * (1. - np.exp(- _tau_gf[i,:,:])) \
            - Idust # contsub
            for i in range(nv)])

        # Convolve beam if given
        if beam is not None:
            I_cube = beam_convolution(xx, yy, I_cube, [beam[0] * dist, beam[1] * dist, beam[2]])

        return I_cube


    def build_cube_subgrid(self, xx, yy, v, nsub = 2,
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,):
        if nsub < 1:
            print('ERROR\tbuild_cube_subgrid: nsub must be >= 2.')
            return 0
        subgrid = SubGrid2D(xx, yy, nsub = nsub)
        _xx, _yy = subgrid.xx_sub, subgrid.yy_sub
        I_cube_sub = self.build_cube(_xx, _yy, v, beam, dist, Tcmb, f0)
        nv = len(v)
        ny, nx = xx.shape
        return subgrid.binning_onsubgrid_layered(I_cube_sub)


    def build_nested_cube(self, xx, yy, v, xscale, yscale, n_subgrid,
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,):
        # original grid
        ny, nx = xx.shape

        # nested grid
        nstgrid = Nested2DGrid(xx, yy)
        xlim = [- np.nanmax(xx) * xscale, np.nanmax(xx) * yscale]
        ylim = [- np.nanmax(yy) * xscale, np.nanmax(yy) * yscale]
        xx_sub, yy_sub = nstgrid.nest(xlim, ylim, n_subgrid)

        # cube on the original grid
        I_cube = self.build_cube(xx, yy, v, 
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0,)

        # cube on the nested grid
        I_cube_sub = self.build_cube(xx_sub, yy_sub, v, 
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0,)

        # cut off edge
        xi, yi, xi0, yi0 = nstgrid.edgecut_indices(
            beam[0] * dist * 1.3, beam[0] * dist * 1.3) # * 1.5
        I_cube_sub = I_cube_sub[:, yi:-yi, xi:-xi]
        # replace
        for i in range(len(v)):
            I_cube[i, yi0:-yi0, xi0:-xi0] = \
            nstgrid.binning_onsubgrid(I_cube_sub[i,:,:])

        return I_cube



@dataclass(slots=True)
class SingleLayerDisk:

    # params for the disk
    # whichever gas or dust
    T0: float = 400. # temperature
    q: float = 0.5   # slope of temperature prof
    log_tau_c: float = 0. # tau at rc
    rc: float = 100.
    gamma: float = 1.
    # make it flared if you like
    z0: float = 0.
    hp: float = 1.25
    # geometry
    inc: float = 0.
    pa: float = 0.
    # stellar mass and vsys matter only for line case
    ms: float = 1.
    vsys: float = 0.
    # positional offsets
    dx0: float = 0.
    dy0: float = 0.
    # reference radius
    r0: float = 1.
    # line width
    delv: float = 0.


    def set_params(self, 
        T0 = 400., q = 0.5, log_tau_c = 0., rc = 100., gamma = 1., 
        z0 = 0., hp = 1.25, inc = 0., pa = 0., ms = 1., vsys = 0., 
        dx0=0., dy0=0., r0 = 1., delv = 0.):
        '''

        Parameters
        ----------
         T0
         q
         z0 (float): au
         hp (float):
         r0 (float): au
         log_tau_c (float):
         rc (float): au
         gamma (float):
         inc (float): deg
         pa (float): deg
         ms (float): Msun
         vsys (float): km/s
        '''
        # initialize parameters
        # dust layer
        self.T0 = T0
        self.q  = q
        self.log_tau_c = log_tau_c
        self.rc = rc
        self.gamma = gamma
        # height
        self.z0 = z0
        self.hp = hp
        # geometry
        self.inc = inc
        self.pa = pa
        # velocity
        self.ms = ms
        self.vsys = vsys
        # positional offsets
        self.dx0 = dx0
        self.dy0 = dy0
        # reference radius
        self.r0 = r0
        # line width
        self.delv = delv


    def get_paramkeys(self):
        return list(self.__annotations__.keys())


    def print_params(self):
        fields = dataclasses.fields(self)
        for v in fields:
            print(f'{v.name}: ({v.type.__name__}) = {getattr(self, v.name)}')


    def build(self, xx_sky, yy_sky, rin = 0.1):
        '''
        Build a model given sky coordinates and return a info for making a image cube.
        '''
        # parameters
        _inc_rad = np.radians(self.inc)
        _pa_rad = np.radians(self.pa)
        _fz = lambda r, z0, r0, hp: z0*(r/r0)**hp
        _dfz = lambda x, y, z0, r0, hp: 2. * y * 0.5 / np.sqrt(x*x + y*y) \
        / r0 * z0 * hp * (np.sqrt(x*x + y*y)/r0)*(hp - 1.)

        # calculate temperature (T), velocity (v) and tau (t)
        def get_Tvt(xx_sky, yy_sky, 
            _fz, zargs, inc, pa, ms, T0, q, r0, tau_c, rc, gamma, _dfz=None):
            # deprojection
            #print('Start deprojection', self.z0, self.hp, self.inc)
            depr = sky_to_local(xx_sky, yy_sky, 
                inc, pa + 0.5 * np.pi, _fz, 
                zargs, _dfz, zarg_lims = [[-0.3, 0.3], [0.1, 100.1], [0., 2.]]) # inc_max = 85.
            #print('Done deprojection')
            if type(depr) == int:
                T = np.full(xx_sky.shape, 1.) # 1 instead of zero to prevent numerical errors
                vlos  = np.zeros(xx_sky.shape)
                tau = np.zeros(xx_sky.shape)
                return T, vlos, tau
            else:
                xx, yy = depr

            # local coordinates
            rr = np.sqrt(xx * xx + yy * yy) # radius
            phph = np.arctan2(yy, xx) # azimuthal angle (rad)
            zz = _fz(rr, *zargs) # height
            # prevent r=0
            rr[rr == 0.] = np.nan

            # quantities
            # temperature
            T = T0 * (rr / r0)**(-q)
            T[np.isnan(T)] = 1. # to prevent computational errors
            T[T <= 1.] = 1. # safty net

            # line of sight velocity
            # take y-axis as the line of sight
            vlos = vkep(rr * auTOcm, ms * Msun, zz * auTOcm) \
            * np.cos(phph) * np.sin(_inc_rad) * 1.e-5 # cm/s --> km/s

            # tau
            tau = ssdisk(rr, tau_c, rc, gamma, beta = None)
            tau[np.isnan(tau)] = 0.  # to prevent computational errors
            tau[tau < 0.] = 0. # safty net

            T[rr < rin] = 0.
            vlos[rr < rin] = 0.
            tau[rr < rin] = 0.

            return T, vlos, tau


        # for a layer
        T, vlos, tau = get_Tvt(
            xx_sky - self.dx0, yy_sky - self.dy0, 
            _fz, [self.z0, self.r0, self.hp],
            _inc_rad, _pa_rad, self.ms,
            self.T0, self.q, self.r0,
            10.**self.log_tau_c, self.rc, self.gamma,)

        return T, vlos, tau


    def build_cube(self, xx, yy, v, 
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,
        return_tau = False):
        # get quantities
        T, vlos, tau = self.build(xx, yy)
        vlos += self.vsys

        # velocity grid
        ny, nx = xx.shape
        dx = xx[0,1] - xx[0,0]
        dy = yy[1,0] - yy[0,0]
        nv = len(v)
        dv = np.mean(v[1:] - v[:-1])
        ve = np.hstack([v - dv * 0.5, v[-1] + 0.5 * dv])

        # making a cube
        _tau = np.zeros((nv, ny, nx))
        _Bv = lambda T, v: Bvppx(T, v, dx, dy, dist = dist, au = True)
        # calculate tau_v
        for i in range(nv):
            #print('vrange: %.2f-%.2f'%(ve[i],ve[i+1]))
            _tau[i,:,:] = np.where(
                (ve[i] <= vlos) & (vlos < ve[i+1]),
                tau, 0.
                )
        # line width
        if self.delv > 0.:
            _tau = glnprof_conv(_tau, v, self.delv)

        if return_tau:
            return _tau #np.exp(- _tau) #(1. - np.exp(- _tau))

        #_tau[_tau > 0.] = 1000.
        # radiative transfer
        _Bv_bg = np.tile(_Bv(Tcmb, f0), (nv,1,1))
        _Bv_T  = np.tile(_Bv(T, f0), (nv,1,1))
        I_cube = (_Bv_T - _Bv_bg) * (1. - np.exp(- _tau))

        # Convolve beam if given
        if beam is not None:
            I_cube = beam_convolution(xx, yy, I_cube, 
                [beam[0] * dist, beam[1] * dist, beam[2]])

        # return intensity
        return I_cube


    def build_nested_cube(self, xx, yy, v, xscale, yscale, n_subgrid,
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230., return_tau = False):
        # original grid
        ny, nx = xx.shape

        # nested grid
        nstgrid = Nested2DGrid(xx, yy)
        xlim = [- np.nanmax(xx) * xscale, np.nanmax(xx) * yscale]
        ylim = [- np.nanmax(yy) * xscale, np.nanmax(yy) * yscale]
        xx_sub, yy_sub = nstgrid.nest(xlim, ylim, n_subgrid)

        # cube on the original grid
        I_cube = self.build_cube(xx, yy, v, 
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0, return_tau = return_tau)

        # cube on the nested grid
        I_cube_sub = self.build_cube(xx_sub, yy_sub, v, 
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0, return_tau = return_tau)

        # cut off edge
        if (beam is not None) & (return_tau == False):
            xi, yi, xi0, yi0 = nstgrid.edgecut_indices(beam[0] * dist * 1.5, beam[0] * dist * 1.5)
            I_cube_sub = I_cube_sub[:, yi:-yi, xi:-xi]
            xi1, yi1 = nstgrid.nx - xi0, nstgrid.ny - yi0
        else:
            xi0, yi0 = nstgrid.xi0, nstgrid.yi0
            xi1, yi1 = nstgrid.xi1, nstgrid.yi1
        # replace
        for i in range(len(v)):
            #I_cube[i, yi0:yi1, xi0:xi1] = \
            I_cube[i, yi0:-yi0, xi0:-xi0] = \
            nstgrid.binning_onsubgrid(I_cube_sub[i,:,:])

        return I_cube


    def build_cube_subgrid(self, xx, yy, v, nsub = 2,
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,):
        if nsub < 1:
            print('ERROR\tbuild_cube_subgrid: nsub must be >= 2.')
            return 0
        subgrid = SubGrid2D(xx, yy, nsub = nsub)
        _xx, _yy = subgrid.xx_sub, subgrid.yy_sub
        I_cube_sub = self.build_cube(_xx, _yy, v, beam, dist, Tcmb, f0)
        nv = len(v)
        ny, nx = xx.shape
        return subgrid.binning_onsubgrid_layered(I_cube_sub)


    def build_cont(self, xx, yy, 
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.):
        # get quantities
        T, _, tau = self.build(xx, yy)

        # velocity grid
        ny, nx = xx.shape
        dx = xx[0,1] - xx[0,0]
        dy = yy[1,0] - yy[0,0]

        # radiative transfer
        _Bv = lambda T, v: Bvppx(T, v, dx, dy, dist = dist, au = True)
        Iv = (_Bv(T, f0) - _Bv(Tcmb, f0)) * (1. - np.exp(- tau))

        # Convolve beam if given
        if beam is not None:
            Iv = beam_convolution(xx, yy, Iv, 
                [beam[0] * dist, beam[1] * dist, beam[2]])

        # return intensity
        return Iv


    def build_cont_subgrid(self, xx, yy, 
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,
        nsub = 2):
        # subgrid
        subgrid = SubGrid2D(xx, yy, nsub = nsub)
        _xx, _yy = subgrid.xx_sub, subgrid.yy_sub

        # cube on the original grid
        Iv = self.build_cont(_xx, _yy,
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0,)

        # return intensity
        return subgrid.binning_onsubgrid_layered(Iv)


    def build_nested_cont(self, xx, yy, xscale, yscale, n_subgrid,
        beam = None, dist = 140., Tcmb = 2.73, f0 = 230.,):
        # original grid
        ny, nx = xx.shape

        # nested grid
        nstgrid = Nested2DGrid(xx, yy)
        xlim = [- np.nanmax(xx) * xscale, np.nanmax(xx) * yscale]
        ylim = [- np.nanmax(yy) * xscale, np.nanmax(yy) * yscale]
        xx_sub, yy_sub = nstgrid.nest(xlim, ylim, n_subgrid)

        # cube on the original grid
        Iv = self.build_cont(xx, yy,
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0,)

        # cube on the nested grid
        Iv_sub = self.build_cont(xx_sub, yy_sub,
        beam = beam, dist = dist, Tcmb = Tcmb, f0 = f0,)

        # cut off edge
        xi, yi, xi0, yi0 = \
        nstgrid.edgecut_indices(beam[0] * dist * 1.3, beam[0] * dist * 1.3)
        Iv_sub = Iv_sub[yi:-yi, xi:-xi]
        # replace
        Iv[yi0:-yi0, xi0:-xi0] = nstgrid.binning_onsubgrid(Iv_sub)

        return Iv



@dataclass(slots=True)
class SSDisk:

    # intensity distribution
    Ic: float = 1. # intensity at rc
    rc: float = 1. # critical radius
    beta: float = 0. # beta
    gamma: float = 0. # gamma
    inc: float = 0.
    pa: float = 0.
    z0: float = 0.
    r0: float = 1.
    hp: float = 0.
    ms: float = 0.
    vsys: float = 0.
    delv: float = 0. # line width

    def set_params(self, Ic = 0, rc = 0, beta = 0, gamma = 0, 
        inc = 0, pa = 0, z0 = 0, r0 = 0, hp = 0, ms = 0, vsys = 0, delv = 0.):
        '''

        Parameters
        ----------
         Ic (float): 
         rc (float): au
         inc (float): deg
         pa (float): deg
         z0 (float): au
         r0 (float): au
         ms (float): Msun
        '''
        # initialize parameters
        self.Ic = Ic
        self.rc = rc
        self.beta = beta
        self.gamma = gamma
        self.inc = inc
        self.pa = pa
        self.z0 = z0
        self.r0 = r0
        self.hp = hp
        self.ms = ms
        self.vsys = vsys
        self.delv = delv


    def get_paramkeys(self):
        return list(self.__annotations__.keys())


    def build(self, xx_sky, yy_sky, rin = 0.1):
        '''
        Build a model given sky coordinates and return a info for making a image cube.
        '''
        # parameters
        _inc_rad = np.radians(self.inc)
        _pa_rad = np.radians(self.pa)
        _fz = lambda r, z0, r0, hp: z0*(r/r0)**hp
        _dfz = lambda x, y, z0, h0, hp: 2. * y * 0.5 / np.sqrt(xp*xp + y*y) \
        / r0 * z0 * hp * (np.sqrt(xp*xp + y*y)/r0)*(hp - 1.)
        _zargs = [self.z0, self.r0, self.hp]

        # deprojection
        depr = sky_to_local(xx_sky.ravel(), yy_sky.ravel(), 
        _inc_rad, _pa_rad + 0.5 * np.pi, _fz, _zargs,)
        if type(depr) == int:
            I_int = np.zeros(xx_sky.shape)
            vlos  = np.zeros(xx_sky.shape)
            return I_int, vlos
        else:
            xx, yy = depr
            xx = xx.reshape(xx_sky.shape)
            yy = yy.reshape(yy_sky.shape)

        # local coordinates
        rr = np.sqrt(xx * xx + yy * yy) # radius
        rr[rr < rin] = np.nan
        phph = np.arctan2(yy, xx) # azimuthal angle (rad)
        zz = _fz(rr, *_zargs) # height

        # take y-axis as the line of sight
        vlos = vkep(rr * auTOcm, self.ms * Msun, zz * auTOcm) \
        * np.cos(phph) * np.sin(_inc_rad) * 1.e-5 + self.vsys # cm/s --> km/s
        I_int = ssdisk(rr, self.Ic, self.rc, self.gamma, self.beta)

        return I_int, vlos


    def build_cube(self, xx, yy, v, beam = None, dist = 140.):
        I_int, vlos = self.model.build(xx, yy)

        ny, nx = xx.shape
        nv = len(v)
        delv = np.mean(v[1:] - v[:-1])
        ve = np.hstack([v - delv * 0.5, v[-1] + 0.5 * delv])
        I_cube = np.zeros((nv, ny, nx))
        # making a cube
        for i in range(nv):
            vindx = np.where((ve[i] <= vlos) & (vlos < ve[i+1]))
            I_cube[i,vindx[0], vindx[1]] = I_int[vindx]

        #print('convolve beam..')
        # Convolve beam if given
        if beam is not None:
            gaussbeam = gaussian2d(xx, yy, 1., 0., 0., 
            beam[1] * dist / 2.35, beam[0] * dist / 2.35, beam[2], peak=True)

            I_cube /= np.abs((xx[0,0] - xx[0,1])*(yy[1,0] - yy[0,0])) # per pixel to per arcsec^2
            I_cube *= np.pi/(4.*np.log(2.)) * beam[0] * beam[1] # per arcsec^2 --> per beam

            # beam convolution
            I_cube = np.where(np.isnan(I_cube), 0., I_cube)
            I_cube = convolve(I_cube, np.array([gaussbeam]), mode='same')
        #print('done.')

        # return intensity
        return I_cube



# Radial profiles
def powerlaw_profile(r, p, I0, r0=1.):
    return I0*(r/r0)**(-p)

def mdpowerlaw_profile(r, a0, alpha, p):
    return a0*(1. + alpha*r)**(-p)

def nuker_profile(r, rt, It, alpha, beta, gamma):
    return It*(r/rt)**(-gamma) * (1. + (r/rt)**alpha)**((gamma-beta)/alpha)

def ssdisk(r, Ic, rc, gamma, beta = None):
    beta_p = gamma if beta is None else beta # - beta = - gamma - q
    return Ic * (r/rc)**(- beta_p) * np.exp(-(r/rc)**(2. - gamma))

def gaussian_profile(r, I0, sigr):
    return I0*np.exp(-r**2./(2.*sigr**2))


# Gaussians
# 1D
def gaussian1d(x, amp, mx, sig):
    return amp*np.exp(- 0.5 * (x - mx)**2/(sig**2)) #+ offset

# 2D
def gaussian2d(x, y, A, mx, my, sigx, sigy, pa=0, peak=True):
    '''
    Generate normalized 2D Gaussian

    Parameters
    ----------
     x: x value (coordinate)
     y: y value
     A: Amplitude. Not a peak value, but the integrated value.
     mx, my: mean values
     sigx, sigy: standard deviations
     pa: position angle [deg]. Counterclockwise is positive.
    '''
    if pa: # skip if pa == 0.
        x, y = rotate2d(x,y,pa)

    coeff = A if peak else A/(2.0*np.pi*sigx*sigy)
    expx = np.exp(-(x-mx)*(x-mx)/(2.0*sigx*sigx))
    expy = np.exp(-(y-my)*(y-my)/(2.0*sigy*sigy))
    return coeff*expx*expy


# 2D rotation
def rotate2d(x, y, angle, deg=True, coords=False):
    '''
    Rotate Cartesian coordinates.
    Right hand direction will be positive.

    array2d: input array
    angle: rotational angle [deg or radian]
    axis: rotatinal axis. (0,1,2) mean (x,y,z). Default x.
    deg (bool): If True, angle will be treated as in degree. If False, as in radian.
    '''

    # degree --> radian
    if deg:
        angle = np.radians(angle)
    else:
        pass

    if coords:
        angle = -angle
    else:
        pass

    cos = np.cos(angle)
    sin = np.sin(angle)

    xrot = x*cos - y*sin
    yrot = x*sin + y*cos

    return xrot, yrot


def sky_to_local(x, y, inc, pa, fz, zargs, dfz = None,
    inc_max = 85., zarg_lims = None, method = 'krylov'):
    '''
    Conversion from the plane-of-sky coordinate to the local coordinate.

    x, y (array): Plane-of-sky coordinates.
    inc (float): Inclination angle of the object.
    pa (float): Position angle of the object.

    method: krylov, anderson

    '''
    _inc_max_rad = np.radians(inc_max)
    jac = True if dfz is not None else False
    if (jac == True) & ((np.array(['hybr', 'lm']) != method).all()):
        method = 'hybr'
    # safety net
    if zarg_lims is not None:
        limit_check = np.array([
            (zargs[i] >= zarg_lims[i][0]) & (zargs[i] < zarg_lims[i][1]) 
            for i in range(len(zargs))])
        if limit_check.all():
            pass
        else:
            return 0

    # root function
    def y_solver(y, xp, yp, inc, fz, zargs, dfz = None):
        if dfz is None:
            return y * np.cos(inc) + fz(np.sqrt(xp*xp + y*y), *zargs) * np.sin(inc) - yp
        else:
            _jac = 1. / (np.cos(inc) + dfz(xp, y, *zargs) * np.sin(inc))
            return y * np.cos(inc) + fz(np.sqrt(xp*xp + y*y), *zargs) * np.sin(inc) - yp, np.diag(_jac)


    # rotation
    if pa != 0.:
        x, y = rotate2d(x, y, pa, coords=False, deg=False)

    # check inclination limit
    if inc < 0.:# inc = 0.
        return 0
    if inc > _inc_max_rad:# inc = _inc_max_rad
        return 0

    # deprojection limit
    # only valid for power-law flaring case
    z0, r0, hp = zargs
    #print('z0, hp, inc: %.2f %.2f %.2f'%(z0, hp, inc * 180./np.pi))
    if z0 == 0.:
        #sol = root(y_solver, y, 
        #    args=(x, y, inc, fz, zargs, dfz), method = method,)
        #ydep = sol.x
        ydep = y / np.cos(inc)
    elif hp == 1.:
        _r = 1.
        _z = np.abs(fz(_r, *zargs))
        th_lim = 0.5 * np.pi - np.arctan2(_z, _r) - inc
        if th_lim <= 0.:
            return 0
        else:
            sol = root(y_solver, y, 
                args=(x, y, inc, fz, zargs, dfz), method = method,)
            ydep = sol.x
    else:
        sol = root(y_solver, y, 
                args=(x, y, inc, fz, zargs, dfz), 
                method = method, options={'maxiter': 20})
        if sol.success:
            ydep = sol.x
        else:
            return 0

        '''
        y_ypmin = (1. / np.tan(inc) * r0**hp / np.abs(z0) / hp)**( 1. / (hp - 1.))
        yp_lim = y_ypmin * np.cos(inc) + fz(np.sqrt(y_ypmin * y_ypmin), *zargs) \
        * np.sin(inc)
        print(yp_lim)

        # deprojection
        ydep = np.empty(y.shape)
        cond = y > - yp_lim if z0 > 0. else y < yp_lim
        if np.count_nonzero(cond) > 0:
            #print('gonna pass')
            sol = root(y_solver, y[np.where(cond)], 
                args=(x[np.where(cond)], y[np.where(cond)], inc, fz, zargs, dfz), 
                method = method, options={'maxiter': 20})
            if sol.success:
                ydep[np.where(cond)] = sol.x
            else:
                return 0
            #print('passed')
            #print(sol.nit, sol.success)
            #ydep[np.where(cond)] = sol.x
        if np.count_nonzero(~cond) > 0:
            ydep[np.where(~cond)] = np.nan
        '''

    return x, ydep


def vkep(r, ms, z = 0.):
    return np.sqrt(Ggrav * ms * r * r / (r*r + z*z)**(1.5))


# Planck function
def Bv(T,v):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    '''
    v = v * 1.e9 # GHz --> Hz
    #print((hp*v)/(kb*T))
    exp=np.exp((hp*v)/(kb*T)) - 1.0
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    Bv=fterm/exp
    #print(exp, T, v)
    return Bv



# Planck function
def Bvppx(T, v, px, py, dist = 140., au = True):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    '''
    # unit
    v = v * 1.e9 # GHz --> Hz

    # Bv in cgs
    exp = np.exp((hp*v)/(kb*T)) - 1.0
    fterm = (2.0*hp*v*v*v)/(clight*clight)
    Bv = fterm / exp

    # From cgs to Jy/str
    Bv = Bv*1.e-7*1.e4 # cgs --> MKS
    Bv = Bv*1.0e26     # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)

    # Jy/str -> Jy/pixel
    if au:
        px = np.radians(px / dist / 3600.) # au --> radian
        py = np.radians(py / dist / 3600.) # au --> radian
    else:
        px = np.radians(px) # deg --> rad
        py = np.radians(py) # deg --> rad
    # one_pixel_area = pixel*pixel (rad^2)
    # Exactly, one_pixel_area = 4.*np.arcsin(np.sin(psize*0.5)*np.sin(psize*0.5))
    #  but the result is almost the same pixel cuz pixel area is much small.
    # (When psize = 20 au and dist = 140 pc, S_apprx/S_acc = 1.00000000000004)
    # I [Jy/pixel]   = I [Jy/sr] * one_pixel_area
    one_pixel_area = np.abs(px*py)
    Bv *= one_pixel_area # Iv (Jy per pixel)
    return Bv



# Jy/beam
def Bv_Jybeam(T,v,bmaj,bmin):
    '''
    Calculate Plack function of frequency in cgs.
    Unit is [erg s-1 cm-2 Hz-1 str-1].

    T: temprature [K]
    v: frequency [GHz]
    bmaj, bmin: beamsize [arcsec]
    '''

    # units
    bmaj = np.radians(bmaj / 3600.) # arcsec --> radian
    bmin = np.radians(bmin / 3600.) # arcsec --> radian
    v = v * 1.e9 # GHz --> Hz

    # coefficient for unit convertion
    # Omg_beam (sr) = (pi/4ln(2))*beam (rad^2)
    # I [Jy/beam] / Omg_beam = I [Jy/sr]
    C2 = np.pi/(4.*np.log(2.))  # beam(rad) -> beam (sr)
    bTOstr = bmaj * bmin * C2  # beam --> str


    #print(np.nanmax((hp*v)/(kb*T)), np.nanmin(T))
    exp = np.exp((hp*v)/(kb*T)) - 1.0
    #print(np.nanmax(exp))
    fterm=(2.0*hp*v*v*v)/(clight*clight)
    Bv = fterm / exp

    # cgs --> Jy/beam
    Bv = Bv*1.e-7*1.e4 # cgs --> MKS
    Bv = Bv*1.0e26     # MKS --> Jy (Jy = 10^-26 Wm-2Hz-1)
    Bv = Bv*bTOstr     # Jy/str --> Jy/beam
    return Bv

def doppler_f2v(f, f0, definition = 'radio'):
    return (f0 - f) / f0 * clight

def doppler_df2dv(df, f0, definition = 'radio'):
    return - df * clight / f0

def doppler_v2f(v, f0, definition = 'radio'):
    return f0 - v * f0 / clight

def doppler_dv2df(dv, f0, definition = 'radio'):
    return - dv * f0 / clight