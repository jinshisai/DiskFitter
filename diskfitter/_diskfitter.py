# import modules
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.optimize import root, minimize
from scipy.signal import convolve
from astropy import constants, units
import emcee
from dataclasses import dataclass

from .grid import Nested2DGrid, SubGrid2D
from .mpe import BayesEstimator



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


# Disk Fitter
class Fitter(object):
    """docstring for Fitter

    Args
    ----
     model: Model you select. Must be an object.
     params_free (dict): free parameters.
     params_fixed (dict): Fixed parameters
    """
    def __init__(self, model, params_free, params_fixed, 
        beam = None, dist = 140., build_args = None, 
        sampling = False, n_subgrid = 3,
        n_nstgrid = 1, xscale = 0.5, yscale = 0.5):
        super(Fitter, self).__init__()
        # parameter checks
        _model = model()
        _model_keys = _model.get_paramkeys()
        _input_keys = list(params_free.keys()) + list(params_fixed.keys())
        if sorted(_model_keys) != sorted(_input_keys):
            print('ERROR\tModelFitter: input keys do not match model input parameters.')
            print('ERROR\tModelFitter: input parameters must be as follows:')
            print(_model_keys)
            return 0

        # set model
        self.model_keys = _model_keys
        self.params_free = params_free
        self.pfree_keys = list(params_free.keys())
        self.params_fixed = params_fixed
        self.pfixed_keys = list(params_fixed.keys())
        _params = merge_dictionaries(params_free, params_fixed)
        self.params_ini = list(
            {k: _params[k] for k in _model_keys}.values()
            ) # re-ordered elements
        self.model = model #(*self.params_ini)
        self.beam = beam
        self.dist = dist
        self.build_args = [beam, dist] + build_args if build_args is not None\
        else [beam, dist]
        self.sampling = sampling
        self.n_subgrid = n_subgrid
        self.n_nstgrid = n_nstgrid
        self.xscale, self.yscale = xscale, yscale


    # define fitting function
    def fit_cube(self, params, pranges, d, derr, xx, yy, v,
        outname = 'modelfitter_results', nwalkers=None, 
        nrun=2000, nburn=1000, labels=[], show_progress=True, 
        optimize_ini=False, moves=emcee.moves.WalkMove(), symmetric_error=False,
        npool = 1,):
        # drop unecessary axis
        d = np.squeeze(d)
        # sampling step
        if self.sampling:
            delx = - (xx[0,1] - xx[0,0]) / self.dist # arcsec
            dely = (yy[1,0] - yy[0,0]) / self.dist   # arcsec
            smpl_x = int(self.beam[1] * 0.5 / delx)
            smpl_y = int(self.beam[1] * 0.5 / dely)
            d_smpld = d[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
        else:
            smpl_x, smpl_y = 1, 1
            d_smpld = d.copy()
        # labels
        if len(labels) != len(params): labels = self.pfree_keys

        # gridding
        if self.n_subgrid > 1:
            # subgrid
            subgrid = SubGrid2D(xx, yy)
            xx_sub, yy_sub = subgrid.xx_sub, subgrid.yy_sub

            # fitting function
            def fitfuc(xx, yy, v, *params):
                # safty net
                if np.all((pranges[0] < np.array([*params])) \
                    * (np.array([*params]) < pranges[1])) == False:
                    return np.zeros(
                        (len(v), xx.shape[1], xx.shape[0])
                        )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]

                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                    ) # reordered elements

                # build model cube
                model = self.model(*params_full)
                # cube on the original grid
                modelcube = model.build_cube_subgrid(
                    xx, yy, v, self.n_subgrid, 
                    *self.build_args)
                return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
        elif self.n_nstgrid > 1:
            nstgrid = Nested2DGrid(xx, yy)
            xlim = [-np.nanmax(xx) * self.xscale, np.nanmax(xx) * self.xscale]
            ylim = [-np.nanmax(yy) * self.yscale, np.nanmax(yy) * self.yscale]
            xx_sub, yy_sub = nstgrid.nest(xlim, ylim, self.n_subgrid)
            if self.beam is not None:
                xi, yi, xi0, yi0 = nstgrid.edgecut_indices(
                    self.beam[0] * self.dist * 1.3, self.beam[0] * self.dist * 1.3)
                # fitting function
                def fitfuc(xx, yy, v, *params):
                    # safty net
                    if np.all((pranges[0] < np.array([*params])) \
                        * (np.array([*params]) < pranges[1])) == False:
                        return np.zeros(
                            (len(v), xx.shape[1], xx.shape[0])
                            )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
                    # merge free parameters to fixed parameters
                    params_free = dict(zip(self.pfree_keys, [*params]))
                    _params_full = merge_dictionaries(params_free, self.params_fixed)
                    params_full = list(
                        {k: _params_full[k] for k in self.model_keys}.values()
                        ) # reordered elements

                    # build model cube
                    model = self.model(*params_full)
                    # cube on the original grid
                    modelcube = model.build_cube(xx, yy, v, *self.build_args)
                    # cube on the nested grid
                    modelcube_sub = model.build_cube(xx_sub, yy_sub, 
                        v, *self.build_args)[:, yi:-yi, xi:-xi]
                    # replace
                    for i in range(len(v)):
                        modelcube[i, yi0:-yi0, xi0:-xi0] = \
                        nstgrid.binning_onsubgrid(modelcube_sub[i,:,:])
                    return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
            else:
                # fitting function
                def fitfuc(xx, yy, v, *params):
                    where_sub = nstgrid.where_subgrid()
                    # safty net
                    if np.all((pranges[0] < np.array([*params])) \
                        * (np.array([*params]) < pranges[1])) == False:
                        return np.zeros(
                            (len(v), xx.shape[1], xx.shape[0])
                            )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
                    # merge free parameters to fixed parameters
                    params_free = dict(zip(self.pfree_keys, [*params]))
                    _params_full = merge_dictionaries(params_free, self.params_fixed)
                    params_full = list(
                        {k: _params_full[k] for k in self.model_keys}.values()
                        ) # reordered elements
                    #del params_free, _params_full # release memory

                    # build model cube
                    model = self.model(*params_full)
                    # cube on the original grid
                    modelcube = model.build_cube(xx, yy, v, *self.build_args)
                    # cube on the nested grid
                    modelcube_sub = model.build_cube(xx_sub, yy_sub, 
                        v, *self.build_args)
                    # replace
                    for i in range(len(v)):
                        modelcube[i,where_sub[0], where_sub[1]] = \
                        nstgrid.binning_onsubgrid(modelcube_sub[i,:,:]).ravel()
                    return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
        else:
            def fitfuc(xx, yy, v, *params):
                # safty net
                if np.all((pranges[0] < np.array([*params])) \
                    * (np.array([*params]) < pranges[1])) == False:
                    return np.zeros(
                        (len(v), xx.shape[1], xx.shape[0])
                        )[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]
                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                    ) # reordered elements
                #del params_free, _params_full # release memory

                # build model cube
                model = self.model(*params_full)
                modelcube = model.build_cube(xx, yy, v, *self.build_args)
                return modelcube[:, smpl_y//2::smpl_y, smpl_x//2::smpl_x]


        # fitting
        p0 = list(self.params_free.values())
        BE = BayesEstimator([xx, yy, v], d_smpld, derr, fitfuc)
        BE.run_mcmc(p0, pranges, outname=outname,
            nwalkers = nwalkers, nrun = nrun, nburn = nburn, labels = labels,
            show_progress = show_progress, optimize_ini = optimize_ini, moves = moves,
            symmetric_error = symmetric_error, npool = npool, f_rand_init=0.1)
        self.popt = BE.pfit[0]
        self.perr = BE.pfit[1:]


        # best solution
        params_free = dict(zip(self.pfree_keys, [*self.popt]))
        _params_full = merge_dictionaries(params_free, self.params_fixed)
        params_full = list(
            {k: _params_full[k] for k in self.model_keys}.values()
            )
        smpl_y, smpl_x = 1, 1
        modelcube = fitfuc(xx, yy, v, *self.popt)
        self.modelcube = modelcube
        return modelcube



    # define fitting function
    def fit_cont(self, params, pranges, d, derr, xx, yy,
        outname = 'modelfitter_results', nwalkers=None, 
        nrun=2000, nburn=1000, labels=[], show_progress=True, 
        optimize_ini=False, moves=emcee.moves.WalkMove(), symmetric_error=False,
        npool = 1,):
        # drop unecessary axis
        d = np.squeeze(d)
        # sampling step
        if self.sampling:
            delx = - (xx[0,1] - xx[0,0]) / self.dist # arcsec
            dely = (yy[1,0] - yy[0,0]) / self.dist # arcsec
            smpl_x = int(self.beam[1] * 0.5 / delx)
            smpl_y = int(self.beam[1] * 0.5 / dely)
            d_smpld = d[::smpl_y, ::smpl_x]
        else:
            smpl_x, smpl_y = 1, 1
            d_smpld = d.copy()
        # labels
        if len(labels) != len(params): labels = self.pfree_keys

        # nested grid
        if self.n_subgrid > 1:
            nstgrid = Nested2DGrid(xx, yy)
            xlim = [-np.nanmax(xx) * self.xscale, np.nanmax(xx) * self.xscale]
            ylim = [-np.nanmax(yy) * self.yscale, np.nanmax(yy) * self.yscale]
            xx_sub, yy_sub = nstgrid.nest(xlim, ylim, self.n_subgrid)
            if self.beam is not None:
                xi, yi, xi0, yi0 = nstgrid.edgecut_indices(
                    self.beam[0] * self.dist * 1.3, self.beam[0] * self.dist * 1.3)
                # fitting function
                def fitfuc(xx, yy, *params):
                    # safty net
                    if np.all((pranges[0] < np.array([*params])) \
                        * (np.array([*params]) < pranges[1])) == False:
                        return np.zeros(xx.shape)[::smpl_y, ::smpl_x]
                    # merge free parameters to fixed parameters
                    params_free = dict(zip(self.pfree_keys, [*params]))
                    _params_full = merge_dictionaries(params_free, self.params_fixed)
                    params_full = list(
                        {k: _params_full[k] for k in self.model_keys}.values()
                        ) # reordered elements

                    # build model cube
                    model = self.model(*params_full)
                    # cube on the original grid
                    modelim = model.build_cont(xx, yy, *self.build_args)
                    # cube on the nested grid
                    modelim_sub = model.build_cont(xx_sub, yy_sub, 
                        *self.build_args)[yi:-yi, xi:-xi]
                    # replace
                    modelim[yi0:-yi0, xi0:-xi0] = \
                    nstgrid.binning_onsubgrid(modelim_sub)
                    return modelim[::smpl_y, ::smpl_x]
            else:
                # fitting function
                def fitfuc(xx, yy, *params):
                    where_sub = nstgrid.where_subgrid()
                    # safty net
                    if np.all((pranges[0] < np.array([*params])) \
                        * (np.array([*params]) < pranges[1])) == False:
                        return np.zeros(xx.shape)[::smpl_y, ::smpl_x]
                    # merge free parameters to fixed parameters
                    params_free = dict(zip(self.pfree_keys, [*params]))
                    _params_full = merge_dictionaries(params_free, self.params_fixed)
                    params_full = list(
                        {k: _params_full[k] for k in self.model_keys}.values()
                        ) # reordered elements
                    #del params_free, _params_full # release memory

                    # build model cube
                    model = self.model(*params_full)
                    # cube on the original grid
                    modelim = model.build_cont(xx, yy, *self.build_args)
                    # cube on the nested grid
                    modelim_sub = model.build_cont(xx_sub, yy_sub, 
                        *self.build_args)
                    # replace
                    modelim[where_sub] = \
                    nstgrid.binning_onsubgrid(modelim_sub).ravel()
                    return modelcube[::smpl_y, ::smpl_x]
        else:
            def fitfuc(xx, yy, *params):
                # safty net
                if np.all((pranges[0] < np.array([*params])) \
                    * (np.array([*params]) < pranges[1])) == False:
                    return np.zeros(xx.shape)[::smpl_y, ::smpl_x]
                # merge free parameters to fixed parameters
                params_free = dict(zip(self.pfree_keys, [*params]))
                _params_full = merge_dictionaries(params_free, self.params_fixed)
                params_full = list(
                    {k: _params_full[k] for k in self.model_keys}.values()
                    ) # reordered elements
                #del params_free, _params_full # release memory

                # build model cube
                model = self.model(*params_full)
                modelim = model.build_cont(xx, yy, *self.build_args)
                return modelim[::smpl_y, ::smpl_x]


        # fitting
        p0 = list(self.params_free.values())
        BE = BayesEstimator([xx, yy], d_smpld, derr, fitfuc)
        BE.run_mcmc(p0, pranges, outname=outname,
            nwalkers = nwalkers, nrun = nrun, nburn = nburn, labels = labels,
            show_progress = show_progress, optimize_ini = optimize_ini, moves = moves,
            symmetric_error = symmetric_error, npool = npool, f_rand_init=0.1)
        self.popt = BE.pfit[0]
        self.perr = BE.pfit[1:]


        # best solution
        params_free = dict(zip(self.pfree_keys, [*self.popt]))
        _params_full = merge_dictionaries(params_free, self.params_fixed)
        params_full = list(
            {k: _params_full[k] for k in self.model_keys}.values()
            )
        smpl_y, smpl_x = 1, 1
        modelim = fitfuc(xx, yy, *self.popt)
        self.modelim = modelim
        return modelim


def merge_dictionaries(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict