import numpy as np

from imfits import Imfits


def cubetofits(temp, model, params, 
    beamconv = True, dist = 140., build_args = None,
    n_subgrid = 1, n_nstgrid = 1, xscale = 0.5, yscale = 0.5):
    cube = Imfits(temp)
    xx = cube.xx * 3600. * dist
    yy = cube.yy * 3600. * dist
    v = cube.vaxis
    beam = cube.beam
    f0 = cube.restfreq * 1.e-9 # GHz
