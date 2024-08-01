import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('.')
from grid import Nested2DGrid



def main():
    # ------ input -------
    x = np.linspace(-100, 100, 64) # dx~3
    y = x.copy()

    nlevels = 2
    nsub = [4, 10] # dx~0.8, 0.08
    xlim = [[-10.,10.], [-3., 3.]]
    ylim = xlim.copy()

    gfunc = lambda x, y: np.exp( - x**2 / 30**2 - y**2 / 60**2)
    # -------------------

    # original data
    xx, yy = np.meshgrid(x,y)
    d_or = gfunc(xx, yy)


    # nested grid
    nstg = Nested2DGrid(x,y, xlim,ylim,nsub,nlevels)
    print(nstg.xinest)
    d = [None] * nstg.nlevels
    for l in range(nstg.nlevels):
        _x, _y = nstg.xnest[l], nstg.ynest[l]
        d[l] = gfunc(_x, _y)

    # collapse
    d_col = nstg.collapse(d)
    #print(np.nonzero(d_or - d_col))


    fig, axes = plt.subplots(1,4, figsize = (11.69, 4))
    ax1, ax2, ax3, ax4 = axes

    # original data without nested grid
    imc = ax1.pcolormesh(xx, yy, d_or)
    plt.colorbar(imc, ax=ax1)

    _xx, _yy = nstg.get_grid(1)
    _d = gfunc(_xx, _yy)
    imc2 = ax2.pcolormesh(_xx, _yy, _d)
    plt.colorbar(imc2, ax=ax2)

    _d1 = nstg.collapse(d, upto = 1)
    #print(nstg.ngrids[1])
    #print(np.nonzero(_d - _d1))
    imc3 = ax3.pcolormesh(_xx, _yy, _d1)
    plt.colorbar(imc3, ax=ax3)

    imc4 = ax4.pcolormesh(xx, yy, d_col)
    plt.colorbar(imc2, ax=ax4)

    for i, ax in enumerate(axes):
        #ax.set_xlim(-800., 800.)
        #ax.set_ylim(-800., 800.)

        if i != 0:
           ax.set_yticklabels('')
    plt.show()



    # test 2
    fig, axes = plt.subplots(1,3, figsize = (11.69, 4))
    ax1, ax2, ax3, = axes

    # original data without nested grid
    _xx, _yy = nstg.get_grid(1)
    _d = gfunc(_xx, _yy)
    imc1 = ax1.pcolormesh(_xx, _yy, _d)
    plt.colorbar(imc1, ax=ax1)


    # nested grid
    _xx2, _yy2 = nstg.get_grid(2)
    imc2 = ax2.pcolormesh(_xx2, _yy2, 
        d[-1].reshape(nstg.ngrids[-1]))
    plt.colorbar(imc2, ax=ax2)

    # binned back to the original grid
    _d_bin = nstg.binning_onsubgrid_layered(
        d[-1].reshape(nstg.ngrids[-1]), 
        nstg.nsub[-1])
    ximin, ximax = nstg.xinest[2]
    yimin, yimax = nstg.yinest[2]
    imc3 = ax3.pcolormesh(
        _xx[yimin:yimax+1, ximin:ximax+1], 
        _yy[yimin:yimax+1, ximin:ximax+1], 
        _d_bin)
    plt.colorbar(imc3, ax=ax3)

    for ax in axes:
        ax.set_xlim(-5.,5)
        ax.set_ylim(-5.,5)

    plt.show()







if __name__ == '__main__':
    main()

