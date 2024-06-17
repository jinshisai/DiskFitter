# modules
import numpy as np
import matplotlib.pyplot as plt


class Nested2DGrid(object):
    """docstring for NestedGrid"""
    def __init__(self, xx, yy, precision = 4):
        super(Nested2DGrid, self).__init__()
        self.xx = xx
        self.yy = yy
        ny, nx = xx.shape
        self.ny, self.nx = ny, nx
        self.dx = xx[0,1] - xx[0,0]
        self.dy = yy[1,0] - yy[0,0]
        self.xc = xx[ny//2, nx//2]
        self.yc = yy[ny//2, nx//2]
        self.yci, self.xci = ny//2, nx//2

        if (_check := self.check_symmetry(precision))[0]:
            pass
        else:
            print('ERROR\tNested2DGrid: Input grid must be symmetric but not.')
            print('ERROR\tNested2DGrid: Condition.')
            print('ERROR\tNested2DGrid: [xcent, ycent, dx, dy]')
            print('ERROR\tNested2DGrid: ', _check[1])
            return None

        # retrive x and y
        x = self.xx[0,:]
        y = self.yy[:,0]
        xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        ye = np.hstack([y - self.dy * 0.5, y[-1] + self.dy * 0.5])
        self.x, self.y = x, y
        self.xe, self.ye = xe, ye
        self.decimals = precision


    def check_symmetry(self, decimals = 5):
        nx, ny = self.nx, self.ny
        xc = np.round(self.xc, decimals)
        yc = np.round(self.yc, decimals)
        _xcent = (xc == 0.) if nx%2 == 1 else (xc == - np.round(self.xx[ny//2 - 1, nx//2 - 1], decimals))
        _ycent = (yc == 0.) if ny%2 == 1 else (yc == - np.round(self.yy[ny//2 - 1, nx//2 - 1], decimals))
        delxs = (self.xx[1:,1:] - self.xx[:-1,:-1]) / self.dx
        delys = (self.yy[1:,1:] - self.yy[:-1,:-1]) / self.dy
        _xdel = (np.round(delxs, decimals) == 1. ).all()
        _ydel = (np.round(delys, decimals)  == 1. ).all()
        cond = [_xdel, _ydel] # _xcent, _ycent,
        return all(cond), cond
    

    def nest(self, xlim,  ylim, nsub = 2):
        # error check
        if (len(xlim) != 2) | (len(ylim) != 2):
            print('ERROR\tnest: Input xlim and/or ylim is not valid.')
            return 0
        # decimals
        xlim = [np.round(xlim[0], self.decimals), np.round(xlim[1], self.decimals)]
        ylim = [np.round(ylim[0], self.decimals), np.round(ylim[1], self.decimals)]

        self.nsub = nsub
        self.xlim_sub, self.ylim_sub = xlim, ylim
        ximin, ximax = index_between(self.x, xlim, mode='edge')[0]
        yimin, yimax = index_between(self.y, ylim, mode='edge')[0]
        _nx = ximax - ximin + 1
        _ny = yimax - yimin + 1
        xemin, xemax = self.xe[ximin], self.xe[ximax + 1]
        yemin, yemax = self.ye[yimin], self.ye[yimax + 1]
        self.xi0, self.xi1 = ximin, ximax # Starting and ending indices of nested grid
        self.yi0, self.yi1 = yimin, yimax # Starting and ending indices of nested grid

        # nested grid
        xe_sub = np.linspace(xemin, xemax, _nx * nsub + 1)
        ye_sub = np.linspace(yemin, yemax, _ny * nsub + 1)
        x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
        y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
        xx_sub, yy_sub = np.meshgrid(x_sub, y_sub)
        self.xe_sub, self.ye_sub = xe_sub, ye_sub
        self.x_sub, self.y_sub = x_sub, y_sub
        self.xx_sub, self.yy_sub = xx_sub, yy_sub
        self.dx_sub, self.dy_sub = self.dx / nsub, self.dy / nsub
        self.nx_sub, self.ny_sub = len(x_sub), len(y_sub)
        return xx_sub, yy_sub


    def where_subgrid(self):
        return np.where(
            (self.xx >= self.xlim_sub[0]) * (self.xx <= self.xlim_sub[1]) \
            * (self.yy >= self.ylim_sub[0]) * (self.yy <= self.ylim_sub[1]))


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        d_avg = np.array([
            data[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        return np.nanmean(d_avg, axis = 0)


    def edgecut_indices(self, xlength, ylength):
        # odd or even
        x_oddeven = self.nx%2
        y_oddeven = self.ny%2
        # edge indices for subgrid
        xi = int(xlength / self.dx_sub) if self.dx_sub > 0 else int(- xlength / self.dx_sub)
        yi = int(ylength / self.dy_sub)
        _nx_resub = int(self.nx_sub - 2 * xi) // self.nsub # nx of subgrid after cutting edge
        _ny_resub = int(self.ny_sub - 2 * yi) // self.nsub # ny of subgrid after cutting edge
        # fit odd/even
        if _nx_resub%2 != x_oddeven: _nx_resub += 1
        if _ny_resub%2 != y_oddeven: _ny_resub += 1
        # nx, ny of the new subgrid and new xi and yi
        nx_resub = _nx_resub * self.nsub
        ny_resub = _ny_resub * self.nsub
        xi = (self.nx_sub - nx_resub) // 2
        yi = (self.ny_sub - ny_resub) // 2
        # for original grid
        xi0 = int((self.nx - nx_resub / self.nsub) * 0.5)
        yi0 = int((self.ny - ny_resub / self.nsub) * 0.5)
        #print(nx_resub / self.nsub, self.nx - xi0 - xi0)
        return xi, yi, xi0, yi0


    def binning(self, nbin):
        if nbin%2 == 0:
            xx, yy = self.shift()
        else:
            xx, yy = self.xx.copy(), self.yy.copy()

        xcut = self.nx%nbin
        ycut = self.ny%nbin
        _xx = xx[ycut//2:-ycut//2, xcut//2:-xcut//2]
        _yy = yy[ycut//2:-ycut//2, xcut//2:-xcut//2]
        xx_avg = np.array([
            _xx[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        yy_avg = np.array([
            _yy[i::nbin, i::nbin]
            for i in range(nbin)
            ])

        return np.average(xx_avg, axis= 0), np.average(yy_avg, axis= 0)


    def shift(self):
        rex = np.arange(-self.nx//2, self.nx//2+1, 1) + 0.5
        rex *= self.dx
        rey = np.arange(-self.ny//2, self.ny//2+1, 1) + 0.5
        rey *= self.dy
        return np.meshgrid(rex, rey)


class SubGrid2D(object):
    """docstring for NestedGrid"""
    def __init__(self, xx, yy, nsub = 2):
        super(SubGrid2D, self).__init__()
        # save grid info
        self.xx = xx
        self.yy = yy
        ny, nx = xx.shape
        self.ny, self.nx = ny, nx
        self.dx = xx[0,1] - xx[0,0]
        self.dy = yy[1,0] - yy[0,0]
        self.xc = xx[ny//2, nx//2]
        self.yc = yy[ny//2, nx//2]
        self.yci, self.xci = ny//2, nx//2

        # retrive x and y
        x = self.xx[0,:]
        y = self.yy[:,0]
        xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        ye = np.hstack([y - self.dy * 0.5, y[-1] + self.dy * 0.5])
        self.x, self.y = x, y
        self.xe, self.ye = xe, ye

        # subgrid
        self.subgrid(nsub = nsub)


    def subgrid(self, nsub = 2):
        self.nsub = nsub
        nx_sub, ny_sub = self.nx * nsub, self.ny * nsub
        self.nx_sub, self.ny_sub = nx_sub, ny_sub

        # sub grid
        xemin, xemax = self.xe[0], self.xe[-1] # edge of the original grid
        yemin, yemax = self.ye[0], self.ye[-1] # edge of the original grid
        xe_sub = np.linspace(xemin, xemax, nx_sub + 1)
        ye_sub = np.linspace(yemin, yemax, ny_sub + 1)
        x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
        y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
        xx_sub, yy_sub = np.meshgrid(x_sub, y_sub)
        self.xe_sub, self.ye_sub = xe_sub, ye_sub
        self.x_sub, self.y_sub = x_sub, y_sub
        self.xx_sub, self.yy_sub = xx_sub, yy_sub
        self.dx_sub, self.dy_sub = self.dx / nsub, self.dy / nsub
        #self.nx_sub, self.ny_sub = len(x_sub), len(y_sub)
        return xx_sub, yy_sub


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        d_avg = np.array([
            data[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        return np.nanmean(d_avg, axis = 0)


    def binning(self, nbin):
        if nbin%2 == 0:
            xx, yy = self.shift()
        else:
            xx, yy = self.xx.copy(), self.yy.copy()

        xcut = self.nx%nbin
        ycut = self.ny%nbin
        _xx = xx[ycut//2:-ycut//2, xcut//2:-xcut//2]
        _yy = yy[ycut//2:-ycut//2, xcut//2:-xcut//2]
        xx_avg = np.array([
            _xx[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        yy_avg = np.array([
            _yy[i::nbin, i::nbin]
            for i in range(nbin)
            ])

        return np.average(xx_avg, axis= 0), np.average(yy_avg, axis= 0)


    def shift(self):
        rex = np.arange(-self.nx//2, self.nx//2+1, 1) + 0.5
        rex *= self.dx
        rey = np.arange(-self.ny//2, self.ny//2+1, 1) + 0.5
        rey *= self.dy
        return np.meshgrid(rex, rey)


def index_between(t, tlim, mode='all'):
    if not (len(tlim) == 2):
        if mode=='all':
            return np.full(np.shape(t), True)
        elif mode == 'edge':
            if len(t.shape) == 1:
                return tuple([[0, len(t)-1]])
            else:
                return tuple([[0, t.shape[i]] for i in range(len(t.shape))])
        else:
            print('index_between: mode parameter is not right.')
            return np.full(np.shape(t), True)
    else:
        if mode=='all':
            return (tlim[0] <= t) * (t <= tlim[1])
        elif mode == 'edge':
            nonzero = np.nonzero((tlim[0] <= t) * (t <= tlim[1]))
            return tuple([[np.min(i), np.max(i)] for i in nonzero])
        else:
            print('index_between: mode parameter is not right.')
            return (tlim[0] <= t) * (t <= tlim[1])


class Nested1DGrid(object):
    """docstring for NestedGrid"""
    def __init__(self, x, precision = 4):
        super(Nested1DGrid, self).__init__()
        self.x = x
        nx = len(x)
        self.nx = nx
        self.dx = x[1] - x[0]
        self.xc = x[nx//2]
        self.xci = nx//2

        if (_check := self.check_symmetry(precision))[0]:
            pass
        else:
            print('ERROR\tNested2DGrid: Input grid must be symmetric but not.')
            print('ERROR\tNested2DGrid: Condition.')
            print('ERROR\tNested2DGrid: [dx]')
            print('ERROR\tNested2DGrid: ', _check[1])
            return None

        # retrive x and y
        xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        self.xe = xe
        self.decimals = precision


    def check_symmetry(self, decimals = 5):
        nx = self.nx
        xc = np.round(self.xc, decimals)
        #_xcent = (xc == 0.) if nx%2 == 1 else (xc == - np.round(self.x[nx//2 - 1], decimals))
        delxs = (self.x[1:] - self.x[:-1]) / self.dx
        _xdel = (np.round(delxs, decimals) == 1. ).all()
        cond = [_xdel] # _xcent
        return all(cond), cond
    

    def nest(self, nsub = 3, xlim = None):
        # subgrid
        self.nsub = nsub
        self.xlim_sub = xlim
        if xlim is not None:
            # error check
            if len(xlim) != 2:
                print('ERROR\tnest: Input xlim is not valid.')
                return 0
            # decimals
            xlim = [np.round(xlim[0], self.decimals), np.round(xlim[1], self.decimals)]
            ximin, ximax = index_between(self.x, xlim, mode='edge')[0]
            _nx = ximax - ximin + 1
            xemin, xemax = self.xe[ximin], self.xe[ximax + 1]
            self.xi0, self.xi1 = ximin, ximax # Starting and ending indices of nested grid

            # nested grid
            xe_sub = np.linspace(xemin, xemax, _nx * nsub + 1)
            x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
            self.xe_sub = xe_sub
            self.x_sub = x_sub
            self.dx_sub = self.dx / nsub
            self.nx_sub = len(x_sub)
        else:
            xe_sub = np.arange(0., self.nx * nsub + 1, 1.) # delta in pixel
            dx_sub = self.dx / nsub # delta for subgrid
            xe_sub *= dx_sub # increase
            xe_sub += self.xe[0] # cell edge
            x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:]) # cell center
            # save
            self.xe_sub = xe_sub
            self.x_sub = x_sub
            self.dx_sub = dx_sub
            self.nx_sub = len(x_sub)
        return x_sub


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        if len(data.shape) == 1:
            d_avg = np.array([
                data[i::nbin]
                for i in range(nbin)
                ])
        elif len(data.shape) == 2:
            d_avg = np.array([
                data[i::nbin, :]
                for i in range(nbin)
                ])
        elif len(data.shape) == 3:
            d_avg = np.array([
                data[i::nbin, :, :]
                for i in range(nbin)
                ])
        else:
            print('Ndim is expected to be <= 3.')
            return 0
        return np.nanmean(d_avg, axis = 0)


    def shift(self):
        rex = np.arange(-self.nx//2, self.nx//2+1, 1) + 0.5
        rex *= self.dx
        return rex



def main():
    # ---------- input -----------
    nx, ny = 32, 33
    xe = np.linspace(-10, 10, nx+1)
    #xe = np.logspace(-1, 1, nx+1)
    ye = np.linspace(-10, 10, ny+1)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    xx, yy = np.meshgrid(xc, yc)
    # ----------------------------


    # ---------- debug ------------
    '''
    # 2D
    # model on an input grid
    dd = np.exp( - (xx**2. / 18.) - (yy**2. / 18.))

    # nested grid
    gridder = Nested2DGrid(xx,yy)
    xx_sub, yy_sub = gridder.nest([-3., 3.], [-3., 3.], 2)
    # model on the nested grid
    dd_sub = np.exp( - (xx_sub**2. / 18.) - (yy_sub**2. / 18.))
    # binned
    dd_binned = gridder.binning_onsubgrid(dd_sub)
    dd_re = dd.copy()
    #print(gridder.where_subgrid())
    dd_re[gridder.where_subgrid()] = dd_binned.ravel()



    # plot
    fig, axes = plt.subplots(1,3)
    ax1, ax2, ax3 = axes

    xx_plt, yy_plt = np.meshgrid(xe, ye)
    ax1.pcolor(xx_plt, yy_plt, dd, vmin = 0., vmax = 1.)
    #ax1.pcolor(xx_sub_plt, yy_sub_plt, dd_sub)

    xx_sub_plt, yy_sub_plt = np.meshgrid(gridder.xe_sub, gridder.ye_sub)
    ax2.pcolor(xx_sub_plt, yy_sub_plt, dd_sub, vmin = 0., vmax = 1)
    c = ax3.pcolor(xx_plt, yy_plt, dd - dd_re, vmin = 0., vmax = 1)


    for axi in [ax1, ax2, ax3]:
        axi.set_xlim(-10,10)
        axi.set_ylim(-10,10)

    for axi in [ax2, ax3]:
        axi.set_xticklabels('')
        axi.set_yticklabels('')

    cax = ax3.inset_axes([1.03, 0., 0.03, 1.])
    plt.colorbar(c, cax=cax)
    plt.show()
    '''


    # 1D
    # model on an input grid
    model = lambda x: np.exp( - (x**2. / 18.))
    d = model(xc)
    # nested grid
    nstg1D = Nested1DGrid(xc)
    x_sub = nstg1D.nest(3)
    # model on the nested grid
    d_sub = model(x_sub)
    # binned
    d_binned = nstg1D.binning_onsubgrid(d_sub)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for xi, di, label, ls in zip(
        [xc, x_sub, xc],
        [d, d_sub, d_binned],
        ['Original', 'Subgrid', 'Binned'],
        ['-', '-', '--']
        ):
        ax.step(xi, di, where = 'mid', lw = 2., alpha = 0.5, ls = ls)

    plt.show()



if __name__ == '__main__':
    main()