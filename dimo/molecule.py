# import modules
import os
import sys
import math
import glob
import numpy as np
from astropy import constants, units

# constants (in cgs)

Ggrav  = constants.G.cgs.value        # Gravitational constant
ms     = constants.M_sun.cgs.value    # Solar mass (g)
ls     = constants.L_sun.cgs.value    # Solar luminosity (erg s^-1)
rs     = constants.R_sun.cgs.value    # Solar radius (cm)
au     = units.au.to('cm')            # 1 au (cm)
pc     = units.pc.to('cm')            # 1 pc (cm)
clight = constants.c.cgs.value        # light speed (cm s^-1)
kb     = constants.k_B.cgs.value      # Boltzman coefficient
hp     = constants.h.cgs.value        # Planck constant
sigsb  = constants.sigma_sb.cgs.value # Stefan-Boltzmann constant (erg s^-1 cm^-2 K^-4)
mp     = constants.m_p.cgs.value      # Proton mass (g)

# path to here
path_to_here = os.path.dirname(__file__)
path_to_library = path_to_here[:-11]



class MolData():


    def __init(self, line):
        self.line = line
        self.read_lamda_moldata()


        # read molecular data
    def read_lamda_moldata(self):
        '''
        Read a molecular data file from LAMDA (Leiden Atomic and Molecular Database)
        '''
        # find path
        line = self.line.lower()
        if '+' in line: line = line.replace('+','p')
        infile = glob.glob(path_to_library+'moldata/'+line+'.dat')
        if len(infile) == 0:
            print('ERROR\tread_lamda_moldata: Cannot find LAMDA file.')
            print('ERROR\tread_lamda_moldata: Only C18O, CO, and N2H+ are \
                supported for now')
            return
        else:
            data = pd.read_csv(infile[0], comment='!', 
                delimiter='\r\n', header=None, engine='python')

        # get
        # line name, weight, nlevels
        _, weight, nlevels = data[0:3][0].values
        weight  = float(weight)
        nlevels = int(nlevels)

        # energy on each excitation level
        elevels = data[3:3+nlevels].values
        elevels = np.array([ elevels[i][0].split() for i in range(nlevels)])
        lev, EJ, gJ, J = elevels.T
        lev = np.array([ int(lev[i]) for i in range(nlevels)])
        EJ  = np.array([ float(EJ[i]) for i in range(nlevels)]) \
        * clight * hp / kb # in K
        gJ  = np.array([ float(gJ[i]) for i in range(nlevels)])
        J   = np.array([ int(J[i]) for i in range(nlevels)])

        # number of transition
        ntrans = data[0][3+nlevels].strip()
        ntrans = int(ntrans)

        # Einstein A coefficient
        vtrans = data[3+nlevels+1:3+nlevels+1+ntrans].values
        vtrans = np.array([vtrans[i][0].split() for i in range(ntrans)])

        itrans, Jup, Jlow, Acoeff, freq, dE = vtrans.T
        itrans = np.array([ int(itrans[i]) for i in range(ntrans)])
        Jup    = np.array([ int(Jup[i]) for i in range(ntrans)])
        Jlow   = np.array([ int(Jlow[i]) for i in range(ntrans)])
        Acoeff = np.array([ float(Acoeff[i]) for i in range(ntrans)])
        freq   = np.array([ float(freq[i]) for i in range(ntrans)])
        dE = np.array([ float(dE[i]) for i in range(ntrans)])
        #for i in range(ntrans):
        #    print('Eu, Eu: %.4f %.4f'%(EJ[i+1], delE[i]))

        # transitions
        trans = [ str(J[ int(Jup[i] - 1)]) + '-' \
        + str( J[ int(Jlow[i] - 1)]) for i in range(len(itrans))]

        # save
        self.weight = weight
        self.nlevels = nlevels
        self.EJ = EJ
        self.gJ = gJ
        self.J = J
        self.ntrans = ntrans
        self.trans = trans
        self.Jup = Jup
        self.Jlow = Jlow
        self.Acoeff = Acoeff
        self.freq = freq
        self.dE = dE


    def params_ul(self, Ju):
        # line Ju --> Jl
        trans = self.trans[Ju]
        freq = self.freq[Ju-1] * 1e9 # Hz
        Aul     = self.Acoeff[Ju-1]
        gu      = self.gJ[Ju]
        gl      = self.gJ[Ju-1]
        Eu     = self.EJ[Ju]
        El     = self.EJ[Ju-1]
        return trans, freq, Aul, gu, gl, Eu, El


    def partfunc_grid(self, Tmin, Tmax, ngrid, scale = 'linear'):
        '''
        Make a grid for the partition function.

        Parameters
        ----------
         Tmin, Tmax (float): Minimum and maximum temperature for the grid calculation.
         ngrid (int): Number of temperature grid.
         scale (str): If the temperature grid is the linear or log scale.
        '''
        if scale == 'linear':
            Tgrid = np.linspace(Tmin, Tmax, ngrid)
        elif scale == 'log':
            Tgrid = np.linspace(np.log10(Tmin), np.log10(Tmax), ngrid)
        else:
            print('ERROR\tPfunc_grid: scale must be linear or log.')
            print('ERROR\tPfunc_grid: Ignore the input and assume linear scale.')
            Tgrid = np.linspace(Tmin, Tmax, ngrid)

        self._Tgrid = Tgrid
        self._PFgrid = np.array([
            np.sum(
            np.array([self.gJ[j]*np.exp(-self.EJ[j]/Tex) for j in range(len(self.J))])
            ) for Tex in Tgrid ])

    def partition_function(self, Tex):
        return np.sum(
            np.array([self.gJ[j]*np.exp(-self.EJ[j]/Tex) for j in range(len(self.J))])
            )



def Molecule():


    def __init(self, molecules):
        self.moldata = {}

        if (type(molecules) == list) or (type(molecules) == tuple):
            for mol in molecules:
                self.moldata[mol] = MolData(mol)
        elif type(molecules) == str:
            self.moldata[molecules] = MolData(molecules)
        else:
            print('ERROR\tLTEAnalysis: molecules must be str, or list or tuple of strings.')


    def get_tau(self, line, Ju, Ntot, Tex, delv = None, grid_approx = True):
        '''
        Calculate the optical depth under the local thermal equilibrium (LTE) assumption.

        Parameters
        ----------
         line (str): The line for which you calculate tau.
         Ju (int): J of or just an index for the upper energy state.
         Ntot (float or array): Total number column density of the molecule (cm^-2)
         Tex (float or array): Excitation temperature (K)
         delv (float): Line width (FWHM) of the line (cm s^-1). If given, tau_v will be calculated,
                       else tau_total integrated over frequency will be returned.
        '''
        trans, freq, Aul, gu, gl, Eu, El = self.moldata[line].params_ul(Ju)

        # partition function
        if grid_approx:
            # currently 0th order approx. only
            Qrot = self.moldata[line]._PFgrid[np.nanargmin(
                (self.moldata[line]._Tgrid - Tex)**2.)]
        else:
            Qrot = self.moldata[line].partition_function(Tex)

        if delv is not None:
            # return tau_v
            return (clight*clight*clight)/(8.*np.pi*freq_ul*freq_ul*freq_ul)*(gu/Qrot)\
            *np.exp(-Eu/Tex)*Ntot*Aul*(np.exp(hp*freq_ul/(kb*Tex)) - 1.) / delv
        else:
            # return tau_total, integrated over frequency
            return (clight*clight)/(8.*np.pi*freq_ul*freq_ul)*(gu/Qrot)\
            *np.exp(-Eu/Tex)*Ntot*Aul*(np.exp(hp*freq_ul/(kb*Tex)) - 1.)

