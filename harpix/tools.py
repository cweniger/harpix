#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import healpy as hp
import numpy as np
import harpix as harp
from scipy.sparse import linalg as la
from scipy.integrate import quad

class HarpixSigma1D(la.LinearOperator):
    """Covariance matrix generator for harpix objects."""
    def __init__(self, err, corrlength = None):
        """
        Parameters
        ----------
        * `harpix` [Harpix object]: Harpix object on which (flattened) data the
        covariance matrix will act.
        * `err` [Harpix object]: Skymap that defines the magnitude of the error
        * `corrlength` [float]: Correlation length in deg.
        * `Sigma` [matrix-like]: Covariance matrix for (flattened) non-spatial
          dimensions of harpix map.
        * `nside` [int, None]: If `nside` equals None, use dense spatial
          correlation matrix.  Otherwise internally convert to `nside` healpix
          map, use `healpy` to perform convolution, and then convert back.
        * `mul_sr` [boolean]: If `true`, `err` will be effectively multiplied
          by `sr` when generating the covariance matrix.

        Returns
        -------
        * `self`
        """
        N = len(err.data)
        super(HarpixSigma1D, self).__init__(None, (N, N))

        self.F = err.getdata(mul_sr = False)
        vec = err.getvec()
        M = np.zeros((N, N))
        sigma_sq = np.deg2rad(corrlength)**2
        for i in range(N):
            dist = hp.rotator.angdist(vec[:,i], vec)
            M[i] = np.exp(-dist**2/2/sigma_sq)
        self.M = M

    def _matvec(self, x):
        return (self.M.dot((x.T*self.F).T).T*self.F).T

#SigmaDims(S, x0, x_err, sigma = ...)
#comp1 = Sigma1 * Sigma2
#the_sum = comp1 + comp2 + comp3
#SF = the_sum.getSwordfish()
#
#SigmaHarpix(M, x0, x_err, sigma = ...)
#
#
#
#
#
#
#Swordfish(B, T, C, E)
#
#
#Swordfish(B = Btot, Sigma = Sigmatot, E = ...)
#
class HarpixSigma(la.LinearOperator):
    """Covariance matrix generator for harpix objects."""
    def __init__(self, harpix):
        """
        Parameters
        ----------
        * `harpix` [Harpix object]: Harpix object on which (flattened) data the
        covariance matrix will act.
        """
        self.harpix = harpix
        self.N = np.prod(np.shape(self.harpix.data))
        super(HarpixSigma, self).__init__(None, (self.N, self.N))
        self.Flist = []
        self.Xlist = []

#    def add2(self, M, S, xm, xs, err_xm, err_xs, sigma_m, Sigma_S, nside
#            = None):
#        """
#
#        Parameters
#        ----------
#        `M`: Harpix map (no data)
#        `S`: Spectrum
#        `ConstM`: Constraints (potentially)
#        `ConstS`: Constraints (potentially)
#        `SigmaM`: Constraints (potentially)
#        `SigmaS`: Constraints (potentially)
#        """
#        # Construct M0, dM, and constM
#        # Construct S0, dS, and constS
#        # Get map from H onto harpix(M0 x S0)
#        # Get inverse map
#        # Generate SigmaM contribution
#        # Generate SigmaS contribution
#
    def add(self, err = None, corrlength = None, Sigma = None,
            nside = None, mul_sr=False):
        """Add contribution to covariance matrix.

        Parameters
        ----------
        * `err` [Harpix object]: Skymap that defines the magnitude of the error
        * `corrlength` [float]: Correlation length in deg.
        * `Sigma` [matrix-like]: Covariance matrix for (flattened) non-spatial
          dimensions of harpix map.
        * `nside` [int, None]: If `nside` equals None, use dense spatial
          correlation matrix.  Otherwise internally convert to `nside` healpix
          map, use `healpy` to perform convolution, and then convert back.
        * `mul_sr` [boolean]: If `true`, `err` will be effectively multiplied
          by `sr` when generating the covariance matrix.

        Returns
        -------
        * `self`
        """
        F = err.getformattedlike(self.harpix).getdata(mul_sr=mul_sr)
        self.Flist.append(F)

        if nside is not None:
            # Use healpy to do convolution
            lmax = 3*nside - 1  # default from hp.smoothing
            Nalm = hp.Alm.getsize(lmax)
            G = np.zeros(Nalm, dtype = 'complex128')
            H = hp.smoothalm(np.ones(Nalm), sigma = np.deg2rad(corrlength), inplace = False, verbose = False)
            npix = hp.nside2npix(nside)
            m = np.zeros(npix)
            m[10] = 1
            M = hp.alm2map(hp.map2alm(m)*H, nside, verbose = False)
            G += H/max(M)
            T = harp.gettransmatrix(self.harpix, nside, nest = False, counts = True)

            def X1(x):
                z = harp.trans_data(T, x)
                b = np.zeros_like(z)
                if self.harpix.dims is not ():
                    for i in range(self.harpix.dims[0]):
                        alm = hp.map2alm(z[:,i])
                        alm *= G
                        b[:,i] = hp.alm2map(alm, nside, verbose = False)
                else:
                    alm = 1e30*hp.map2alm(z/1e30, iter = 0)  # Older but faster routine
                    alm *= G
                    b = hp.alm2map(alm, nside, verbose = False)
                return harp.trans_data(T.T, b)
        else:
            vec = self.harpix.getvec()
            N = len(self.harpix.data)
            M = np.zeros((N, N))
            #corr = 2.**(self.harpix.order-2)
            sigma2 = np.deg2rad(corrlength)**2
            for i in range(N):
                dist = hp.rotator.angdist(vec[:,i], vec)
                M[i] = np.exp(-dist**2/2/sigma2)
            #for i in range(N):
            #   for j in range(N):
            #      pass
            #      #M[i,j] *= (corr[i]*corr[j])**2
            def X1(x):
                return M.dot(x)

        def X0(x):
            if Sigma is not None:
                x = Sigma.dot(x.T).T
            return x

        self.Xlist.append([X0, X1])

        return self

    def _matvec(self,x):
        result = np.zeros(self.N)
        for F, X in zip(self.Flist, self.Xlist):
            Y = x.reshape((-1,)+self.harpix.dims)*F

            Y = X[0](Y)
            Y = X[1](Y)

            result += (Y.reshape((-1,)+self.harpix.dims)*F).flatten()
        return result

def getsigma(x, f):
    X, Y = np.meshgrid(x,x)
    Sigma = f(X,Y)
    A = 1/np.sqrt(np.diag(Sigma))
    Sigma = np.diag(A).dot(Sigma).dot(np.diag(A))
    return Sigma

def getmodelinput(signals, noise, systematics, exposure):
    # Everything is intensity
    S = [sig.getformattedlike(signals[0]).getdata(mul_sr=True).flatten() for sig in signals]
    N = noise.getformatte_like(signals[0]).getdata(mul_sr=True).flatten()
    SYS = HarpixSigma(signals[0])
    if systematics is None:
        SYS = None
    else:
        for sys in systematics:
            SYS.addsystematics(**sys)
    if isinstance(exposure, float):
        E = np.ones_like(N)*exposure
    else:
        E = exposure.getformattedlike(signals[0]).getdata().flatten()
    return S, N, SYS, E

class Logbins(object):
    def __init__(self, start, stop, num):
        """Return log-bin object.

        Parameters
        ----------
        start : float
            ``base ** start`` is the starting value of the sequence.
        stop : float
             ``base ** stop`` is the final value of the sequence, unless `endpoint`
             is False.  In that case, ``num + 1`` values are spaced over the
             interval in log-space, of which all but the last (a sequence of
             length `num`) are returned.
         num : integer
             Number of samples to generate.

         Returns
         -------
         out : Logbins

         """

        assert start < stop
        self.start = start
        self.stop = stop
        self.num = num 

        self.bounds = np.logspace(start, stop, num+1)
        self.bins = np.array(zip(self.bounds[:-1], self.bounds[1:]))
        self.widths= self.bins[:,1]-self.bins[:,0]
        self.means = self.bins.prod(axis=1)**0.5

    def integrate(self, function):
        out = []
        for xmin, xmax in self.bins:
            o, oerr = quad(function, xmin, xmax)
            out.append(o)
        return np.array(out)

#    def average(self, function):
#        out = []
#        for xmin, xmax in self.bins:
#            o, oerr = quad(function, xmin, xmax)
#            out.append(o/(xmax-xmin))
#        return np.array(out)
#

class Convolution1D(object):
    """General 1-D covolution object."""
    def __init__(self, bins, sigma):
        """Create Convolution1D object, assuming Gaussian kernal.

        Arguments
        ---------
        bins : Logbin or Linbin object
        sigma: {float, function}
            Relative 1-sigma width.
        """
        self.bins = bins
        if isinstance(sigma, float):
            self.sigma = lambda x: np.ones_like(x)*sigma
        else:
            self.sigma = sigma
        self.K = self._create_response_matrix()

    @staticmethod
    def _lognormal_kernal(x, x_c, sigma):
        """Log-normal distrbution, centered on log(x_c), width sigma."""
        return 1/np.sqrt(2*np.pi)/x/sigma*np.exp(-0.5*(np.log(x/x_c))**2/sigma**2)

    # FIXME:  Improve treatment at boundaries
    def _create_response_matrix(self):
        K = []
        for x0 in self.bins.means:
            s0 = self.sigma(x0)
            kernal = lambda x: self._lognormal_kernal(x, x0, s0)
            k = self.bins.integrate(kernal)
            K.append(k)
        K = np.array(K)
        return K

    def __call__(self, mu):
        return self.K.dot(mu)

