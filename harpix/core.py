#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import healpy as hp
import pylab as plt
import scipy.sparse as sp
import inspect
from copy import deepcopy
from sklearn.neighbors import BallTree

# Hierarchical Adaptive Resolution Pixelization of the Sphere
# A thin python wrapper around healpix

def zeroslike(h):
    """Create zero Harpix object with shape of `h`.

    Parameters
    ----------
    * `h` [Harpix]: Harpix object.
    """
    H = deepcopy(h)
    H.data *= 0.
    return H

def _trans_data(T, data):
    # FIXME: Rewrite just as matrix products
    dims = np.shape(data)[1:]
    Nout, Nin = T.shape
    out = np.zeros((Nout,)+dims)
    if len(dims) == 0:
        out = T.dot(data)
    elif len(dims) == 1:
        for i in range(dims[0]):
            out[:,i] = T.dot(data[:,i])
    elif len(dims) == 2:
        for i in range(dims[0]):
            for j in range(dims[1]):
                out[:,i,j] = T.dot(data[:,i,j])
    else:
        raise NotImplementedError()
    return out

def _get_trans_matrix(IN, OUT, nest = True, counts = False):
    """Return transformation matrix.

    Parameters
    ----------
    * `IN` [Harpix]: Harpix input map.
    * `OUT` [Harpix]: Harpix output map.

    Returns
    -------
    * `M` [array-like]: Matrix that transforms flattened data of format IN onto
      flattened data of format OUT.
    """
    # FIXME: Update description
    if isinstance(IN, Harpix) and isinstance(OUT, Harpix):
        if counts: raise NotImplementedError()
        return _get_trans_matrix_HARP2HARP(IN, OUT)
    elif isinstance(IN, Harpix) and isinstance(OUT, int):
        return _get_trans_matrix_HARP2HPX(IN, OUT, nest = nest, counts = counts)
    elif isinstance(IN, int) and isinstance(OUT, Harpix):
        if counts: raise NotImplementedError()
        return _get_trans_matrix_HPX2HARP(IN, OUT, nest = nest)
    else:
        raise TypeError("Invalid types.")

def _get_trans_matrix_HARP2HPX(Hin, nside, nest = True, counts = False):
    npix = hp.nside2npix(nside)
    fullorder = hp.nside2order(nside)
    fullmap = np.zeros(npix)
    N = len(Hin.data)

    # COO matrix setup
    row =  []
    col =  []
    data = []
    shape = (npix, N)
    num = np.arange(N)

    for o in np.unique(Hin.order):
        mask = Hin.order == o
        if o > fullorder:
            idx = Hin.ipix[mask] >> (o-fullorder)*2
            if not counts:
                dat = np.ones(len(idx)) / 4**(o-fullorder)
            else:
                dat = np.ones(len(idx))
            if not nest: idx = hp.nest2ring(nside, idx)
            row.extend(idx)
            col.extend(num[mask])
            data.extend(dat)
        elif o == fullorder:
            idx = Hin.ipix[mask]
            if not nest: idx = hp.nest2ring(nside, idx)
            dat = np.ones(len(idx))
            row.extend(idx)
            col.extend(num[mask])
            data.extend(dat)
        elif o < fullorder:
            idx = Hin.ipix[mask] << (fullorder-o)*2
            if not counts:
                dat = np.ones(len(idx))
            else:
                dat = np.ones(len(idx)) / 4**(fullorder-o)
            for i in range(0, 4**(fullorder-o)):
                if not nest:
                    row.extend(hp.nest2ring(nside, idx+i))
                else:
                    row.extend(idx+i)
                col.extend(num[mask])
                data.extend(dat)

    M = sp.coo_matrix((data, (row, col)), shape = shape)
    M = M.tocsr()
    return M

def _get_trans_matrix_HPX2HARP(nside, Hout, nest = True):
    Hin = Harpix()
    Hin.addiso(nside)
    if not nest:
        Hin.ipix = hp.ring2nest(nside, Hin.ipix)
    T = _get_trans_matrix(Hin, Hout)
    return T

def _get_trans_matrix_HARP2HARP(Hin, Hout):
    orders1 = np.unique(Hout.order)
    orders2 = np.unique(Hin.order)
    N1 = len(Hout.data)
    N2 = len(Hin.data)
    num1 = np.arange(N1)
    num2 = np.arange(N2)
    A = sp.coo_matrix((N1,N2))
    for o1 in orders1:
        for o2 in orders2:
            #print o1, o2
            order = min(o1, o2)
            npix = 12*(2**order)**2

            mask1 = Hout.order == o1
            idx1 = Hout.ipix[mask1] >> (o1-order)*2
            dat1 = np.ones(len(idx1))
            row1 = num1[mask1]
            col1 = idx1
            M1 = sp.coo_matrix((dat1, (row1, col1)), shape=(N1,npix))


            mask2 = Hin.order == o2
            idx2 = Hin.ipix[mask2] >> (o2-order)*2
            dat2 = np.ones(len(idx2)) / 4**(o2-order)
            row2 = idx2
            col2 = num2[mask2]
            M2 = sp.coo_matrix((dat2, (row2, col2)), shape=(npix,N2))

            M1 = M1.tocsr()
            M2 = M2.tocsc()
            A += M1.dot(M2)

    return A.tocsr()


class Harpix():
    """Thin healpy wrapper to allow multi-resolution maps.
    """
    def __init__(self, dims = ()):
        """Constructor.

        The constructor returns an empty Harpix object without any pixels set.

        Parameters
        ----------
        * `dims` [tuple of integers]: Dimensions of per pixel data
        """
        self.ipix = np.empty((0,), dtype=np.int64)
        self.order = np.empty((0,), dtype=np.int8)
        self.dims = dims
        self.data = np.empty((0,)+self.dims, dtype=np.float64)

    @classmethod
    def fromhealpix(cls, m, indices = None, nside = None, nest = True, div_sr
            = False):
        """Construct Harpix object from regular healpix data.

        Parameters
        ----------
        * `m` [array (N, d1, d2, d3, ...)]: Data array, with `N` sky pixels.
        * `indices` [vector with length `N` OR `None`]: Healpix indices.  If
          `None`, assume full sky covery and proper pixel ordering.
        * `nside` [integer]: `nside` of map, only required if `indices` is not
          `None`.
        * `nest` [boolean]: If `True`, assume that `m` is in nest format.
        * `div_sr` [boolean]: If `True`, divide by sr pixel size before storing
          data.
        """
        dims = np.shape(m[0])
        if indices is None:
            npix = len(m)
            nside = hp.npix2nside(npix)
            if nest:
                data = m
            else:
                i = hp.nest2ring(nside, r.ipix)
                data = m[i]  # Re-order data
            ipix = np.arange(npix, dtype=np.int64)  # nest indices
        elif indices is not None and nside is not None:
            npix = hp.nside2npix(nside)
            data = m
            if nest:
                ipix = indices
            else:
                ipix = hp.ring2nest(nside, indices)
        else:
            raise KeyError("nside not set.")
        order = hp.nside2order(nside)

        r = cls(dims = dims)
        r.ipix = ipix  # Nest indices
        r.order = np.ones(len(ipix), dtype=np.int8)*order
        r.data = data

        if div_sr:
            r._div_sr()

        return r

    @classmethod
    def fromfile(cls, filename):
        """Construct HARPIx object from *.npy file.

        Parameters
        ----------
        * `filename` [string]: Filename of *.npy file.
        """
        data = np.load(filename)
        r = cls(dims = data['dims'])
        r.ipix = data['ipix']
        r.order = data['order']
        r.data = data['data']
        return r

    def expand(self, values):
        """Return new Harpix object with expanded data.

        If `n = len(values)` and `dim = (k, m)`, this method returns an object
        with `dim = (k, m, n)`, and `data[k, m, n] = data_old[k, m] * values[n]`.

        Parameters
        ----------
        * `values` [1-D floats]: List of floats to use in expansion.
        """
        n = len(values)
        dims = self.dims + (n,)
        r = Harpix(dims = dims)
        r.order = self.order
        r.ipix = self.ipix
        data = np.repeat(self.data.flatten(), n)*np.tile(values, len(self.data.flatten()))
        r.data = data.reshape((-1,)+dims)
        return r

    def writefile(self, filename):
        np.savez(filename, ipix = self.ipix, order = self.order, dims = self.dims, data = self.data)
        return self

    def _set_data(self, data):
        """Overwrite data.

        Parameters
        ----------
        * `data` [array-like]: Data that overwrites internal data.
        """
        assert np.prod(np.shape(self.data)) == np.prod(np.shape(data))
        self.data = data.reshape((-1,)+self.dims)
        return self

    @classmethod
    def _from_data(cls, h, data, div_sr = False):
        H = deepcopy(h)
        H._set_data(data)
        if div_sr:
            H._div_sr()
        return H

    def getdata(self, mul_sr = False):
        """Return data array.

        Parameters
        ----------
        * `mul_sr` [boolean]: If `True`, multiply data with pixel size in
        steradian before returning.
        """
        if not mul_sr:
            return deepcopy(self.data)
        else:
            sr = 4*np.pi/12*4.**-self.order
            return (self.data.T*sr).T

    def printinfo(self):
        """Print summary information."""
        print "Number of pixels: %i"%len(self.data)
        print "Minimum nside:    %i"%hp.order2nside(min(self.order))
        print "Maximum nside:    %i"%hp.order2nside(max(self.order))
        return self

    def addsingularity(self, vec, r0, r1, n = 100):
        """Add region with centrally increasing pixel density.

        The grid is defined such that each radius r with r0<r<r1 contains
        at least `n` pixels with decreasing size towards the center.

        Parameters
        ----------
        * `vec` [3-tupel]: Central direction.
        * `r0` [float]: Inner radius [deg]
        * `r1` [float]: Outer radius [deg]
        * `n` [int]: Number of pixels.
        """
        sr0 = np.deg2rad(r0)**2*np.pi/n
        sr1 = np.deg2rad(r1)**2*np.pi/n
        order0 = int(np.log(4*np.pi/12/sr0)/np.log(4))+1
        order1 = int(np.log(4*np.pi/12/sr1)/np.log(4))+1
        for o in range(order1, order0+1):
            r = r1/2**(o-order1)
            nside = hp.order2nside(o)
            self.adddisc(vec, r, nside, clean = False)
        self._clean()
        return self

    def addipix(self, ipix, order, clean = True, fill = 0., insert_first = False):
        """Add pixels according to index.

        Parameters
        ----------
        * `ipix` [integers]: 
        """
        # TODO
        if insert_first:
            self.ipix = np.append(ipix, self.ipix)
            self.order = np.append(order, self.order)
            self.data = np.append(np.ones((len(ipix),)+self.dims)*fill,
                    self.data, axis=0)
        else:
            self.ipix = np.append(self.ipix, ipix)
            self.order = np.append(self.order, order)
            self.data = np.append(self.data,
                    np.ones((len(ipix),)+self.dims)*fill, axis=0)
        if clean:
            self._clean()
        return self

    def addiso(self, nside = 1, clean = True, fill = 0.):
        """Add isotropic component with nside.

        Parameters
        ----------
        * `nside` [integer]: Healpix parameter for isotropic map.
        * `clean` [boolean]: If `False`, skip internal clean-up of overlapping
          pixels.
        * `fill` [float]: Isotropic fill.

        Returns
        -------
        * `self`: Returns this Harpix instance.
        """
        order = hp.nside2order(nside)
        npix = hp.nside2npix(nside)
        ipix = np.arange(0, npix)
        self.ipix = np.append(self.ipix, ipix)
        self.data = np.append(self.data,
                np.ones((len(ipix),)+self.dims)*fill, axis=0)
        self.order = np.append(self.order, order*np.ones(len(ipix), dtype=np.int8))
        if clean:
            self._clean()
        return self

    def adddisc(self, vec, radius, nside, clean = True, fill = 0.):
        """Add disc component.

        Parameters
        ----------
        * `vec` [tuple]: Center of disc.
        * `radius` [float]: Radius of disc to add.
        * `nside` [integer]: Healpix parameter for isotropic map.
        * `clean` [boolean]: If `False`, skip internal clean-up of overlapping
          pixels.
        * `fill` [float]: Value inside disk.

        Returns
        -------
        * `self`: Returns this Harpix instance.
        """
        if len(vec) == 2:
            vec = hp.ang2vec(vec[0], vec[1], lonlat=True)
        radius = np.deg2rad(radius)
        order = hp.nside2order(nside)
        ipix = hp._query_disc.query_disc(nside, vec, radius, nest=True)
        self.ipix = np.append(self.ipix, ipix)
        self.data = np.append(self.data,
                np.ones((len(ipix),)+self.dims)*fill, axis=0)
        self.order = np.append(self.order, order*np.ones(len(ipix), dtype=np.int8))
        if clean:
            self._clean()
        return self

    def addpolygon(self, vertices, nside, clean = True, fill = 0.):
        """Add polygon component.

        Parameters
        ----------
        * `vertices` [tuple]: List of vertices defining polygon.
        * `nside` [integer]: Healpix parameter for isotropic map.
        * `clean` [boolean]: If `False`, skip internal clean-up of overlapping
          pixels.
        * `fill` [float]: Value inside disk.

        Returns
        -------
        * `self`: Returns this Harpix instance.
        """
        order = hp.nside2order(nside)
        ipix = hp._query_disc.query_polygon(nside, vertices, nest=True)
        self.ipix = np.append(self.ipix, ipix)
        self.data = np.append(self.data,
                np.ones((len(ipix),)+self.dims)*fill, axis=0)
        self.order = np.append(self.order, order*np.ones(len(ipix), dtype=np.int8))
        if clean:
            self._clean()
        return self

    def getformattedlike(self, h):
        """Returns new reformatted Harpix object.

        Parameters
        ----------
        * `h` [Harpix]: Harpix object with template format

        Returns
        -------
        * `H` [Harpix]: New Harpix object with data from `self` and format from
          `h`.
        """
        T = _get_trans_matrix(self, h)
        H = deepcopy(h)
        H.data = _trans_data(T, self.data)
        return H

    def gethealpix(self, nside, idxs = (), nest = True):
        T = _get_trans_matrix(self, nside, nest = nest)
        """Returns healpix map.

        Parameters
        ----------
        * `nside` [integral]: Healpix `nside` parameter.
        * `idxs` [integer tupel]: Indices of data slice.
        * `nest` [boolean]: If `False`, return ring-ordered map.

        Returns
        -------
        * `map` [array]: Healpix map.
        """
        if idxs == ():
            return T.dot(self.data)
        elif len(idxs) == 1:
            return T.dot(self.data[:,idxs[0]])
        elif len(idxs) == 2:
            return T.dot(self.data[:,idxs[0], idxs[1]])
        elif len(idxs) == 3:
            return T.dot(self.data[:,idxs[0], idxs[1], idxs[2]])
        else:
            raise NotImplementedError()

    def _clean(self):
        """Iteratively merge overlapping pixels while adding data.
        """
        orders = np.unique(self.order)
        clean_ipix = []
        clean_data = []
        clean_order = []

        for o in np.arange(min(orders), max(orders)+1):
            mask = self.order == o
            maskS = self.order > o

            sub_ipix = self.ipix[maskS] >> 2*(self.order[maskS] - o)

            if o > orders[0]:
                unsubbed = np.in1d(spill_ipix, sub_ipix, invert = True)
                clean_ipix1 = spill_ipix[unsubbed]
                clean_data1 = spill_data[unsubbed]
                spill_ipix1 = np.repeat(spill_ipix[~unsubbed] << 2, 4)
                spill_ipix1 += np.tile(np.arange(4), int(len(spill_ipix1)/4))
                spill_data1 = np.repeat(spill_data[~unsubbed], 4, axis=0)
            else:
                clean_ipix1 = np.empty((0,), dtype=np.int64)
                clean_data1 = np.empty((0,)+self.dims, dtype=np.float64)
                spill_ipix1 = np.empty((0,), dtype=np.int64)
                spill_data1 = np.empty((0,)+self.dims, dtype=np.float64)

            unsubbed = np.in1d(self.ipix[mask], sub_ipix, invert = True)
            clean_ipix2 = self.ipix[mask][unsubbed]
            clean_data2 = self.data[mask][unsubbed]
            spill_ipix2 = np.repeat(self.ipix[mask][~unsubbed] << 2, 4)
            spill_ipix2 += np.tile(np.arange(4), int(len(spill_ipix2)/4))
            spill_data2 = np.repeat(self.data[mask][~unsubbed], 4, axis=0)

            clean_ipix_mult = np.append(clean_ipix1, clean_ipix2)
            clean_data_mult = np.append(clean_data1, clean_data2, axis=0)
            clean_ipix_sing, inverse = np.unique(clean_ipix_mult,
                    return_inverse = True)
            clean_data_sing = np.zeros((len(clean_ipix_sing),)+self.dims)
            np.add.at(clean_data_sing, inverse, clean_data_mult)
            clean_ipix.extend(clean_ipix_sing)
            clean_data.extend(clean_data_sing)
            clean_order.extend(np.ones(len(clean_ipix_sing), dtype=np.int8)*o)

            spill_ipix = np.append(spill_ipix1, spill_ipix2)
            spill_data = np.append(spill_data1, spill_data2, axis=0)

        self.ipix = np.array(clean_ipix)
        self.data = np.array(clean_data)
        self.order = np.array(clean_order)
        return self

    def __iadd__(self, other):
        """Increment map, keeping pixels of original map."""
        T = _get_trans_matrix(other, self)
        self.data += _trans_data(T, other.data)
        return self

    def __mul__(self, other):
        """Multiply maps, and merge pixelization."""
        if isinstance(other, Harpix):
            h1 = deepcopy(self)
            h2 = deepcopy(other)
            h1.addipix(other.ipix, other.order, insert_first=True)
            h2.addipix(self.ipix, self.order)
            h1.data *= h2.data
            return h1
        elif isinstance(other, float):
            this = deepcopy(self)
            this.data *= other
            return this
        else:
            raise NotImplementedError

    def __add__(self, other):
        """Multiply maps, and merge pixelization."""
        if isinstance(other, Harpix):
            this = deepcopy(self)
            this.data = np.append(this.data, other.data, axis=0)
            this.ipix = np.append(this.ipix, other.ipix)
            this.order = np.append(this.order, other.order)
            this._clean()
            return this
        else:
            raise NotImplementedError

    def removezeros(self):
        """Remove pixels with zero data.

        Returns
        -------
        * `self`
        """
        mask = self.data != 0.
        self.ipix = self.ipix[mask]
        self.data = self.data[mask]
        self.order = self.order[mask]
        return self

    def getsr(self):
        """Return pixel size in steradian.

        Returns
        -------
        * `sr` [array]: Pixel size.
        """
        sr = 4*np.pi/12*4.**-self.order
        return sr

    def getarea(self):
        """Return covered area in steradian.

        Returns
        -------
        * `area` [float]: Covered area.
        """
        sr = 4*np.pi/12*4.**-self.order
        return sum(sr)

    def getintegral(self):
        """Return integrated flux over area.

        Returns
        -------
        * `f` [float]: Integral.
        """
        # FIXME: Works only for 0-dim data.
        sr = 4*np.pi/12*4.**-self.order
        M = (self.data.T*sr).T
        return M.sum(axis=0)

    def _mul_sr(self):
        sr = 4*np.pi/12*4.**-self.order
        self.data = (self.data.T*sr).T
        return self

    def _div_sr(self):
        sr = 4*np.pi/12*4.**-self.order
        self.data = (self.data.T/sr).T
        return self

    def mulfunc(self, func, mode = 'lonlat', **kwargs):
        """Evaluate function on map and multiply with data.

        Parameters
        ----------
        * `func` [function]: Function on map.
        * `mode` [str]: Parametrization. `lonlat` or `dist`.
        """
        # FIXME: Better documentation.
        values = self._evalulate(func, mode = mode, **kwargs)
        self.data *= values
        return self

    def addfunc(self, func, mode = 'lonlat', **kwargs):
        """Equivalent to `mulfunc`."""
        # FIXME: Better documentation.
        values = self._evalulate(func, mode = mode, **kwargs)
        self.data += values
        return self

    def _evalulate(self, func, mode = 'lonlat', center = None, single_valued =
            False):
        nargs = len(inspect.getargspec(func).args)
        signature = "()"
        signature += ",()"*(nargs-1)
        signature += "->"
        if self.dims== ():
            signature += "()" 
        elif len(self.dims) == 1:
            signature += "(n)"
        elif len(self.dims) == 2:
            signature += "(n,m)"
        elif len(self.dims) == 3:
            signature += "(n,m,k)"
        else:
            raise NotImplementedError()

        if single_valued:
            f = np.vectorize(func)
        else:
            f = np.vectorize(func, signature = signature)

        if mode == 'lonlat':
            lon, lat = self.getlonlat()
            values = f(lon, lat)
        elif mode == 'dist':
            dist = self.getdist(center[0], center[1])
            values = f(dist)
        else:
            raise KeyError("Mode unknown.")
        return values

    def applymask(self, mask_func, mode = 'lonlat', **kwargs):
        """Apply mask to map.

        Parameters
        ----------
        * `mask_func` [function]: Functional definition of pixel masp.
        """
        mask = self._evalulate(mask_func, mode = mode, single_valued = True, **kwargs)
        self.ipix = self.ipix[mask]
        self.order = self.order[mask]
        self.data = self.data[mask]
        return self

    def getdist(self, lon, lat):
        lonV, latV = self.getlonlat()
        dist = hp.rotator.angdist([lon, lat], [lonV, latV], lonlat=True)
        dist = np.rad2deg(dist)
        return dist

    def addrandom(self):
        self.data += np.random.random(np.shape(self.data))
        return self

    def getlonlat(self):
        return self._get_position(lonlat = True)

    def getvec(self):
        return self._get_position(lonlat = False)

    def _get_position(self, lonlat = False):
        orders = np.unique(self.order)
        if lonlat:
            lon = np.zeros(len(self.data))
            lat = np.zeros(len(self.data))
        else:
            vec = np.zeros((3, len(self.data)))
        for o in orders:
            nside = hp.order2nside(o)
            mask = self.order == o
            ipix = self.ipix[mask]
            if lonlat:
                lon[mask], lat[mask] = hp.pix2ang(nside, ipix, nest = True, lonlat = True)
            else:
                vec[:,mask] = hp.pix2vec(nside, ipix, nest = True)
        if lonlat:
            lon = np.mod(lon+180, 360) - 180
            return lon, lat
        else:
            return vec

    def smooth(self, sigma, sigmacut = 3):
        self._smoothing(self, self, sigma, sigmacut = sigmacut)
        return self

    def smooth_into(self, outmap, sigma, sigmacut = 3):
        self._smoothing(outmap, self, sigma, sigmacut = sigmacut)
        return self

    @staticmethod
    def _smoothing(outmap, inmap, sigma, sigmacut = 3, verbose = False,
            renormalize_kernel = False):
        """
        Kernel smoothing of harpix maps in real space.

        Works best for small patches of the sky.  For the convolution of full sky maps
        use standard healpy routines, based on spherical harmonics.

        NOTE: When setting `renormalize_kernel` to `True`, the sum over weights is only
        taken over pixels that are included in `inmap`.  This implies that, in order to
        get the expected results, `inmap` must cover the area of the `outmap` plus the
        tails of the kernel.

        Arguments
        ---------
        * `outmap` [harpix] : Output map.  Only pixelization is relevant, data will be
          overwritten (but should have same dimensionality as input map).
        * `inmap` [harpix] : Input map.
        * `sigma` [float] : Width of gaussian kernel in standard deviations (rad).
        * `sigmacut` [float] : Effective size of convolution kernel, in standard
          deviations.
        * `renormalize_kernel` [boolean] (default False) : Ensure that kernel weights
          sum up to one.  See above note for more details.
        * `verbose` [boolean] : Report progress.

        """
        if type(sigma) == float:
            sigma = np.ones(inmap.dims)*sigma
        invecs = inmap.getvec().T
        outvecs = outmap.getvec().T
        indata = inmap.getdata(mul_sr = not renormalize_kernel)
        in_sr = inmap.getsr()
        out_sr = outmap.getsr()
        in_pixsize= in_sr**0.5
        out_pixsize= out_sr**0.5
        if (sigma.min() < out_pixsize.max()) or (sigma.min() < in_pixsize.max()):
            print 'ERROR: Pixel size of input or output image is larger than kernel width.'
            print "Minimum kernel width:", sigma.min()
            print "Input max pixsize:", in_pixsize.max()
            print "Output max pixsize:", out_pixsize.max()
            raise ValueError("ERROR: Maximum pixel size of input or output image larger than kernel width.")
        if np.rad2deg(sigma.max()) > 20:
            raise ValueError('WARNING: Maximum kernel width too large (limit is 20 deg).')
        D = 2*np.sin(sigma/2)
        # Factor 1.01 makes sure D_max covers all relevant points.
        D_max = 2*np.sin(max(sigmacut*sigma.max(), np.pi)/2)*1.01
        inv_D = 1/D
        D2 = 2*np.pi*D*D
        if verbose: print 'Searching nearest neigbors.'''
        tree = BallTree(invecs)
        ind, dist = tree.query_radius(outvecs, r = D_max, return_distance = True)
        #for i in tqdm(range(len(outvecs)), desc = "Convolution"):
        if verbose: print 'Convolving...'
        for i in range(len(outvecs)), desc = "Convolution":
            if renormalize_kernel:
                weights = np.exp(-0.5*np.multiply.outer(dist[i], inv_D)**2)
                weightssr = (in_sr[ind[i]]*weights.T).T
                norm = weightssr.sum(axis=0)+1e-100
                weightssr = weightssr/np.expand_dims(norm, axis=0)
                x = (indata[ind[i]]*weightssr).sum(axis=0)
                outmap.data[i] = x
            else:
                weights = np.exp(-0.5*np.multiply.outer(dist[i], inv_D)**2)/D2
                x = (indata[ind[i]]*weights).sum(axis=0)
                outmap.data[i] = x
