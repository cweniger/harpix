#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import healpy as hp
import pylab as plt
import scipy.sparse as sp
import inspect
from copy import deepcopy

# Hierarchical Adaptive Resolution Pixelization of the Sphere
# A thin python wrapper around healpix

def zeros_like(h):
    H = deepcopy(h)
    H.data *= 0.
    return H

def trans_data(T, data):
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

def get_trans_matrix(IN, OUT, nest = True, counts = False):
    if isinstance(IN, HARPix) and isinstance(OUT, HARPix):
        if counts: raise NotImplementedError()
        return _get_trans_matrix_HARP2HARP(IN, OUT)
    elif isinstance(IN, HARPix) and isinstance(OUT, int):
        return _get_trans_matrix_HARP2HPX(IN, OUT, nest = nest, counts = counts)
    elif isinstance(IN, int) and isinstance(OUT, HARPix):
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
    Hin = HARPix()
    Hin.add_iso(nside)
    if not nest:
        Hin.ipix = hp.ring2nest(nside, Hin.ipix)
    T = get_trans_matrix(Hin, Hout)
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


class HARPix():
    def __init__(self, dims = ()):
        self.ipix = np.empty((0,), dtype=np.int64)
        self.order = np.empty((0,), dtype=np.int8)
        self.dims = dims
        self.data = np.empty((0,)+self.dims, dtype=np.float64)

    @classmethod
    def from_healpix(cls, m, nest = True):
        npix = len(m)
        nside = hp.npix2nside(npix)
        order = hp.nside2order(nside)

        dims = np.shape(m[0])
        r = cls(dims = dims)
        r.ipix = np.arange(npix, dtype=np.int64)
        if not nest:
            i = hp.nest2ring(nside, r.ipix)
            r.data = m[i]
        else:
            r.data = m
        r.order = np.ones(npix, dtype=np.int8)*order
        return r

    @classmethod
    def from_file(cls, filename):
        data = np.load(filename)
        r = cls(dims = data['dims'])
        r.ipix = data['ipix']
        r.order = data['order']
        r.data = data['data']
        return r

    def expand(self, values):
        """Return new HARPix object with expanded data."""
        n = len(values)
        dims = self.dims + (n,)
        r = HARPix(dims = dims)
        r.order = self.order
        r.ipix = self.ipix
        data = np.repeat(self.data.flatten(), n)*np.tile(values, len(self.data.flatten()))
        r.data = data.reshape((-1,)+dims)
        return r

    def write_file(self, filename):
        np.savez(filename, ipix = self.ipix, order = self.order, dims = self.dims, data = self.data)
        return self

    def set_data(self, data):
        assert np.prod(np.shape(self.data)) == np.prod(np.shape(data))
        self.data = data.reshape((-1,)+self.dims)
        return self

    @classmethod
    def from_data(cls, h, data, div_sr = False):
        H = deepcopy(h)
        H.set_data(data)
        if div_sr:
            H._div_sr()
        return H

    def get_data(self, mul_sr = False):
        if not mul_sr:
            return deepcopy(self.data)
        else:
            sr = 4*np.pi/12*4.**-self.order
            return (self.data.T*sr).T

    def print_info(self):
        print "Number of pixels: %i"%len(self.data)
        print "Minimum nside:    %i"%hp.order2nside(min(self.order))
        print "Maximum nside:    %i"%hp.order2nside(max(self.order))
        return self

    def add_singularity(self, vec, r0, r1, n = 100):
        sr0 = np.deg2rad(r0)**2*np.pi/n
        sr1 = np.deg2rad(r1)**2*np.pi/n
        order0 = int(np.log(4*np.pi/12/sr0)/np.log(4))+1
        order1 = int(np.log(4*np.pi/12/sr1)/np.log(4))+1
        for o in range(order1, order0+1):
            r = r1/2**(o-order1)
            nside = hp.order2nside(o)
            self.add_disc(vec, r, nside, clean = False)
        self._clean()
        return self

    def add_ipix(self, ipix, order, clean = True, fill = 0., insert_first = False):
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

    def add_iso(self, nside = 1, clean = True, fill = 0.):
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

    def add_disc(self, vec, radius, nside, clean = True, fill = 0.):
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

    def add_polygon(self, vertices, nside, clean = True, fill = 0.):
        order = hp.nside2order(nside)
        ipix = hp._query_disc.query_polygon(nside, vertices, nest=True)
        self.ipix = np.append(self.ipix, ipix)
        self.data = np.append(self.data,
                np.ones((len(ipix),)+self.dims)*fill, axis=0)
        self.order = np.append(self.order, order*np.ones(len(ipix), dtype=np.int8))
        if clean:
            self._clean()
        return self

    def get_formatted_like(self, h):
        T = get_trans_matrix(self, h)
        H = deepcopy(h)
        H.data = trans_data(T, self.data)
        return H

    def get_healpix(self, nside, idxs = (), nest = True):
        T = get_trans_matrix(self, nside, nest = nest)
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
        T = get_trans_matrix(other, self)
        self.data += trans_data(T, other.data)
        return self

    def __mul__(self, other):
        if isinstance(other, HARPix):
            h1 = deepcopy(self)
            h2 = deepcopy(other)
            h1.add_ipix(other.ipix, other.order, insert_first=True)
            h2.add_ipix(self.ipix, self.order)
            h1.data *= h2.data
            return h1
        elif isinstance(other, float):
            this = deepcopy(self)
            this.data *= other
            return this
        else:
            raise NotImplementedError

    def __add__(self, other):
        """Add to dense map."""
        if isinstance(other, HARPix):
            this = deepcopy(self)
            this.data = np.append(this.data, other.data, axis=0)
            this.ipix = np.append(this.ipix, other.ipix)
            this.order = np.append(this.order, other.order)
            this._clean()
            return this
        else:
            raise NotImplementedError

    def remove_zeros(self):
        mask = self.data != 0.
        self.ipix = self.ipix[mask]
        self.data = self.data[mask]
        self.order = self.order[mask]
        return self

    def get_area(self):
        """Return area covered by map in steradian."""
        sr = 4*np.pi/12*4.**-self.order
        return sum(sr)

    def get_integral(self):
        """Return area covered by map in steradian."""
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

    def mul_func(self, func, mode = 'lonlat', **kwargs):
        values = self._evalulate(func, mode = mode, **kwargs)
        self.data *= values
        return self

    def add_func(self, func, mode = 'lonlat', **kwargs):
        values = self._evalulate(func, mode = mode, **kwargs)
        self.data += values
        return self

    def _evalulate(self, func, mode = 'lonlat', center = None):
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
        f = np.vectorize(func, signature = signature)
        if mode == 'lonlat':
            lon, lat = self.get_lonlat()
            values = f(lon, lat)
        elif mode == 'dist':
            dist = self.get_dist(center[0], center[1])
            values = f(dist)
        else:
            raise KeyError("Mode unknown.")
        return values

    def apply_mask(self, mask_func, mode = 'lonlat'):
        self.mul_func(mask_func, mode = mode)
        self.remove_zeros()
        return self

    def get_dist(self, lon, lat):
        lonV, latV = self.get_lonlat()
        dist = hp.rotator.angdist([lon, lat], [lonV, latV], lonlat=True)
        dist = np.rad2deg(dist)
        return dist

    def add_random(self):
        self.data += np.random.random(np.shape(self.data))
        return self

    def get_lonlat(self):
        return self._get_position(lonlat = True)

    def get_vec(self):
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

def _test():
    dims = (10,3)
    D = np.ones(dims)
    h1 = HARPix(dims = dims).add_iso(32).add_random()
    h2 = HARPix(dims = dims).add_iso(64).add_random()

    h1.add_singularity((0,0), 0.1, 20, n = 100)#.add_random()
    h1.add_singularity((0,-10), 0.1, 20, n = 100)#.add_random()
    h2.add_singularity((0,10), 0.1, 20, n = 100)#.add_random()
    h1.add_func(lambda d: D*0.1/d, center = (0,10.0), mode='dist')
    h1.add_func(lambda d: D*0.1/d, center = (0,0.0), mode='dist')
    h2.add_func(lambda d: D*0.1/d, center = (0,-10.0), mode='dist')

    h1 += h2
    quit()

    T = get_trans_matrix(h1, h2)
    h2.data = trans_data(T, h1.data)

    #print h2.get_integral()
#    T1 = get_trans_matrix(h1, 64)
#    T2 = get_trans_matrix(64, h1)
#    print h1.get_integral()

#    h1.data = T2.dot(T1.dot(h1.data))

    m = h1.get_healpix(256)
    hp.mollview(np.log10(m), nest =True)
    plt.savefig('test.eps')
    quit()
    #h2.add_random()
    T = get_trans_matrix(h1, h2)
    h2.data += T.dot(h1.data)
    print h2.get_integral()
    m = h2.get_healpix(256)
    hp.mollview(np.log10(m), nest =True)
    plt.savefig('test.eps')

    quit()

    h.add_iso(nside = 1)
    h.add_singularity((0,0.0), 0.1, 100, n = 100)#.add_random()
    h.add_func(lambda d: 0.1/d, center = (0,0.0), mode='dist')
    h.add_singularity((80,20), 0.1, 100, n = 100)#.add_random()
    #h.add_singularity((30,20), 0.1, 100, n = 1000)#.add_random()
    #h.add_func(lambda d: 0.1/d, center = (30,20), mode='dist')
    #h.add_random()
    #h.apply_mask(lambda l, b: abs(b) < 3)
    h.print_info()

    m = h.get_healpix(32)
    hp.mollview(np.log10(m), nest = True, cmap='gnuplot')
    #hp.cartview(np.log10(m), nest = True, cmap='gnuplot', lonra = [-1, 1],
          #  latra = [-1, 1])
    print h.get_integral()
    h.add_iso(nside = 4)
    h.add_singularity((0,0), 0.1, 100, n = 1000)#.add_random()
    h.add_func(lambda d: 0.1/d, center = (0,0), mode='dist')
    h.add_singularity((30,20), 0.1, 100, n = 1000)#.add_random()
    h.add_func(lambda d: 0.1/d, center = (30,20), mode='dist')
    h.add_random()
    #h.apply_mask(lambda l, b: abs(b) < 3)
    h.print_info()

    m = h.get_healpix(32)
    m = hp.smoothing(m, sigma =1, nest=True)
    hp.mollview(np.log10(m), nest = True, cmap='gnuplot')
    quit()
    quit()

    npix = hp.nside2npix(8)
    m = np.random.random((npix, 2,3))
    h=HARPix.from_healpix(m)
    m = h.get_healpix(128, idxs = (1,1))
    h.print_info()
    hp.mollview(np.log10(m), nest = True, cmap='gnuplot')
    plt.savefig('test2.eps')

    h = HARPix(dims=(10,)).add_iso(fill = 100)
    for i in range(10):
        lonlat = (40*i, 10*i)
        h0 = HARPix(dims=(10,))
        h0.add_peak(lonlat, .01, 10)
        print np.shape(h0.data)
        x = np.linspace(1, 10, 10)
        h0.add_func(lambda dist: x/(dist+0.01), mode = 'dist', center = lonlat)
        h += h0
    m = h.get_healpix(128, idxs=(4,))
    h.print_info()
    #hp.mollview(np.log10(m), nest = True, cmap='gnuplot', min = 1, max = 4)
    hp.cartview(np.log10(m), cmap='gnuplot', min = 1, max = 4)
    plt.savefig('test.eps')
    hp.mollview(np.log10(m), nest = True, cmap='gnuplot', min = 1, max = 4)
    plt.savefig('test3.eps')

if __name__ == "__main__":
    _test()
