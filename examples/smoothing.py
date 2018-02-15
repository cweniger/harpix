#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import harpix as harp
import healpy as hp
import pylab as plt
from sklearn.neighbors import BallTree
from tqdm import tqdm

def smoothing(outmap, inmap, sigma, sigmacut = 3, verbose = False,
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
    if verbose: print 'Searching nearest neigbors'
    tree = BallTree(invecs)
    ind, dist = tree.query_radius(outvecs, r = D_max, return_distance = True)
    for i in tqdm(range(len(outvecs)), desc = "Convolution"):
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

def integral_test1():
    for renkern in [True, False]:
        HP = harp.Harpix(dims = ())
        HPout = harp.Harpix(dims = ())

        #HP.addiso(nside=4)
        HP.adddisc((0.5, 0.), 1.0, 256, fill = 2.)

        HPout.adddisc((0., 0.), 50., 64)
        smoothing(HPout, HP, 0.2, verbose = True, renormalize_kernel =
                renkern)
        print "renormalize_kernel =", renkern
        print "inmap integral =", HP.getintegral()
        print "outmap integral =", HPout.getintegral()

def integral_test2():
    for renkern in [True, False]:
        HP = harp.Harpix(dims = ())
        HPout = harp.Harpix(dims = ())

        #HP.addiso(nside=1)
        HP.adddisc((0.0, 0.), 1.5, 32, fill = 2.)

        HPout.adddisc((0., 0.), 40., 32)
        smoothing(HPout, HP, 0.20, verbose = True, renormalize_kernel =
                renkern)

        print "renormalize_kernel =", renkern
        print "inmap integral =", HP.getintegral()
        print "outmap integral =", HPout.getintegral()

def integral_test3():
    for renkern in [True, False]:
        HP = harp.Harpix(dims = ())
        HPout = harp.Harpix(dims = ())

        #HP.addiso(nside=4)
        HP.adddisc((0.5, 0.), 1.0, 256, fill = 2.)

        HPout.adddisc((0., 0.), 50., 64)
        smoothing(HPout, HP, 0.2, verbose = True, renormalize_kernel =
                renkern)
        print "renormalize_kernel =", renkern
        print "inmap integral =", HP.getintegral()
        print "outmap integral =", HPout.getintegral()

def test_size():
    phi = 0.345
    print "Size [deg]", np.rad2deg(phi)
    area = phi**2*np.pi
    print "Plain area [deg]", area
    area = 2*np.pi*(1-np.cos(phi))
    print "True area [deg]", area

def test():
    HP = harp.Harpix(dims = ())
    #HP.adddisc((0., 0.), 50., 1, fill = 0.)
    HP.addiso(nside=32)
    HP.adddisc((0., 0.), 1.0, 64, fill = 2.)

    HPout = harp.Harpix(dims = ())
    HPout.adddisc((0., 0.), 10., 64)
    #HPout.addiso(nside=32)

    smoothing(HPout, HP, 0.001, verbose = True, sigmacut = 3.0)
#    print len(HP.data)
#    print len(HPout.data)
    print HP.getintegral()
    print HPout.getintegral()

#    m = HPout.gethealpix(nside=64)
#    hp.mollview(m[:], nest=True)
#    plt.show()

if __name__ == "__main__":
    #integral_test1()
    integral_test2()
    #integral_test3()
    #test_size()
