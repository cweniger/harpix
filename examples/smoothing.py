#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import harpix as harp
import healpy as hp
import pylab as plt
from sklearn.neighbors import BallTree
from tqdm import tqdm

"""
Kernal smoothing of harpix maps in real space.

Works best for small patches of the sky.  For the convolution of full sky maps
use standard healpy routines, based on spherical harmonics.

Arguments
---------
* `outmap` [harpix] : Output map.  Only pixelization is relevant, data will be
  overwritten (but should have same dimensionality as input map).
* `inmap` [harpix] : Input map.
* `sigma` [float] : Width of gaussian kernel in standard deviations (rad).
* `sigmacut` [float] : Effective size of convolution kernel, in standard
  deviations.
* `verbose` [boolean] : Report progress.

"""
def smoothing(outmap, inmap, sigma, sigmacut = 3, renormalize_kernel = True, verbose = False):
    if type(sigma) == float:
        sigma = np.ones(inmap.dims)*sigma
    invecs = inmap.getvec().T
    outvecs = outmap.getvec().T
    indata = inmap.getdata(mul_sr = not renormalize_kernel)
    in_sr = inmap.getsr()
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
            norm = weightssr.sum(axis=0)
            weightssr = weightssr/np.expand_dims(norm, axis=0)
            x = (indata[ind[i]]*weightssr).sum(axis=0)
            outmap.data[i] = x
        else:
            weights = np.exp(-0.5*np.multiply.outer(dist[i], inv_D)**2)/D2
            x = (indata[ind[i]]*weights).sum(axis=0)
            outmap.data[i] = x

def test():
    HP = harp.Harpix(dims = ())
    HP.addiso(nside=32)
    HP.adddisc((0., 0.), 10., 64, fill = 1.)
    HP.adddisc((0., 0.), 2.0, 256, fill = 2.)

    HPout = harp.Harpix(dims = ())
    HPout.adddisc((0., 0.), 40., 64)
    #HPout.addiso(nside=32)

    smoothing(HPout, HP, 0.06, verbose = True, sigmacut = 3.0,
            renormalize_kernel = True)
    print HP.getintegral()
    print HPout.getintegral()

    m = HPout.gethealpix(nside=64)
    hp.mollview(m[:], nest=True)
    plt.show()

if __name__ == "__main__":
    test()
