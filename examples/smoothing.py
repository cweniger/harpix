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
def smoothing(outmap, inmap, sigma, sigmacut = 3, verbose = False):
    if type(sigma) == float:
        sigma = np.ones(inmap.dims)*sigma
    invecs = inmap.getvec().T
    outvecs = outmap.getvec().T
    indata = inmap.getdata(mul_sr = False)
    sr = inmap.getsr()
    D = 2*np.sin(sigma/2)
    # Factor 1.01 makes sure D_max covers all relevant points.
    D_max = 2*np.sin(max(sigmacut*sigma.max(), np.pi)/2)*1.01
    inv_D = 1/D
    if verbose: print 'Searching nearest neigbors'
    tree = BallTree(invecs)
    ind, dist = tree.query_radius(outvecs, r = D_max, return_distance = True)
    for i in tqdm(range(len(outvecs)), desc = "Convolution"):
        weights = np.exp(-0.5*np.multiply.outer(dist[i], inv_D)**2)
        weightssr = (sr[ind[i]]*weights.T).T
        norm = weightssr.sum(axis=0)
        weightssrnorm = weightssr/np.expand_dims(norm, axis=0)
        x = (indata[ind[i]]*weightssrnorm).sum(axis=0)
        outmap.data[i] = x

def test():
    HP = harp.Harpix(dims = ())
    HP.addiso(nside=32)
    HPout = harp.Harpix(dims = ())
    HPout.addiso(nside=32)

    HP.data[0] = 1.3

    smoothing(HPout, HP, 0.1, verbose = True, sigmacut = 3.0)
    print HPout.getintegral()
    print HP.getintegral()

    m = HPout.gethealpix(nside=64)
    hp.mollview(m[:], nest=True)
    plt.show()

if __name__ == "__main__":
    test()
