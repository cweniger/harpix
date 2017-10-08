"""`harpix` extends `healpy` to multi-resolution partial sky maps.

NOTE: The package is still in development phase.  Use at your own risk.


Motivation
----------

With `harpix` you can generate, handle and modify multi-resolution partial sky
maps, based on the HEALPix pixelization scheme.


Documentation
-------------

The documentation of `harpix` can be found on
[github.io](https://cweniger.github.io/harpix).


Installation
------------

`swordfish` has been tested with Python 2.7.13 and the packages

- `numpy 1.13.1`
- `scipy 0.19.0`
- `matplotlib 2.0.0`
- `healpy 1.10.3`

Let us know if you run into problems.

`harpix` can be installed by invoking

    git clone https://github.com/cweniger/harpix
    cd harpix
    python setup.py install


Citation
--------

If you use `harpix` for scientific publications, please acknowledge the use of
this package with a link to this website.
"""

from harpix.core import *
__all__ = ["zeros_like", "get_trans_matrix", "Harpix"]
