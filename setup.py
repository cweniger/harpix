from distutils.core import setup #, Extension
#import numpy
#from Cython.Distutils import build_ext
#import os

#os.environ["CC"] = "g++"
#os.environ["CXX"] = "g++"

setup(
    name='harpix',
    version='0.1',
    description='A thin healpy wrapper to enable multi-resolution pixelization of the sphere.',
    author='Christoph Weniger',
    author_mail='c.weniger@uva.nl',
    packages=['harpix'],
    package_data={'harpix': [] },
    long_description="""Hierachical Adaptive Resolution PIXelization.""",
)
