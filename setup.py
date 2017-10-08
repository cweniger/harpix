from distutils.core import setup #, Extension

setup(
    name='harpix',
    version='0.1',
    description='A Python healpy extension for multi-resolution partial sky maps.',
    author='Christoph Weniger',
    author_mail='c.weniger@uva.nl',
    packages=['harpix'],
    package_data={'harpix': [] },
    long_description="""Hierachical Adaptive Resolution PIXelization of the sphere.""",
)
