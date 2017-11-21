"""
setup.py

This is the distutils setup file used to build the Cython utilities used by draculab.
"""
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("cython_utils", ["cython_utils.c"],
            include_dirs=[numpy.get_include()]),
    ],
)


setup(
    name = 'draculab cythonized functions',
    ext_modules = cythonize("cython_utils.pyx"),
    include_dirs=[numpy.get_include()]
)
