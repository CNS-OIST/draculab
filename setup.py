"""
setup.py

This is the distutils setup file used to build the Cython utilities used by draculab.
"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'draculab cythonized functions',
    ext_modules = cythonize("cython_utils.pyx"),
)
