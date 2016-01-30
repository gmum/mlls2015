from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("alpy2/cython_routines.pyx")
)