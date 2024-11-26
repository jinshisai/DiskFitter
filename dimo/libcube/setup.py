#from distutils.core import setup
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'ttldisk',
    ext_modules = cythonize('linecube.pyx'),
    include_dirs = [numpy.get_include()],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )