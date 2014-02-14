from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

epyglpki_ext = Extension('epyglpki', ['epyglpki.pyx'], libraries=['glpk'])
epyglpki_ext.cython_directives = {'embedsignature': True}

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [epyglpki_ext]
)
