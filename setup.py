# setup.py: setup script for ecyglpki

###############################################################################
#
#  This code is part of ecyglpki (a Cython GLPK interface).
#
#  Copyright (C) 2014 Erik Quaeghebeur. All rights reserved.
#
#  epyglpki is free software: you can redistribute it and/or modify it under
#  the terms of the GNU General Public License as published by the Free
#  Software Foundation, either version 3 of the License, or (at your option)
#  any later version.
#
#  epyglpki is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#  details.
#
#  You should have received a copy of the GNU General Public License
#  along with epyglpki. If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Operating System :: POSIX",
    "Natural Language :: English",
    "Programming Language :: Cython",
    "Programming Language :: Python :: 3.3",
    "Topic :: Scientific/Engineering :: Mathematics"
    ]

epyglpki_ext = Extension('ecyglpki', ['ecyglpki.pyx'], libraries=['glpk'])
epyglpki_ext.cython_directives = {'embedsignature': True}

setup(
    name = 'ecyglpki',
    url = 'https://github.com/equaeghe/ecyglpki',
    author = 'Erik Quaeghebeur',
    author_email = 'epyglpki@equaeghe.nospammail.net',
    license = 'GPL',
    classifiers = classifiers,
    cmdclass = {'build_ext': build_ext},
    ext_modules = [epyglpki_ext]
)
