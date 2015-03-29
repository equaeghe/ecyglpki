# setup.py: setup script for ecyglpki

###############################################################################
#
#  This code is part of ecyglpki (a Cython GLPK interface).
#
#  Copyright (C) 2014 Erik Quaeghebeur. All rights reserved.
#
#  ecyglpki is free software: you can redistribute it and/or modify it under
#  the terms of the GNU General Public License as published by the Free
#  Software Foundation, either version 3 of the License, or (at your option)
#  any later version.
#
#  ecyglpki is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#  details.
#
#  You should have received a copy of the GNU General Public License
#  along with ecyglpki. If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

from distutils.core import setup
from distutils.extension import Extension

import os.path
if os.path.exists('ecyglpki.c'):
    from distutils.command.build_ext import build_ext
    ecyglpki_ext = Extension('ecyglpki', ['ecyglpki.c'], libraries=['glpk'])
else:  # Cython needs to create the c file
    from Cython.Distutils import build_ext
    ecyglpki_ext = Extension('ecyglpki', ['ecyglpki.pyx'], libraries=['glpk'])
    ecyglpki_ext.cython_directives = {'embedsignature': True}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Natural Language :: English",
    "Programming Language :: Cython",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Mathematics"
    ]

setup(
    name = 'ecyglpki',
    description = 'A Cython GLPK interface',
    version = '0.2.0',
    url = 'https://github.com/equaeghe/ecyglpki',
    author = 'Erik Quaeghebeur',
    author_email = 'ecyglpki@equaeghe.nospammail.net',
    license = 'GPL',
    classifiers = classifiers,
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ecyglpki_ext]
)
