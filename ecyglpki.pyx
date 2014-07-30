# ecyglpki.pyx: Cython interface for GLPK

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
#  epyglpki is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#  details.
#
#  You should have received a copy of the GNU General Public License
#  along with epyglpki. If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################


cimport glpk
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
import numbers
import collections.abc


def GLPK_version():
    return glpk.version().decode()


cdef char* str2chars(str string) except NULL:
    cdef bytes encoded_string = string.encode()
    return encoded_string


cdef char* name2chars(str name) except NULL:
    """Check whether a name is acceptable to GLPK

    Names acceptable to GLPK may not exceed 255 bytes. We encode a string in
    UTF-8, so it must not exceed 255 bytes *encoded as UTF-8*.

    """
    cdef bytes encoded_name = name.encode()
    if len(encoded_name) > 255:
        raise ValueError("Name must not exceed 255 bytes.")
    return encoded_name

cdef str chars2name(const char* chars):
    return '' if chars is NULL else chars.decode()


include 'ecyglpki-smcp.pxi'

include 'ecyglpki-bfcp.pxi'

include 'ecyglpki-iptcp.pxi'

include 'ecyglpki-iocp.pxi'

include 'ecyglpki-problem.pxi'

include 'ecyglpki-tree.pxi'

include 'ecyglpki-graph.pxi'
