# epyglpki.pyx: Cython/Python interface for GLPK

###############################################################################
#
#  This code is part of epyglpki (a Cython/Python GLPK interface).
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


cimport glpk
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
import numbers
import collections.abc
import itertools

include 'glpk-constants.pxi'


def GLPK_version():
    return glpk.version().decode()

cdef class Name(unicode):
    """A name acceptable to GLPK

    Names acceptable to GLPK may not exceed 255 bytes. We encode a string in
    UTF-8, so it must not exceed 255 bytes *encoded as UTF-8*.

    """

    cdef bytes _encoded

    def __init__(self, unicode name):
        self._encoded = name.encode()
        n = len(self._encoded)
        if n > 255:
            raise ValueError("'Name' must not exceed 255 bytes encoded as " +
                             "UTF-8, yours is " + str(n) + ".")

    cdef const char* _to_chars(self) except NULL:
        return self._encoded

    @classmethod
    def _from_chars(cls, const char* chars):
        return '' if chars is NULL else chars.decode()


include 'epyglpki-program.pxi'


include 'epyglpki-varstraints.pxi'


include 'epyglpki-varstraint.pxi'


include 'epyglpki-objective.pxi'


include 'epyglpki-solvers.pxi'
