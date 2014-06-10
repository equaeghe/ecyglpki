# ecyglpki-bfcp.pxi: Cython interface for GLPK basis factorization controls

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


# basis factorization approach
cdef strpair2bftype = {
    ('LU', 'Forrest-Tomlin'): glpk.BF_LUF + glpk.BF_FT,
    ('LU', 'Bartels-Golub'): glpk.BF_LUF + glpk.BF_BG,
    ('LU', 'Givens'): glpk.BF_LUF + glpk.BF_GR,
    ('BTLU', 'Bartels-Golub'): glpk.BF_BTF + glpk.BF_BG,
    ('BTLU', 'Givens'): glpk.BF_BTF + glpk.BF_GR
    }
cdef bftype2strpair = {bftype: stringpair
                       for stringpair, bftype in strpair2bftype.items()}


cdef class FactorizationControls:
    """The basis factorization control parameter object

    .. doctest:: FactorizationControls

        >>> p = Problem()
        >>> r = FactorizationControls(p)

    """

    cdef glpk.BasFacCP _bfcp

    def __cinit__(self, Problem problem):
        cdef glpk.ProbObj* _problem = <glpk.ProbObj*>PyCapsule_GetPointer(
                                                problem._problem_ptr(), NULL)
        glpk.get_bfcp(_problem, &self._bfcp)

    property type:
        """The basis factorization type, `str` pairs

        Possible first components:

        * `'LU'`: plain LU factorization
        * `'BTLU'`: block-triangular LU factorization

        Possible second components

        * `'Forrest-Tomlin'`: `Forrest–Tomlin`_ update applied to U
          (only with plain LU factorization)
        * `'Bartels-Golub'`: `Bartels–Golub`_ update applied to Schur
          complement
        * `'Givens'`: Givens rotation update applied to Schur complement

        .. _Forrest–Tomlin: http://dx.doi.org/10.1007/BF01584548
        .. _Bartels–Golub: http://dx.doi.org/10.1145/362946.362974

        .. doctest:: FactorizationControls

            >>> r.type  # the GLPK default
            ('LU', 'Forrest-Tomlin')

        """
        def __get__(self):
            return bftype2strpair[self._bfcp.type]
        def __set__(self, value):
            self._bfcp.type = strpair2bftype[value]

    property piv_tol:
        """Markowitz threshold pivoting tolerance, a |Real| number

        (Value must lie between 0 and 1.)

        """
        def __get__(self):
            return self._bfcp.piv_tol
        def __set__(self, value):
            self._bfcp.piv_tol = float(value)

    property piv_lim:
        """Number of pivot candidates that need to be considered, an `int` ≥1"""
        def __get__(self):
            return self._bfcp.piv_lim
        def __set__(self, value):
            self._bfcp.piv_lim = int(value)

    property suhl:
        """Whether to use Suhl heuristic, a `bool`"""
        def __get__(self):
            return self._bfcp.suhl
        def __set__(self, value):
            self._bfcp.suhl = bool(value)

    property eps_tol:
        """Tolerance below which numbers are replaced by zero, a |Real| number"""
        def __get__(self):
            return self._bfcp.eps_tol
        def __set__(self, value):
            self._bfcp.eps_tol = float(value)

    property nfs_max:
        """Maximal number of additional row-like factors, an `int`

        (Used only when *type* contains `'Forrest-Tomlin'`.)

        """
        def __get__(self):
            return self._bfcp.nfs_max
        def __set__(self, value):
            self._bfcp.nfs_max = int(value)

    property nrs_max:
        """Maximal number of additional row and columns, an `int`

        (Used only when *type* contains `'Bartels-Golub'` or `'Givens'`.)

        """
        def __get__(self):
            return self._bfcp.nrs_max
        def __set__(self, value):
            self._bfcp.nrs_max = int(value)
