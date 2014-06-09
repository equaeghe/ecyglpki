# epyglpki-iptcp.pxi: Cython interface for GLPK interior point solver controls

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


# ordering algorithm
cdef str2ordalg = {
    'orig': glpk.ORD_NONE,
    'qmd': glpk.ORD_QMD,
    'amd': glpk.ORD_AMD,
    'symamd': glpk.ORD_SYMAMD
    }
cdef ordalg2str = {ord_alg: string for string, ord_alg in str2ordalg.items()}


cdef class IPointControls:
    """The interior point solver control parameter object

    .. doctest:: IPointControls

        >>> r = IPointControls()

    """

    cdef glpk.IPointCP _iptcp

    def __cinit__(self):
        glpk.init_iptcp(&self._iptcp)

    property msg_lev:
        """The message level, a `str`

        The possible values are

        * `'no'`: no output
        * `'warnerror'`: warnings and errors only
        * `'normal'`: normal output
        * `'full'`: normal output and informational messages

        .. doctest:: IPointControls

            >>> r.msg_lev  # the GLPK default
            'full'
            >>> r.msg_lev = 'no'
            >>> r.msg_lev
            'no'

        """
        def __get__(self):
            return msglev2str[self._iptcp.msg_lev]
        def __set__(self, value):
            self._iptcp.msg_lev = str2msglev[value]

    property ord_alg:
        """The ordering algorithm used prior to Cholesky factorization, a `str`

        The possible values are

        * `'orig'`: normal (original)
        * `'qmd'`: quotient minimum degree
        * `'amd'`: approximate minimum degree
        * `'symamd'`: approximate minimum degree for symmetric matrices

        .. doctest:: IPointControls

            >>> r.ord_alg  # the GLPK default
            'amd'
            >>> r.ord_alg = 'qmd'
            >>> r.ord_alg
            'qmd'

        """
        def __get__(self):
            return ordalg2str[self._iptcp.ord_alg]
        def __set__(self, value):
            self._iptcp.ord_alg = str2ordalg[value]
