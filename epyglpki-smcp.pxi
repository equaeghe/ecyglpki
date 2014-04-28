# epyglpki-solvers.pxi: Cython/Python interface for GLPK simplex solver controls

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


# simplex method
cdef str2meth = {
    'primal': glpk.PRIMAL,
    'dual': glpk.DUAL,
    'dual_fail_primal': glpk.DUALP
    }
cdef meth2str = {meth: string for string, meth in str2meth.items()}

# pricing strategy
cdef str2pricing = {
    'Dantzig': glpk.PT_STD,
    'steepest': glpk.PT_PSE
    }
cdef pricing2str = {pricing: string for string, pricing in str2pricing.items()}

# ratio test type
cdef str2rtest = {
    'standard': glpk.RT_STD,
    'Harris': glpk.RT_HAR
    }
cdef rtest2str = {r_test: string for string, r_test in str2rtest.items()}

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

# variable status
cdef str2varstat = {
    'basic': glpk.BS,
    'lower': glpk.NL,
    'upper': glpk.NU,
    'free': glpk.NF,
    'fixed': glpk.NS
    }
cdef varstat2str = {varstat: string for string, varstat in str2varstat.items()}


cdef class SimplexControls:
    """The simplex solver control parameter object

    .. doctest:: SimplexControls

        >>> r = SimplexControls()

    """

    cdef glpk.SimplexCP _smcp

    def __cinit__(self):
        glpk.init_smcp(&self._smcp)

    property msg_lev:
        """The message level, a `str`

        The possible values are

        * `'no'`: no output
        * `'warnerror'`: warnings and errors only
        * `'normal'`: normal output
        * `'full'`: normal output and informational messages

        .. doctest:: SimplexControls

            >>> r.msg_lev  # the GLPK default
            'full'
            >>> r.msg_lev = 'no'
            >>> r.msg_lev
            'no'

        """
        def __get__(self):
            return msglev2str[self._smcp.msg_lev]
        def __set__(self, value):
            self._smcp.msg_lev = str2msglev[value]

    property meth:
        """The simplex method, a `str`

        The possible values are

        * `'primal'`: two-phase primal simplex
        * `'dual'`: two-phase dual simplex
        * `'dual_fail_primal'`: two-phase dual simplex and, if it fails,
          switch to primal simplex

        .. doctest:: SimplexControls

            >>> r.meth  # the GLPK default
            'primal'
            >>> r.meth = 'dual_fail_primal'
            >>> r.meth
            'dual_fail_primal'

        """
        def __get__(self):
            return meth2str[self._smcp.meth]
        def __set__(self, value):
            self._smcp.meth = str2meth[value]

    property pricing:
        """The pricing technique, a `str`

        The possible values are

        * `'Dantzig'`: standard ‘textbook’
        * `'steepest'`: projected steepest edge

        .. doctest:: SimplexControls

            >>> r.pricing  # the GLPK default
            'steepest'
            >>> r.pricing = 'Dantzig'
            >>> r.pricing
            'Dantzig'

        """
        def __get__(self):
            return pricing2str[self._smcp.pricing]
        def __set__(self, value):
            self._smcp.pricing = str2pricing[value]

    property r_test:
        """The ratio test technique, a `str`

        The possible values are

        * `'standard'`: standard ‘textbook’
        * `'Harris'`: Harris’s two-pass ratio test

        .. doctest:: SimplexControls

            >>> r.r_test  # the GLPK default
            'Harris'
            >>> r.r_test = 'standard'
            >>> r.r_test
            'standard'

        """
        def __get__(self):
            return rtest2str[self._smcp.r_test]
        def __set__(self, value):
            self._smcp.r_test = str2rtest[value]

    property tol_bnd:
        """Tolerance to check if the solution is primal feasible, a |Real| number"""
        def __get__(self):
            return self._smcp.tol_bnd
        def __set__(self, value):
            self._smcp.tol_bnd = float(value)

    property tol_dj:
        """Tolerance to check if the solution is dual feasible, a |Real| number"""
        def __get__(self):
            return self._smcp.tol_dj
        def __set__(self, value):
            self._smcp.tol_dj = float(value)

    property tol_piv:
        """Tolerance to choose eligble pivotal elements, a |Real| number"""
        def __get__(self):
            return self._smcp.tol_piv
        def __set__(self, value):
            self._smcp.tol_piv = float(value)

    property obj_ll:
        """Lower limit of the objective function, a |Real| number

        (Used only if *meth* is `'dual'`.)

        """
        def __get__(self):
            return self._smcp.obj_ll
        def __set__(self, value):
            self._smcp.obj_ll = float(value)

    property obj_ul:
        """Upper limit of the objective function, a |Real| number

        (Used only if *meth* is `'dual'`.)

        """
        def __get__(self):
            return self._smcp.obj_ul
        def __set__(self, value):
            self._smcp.obj_ul = float(value)

    property it_lim:
        """Iteration limit, an `int`"""
        def __get__(self):
            return self._smcp.it_lim
        def __set__(self, value):
            self._smcp.it_lim = int(value)

    property tm_lim:
        """Time limit [ms], an `int`"""
        def __get__(self):
            return self._smcp.tm_lim
        def __set__(self, value):
            self._smcp.tm_lim = int(value)

    property out_frq:
        """Output frequency [iterations] of informational messages, an `int`"""
        def __get__(self):
            return self._smcp.out_frq
        def __set__(self, value):
            self._smcp.out_frq = int(value)

    property out_dly:
        """Output delay [ms] of solution process information, an `int`"""
        def __get__(self):
            return self._smcp.out_dly
        def __set__(self, value):
            self._smcp.out_dly = int(value)

    property presolve:
        """Whether to use the LP presolver, a `bool`"""
        def __get__(self):
            return self._smcp.presolve
        def __set__(self, value):
            self._smcp.presolve = bool(value)
