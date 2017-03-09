# ecyglpki-smcp.pxi: Cython interface for GLPK simplex solver controls

###############################################################################
#
#  This code is part of ecyglpki (a Cython GLPK interface).
#
#  Copyright ⓒ 2017 Erik Quaeghebeur. All rights reserved.
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
    'Harris': glpk.RT_HAR,
    'flip-flop': glpk.RT_FLIP
    }
cdef rtest2str = {r_test: string for string, r_test in str2rtest.items()}


cdef class SimplexControls:
    """The simplex solver control parameter object

    .. doctest:: SimplexControls

        >>> r = SimplexControls()

    """

    cdef glpk.SmCp _smcp

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
        """Tolerance to check if the solution is primal feasible, a |Real| number

        .. doctest:: SimplexControls

            >>> r.tol_bnd  # the GLPK default
            1e-07
            >>> r.tol_bnd = 0.2
            >>> r.tol_bnd
            0.2

        """
        def __get__(self):
            return self._smcp.tol_bnd
        def __set__(self, value):
            self._smcp.tol_bnd = float(value)

    property tol_dj:
        """Tolerance to check if the solution is dual feasible, a |Real| number

        .. doctest:: SimplexControls

            >>> r.tol_dj  # the GLPK default
            1e-07
            >>> r.tol_dj = 0.01
            >>> r.tol_dj
            0.01

        """
        def __get__(self):
            return self._smcp.tol_dj
        def __set__(self, value):
            self._smcp.tol_dj = float(value)

    property tol_piv:
        """Tolerance to choose eligble pivotal elements, a |Real| number

        .. doctest:: SimplexControls

            >>> r.tol_piv  # the GLPK default
            1e-09
            >>> r.tol_piv = 1e-3
            >>> r.tol_piv
            0.001

        """
        def __get__(self):
            return self._smcp.tol_piv
        def __set__(self, value):
            self._smcp.tol_piv = float(value)

    property obj_ll:
        """Lower limit of the objective function, a |Real| number

        (Used only if *meth* is `'dual'`.)

        .. doctest:: SimplexControls

            >>> r.obj_ll  # the GLPK default
            -1.7976931348623157e+308
            >>> r.obj_ll = -1234.0
            >>> r.obj_ll
            -1234.0

        """
        def __get__(self):
            return self._smcp.obj_ll
        def __set__(self, value):
            self._smcp.obj_ll = float(value)

    property obj_ul:
        """Upper limit of the objective function, a |Real| number

        (Used only if *meth* is `'dual'`.)

        .. doctest:: SimplexControls

            >>> r.obj_ul  # the GLPK default
            1.7976931348623157e+308
            >>> r.obj_ul = 123.4
            >>> r.obj_ul
            123.4

        """
        def __get__(self):
            return self._smcp.obj_ul
        def __set__(self, value):
            self._smcp.obj_ul = float(value)

    property it_lim:
        """Iteration limit, an `int`

        .. doctest:: SimplexControls

            >>> r.it_lim  # the GLPK default
            2147483647
            >>> r.it_lim = 10
            >>> r.it_lim
            10

        """
        def __get__(self):
            return self._smcp.it_lim
        def __set__(self, value):
            self._smcp.it_lim = int(value)

    property tm_lim:
        """Time limit [ms], an `int`

        .. doctest:: SimplexControls

            >>> r.tm_lim  # the GLPK default
            2147483647
            >>> r.tm_lim = 1e7
            >>> r.tm_lim
            10000000

        """
        def __get__(self):
            return self._smcp.tm_lim
        def __set__(self, value):
            self._smcp.tm_lim = int(value)

    property out_frq:
        """Output frequency [iterations] of informational messages, an `int`

        .. doctest:: SimplexControls

            >>> r.out_frq  # the GLPK default
            500
            >>> r.out_frq = 50
            >>> r.out_frq
            50

        """
        def __get__(self):
            return self._smcp.out_frq
        def __set__(self, value):
            self._smcp.out_frq = int(value)

    property out_dly:
        """Output delay [ms] of solution process information, an `int`

        .. doctest:: SimplexControls

            >>> r.out_dly  # the GLPK default
            0
            >>> r.out_dly = 25
            >>> r.out_dly
            25

        """
        def __get__(self):
            return self._smcp.out_dly
        def __set__(self, value):
            self._smcp.out_dly = int(value)

    property presolve:
        """Whether to use the LP presolver, a `bool`

        .. doctest:: SimplexControls

            >>> r.presolve  # the GLPK default
            False
            >>> r.presolve = True
            >>> r.presolve
            True

        """
        def __get__(self):
            return self._smcp.presolve
        def __set__(self, value):
            self._smcp.presolve = bool(value)
