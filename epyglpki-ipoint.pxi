# epyglpki-solvers.pxi: Cython/Python interface for GLPK interior point solver

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


cdef class IPointControls:
    """The interior point solver (`.IPointSolver`) control parameter object

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


cdef class IPointSolver(_Solver):
    """An interior point solver

    .. doctest:: IPointSolver

        >>> p = MILProgram()
        >>> s = p.ipoint
        >>> isinstance(s, IPointSolver)
        True

    """

    def solve(self, controls):
        """Solve the linear program

        :param controls: the control parameters
        :type controls: `.IPointControls`
        :returns: solution status; see `.status` for details
        :rtype: `str`
        :raises ValueError: if the problem has no rows/columns
        :raises ArithmeticError: if there occurs very slow convergence or
            divergence
        :raises StopIteration: if the iteration limit is exceeded
        :raises ArithmeticError: if numerical instability occurs on solving
            the Newtonian system

        .. todo::

            Add doctest

        """
        cdef glpk.IPointCP iptcp = controls._iptcp
        retcode = glpk.interior(self._problem, &iptcp)
        if retcode is 0:
            return self.status()
        else:
            raise iptretcode2error[retcode]

    property status:
        """The current solution status, a `str`

        The possible values are `'undefined'`, `'optimal'`, `'infeasible'`,
        and `'no feasible'`.

        .. doctest:: IPointSolver

            >>> s.status
            'undefined'

        """
        def __get__(self):
            return solstat2str[glpk.ipt_status(self._problem)]

    property objective:
        """The objective value for the current solution, a |Real| number

        .. doctest:: IPointSolver

            >>> s.objective
            0.0

        """
        def __get__(self):
            return glpk.ipt_obj_val(self._problem)

    def primal(self, varstraint):
        """Return primal value for the current solution

        :param varstraint: variable or constraint to return the primal value of
        :type varstraint: `.Varstraint`
        :returns: the value of *varstraint* for the current solution
        :rtype: `float` or `int`
        :raises TypeError: if varstraint is not `.Varstraint`

        .. todo::

            Add doctest

        """
        return self._value(varstraint, glpk.ipt_col_prim, glpk.ipt_row_prim)

    def primal_error(self):
        """Return absolute and relative primal solution errors

        :returns: a |Mapping| of {`'equalities'`, `'bounds'`} to |Mapping|
            of {`'abs'`, `'rel'`} to pairs consisting of an error (`float`)
            and a `.Varstraint` where it is attained

        The errors returned by this function quantify to what degree the
        current primal solution does not satisfy the Karush-Kuhn-Tucker
        conditions for equalities and bounds
        (possibly due to floating-point errors).

        .. todo::

            Add doctest

        """
        return self._error('primal', glpk.IPT)

    def dual(self, varstraint):
        """Return dual value for the current solution

        :param varstraint: variable or constraint to return the dual value of
        :type varstraint: `.Varstraint`
        :returns: the value of *varstraint* for the current solution
        :rtype: `float` or `int`
        :raises TypeError: if varstraint is not `.Varstraint`

        .. todo::

            Add doctest

        """
        return self._value(varstraint, glpk.ipt_col_dual, glpk.ipt_row_dual)

    def dual_error(self):
        """Return absolute and relative dual solution errors

        :returns: a |Mapping| of {`'equalities'`, `'bounds'`} to |Mapping|
            of {`'abs'`, `'rel'`} to pairs consisting of an error (`float`)
            and a `.Varstraint` where it is attained

        The errors returned by this function quantify to what degree the
        current dual solution does not satisfy the Karush-Kuhn-Tucker
        conditions for equalities and bounds
        (possibly due to floating-point errors).

        .. todo::

            Add doctest

        """
        return self._error('dual', glpk.IPT)

    def print_solution(self, fname):
        """Write the solution to a file in a readable format

        :param fname: the name of the file to write to
        :type fname: `str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.print_ipt, fname)

    def read_solution(self, fname):
        """Read the solution from a file

        :param fname: the name of the file to read from
            (written by `.write_solution`)
        :type fname: `str`
        :raises RuntimeError: if there is an error reading the file

        .. todo::

            Add doctest

        """
        self._read(glpk.read_ipt, fname)

    def write_solution(self, fname):
        """Write the solution to a file

        :param fname: the name of the file to write to
        :type fname: `str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.write_ipt, fname)
