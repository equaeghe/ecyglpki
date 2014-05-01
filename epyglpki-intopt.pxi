# epyglpki-intopt.pxi: Cython/Python interface for GLPK integer optimization solver

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


cdef class IntOptSolver(_Solver):
    """An integer optimization solver

    .. doctest:: IntOptSolver

        >>> p = MILProgram()
        >>> s = p.intopt
        >>> isinstance(s, IntOptSolver)
        True

    """

    def solve(self, controls=IntOptControls(),
              solver='branchcut', obj_bound=None):
        """Solve the mixed-integer linear program

        :param controls: the control parameters (uses defaults if omitted)
        :type controls: `.IntOptControls`
        :param solver: which solver to use, chosen from

            * `'branchcut'`: a branch-and-cut solver
            * `'intfeas1'`: a SAT solver based integer feasibility solver;
              applicable only to problems

              * that are feasibility problems (but see `obj_bound` description)
              * with only binary variables
                (and integer variables with coinciding lower and upper bound)
              * with all-integer coefficients

              and furthermore efficient mainly for problems with constraints
              that are covering, packing, or partitioning inequalities, i.e.,
              sums of binary variables :math:`x` or their ‘negation’
              :math:`1-x`, smaller than, equal to, or larger than 1.

        :type solver: `str`
        :param obj_bound: if *solver* is `'intfeas1'`, a solution is
            considered feasible only if the corresponding objective value is
            not worse than this bound (not used if solver is `'branchcut'`)
        :type obj_bound: |Integral|
        :returns: solution status; see `.status` for details
        :rtype: `str`
        :raises ValueError: if *solver* is neither `'branchcut'` nor
            `'intfeas1'`
        :raises TypeError: if `obj_bound` is not |Integral|
        :raises ValueError: if incorrect bounds are given
        :raises ValueError: if no optimal LP relaxation basis has been provided
        :raises ValueError: if the LP relaxation is infeasible
        :raises ValueError: if the LP relaxation is unbounded
        :raises RuntimeError: in case of solver failure
        :raises StopIteration: if the relative mip gap tolerance has been
            reached
        :raises StopIteration: if the time limit has been exceeded
        :raises StopIteration: if the branch-and-cut callback terminated the
            solver
        :raises ValueError: if not all problem parameters are integer
            (only relevant if *solver* is `'intfeas1'`)
        :raises OverflowError: if an integer overflow occurred when
            transforming to CNF SAT format

        .. todo::

            Add doctest

        """
        cdef glpk.IntOptCP iocp = controls._iocp
        if solver is 'branchcut':
            retcode = glpk.intopt(self._problem, &iocp)
        elif solver is 'intfeas1':
            if obj_bound is None:
                retcode = glpk.intfeas1(self._problem, False, 0)
            elif isinstance(obj_bound, numbers.Integral):
                retcode = glpk.intfeas1(self._problem, True, obj_bound)
            else:
                raise TypeError("'obj_bound' must be an integer.")
        else:
            raise ValueError("The only available solvers are 'branchcut' " +
                             "and 'intfeas1'.")
        if retcode is 0:
            return self.status()
        else:
            raise ioretcode2error[retcode]

    property status:
        """The current solution status, a `str`

        The possible values are `'undefined'`, `'optimal'`, `'no feasible'`,
        and `'feasible'`.

        .. doctest:: IntOptSolver

            >>> s.status
            'undefined'

        """
        def __get__(self):
            return solstat2str[glpk.mip_status(self._problem)]

    def error(self):
        """Return absolute and relative solution errors

        :returns: a |Mapping| of {`'equalities'`, `'bounds'`} to |Mapping|
            of {`'abs'`, `'rel'`} to pairs consisting of an error (`float`)
            and a `.Variable` or `.Constraint` where it is attained

        The errors returned by this function quantify to what degree the
        current solution does not satisfy the Karush-Kuhn-Tucker conditions
        for equalities and bounds (possibly due to floating-point errors).

        .. todo::

            Add doctest

        """
        return self._error('primal', glpk.MIP)

    def print_solution(self, fname):
        """Write the solution to a file in a readable format

        :param fname: the name of the file to write to
        :type fname: `str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.print_mip, fname)

    def read_solution(self, fname):
        """Read the solution from a file

        :param fname: the name of the file to read from
            (written by `.write_solution`)
        :type fname: `str`
        :raises RuntimeError: if there is an error reading the file

        .. todo::

            Add doctest

        """
        self._read(glpk.read_mip, fname)

    def write_solution(self, fname):
        """Write the solution to a file

        :param fname: the name of the file to write to
        :type fname: `str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.write_mip, fname)
