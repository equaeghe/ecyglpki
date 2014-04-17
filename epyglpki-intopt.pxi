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


cdef class IntOptControls:
    """The integer optimization solver (`.IntOptSolver`) control parameter object

    .. doctest:: IntOptControls

        >>> r = IntOptControls()

    """

    cdef glpk.IntOptCP _iocp

    def __cinit__(self):
        glpk.init_iocp(&self._iocp)

    property msg_lev:
        """The message level, a `str`

        The possible values are

        * `'no'`: no output
        * `'warnerror'`: warnings and errors only
        * `'normal'`: normal output
        * `'full'`: normal output and informational messages

        .. doctest:: IntOptControls

            >>> r.msg_lev  # the GLPK default
            'full'
            >>> r.msg_lev = 'no'
            >>> r.msg_lev
            'no'

        """
        def __get__(self):
            return msglev2str[self._iocp.msg_lev]
        def __set__(self, value):
            self._iocp.msg_lev = str2msglev[value]

    property out_frq:
        """Output frequency [ms] of informational messages, an `int`"""
        def __get__(self):
            return self._iocp.out_frq
        def __set__(self, value):
            self._iocp.out_frq = int(value)

    property out_dly:
        """Output delay [ms] of current LP relaxation solution, an `int`"""
        def __get__(self):
            return self._iocp.out_dly
        def __set__(self, value):
            self._iocp.out_dly = int(value)

    property tm_lim:
        """Time limit [ms], an `int`"""
        def __get__(self):
            return self._iocp.tm_lim
        def __set__(self, value):
            self._iocp.tm_lim = int(value)

    property br_tech:
        """The branching technique, a `str`

        The possible values are

        * `'first_fracvar'`: first fractional variable
        * `'last_fracvar'`: last fractional variable
        * `'most_fracvar'`: most fractional variable
        * `'Driebeek-Tomlin'`: heuristic by Driebeek_ & Tomlin
        * `'hybrid_peudocost'`: hybrid pseudocost heuristic

        .. _Driebeek: http://dx.doi.org/10.1287/ijoc.4.3.267

        """
        def __get__(self):
            return brtech2str[self._iocp.br_tech]
        def __set__(self, value):
            self._iocp.br_tech = str2brtech[value]

    property bt_tech:
        """The backtracking technique, a `str`

        The possible values are

        * `'depth'`: depth first search
        * `'breadth'`: breadth first search
        * `'bound'`: best local bound
        * `'projection'`: best projection heuristic

        """
        def __get__(self):
            return bttech2str[self._iocp.bt_tech]
        def __set__(self, value):
            self._iocp.bt_tech = str2bttech[value]

    property pp_tech:
        """The preprocessing technique, a `str`

        The possible values are

        * `'none'`: disable preprocessing
        * `'root'`: preprocessing only on the root level
        * `'all'`: preprocessing on all levels

        """
        def __get__(self):
            return pptech2str[self._iocp.pp_tech]
        def __set__(self, value):
            self._iocp.pp_tech = str2pptech[value]

    property fp_heur:
        """Whether to apply the `feasibility pump`_ heuristic, a `bool`

        .. _feasibility pump: http://dx.doi.org/10.1007/s10107-004-0570-3

        """
        def __get__(self):
            return self._iocp.fp_heur
        def __set__(self, value):
            self._iocp.fp_heur = bool(value)

    property ps_heur:
        """Whether to apply the `proximity search`_ heuristic, a `bool`

        .. _proximity search: http://www.dei.unipd.it/~fisch/papers/proximity_search.pdf

        """
        def __get__(self):
            return self._iocp.ps_heur
        def __set__(self, value):
            self._iocp.ps_heur = bool(value)

    property ps_tm_lim:
        """Time limit [ms] for the proximity earch heuristic, an `int`"""
        def __get__(self):
            return self._iocp.ps_tm_lim
        def __set__(self, value):
            self._iocp.ps_tm_lim = int(value)

    property gmi_cuts:
        """Whether to generate Gomory’s mixed integer cuts, a `bool`"""
        def __get__(self):
            return self._iocp.gmi_cuts
        def __set__(self, value):
            self._iocp.gmi_cuts = bool(value)

    property mir_cuts:
        """Whether to generate mixed integer rounding cuts, a `bool`"""
        def __get__(self):
            return self._iocp.mir_cuts
        def __set__(self, value):
            self._iocp.mir_cuts = bool(value)

    property cov_cuts:
        """Whether to generate mixed cover cuts, a `bool`"""
        def __get__(self):
            return self._iocp.cov_cuts
        def __set__(self, value):
            self._iocp.cov_cuts = bool(value)

    property clq_cuts:
        """Whether to generate generate clique cuts, a `bool`"""
        def __get__(self):
            return self._iocp.clq_cuts
        def __set__(self, value):
            self._iocp.clq_cuts = bool(value)

    property tol_int:
        """Abs. tolerance for LP solution integer feasibility, a |Real| number

        This is the absolute tolerance used to check if the optimal solution to the current LP relaxation is integer feasible.

        """
        def __get__(self):
            return self._iocp.tol_int
        def __set__(self, value):
            self._iocp.tol_int = float(value)

    property tol_obj:
        """Rel. tolerance of LP objective optimality, a |Real| number

        This is the relative tolerance used to check if the objective value in
        the optimal solution to the current LP relaxation is not better than in
        the best known integer feasible solution.

        """
        def __get__(self):
            return self._iocp.tol_obj
        def __set__(self, value):
            self._iocp.tol_obj = float(value)

    property mip_gap:
        """The relative MIP-gap tolerance, a |Real| number

        The search stops once the relative MIP-gap falls below this value.

        """
        def __get__(self):
            return self._iocp.mip_gap
        def __set__(self, value):
            self._iocp.mip_gap = float(value)

    property presolve:
        """Whether to use the MIP presolver, a `bool`

        Using the MIP presolver may simplify the problem

        """
        def __get__(self):
            return self._iocp.presolve
        def __set__(self, value):
            self._iocp.presolve = bool(value)

    property binarize:
        """Whether to binarize integer variables, a `bool`

        This option is only used if *presolve* is `True`.

        """
        def __get__(self):
            return self._iocp.binarize
        def __set__(self, value):
            self._iocp.binarize = bool(value)


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

    property objective:
        """The objective value for the current solution, a |Real| number

        .. doctest:: IntOptSolver

            >>> s.objective
            0.0

        """
        def __get__(self):
            return glpk.mip_obj_val(self._problem)

    def value(self, varstraint):
        """Return the variable or constraint value for the current solution

        :param varstraint: variable or constraint to return the value of
        :type varstraint: `.Variable` or `.Constraint`
        :returns: the value of *varstraint* for the current solution
        :rtype: `float` or `int`
        :raises TypeError: if varstraint is not `.Variable` or `.Constraint`
        :raises ValueError: if a variable with `'integer'` or `'binary'` kind
            has a non-integer value

        .. todo::

            Add doctest

        """
        val = self._value(varstraint, glpk.mip_col_val, glpk.mip_row_val)
        if isinstance(varstraint, Variable):
            if varstraint.kind in {'binary', 'integer'}:
                if val.isinteger():
                    val = int(val)
                else:
                    raise ValueError("Variable with binary or integer kind " +
                                     "has non-integer value " + str(val))
        return val

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
