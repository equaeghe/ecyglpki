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

    cdef glpk.IntOptCP _iocp

    def __cinit__(self, program):
        glpk.init_iocp(&self._iocp)

    def controls(self, defaults=False, **controls):
        """Change or retrieve the solver's control parameters

        :param defaults: whether to set the parameters back to their default
            values or not
        :type defaults: `bool`
        :param controls: zero or more named parameters to change from the
            following list:

            * **msg_lev** (`str`) – the message level, with possible values

              * `'no'`: no output
              * `'warnerror'`: warnings and errors only
              * `'normal'`: normal output
              * `'full'`: normal output and informational messages

            * **out_frq** (|Integral|) – output frequency [ms] of informational
              messages
            * **out_dly** (|Integral|) – output delay [ms] of current LP
              relaxation solution
            * **tm_lim** (|Integral|) – time limit [ms]
            * **br_tech** (`str`) – the branching technique, with possible
              values

              * `'first_fracvar'`: first fractional variable
              * `'last_fracvar'`: last fractional variable
              * `'most_fracvar'`: most fractional variable
              * `'Driebeek-Tomlin'`: heuristic by Driebeek_ & Tomlin
              * `'hybrid_peudocost'`: hybrid pseudocost heuristic

            * **bt_tech** (`str`) – the backtracking technique, with possible
              values

              * `'depth'`: depth first search
              * `'breadth'`: breadth first search
              * `'bound'`: best local bound
              * `'projection'`: best projection heuristic

            * **pp_tech** (`str`) – the preprocessing technique, with possible
              values

              * `'none'`: disable preprocessing
              * `'root'`: preprocessing only on the root level
              * `'all'`: preprocessing on all levels

            * **fp_heur** (`bool`) – apply `feasibility pump`_ heuristic
            * **ps_heur** (`bool`) – apply `proximity search`_ heuristic
            * **ps_tm_lim** (|Integral|) –  time limit [ms] for the proximity
              search heuristic
            * **gmi_cuts** (`bool`) – generate Gomory’s mixed integer cuts
            * **mir_cuts** (`bool`) – generate mixed integer rounding cuts
            * **cov_cuts** (`bool`) – generate mixed cover cuts
            * **clq_cuts** (`bool`) – generate clique cuts
            * **tol_int** (|Real|) – absolute tolerance used to check if the
              optimal solution to the current LP relaxation is integer feasible
            * **tol_obj** (|Real|) – relative tolerance used to check if the
              objective value in the optimal solution to the current LP
              relaxation is not better than in the best known integer feasible
              solution.
            * **mip_gap** (|Real|) – relative MIP-gap tolerance
              (search stops once the relative MIP-gap falls below this value)
            * **presolve** (`bool`) – use MIP presolver, may simplify the
              problem
            * **binarize** (`bool`) – binarize integer variables
              (only used if *presolve* is `True`)

        :raises ValueError: if a non-existing control name is given

        .. todo::

            Add doctest

        .. _Driebeek: http://dx.doi.org/10.1287/ijoc.4.3.267
        .. _feasibility pump: http://dx.doi.org/10.1007/s10107-004-0570-3
        .. _proximity search: http://www.dei.unipd.it/~fisch/papers/proximity_search.pdf

        """
        if defaults:
            glpk.init_iocp(&self._iocp)
        for control, val in controls.items():
            if control is 'msg_lev':
                self._iocp.msg_lev = str2msglev[val]
            elif control is 'br_tech':
                self._iocp.br_tech = str2brtech[val]
            elif control is 'bt_tech':
                self._iocp.bt_tech = str2bttech[val]
            elif control is 'pp_tech':
                self._iocp.pp_tech = str2pptech[val]
            elif control in {'tol_int', 'tol_obj', 'mip_gap'}:
                if not isinstance(val, numbers.Real):
                    raise TypeError("'" + control + "' value must be real.")
                elif control is 'tol_int':
                    self._iocp.tol_int = val
                elif control is 'tol_obj':
                    self._iocp.tol_obj = val
                elif control is 'mip_gap':
                    self._iocp.mip_gap = val
            elif control in {'tm_lim', 'out_frq', 'out_dly'}:
                if not isinstance(val, numbers.Integral):
                    raise TypeError("'" + control + "' value must be integer.")
                elif control is 'tm_lim':
                    self._iocp.tm_lim = val
                elif control is 'out_frq':
                    self._iocp.out_frq = val
                elif control is 'out_dly':
                    self._iocp.out_dly = val
            elif control in {'mir_cuts', 'gmi_cuts', 'cov_cuts', 'clq_cuts',
                             'presolve', 'binarize', 'fp_heur'}:
                if not isinstance(val, bool):
                    raise TypeError("'" + control + "' value must be Boolean.")
                elif control is 'mir_cuts':
                    self._iocp.mir_cuts = val
                elif control is 'gmi_cuts':
                    self._iocp.gmi_cuts = val
                elif control is 'cov_cuts':
                    self._iocp.cov_cuts = val
                elif control is 'clq_cuts':
                    self._iocp.clq_cuts = val
                elif control is 'presolve':
                    self._iocp.presolve = val
                elif control is 'binarize':
                    self._iocp.binarize = val
                elif control is 'fp_heur':
                    self._iocp.fp_heur = val
            else:
                raise ValueError("Non-existing control: " + repr(control))
        controls = {}
        controls = self._iocp
        controls['msg_lev'] = msglev2str[controls['msg_lev']]
        controls['br_tech'] = brtech2str[controls['br_tech']]
        controls['bt_tech'] = bttech2str[controls['bt_tech']]
        controls['pp_tech'] = pptech2str[controls['pp_tech']]
        return controls

    def solve(self, solver='branchcut', obj_bound=None):
        """Solve the mixed-integer linear program

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
        if solver is 'branchcut':
            retcode = glpk.intopt(self._problem, &self._iocp)
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

    def status(self):
        """Return the current solution status

        :returns: the current solution status, either `'undefined'`,
            `'optimal'`, `'no feasible'`, or `'feasible'`
        :rtype: `str`

        .. todo::

            Add doctest

        """
        return solstat2str[glpk.mip_status(self._problem)]

    def objective(self):
        """Return the objective value for the current solution

        :returns: the objective value for the current solution
        :rtype: `float`

        .. todo::

            Add doctest

        """
        return glpk.mip_obj_val(self._problem)

    def value(self, varstraint):
        """Return the variable or constraint value for the current solution

        :param varstraint: variable or constraint to return the value of
        :type varstraint: `.Varstraint`
        :returns: the value of *varstraint* for the current solution
        :rtype: `float` or `int`
        :raises TypeError: if varstraint is not `.Varstraint`
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
            and a `.Varstraint` where it is attained

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
