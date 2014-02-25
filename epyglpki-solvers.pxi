# epyglpki-solvers.pxi: Cython/Python interface for GLPK solvers

###############################################################################
#
#  This code is part of epyglpki (a Cython/Python GLPK interface).
#
#  Copyright (C) 2014 erik Quaeghebeur. All rights reserved.
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


cdef class _Solver(_ProgramComponent):

    cdef _faccess(self,
                  int (*faccess_function)(glpk.ProbObj*, const char*),
                  fname, error_msg):
        cdef char* chars
        fname = name2chars(fname)
        chars = fname
        retcode = faccess_function(self._problem, chars)
        if retcode is not 0:
            raise RuntimeError(error_msg + " '" + fname + "'.")

    cdef _read(self,
               int (*faccess_function)(glpk.ProbObj*, const char*), fname):
        self._faccess(faccess_function, fname, "Error reading file")

    cdef _write(self,
                int (*faccess_function)(glpk.ProbObj*, const char*), fname):
        self._faccess(faccess_function, fname, "Error writing file")


cdef class _LPSolver(_Solver):

    cdef _solution(self, varstraints,
                   double (*primal_func)(glpk.ProbObj*, int),
                   double (*dual_func)(glpk.ProbObj*, int), dual):
        if not isinstance(dual, bool):
            raise TypeError("Parameter dual must be True or False")
        solution = {}
        if not dual:
            for varstraint in varstraints:
                val = primal_func(self._problem,
                                  self._program._ind(varstraint))
                if val != 0.0:
                    solution[varstraint] = val
        else:
            for varstraint in varstraints:
                val = dual_func(self._problem, self._program._ind(varstraint))
                if val != 0.0:
                    solution[varstraint] = val
        return solution


cdef class SimplexSolver(_LPSolver):
    """The problem's simplex solver"""

    cdef glpk.SimplexCP _smcp
    cdef glpk.BasFacCP _bfcp

    def __cinit__(self, program):
        glpk.init_smcp(&self._smcp)
        glpk.get_bfcp(self._problem, &self._bfcp)

    def controls(self, defaults=False, **controls):
        """Change or retrieve the solver's control parameters

        :param defaults: whether to set the parameters back to their default
            values or not
        :type defaults: :class:`bool`
        :param controls: zero or more named parameters to change from the
            following list:

            * :data:`msg_lev` (:class:`str`) – the message level,
              with possible values

              * :data:`'no'`: no output
              * :data:`'warnerror'`: warnings and errors only
              * :data:`'normal'`: normal output
              * :data:`'full'`: normal output and informational messages

            * :data:`meth` (:class:`str`) – simplex method,
              with possible values

              * :data:`'primal'`: two-phase primal simplex
              * :data:`'dual'`: two-phase dual simplex
              * :data:`'dual_fail_primal'`: two-phase dual simplex and, if it
                fails, switch to primal simplex

            * :data:`pricing` (:class:`str`) – pricing technique,
              with possible values

              * :data:`'Dantzig'`: standard ‘textbook’
              * :data:`'steepest'`: projected steepest edge

            * :data:`r_test` (:class:`str`) – ratio test technique,
              with possible values

              * :data:`'standard'`: standard ‘textbook’
              * :data:`'Harris'`: Harris’s two-pass ratio test

            * :data:`tol_bnd` (:class:`~numbers.Real`) – tolerance used to
              check if the basic solution is primal feasible
            * :data:`tol_dj` (:class:`~numbers.Real`) – tolerance used to check
              if the basic solution is dual feasible
            * :data:`tol_piv` (:class:`~numbers.Real`) – tolerance used to
              choose eligble pivotal elements of the simplex table
            * :data:`obj_ll` (:class:`~numbers.Real`) – lower limit of the
              objective function (only if :data:`meth` is :data:`'dual'`)
            * :data:`obj_ul` (:class:`~numbers.Real`) – upper limit of the
              objective function (only if :data:`meth` is :data:`'dual'`)
            * :data:`it_lim` (:class:`~numbers.Integral`) – iteration limit
            * :data:`tm_lim` (:class:`~numbers.Integral`) – time limit [ms]
            * :data:`out_frq` (:class:`~numbers.Integral`) – output frequency
              [iterations] of informational messages
            * :data:`out_dly` (:class:`~numbers.Integral`) – output delay
              [ms] of solution process information
            * :data:`presolve` (:class:`bool`) – use LP presolver

            or, for basis factorization, from the following list:

            * :data:`type` (:class:`str`) – basis factorization type,
              with possible values

              * :data:`'Forrest-Tomlin'`: LU + Forrest–Tomlin update
              * :data:`'Bartels-Golub'`: LU + Schur complement +
                Bartels–Golub update
              * :data:`'Givens'`: LU + Schur complement +
                Givens rotation update

            * :data:`lu_size` (:class:`~numbers.Integral`) – 
            * :data:`piv_tol` (:class:`~numbers.Real`) – 
            * :data:`piv_lim` (:class:`~numbers.Integral`) – 
            * :data:`suhl` (:class:`bool`) – 
            * :data:`eps_tol` (:class:`~numbers.Real`) – 
            * :data:`max_gro` (:class:`~numbers.Real`) – 
            * :data:`nfs_max` (:class:`~numbers.Integral`) – 
            * :data:`upd_tol` (:class:`~numbers.Real`) – 
            * :data:`nrs_max` (:class:`~numbers.Integral`) – 
            * :data:`rs_size` (:class:`~numbers.Integral`) – 

        :raises ValueError: if a non-existing control name is given

        .. todo::

            Add doctest

        """
        if defaults:
            glpk.init_smcp(&self._smcp)
            glpk.set_bfcp(self._problem, NULL)
            glpk.get_bfcp(self._problem, &self._bfcp)
        for control, val in controls.items():
            # smcp enumerated parameters
            if control is 'msg_lev':
                self._smcp.msg_lev = str2msglev[val]
            elif control is 'meth':
                self._smcp.meth = str2meth[val]
            elif control is 'pricing':
                self._smcp.meth = str2pricing[val]
            elif control is 'r_test':
                self._smcp.r_test = str2rtest[val]
            # bfcp enumerated parameters
            elif control is 'type':
                self._bfcp.type = str2bftype[val]
            # double parameters
            elif control in {
                # smcp
                'tol_bnd', 'tol_dj', 'tol_piv', 'obj_ll', 'obj_ul',
                # bfcp
                'piv_tol', 'eps_tol', 'max_gro', 'upd_tol'
                }:
                if not isinstance(val, numbers.Real):
                    raise TypeError("'" + control + "' value must be real.")
                # smcp
                elif control is 'tol_bnd':
                    self._smcp.tol_bnd = val
                elif control is 'tol_dj':
                    self._smcp.tol_dj = val
                elif control is 'tol_piv':
                    self._smcp.tol_piv = val
                elif control is 'obj_ll':
                    self._smcp.obj_ll = val
                elif control is 'obj_ul':
                    self._smcp.obj_ul = val
                # bfcp
                elif control is 'piv_tol':
                    self._bfcp.piv_tol = val
                elif control is 'eps_tol':
                    self._bfcp.eps_tol = val
                elif control is 'max_gro':
                    self._bfcp.max_gro = val
                elif control is 'upd_tol':
                    self._bfcp.upd_tol = val
            # int parameters
            elif control in {
                # smcp
                'it_lim', 'tm_lim', 'out_frq', 'out_dly',
                # bfcp
                'lu_size', 'piv_lim', 'nfs_max', 'nrs_max', 'rs_size'
                }:
                if not isinstance(val, numbers.Integral):
                    raise TypeError("'" + control + "' value must be integer.")
                # smcp
                elif control is 'it_lim':
                    self._smcp.it_lim = val
                elif control is 'tm_lim':
                    self._smcp.tm_lim = val
                elif control is 'out_frq':
                    self._smcp.out_frq = val
                elif control is 'out_dly':
                    self._smcp.out_dly = val
                # bfcp
                elif control is 'lu_size':
                    self._bfcp.lu_size = val
                elif control is 'piv_lim':
                    self._bfcp.piv_lim = val
                elif control is 'nfs_max':
                    self._bfcp.nfs_max = val
                elif control is 'nrs_max':
                    self._bfcp.nrs_max = val
                elif control is 'rs_size':
                    self._bfcp.rs_size = val
            # bint parameters
            elif control in {
                # smcp
                'presolve',
                # bfcp
                'suhl'
                }:
                if not isinstance(val, bool):
                    raise TypeError("'" + control + "' value must be Boolean.")
                # smcp
                elif control is 'presolve':
                    self._smcp.presolve = val
                # bfcp
                elif control is 'suhl':
                    self._bfcp.suhl = val
            else:
                raise ValueError("Non-existing control: " + repr(control))
        glpk.set_bfcp(self._problem, &self._bfcp)
        scontrols = {}
        scontrols = self._smcp
        scontrols['msg_lev'] = msglev2str[scontrols['msg_lev']]
        scontrols['meth'] = meth2str[scontrols['meth']]
        scontrols['pricing'] = pricing2str[scontrols['pricing']]
        scontrols['r_test'] = rtest2str[scontrols['r_test']]
        fcontrols = {}
        fcontrols = self._bfcp
        fcontrols['type'] = bftype2str[fcontrols['type']]
        controls = {}
        controls.update(scontrols)
        controls.update(fcontrols)
        return controls

    def solve(self, exact=False):
        """Solve the linear program

        :param exact: whether to use exact arithmetic or not
            (only if the :class:`meth` control parameter is :data:`'primal'`)
        :type exact: :class:`bool`
        :returns: solution status; see :meth:`SimplexSolver.status` for
            details, or :data:`"obj_ll reached"` or :data:`"obj_ul reached"` in
            case that happens
        :rtype: :class:`str`
        :raises ValueError: if `exact` is :data:`True` but the :data:`meth`
            control parameter is not :data:`'primal'`
        :raises ValueError: if finite values are set for :data:`obj_ll` or
            :data:`obj_ll` while the :data:`meth` control parameter is not
            :data:`'dual'`
        :raises ValueError: if the basis is invalid
        :raises ValueError: if the basis matrix is singular
        :raises ValueError: if the basis matrix is ill-conditioned
        :raises ValueError: if incorrect bounds are given
        :raises RuntimeError: in case of solver failure
        :raises StopIteration: if the iteration limit is exceeded
        :raises StopIteration: if the time limit is exceeded
        :raises StopIteration: if the presolver detects the problem has no
            primal feasible solution
        :raises StopIteration: if the presolver detects the problem has no dual
            feasible solution

        .. todo::

            Add doctest

        """
        if exact:
            if meth2str[self._smcp.meth] is not 'primal':
                raise ValueError("Only primal simplex with exact arithmetic.")
            retcode = glpk.simplex_exact(self._problem, &self._smcp)
        else:
            if ((meth2str[self._smcp.meth] is not 'dual') and
                ((self._smcp.obj_ll > -DBL_MAX) or
                 (self._smcp.obj_ul < +DBL_MAX))):
                 raise ValueError("Objective function limits only with " +
                                  "dual simplex.")
            retcode = glpk.simplex(self._problem, &self._smcp)
        if retcode is 0:
            return self.status()
        elif retcode in {glpk.EOBJLL, glpk.EOBJUL}:
            return smretcode2str[retcode]
        else:
            raise smretcode2error[retcode]

    def status(self, detailed=False):
        """Return the current solution status

        :param detailed: whether to give a detailed solution status
        :type detailed: :class:`bool`
        :returns: the current solution status

            * in case `detailed` is :data:`False`, either :data:`'undefined'`,
              :data:`'optimal`, :data:`'infeasible'`, :data:`'no feasible'`,
              :data:`'feasible'`, or :data:`'unbounded'`
            * in case `detailed` is :data:`True`, a pair of statuses is given,
              one for the primal solution and one for the dual solution, either
              :data:`'undefined'`, :data:`'infeasible'`, :data:`'no feasible'`,
              or :data:`'feasible'`

        :rtype: :class:`str` or length-2 :class:`tuple` of :class:`str`

        .. todo::

            Add doctest

        """
        status = solstat2str[glpk.sm_status(self._problem)]
        if detailed:
            return (status, (solstat2str[glpk.sm_prim_stat(self._problem)],
                             solstat2str[glpk.sm_dual_stat(self._problem)]))
        else:
            return status

    def objective(self):
        """Return the objective value for the current solution

        :returns: the objective value for the current solution
        :rtype: :class:`float`

        .. todo::

            Add doctest

        """
        return glpk.sm_obj_val(self._problem)

    def variables(self, dual=False):
        """Returns the values of the variables for the current solution

        :param dual: whether to return dual or primal values
        :type dual: :class:`bool`
        :returns: the nonzero values of the variables for the current
            solution
        :rtype: :class:`dict` from :class:`Variable` to :class:`float`
        :raises TypeError: if `dual` is not :class:`bool`

        .. todo::

            Add doctest

        """
        return self._solution(self._program._variables,
                              glpk.sm_col_prim, glpk.sm_col_dual, dual)

    def constraints(self, dual=False):
        """Returns the values of the constraints for the current solution

        :param dual: whether to return dual or primal values
        :type dual: :class:`bool`
        :returns: the nonzero values of the constraints for the current
            solution
        :rtype: :class:`dict` from :class:`Constraint` to :class:`float`
        :raises TypeError: if `dual` is not :class:`bool`

        .. todo::

            Add doctest

        """
        return self._solution(self._program._constraints,
                              glpk.sm_row_prim, glpk.sm_row_dual, dual)

    def unboundedness(self):
        ind = glpk.sm_unbnd_ray(self._problem)
        constraints = len(self._program._constraints)
        if ind is 0:
            return (None, '')
        elif ind <= constraints:
            varstraint = self._program._constraints[ind]
            varstat = glpk.get_row_stat(self._problem, ind)
        else:
            ind -= constraints
            varstraint = self._program._variables[ind]
            varstat = glpk.get_col_stat(self._problem, ind)
        nature = 'dual' if varstat is glpk.BS else 'primal'
        return (varstraint, nature)

    def basis(self, basis=None, warmup=False):
        if basis is 'standard':
            glpk.std_basis(self._problem)
        elif basis is 'advanced':
            glpk.adv_basis(self._problem, 0)
        elif basis is 'Bixby':
            glpk.cpx_basis(self._problem)
        elif basis is not None:
            for varstraint, string in basis.items():
                varstat = str2varstat[string]
                if isinstance(varstraint, Variable):
                    col = self._program._ind(varstraint)
                    glpk.set_col_stat(self._problem, col, varstat)
                elif isinstance(varstraint, Constraint):
                    row = self._program._ind(varstraint)
                    glpk.set_row_stat(self._problem, row, varstat)
                else:
                    raise TypeError("Only 'Variable' and 'Constraint' " +
                                    "can have a status.")
        if warmup:
            retcode = glpk.warm_up(self._problem)
            if retcode is not 0:
                raise smretcode2error[retcode]
        basis = {}
        for col, variable in enumerate(self._program._variables, start=1):
            varstat = glpk.get_col_stat(self._problem, col)
            basis[variable] = varstat2str[varstat]
        for row, constraint in enumerate(self._program._constraints, start=1):
            varstat = glpk.get_row_stat(self._problem, row)
            basis[constraint] = varstat2str[varstat]
        return basis

    def print_solution(self, fname):
        """Write the solution to a file in a readable format

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.print_sol, fname)

    def read_solution(self, fname):
        """Read the solution from a file

        :param fname: the name of the file to read from
            (written by :meth:`SimplexSolver.write_solution`)
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error reading the file

        .. todo::

            Add doctest

        """
        self._read(glpk.read_sol, fname)

    def write_solution(self, fname):
        """Write the solution to a file

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.write_sol, fname)

    def print_ranges(self, varstraints, fname):
        if self.status() is not 'optimal':
            raise Exception("Solution must be optimal.")
        if not isinstance(varstraints, collections.abc.Sequence):
            raise TypeError("'varstraints' must be a sequence.")
        length = len(varstraints)
        cdef char* chars
        fname = name2chars(fname)
        chars = fname
        cdef int* indlist = <int*>glpk.calloc(1+length, sizeof(int))
        try:
            rows = len(self._program._constraints)
            for pos, varstraint in enumerate(varstraints, start=1):
                ind = self._program._ind(varstraint)
                if isinstance(varstraint, Variable):
                    ind += rows
                indlist[pos] = ind
            if not glpk.bf_exists(self._problem):
                retcode = glpk.factorize(self._problem)
                if retcode is not 0:
                    raise smretcode2error[retcode]
            glpk.print_ranges(self._problem, length, indlist, 0, chars)
        finally:
            glpk.free(indlist)


cdef class IPointSolver(_LPSolver):
    """The problem's interior point solver"""

    cdef glpk.IPointCP _iptcp

    def __cinit__(self, program):
        glpk.init_iptcp(&self._iptcp)

    def controls(self, defaults=False, **controls):
        """Change or retrieve the solver's control parameters

        :param defaults: whether to set the parameters back to their default
            values or not
        :type defaults: :class:`bool`
        :param controls: zero or more named parameters to change from the
            following list:

            * :data:`msg_lev` (:class:`str`) – the message level,
              with possible values

              * :data:`'no'`: no output
              * :data:`'warnerror'`: warnings and errors only
              * :data:`'normal'`: normal output
              * :data:`'full'`: normal output and informational messages

            * :data:`ord_alg` (:class:`str`) – the ordering algorithm used
              prior to Cholesky factorization, with possible values

              * :data:`'orig'`: normal (original)
              * :data:`'qmd'`: quotient minimum degree
              * :data:`'amd'`: approximate minimum degree
              * :data:`'symamd'`: approximate minimum degree for symmetric
                matrices

        :raises ValueError: if a non-existing control name is given

        .. todo::

            Add doctest

        """
        if defaults:
            glpk.init_iptcp(&self._iptcp)
        for control, val in controls.items():
            if control is 'msg_lev':
                self._iptcp.msg_lev = str2msglev[val]
            elif control is 'ord_alg':
                self._iptcp.ord_alg = str2ordalg[val]
            else:
                raise ValueError("Non-existing control: " + repr(control))
        controls = {}
        controls = self._iptcp
        controls['msg_lev'] = msglev2str[controls['msg_lev']]
        controls['ord_alg'] = ordalg2str[controls['ord_alg']]
        return controls

    def solve(self):
        """Solve the linear program

        :returns: solution status; see :meth:`IPointSolver.status` for details
        :rtype: :class:`str`
        :raises ValueError: if the problem has no rows/columns
        :raises ArithmeticError: if there occurs very slow convergence or
            divergence
        :raises StopIteration: if the iteration limit is exceeded
        :raises ArithmeticError: if numerical instability occurs on solving
            the Newtonian system

        .. todo::

            Add doctest

        """
        retcode = glpk.interior(self._problem, &self._iptcp)
        if retcode is 0:
            return self.status()
        else:
            raise iptretcode2error[retcode]

    def status(self):
        """Return the current solution status

        :returns: the current solution status, either :data:`'undefined'`,
            :data:`'optimal`, :data:`'infeasible'`, or :data:`'no feasible'`
        :rtype: :class:`str`

        .. todo::

            Add doctest

        """
        return solstat2str[glpk.ipt_status(self._problem)]

    def objective(self):
        """Return the objective value for the current solution

        :returns: the objective value for the current solution
        :rtype: :class:`float`

        .. todo::

            Add doctest

        """
        return glpk.ipt_obj_val(self._problem)

    def variables(self, dual=False):
        """Returns the values of the variables for the current solution

        :param dual: whether to return dual or primal values
        :type dual: :class:`bool`
        :returns: the nonzero values of the variables for the current
            solution
        :rtype: :class:`dict` from :class:`Variable` to :class:`float`
        :raises TypeError: if `dual` is not :class:`bool`

        .. todo::

            Add doctest

        """
        return self._solution(self._program._variables,
                              glpk.ipt_col_prim, glpk.ipt_col_dual, dual)

    def constraints(self, dual=False):
        """Returns the values of the constraints for the current solution

        :param dual: whether to return dual or primal values
        :type dual: :class:`bool`
        :returns: the nonzero values of the constraints for the current
            solution
        :rtype: :class:`dict` from :class:`Constraint` to :class:`float`
        :raises TypeError: if `dual` is not :class:`bool`

        .. todo::

            Add doctest

        """
        return self._solution(self._program._constraints,
                              glpk.ipt_row_prim, glpk.ipt_row_dual, dual)

    def print_solution(self, fname):
        """Write the solution to a file in a readable format

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.print_ipt, fname)

    def read_solution(self, fname):
        """Read the solution from a file

        :param fname: the name of the file to read from
            (written by :meth:`IPointSolver.write_solution`)
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error reading the file

        .. todo::

            Add doctest

        """
        self._read(glpk.read_ipt, fname)

    def write_solution(self, fname):
        """Write the solution to a file

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.write_ipt, fname)


cdef class IntOptSolver(_Solver):
    """The problem's integer optimization solver"""

    cdef glpk.IntOptCP _iocp

    def __cinit__(self, program):
        glpk.init_iocp(&self._iocp)

    def controls(self, defaults=False, **controls):
        """Change or retrieve the solver's control parameters

        :param defaults: whether to set the parameters back to their default
            values or not
        :type defaults: :class:`bool`
        :param controls: zero or more named parameters to change from the
            following list:

            * :data:`msg_lev` (:class:`str`) – the message level,
              with possible values

              * :data:`'no'`: no output
              * :data:`'warnerror'`: warnings and errors only
              * :data:`'normal'`: normal output
              * :data:`'full'`: normal output and informational messages

            * :data:`out_frq` (:class:`~numbers.Integral`) – output frequency
              [ms] of informational messages
            * :data:`out_dly` (:class:`~numbers.Integral`) – output delay
              [ms] of current LP relaxation solution
            * :data:`tm_lim` (:class:`~numbers.Integral`) – time limit [ms]
            * :data:`br_tech` (:class:`str`) – the branching technique,
              with possible values

              * :data:`'first_fracvar'`: first fractional variable
              * :data:`'last_fracvar'`: last fractional variable
              * :data:`'most_fracvar'`: most fractional variable
              * :data:`'Driebeek-Tomlin'`: heuristic by `Driebeek
                <http://www.jstor.org/discover/10.2307/2627887>`_ & Tomlin
              * :data:`'hybrid_peudocost'`: hybrid pseudocost heuristic

            * :data:`bt_tech` (:class:`str`) – the backtracking technique,
              with possible values

              * :data:`'depth'`: depth first search
              * :data:`'breadth'`: breadth first search
              * :data:`'bound'`: best local bound
              * :data:`'projection'`: best projection heuristic

            * :data:`pp_tech` (:class:`str`) – the preprocessing technique,
              with possible values

              * :data:`'none'`: disable preprocessing
              * :data:`'root'`: preprocessing only on the root level
              * :data:`'all'`: preprocessing on all levels

            * :data:`mir_cuts` (:class:`bool`) –
              generate mixed integer rounding cuts
            * :data:`gmi_cuts` (:class:`bool`) –
              generate Gomory’s mixed integer cuts
            * :data:`cov_cuts` (:class:`bool`) –
              generate mixed cover cuts
            * :data:`clq_cuts` (:class:`bool`) –
              generate clique cuts
            * :data:`fp_heur` (:class:`bool`) –
              apply feasibility pump heuristic
            * :data:`tol_int` (:class:`~numbers.Real`) – absolute tolerance
              used to check if the optimal solution to the current LP
              relaxation is integer feasible
            * :data:`tol_obj` (:class:`~numbers.Real`) – relative tolerance
              used to check if the objective value in the optimal solution to
              the current LP relaxation is not better than in the best known
              integer feasible solution.
            * :data:`mip_gap` (:class:`~numbers.Real`) – relative MIP-gap
              tolerance
              (search stops once the relative MIP-gap falls below this value)
            * :data:`presolve` (:class:`bool`) – use MIP presolver,
              may simplify the problem
            * :data:`binarize` (:class:`bool`) – binarize integer variables
              (only used if :data:`presolve` is :data:`True`)

        :raises ValueError: if a non-existing control name is given

        .. todo::

            Add doctest

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

            * :data:`'branchcut'`: a branch-and-cut solver
            * :data:`'intfeas1'`: a SAT solver based integer feasibility
              solver; applicable only to problems

              * that are feasibility problems (but see `obj_bound` description)
              * with only binary variables
                (and integer variables with coinciding lower and upper bound)
              * with all-integer coefficients

              and furthermore efficient mainly for problems with constraints
              that are covering, packing, or partitioning inequalities, i.e.,
              sums of binary variables :math:`x` or their ‘negation’
              :math:`1-x`, smaller than, equal to, or larger than :math:`1`.

        :type solver: :class:`str`
        :param obj_bound: if `solver` is :data:`'intfeas1'`, a solution is
            considered feasible only if the corresponding objective value is
            not worse than this bound (not used if solver is
            :data:`'branchcut'`)
        :type obj_bound: :class:`~numbers.Integral`
        :returns: solution status; see :meth:`IntOptSolver.status` for details
        :rtype: :class:`str`
        :raises ValueError: if `solver` is neither :data:`'branchcut'` nor
            :data:`'intfeas1'`
        :raises TypeError: if `obj_bound` is not :class:`~numbers.Integral`
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
            (only relevant if `solver` is :data:`'intfeas1'`)
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

        :returns: the current solution status, either :data:`'undefined'`,
            :data:`'optimal`, :data:`'no feasible'`, or :data:`'feasible'`
        :rtype: :class:`str`

        .. todo::

            Add doctest

        """
        return solstat2str[glpk.mip_status(self._problem)]

    def objective(self):
        """Return the objective value for the current solution

        :returns: the objective value for the current solution
        :rtype: :class:`float`

        .. todo::

            Add doctest

        """
        return glpk.mip_obj_val(self._problem)

    def variables(self):
        """Returns the values of the variables for the current solution

        :returns: the nonzero values of the variables for the current
            solution
        :rtype: :class:`dict` from :class:`Variable` to :class:`float` or
            :class:`int`
        :raises ValueError: if a variable with :data:`'integer'` or
            :data:`'binary'` kind has a non-integer value

        .. todo::

            Add doctest

        """
        solution = {}
        for col, variable in enumerate(self._program._variables, start=1):
            val = glpk.mip_col_val(self._problem, col)
            if variable.kind() in {'integer', 'binary'}:
                if val.is_integer():
                    val = int(val)
                else:
                    raise ValueError("Variable with integer or binary kind" +
                                     "has non-integer value")
            if val != 0:
                solution[variable] = val
        return solution

    def constraints(self):
        """Returns the values of the constraints for the current solution

        :returns: the nonzero values of the constraints for the current
            solution
        :rtype: :class:`dict` from :class:`Constraint` to :class:`float`

        .. todo::

            Add doctest

        """
        solution = {}
        for row, constraint in enumerate(self._program._constraints, start=1):
            val = glpk.mip_row_val(self._problem, row)
            if val != 0:
                solution[constraint] = val
        return solution

    def print_solution(self, fname):
        """Write the solution to a file in a readable format

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.print_mip, fname)

    def read_solution(self, fname):
        """Read the solution from a file

        :param fname: the name of the file to read from
            (written by :meth:`IntOptSolver.write_solution`)
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error reading the file

        .. todo::

            Add doctest

        """
        self._read(glpk.read_mip, fname)

    def write_solution(self, fname):
        """Write the solution to a file

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.write_mip, fname)
