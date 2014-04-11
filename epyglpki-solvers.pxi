# epyglpki-solvers.pxi: Cython/Python interface for GLPK solvers

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


cdef class _Solver(_ProgramComponent):

    cdef _value(self, varstraint,
                double (*variable_func)(glpk.ProbObj*, int),
                double (*constraint_func)(glpk.ProbObj*, int)):
        if isinstance(varstraint, Variable):
            col = self._program._col(varstraint)
            return variable_func(self._problem, col)
        elif isinstance(varstraint, Constraint):
            row = self._program._row(varstraint)
            return constraint_func(self._problem, row)
        else:
            raise TypeError("varstraint must be a Variable or Constraint")

    def _error(self, soltype, solver):
        cdef double* ae_max = <double*>glpk.alloc(1, sizeof(double))
        cdef int* ae_ind = <int*>glpk.alloc(1, sizeof(int))
        cdef double* re_max = <double*>glpk.alloc(1, sizeof(double))
        cdef int* re_ind = <int*>glpk.alloc(1, sizeof(int))
        try:
            if soltype is 'primal':
                eqtype = glpk.KKT_PE
                bndtype = glpk.KKT_PB
            elif soltype is 'dual':
                eqtype = glpk.KKT_DE
                bndtype = glpk.KKT_DB
            else:
                raise ValueError("soltype should be 'primal' or 'dual'.")
            error = {}
            # equalities
            glpk.check_kkt(self._problem, solver, eqtype,
                           ae_max, ae_ind, re_max, re_ind)
            a_max = ae_max[0]
            a_ind = ae_ind[0]
            r_max = re_max[0]
            r_ind = re_ind[0]
            if eqtype is glpk.KKT_PE:
                a_varstraint = self._program._constraint(a_ind)
                r_varstraint = self._program._constraint(r_ind)
            elif eqtype is glpk.KKT_DE:
                a_varstraint = self._program._variable(a_ind)
                r_varstraint = self._program._variable(r_ind)
            error['equalities'] = {'abs': (a_max, a_varstraint),
                                   'rel': (r_max, r_varstraint)}
            # bounds
            glpk.check_kkt(self._problem, solver, bndtype,
                           ae_max, ae_ind, re_max, re_ind)
            a_max = ae_max[0]
            a_ind = ae_ind[0]
            r_max = re_max[0]
            r_ind = re_ind[0]
            a_varstraint = self._program._varstraint(a_ind)
            r_varstraint = self._program._varstraint(r_ind)
            error['bounds'] = {'abs': (a_max, a_varstraint),
                               'rel': (r_max, r_varstraint)}
            return error
        finally:
            glpk.free(ae_max)
            glpk.free(ae_ind)
            glpk.free(re_max)
            glpk.free(ae_ind)

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


cdef class SimplexSolver(_Solver):
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
        :type defaults: `bool`
        :param controls: zero or more named parameters to change from the
            following list:

            * **msg_lev** (`str`) – the message level, with possible values

              * `'no'`: no output
              * `'warnerror'`: warnings and errors only
              * `'normal'`: normal output
              * `'full'`: normal output and informational messages

            * **meth** (`str`) – simplex method, with possible values

              * `'primal'`: two-phase primal simplex
              * `'dual'`: two-phase dual simplex
              * `'dual_fail_primal'`: two-phase dual simplex and, if it fails,
                switch to primal simplex

            * **pricing** (`str`) – pricing technique, with possible values

              * `'Dantzig'`: standard ‘textbook’
              * `'steepest'`: projected steepest edge

            * **r_test** (`str`) – ratio test technique, with possible values

              * `'standard'`: standard ‘textbook’
              * `'Harris'`: Harris’s two-pass ratio test

            * **tol_bnd** (|Real|) – tolerance used to check if the basic
              solution is primal feasible
            * **tol_dj** (|Real|) – tolerance used to check if the basic
              solution is dual feasible
            * **tol_piv** (|Real|) – tolerance used to choose eligble pivotal
              elements of the simplex table
            * **obj_ll** (|Real|) – lower limit of the objective function
              (only if *meth* is `'dual'`)
            * **obj_ul** (|Real|) – upper limit of the objective function
              (only if *meth* is `'dual'`)
            * **it_lim** (|Integral|) – iteration limit
            * **tm_lim** (|Integral|) – time limit [ms]
            * **out_frq** (|Integral|) – output frequency [iterations] of
              informational messages
            * **out_dly** (|Integral|) – output delay [ms] of solution process
              information
            * **presolve** (`bool`) – use LP presolver

            or, for basis factorization, from the following list:

            * **type** (length-2 `tuple` of `str`) – basis factorization type,
              pairs with possible first components

              * `'LU'`: plain LU factorization
              * `'BTLU'`: block-triangular LU factorization

              and possible second components

              * `'Forrest-Tomlin'`: `Forrest–Tomlin`_ update applied to U
                (only with plain LU factorization)
              * `'Bartels-Golub'`: `Bartels–Golub`_ update applied to Schur
                complement
              * `'Givens'`: Givens rotation update applied to Schur complement

            * **piv_tol** (|Real|) – Markowitz threshold pivoting tolerance
              (value must lie between 0 and 1)
            * **piv_lim** (|Integral|) – number of pivot
              candidates that need to be considered on choosing a pivot element
              (at least 1)
            * **suhl** (`bool`) – use Suhl heuristic
            * **eps_tol** (|Real|) – tolerance below which numbers are replaced
              by zero
            * **nfs_max** (|Integral|) – maximal number of additional row-like
              factors (used only when *type* is `'Forrest-Tomlin'`)
            * **nrs_max** (|Integral|) – maximal number of additional row and
              columns
              (used only when *type* is `'Bartels-Golub'` or `'Givens'`)

        :raises ValueError: if a non-existing control name is given

        .. todo::

            Add doctest

        .. _Forrest–Tomlin: http://dx.doi.org/10.1007/BF01584548
        .. _Bartels–Golub: http://dx.doi.org/10.1145/362946.362974

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
                self._bfcp.type = strpair2bftype[val]
            # double parameters
            elif control in {
                # smcp
                'tol_bnd', 'tol_dj', 'tol_piv', 'obj_ll', 'obj_ul',
                # bfcp
                'piv_tol', 'eps_tol'
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
            # int parameters
            elif control in {
                # smcp
                'it_lim', 'tm_lim', 'out_frq', 'out_dly',
                # bfcp
                'piv_lim', 'nfs_max', 'nrs_max'
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
                elif control is 'piv_lim':
                    self._bfcp.piv_lim = val
                elif control is 'nfs_max':
                    self._bfcp.nfs_max = val
                elif control is 'nrs_max':
                    self._bfcp.nrs_max = val
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
        fcontrols['type'] = bftype2strpair[fcontrols['type']]
        controls = {}
        controls.update(scontrols)
        controls.update(fcontrols)
        return controls

    def solve(self, exact=False):
        """Solve the linear program

        :param exact: whether to use exact arithmetic or not
            (only if the *meth* control parameter is `'primal'`)
        :type exact: `bool`
        :returns: solution status; see `.status` for details,
            or `'obj_ll reached'` or `'obj_ul reached'` in case that happens
        :rtype: `str`
        :raises ValueError: if *exact* is `True` but the *meth* control
            parameter is not `'primal'`
        :raises ValueError: if finite values are set for *obj_ll* or *obj_ul*
            while the *meth* control parameter is not `'dual'`
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
        :type detailed: `bool`
        :returns: the current solution status

            * in case *detailed* is `False`, either `'undefined'`, `'optimal'`,
              `'infeasible'`, `'no feasible'`, `'feasible'`, or `'unbounded'`
            * in case *detailed* is `True`, a pair of statuses is given,
              one for the primal solution and one for the dual solution, either
              `'undefined'`, `'infeasible'`, `'no feasible'`, or `'feasible'`

        :rtype: `str` or length-2 `tuple` of `str`

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
        :rtype: `float`

        .. todo::

            Add doctest

        """
        return glpk.sm_obj_val(self._problem)


    def primal(self, varstraint):
        """Return primal value for the current solution

        :param varstraint: variable or constraint to return the primal value of
        :type varstraint: `.Variable` or `.Constraint`
        :returns: the value of *varstraint* for the current solution
        :rtype: `float` or `int`
        :raises TypeError: if varstraint is neither `.Variable` nor
            `.Constraint`

        .. todo::

            Add doctest

        """
        return self._value(varstraint, glpk.sm_col_prim, glpk.sm_row_prim)


    def primal_error(self):
        """Return absolute and relative primal solution errors

        :returns: a |Mapping| of {`'equalities'`, `'bounds'`} to |Mapping|
            of {`'abs'`, `'rel'`} to pairs consisting of an error (`float`)
            and a `.Variable` or `.Constraint` where it is attained

        The errors returned by this function quantify to what degree the
        current primal solution does not satisfy the Karush-Kuhn-Tucker
        conditions for equalities and bounds
        (possibly due to floating-point errors).

        .. todo::

            Add doctest

        """
        return self._error('primal', glpk.SOL)

    def dual(self, varstraint):
        """Return dual value for the current solution

        :param varstraint: variable or constraint to return the dual value of
        :type varstraint: `.Variable` or `.Constraint`
        :returns: the value of *varstraint* for the current solution
        :rtype: `float` or `int`
        :raises TypeError: if varstraint is neither `.Variable` nor
            `.Constraint`

        .. todo::

            Add doctest

        """
        return self._value(varstraint, glpk.sm_col_dual, glpk.sm_row_dual)

    def dual_error(self):
        """Return absolute and relative dual solution errors

        :returns: a |Mapping| of {`'equalities'`, `'bounds'`} to |Mapping|
            of {`'abs'`, `'rel'`} to pairs consisting of an error (`float`)
            and a `.Variable` or `.Constraint` where it is attained

        The errors returned by this function quantify to what degree the
        current dual solution does not satisfy the Karush-Kuhn-Tucker
        conditions for equalities and bounds
        (possibly due to floating-point errors).

        .. todo::

            Add doctest

        """
        return self._error('dual', glpk.SOL)

    def unboundedness(self):
        """Return a variable or constraint causing unboundedness

        :returns: a variable or constraint causing unboundedness (if any) and
            the nature of the unboundedness, either `'primal'` or `'dual'`
        :rtype: length-2 `tuple` of `.Variable` or `.Constraint` and `str`

        .. todo::

            Add doctest

        """
        ind = glpk.sm_unbnd_ray(self._problem)
        varstraint = self._program._varstraint(ind)
        nature = 'primal'
        constraints = len(self._program._constraints)
        if ind is 0:
            nature = ''
        elif ind <= constraints:
            if glpk.get_row_stat(self._problem, ind) is glpk.BS:
                nature = 'dual'
        else:
            if glpk.get_col_stat(self._problem, ind) is glpk.BS:
                nature = 'dual'
        return (varstraint, nature)

    def basis(self, algorithm=None, status=None, warmup=False):
        """Change or retrieve basis

        A basis is defined by the statuses assigned to all variables and
        constraints; the possible statuses are

        * `'basic'`: basic
        * `'lower'`: non-basic with active lower bound
        * `'upper'`: non-basic with active upper bound
        * `'free'`: non-basic free (unbounded)
        * `'fixed'`: non-basic fixed

        A basis is valid if the basis matrix is non-singular, which implies
        that the number of basic variables and constraints is equal to the
        total number of constraints.

        :param algorithm: an algorithm for generating a basis (omit to not
            replace the current basis), chosen from

            * `'standard'`: sets all constraints as basic
            * `'advanced'`: sets as basic

              #. all non-fixed constraints
              #. as many non-fixed variables as possible, while preserving the
                 lower triangular structure of the basis matrix
              #. appropriate fixed constraints to complete the basis

            * `'Bixby'`: algorithm used by CPLEX, as discussed by Bixby_

        :type algorithm: `str`
        :param status: the mapping of statuses to change
            (omit to not modify the basis)
        :type status: |Mapping| from `.Variable` or `.Constraint` to `str`
        :param warmup: whether to ‘warm up’ the basis, so that `.solve` can be
            used without presolving
        :type warmup: `bool`
        :returns: a mapping of the basis statuses of all variables and
            constraints
        :rtype: `dict` from `.Variable` or `.Constraint` to `str`
        :raises ValueError: if *algorithm* is neither `'standard'`,
            `'advanced'`, nor `'Bixby'`
        :raises TypeError: if *status* is not |Mapping|
        :raises TypeError: if *status* keys are neither `.Variable` nor
            `.Constraint`
        :raises ValueError: if the basis is invalid
        :raises ValueError: if the basis matrix is singular
        :raises ValueError: if the basis matrix is ill-conditioned

        .. todo::

            Add doctest

        .. note::

            After `.solve` has been run successfully, the
            basis is left in a valid state. So it is not necessary to run this
            method before, e.g., re-optimizating after only the objective has
            been changed.

        .. _Bixby: http://dx.doi.org/10.1287/ijoc.4.3.267

        """
        if algorithm is not None:
            if algorithm is 'standard':
                glpk.std_basis(self._problem)
            elif algorithm is 'advanced':
                glpk.adv_basis(self._problem, 0)
            elif algorithm is 'Bixby':
                glpk.cpx_basis(self._problem)
            else:
                raise ValueError(repr(algorithm)
                                 + " is not a basis generation algorithm.")
        if isinstance(status, collections.abc.Mapping):
            for varstraint, string in status.items():
                varstat = str2varstat[string]
                if isinstance(varstraint, Variable):
                    col = self._program._col(varstraint)
                    glpk.set_col_stat(self._problem, col, varstat)
                elif isinstance(varstraint, Constraint):
                    row = self._program._row(varstraint)
                    glpk.set_row_stat(self._problem, row, varstat)
                else:
                    raise TypeError("Only 'Variable' and 'Constraint' " +
                                    "can have a status.")
        elif status is not None:
            raise TypeError("Statuses must be specified using a Mapping.")
        if warmup:
            retcode = glpk.warm_up(self._problem)
            if retcode is not 0:
                raise smretcode2error[retcode]
        status = {}
        for col, variable in enumerate(self._program._variables, start=1):
            varstat = glpk.get_col_stat(self._problem, col)
            status[variable] = varstat2str[varstat]
        for row, constraint in enumerate(self._program._constraints, start=1):
            varstat = glpk.get_row_stat(self._problem, row)
            status[constraint] = varstat2str[varstat]
        return status

    def print_solution(self, fname):
        """Write the solution to a file in a readable format

        :param fname: the name of the file to write to
        :type fname: `str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.print_sol, fname)

    def read_solution(self, fname):
        """Read the solution from a file

        :param fname: the name of the file to read from
            (written by `.write_solution`)
        :type fname: `str`
        :raises RuntimeError: if there is an error reading the file

        .. todo::

            Add doctest

        """
        self._read(glpk.read_sol, fname)

    def write_solution(self, fname):
        """Write the solution to a file

        :param fname: the name of the file to write to
        :type fname: `str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.write_sol, fname)

    def print_ranges(self, varstraints, fname):
        """Write a sensitivity analysis report to file in readable format

        :param varstraints: sequence of variables and/or constraints to analyze
        :type varstraints: |Sequence| of `.Variable` and/or `.Constraint`
        :param fname: the name of the file to write to
        :type fname: `str`
        :raises Exception: if the current solution is not optimal
        :raises TypeError: if *varstraints* is not  |Sequence| of
            `.Variable` and/or `.Constraint`
        :raises ValueError: if the current basis is invalid
        :raises ValueError: if the current basis matrix is singular
        :raises ValueError: if the current basis matrix is ill-conditioned

        .. todo::

            Add doctest

        """
        if self.status() is not 'optimal':
            raise Exception("Solution must be optimal.")
        if not isinstance(varstraints, collections.abc.Sequence):
            raise TypeError("'varstraints' must be a sequence.")
        length = len(varstraints)
        cdef char* chars
        fname = name2chars(fname)
        chars = fname
        cdef int* indlist = <int*>glpk.alloc(1+length, sizeof(int))
        try:
            for pos, varstraint in enumerate(varstraints, start=1):
                ind = self._program._ind(varstraint, alternate=True)
                indlist[pos] = ind
            if not glpk.bf_exists(self._problem):
                retcode = glpk.factorize(self._problem)
                if retcode is not 0:
                    raise smretcode2error[retcode]
            glpk.print_ranges(self._problem, length, indlist, 0, chars)
        finally:
            glpk.free(indlist)


cdef class IPointSolver(_Solver):
    """The problem's interior point solver"""

    cdef glpk.IPointCP _iptcp

    def __cinit__(self, program):
        glpk.init_iptcp(&self._iptcp)

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

            * **ord_alg** (`str`) – the ordering algorithm used prior to
              Cholesky factorization, with possible values

              * `'orig'`: normal (original)
              * `'qmd'`: quotient minimum degree
              * `'amd'`: approximate minimum degree
              * `'symamd'`: approximate minimum degree for symmetric matrices

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
        retcode = glpk.interior(self._problem, &self._iptcp)
        if retcode is 0:
            return self.status()
        else:
            raise iptretcode2error[retcode]

    def status(self):
        """Return the current solution status

        :returns: the current solution status, either `'undefined'`,
            `'optimal'`, `'infeasible'`, or `'no feasible'`
        :rtype: `str`

        .. todo::

            Add doctest

        """
        return solstat2str[glpk.ipt_status(self._problem)]

    def objective(self):
        """Return the objective value for the current solution

        :returns: the objective value for the current solution
        :rtype: `float`

        .. todo::

            Add doctest

        """
        return glpk.ipt_obj_val(self._problem)

    def primal(self, varstraint):
        """Return primal value for the current solution

        :param varstraint: variable or constraint to return the primal value of
        :type varstraint: `.Variable` or `.Constraint`
        :returns: the value of *varstraint* for the current solution
        :rtype: `float` or `int`
        :raises TypeError: if varstraint is neither `.Variable` nor
            `.Constraint`

        .. todo::

            Add doctest

        """
        return self._value(varstraint, glpk.ipt_col_prim, glpk.ipt_row_prim)

    def primal_error(self):
        """Return absolute and relative primal solution errors

        :returns: a |Mapping| of {`'equalities'`, `'bounds'`} to |Mapping|
            of {`'abs'`, `'rel'`} to pairs consisting of an error (`float`)
            and a `.Variable` or `.Constraint` where it is attained

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
        :type varstraint: `.Variable` or `.Constraint`
        :returns: the value of *varstraint* for the current solution
        :rtype: `float` or `int`
        :raises TypeError: if varstraint is neither `.Variable` nor
            `.Constraint`

        .. todo::

            Add doctest

        """
        return self._value(varstraint, glpk.ipt_col_dual, glpk.ipt_row_dual)

    def dual_error(self):
        """Return absolute and relative dual solution errors

        :returns: a |Mapping| of {`'equalities'`, `'bounds'`} to |Mapping|
            of {`'abs'`, `'rel'`} to pairs consisting of an error (`float`)
            and a `.Variable` or `.Constraint` where it is attained

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


cdef class IntOptSolver(_Solver):
    """The problem's integer optimization solver"""

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
        :type varstraint: `.Variable` or `.Constraint`
        :returns: the value of *varstraint* for the current solution
        :rtype: `float` or `int`
        :raises TypeError: if varstraint is neither `.Variable` nor
            `.Constraint`
        :raises ValueError: if a variable with `'integer'` or `'binary'` kind
            has a non-integer value

        .. todo::

            Add doctest

        """
        val = self._value(varstraint, glpk.mip_col_val, glpk.mip_row_val)
        if isinstance(varstraint, Variable):
            if varstraint.kind() in {'binary', 'integer'}:
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
