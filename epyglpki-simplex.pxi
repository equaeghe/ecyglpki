# epyglpki-solvers.pxi: Cython/Python interface for GLPK simplex solver

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


cdef class SimplexSolver(_Solver):
    """A simplex solver

    .. doctest:: SimplexSolver

        >>> p = MILProgram()
        >>> s = p.simplex
        >>> isinstance(s, SimplexSolver)
        True

    """

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

    property status:
        """The current solution status, a `str`

        The possible values are `'undefined'`, `'optimal'`, `'infeasible'`,
        `'no feasible'`, `'feasible'`, or `'unbounded'`.

        .. doctest:: SimplexSolver

            >>> s.status
            'undefined'

        """
        def __get__(self):
            return solstat2str[glpk.sm_status(self._problem)]

    property status_primal:
        """The current primal solution status, a `str`

        The possible values are `'undefined'`, `'infeasible'`, `'no feasible'`,
        or `'feasible'`.

        .. doctest:: SimplexSolver

            >>> s.status_primal
            'undefined'

        """
        def __get__(self):
            return solstat2str[glpk.sm_prim_stat(self._problem)]

    property status_dual:
        """The current solution status, a `str`

        The possible values are `'undefined'`, `'infeasible'`, `'no feasible'`,
        or `'feasible'`.

        .. doctest:: SimplexSolver

            >>> s.status_dual
            'undefined'

        """
        def __get__(self):
            return solstat2str[glpk.sm_dual_stat(self._problem)]

    property objective:
        """The objective value for the current solution, a |Real| number

        .. doctest:: SimplexSolver

            >>> s.objective
            0.0

        """
        def __get__(self):
            return glpk.sm_obj_val(self._problem)

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
        return self._value(varstraint, glpk.sm_col_prim, glpk.sm_row_prim)


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
        return self._error('primal', glpk.SOL)

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
        return self._value(varstraint, glpk.sm_col_dual, glpk.sm_row_dual)

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
        return self._error('dual', glpk.SOL)

    def unboundedness(self):
        """Return a variable or constraint causing unboundedness

        :returns: a variable or constraint causing unboundedness (if any) and
            the nature of the unboundedness, either `'primal'` or `'dual'`
        :rtype: length-2 `tuple` of `.Varstraint` and `str`

        .. todo::

            Add doctest

        """
        ind = glpk.sm_unbnd_ray(self._problem)
        varstraint = self._program._from_varstraintind(ind)
        nature = 'primal'
        constraints = len(self._program.constraints)
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
        :type status: |Mapping| from `.Varstraint` to `str`
        :param warmup: whether to ‘warm up’ the basis, so that `.solve` can be
            used without presolving
        :type warmup: `bool`
        :returns: a mapping of the basis statuses of all variables and
            constraints
        :rtype: `dict` from `.Varstraint` to `str`
        :raises ValueError: if *algorithm* is neither `'standard'`,
            `'advanced'`, nor `'Bixby'`
        :raises TypeError: if *status* is not |Mapping|
        :raises TypeError: if *status* keys are not `.Varstraint`
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
                    glpk.set_col_stat(self._problem, varstraint._ind, varstat)
                elif isinstance(varstraint, Constraint):
                    glpk.set_row_stat(self._problem, varstraint._ind, varstat)
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
        for col, variable in enumerate(self._program.variables, start=1):
            varstat = glpk.get_col_stat(self._problem, col)
            status[variable] = varstat2str[varstat]
        for row, constraint in enumerate(self._program.constraints, start=1):
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
        :type varstraints: |Sequence| of `.Varstraint`
        :param fname: the name of the file to write to
        :type fname: `str`
        :raises Exception: if the current solution is not optimal
        :raises TypeError: if *varstraints* is not |Sequence| of `.Varstraint`
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
        k = len(varstraints)
        cdef int* inds = <int*>glpk.alloc(1+k, sizeof(int))
        try:
            for i, varstraint in enumerate(varstraints, start=1):
                inds[i] = varstraint._varstraintind
            if not glpk.bf_exists(self._problem):
                retcode = glpk.factorize(self._problem)
                if retcode is not 0:
                    raise smretcode2error[retcode]
            glpk.print_ranges(self._problem, k, inds, 0, name2chars(fname))
        finally:
            glpk.free(inds)
