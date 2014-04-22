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


cdef class SimplexControls:
    """The simplex solver (`.SimplexSolver`) control parameter object

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


cdef class Basis(_Component):
    """The basis of a simplex solver"""

    cdef readonly Factorization factorization
    """The basis factorization, a `.Factorization` object"""

    def __cinit__(self, program):
        self.factorization = Factorization(self._program)


cdef class Factorization(_Component):
    """The basis factorization"""

    property controls:
        """The basis factorization controls"""
        def __get__(self):
            return FactorizationControls(self._program)
        def __set__(self, controls):
            cdef glpk.BasFacCP bfcp = controls._bfcp
            glpk.set_bfcp(self._problem, &bfcp)
        def __del__(self):
            glpk.set_bfcp(self._problem, NULL)


cdef class FactorizationControls(_Component):
    """The basis factorization control parameter object

    .. doctest:: FactorizationControls

        >>> r = MILProgram().simplex.basis.factorization.controls
        >>> isinstance(r, FactorizationControls)
        True

    """

    cdef glpk.BasFacCP _bfcp

    def __cinit__(self, program):
        glpk.get_bfcp(self._problem, &self._bfcp)

    property type:
        """The basis factorization type, `str` pairs

        Possible first components:

        * `'LU'`: plain LU factorization
        * `'BTLU'`: block-triangular LU factorization

        Possible second components

        * `'Forrest-Tomlin'`: `Forrest–Tomlin`_ update applied to U
          (only with plain LU factorization)
        * `'Bartels-Golub'`: `Bartels–Golub`_ update applied to Schur
          complement
        * `'Givens'`: Givens rotation update applied to Schur complement

        .. _Forrest–Tomlin: http://dx.doi.org/10.1007/BF01584548
        .. _Bartels–Golub: http://dx.doi.org/10.1145/362946.362974

        .. doctest:: FactorizationControls

            >>> r.type  # the GLPK default
            ('LU', 'Forrest-Tomlin')

        """
        def __get__(self):
            return bftype2strpair[self._bfcp.type]
        def __set__(self, value):
            self._bfcp.type = strpair2bftype[value]

    property piv_tol:
        """Markowitz threshold pivoting tolerance, a |Real| number

        (Value must lie between 0 and 1.)

        """
        def __get__(self):
            return self._bfcp.piv_tol
        def __set__(self, value):
            self._bfcp.piv_tol = float(value)

    property piv_lim:
        """Number of pivot candidates that need to be considered, an `int` ≥1"""
        def __get__(self):
            return self._bfcp.piv_lim
        def __set__(self, value):
            self._bfcp.piv_lim = int(value)

    property suhl:
        """Whether to use Suhl heuristic, a `bool`"""
        def __get__(self):
            return self._bfcp.suhl
        def __set__(self, value):
            self._bfcp.suhl = bool(value)

    property eps_tol:
        """Tolerance below which numbers are replaced by zero, a |Real| number"""
        def __get__(self):
            return self._bfcp.eps_tol
        def __set__(self, value):
            self._bfcp.eps_tol = float(value)

    property nfs_max:
        """Maximal number of additional row-like factors, an `int`

        (Used only when *type* contains `'Forrest-Tomlin'`.)

        """
        def __get__(self):
            return self._bfcp.nfs_max
        def __set__(self, value):
            self._bfcp.nfs_max = int(value)

    property nrs_max:
        """Maximal number of additional row and columns, an `int`

        (Used only when *type* contains `'Bartels-Golub'` or `'Givens'`.)

        """
        def __get__(self):
            return self._bfcp.nrs_max
        def __set__(self, value):
            self._bfcp.nrs_max = int(value)


cdef class SimplexSolver(_Solver):
    """A simplex solver

    .. doctest:: SimplexSolver

        >>> p = MILProgram()
        >>> s = p.simplex
        >>> isinstance(s, SimplexSolver)
        True

    """

    cdef readonly Basis basis
    """The simplex basis, a `.Basis` object"""

    def __cinit__(self, program):
        self.basis = Basis(self._program)

    def solve(self, controls=SimplexControls(), exact=False):
        """Solve the linear program

        :param controls: the control parameters (uses defaults if omitted)
        :type controls: `.SimplexControls`
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
        cdef glpk.SimplexCP smcp = controls._smcp
        if exact:
            if meth2str[self._smcp.meth] is not 'primal':
                raise ValueError("Only primal simplex with exact arithmetic.")
            retcode = glpk.simplex_exact(self._problem, &smcp)
        else:
            if ((meth2str[smcp.meth] is not 'dual') and
                ((smcp.obj_ll > -DBL_MAX) or (smcp.obj_ul < +DBL_MAX))):
                 raise ValueError("Objective function limits only with " +
                                  "dual simplex.")
            retcode = glpk.simplex(self._problem, &smcp)
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
        :type varstraint: `.Variable` or `.Constraint`
        :returns: the value of *varstraint* for the current solution
        :rtype: `float` or `int`
        :raises TypeError: if varstraint is not `.Variable` or `.Constraint`

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
        :raises TypeError: if varstraint is not `.Variable` or `.Constraint`

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

    def basis_stuff(self, algorithm=None, status=None, warmup=False):
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
        :raises TypeError: if *status* keys are not `.Variable` or `.Constraint`
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
        :type varstraints: |Sequence| of `.Variable` or `.Constraint`
        :param fname: the name of the file to write to
        :type fname: `str`
        :raises Exception: if the current solution is not optimal
        :raises TypeError: if *varstraints* is not |Sequence| of `.Variable` or `.Constraint`
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
