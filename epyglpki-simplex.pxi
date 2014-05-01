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

    cdef readonly Basis basis
    """The simplex basis, a `.Basis` object"""

    def __cinit__(self, program):
        self.basis = Basis(self._program)

    def solve(self, controls=None):
        """Solve the linear program using floating point arithmetic

        :param controls: the control parameters (uses defaults if omitted)
        :type controls: `.SimplexControls`
        :returns: solution status; see `.status` for details,
            or `'obj_ll reached'` or `'obj_ul reached'` in case that happens
        :rtype: `str`
        :raises ValueError: if finite values are set for `obj_ll` or `obj_ul`
            while the `meth` control parameter is not `'dual'`
        :raises ValueError: if the basis is invalid
        :raises ValueError: if the basis matrix is singular or ill-conditioned
        :raises ValueError: if incorrect bounds are given
        :raises RuntimeError: in case of solver failure
        :raises StopIteration: if the iteration or time limit is exceeded
        :raises StopIteration: if the presolver detects the problem has no
            primal or dual feasible solution

        .. todo::

            Add doctest

        """
        cdef glpk.SimplexCP smcp
        if controls is None:
            glpk.init_smcp(&smcp)
        elif isinstance(controls, SimplexControls):
            smcp = controls._smcp
        else:
            raise TypeError("Controls parameter should be given as a " +
                            "'SimplexControls' object, not " +
                            type(controls).__name__)
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

    def solve_exactly(self, controls=None):
        """Solve the linear program using exact, integer arithmetic

        :param controls: the control parameters
            (only uses `it_lim` and `tm_lim`; uses defaults if omitted)
        :type controls: `.SimplexControls`
        :returns: solution status; see `.status` for details
        :rtype: `str`
        :raises ValueError: if the `meth` control parameter is not `'primal'`
            (only primal simplex is implemented in exact arithmetic)
        :raises ValueError: if the basis is invalid
        :raises ValueError: if the basis matrix is singular
        :raises ValueError: if incorrect bounds are given
        :raises RuntimeError: in case of solver failure
        :raises StopIteration: if the iteration or time limit is exceeded

        .. todo::

            Add doctest

        .. note::

            Computations in exact arithmetic are relatively time-consuming. So
            it is advised to first find an optimal basis with the
            `.SimplexSolver.solve` method and only then call this one, in which
            case only a few simplex iterations need to be performed in exact arithmetic.

        """
        cdef glpk.SimplexCP smcp
        if controls is None:
            glpk.init_smcp(&smcp)
        elif isinstance(controls, SimplexControls):
            smcp = controls._smcp
        else:
            raise TypeError("Controls parameter should be given as a " +
                            "'SimplexControls' object, not " +
                            type(controls).__name__)
        if meth2str[self._smcp.meth] is not 'primal':
            raise ValueError("Only primal simplex with exact arithmetic.")
        retcode = glpk.exact(self._problem, &smcp)
        if retcode is 0:
            return self.status()
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
            return solstat2str[glpk.get_status(self._problem)]

    property status_primal:
        """The current primal solution status, a `str`

        The possible values are `'undefined'`, `'infeasible'`, `'no feasible'`,
        or `'feasible'`.

        .. doctest:: SimplexSolver

            >>> s.status_primal
            'undefined'

        """
        def __get__(self):
            return solstat2str[glpk.get_prim_stat(self._problem)]

    property status_dual:
        """The current solution status, a `str`

        The possible values are `'undefined'`, `'infeasible'`, `'no feasible'`,
        or `'feasible'`.

        .. doctest:: SimplexSolver

            >>> s.status_dual
            'undefined'

        """
        def __get__(self):
            return solstat2str[glpk.get_dual_stat(self._problem)]

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
        ind = glpk.get_unbnd_ray(self._problem)
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


cdef class Basis(_Component):
    """The basis of a simplex solver

    .. doctest:: Basis

        >>> b = MILProgram().simplex.basis
        >>> isinstance(b, Basis)
        True

    .. todo::

        Add doctest showing use of `__len__` and `__getitem__`

    """

    def __len__(self):
        return len(self._program.constraints)

    def __getitem__(self, col):
        n = len(self.constraints)
        if (col <= 0) or (col > n):
            raise IndexError("The basis only has " + str(n) + "components, " +
                             "indexed from 1 to " + str(n) + ".")
        ind = glpk.get_bhead(self._problem, col)
        return self._program._from_varstraintind(ind)

    property controls:
        """The basis factorization controls, a `.FactorizationControls` object"""
        def __get__(self):
            return FactorizationControls(self._program)
        def __set__(self, controls):
            cdef glpk.BasFacCP bfcp = controls._bfcp
            glpk.set_bfcp(self._problem, &bfcp)
        def __del__(self):
            glpk.set_bfcp(self._problem, NULL)

    def generate(self, algorithm):
        """Generate an initial basis using an algorithm

        :type algorithm: `str`
        :param algorithm: an algorithm for generating a basis, chosen from

            * `'standard'`: sets all constraints as basic
            * `'advanced'`: sets as basic

              #. all non-fixed constraints
              #. as many non-fixed variables as possible, while preserving the
                 lower triangular structure of the basis matrix
              #. appropriate fixed constraints to complete the basis

            * `'Bixby'`: algorithm used by CPLEX, as discussed by Bixby_

        :raises ValueError: if *algorithm* is neither `'standard'`,
            `'advanced'`, nor `'Bixby'`

        .. todo::

            Add doctest

        .. _Bixby: http://dx.doi.org/10.1287/ijoc.4.3.267

        """
        if algorithm is 'standard':
            glpk.std_basis(self._problem)
        elif algorithm is 'advanced':
            glpk.adv_basis(self._problem, 0)
        elif algorithm is 'Bixby':
            glpk.cpx_basis(self._problem)
        else:
            raise ValueError(repr(algorithm)
                             + " is not a basis generation algorithm.")

    def warm_up(self):
        """‘Warm up’ the basis

        :raises ValueError: if the basis is invalid
        :raises ValueError: if the basis matrix is singular
        :raises ValueError: if the basis matrix is ill-conditioned

        A basis must be ‘warmed up’ to use `.SimplexSolver.solve` without
        presolving.

        .. todo::

            Add doctest

        .. note::

            After `.SimplexSolver.solve` has been run successfully, the basis
            is left in a valid state. So it is not necessary to run this method
            before, e.g., re-optimizating after only the objective has been
            changed.

        """
        retcode = glpk.warm_up(self._problem)
        if retcode is not 0:
            raise smretcode2error[retcode]
