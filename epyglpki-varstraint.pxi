# epyglpki-solvers.pxi: Cython/Python interface for GLPK variables/constraints

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


cdef class _Varstraint(_Component):

    cdef str _alias  # unique identifier; invariant after initialization
    cdef str _name  # always identical to
                    # glpk.get_col/row_name(self._problem, self._col/row)

    def __hash__(self):
        return hash((id(self._program), self._alias))


cdef class Variable(_Varstraint):
    """One of the problem's variables

    .. doctest:: Variable

        >>> p = MILProgram()
        >>> x = p.variables.add()
        >>> isinstance(x, Variable)
        True

    """

    cdef readonly Bounds bounds
    """The variable bounds, a `.Bounds` object"""
    cdef readonly SimplexSolution simplex
    """The variable's simplex solution, a `.SimplexSolution` object"""
    cdef readonly IPointSolution ipoint
    """The variable's interior point solution, a `.IPointSolution` object"""
    cdef readonly IntOptSolution intopt
    """The variable's integer optimization solution, a `.IntOptSolution` object"""

    def __cinit__(self, program):
        self._alias = program._generate_alias()
        col = len(program.variables) + 1
                        # + 1 because variable not yet added to
                        # self._program.variables._varstraints at this point
        cdef const char* chars = glpk.get_col_name(self._problem, col)
        if chars is NULL:
            self._name = self._alias
            glpk.set_col_name(self._problem, col, name2chars(self._name))
        else:
            self._name = chars.decode()
        self.bounds = Bounds(self)
        self.simplex = SimplexSolution(self)
        self.ipoint = IPointSolution(self)
        self.intopt = IntOptSolution(self)

    property _ind:
        """Return the column index"""
        def __get__(self):
            col = glpk.find_col(self._problem, name2chars(self._name))
            if col is 0:
                raise IndexError("Variable does not belong to this program.")
            return col

    property _varstraintind:
        """Return the variable index

        (GLPK sometimes indexes variables after constraints.)

        """
        def __get__(self):
            return len(self._program.constraints) + self._ind

    property name:
        """The variable name, a `str` of ≤255 bytes UTF-8 encoded

        .. doctest:: Variable

            >>> x.name  # the GLPK default
            ''
            >>> x.name = 'Stake'
            >>> x.name
            'Stake'
            >>> del x.name  # clear name
            >>> x.name
            ''

        """
        def __get__(self):
            return '' if self._name is self._alias else self._name
        def __set__(self, name):
            col = self._ind  # get col with old name
            self._name = self._alias if name is '' else name
            glpk.set_col_name(self._problem, col, name2chars(self._name))
        def __del__(self):
            self.name = self._alias

    property kind:
        """The variable kind, either `'continuous'`, `'integer'`, or `'binary'`

        .. doctest:: Variable

            >>> x.kind  # the GLPK default
            'continuous'
            >>> x.kind = 'integer'
            >>> x.kind
            'integer'

        .. note::

            A variable has `'binary'` kind if and only if it is an integer
            variable with lower bound zero and upper bound one:

            .. doctest:: Variable

                >>> x.kind
                'integer'
                >>> x.bounds(0, 1)
                >>> x.bounds.lower, x.bounds.upper
                (0.0, 1.0)
                >>> x.kind
                'binary'
                >>> x.bounds.upper = 3
                >>> x.bounds.lower, x.bounds.upper
                (0.0, 3.0)
                >>> x.kind
                'integer'
                >>> x.kind = 'binary'
                >>> x.bounds.lower, x.bounds.upper
                (0.0, 1.0)

        """
        def __get__(self):
            return varkind2str[glpk.get_col_kind(self._problem, self._ind)]
        def __set__(self, kind):
            if kind in str2varkind:
                glpk.set_col_kind(self._problem, self._ind, str2varkind[kind])
            else:
                raise ValueError("Kind must be 'continuous', 'integer', " +
                                 "or 'binary'.")

    def _lower_bound(self):
        return glpk.get_col_lb(self._problem, self._ind)

    def _upper_bound(self):
        return glpk.get_col_ub(self._problem, self._ind)

    def _set_bounds(self, int vartype, double lb, double ub):
        glpk.set_col_bnds(self._problem, self._ind, vartype, lb, ub)

    def _vartype(self):
        return glpk.get_col_type(self._problem, self._ind)

    property coeffs:
        """Nonzero coefficients, a |Mapping| of `.Constraint` to |Real|"""
        def __get__(self):
            k = glpk.get_mat_col(self._problem, self._ind, NULL, NULL)
            cdef int* rows =  <int*>glpk.alloc(1+k, sizeof(int))
            cdef double* vals =  <double*>glpk.alloc(1+k, sizeof(double))
            try:
                glpk.get_mat_col(self._problem, self._ind, rows, vals)
                coeffs = {self._program.constraints._from_ind(rows[i]): vals[i]
                          for i in range(1, 1+k)}
            finally:
                glpk.free(rows)
                glpk.free(vals)
            return coeffs
        def __set__(self, coeffs):
            coeffscheck(coeffs)
            k = len(coeffs)
            cdef int* rows =  <int*>glpk.alloc(1+k, sizeof(int))
            cdef double* vals =  <double*>glpk.alloc(1+k, sizeof(double))
            try:
                for i, item in enumerate(coeffs, start=1):
                    if isinstance(item[0], Constraint):
                        row = item[0]._ind
                    else:  #  assume name
                        row = self._program.constraints._find_ind(item[0])
                    rows[i] = row
                    vals[i] = item[1]
                glpk.set_mat_col(self._problem, self._ind, 0, rows, vals)
            finally:
                glpk.free(rows)
                glpk.free(vals)
        def __del__(self):
            glpk.set_mat_col(self._problem, self._ind, 0, NULL, NULL)

    property weight:
        """The weight of the variable in the problem's objective

        .. doctest:: Variable

            >>> x.weight  # the GLPK default
            0.0
            >>> x.weight = -3.5
            >>> x.weight
            -3.5

        """
        def __get__(self):
            cdef double weight
            weight = glpk.get_obj_coef(self._problem, self._ind)
            return weight
        def __set__(self, weight):
            if isinstance(weight, numbers.Real):
                glpk.set_obj_coef(self._problem, self._ind, weight)
            else:
                raise TypeError("Coefficient values must be 'numbers.Real' " +
                                "instead of '" + type(weight).__name__ + "'.")

    property scale:
        """The constraint matrix scaling factor for the variable's column

        .. doctest:: Variable

            >>> x.scale  # the GLPK default
            1.0
            >>> x.scale = 1/5
            >>> x.scale
            0.2

        """
        def __get__(self):
            cdef double factor = glpk.get_col_sf(self._problem, self._ind)
            return factor
        def __set__(self, factor):
            glpk.set_col_sf(self._problem, self._ind, factor)

    def remove(self):
        """Remove the variable from the problem"""
        del self._program.variables[self._name]

    def _mip_val(self):
        return glpk.mip_col_val(self._problem, self._ind)

    def _ipt_prim(self):
        return glpk.ipt_col_prim(self._problem, self._ind)

    def _ipt_dual(self):
        return glpk.ipt_col_dual(self._problem, self._ind)

    def _sm_prim(self):
        return glpk.sm_col_prim(self._problem, self._ind)

    def _sm_dual(self):
        return glpk.sm_col_dual(self._problem, self._ind)

    def _get_stat(self):
        return glpk.get_col_stat(self._problem, self._ind)

    def _set_stat(self, varstat):
        glpk.set_col_stat(self._problem, self._ind, varstat)


cdef class Constraint(_Varstraint):
    """One of the problem's constraints

    .. doctest:: Constraint

        >>> p = MILProgram()
        >>> c = p.constraints.add()
        >>> isinstance(c, Constraint)
        True

    """

    cdef readonly Bounds bounds
    """The constraint bounds, a `.Bounds` object"""
    cdef readonly SimplexSolution simplex
    """The constraint's simplex solution, a `.SimplexSolution` object"""
    cdef readonly IPointSolution ipoint
    """The constraint's interior point solution, a `.IPointSolution` object"""
    cdef readonly IntOptSolution intopt
    """The constraint's integer optimization solution, a `.IntOptSolution` object"""

    def __cinit__(self, program):
        self._alias = program._generate_alias()
        row = len(program.constraints) + 1
                        # + 1 because variable not yet added to
                        # self._program.constraints._varstraints at this point
        cdef const char* chars = glpk.get_row_name(self._problem, row)
        if chars is NULL:
            self._name = self._alias
            glpk.set_row_name(self._problem, row, name2chars(self._name))
        else:
            self._name = chars.decode()
        self.bounds = Bounds(self)
        self.simplex = SimplexSolution(self)
        self.ipoint = IPointSolution(self)
        self.intopt = IntOptSolution(self)

    property _ind:
        """Return the row index"""
        def __get__(self):
            row = glpk.find_row(self._problem, name2chars(self._name))
            if row is 0:
                raise IndexError("Constraint does not belong to this program.")
            return row

    property _varstraintind:
        """Return the variable index

        (GLPK sometimes indexes structural variables after auxiliary ones,
        i.e., constraints.)

        """
        def __get__(self):
            return self._ind

    property name:
        """The constraint name, a `str` of ≤255 bytes UTF-8 encoded

        .. doctest:: Constraint

            >>> c.name  # the GLPK default
            ''
            >>> c.name = 'Budget'
            >>> c.name
            'Budget'
            >>> del c.name  # clear name
            >>> c.name
            ''

        """
        def __get__(self):
            return '' if self._name is self._alias else self._name
        def __set__(self, name):
            row = self._ind  # get row with old name
            self._name = self._alias if name is '' else name
            glpk.set_row_name(self._problem, row, name2chars(self._name))
        def __del__(self):
            self.name = self._alias

    def _lower_bound(self):
        return glpk.get_row_lb(self._problem, self._ind)

    def _upper_bound(self):
        return glpk.get_row_ub(self._problem, self._ind)

    def _set_bounds(self, int vartype, double lb, double ub):
        glpk.set_row_bnds(self._problem, self._ind, vartype, lb, ub)

    def _vartype(self):
        return glpk.get_row_type(self._problem, self._ind)

    property coeffs:
        """Nonzero coefficients, a |Mapping| of `.Variable` to |Real|"""
        def __get__(self):
            k = glpk.get_mat_row(self._problem, self._ind, NULL, NULL)
            cdef int* cols =  <int*>glpk.alloc(1+k, sizeof(int))
            cdef double* vals =  <double*>glpk.alloc(1+k, sizeof(double))
            try:
                glpk.get_mat_row(self._problem, self._ind, cols, vals)
                coeffs = {self._program.variables._from_ind(cols[i]): vals[i]
                          for i in range(1, 1+k)}
            finally:
                glpk.free(cols)
                glpk.free(vals)
            return coeffs
        def __set__(self, coeffs):
            coeffscheck(coeffs)
            k = len(coeffs)
            cdef int* cols =  <int*>glpk.alloc(1+k, sizeof(int))
            cdef double* vals =  <double*>glpk.alloc(1+k, sizeof(double))
            try:
                for i, item in enumerate(coeffs, start=1):
                    if isinstance(item[0], Variable):
                        col = item[0]._ind
                    else:  #  assume name
                        col = self._program.variables._find_ind(item[0])
                    cols[i] = col
                    vals[i] = item[1]
                glpk.set_mat_row(self._problem, self._ind, 0, cols, vals)
            finally:
                glpk.free(cols)
                glpk.free(vals)
        def __del__(self):
            glpk.set_mat_row(self._problem, self._ind, 0, NULL, NULL)

    property scale:
        """The constraint matrix scaling factor for the constraint's row

        .. doctest:: Constraint

            >>> c.scale  # the GLPK default
            1.0
            >>> c.scale = 3
            >>> c.scale
            3.0

        """
        def __get__(self):
            cdef double factor = glpk.get_row_sf(self._problem, self._ind)
            return factor
        def __set__(self, factor):
            glpk.set_row_sf(self._problem, self._ind, factor)

    def remove(self):
        """Remove the constraint from the problem"""
        del self._program.constraints[self._name]

    def _mip_val(self):
        return glpk.mip_row_val(self._problem, self._ind)

    def _ipt_prim(self):
        return glpk.ipt_row_prim(self._problem, self._ind)

    def _ipt_dual(self):
        return glpk.ipt_row_dual(self._problem, self._ind)

    def _sm_prim(self):
        return glpk.sm_row_prim(self._problem, self._ind)

    def _sm_dual(self):
        return glpk.sm_row_dual(self._problem, self._ind)

    def _get_stat(self):
        return glpk.get_row_stat(self._problem, self._ind)

    def _set_stat(self, varstat):
        glpk.set_row_stat(self._problem, self._ind, varstat)


cdef class Bounds:
    """The bounds of a variable or constraint

    .. doctest:: Bounds

        >>> b = MILProgram().variables.add().bounds
        >>> isinstance(b, Bounds)
        True
        >>> b.lower, b.upper  # the GLPK default for variables
        (0.0, 0.0)

    .. doctest:: Bounds

        >>> b = MILProgram().constraints.add().bounds
        >>> isinstance(b, Bounds)
        True
        >>> b.lower, b.upper  # the GLPK default for constraints
        (-inf, inf)

    Changing both lower and upper bounds can be done by passing them as an
    argument to the `.Bounds` object, using `None` for no bound:

    .. doctest:: Bounds

        >>> b(0, 4)
        >>> b.lower, b.upper
        (0.0, 4.0)
        >>> b(None, None)
        >>> b.lower, b.upper
        (-inf, inf)

    The lower and upper bounds can retrieved also be changed using the
    attributes below:

    """

    cdef _Varstraint _varstraint

    def __cinit__(self, varstraint):
        self._varstraint = varstraint

    def __str__(self):
        return str((self.lower, self.upper))

    def __repr__(self):
        return type(self).__name__ + str(self)

    def __call__(self, lower, upper):
        cdef double lb
        cdef double ub
        if isinstance(lower, numbers.Real):
            lb = lower
            if lb <= -DBL_MAX:
                lb = -DBL_MAX
                lower = None
        elif lower is None:
            lb = -DBL_MAX
        else:
            raise TypeError("Lower bound value must be 'None' or 'Real'.")
        if isinstance(upper, numbers.Real):
            ub = upper
            if ub >= +DBL_MAX:
                ub = +DBL_MAX
                upper = None
        elif upper is None:
            ub = +DBL_MAX
        else:
            raise TypeError("Upper bound value must be 'None' or 'Real'.")
        if lb > ub:
            raise ValueError("Lower bound must not dominate upper bound.")
        vartype = pair2vartype[(lower is None, upper is None)]
        if vartype is glpk.DB:
            if lb == ub:
                vartype = glpk.FX
        self._varstraint._set_bounds(vartype, lb, ub)

    property lower:
        """The lower bound

        .. doctest:: Bounds

            >>> b.lower = -1
            >>> b.lower
            -1.0
            >>> del b.lower  # remove the lower bound
            >>> b.lower
            -inf

        """
        def __get__(self):
            cdef double lb = self._varstraint._lower_bound()
            return -float('inf') if lb == -DBL_MAX else lb
        def __set__(self, value):
            self(value, self.upper)
        def __del__(self):
            self.lower = None

    property upper:
        """The upper bound

        .. doctest:: Bounds

            >>> b.upper = 5/2
            >>> b.upper
            2.5
            >>> del b.upper  # remove the upper bound
            >>> b.upper
            inf

        """
        def __get__(self):
            cdef double ub = self._varstraint._upper_bound()
            return float('inf') if ub == +DBL_MAX else ub
        def __set__(self, value):
            self(self.lower, value)
        def __del__(self):
            self.upper = None

    property type:
        """The type of bounds, a `str`

        The possible values are:

        * `'free'`: has no bounds
        * `'dominating'`: has a lower bound
        * `'dominated'`: has an upper bound
        * `'bounded'`: has both a lower and an upper bound
        * `'fixed'`: has identical lower and upper bounds

        .. doctest:: Bounds

            >>> b.lower, b.upper
            (-inf, inf)
            >>> b.type
            'free'
            >>> b.lower = 0
            >>> b.type
            'dominating'
            >>> b.upper = 1
            >>> b.type
            'bounded'

        """
        def __get__(self):
            return vartype2str[self._varstraint._vartype()]


cdef class SimplexSolution:
    """The solution produced by the simplex solver

    .. doctest:: SimplexSolution

        >>> s = MILProgram().variables.add().simplex
        >>> isinstance(s, SimplexSolution)
        True
        >>> s.primal, s.dual, s.status  # the GLPK default before solving
        (0.0, 0.0, 'fixed')

    .. doctest:: SimplexSolution

        >>> c = MILProgram().constraints.add().simplex
        >>> isinstance(s, SimplexSolution)
        True
        >>> s.primal, s.dual, s.status  # the GLPK default before solving
        (0.0, 0.0, 'fixed')

    """

    cdef _Varstraint _varstraint

    def __cinit__(self, varstraint):
        self._varstraint = varstraint

    property primal:
        """The primal solution value, a |Real| number"""
        def __get__(self):
            cdef double val = self._varstraint._sm_prim()
            return val

    property dual:
        """The dual solution value, a |Real| number"""
        def __get__(self):
            cdef double val = self._varstraint._sm_dual()
            return val

    property status:
        """The basic status, a `str`

        The possible values are:

        * `'basic'`: basic
        * `'lower'`: non-basic with active lower bound
        * `'upper'`: non-basic with active upper bound
        * `'free'`: non-basic free (unbounded)
        * `'fixed'`: non-basic fixed

        .. doctest:: SimplexSolution

            >>> s.status = 'basic'
            >>> s.status
            'basic'

        .. note::

            A basis is valid if the basis matrix is non-singular, which implies
            that the number of basic variables and constraints is equal to the
            total number of constraints.

        """
        def __get__(self):
            return varstat2str[self._varstraint._get_stat()]
        def __set__(self, status):
            varstat = str2varstat[status]
            self._varstraint._set_stat(varstat)


cdef class IPointSolution:
    """The solution produced by the interior point solver

    .. doctest:: IPointSolution

        >>> s = MILProgram().variables.add().ipoint
        >>> isinstance(s, IPointSolution)
        True
        >>> s.primal, s.dual  # the GLPK default before solving
        (0.0, 0.0)

    .. doctest:: IPointSolution

        >>> c = MILProgram().constraints.add().ipoint
        >>> isinstance(s, IPointSolution)
        True
        >>> s.primal, s.dual  # the GLPK default before solving
        (0.0, 0.0)

    """

    cdef _Varstraint _varstraint

    def __cinit__(self, varstraint):
        self._varstraint = varstraint

    property primal:
        """The primal solution value, a |Real| number"""
        def __get__(self):
            cdef double val = self._varstraint._ipt_prim()
            return val

    property dual:
        """The dual solution value, a |Real| number"""
        def __get__(self):
            cdef double val = self._varstraint._ipt_dual()
            return val


cdef class IntOptSolution:
    """The solution produced by the integer optimization solver

    .. doctest:: IntOptSolution

        >>> s = MILProgram().variables.add().intopt
        >>> isinstance(s, IntOptSolution)
        True
        >>> s.value  # the GLPK default before solving
        0.0

    .. doctest:: IntOptSolution

        >>> c = MILProgram().constraints.add().intopt
        >>> isinstance(s, IntOptSolution)
        True
        >>> s.value  # the GLPK default before solving
        0.0

    """

    cdef _Varstraint _varstraint

    def __cinit__(self, varstraint):
        self._varstraint = varstraint

    property value:
        """The solution value, a |Real| number"""
        def __get__(self):
            cdef double val = self._varstraint._mip_val()
            return val
