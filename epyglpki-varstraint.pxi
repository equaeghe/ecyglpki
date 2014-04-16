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
        cdef int* ptr = &self._problem
        return hash((ptr, self._alias))

    @classmethod
    cdef _coeffscheck(cls, coeffs):
        if not isinstance(coeffs, collections.abc.Mapping):
            raise TypeError("Coefficients must be passed in a Mapping, not " +
                            type(coeffs).__name__)

    @classmethod
    cdef _coeff2real(cls, coeff)
        if isinstance(coeff, numbers.Real):
            return coeff
        else:
            raise TypeError("Coefficient values must be Real instead of " +
                            type(item[1]).__name__)


cdef class Variable(_Varstraint):

    cdef readonly Bounds bounds
    """The variable bounds, a `.Bounds` object"""

    def __cinit__(self, program):
        self._name = self._alias = self._program._generate_alias()
        col = len(self.program.variables) + 1
                        # +1 because variable not yet added to
                        # self._program.variables._varstraints at this point
        # TODO: deal with name properly when using _link
        glpk.set_col_name(self._problem, col, name2chars(self._name))
        bounds = Bounds(self)

    property _ind:
        """Return the column index"""
        def __get__(self):
            col = glpk.find_col(self._problem, name2chars(self._name))
            return None if col is 0 else col

    property _col_after_rows:
        """Return the variable index

        (GLPK sometimes indexes variables after constraints.)

        """
        def __get__(self):
            return len(self._problem._constraints) + self._col

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
            self._name = self._alias if name is '' else name
            glpk.set_col_name(self._problem, self._col, name2chars(self._name))
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
                >>> x.bounds(lower=0, upper=1)
                (0.0, 1.0)
                >>> x.kind
                'binary'
                >>> x.bounds(upper=3)
                (0.0, 3.0)
                >>> x.kind
                'integer'
                >>> x.kind = 'binary'
                >>> x.bounds()
                (0.0, 1.0)

        """
        def __get__(self):
            return varkind2str[glpk.get_col_kind(self._problem, self._col)]
        def __set__(self, kind):
            if kind in str2varkind:
                glpk.set_col_kind(self._problem, self._col, str2varkind[kind])
            else:
                raise ValueError("Kind must be 'continuous', 'integer', " +
                                 "or 'binary'.")

    cdef double _lower_bound(self):
        return glpk.get_col_lb(self._problem, self._col)

    cdef double _upper_bound(self):
        return glpk.get_col_ub(self._problem, self._col)

    cdef _set_bounds(int vartype, double lb, double ub):
        glpk.set_col_bnds(self._problem, self._col, vartype, lb, ub)

    property coeffs:
        """Nonzero coefficients, a |Mapping| of `.Constraint` to |Real|"""
        def __get__(self):
            k = glpk.get_mat_col(self._problem, self._ind, NULL, NULL)
            cdef int* rows =  <int*>glpk.alloc(1+k, sizeof(int))
            cdef double* vals =  <double*>glpk.alloc(1+k, sizeof(double))
            try:
                glpk.get_mat_col(self._problem, self._ind, rows, vals)
                coeffs = {self._problem.constraints[rows[i]]: vals[i]
                          for i in range(1, 1+k)}
            finally:
                glpk.free(vals)
                glpk.free(inds)
            return coeffs
        def __set__(self, coeffs):
            _Varstraint._coeffscheck(coeffs)
            k = len(coeffs)
            cdef int* rows =  <int*>glpk.alloc(1+k, sizeof(int))
            cdef double* vals =  <double*>glpk.alloc(1+k, sizeof(double))
            try:
                for i, item in enumerate(coeffs, start=1):
                    if isinstance(item[0], Constraint):
                        row = item[0]._ind
                    else:
                        row = self._program.constraints(item[0])
                    inds[i] = row
                    vals[i] = _Varstraint.coeff2real(item[1])
                glpk.set_mat_col(self._problem, self._ind, 0, inds, vals)
            finally:
                glpk.free(vals)
                glpk.free(inds)
        def __del__(self):
            glpk.set_mat_col(self._problem, self._ind, 0, NULL, NULL)

    def remove(self):
        """Remove the variable from the problem"""
        del self._problem.variables[self._name]


cdef class Constraint(_Varstraint):

    cdef readonly Bounds bounds
    """The constraint bounds, a `.Bounds` object"""

    def __cinit__(self, program):
        self._alias = self._program._generate_alias()
        row = len(self._program.constraints)
                        # +1 because variable not yet added to
                        # self._program.constraints._varstraints at this point
        # TODO: deal with name properly when using _link
        glpk.set_row_name(self._problem, row, name2chars(self._name))
        bounds = Bounds(self)

    property _ind:
        """Return the row index"""
        def __get__(self):
            row = glpk.find_row(self._problem, name2chars(self._name))
            return None if row is 0 else row

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
            self._name = self._alias if name is '' else name
            glpk.set_row_name(self._problem, self._row, name2chars(self._name))
        def __del__(self):
            self.name = self._alias

    cdef double _lower_bound(self):
        return glpk.get_row_lb(self._problem, self._row)

    cdef double _upper_bound(self):
        return glpk.get_row_ub(self._problem, self._row)

    cdef _set_bounds(int vartype, double lb, double ub):
        glpk.set_row_bnds(self._problem, self._row, vartype, lb, ub)

    property coeffs:
        """Nonzero coefficients, a |Mapping| of `.Variable` to |Real|"""
        def __get__(self):
            k = glpk.get_mat_row(self._problem, self._ind, NULL, NULL)
            cdef int* cols =  <int*>glpk.alloc(1+k, sizeof(int))
            cdef double* vals =  <double*>glpk.alloc(1+k, sizeof(double))
            try:
                glpk.get_mat_row(self._problem, self._ind, cols, vals)
                coeffs = {self._problem.variables[cols[i]]: vals[i]
                          for i in range(1, 1+k)}
            finally:
                glpk.free(vals)
                glpk.free(inds)
            return coeffs
        def __set__(self, coeffs):
            _Varstraint._coeffscheck(coeffs)
            k = len(coeffs)
            cdef int* cols =  <int*>glpk.alloc(1+k, sizeof(int))
            cdef double* vals =  <double*>glpk.alloc(1+k, sizeof(double))
            try:
                for i, item in enumerate(coeffs, start=1):
                    if isinstance(item[0], Variable):
                        col = item[0]._ind
                    else:
                        col = self._program.variables(item[0])
                    inds[i] = col
                    vals[i] = _Varstraint.coeff2real(item[1])
                glpk.set_mat_row(self._problem, self._ind, 0, inds, vals)
            finally:
                glpk.free(vals)
                glpk.free(inds)
        def __del__(self):
            glpk.set_mat_row(self._problem, self._ind, 0, NULL, NULL)

    def remove(self):
        """Remove the constraint from the problem"""
        del self._problem.constraints[self._name]


cdef class Bounds(_Component):

    cdef _Varstraint _varstraint

    def __cinit__(self, varstraint):
        self._varstraint = varstraint

    def __str__(self):
        return str((self.lower, self.upper))

    def __repr__(self):
        return type(self).__name__ + str(self)

    @classmethod
    cdef double _arg2double(cls, value):
        if value is None:
            return -DBL_MAX
        elif isinstance(value, numbers.Real):
            return value
        else:
            raise TypeError("Bound value must be 'None' or 'Real'.")

    def __call__(self, lower, upper):
        cdef double lb
        cdef double ub
        lb = Bounds._arg2double(lower)
        ub = Bounds._arg2double(upper)
        if lb > ub:
            raise ValueError("Lower bound must not dominate upper bound.")
        vartype = pair2vartype[(lower, upper)]
        if vartype == glpk.DB:
            if lb == ub:
                vartype = glpk.FX
        self._varstraint._set_bounds(vartype, lb, ub)

    property lower:
        """The lower bound"""
        def __get__(self):
            lb = self._varstraint._lower_bound()
            return None if lb == -DBL_MAX else lb
        def __set__(self, value):
            self(value, self.upper)
        def __del__(self):
            self.lower = None

    property upper:
        """The upper bound"""
        def __get__(self):
            ub = self._varstraint._upper_bound()
            return None if ub == +DBL_MAX else ub
        def __set__(self, value):
            self(self.lower, value)
        def __del__(self):
            self.upper = None
