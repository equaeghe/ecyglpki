# epyglpki-components.pxi: Cython/Python interface for GLPK problem components

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


cdef class MILProgramOld:

    def _col(self, variable, alternate=False):
        """Return the column index of a Variable"""
        try:
            col = 1 + self._variables.index(variable)
            if alternate: # GLPK sometimes indexes variables after constraints
                col += len(self._constraints)
            return col
                # GLPK indices start at 1
        except ValueError:
            raise IndexError("This is possibly a zombie; kill it using 'del'.")

    def _variable(self, col, alternate=False):
        """Return the Variable corresponding to a column index"""
        if alternate: # GLPK sometimes indexes variables after constraints
            rows = len(self._constraints)
            if col <= rows:
                raise IndexError("Alternate column index cannot be smaller " +
                                 "than total number of rows")
            else:
                col -= rows
        return None if col is 0 else self._variables[col-1]

    def _row(self, constraint):
        """Return the row index of a Constraint"""
        try:
            return 1 + self._constraints.index(constraint)
                # GLPK indices start at 1
        except ValueError:
            raise IndexError("This is possibly a zombie; kill it using 'del'.")

    def _constraint(self, row):
        """Return the Constraint corresponding to a row index"""
        return None if row is 0 else self._constraints[row-1]

    def _ind(self, varstraint, alternate=False):
        """Return the column/row index of a Variable/Constraint"""
        if isinstance(varstraint, Variable):
            return self._col(varstraint, alternate)
        elif isinstance(varstraint, Constraint):
            return self._row(varstraint)
        else:
            raise TypeError("No index available for this object type.")

    def _varstraint(self, ind):
        """Return the Variable/Constraint corresponding to an alternate index"""
        if ind > len(self._constraints):
            return self._variable(ind, alternate=True)
        else:
            return self._constraint(ind)

    def _del_varstraint(self, varstraint):
        """Remove a Variable or Constraint from the problem"""
        if isinstance(varstraint, Variable):
            self._variables.remove(varstraint)
        elif isinstance(varstraint, Constraint):
            self._constraints.remove(varstraint)
        else:
            raise TypeError("No index available for this object type.")

    def add_variable(self, coeffs={}, lower_bound=False, upper_bound=False,
                     kind=None, name=None):
        """Add and obtain new variable object

        :param coeffs: set variable coefficients; see `.Variable.coeffs`
        :param lower_bound: set variable lower bound;
            see `.Varstraint.bounds`, parameter *lower*
        :param upper_bound: set variable upper bound;
            see `.Varstraint.bounds`, parameter *upper*
        :param kind: set variable kind; see `.Variable.kind`
        :param name: set variable name; see `.Variable.name`
        :returns: variable object
        :rtype: `.Variable`

        .. doctest:: MILProgram.add_variable

            >>> p = MILProgram()
            >>> x = p.add_variable()
            >>> x
            <epyglpki.Variable object at 0x...>

        """
        variable = Variable(self)
        self._variables.append(variable)
        assert len(self._variables) is glpk.get_num_cols(self._problem)
        variable.coeffs(None if not coeffs else coeffs)
        variable.bounds(lower_bound, upper_bound)
        if kind is not None:
            variable.kind = kind
        if name is not None:
            variable.name = name
        return variable

    def variables(self):
        """A list of the problem's variables

        :returns: a list of the problem's variables
        :rtype: `list` of `.Variable`

        .. doctest:: MILProgram.variables

            >>> p = MILProgram()
            >>> x = p.add_variable()
            >>> p.variables()
            [<epyglpki.Variable object at 0x...>]
            >>> y = p.add_variable()
            >>> v = p.variables()
            >>> (x in v) and (y in v)
            True

        """
        return self._variables

    def add_constraint(self, coeffs={}, lower_bound=False, upper_bound=False,
                       name=None):
        """Add and obtain new constraint object

        :param coeffs: set constraint coefficients; see `.Constraint.coeffs`
        :param lower_bound: set constraint lower bound;
            see `.Varstraint.bounds`, parameter *lower*
        :param upper_bound: set constraint upper bound;
            see `.Varstraint.bounds`, parameter *upper*
        :param name: set constraint name; see `.Constraint.name`
        :returns: constraint object
        :rtype: `.Constraint`

        .. doctest:: MILProgram.add_constraint

            >>> p = MILProgram()
            >>> c = p.add_constraint()
            >>> c
            <epyglpki.Constraint object at 0x...>

        """
        constraint = Constraint(self)
        self._constraints.append(constraint)
        assert len(self._constraints) is glpk.get_num_rows(self._problem)
        constraint.coeffs(None if not coeffs else coeffs)
        constraint.bounds(lower_bound, upper_bound)
        if name is not None:
            constraint.name = name
        return constraint

    def constraints(self):
        """Return a list of the problem's constraints

        :returns: a list of the problem's constraints
        :rtype: `list` of `.Constraint`

        .. doctest:: MILProgram.constraints

            >>> p = MILProgram()
            >>> c = p.add_constraint()
            >>> p.constraints()
            [<epyglpki.Constraint object at 0x...>]
            >>> d = p.add_constraint()
            >>> w = p.constraints()
            >>> (c in w) and (d in w)
            True

        """
        return self._constraints


cdef class Varstraint(_Component):
    """One of the program's variables or constraints

    Any constraint :math:`a\leq c'x\leq b` can be represented as a combination
    of an ‘auxiliary’ variable :math:`z_c`, an equality constraint
    :math:`z_c=c'x` linking that auxiliary variable to the (vector of)
    ‘structural’ variables :math:`x`, and bounds :math:`a\leq z_c\leq b` on
    that auxiliary variable. Therefore it is useful to have this class, which
    contains the methods common to both `.Variable` and `.Constraint` classes.

    .. doctest:: Varstraint

        >>> p = MILProgram()
        >>> x = p.add_variable()
        >>> isinstance(x, Varstraint)
        True
        >>> c = p.add_constraint()
        >>> isinstance(c, Varstraint)
        True

    """

    cdef int _unique_id

    def __cinit__(self, program):
        self._unique_id = self._program._generate_unique_id()

    def __hash__(self):
        return self._unique_id

    def __str__(self):
        return str(self._unique_id) + ':' + self.name

    #  how to really del?
    def remove(self):
        """Remove the varstraint from the problem

        .. doctest:: Varstraint

            >>> x in p.variables()
            True
            >>> x.remove()
            >>> x in p.variables()
            False

        .. doctest:: Varstraint

            >>> c in p.constraints()
            True
            >>> c.remove()
            >>> c in p.constraints()
            False

        .. note::

            Removing a varstraint from a problem object does not delete the
            referencing `.Varstraint` objects, which in some sense become
            ‘zombie’ objects; they should best be deleted manually:

            .. doctest:: Varstraint

                >>> c in p.constraints()
                False
                >>> c
                <epyglpki.Constraint object at 0x...>
                >>> c.name # doctest: +SKIP
                Traceback (most recent call last):
                  ...
                ValueError: <epyglpki.Constraint object at 0x...> is not in list
                <BLANKLINE>
                During handling of the above exception, another exception occurred:
                <BLANKLINE>
                Traceback (most recent call last):
                  ...
                IndexError: This is possibly a zombie; kill it using 'del'.
                >>> del c
                >>> c
                Traceback (most recent call last):
                  ...
                NameError: name 'c' is not defined

        """
        self._remove()
        self._program._del_varstraint(self)

    def bounds(self, lower=None, upper=None):
        """Change or retrieve varstraint bounds

        :param lower: the varstraint's lower bound
            (`False` to remove bound; omit for retrieval only)
        :type lower: |Real| or `False`
        :param upper: the varstraint's upper bound
            (`False` to remove bound; omit for retrieval only)
        :type upper: |Real| or `False`
        :returns: the varstraint's bounds
        :rtype: length-2 `tuple` of `float` (or `False`)
        :raises TypeError: if *lower* or *upper* is not |Real| or `False`
        :raises ValueError: if *lower* is larger than *upper*

        .. doctest:: Varstraint

            >>> x.bounds()
            (False, False)
            >>> x.bounds(lower=0, upper=5.5)
            (0.0, 5.5)
            >>> x.bounds(upper=False)
            (0.0, False)

        .. doctest:: Varstraint

            >>> c.bounds()
            (False, False)
            >>> c.bounds(lower=-1/2, upper=1/2)
            (-0.5, 0.5)
            >>> c.bounds(lower=False)
            (False, 0.5)


        """
        cdef double lb
        cdef double ub
        ind = self._program._ind(self)
        if lower is False:
            lb = -DBL_MAX
        else:
            if lower is None:
                lb = self._get_lb(ind)
            elif isinstance(lower, numbers.Real):
                lb = lower
            else:
                raise TypeError("Lower bound must be real numbers or 'False'.")
            lower = False if lb == -DBL_MAX else True
        if upper is False:
            ub = +DBL_MAX
        else:
            if upper is None:
                ub = self._get_ub(ind)
            elif isinstance(upper, numbers.Real):
                ub = upper
            else:
                raise TypeError("Upper bound must be real numbers or 'False'.")
            upper = False if ub == +DBL_MAX else True
        if lb > ub:
            raise ValueError("Lower bound must not dominate upper bound.")
        vartype = pair2vartype[(lower, upper)]
        if vartype == glpk.DB:
            if lb == ub:
                vartype = glpk.FX
        self._set_bnds(ind, vartype, lb, ub)
        return (lb if lower else False, ub if upper else False)

    cdef _coeffs(self,
                 int (*get_function)(glpk.ProbObj*, int, int[], double[]),
                 void (*set_function)(glpk.ProbObj*, int, int,
                                      int[], double[]),
                 argtypename, varstraints, coeffs=None):
        ind = self._program._ind(self)
        if coeffs is None:
            length = get_function(self._problem, ind, NULL, NULL)
        elif isinstance(coeffs, collections.abc.Mapping):
            length = len(coeffs)
        else:
            raise TypeError("Coefficients must be given using a " +
                            "collections.abc.Mapping.")
        cdef double* vals =  <double*>glpk.alloc(1+length, sizeof(double))
        cdef int* inds =  <int*>glpk.alloc(1+length, sizeof(int))
        try:
            if coeffs is not None:
                if length is 0:
                    set_function(self._problem, ind, length, NULL, NULL)
                else:
                    for other_ind, item in enumerate(coeffs.items(), start=1):
                        val = vals[other_ind] = item[1]
                        if not isinstance(val, numbers.Real):
                            raise TypeError("Coefficient values must be " +
                                            "'numbers.Real' instead of '" +
                                            type(val).__name__ + "'.")
                        if not isinstance(item[0], type(self)):
                            inds[other_ind] = self._program._ind(item[0])
                        else:
                            raise TypeError("Coefficient keys must be '" +
                                            argtypename + "' instead of '" +
                                            type(item[0]).__name__ +"'.")
                    set_function(self._problem, ind, length, inds, vals)
            length = get_function(self._problem, ind, inds, vals)
            coeffs = {}
            for other_ind in range(1, 1+length):
                coeffs[varstraints[inds[other_ind]-1]] = vals[other_ind]
        finally:
            glpk.free(vals)
            glpk.free(inds)
        return coeffs


cdef class Variable(Varstraint):
    """One of the problem's variables

    .. doctest:: Variable

        >>> p = MILProgram()
        >>> x = p.add_variable()
        >>> isinstance(x, Variable)
        True

    """

    def __cinit__(self, program):
        glpk.add_cols(self._problem, 1)

    def _remove(self):
        cdef int ind[2]
        ind[1] = self._program._col(self)
        glpk.del_cols(self._problem, 1, ind)

    def _get_lb(self, int ind):
        return glpk.get_col_lb(self._problem, ind)

    def _get_ub(self, int ind):
        return glpk.get_col_ub(self._problem, ind)

    def _set_bnds(self, int ind, int vartype, double lb, double ub):
        glpk.set_col_bnds(self._problem, ind, vartype, lb, ub)

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
            col = self._program._col(self)
            return varkind2str[glpk.get_col_kind(self._problem, col)]
        def __set__(self, kind):
            col = self._program._col(self)
            if kind in str2varkind:
                glpk.set_col_kind(self._problem, col, str2varkind[kind])
            else:
                raise ValueError("Kind must be 'continuous', 'integer', " +
                                 "or 'binary'.")

    def coeffs(self, coeffs=None):
        """Replace or retrieve variable coefficients (constraint matrix column)

        :param coeffs: the mapping with the new coefficients
            (``{}`` to set all coefficients to 0; omit for retrieval only)
        :type coeffs: |Mapping| of `.Constraint` to |Real|
        :returns: the coefficient mapping, which only contains nonzero
            coefficients
        :rtype: `dict` of `.Constraint` to `float`
        :raises TypeError: if *coeffs* is not |Mapping|
        :raises TypeError: if a coefficient key is not `.Variable`
        :raises TypeError: if a coefficient value is not |Real|

        .. doctest:: Variable.coeffs

            >>> p = MILProgram()
            >>> c = p.add_constraint()
            >>> d = p.add_constraint()
            >>> x = p.add_variable()
            >>> x.coeffs()
            {}
            >>> x.coeffs({c: 10/9, d: 0})
            {<epyglpki.Constraint object at 0x...>: 1.1111...}
            >>> x.coeffs({})
            {}

        """
        return self._coeffs(glpk.get_mat_col, glpk.set_mat_col,
                            Constraint.__name__, self._program._constraints,
                            coeffs)

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
            col = self._program._col(self)
            cdef char* chars = glpk.get_col_name(self._problem, col)
            return '' if chars is NULL else chars.decode()
        def __set__(self, name):
            col = self._program._col(self)
            glpk.set_col_name(self._problem, col, name2chars(name))
        def __del__(self):
            col = self._program._col(self)
            glpk.set_col_name(self._problem, col, NULL)


cdef class Constraint(Varstraint):
    """One of the problem's constraints

    .. doctest:: Constraint

        >>> p = MILProgram()
        >>> c = p.add_constraint()
        >>> isinstance(c, Constraint)
        True

    """

    def __cinit__(self, program):
        glpk.add_rows(self._problem, 1)

    def _remove(self):
        cdef int ind[2]
        ind[1] = self._program._row(self)
        glpk.del_rows(self._problem, 1, ind)

    def _get_lb(self, int ind):
        return glpk.get_row_lb(self._problem, ind)

    def _get_ub(self, int ind):
        return glpk.get_row_ub(self._problem, ind)

    def _set_bnds(self, int ind, int vartype, double lb, double ub):
        glpk.set_row_bnds(self._problem, ind, vartype, lb, ub)

    def coeffs(self, coeffs=None):
        """Replace or retrieve constraint coefficients (constraint matrix row)

        :param coeffs: the mapping with the new coefficients
            (``{}`` to set all coefficients to 0; omit for retrieval only)
        :type coeffs: |Mapping| of `.Variable` to |Real|
        :returns: the coefficient mapping, which only contains nonzero
            coefficients
        :rtype: `dict` of `.Variable` to `float`
        :raises TypeError: if *coeffs* is not |Mapping|
        :raises TypeError: if a coefficient key is not `.Variable`
        :raises TypeError: if a coefficient value is not |Real|

        .. doctest:: Constraint.coeffs

            >>> p = MILProgram()
            >>> x = p.add_variable()
            >>> y = p.add_variable()
            >>> c = p.add_constraint()
            >>> c.coeffs()
            {}
            >>> c.coeffs({x: .5, y: 0})
            {<epyglpki.Variable object at 0x...>: 0.5}
            >>> c.coeffs({})
            {}

        """
        return self._coeffs(glpk.get_mat_row, glpk.set_mat_row,
                            Variable.__name__, self._program._variables,
                            coeffs)

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
            row = self._program._row(self)
            cdef char* chars = glpk.get_row_name(self._problem, row)
            return '' if chars is NULL else chars.decode()
        def __set__(self, name):
            row = self._program._row(self)
            glpk.set_row_name(self._problem, row, name2chars(name))
        def __del__(self):
            row = self._program._row(self)
            glpk.set_row_name(self._problem, row, NULL)
