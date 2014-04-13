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


cdef class _Component:

    cdef MILProgram _program
    cdef glpk.ProbObj* _problem

    def __cinit__(self, program):
        self._program = program
        self._problem = <glpk.ProbObj*>PyCapsule_GetPointer(
                                                program._problem_ptr(), NULL)


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


cdef class Objective(_Component):
    """The problem's objective

    .. doctest:: Objective

        >>> p = MILProgram()
        >>> o = p.objective
        >>> isinstance(o, Objective)
        True

    """

    property direction:
        """The objective direction, either `'minimize'` or `'maximize'`

        .. doctest:: Objective

            >>> o.direction  # the GLPK default
            'minimize'
            >>> o.direction = 'maximize'
            >>> o.direction
            'maximize'

        """
        def __get__(self):
            return optdir2str[glpk.get_obj_dir(self._problem)]
        def __set__(self, direction):
            if direction in str2optdir:
                glpk.set_obj_dir(self._problem, str2optdir[direction])
            else:
                raise ValueError("Direction must be 'minimize' or 'maximize'.")

    def coeffs(self, coeffs=None):
        """Change or retrieve objective function coefficients

        :param coeffs: the mapping with coefficients to change
            (``{}`` to set all coefficients to 0; omit for retrieval only)
        :type coeffs: |Mapping| of `.Variable` to |Real|
        :returns: the coefficient mapping, which only contains nonzero
            coefficients
        :rtype: `dict` of `.Variable` to `float`
        :raises TypeError: if *coeffs* is not |Mapping|
        :raises TypeError: if a coefficient key is not `.Variable`
        :raises TypeError: if a coefficient value is not |Real|

        .. doctest:: Objective.coeffs

            >>> p = MILProgram()
            >>> x = p.add_variable()
            >>> y = p.add_variable()
            >>> o = p.objective
            >>> o.coeffs()
            {}
            >>> o.coeffs({x: 3, y: 0})
            {<epyglpki.Variable object at 0x...>: 3.0}
            >>> o.coeffs({})
            {}

        """
        if coeffs is not None:
            if not isinstance(coeffs, collections.abc.Mapping):
                raise TypeError("Coefficients must be given using a " +
                                "collections.abc.Mapping.")
            elif not coeffs:
                for variable in self._program._variables:
                    coeffs[variable] = 0.0
            for variable, val in coeffs.items():
                if not isinstance(variable, Variable):
                    raise TypeError("Coefficient keys must be 'Variable' " +
                                    "instead of '"
                                    + type(variable).__name__ + "'.")
                else:
                    col = self._program._col(variable)
                    if isinstance(val, numbers.Real):
                        glpk.set_obj_coef(self._problem, col, val)
                    else:
                        raise TypeError("Coefficient values must be " +
                                        "'numbers.Real' instead of '" +
                                        type(val).__name__ + "'.")
        coeffs = {}
        for col, variable in enumerate(self._program._variables, start=1):
            val = glpk.get_obj_coef(self._problem, col)
            if val != 0.0:
                coeffs[variable] = val
        return coeffs

    property constant:
        """The objective function constant, a |Real| number

        .. doctest:: Objective

            >>> o.constant  # the GLPK default
            0.0
            >>> o.constant = 3
            >>> o.constant
            3.0
            >>> del o.constant
            >>> o.constant
            0.0

        """
        def __get__(self):
            return glpk.get_obj_coef(self._problem, 0)
        def __set__(self, constant):
            if isinstance(constant, numbers.Real):
                glpk.set_obj_coef(self._problem, 0, constant)
            else:
                raise TypeError("Objective constant must be a real number.")
        def __del__(self):
            glpk.set_obj_coef(self._problem, 0, 0.0)

    property name:
        """The objective function name, a `str` of ≤255 bytes UTF-8 encoded

        .. doctest:: Objective

            >>> o.name  # the GLPK default
            ''
            >>> o.name = 'σκοπός'
            >>> o.name
            'σκοπός'
            >>> del o.name  # clear name
            >>> o.name
            ''

        """
        def __get__(self):
            cdef char* chars = glpk.get_obj_name(self._problem)
            return '' if chars is NULL else chars.decode()
        def __set__(self, name):
            glpk.set_obj_name(self._problem, name2chars(name))
        def __del__(self):
            glpk.set_obj_name(self._problem, NULL)
