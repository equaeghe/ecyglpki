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


cdef class _ProgramComponent:

    cdef MILProgram _program
    cdef glpk.ProbObj* _problem

    def __cinit__(self, program):
        self._program = program
        self._problem = <glpk.ProbObj*>PyCapsule_GetPointer(
                                                program._problem_ptr(), NULL)


cdef class _Varstraint(_ProgramComponent):

    cdef int _unique_id

    def __cinit__(self, program):
        self._unique_id = self._program._generate_unique_id()

    def __hash__(self):
        return self._unique_id

    def __str__(self):
        return str(self._unique_id) + ':' + self.name()

    #  how to really del?
    cdef _zombify(self, void (*del_function)(glpk.ProbObj*, int, const int[])):
        cdef int ind[2]
        ind[1] = self._program._ind(self)
        del_function(self._problem, 1, ind)
        self._program._del_varstraint(self)

    cdef _bounds(self,
                 double (*get_lb_function)(glpk.ProbObj*, int),
                 double (*get_ub_function)(glpk.ProbObj*, int),
                 void (*set_bnds_function)(glpk.ProbObj*,
                                           int, int, double, double),
                 lower=None, upper=None):
        cdef double lb
        cdef double ub
        ind = self._program._ind(self)
        if lower is False:
            lb = -DBL_MAX
        else:
            if lower is None:
                lb = get_lb_function(self._problem, ind)
            elif isinstance(lower, numbers.Real):
                lb = lower
            else:
                raise TypeError("Lower bound must be real numbers or 'False'.")
            lower = False if lb == -DBL_MAX else True
        if upper is False:
            ub = +DBL_MAX
        else:
            if upper is None:
                ub = get_ub_function(self._problem, ind)
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
        set_bnds_function(self._problem, ind, vartype, lb, ub)
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

    cdef _name(self,
               const char* (*get_name_function)(glpk.ProbObj*, int),
               void (*set_name_function)(glpk.ProbObj*, int, const char*),
               name=None):
        cdef char* chars
        ind = self._program._ind(self)
        if name is '':
            set_name_function(self._problem, ind, NULL)
        elif name is not None:
            name = name2chars(name)
            chars = name
            set_name_function(self._problem, ind, chars)
        chars = get_name_function(self._problem, ind)
        return '' if chars is NULL else chars.decode()


cdef class Variable(_Varstraint):
    """One of the problem's variables"""

    def __cinit__(self, program):
        glpk.add_cols(self._problem, 1)

    def remove(self):
        """Remove the variable from the problem

        .. doctest:: Variable.remove

            >>> p = MILProgram()
            >>> x = p.add_variable()
            >>> p.variables()
            [<epyglpki.Variable object at 0x...>]
            >>> x.remove()
            >>> p.variables()
            []

        .. note::

            Removing a variable from a problem object does not delete the
            referencing Variable objects, which in some sense become 'zombie'
            objects; they should best be deleted manually:

            .. doctest:: Variable.remove

                >>> p.variables()
                []
                >>> x
                <epyglpki.Variable object at 0x...>
                >>> x.name()
                Traceback (most recent call last):
                  ...
                ValueError: <epyglpki.Variable object at 0x...> is not in list
                <BLANKLINE>
                During handling of the above exception, another exception occurred:
                <BLANKLINE>
                Traceback (most recent call last):
                  ...
                IndexError: This is possibly a zombie; kill it using 'del'.
                >>> del x
                >>> x
                Traceback (most recent call last):
                  ...
                NameError: name 'x' is not defined

        """
        self._zombify(glpk.del_cols)

    def bounds(self, lower=None, upper=None):
        """Change or retrieve variable bounds

        :param lower: the variable's lower bound
            (`False` to remove bound; omit for retrieval only)
        :type lower: |Real| or `False`
        :param upper: the variable's upper bound
            (`False` to remove bound; omit for retrieval only)
        :type upper: |Real| or `False`
        :returns: the variable's bounds
        :rtype: length-2 `tuple` of `float` (or `False`)
        :raises TypeError: if *lower* or *upper* is not |Real| or `False`
        :raises ValueError: if *lower* is larger than *upper*

        .. doctest:: Variable.bounds

            >>> p = MILProgram()
            >>> x = p.add_variable()
            >>> x.bounds()
            (False, False)
            >>> x.bounds(lower=0, upper=5.5)
            (0.0, 5.5)
            >>> x.bounds(upper=False)
            (0.0, False)

        """
        return self._bounds(glpk.get_col_lb, glpk.get_col_ub,
                            glpk.set_col_bnds, lower, upper)

    def kind(self, kind=None):
        """Change or retrieve variable kind

        :param kind: the new variable kind (omit for retrieval only)
        :type kind: `str`, either `'continuous'`, `'integer'`, or `'binary'`
        :returns: the variable kind
        :rtype: `str`
        :raises ValueError: if *kind* is not `'continuous'`, `'integer'`,
            or `'binary'`

        .. doctest:: Variable.kind

            >>> p = MILProgram()
            >>> x = p.add_variable()
            >>> x.kind()
            'continuous'
            >>> x.kind('integer')
            'integer'

        .. note::

            A variable has `'binary'` kind if and only if it is an
            integer variable with lower bound zero and upper bound one:

            .. doctest:: Variable.kind

                >>> x.kind()
                'integer'
                >>> x.bounds(lower=0, upper=1)
                (0.0, 1.0)
                >>> x.kind()
                'binary'
                >>> x.bounds(upper=3)
                (0.0, 3.0)
                >>> x.kind()
                'integer'
                >>> x.kind('binary')
                'binary'
                >>> x.bounds()
                (0.0, 1.0)

        """
        col = self._program._col(self)
        if kind is not None:
            if kind in str2varkind:
                glpk.set_col_kind(self._problem, col, str2varkind[kind])
            else:
                raise ValueError("Kind must be 'continuous'," +
                                 "'integer', or 'binary'.")
        return varkind2str[glpk.get_col_kind(self._problem, col)]

    def coeffs(self, coeffs=None):
        """Change or retrieve variable coefficients (constraint matrix column)

        :param coeffs: the mapping with coefficients to change
            (``{}`` to set all coefficients to `0`; omit for retrieval only)
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

    def name(self, name=None):
        """Change or retrieve variable name

        :param name: the new variable name (omit for retrieval only)
        :type name: `str`
        :returns: the variable name
        :rtype: `str`
        :raises TypeError: if *name* is not a `str`
        :raises ValueError: if *name* exceeds 255 bytes encoded in UTF-8

        .. doctest:: Variable.name

            >>> p = MILProgram()
            >>> x = p.add_variable()
            >>> x.name()
            ''
            >>> x.name('Stake')
            'Stake'
            >>> x.name()
            'Stake'

        """
        return self._name(glpk.get_col_name, glpk.set_col_name, name)


cdef class Constraint(_Varstraint):
    """One of the problem's constraints"""

    def __cinit__(self, program):
        glpk.add_rows(self._problem, 1)

    def remove(self):
        """Remove the constraint from the problem

        .. doctest:: Constraint.remove

            >>> p = MILProgram()
            >>> c = p.add_constraint()
            >>> p.constraints()
            [<epyglpki.Constraint object at 0x...>]
            >>> c.remove()
            >>> p.constraints()
            []

        .. note::

            Removing a constraint from a problem object does not delete the
            referencing Constraint objects, which in some sense become 'zombie'
            objects; they should best be deleted manually:

            .. doctest:: Constraint.remove

                >>> p.constraints()
                []
                >>> c
                <epyglpki.Constraint object at 0x...>
                >>> c.name()
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
        self._zombify(glpk.del_rows)

    def bounds(self, lower=None, upper=None):
        """Change or retrieve constraint bounds

        :param lower: the constraint's lower bound
            (`False` to remove bound; omit for retrieval only)
        :type lower: |Real| or `False`
        :param upper: the constraint's upper bound
            (`False` to remove bound; omit for retrieval only)
        :type upper: |Real| or `False`
        :returns: the constraint's bounds
        :rtype: length-2 `tuple` of `float` (or `False`)
        :raises TypeError: if *lower* or *upper* is not |Real| or `False`
        :raises ValueError: if *lower* is larger than *upper*

        .. doctest:: Constraint.bounds

            >>> p = MILProgram()
            >>> c = p.add_constraint()
            >>> c.bounds()
            (False, False)
            >>> c.bounds(lower=-1/2, upper=1/2)
            (-0.5, 0.5)
            >>> c.bounds(lower=False)
            (False, 0.5)

        """
        return self._bounds(glpk.get_row_lb, glpk.get_row_ub,
                            glpk.set_row_bnds, lower, upper)

    def coeffs(self, coeffs=None):
        """Change or retrieve constraint coefficients (constraint matrix row)

        :param coeffs: the mapping with coefficients to change
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

    def name(self, name=None):
        """Change or retrieve constraint name

        :param name: the new constraint name (omit for retrieval only)
        :type name: `str`
        :returns: the constraint name
        :rtype: `str`
        :raises TypeError: if *name* is not a `str`
        :raises ValueError: if *name* exceeds 255 bytes encoded in UTF-8

        .. doctest:: Constraint.name

            >>> p = MILProgram()
            >>> c = p.add_constraint()
            >>> c.name()
            ''
            >>> c.name('Budget')
            'Budget'
            >>> c.name()
            'Budget'

        """
        return self._name(glpk.get_row_name, glpk.set_row_name, name)


cdef class Objective(_ProgramComponent):
    """The problem's objective function"""

    def direction(self, direction=None):
        """Change or retrieve objective direction

        :param direction: the new objective direction (omit for retrieval only)
        :type direction: `str`, either `'minimize'` or `'maximize'`
        :returns: the objective direction
        :rtype: `str`
        :raises ValueError: if *direction* is not `'minimize'` or `'maximize'`

        .. doctest:: Objective.direction

            >>> p = MILProgram()
            >>> o = p.objective()
            >>> o.direction()
            'minimize'
            >>> o.direction('maximize')
            'maximize'

        """
        if direction is not None:
            if direction in str2optdir:
                glpk.set_obj_dir(self._problem, str2optdir[direction])
            else:
                raise ValueError("Direction must be 'minimize' or 'maximize'.")
        return optdir2str[glpk.get_obj_dir(self._problem)]

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
            >>> o = p.objective()
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

    def constant(self, constant=None):
        """Change or retrieve objective function constant

        :param constant: the new objective function constant
            (omit for retrieval only)
        :type constant: |Real|
        :returns: the objective function constant
        :rtype: `float`
        :raises TypeError: if *constant* is not |Real|

        .. doctest:: Objective.constant

            >>> p = MILProgram()
            >>> o = p.objective()
            >>> o.constant()
            0.0
            >>> o.constant(3)
            3.0

        """
        if constant is not None:
            if isinstance(constant, numbers.Real):
                glpk.set_obj_coef(self._problem, 0, constant)
            else:
                raise TypeError("Objective constant must be a real number.")
        return glpk.get_obj_coef(self._problem, 0)

    def name(self, name=None):
        """Change or retrieve objective function name

        :param name: the new objective function name (omit for retrieval only)
        :type name: `str`
        :returns: the objective function name
        :rtype: `str`
        :raises TypeError: if *name* is not a `str`
        :raises ValueError: if *name* exceeds 255 bytes encoded in UTF-8

        .. doctest:: Objective.name

            >>> p = MILProgram()
            >>> o = p.objective()
            >>> o.name()
            ''
            >>> o.name('σκοπός')
            'σκοπός'
            >>> o.name()
            'σκοπός'

        """
        cdef char* chars
        if name is '':
            glpk.set_obj_name(self._problem, NULL)
        elif name is not None:
            name = name2chars(name)
            chars = name
            glpk.set_obj_name(self._problem, chars)
        chars = glpk.get_obj_name(self._problem)
        return '' if chars is NULL else chars.decode()
