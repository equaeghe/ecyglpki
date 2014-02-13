# epyglpki.pyx: Cython/Python interface for GLPK

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


cimport glpk
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
import numbers
import collections.abc

include 'glpk-constants.pxi'


def GLPK_version():
    return glpk.version().decode()


cdef name2chars(name):
    cdef char* chars
    if not isinstance(name, str):
        raise TypeError("Name must be a 'str'.")
    else:
        name = name.encode()
        if len(name) > 255:
            raise ValueError("Name must not exceed 255 bytes.")
        chars = name
    return chars


cdef class MILProgram:

    cdef glpk.ProbObj* _problem

    def __cinit__(self):
        self._problem = glpk.create_prob()

    def _problem_ptr(self):
        return PyCapsule_New(self._problem, NULL, NULL)

    cdef int _unique_ids
    cdef object _variables
    cdef object _constraints

    def __init__(self):
        self._unique_ids = 0
        self._variables = []
        self._constraints = []

    @classmethod
    def read(cls, fname, format='GLPK', mpsfmt='free'):
        cdef char* chars
        cdef glpk.ProbObj* problem
        fname = name2chars(fname)
        chars = fname
        program = cls()
        problem = <glpk.ProbObj*>PyCapsule_GetPointer(
                                                program._problem_ptr(), NULL)
        if format is 'GLPK':
            retcode = glpk.read_prob(problem, 0, chars)
        elif format is 'LP':
            retcode = glpk.read_lp(problem, NULL, chars)
        elif format is 'MPS':
            retcode = glpk.read_mps(problem, str2mpsfmt[mpsfmt], NULL, chars)
        else:
            raise ValueError("Only 'GLPK', 'LP', and 'MPS' formats are " +
                             "supported.")
        if retcode is 0:
            for col in range(glpk.get_num_cols(problem)):
                variable = Variable(program)
                program._variables.append(variable)
            for row in range(glpk.get_num_rows(problem)):
                constraint = Constraint(program)
                program._constraints.append(constraint)
        else:
            raise RuntimeError("Error reading " + format + " file.")
        return program

    def write(self, fname, format='GLPK', mpsfmt='free'):
        cdef char* chars
        fname = name2chars(fname)
        chars = fname
        if format is 'GLPK':
            retcode = glpk.write_prob(self._problem, 0, chars)
        elif format is 'LP':
            retcode = glpk.write_lp(self._problem, NULL, chars)
        elif format is 'MPS':
            retcode = glpk.write_mps(self._problem,
                                     str2mpsfmt[mpsfmt], NULL, chars)
        else:
            raise ValueError("Only 'GLPK', 'LP', and 'MPS' formats are " +
                             "supported.")
        if retcode is not 0:
            raise RuntimeError("Error writing " + format + " file.")

    def __dealloc__(self):
        glpk.delete_prob(self._problem)

    def _generate_unique_id(self):
        self._unique_ids += 1
        return self._unique_ids

    def name(self, name=None):
        """Change or retrieve problem name

          :type `name`: :class:`str` or `None`
          :returns: the problem name
          :rtype: :class:`str`

        >>> p = MILProgram()
        >>> p.name()
        ''
        >>> p.name('Programme Linéaire')
        'Programme Linéaire'
        >>> p.name()
        'Programme Linéaire'
        >>> p.name('')
        ''

        """
        cdef char* chars
        if name is '':
            glpk.set_prob_name(self._problem, NULL)
        elif name is not None:
            name = name2chars(name)
            chars = name
            glpk.set_prob_name(self._problem, chars)
        chars = glpk.get_prob_name(self._problem)
        return '' if chars is NULL else chars.decode()

    def _ind(self, varstraint):
        try:
            if isinstance(varstraint, Variable):
                ind = self._variables.index(varstraint)
            elif isinstance(varstraint, Constraint):
                ind = self._constraints.index(varstraint)
            else:
                raise TypeError("No index available for this object type.")
        except ValueError:
            raise IndexError("This is possibly a zombie; kill it using 'del'.")
        return 1 + ind

    def _del(self, varstraint):
        if isinstance(varstraint, Variable):
            self._variables.remove(varstraint)
        elif isinstance(varstraint, Constraint):
            self._constraints.remove(varstraint)
        else:
            raise TypeError("No index available for this object type.")

    def add_variable(self, coeffs=False, lower_bound=False, upper_bound=False,
                     kind='continuous', name=None):
        variable = Variable(self)
        self._variables.append(variable)
        assert len(self._variables) is glpk.get_num_cols(self._problem)
        variable.coeffs(coeffs)
        variable.bounds(lower_bound, upper_bound)
        variable.kind(kind)
        variable.name(name)
        return variable

    def variables(self):
        return self._variables

    def add_constraint(self, coeffs=False,
                       lower_bound=False, upper_bound=False, name=None):
        constraint = Constraint(self)
        self._constraints.append(constraint)
        assert len(self._constraints) is glpk.get_num_rows(self._problem)
        constraint.coeffs(coeffs)
        constraint.bounds(lower_bound, upper_bound)
        constraint.name(name)
        return constraint

    def constraints(self):
        return self._constraints

    def coeffs(self, coeffs):
        elements = 0 if coeffs is False else len(coeffs)
        cdef double* vals = <double*>glpk.calloc(1+elements, sizeof(double))
        cdef int* cols = <int*>glpk.calloc(1+elements, sizeof(int))
        cdef int* rows = <int*>glpk.calloc(1+elements, sizeof(int))
        try:
            if elements is 0:
                glpk.load_matrix(self._problem, elements, NULL, NULL, NULL)
            else:
                for ind, item in enumerate(coeffs.items(), start=1):
                    vals[ind] = item[1]
                    if len(item[0]) != 2:
                        raise ValueError("Coefficient position must have " +
                                         "exactly two components.")
                    elif (isinstance(item[0][0], Variable) and
                        isinstance(item[0][1], Constraint)):
                        cols[ind] = self._ind(item[0][0])
                        rows[ind] = self._ind(item[0][1])
                    elif (isinstance(item[0][0], Constraint) and
                            isinstance(item[0][1], Variable)):
                        rows[ind] = self._ind(item[0][0])
                        cols[ind] = self._ind(item[0][1])
                    else:
                        raise TypeError("Coefficient position components " +
                                        "must be one Variable and one " +
                                        "Constraint.")
                glpk.load_matrix(self._problem, elements, rows, cols, vals)
                assert elements is glpk.get_num_nz(self._problem)
        finally:
            glpk.free(vals)
            glpk.free(cols)
            glpk.free(rows)

    def scaling(self, scaling=None):
        if scaling is False:
            glpk.unscale_prob(self._problem)
            scaling = dict()
        elif scaling is None:
            scaling = dict()
        elif isinstance(scaling, collections.abc.Mapping):
            for varstraint, factor in scaling.items():
                if not isinstance(factor, numbers.Real):
                    raise TypeError("Scaling factors must be real numbers.")
                if isinstance(varstraint, Variable):
                    glpk.set_col_sf(self._problem, self._ind(varstraint),
                                    factor)
                elif isinstance(varstraint, Constraint):
                    glpk.set_row_sf(self._problem, self._ind(varstraint),
                                    factor)
                else:
                    raise TypeError("Only 'Variable' and 'Constraint' can " +
                                    "have a scaling factor.")
        elif isinstance(scaling, collections.abc.Sequence):
            if 'auto' in scaling:
                glpk.scale_prob(self._problem, str2scalopt['auto'])
            else:
                glpk.scale_prob(self._problem, sum(str2scalopt[string]
                                                   for string in scaling))
        for variable in self._variables:
            factor = glpk.get_col_sf(self._problem, self._ind(variable))
            if factor != 1.0:
                scaling[variable] = factor
        for constraint in self._constraints:
            factor = glpk.get_row_sf(self._problem, self._ind(constraint))
            if factor != 1.0:
                scaling[constraint] = factor
        return scaling

    def objective(self, coeffs=False, constant=0, direction='minimize',
                  name=None):
        objective = Objective(self)
        objective.coeffs(coeffs)
        objective.constant(constant)
        objective.direction(direction)
        objective.name(name)
        return objective

    def simplex(self, controls=None, fcontrols=None):
        simplex_solver = SimplexSolver(self)
        simplex_solver.controls(controls, fcontrols)
        return simplex_solver

    def ipoint(self, controls=None):
        ipoint_solver = IPointSolver(self)
        ipoint_solver.controls(controls)
        return ipoint_solver

    def intopt(self, controls=None):
        intopt_solver = IntOptSolver(self)
        intopt_solver.controls(controls)
        return intopt_solver


cdef class _ProgramComponent:

    cdef MILProgram _program
    cdef glpk.ProbObj* _problem

    def __cinit__(self, program):
        self._program = program
        self._problem = <glpk.ProbObj*>PyCapsule_GetPointer(
                                                program._problem_ptr(), NULL)


cdef class _Varstraint(_ProgramComponent):

    cdef int _unique_id

    def __init__(self, program):
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
        self._program._del(self)

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
        elif coeffs is False:
            length = 0
        else:
            length = len(coeffs)
        cdef double* vals =  <double*>glpk.calloc(1+length, sizeof(double))
        cdef int* inds =  <int*>glpk.calloc(1+length, sizeof(int))
        try:
            if coeffs is not None:
                if length is 0:
                    set_function(self._problem, ind, length, NULL, NULL)
                else:
                    for other_ind, item in enumerate(coeffs.items(), start=1):
                        vals[other_ind] = item[1]
                        if not isinstance(item[0], type(self)):
                            inds[other_ind] = self._program._ind(item[0])
                        else:
                            raise TypeError("Coefficient keys must be '" +
                                            argtypename + "' instead of '" +
                                            type(item[0]).__name__ +"'.")
                    set_function(self._problem, ind, length, inds, vals)
            length = get_function(self._problem, ind, inds, vals)
            coeffs = dict()
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

    def name(self, name=None):
        return NotImplemented  # should be implemented in public child classes


cdef class Variable(_Varstraint):

    def __init__(self, program):
        glpk.add_cols(self._problem, 1)
        super().__init__(program)

    def zombify(self):
        self._zombify(glpk.del_cols)

    def bounds(self, lower=None, upper=None):
        if self.kind() in {'integer', 'binary'}:
            if any((bound not in {False, None}) and
                   not isinstance(bound, numbers.Integral)
                   for bound in (lower, upper)):
                raise TypeError("Integer variable must have integer bounds.")
        return self._bounds(glpk.get_col_lb, glpk.get_col_ub,
                            glpk.set_col_bnds, lower, upper)

    def kind(self, kind=None):
        col = self._program._ind(self)
        if kind is not None:
            if kind in str2varkind:
                glpk.set_col_kind(self._problem, col, str2varkind[kind])
            else:
                raise ValueError("Kind must be 'continuous'," +
                                 "'integer', or 'binary'.")
        return varkind2str[glpk.get_col_kind(self._problem, col)]

    def coeffs(self, coeffs=None):
        return self._coeffs(glpk.get_mat_col, glpk.set_mat_col,
                            Constraint.__name__, self._program._constraints, 
                            coeffs)

    def name(self, name=None):
        return self._name(glpk.get_col_name, glpk.set_col_name, name)


cdef class Constraint(_Varstraint):

    def __init__(self, program):
        glpk.add_rows(self._problem, 1)
        super().__init__(program)

    def zombify(self):
        self._zombify(glpk.del_rows)

    def bounds(self, lower=None, upper=None):
        return self._bounds(glpk.get_row_lb, glpk.get_row_ub,
                            glpk.set_row_bnds, lower, upper)

    def coeffs(self, coeffs=None):
        return self._coeffs(glpk.get_mat_row, glpk.set_mat_row,
                            Variable.__name__, self._program._variables,
                            coeffs)

    def name(self, name=None):
        return self._name(glpk.get_row_name, glpk.set_row_name, name)


cdef class Objective(_ProgramComponent):

    def direction(self, direction=None):
        """Change or retrieve objective direction

          :type `direction`: :class:`str`, either 'minimize' or 'maximize',
            or `None`
          :returns: the objective direction
          :rtype: :class:`str`

        >>> p = MILProgram()
        >>> o = p.objective()
        >>> o.direction()
        'minimize'
        >>> o.constant('maximize')
        'maximize'

        """
        if direction is not None:
            if direction in str2optdir:
                glpk.set_obj_dir(self._problem, str2optdir[direction])
            else:
                raise ValueError("Direction must be 'minimize' or 'maximize'.")
        return optdir2str[glpk.get_obj_dir(self._problem)]

    def coeffs(self, coeffs=None):
        if coeffs is False:
            coeffs = dict()
            for col in range(1, 1+len(self._program._variables)):
                coeffs[self._program._variables[col-1]] = 0.0
        if coeffs is not None:
            for variable, val in coeffs.items():
                if isinstance(variable, Variable):
                    col = self._program._ind(variable)
                    glpk.set_obj_coef(self._problem, col, val)
                else:
                    raise TypeError("Coefficient keys must be 'Variable' " +
                                    "instead of '"
                                    + type(variable).__name__ + "'.")
        coeffs = dict()
        for col in range(1, 1+len(self._program._variables)):
            val = glpk.get_obj_coef(self._problem, col)
            if val != 0.0:
                coeffs[self._program._variables[col-1]] = val
        return coeffs

    def constant(self, constant=None):
        """ Change or retrieve objective function constant

          :type `constant`: :class:`numbers.Real` or `None`
          :returns: the objective function constant
          :rtype: :class:`float`

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
                raise TypeError("Objective constant must be 'numbers.Real'.")
        return glpk.get_obj_coef(self._problem, 0)

    def name(self, name=None):
        """Change or retrieve objective function name

          :type `name`: :class:`str` or `None`
          :returns: the objective function name
          :rtype: :class:`str`

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


include "epyglpki-solvers.pxi"
