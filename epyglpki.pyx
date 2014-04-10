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
    """Main problem object

    .. doctest:: MILProgram

        >>> p = MILProgram()
        >>> p
        <epyglpki.MILProgram object at 0x...>

    """

    cdef glpk.ProbObj* _problem
    cdef int _unique_ids
    cdef object _variables
    cdef object _constraints

    def __cinit__(self):
        self._problem = glpk.create_prob()
        self._unique_ids = 0
        self._variables = []
        self._constraints = []

    def _problem_ptr(self):
        """Encapsulate the pointer to the problem object

        The problem object pointer `self._problem` cannot be passed as such as
        an argument to other functions. Therefore we encapsulate it in a
        capsule that can be passed. It has to be unencapsulated after
        reception.

        """
        return PyCapsule_New(self._problem, NULL, NULL)

    @classmethod
    def read(cls, fname, format='GLPK', mpsfmt='free'):
        """Read a problem from a file (class method)

        :param fname: the name of the file to read from
        :type fname: :class:`str`
        :param format: the format of the file read from; either :data:`'GLPK'`,
            :data:`'LP'`, :data:`'MPS'`, or :data:`'CNFSAT'`
        :type format: :class:`str`
        :param mpsfmt: MPS-subformat; either :data:`'free'` or :data:`'fixed'`
            (ignored when `format` is not :data:`'MPS'`)
        :type mpsfmt: :class:`str`
        :raises ValueError: if `format` is not :data:`'GLPK'`, :data:`'LP'`,
            :data:`'MPS'`, or :data:`'CNFSAT'`
        :raises RuntimeError: if an error occurred reading the file

        .. todo::

            Add doctest

        """
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
        elif format is 'CNFSAT':
            retcode = glpk.read_cnfsat(problem, chars)
        else:
            raise ValueError("Only 'GLPK', 'LP', 'MPS', and 'CNFSAT' " +
                             "formats are supported.")
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
        """Write the problem to a file

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :param format: the format of the file written to; either
            :data:`'GLPK'`, :data:`'LP'`, :data:`'MPS'`, or :data:`'CNFSAT'`
        :type format: :class:`str`
        :param mpsfmt: MPS-subformat; either :data:`'free'` or :data:`'fixed'`
            (ignored when `format` is not :data:`'MPS'`)
        :type mpsfmt: :class:`str`
        :raises ValueError: if `format` is not :data:`'GLPK'`, :data:`'LP'`,
            :data:`'MPS'`, or :data:`'CNFSAT'`
        :raises RuntimeError: if an error occurred writing the file

        .. todo::

            Add doctest

        """
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
        if format is 'CNFSAT':
            retcode = glpk.write_cnfsat(self._problem, chars)
        else:
            raise ValueError("Only 'GLPK', 'LP', 'MPS', and 'CNFSAT' " +
                             "formats are supported.")
        if retcode is not 0:
            raise RuntimeError("Error writing " + format + " file.")

    def __dealloc__(self):
        glpk.delete_prob(self._problem)

    def _generate_unique_id(self):
        self._unique_ids += 1
        return self._unique_ids

    def name(self, name=None):
        """Change or retrieve problem name

        :param name: the new problem name (omit for retrieval only)
        :type name: :class:`str`
        :returns: the problem name
        :rtype: :class:`str`
        :raises TypeError: if `name` is not a :class:`str`
        :raises ValueError: if `name` exceeds 255 bytes encoded in UTF-8

        .. doctest:: MILProgram.name

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

    def _col(self, variable):
        """Return the column index of a Variable"""
        try:
            return 1 + self._variables.index(variable)
                # GLPK indices start at 1
        except ValueError:
            raise IndexError("This is possibly a zombie; kill it using 'del'.")

    def _row(self, constraint):
        """Return the row index of a Constraint"""
        try:
            return 1 + self._constraints.index(constraint)
                # GLPK indices start at 1
        except ValueError:
            raise IndexError("This is possibly a zombie; kill it using 'del'.")

    def _ind(self, varstraint):
        """Return the column/row index of a Variable/Constraint"""
        if isinstance(varstraint, Variable):
            return self._col(varstraint)
        elif isinstance(varstraint, Constraint):
            return self._row(varstraint)
        else:
            raise TypeError("No index available for this object type.")

    def _del_varstraint(self, varstraint):
        """Remove a Variable or Constraint from the problem"""
        if isinstance(varstraint, Variable):
            self._variables.remove(varstraint)
        elif isinstance(varstraint, Constraint):
            self._constraints.remove(varstraint)
        else:
            raise TypeError("No index available for this object type.")

    def add_variable(self, coeffs={}, lower_bound=False, upper_bound=False,
                     kind='continuous', name=''):
        """Add and obtain new variable object

        :param coeffs: set variable coefficients; see :meth:`Variable.coeffs`
        :param lower_bound: set variable lower bound;
            see :meth:`Variable.bounds`, parameter `lower`
        :param upper_bound: set variable upper bound;
            see :meth:`Variable.bounds`, parameter `upper`
        :param kind: set variable kind; see :meth:`Variable.kind`
        :param name: set variable name; see :meth:`Variable.name`
        :returns: variable object
        :rtype: :class:`Variable`

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
        variable.kind(kind)
        variable.name(None if not name else name)
        return variable

    def variables(self):
        """A list of the problem's variables

        :returns: a list of the problem's variables
        :rtype: :class:`list` of :class:`Variable`

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
                       name=''):
        """Add and obtain new constraint object

        :param coeffs: set constraint coefficients;
            see :meth:`Constraint.coeffs`
        :param lower_bound: set constraint lower bound;
            see :meth:`Constraint.bounds`, parameter `lower`
        :param upper_bound: set constraint upper bound;
            see :meth:`Constraint.bounds`, parameter `upper`
        :param name: set constraint name; see :meth:`Constraint.name`
        :returns: constraint object
        :rtype: :class:`Constraint`

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
        constraint.name(None if not name else name)
        return constraint

    def constraints(self):
        """Return a list of the problem's constraints

        :returns: a list of the problem's constraints
        :rtype: :class:`list` of :class:`Constraint`

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

    def coeffs(self, coeffs):
        """Change or retrieve coefficients (constraint matrix)

        :param coeffs: the mapping with coefficients to change
            (`{}` to set all coefficients to `0`)
        :type coeffs: :class:`~collections.abc.Mapping` of length-2
            :class:`~collections.abc.Sequence`, containing one
            :class:`Variable` and one :class:`Constraint`, to
            :class:`~numbers.Real`
        :raises TypeError: if `coeffs` is not :class:`~collections.abc.Mapping`
        :raises TypeError: if a coefficient key component is not a pair of
              :class:`Variable` and :class:`Constraint`
        :raises TypeError: if a coefficient value is not :class:`~numbers.Real`
        :raises ValueError: if the coefficient key does not have two 
              components

        .. doctest:: MILProgram.coeffs

            >>> p = MILProgram()
            >>> x = p.add_variable()
            >>> y = p.add_variable()
            >>> c = p.add_constraint()
            >>> d = p.add_constraint()
            >>> p.coeffs({(x, c): 3, (d, y): 5.5, (x, d): 0})
            >>> x.coeffs()[c] == c.coeffs()[x] == 3
            True
            >>> y.coeffs()[d] == d.coeffs()[y] == 5.5
            True
            >>> len(x.coeffs()) == len(d.coeffs()) == 1
            True

        """
        if isinstance(coeffs, collections.abc.Mapping):
            elements = len(coeffs)
        else:
            raise TypeError("Coefficients must be given using a " +
                            "collections.abc.Mapping.")
        cdef double* vals = <double*>glpk.alloc(1+elements, sizeof(double))
        cdef int* cols = <int*>glpk.alloc(1+elements, sizeof(int))
        cdef int* rows = <int*>glpk.alloc(1+elements, sizeof(int))
        try:
            if elements is 0:
                glpk.load_matrix(self._problem, elements, NULL, NULL, NULL)
            else:
                nz_elements = elements
                for ind, item in enumerate(coeffs.items(), start=1):
                    val = vals[ind] = item[1]
                    if not isinstance(val, numbers.Real):
                        raise TypeError("Coefficient values must be " +
                                        "'numbers.Real' instead of '" +
                                        type(val).__name__ + "'.")
                    elif val == 0.0:
                        nz_elements -= 1
                    if len(item[0]) is not 2:
                        raise ValueError("Coefficient key must have " +
                                         "exactly two components.")
                    elif (isinstance(item[0][0], Variable) and
                        isinstance(item[0][1], Constraint)):
                        cols[ind] = self._col(item[0][0])
                        rows[ind] = self._row(item[0][1])
                    elif (isinstance(item[0][0], Constraint) and
                            isinstance(item[0][1], Variable)):
                        rows[ind] = self._row(item[0][0])
                        cols[ind] = self._col(item[0][1])
                    else:
                        raise TypeError("Coefficient position components " +
                                        "must be one Variable and one " +
                                        "Constraint.")
                glpk.load_matrix(self._problem, elements, rows, cols, vals)
                assert nz_elements is glpk.get_num_nz(self._problem)
        finally:
            glpk.free(vals)
            glpk.free(cols)
            glpk.free(rows)

    def scaling(self, *algorithms, factors=None):
        """Change, apply and unapply scaling factors

        :param algorithms: choose scaling algorithms to apply from among

            * :data:`'auto'`: choose algorithms automatically
              (other arguments are ignored)
            * :data:`'skip'`: skip scaling if the problem is already
              well-scaled
            * :data:`'geometric'`: perform geometric mean scaling
            * :data:`'equilibration'`: perform equilibration scaling
            * :data:`'round'`: round scaling factors to the nearest power of
              two

        :type algorithms: zero or more :class:`str` arguments
        :param factors: the mapping with scaling factors to change
            (`{}` to set all factors to `1`; omit for retrieval only);
            values defined here have precedence over the ones generated by
            `algorithms`
        :type factors: :class:`~collections.abc.Mapping` of
            :class:`Variable` or :class:`Constraint` to :class:`~numbers.Real`
        :returns: the scaling factor mapping, which only contains non-`1`
            factors
        :rtype: :class:`dict` of :class:`Variable` or :class:`Constraint` to
            :class:`float`
        :raises TypeError: if `factors` is not
            :class:`~collections.abc.Mapping`
        :raises TypeError: if the scaling factors are not
            :class:`~numbers.Real`
        :raises TypeError: if a key in the scaling factor mapping is neither
            :class:`Variable` nor :class:`Constraint`

        .. doctest:: MILProgram.scaling

            >>> p = MILProgram()
            >>> x = p.add_variable()
            >>> y = p.add_variable()
            >>> c = p.add_constraint()
            >>> d = p.add_constraint()
            >>> p.coeffs({(x, c): 3e-100, (d, y): 5.5, (x, d): 1.5e200})
            >>> p.scaling()
            {}
            >>> p.scaling('skip', 'geometric', 'equilibration',
            ...           factors={y: 3}) # doctest: +NORMALIZE_WHITESPACE
            {<epyglpki.Variable object at 0x...>: 3.329...e-67,
             <epyglpki.Variable object at 0x...>: 3.0,
             <epyglpki.Constraint object at 0x...>: 1.001...e+166,
             <epyglpki.Constraint object at 0x...>: 2.002...e-134}
            >>> p.scaling(factors={})
            {}

        .. note::

            If a scaling algorithm is given, all factors are first set to `1`:

            .. doctest:: MILProgram.scaling

                >>> p.scaling(factors={d: 5.5})
                {<epyglpki.Constraint object at 0x...>: 5.5}
                >>> p.scaling('round')
                {}

        """
        if algorithms:
            if 'auto' in algorithms:
                glpk.scale_prob(self._problem, str2scalopt['auto'])
            else:
                glpk.scale_prob(self._problem, sum(str2scalopt[algorithm]
                                                for algorithm in algorithms))
        if factors == {}:
            glpk.unscale_prob(self._problem)
        elif isinstance(factors, collections.abc.Mapping):
            for varstraint, factor in factors.items():
                if not isinstance(factor, numbers.Real):
                    raise TypeError("Scaling factors must be real numbers.")
                if isinstance(varstraint, Variable):
                    glpk.set_col_sf(self._problem, self._col(varstraint),
                                    factor)
                elif isinstance(varstraint, Constraint):
                    glpk.set_row_sf(self._problem, self._row(varstraint),
                                    factor)
                else:
                    raise TypeError("Only 'Variable' and 'Constraint' can " +
                                    "have a scaling factor.")
        elif factors is not None:
            raise TypeError("Factors must be given using a " +
                            "collections.abc.Mapping.")
        factors = {}
        for col, variable in enumerate(self._variables, start=1):
            factor = glpk.get_col_sf(self._problem, col)
            if factor != 1.0:
                factors[variable] = factor
        for row, constraint in enumerate(self._constraints, start=1):
            factor = glpk.get_row_sf(self._problem, row)
            if factor != 1.0:
                factors[constraint] = factor
        return factors

    def objective(self, coeffs={}, constant=0, direction='minimize',
                  name=''):
        """Obtain objective object

        :param coeffs: set objective coefficients; see :meth:`Objective.coeffs`
        :param constant: set objective constant; see :meth:`Objective.constant`
        :param direction: set objective direction; see :meth:`Objective.direction`
        :param name: set objective name; see :meth:`Objective.name`
        :returns: objective object
        :rtype: :class:`Objective`

        .. doctest:: MILProgram.objective

            >>> p = MILProgram()
            >>> o = p.objective()
            >>> o
            <epyglpki.Objective object at 0x...>

        """
        objective = Objective(self)
        objective.coeffs(coeffs)
        objective.constant(constant)
        objective.direction(direction)
        objective.name(name)
        return objective

    def simplex(self, **controls):
        """Obtain simplex solver object

        :param controls: set solver and basis factorization control parameters;
            see :meth:`SimplexSolver.controls`, parameter `controls`
        :returns: simplex solver object
        :rtype: :class:`SimplexSolver`

        .. doctest:: MILProgram.simplex

            >>> p = MILProgram()
            >>> s = p.simplex()
            >>> s
            <epyglpki.SimplexSolver object at 0x...>

        """
        simplex_solver = SimplexSolver(self)
        simplex_solver.controls(**controls)
        return simplex_solver

    def ipoint(self, **controls):
        """Obtain interior point solver object

        :param controls: set solver control parameters;
            see :meth:`IPointSolver.controls`, parameter `controls`
        :returns: interior point solver object
        :rtype: :class:`IPointSolver`

        .. doctest:: MILProgram.ipoint

            >>> p = MILProgram()
            >>> s = p.ipoint()
            >>> s
            <epyglpki.IPointSolver object at 0x...>

        """
        ipoint_solver = IPointSolver(self)
        ipoint_solver.controls(**controls)
        return ipoint_solver

    def intopt(self, **controls):
        """Obtain integer optimization solver object

        :param controls: set solver control parameters;
            see :meth:`IntOptSolver.controls`, parameter `controls`
        :returns: integer optimization solver object
        :rtype: :class:`IntOptSolver`

        .. doctest:: MILProgram.intopt

            >>> p = MILProgram()
            >>> s = p.intopt()
            >>> s
            <epyglpki.IntOptSolver object at 0x...>

        """
        intopt_solver = IntOptSolver(self)
        intopt_solver.controls(**controls)
        return intopt_solver


include "epyglpki-components.pxi"


include "epyglpki-solvers.pxi"
