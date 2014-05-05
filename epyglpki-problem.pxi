# epyglpki-solvers.pxi: Cython/Python interface for GLPK problems

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


import itertools


from libc.limits cimport INT_MAX
cdef int int_MAX = int(INT_MAX)

from libc.float cimport DBL_MAX
cdef float double_MAX = float(DBL_MAX)

# message levels
cdef str2msglev = {
    'no': glpk.MSG_OFF,
    'warnerror': glpk.MSG_ERR,
    'normal': glpk.MSG_ON,
    'full': glpk.MSG_ALL
    }
cdef msglev2str = {msg_lev: string for string, msg_lev in str2msglev.items()}


include 'epyglpki-smcp.pxi'


include 'epyglpki-bfcp.pxi'


include 'epyglpki-iptcp.pxi'


include 'epyglpki-iocp.pxi'


# optimization directions
cdef str2optdir = {
    'minimize': glpk.MIN,
    'maximize': glpk.MAX
    }
cdef optdir2str = {optdir: string for string, optdir in str2optdir.items()}


# variable kinds
cdef str2varkind = {
    'continuous': glpk.CV,
    'integer': glpk.IV,
    'binary': glpk.BV
    }
cdef varkind2str = {varkind: string for string, varkind in str2varkind.items()}


# variable types
cdef pair2vartype = {
    (True, True): glpk.FR,
    (False, True): glpk.LO,
    (True, False): glpk.UP,
    (False, False): glpk.DB
    }
cdef vartype2str = {
    glpk.FR: 'free',
    glpk.LO: 'dominating',
    glpk.UP: 'dominated',
    glpk.DB: 'bounded',
    glpk.FX: 'fixed'
    }

cdef _bounds(lower, upper):
    if isinstance(lower, numbers.Real):
        lb = lower
        if lb <= -double_MAX:
            lb = -double_MAX
            lower = None
    elif lower is None:
        lb = -double_MAX
    else:
        raise TypeError("Lower bound must be 'None' or 'Real', not " +
                        type(lower).__name__ + ".")
    if isinstance(upper, numbers.Real):
        ub = upper
        if ub >= +double_MAX:
            ub = +double_MAX
            upper = None
    elif upper is None:
        ub = +double_MAX
    else:
        raise TypeError("Upper bound must be 'None' or 'Real', not " +
                        type(lower).__name__ + ".")
    if lb > ub:
        raise ValueError("Lower bound (" + str(lb) + ") must not dominate " +
                         "upper bound (" + str(ub) + ").")
    vartype = pair2vartype[(lower is None, upper is None)]
    if vartype is glpk.DB:
        if lb == ub:
            vartype = glpk.FX
    return vartype, lb, ub


# variable status
cdef str2varstat = {
    'basic': glpk.BS,
    'lower': glpk.NL,
    'upper': glpk.NU,
    'free': glpk.NF,
    'fixed': glpk.NS
    }
cdef varstat2str = {varstat: string for string, varstat in str2varstat.items()}

cdef compatible_status = {
    'free': frozenset('free'),
    'dominating': frozenset('lower'),
    'dominated': frozenset('upper'),
    'bounded': frozenset('lower', 'upper'),
    'fixed': frozenset('fixed'),
    }

cdef _statuscheck(str vartypestr, str status):
    if status is not 'basic':
        statuses = compatible_status[vartypestr]
        if status not in statuses:
            raise ValueError("Row status must be in " + str(statuses) +
                             ", not " + status + ".")


# scaling options
cdef str2scalopt = {
    'geometric': glpk.SF_GM,
    'equilibration': glpk.SF_EQ,
    'round': glpk.SF_2N,
    'skip': glpk.SF_SKIP,
    'auto': glpk.SF_AUTO
    }


# solution indicators
cdef str2solind = {
    'simplex': glpk.SOL,
    'ipoint': glpk.IPT,
    'intopt': glpk.MIP
    }


# solution statuses
cdef solstat2str = {
    glpk.UNDEF: 'undefined',
    glpk.OPT: 'optimal',
    glpk.INFEAS: 'infeasible',
    glpk.NOFEAS: 'no feasible',
    glpk.FEAS: 'feasible',
    glpk.UNBND: 'unbounded',
    }


# simplex return codes (errors)
cdef smretcode2error = {
    glpk.EBADB: ValueError("Basis is invalid."),
    glpk.ESING: ValueError("Basis matrix is singular."),
    glpk.ECOND: ValueError("Basis matrix is ill-conditioned."),
    glpk.EBOUND: ValueError("Incorrect bounds given."),
    glpk.EFAIL: RuntimeError("Solver failure."),
    glpk.EITLIM: StopIteration("Iteration limit exceeded."),
    glpk.ETMLIM: StopIteration("Time limit exceeded."),
    glpk.ENOPFS:
        StopIteration("Presolver: Problem has no primal feasible solution."),
    glpk.ENODFS:
        StopIteration("Presolver: Problem has no dual feasible solution.")
}

# simplex return codes (message)
cdef smretcode2str = {
    glpk.EOBJLL: "obj_ll reached",
    glpk.EOBJUL: "obj_ul reached",
    }

# interior point return codes (errors)
cdef iptretcode2error = {
    glpk.EFAIL: ValueError("The problem has no rows/columns."),
    glpk.ENOCVG: ArithmeticError("Very slow convergence or divergence."),
    glpk.EITLIM: StopIteration("Iteration limit exceeded."),
    glpk.EINSTAB: ArithmeticError("Numerical instability " +
                                  "on solving Newtonian system.")
    }

# integer optimization return codes (errors)
cdef ioretcode2error = {
    glpk.EBOUND: ValueError("Incorrect bounds given."),
    glpk.EROOT: ValueError("No optimal LP relaxation basis provided."),
    glpk.ENOPFS: ValueError("LP relaxation is infeasible."),
    glpk.ENODFS: ValueError("LP relaxation is unbounded."),
    glpk.EFAIL: RuntimeError("Solver failure."),
    glpk.EMIPGAP: StopIteration("Relative mip gap tolerance has been reached"),
    glpk.ETMLIM: StopIteration("Time limit exceeded."),
    glpk.ESTOP: StopIteration("Branch-and-cut callback terminated solver."),
    glpk.EDATA: ValueError("All problem parameters must be integer."),
    glpk.ERANGE: OverflowError("Integer overflow occurred when transforming " +
                               "to CNF SAT format.")
    }


# condition indicator:
cdef pair2condind = {
    (False, 'equalities'): glpk.KKT_PE,
    (False, 'bounds'): glpk.KKT_PB,
    (True, 'equalities'): glpk.KKT_DE,
    (True, 'bounds'): glpk.KKT_DB
}


# MPS file format
cdef str2mpsfmt = {
    'fixed': glpk.MPS_DECK,
    'free': glpk.MPS_FILE
    }


# problem coefficients
cdef _coeffscheck(coeffs):
    if not isinstance(coeffs, collections.abc.Mapping):
        raise TypeError("Coefficients must be passed in a 'Mapping', not " +
                        type(coeffs).__name__)
    if not all([isinstance(value, numbers.Real) for value in coeffs.values()]):
        raise TypeError("Coefficient values must be 'Real'.")


cdef class Problem:
    """A GLPK problem"""

    ### Object definition, creation, setup, and cleanup ###

    cdef glpk.ProbObj* _problem
    cdef public bint prefer_names
    """Switch to have methods return row and column names, if available"""

    def __cinit__(self):
        self._problem = glpk.create_prob()
        glpk.create_index(self._problem)
        self.prefer_names = False

    def _problem_ptr(self):
        """Encapsulate the pointer to the problem object

        The problem object pointer `self._problem` cannot be passed as such as
        an argument to other functions. Therefore we encapsulate it in a
        capsule that can be passed. It has to be unencapsulated after
        reception.

        """
        return PyCapsule_New(self._problem, NULL, NULL)

    def __dealloc__(self):
        glpk.delete_prob(self._problem)

    ### Translated GLPK functions ###

    def set_prob_name(self, str name):
        """Assign (change) problem name"""
        glpk.set_prob_name(self._problem, name2chars(name))

    def set_obj_name(self, str name):
        """Assign (change) objective function name"""
        glpk.set_obj_name(self._problem, name2chars(name))

    def set_obj_dir(self, str direction):
        """Set (change) optimization direction flag"""
        glpk.set_obj_dir(self._problem, str2optdir(direction))

    def add_rows(self, int number):
        """Add new rows to problem object"""
        glpk.add_rows(self._problem, number)

    def add_named_rows(self, *names):  # variant of add_rows
        """Add new rows to problem object

        :param names: the names (str strings) of the rows to add

        """
        cdef int number = len(names)
        if number is 0:
            return
        cdef int first = self.add_rows(number)
        for row, name in enumerate(names, start=first):
            glpk.set_row_name(self._problem, row, name2chars(name))

    def add_cols(self, int number):
        """Add new columns to problem object"""
        glpk.add_cols(self._problem, number)

    def add_named_cols(self, *names):  # variant of add_cols
        """Add new columns to problem object

        :param names: the names (str strings) of the columns to add

        """
        cdef int number = len(names)
        if number is 0:
            return
        cdef int first = self.add_cols(number)
        for col, name in enumerate(names, start=first):
            glpk.set_col_name(self._problem, col, name2chars(name))

    def set_row_name(self, row, str name):
        """Change row name"""
        row = self.find_row_as_needed(row)
        glpk.set_row_name(self._problem, row, name2chars(name))

    def set_col_name(self, col, str name):
        """Change column name"""
        col = self.find_col_as_needed(col)
        glpk.set_col_name(self._problem, col, name2chars(name))

    def set_row_bnds(self, row, lower, upper):
        """Set (change) row bounds"""
        row = self.find_row_as_needed(row)
        cdef int vartype
        cdef double lb
        cdef double ub
        vartype, lb, ub = _bounds(lower, upper)
        glpk.set_row_bnds(self._problem, row, vartype, lb, ub)

    def set_col_bnds(self, col, lower, upper):
        """Set (change) column bounds"""
        col = self.find_col_as_needed(col)
        cdef int vartype
        cdef double lb
        cdef double ub
        vartype, lb, ub = _bounds(lower, upper)
        glpk.set_col_bnds(self._problem, col, vartype, lb, ub)

    def set_obj_coef(self, col, double coeff):
        """Set (change) obj. coefficient"""
        col = self.find_col_as_needed(col)
        glpk.set_obj_coef(self._problem, col, coeff)

    def set_obj_const(self, double coeff):  # variant of set_obj_coef
        """Set (change) obj. constant term"""
        glpk.set_obj_coef(self._problem, 0, coeff)

    def set_mat_row(self, row, coeffs):
        """Set (replace) row of the constraint matrix

        :param coeffs: |Mapping| from column names (str strings) to
            coefficient values (|Real|).

        """
        _coeffscheck(coeffs)
        row = self.find_row_as_needed(row)
        cdef int k = len(coeffs)
        cdef int* cols = <int*>glpk.alloc(1+k, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+k, sizeof(double))
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                cols[i] = self.find_col_as_needed(item[0])
                vals[i] = item[1]
            glpk.set_mat_row(self._problem, row, k, cols, vals)
        finally:
            glpk.free(cols)
            glpk.free(vals)

    def clear_mat_row(self, row):  # variant of set_mat_row
        """Clear row of the constraint matrix"""
        row = self.find_row_as_needed(row)
        glpk.set_mat_row(self._problem, row, 0, NULL, NULL)

    def set_mat_col(self, col, coeffs):
        """Set (replace) column of the constraint matrix

        :param coeffs: |Mapping| from row names (str strings) to
            coefficient values (|Real|).

        """
        col = self.find_col_as_needed(col)
        _coeffscheck(coeffs)
        cdef int k = len(coeffs)
        cdef int* rows = <int*>glpk.alloc(1+k, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+k, sizeof(double))
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                rows[i] = self.find_row_as_needed(item[0])
                vals[i] = item[1]
            glpk.set_mat_col(self._problem, col, k, rows, vals)
        finally:
            glpk.free(rows)
            glpk.free(vals)

    def clear_mat_col(self, col):  # variant of set_mat_col
        """Clear column of the constraint matrix"""
        col = self.find_col_as_needed(col)
        glpk.set_mat_col(self._problem, col, 0, NULL, NULL)

    def load_matrix(self, coeffs):
        """Load (replace) the whole constraint matrix

        :param coeffs: |Mapping| from row and column name (str string)
            pairs (length-2 `tuple`) to coefficient values (|Real|).

        """
        _coeffscheck(coeffs)
        if not all(isinstance(key, tuple) and (len(key) is 2)
                   for key in coeffs.keys()):
            raise TypeError("Coefficient keys must be pairs, " +
                            "i.e., length-2 tuples.")
        cdef int k = len(coeffs)
        cdef int* rows = <int*>glpk.alloc(1+k, sizeof(int))
        cdef int* cols = <int*>glpk.alloc(1+k, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+k, sizeof(double))
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                rows[i] = self.find_row_as_needed(item[0][0])
                cols[i] = self.find_col_as_needed(item[0][1])
                vals[i] = item[1]
            glpk.load_matrix(self._problem, k, rows, cols, vals)
        finally:
            glpk.free(rows)
            glpk.free(cols)
            glpk.free(vals)

    def sort_matrix(self):
        """Sort elements of the constraint matrix"""
        glpk.sort_matrix(self._problem)

    def clear_matrix(self):  # variant of load_matrix
        """Clear the whole constraint matrix"""
        glpk.load_matrix(self._problem, 0, NULL, NULL, NULL)

    def del_rows(self, *rows):
        """Delete specified rows from problem object

        :param names: the names (str strings) of the rows to delete

        """
        cdef int k = len(rows)
        if k is 0:
            return
        cdef int* rowinds = <int*>glpk.alloc(1+k, sizeof(int))
        try:
            for i, row in enumerate(rows, start=1):
                rowinds[i] = self.find_row_as_needed(row)
            glpk.del_rows(self._problem, k, rowinds)
        finally:
            glpk.free(rowinds)

    def del_cols(self, *cols):
        """Delete specified columns from problem object

        :param names: the names (str strings) of the columns to delete

        """
        cdef int k = len(cols)
        if k is 0:
            return
        cdef int* colinds = <int*>glpk.alloc(1+k, sizeof(int))
        try:
            for i, col in enumerate(cols, start=1):
                colinds[i] = self.find_col_as_needed(col)
            glpk.del_cols(self._problem, k, colinds)
        finally:
            glpk.free(colinds)

    @classmethod
    def copy_prob(cls, Problem source, bint copy_names):
        """Copy problem object content"""
        problem = cls()
        cdef glpk.ProbObj* _problem = <glpk.ProbObj*>PyCapsule_GetPointer(
                                                problem._problem_ptr(), NULL)
        glpk.copy_prob(_problem, source._problem, copy_names)
        return problem

    def erase_prob(self):
        """Erase problem object content"""
        glpk.erase_prob(self._problem)

    def get_prob_name(self):
        """Retrieve problem name"""
        return chars2name(glpk.get_prob_name(self._problem))

    def get_obj_name(self):
        """Retrieve objective function name"""
        return chars2name(glpk.get_obj_name(self._problem))

    def get_obj_dir(self):
        """Retrieve optimization direction flag"""
        return optdir2str[glpk.get_obj_dir(self._problem)]

    def get_num_rows(self):
        """Retrieve number of rows"""
        return glpk.get_num_rows(self._problem)

    def get_num_cols(self):
        """Retrieve number of columns"""
        return glpk.get_num_cols(self._problem)

    def get_row_name(self, int row):
        """Retrieve row name"""
        return chars2name(glpk.get_row_name(self._problem, row))

    def get_row_name_if_available(self, int row):
        if self.prefer_names:
            name = self.get_row_name(row)
            return row if name is '' else name
        else:
            return row

    def get_col_name(self, int col):
        """Retrieve column name"""
        return chars2name(glpk.get_col_name(self._problem, col))

    def get_col_name_if_available(self, int col):
        if self.prefer_names:
            name = self.get_col_name(col)
            return col if name is '' else name
        else:
            return col

    def get_row_or_col_name(self, int ind):  # _get_row/col_name variant
        """Retrieve row or column name"""
        cdef int m = self.get_num_rows()
        if ind > m:  # column
            return 'col', self.get_col_name(ind - m)
        else:  # row
            return 'row', self.get_row_name(ind)

    def get_row_or_col_name_if_available(self, int ind):
        if self.prefer_names:
            row_or_col, name = self.get_row_or_col_name(ind)
            return row_or_col, ind if name is '' else row_or_col, name
        else:
            m = self.get_num_rows()
            if ind > m:  # column
                return 'col', ind - m
            else:  # row
                return 'row', ind

    def get_row_type(self, row):
        """Retrieve row type"""
        row = self.find_row_as_needed(row)
        return vartype2str[glpk.get_row_type(self._problem, row)]

    def get_row_lb(self, row):
        """Retrieve row lower bound"""
        row = self.find_row_as_needed(row)
        cdef double lb = glpk.get_row_lb(self._problem, row)
        return -float('inf') if lb == -DBL_MAX else lb

    def get_row_ub(self, row):
        """Retrieve row upper bound"""
        row = self.find_row_as_needed(row)
        cdef double ub = glpk.get_row_ub(self._problem, row)
        return -float('inf') if ub == -DBL_MAX else ub

    def get_col_type(self, col):
        """Retrieve column type"""
        col = self.find_col_as_needed(col)
        return vartype2str[glpk.get_col_type(self._problem, col)]

    def get_col_lb(self, col):
        """Retrieve column lower bound"""
        col = self.find_col_as_needed(col)
        cdef double lb = glpk.get_col_lb(self._problem, col)
        return -float('inf') if lb == -DBL_MAX else lb

    def get_col_ub(self, col):
        """Retrieve column upper bound"""
        col = self.find_col_as_needed(col)
        cdef double ub = glpk.get_col_ub(self._problem, col)
        return -float('inf') if ub == -DBL_MAX else ub

    def get_obj_coef(self, col):
        """Retrieve obj. coefficient"""
        col = self.find_col_as_needed(col)
        return glpk.get_obj_coef(self._problem, col)

    def get_obj_const(self):  # variant of get_obj_coef
        """Retrieve obj. constant term"""
        return glpk.get_obj_coef(self._problem, 0)

    def get_num_nz(self):
        """Retrieve number of constraint coefficients"""
        return glpk.get_num_nz(self._problem)

    def get_mat_row(self, row):
        """Retrieve row of the constraint matrix"""
        row = self.find_row_as_needed(row)
        cdef int n = self.get_num_cols()
        cdef int* cols =  <int*>glpk.alloc(1+n, sizeof(int))
        cdef double* vals =  <double*>glpk.alloc(1+n, sizeof(double))
        cdef int k
        try:
            k = glpk.get_mat_row(self._problem, row, cols, vals)
            coeffs = {self.get_col_name_if_available(cols[i]): vals[i]
                      for i in range(1, 1+k)}
        finally:
            glpk.free(cols)
            glpk.free(vals)
        return coeffs

    def get_mat_col(self, col):
        """Retrieve column of the constraint matrix"""
        col = self.find_col_as_needed(col)
        cdef int m = self.get_num_rows()
        cdef int* rows =  <int*>glpk.alloc(1+m, sizeof(int))
        cdef double* vals =  <double*>glpk.alloc(1+m, sizeof(double))
        cdef int k
        try:
            k = glpk.get_mat_col(self._problem, col, rows, vals)
            coeffs = {self.get_row_name_if_available(rows[i]): vals[i]
                      for i in range(1, 1+k)}
        finally:
            glpk.free(rows)
            glpk.free(vals)
        return coeffs

    def find_row(self, str name):
        """Find row by its name"""
        cdef int row = glpk.find_row(self._problem, name2chars(name))
        if row is 0:
            raise ValueError("'" + name + "' is not a row name.")
        else:
            return row

    def find_row_as_needed(self, row):
        if isinstance(row, int):
            return row
        elif isinstance(row, str):
            return self.find_row(row)
        else:
            raise TypeError("'row' must be a number ('int') or a name " +
                            "(str 'str'), not '" + type(row).__name__ +
                            "'.")

    def find_col(self, str name):
        """Find column by its name"""
        cdef int col = glpk.find_col(self._problem, name2chars(name))
        if col is 0:
            raise ValueError("'" + name + "' is not a column name.")
        else:
            return col

    def find_col_as_needed(self, col):
        if isinstance(col, int):
            return col
        elif isinstance(col, str):
            return self.find_col(col)
        else:
            raise TypeError("'col' must be a number ('int') or a name " +
                            "(str 'str'), not '" + type(col).__name__ +
                            "'.")

    def find_row_or_col_as_needed(self, ind):
        if isinstance(ind, int):
            return ind
        elif not isinstance(ind, tuple) or (len(ind) is not 2):
            raise TypeError("'ind' must be a number ('int') or a pair, not '" +
                            type(ind).__name__ + "'.")
        elif ind[0] is 'row':
            return self.find_row_as_needed(ind[1])
        elif ind[0] is 'col':
            return self.get_num_rows() + self.find_col_as_needed(ind[1])
        else:
            raise ValueError("'ind[0]' must be 'row' or 'col', not '" +
                             str(ind[0]) + "'.")

    def set_rii(self, row, double sf):
        """Set (change) row scale factor"""
        row = self.find_row_as_needed(row)
        glpk.set_rii(self._problem, row, sf)

    def set_sjj(self, col, double sf):
        """Set (change) column scale factor"""
        col = self.find_col_as_needed(col)
        glpk.set_sjj(self._problem, col, sf)

    def get_rii(self, row):
        """Retrieve row scale factor"""
        row = self.find_row_as_needed(row)
        return glpk.get_rii(self._problem, row)

    def get_sjj(self, col):
        """Retrieve column scale factor"""
        col = self.find_col_as_needed(col)
        return glpk.get_sjj(self._problem, col)

    def scale_prob(self, *algorithms):
        """Scale problem data

        :param algorithms: the algorithms to apply, one or more from `'auto'`:,
            `'skip'`, `'geometric'`, `'equilibration'`, `'round'`

        """
        if len(algorithms) is not 0:
            glpk.scale_prob(self._problem, sum(str2scalopt[algorithm]
                                               for algorithm in algorithms))

    def unscale_prob(self):
        """Unscale problem data"""
        glpk.unscale_prob(self._problem)

    def set_row_stat(self, row, str status):
        """Set (change) row status"""
        row = self.find_row_as_needed(row)
        _statuscheck(self.get_row_type(row), status)
        glpk.set_row_stat(self._problem, row, str2varstat[status])

    def set_col_stat(self, col, str status):
        """Set (change) column status"""
        col = self.find_col_as_needed(col)
        _statuscheck(self.get_col_type(col), status)
        glpk.set_col_stat(self._problem, col, str2varstat[status])

    def std_basis(self):
        """Construct standard initial LP basis"""
        glpk.std_basis(self._problem)

    def adv_basis(self):
        """Construct advanced initial LP basis"""
        glpk.adv_basis(self._problem, 0)

    def cpx_basis(self):
        """Construct Bixby's initial LP basis"""
        glpk.cpx_basis(self._problem)

    def simplex(self, SimplexControls controls):
        """Solve LP problem with the simplex method"""
        if ((controls.meth is not 'dual') and
            ((controls.obj_ll > -double_MAX) or
             (controls.obj_ul < +double_MAX))):
                raise ValueError("Objective function limits only with dual " +
                                 "simplex.")
        cdef int retcode = glpk.simplex(self._problem, &controls._smcp)
        if retcode is 0:
            return self.get_status()
        elif retcode in {glpk.EOBJLL, glpk.EOBJUL}:
            return smretcode2str[retcode]
        else:
            raise smretcode2error[retcode]

    def exact(self, SimplexControls controls):
        """Solve LP problem in exact arithmetic"""
        if controls.meth is not 'primal':
            raise ValueError("Only primal simplex with exact arithmetic.")
        cdef int retcode = glpk.exact(self._problem, &controls._smcp)
        if retcode is 0:
            return self.status()
        else:
            raise smretcode2error[retcode]

    def get_status(self):
        """Retrieve generic status of basic solution"""
        return solstat2str[glpk.get_status(self._problem)]

    def get_prim_stat(self):
        """Retrieve status of primal basic solution"""
        return solstat2str[glpk.get_prim_stat(self._problem)]

    def get_dual_stat(self):
        """Retrieve status of dual basic solution"""
        return solstat2str[glpk.get_dual_stat(self._problem)]

    def get_obj_val(self):
        """Retrieve objective value (basic solution)"""
        return glpk.get_obj_val(self._problem)

    def get_row_stat(self, row):
        """Retrieve row status"""
        row = self.find_row_as_needed(row)
        return varstat2str[glpk.get_row_stat(self._problem, row)]

    def get_row_prim(self, row):
        """Retrieve row primal value (basic solution)"""
        row = self.find_row_as_needed(row)
        return glpk.get_row_prim(self._problem, row)

    def get_row_dual(self, row):
        """Retrieve row dual value (basic solution)"""
        row = self.find_row_as_needed(row)
        return glpk.get_row_dual(self._problem, row)

    def get_col_stat(self, col):
        """Retrieve column status"""
        col = self.find_col_as_needed(col)
        return varstat2str[glpk.get_col_stat(self._problem, col)]

    def get_col_prim(self, col):
        """Retrieve column primal value (basic solution)"""
        col = self.find_col_as_needed(col)
        return glpk.get_col_prim(self._problem, col)

    def get_col_dual(self, col):
        """retrieve column dual value (basic solution)"""
        col = self.find_col_as_needed(col)
        return glpk.get_col_dual(self._problem, col)

    def get_unbnd_ray(self):
        """Determine variable causing unboundedness"""
        return self.get_row_or_col_name_if_available(
                                            glpk.get_unbnd_ray(self._problem))

    def interior(self, IPointControls controls):
        """Solve LP problem with the interior-point method"""
        cdef int retcode = glpk.interior(self._problem, &controls._iptcp)
        if retcode is 0:
            return self.ipt_status()
        else:
            raise iptretcode2error[retcode]

    def ipt_status(self):
        """Retrieve status of interior-point solution"""
        return solstat2str[glpk.ipt_status(self._problem)]

    def ipt_obj_val(self):
        """Retrieve objective value (interior point)"""
        return glpk.ipt_obj_val(self._problem)

    def ipt_row_prim(self, row):
        """Retrieve row primal value (interior point)"""
        row = self.find_row_as_needed(row)
        return glpk.ipt_row_prim(self._problem, row)

    def ipt_row_dual(self, row):
        """Retrieve row dual value (interior point)"""
        row = self.find_row_as_needed(row)
        return glpk.ipt_row_dual(self._problem, row)

    def ipt_col_prim(self, col):
        """Retrieve column primal value (interior point)"""
        col = self.find_col_as_needed(col)
        return glpk.ipt_col_prim(self._problem, col)

    def ipt_col_dual(self, col):
        """Retrieve column dual value (interior point)"""
        col = self.find_col_as_needed(col)
        return glpk.ipt_col_dual(self._problem, col)

    def set_col_kind(self, col, str kind):
        """Set (change) column kind"""
        col = self.find_col_as_needed(col)
        glpk.set_col_kind(self._problem, col, str2varkind[kind])

    def get_col_kind(self, col):
        """Retrieve column kind; returns varkind"""
        col = self.find_col_as_needed(col)
        return varkind2str[glpk.get_col_kind(self._problem, col)]

    def get_num_int(self):
        """Retrieve number of integer columns"""
        return glpk.get_num_int(self._problem)

    def get_num_bin(self):
        """Retrieve number of binary columns"""
        return glpk.get_num_bin(self._problem)

    def intopt(self, IntOptControls controls):
        """Solve MIP problem with the branch-and-bound method"""
        cdef int retcode = glpk.intopt(self._problem, &controls._iocp)
        if retcode is 0:
            return self.status()
        else:
            raise ioretcode2error[retcode]

    def mip_status(self):
        """Retrieve status of MIP solution"""
        return solstat2str[glpk.mip_status(self._problem)]

    def mip_obj_val(self):
        """Retrieve objective value (MIP solution)"""
        return glpk.mip_obj_val(self._problem)

    def mip_row_val(self, row):
        """Retrieve row value (MIP solution)"""
        row = self.find_row_as_needed(row)
        return glpk.mip_row_val(self._problem, row)

    def mip_col_val(self, col):
        """Retrieve column value (MIP solution)"""
        col = self.find_col_as_needed(col)
        return glpk.mip_col_val(self._problem, col)

    def check_kkt(self, str solver, str condition, bint dual=False):
        """Check feasibility/optimality conditions"""
        if dual and (solver is 'intopt'):
            raise ValueError("Dual conditions cannot be checked for the " +
                             "integer optimization solution.")
        cdef int sol = str2solind[solver]
        cdef int cond = pair2condind[(dual, condition)]
        cdef double ae_max
        cdef int ae_ind
        cdef double re_max
        cdef int re_ind
        glpk.check_kkt(self._problem, sol, cond,
                       &ae_max, &ae_ind, &re_max, &re_ind)
        if condition is 'equalities':
            if not dual:
                ae_id = 'row', self.get_row_name_if_available(ae_ind)
                re_id = 'row', self.get_row_name_if_available(re_ind)
            else:
                ae_id = 'col', self.get_col_name_if_available(ae_ind)
                re_id = 'col', self.get_col_name_if_available(re_ind)
        elif condition is 'bounds':
            ae_id = self.get_row_or_col_name_if_available(ae_ind)
            re_id = self.get_row_or_col_name_if_available(re_ind)
        else:
            raise ValueError("Condition is either 'equalities' or 'bounds'.")
        return {'abs': (ae_max, ae_id), 'rel': (re_max, re_id)}

    def print_sol(self, str fname):
        """Write basic solution in printable format"""
        retcode = glpk.print_sol(self._problem, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing printable basic solution file")

    def read_sol(self, str fname):
        """Read basic solution from text file"""
        retcode = glpk.read_sol(self._problem, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading basic solution file")

    def write_sol(self, str fname):
        """Write basic solution to text file"""
        retcode = glpk.write_sol(self._problem, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing basic solution file")

    def print_ranges(self, row_or_cols, str fname):
        """Print sensitivity analysis report"""
        if not isinstance(row_or_cols, collections.abc.Sequence):
            raise TypeError("'row_or_cols' must be a 'Sequence', not " +
                            type(row_or_cols).__name__ + ".")
        cdef int k = len(row_or_cols)
        cdef int* inds = <int*>glpk.alloc(1+k, sizeof(int))
        try:
            for i, row_or_col in enumerate(row_or_cols, start=1):
                inds[i] = self.find_row_or_col_as_needed(row_or_col)
            glpk.print_ranges(self._problem, k, inds, 0, str2chars(fname))
        finally:
            glpk.free(inds)

    def print_ipt(self, str fname):
        """Write interior-point solution in printable format"""
        retcode = glpk.print_ipt(self._problem, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing printable interior point " +
                               "solution file")

    def read_ipt(self, str fname):
        """Read interior-point solution from text file"""
        retcode = glpk.read_ipt(self._problem, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading interior point solution file")

    def write_ipt(self, str fname):
        """Write interior-point solution to text file"""
        retcode = glpk.write_ipt(self._problem, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing interior point solution file")

    def print_mip(self, str fname):
        """Write MIP solution in printable format"""
        retcode = glpk.print_mip(self._problem, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing printable integer " +
                               "optimization solution file")

    def read_mip(self, str fname):
        """Read MIP solution from text file"""
        retcode = glpk.read_mip(self._problem, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading integer optimization solution " +
                               "file")

    def write_mip(self, str fname):
        """Write MIP solution to text file"""
        retcode = glpk.write_mip(self._problem, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing integer optimization solution " +
                               "file")

    def bf_exists(self):
        """Check if LP basis factorization exists"""
        return glpk.bf_exists(self._problem)

    def factorize(self):
        """Compute LP basis factorization"""
        cdef int retcode = glpk.factorize(self._problem)
        if retcode is not 0:
            raise smretcode2error[retcode]

    def bf_updated(self):
        """Check if LP basis factorization has been updated"""
        return glpk.bf_updated(self._problem)

    def get_bfcp(self):
        """Retrieve LP basis factorization control parameters"""
        return FactorizationControls(self)

    def set_bfcp(self, FactorizationControls controls):
        """Change LP basis factorization control parameters"""
        glpk.set_bfcp(self._problem, &controls._bfcp)

    def get_bhead(self, int k):
        """Retrieve LP basis header information"""
        return self.get_row_or_col_name_if_available(
                                            glpk.get_bhead(self._problem, k))

    def get_row_bind(self, row):
        """Retrieve row index in the basis header"""
        row = self.find_row_as_needed(row)
        return glpk.get_row_bind(self._problem, row)

    def get_col_bind(self, col):
        """Retrieve column index in the basis header"""
        col = self.find_col_as_needed(col)
        return glpk.get_col_bind(self._problem, col)

    def ftran(self, tuple rhs):
        """Perform forward transformation (solve system B*x = b)"""
        if not all(isinstance(value, numbers.Real) for value in rhs):
            raise TypeError("Right-hand side must contain 'Real' numbers " +
                            "only.")
        cdef int m = self.get_num_rows()
        if len(rhs) is not m:
            raise ValueError("The right-hand side vector must have the same " +
                             "number of components as the basis, " + str(m) +
                             ".")
        cdef double* rhs_pre_x_post = <double*>glpk.alloc(1+m, sizeof(double))
        for i, value in enumerate(rhs, start=1):
            rhs_pre_x_post[i] = value
        glpk.ftran(self._problem, rhs_pre_x_post)
        return (rhs_pre_x_post[i] for i in range(1, 1+m))

    def btran(self, tuple rhs):
        """Perform backward transformation (solve system B'*x = b)"""
        if not all(isinstance(value, numbers.Real) for value in rhs):
            raise TypeError("Right-hand side must contain 'Real' numbers " +
                            "only.")
        cdef int m = self.get_num_rows()
        if len(rhs) is not m:
            raise ValueError("The right-hand side vector must have the same " +
                             "number of components as the basis, " + str(m) +
                             ".")
        cdef double* rhs_pre_x_post = <double*>glpk.alloc(1+m, sizeof(double))
        for i, value in enumerate(rhs, start=1):
            rhs_pre_x_post[i] = value
        glpk.btran(self._problem, rhs_pre_x_post)
        return (rhs_pre_x_post[i] for i in range(1, 1+m))

    def warm_up(self):
        """“Warm up” LP basis"""
        cdef int retcode = glpk.warm_up(self._problem)
        if retcode is not 0:
            raise smretcode2error[retcode]

    def eval_tab_row(self, ind):
        """Compute row of the simplex tableau"""
        ind = self.find_row_or_col_as_needed(ind)
        cdef int n = self.get_num_cols()
        cdef int* inds = <int*>glpk.alloc(1+n, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+n, sizeof(double))
        cdef int k
        try:
            k = glpk.eval_tab_row(self._problem, ind, inds, vals)
            return {self.get_row_or_col_name_if_available(inds[i]): vals[i]
                    for i in range(1, 1+k)}
        finally:
            glpk.free(inds)
            glpk.free(vals)

    def eval_tab_col(self, ind):
        """Compute column of the simplex tableau"""
        ind = self.find_row_or_col_as_needed(ind)
        cdef int m = self.get_num_rows()
        cdef int* inds = <int*>glpk.alloc(1+m, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+m, sizeof(double))
        cdef int k
        try:
            k = glpk.eval_tab_col(self._problem, ind, inds, vals)
            return {self.get_row_or_col_name_if_available(inds[i]): vals[i]
                    for i in range(1, 1+k)}
        finally:
            glpk.free(inds)
            glpk.free(vals)

    def transform_row(self, coeffs):
        """Transform explicitly specified row"""
        _coeffscheck(coeffs)
        cdef int n = self.get_num_cols()
        cdef int* inds = <int*>glpk.alloc(1+n, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+n, sizeof(double))
        cdef int k
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                inds[i] = self.find_col_as_needed(item[0])
                vals[i] = item[1]
            k = glpk.transform_row(self._problem, len(coeffs), inds, vals)
            return {self.get_row_or_col_name_if_available(inds[i]): vals[i]
                    for i in range(1, 1+k)}
        finally:
            glpk.free(inds)
            glpk.free(vals)

    def transform_col(self, coeffs):
        """Transform explicitly specified column"""
        _coeffscheck(coeffs)
        cdef int m = self.get_num_cols()
        cdef int* inds = <int*>glpk.alloc(1+m, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+m, sizeof(double))
        cdef int k
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                inds[i] = self.find_row_as_needed(item[0])
                vals[i] = item[1]
            k = glpk.transform_col(self._problem, len(coeffs), inds, vals)
            return {self.get_row_or_col_name_if_available(inds[i]): vals[i]
                    for i in range(1, 1+k)}
        finally:
            glpk.free(inds)
            glpk.free(vals)

    def prim_rtest(self, coeffs, int direction, double eps):
        """Perform primal ratio test"""
        _coeffscheck(coeffs)
        if direction not in {-1, +1}:
            raise ValueError("'direction' should be +1 or -1, not " +
                             str(direction) + ".")
        cdef int m = self.get_num_rows()
        cdef int* inds = <int*>glpk.alloc(1+m, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+m, sizeof(double))
        cdef int j
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                inds[i] = self.find_row_or_col_as_needed(item[0])
                vals[i] = item[1]
            j = glpk.prim_rtest(self._problem, len(coeffs), inds, vals,
                                direction, eps)
            return self.get_row_or_col_name_if_available(inds[j])
        finally:
            glpk.free(inds)
            glpk.free(vals)

    def dual_rtest(self, coeffs, int direction, double eps):
        """Perform dual ratio test"""
        _coeffscheck(coeffs)
        if direction not in {-1, +1}:
            raise ValueError("'direction' should be +1 or -1, not " +
                             str(direction) + ".")
        cdef int n = self.get_num_cols()
        cdef int* inds = <int*>glpk.alloc(1+n, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+n, sizeof(double))
        cdef int j
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                inds[i] = self.find_row_or_col_as_needed(item[0])
                vals[i] = item[1]
            j = glpk.dual_rtest(self._problem, len(coeffs), inds, vals,
                                direction, eps)
            return self.get_row_or_col_name_if_available(inds[j])
        finally:
            glpk.free(inds)
            glpk.free(vals)

    def analyze_bound(self, ind):
        """Analyze active bound of non-basic variable"""
        ind = self.find_row_or_col_as_needed(ind)
        cdef double min_bnd
        cdef int min_bnd_k
        cdef double max_bnd
        cdef int max_bnd_k
        glpk.analyze_bound(self._problem, ind,
                           &min_bnd, &min_bnd_k, &max_bnd, &max_bnd_k)
        if min_bnd > -DBL_MAX:
            minimal = min_bnd, self.get_row_or_col_name_if_available(min_bnd_k)
        else:
            minimal = (-float('inf'), None)
        if max_bnd < +DBL_MAX:
            maximal = max_bnd, self.get_row_or_col_name_if_available(max_bnd_k)
        else:
            maximal = (+float('inf'), None)
        return {'minimal': minimal, 'maximal': maximal}

    def analyze_coef(self, ind):
        """Analyze objective coefficient at basic variable"""
        ind = self.find_row_or_col_as_needed(ind)
        cdef double min_coef
        cdef int min_coef_k
        cdef double val_min_coef,
        cdef double max_coef
        cdef int max_coef_k
        cdef double val_max_coef
        glpk.analyze_coef(self._problem, ind,
                          &min_coef, &min_coef_k, &val_min_coef,
                          &max_coef, &max_coef_k, &val_max_coef)
        if val_min_coef <= -DBL_MAX:
            minval = -float('inf')
        elif val_min_coef >= DBL_MAX:
            minval = +float('inf')
        if val_max_coef <= -DBL_MAX:
            maxval = -float('inf')
        elif val_max_coef >= DBL_MAX:
            maxval = +float('inf')
        if min_coef > -DBL_MAX:
            minimal = (min_coef,
                       self.get_row_or_col_name_if_available(min_coef_k),
                       minval)
        else:
            minimal = -float('inf'), None, minval
        if max_coef < +DBL_MAX:
            maximal = (max_coef,
                       self.get_row_or_col_name_if_available(max_coef_k),
                       maxval)
        else:
            maximal = +float('inf'), None, maxval
        return {'minimal': minimal, 'maximal': maximal}

    @classmethod
    def read_mps(cls, str format, str fname):
        """Read problem data in MPS format"""
        problem = cls()
        cdef glpk.ProbObj* _problem = <glpk.ProbObj*>PyCapsule_GetPointer(
                                                problem._problem_ptr(), NULL)
        retcode = glpk.read_mps(_problem, str2mpsfmt[format], NULL,
                                str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading MPS file.")
        return problem

    def write_mps(self, str format, str fname):
        """Write problem data in MPS format"""
        retcode = glpk.write_mps(self._problem, str2mpsfmt[format], NULL,
                                 str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing MPS file.")

    @classmethod
    def read_lp(cls, str fname):
        """Read problem data in CPLEX LP format"""
        problem = cls()
        cdef glpk.ProbObj* _problem = <glpk.ProbObj*>PyCapsule_GetPointer(
                                                problem._problem_ptr(), NULL)
        retcode = glpk.read_lp(_problem, NULL, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading LP file.")
        return problem

    def write_lp(self, str fname):
        """Write problem data in CPLEX LP format"""
        retcode = glpk.write_lp(self._problem, NULL, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing LP file.")

    @classmethod
    def read_prob(cls, str fname):
        """Read problem data in GLPK format"""
        problem = cls()
        cdef glpk.ProbObj* _problem = <glpk.ProbObj*>PyCapsule_GetPointer(
                                                problem._problem_ptr(), NULL)
        retcode = glpk.read_prob(_problem, 0, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading GLPK file.")
        return problem

    def write_prob(self, str fname):
        """Write problem data in GLPK format"""
        retcode = glpk.write_prob(self._problem, 0, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing GLPK file.")

    @classmethod
    def read_cnfsat(cls, str fname):
        """Read CNF-SAT problem data in DIMACS format"""
        problem = cls()
        cdef glpk.ProbObj* _problem = <glpk.ProbObj*>PyCapsule_GetPointer(
                                                problem._problem_ptr(), NULL)
        retcode = glpk.read_cnfsat(_problem, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading CNF-SAT file.")
        return problem

    def check_cnfsat(self):
        """Check for CNF-SAT problem instance"""
        return not bool(glpk.check_cnfsat(self._problem))

    def write_cnfsat(self, str fname):
        """Write CNF-SAT problem data in DIMACS format"""
        retcode = glpk.write_cnfsat(self._problem, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing CNF-SAT file.")

    def minisat1(self):
        """Solve CNF-SAT problem with MiniSat solver"""
        cdef int retcode = glpk.minisat1(self._problem)
        if retcode is 0:
            return self.status()
        else:
            raise ioretcode2error[retcode]

    def intfeas1(self, bint use_bound, int obj_bound):
        """Solve integer feasibility problem"""
        cdef int retcode = glpk.intfeas1(self._problem, use_bound, obj_bound)
        if retcode is 0:
            return self.status()
        else:
            raise ioretcode2error[retcode]
