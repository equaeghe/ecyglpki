# ecyglpki-problem.pxi: Cython interface for GLPK problems

###############################################################################
#
#  This code is part of ecyglpki (a Cython GLPK interface).
#
#  Copyright (C) 2015 Erik Quaeghebeur. All rights reserved.
#
#  ecyglpki is free software: you can redistribute it and/or modify it under
#  the terms of the GNU General Public License as published by the Free
#  Software Foundation, either version 3 of the License, or (at your option)
#  any later version.
#
#  ecyglpki is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#  details.
#
#  You should have received a copy of the GNU General Public License
#  along with ecyglpki. If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################


from libc.limits cimport INT_MAX
cdef int int_MAX = int(INT_MAX)

from libc.float cimport DBL_MAX
cdef double double_MAX = DBL_MAX

# message levels
cdef str2msglev = {
    'no': glpk.MSG_OFF,
    'warnerror': glpk.MSG_ERR,
    'normal': glpk.MSG_ON,
    'full': glpk.MSG_ALL
    }
cdef msglev2str = {msg_lev: string for string, msg_lev in str2msglev.items()}


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
cdef str2vartype = {string: vartype for vartype, string in vartype2str.items()}

cdef _bounds(lower, upper):
    """Return ‘vartype’ and lower and upper bounds as real numbers

    :param lower: a real number or None (if unbounded from below)
    :type lower: |Real| or `NoneType`
    :param lower: a real number or None (if unbounded from above)
    :type lower: |Real| or `NoneType`
    :returns: a 3-tuple of ‘vartype’, lower bound, and upper bound
    :rtype: (`int`, |Real|, |Real|)

    """
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
    'bounded': frozenset(['lower', 'upper']),
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
    if not isinstance(coeffs, Mapping):
        raise TypeError("Coefficients must be passed in a 'Mapping', not " +
                        type(coeffs).__name__)
    if not all([isinstance(value, numbers.Real) for value in coeffs.values()]):
        raise TypeError("Coefficient values must be 'Real'.")


cdef class Problem:
    """A GLPK problem"""

    ### Object definition, creation, setup, and cleanup ###

    cdef glpk.Prob* _prob

    def __cinit__(self, prob_ptr=None):
        if prob_ptr is None:
            self._prob = glpk.create_prob()
            glpk.create_index(self._prob)
        else:
            self._prob = <glpk.Prob*>PyCapsule_GetPointer(prob_ptr(), NULL)

    def _prob_ptr(self):
        """Encapsulate the pointer to the problem object

        The problem object pointer `self._prob` cannot be passed as such as
        an argument to other functions. Therefore we encapsulate it in a
        capsule that can be passed. It has to be unencapsulated after
        reception.

        """
        return PyCapsule_New(self._prob, NULL, NULL)

    def __dealloc__(self):
        glpk.delete_index(self._prob)
        glpk.delete_prob(self._prob)

    ### Translated GLPK functions ###

    def set_prob_name(self, str name):
        """Assign (change) problem name

        :param name: the name of the problem, it may not exceed 255 bytes
            *encoded as UTF-8*
        :type name: `str`

        """
        glpk.set_prob_name(self._prob, name2chars(name))

    def set_obj_name(self, str name):
        """Assign (change) objective function name

        :param name: the name of the objective, it may not exceed 255 bytes
            *encoded as UTF-8*
        :type name: `str`

        """
        glpk.set_obj_name(self._prob, name2chars(name))

    def set_obj_dir(self, str direction):
        """Set (change) optimization direction flag

        :param direction: the objective direction,
            either `'minimize'` or `'maximize'`
        :type direction: `str`

        """
        glpk.set_obj_dir(self._prob, str2optdir[direction])

    def add_rows(self, int number):
        """Add new rows to problem object

        :param number: the number of rows to add
        :type number: `int`
        :returns: the index of the first row added
        :rtype: `int`

        """
        return glpk.add_rows(self._prob, number)

    def add_named_rows(self, *names):  # variant of add_rows
        """Add new rows to problem object

        :param names: the names of the rows to add, none may exceed 255 bytes
            *encoded as UTF-8*
        :type names: `tuple` of `str`

        """
        cdef int number = len(names)
        if number is 0:
            return
        cdef int first = self.add_rows(number)
        for row, name in enumerate(names, start=first):
            glpk.set_row_name(self._prob, row, name2chars(name))

    def add_cols(self, int number):
        """Add new columns to problem object

        :param number: the number of columns to add
        :type number: `int`
        :returns: the index of the first column added
        :rtype: `int`

        """
        return glpk.add_cols(self._prob, number)

    def add_named_cols(self, *names):  # variant of add_cols
        """Add new columns to problem object

        :param names: the names of the columns to add, none may exceed 255
            bytes *encoded as UTF-8*
        :type names: `tuple` of `str`

        """
        cdef int number = len(names)
        if number is 0:
            return
        cdef int first = self.add_cols(number)
        for col, name in enumerate(names, start=first):
            glpk.set_col_name(self._prob, col, name2chars(name))

    def set_row_name(self, row, str name):
        """Change row name

        :param row: the index or name of the row
        :type row: `int` or `str`
        :param name: the name of the row, it may not exceed 255 bytes
            *encoded as UTF-8*
        :type name: `str`

        """
        row = self.find_row_as_needed(row)
        glpk.set_row_name(self._prob, row, name2chars(name))

    def set_col_name(self, col, str name):
        """Change column name

        :param col: the index or name of the column
        :type col: `int` or `str`
        :param name: the name of the column, it may not exceed 255 bytes
            *encoded as UTF-8*
        :type name: `str`

        """
        col = self.find_col_as_needed(col)
        glpk.set_col_name(self._prob, col, name2chars(name))

    def set_row_bnds(self, row, lower, upper):
        """Set (change) row bounds

        :param row: the index or name of the row
        :type row: `int` or `str`
        :param lower: lower bound (`None` if unbounded from below)
        :type lower: |Real| or `NoneType`
        :param upper: upper bound (`None` if unbounded from above)
        :type upper: |Real| or `NoneType`

        """
        row = self.find_row_as_needed(row)
        cdef int vartype
        cdef double lb
        cdef double ub
        vartype, lb, ub = _bounds(lower, upper)
        glpk.set_row_bnds(self._prob, row, vartype, lb, ub)

    def set_col_bnds(self, col, lower, upper):
        """Set (change) column bounds

        :param col: the index or name of the column
        :type col: `int` or `str`
        :param lower: lower bound (`None` if unbounded from below)
        :type lower: |Real| or `NoneType`
        :param upper: upper bound (`None` if unbounded from above)
        :type upper: |Real| or `NoneType`

        """
        col = self.find_col_as_needed(col)
        cdef int vartype
        cdef double lb
        cdef double ub
        vartype, lb, ub = _bounds(lower, upper)
        glpk.set_col_bnds(self._prob, col, vartype, lb, ub)

    def set_obj_coef(self, col, double coeff):
        """Set (change) obj. coefficient

        :param col: the index or name of the column
        :type col: `int` or `str`
        :param coeff: the coefficient value
        :type coeff: |Real|

        """
        col = self.find_col_as_needed(col)
        glpk.set_obj_coef(self._prob, col, coeff)

    def set_obj_const(self, double coeff):  # variant of set_obj_coef
        """Set (change) obj. constant term

        :param coeff: the coefficient value
        :type coeff: |Real|

        """
        glpk.set_obj_coef(self._prob, 0, coeff)

    def set_mat_row(self, row, coeffs):
        """Set (replace) row of the constraint matrix

        :param row: the index or name of the row
        :type row: `int` or `str`
        :param coeffs: |Mapping| from column names (`str` strings) to
            coefficient values (|Real|)

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
            glpk.set_mat_row(self._prob, row, k, cols, vals)
        finally:
            glpk.free(cols)
            glpk.free(vals)

    def clear_mat_row(self, row):  # variant of set_mat_row
        """Clear row of the constraint matrix

        :param row: the index or name of the row
        :type row: `int` or `str`

        """
        row = self.find_row_as_needed(row)
        glpk.set_mat_row(self._prob, row, 0, NULL, NULL)

    def set_mat_col(self, col, coeffs):
        """Set (replace) column of the constraint matrix

        :param col: the index or name of the column
        :type col: `int` or `str`
        :param coeffs: |Mapping| from row names (`str` strings) to
            coefficient values (|Real|)

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
            glpk.set_mat_col(self._prob, col, k, rows, vals)
        finally:
            glpk.free(rows)
            glpk.free(vals)

    def clear_mat_col(self, col):  # variant of set_mat_col
        """Clear column of the constraint matrix

        :param col: the index or name of the column
        :type col: `int` or `str`

        """
        col = self.find_col_as_needed(col)
        glpk.set_mat_col(self._prob, col, 0, NULL, NULL)

    def load_matrix(self, coeffs):
        """Load (replace) the whole constraint matrix

        :param coeffs: |Mapping| from row and column name (str string)
            pairs (length-2 `tuple`) to coefficient values (|Real|)

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
            glpk.load_matrix(self._prob, k, rows, cols, vals)
        finally:
            glpk.free(rows)
            glpk.free(cols)
            glpk.free(vals)

    def clear_matrix(self):  # variant of load_matrix
        """Clear the whole constraint matrix"""
        glpk.load_matrix(self._prob, 0, NULL, NULL, NULL)

    def sort_matrix(self):
        """Sort elements of the constraint matrix"""
        glpk.sort_matrix(self._prob)

    def del_rows(self, *rows):
        """Delete specified rows from problem object

        :param rows: the indices or names of the rows
        :type rows: `tuple` of `int` or `str`

        """
        cdef int k = len(rows)
        if k is 0:
            return
        cdef int* rowinds = <int*>glpk.alloc(1+k, sizeof(int))
        try:
            for i, row in enumerate(rows, start=1):
                rowinds[i] = self.find_row_as_needed(row)
            glpk.del_rows(self._prob, k, rowinds)
        finally:
            glpk.free(rowinds)

    def del_cols(self, *cols):
        """Delete specified columns from problem object

        :param cols: the indices or the names of the columns
        :type cols: `tuple` of `int` or `str`

        """
        cdef int k = len(cols)
        if k is 0:
            return
        cdef int* colinds = <int*>glpk.alloc(1+k, sizeof(int))
        try:
            for i, col in enumerate(cols, start=1):
                colinds[i] = self.find_col_as_needed(col)
            glpk.del_cols(self._prob, k, colinds)
        finally:
            glpk.free(colinds)

    @classmethod # make it a normal method?
    def copy_prob(cls, Problem source, bint copy_names):
        """Copy problem object content

        :param source: the problem to copy
        :type source: `.Problem`
        :param copy_names: whether to copy problem component names or not
        :type copy_names: `bool`
        :returns: the copied problem
        :rtype: `.Problem`

        """
        problem = cls()
        cdef glpk.Prob* _prob = <glpk.Prob*>PyCapsule_GetPointer(
                                                    problem._prob_ptr(), NULL)
        glpk.copy_prob(_prob, source._prob, copy_names)
        return problem

    def erase_prob(self):
        """Erase problem object content"""
        glpk.erase_prob(self._prob)

    def get_prob_name(self):
        """Retrieve problem name

        :returns: the problem name
        :rtype: `str`

        """
        return chars2name(glpk.get_prob_name(self._prob))

    def get_obj_name(self):
        """Retrieve objective function name

        :returns: the objective name
        :rtype: `str`

        """
        return chars2name(glpk.get_obj_name(self._prob))

    def get_obj_dir(self):
        """Retrieve optimization direction flag

        :returns: the objective direction, either `'minimize'` or `'maximize'`
        :rtype: `str`

        """
        return optdir2str[glpk.get_obj_dir(self._prob)]

    def get_num_rows(self):
        """Retrieve number of rows

        :returns: the number of rows
        :rtype: `int`

        """
        return glpk.get_num_rows(self._prob)

    def get_num_cols(self):
        """Retrieve number of columns

        :returns: the number of columns
        :rtype: `int`

        """
        return glpk.get_num_cols(self._prob)

    def get_row_name(self, int row):
        """Retrieve row name

        :param row: the index of the row
        :type row: `int`
        :returns: the row name
        :rtype: `str`

        """
        return chars2name(glpk.get_row_name(self._prob, row))

    def get_row_name_if(self, int row, names_preferred=False):
        """

        :param row: the index of the row
        :type row: `int`
        :param names_preferred: whether to return the row name or index
        :type names_preferred: `bool`
        :returns: the row name or index
        :rtype: `str` or `int`

        """
        if names_preferred:
            name = self.get_row_name(row)
            return row if name is '' else name
        else:
            return row

    def get_col_name(self, int col):
        """Retrieve column name

        :param col: the index of the column
        :type col: `int`
        :returns: the problem name
        :rtype: `str`

        """
        return chars2name(glpk.get_col_name(self._prob, col))

    def get_col_name_if(self, int col, names_preferred=False):
        """

        :param col: the index of the column
        :type col: `int`
        :param names_preferred: whether to return the column name or index
        :type names_preferred: `bool`
        :returns: the column name or index
        :rtype: `str` or `int`

        """
        if names_preferred:
            name = self.get_col_name(col)
            return col if name is '' else name
        else:
            return col

    def get_row_or_col_name(self, int ind):  # _get_row/col_name variant
        """Retrieve row or column name

        :param ind: the row/column index
        :type ind: `int`
        :returns: a pair, either `'row'` or `'col'` and the row or column name
        :rtype: (`str`, `str`)

        """
        cdef int m = self.get_num_rows()
        if ind > m:  # column
            return 'col', self.get_col_name(ind - m)
        else:  # row
            return 'row', self.get_row_name(ind)

    def get_row_or_col_name_if(self, int ind, names_preferred=False):
        """

        :param ind: the row/column index
        :type ind: `int`
        :param names_preferred: whether to return the row or column
            name or index
        :type names_preferred: `bool`
        :returns: a pair, either `'row'` or `'col'` and the row or column name
        :rtype: (`str`, `str`)

        """
        if names_preferred:
            row_or_col, name = self.get_row_or_col_name(ind)
            return row_or_col, ind if name is '' else row_or_col, name
        else:
            m = self.get_num_rows()
            if ind > m:  # column
                return 'col', ind - m
            else:  # row
                return 'row', ind

    def get_row_type(self, row):
        """Retrieve row type

        :param row: the index or name of the row
        :type row: `int` or `str`
        :returns: the row type, either `'free'`, `'dominating'`, `'dominated'`,
            `'bounded'`, or `'fixed'`
        :rtype: `str`

        """
        row = self.find_row_as_needed(row)
        return vartype2str[glpk.get_row_type(self._prob, row)]

    def get_row_lb(self, row):
        """Retrieve row lower bound

        :param row: the index or name of the row
        :type row: `int` or `str`
        :returns: the lower bound
        :rtype: `float`

        """
        row = self.find_row_as_needed(row)
        cdef double lb = glpk.get_row_lb(self._prob, row)
        return -float('inf') if lb == -double_MAX else lb

    def get_row_ub(self, row):
        """Retrieve row upper bound

        :param row: the index or name of the row
        :type row: `int` or `str`
        :returns: the upper bound
        :rtype: `float`

        """
        row = self.find_row_as_needed(row)
        cdef double ub = glpk.get_row_ub(self._prob, row)
        return -float('inf') if ub == -double_MAX else ub

    def get_col_type(self, col):
        """Retrieve column type

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the column type, either `'free'`, `'dominating'`,
            `'dominated'`, `'bounded'`, or `'fixed'`
        :rtype: `str`

        """
        col = self.find_col_as_needed(col)
        return vartype2str[glpk.get_col_type(self._prob, col)]

    def get_col_lb(self, col):
        """Retrieve column lower bound

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the lower bound
        :rtype: `float`

        """
        col = self.find_col_as_needed(col)
        cdef double lb = glpk.get_col_lb(self._prob, col)
        return -float('inf') if lb == -double_MAX else lb

    def get_col_ub(self, col):
        """Retrieve column upper bound

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the upper bound
        :rtype: `float`

        """
        col = self.find_col_as_needed(col)
        cdef double ub = glpk.get_col_ub(self._prob, col)
        return -float('inf') if ub == -double_MAX else ub

    def get_obj_coef(self, col):
        """Retrieve obj. coefficient

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the coefficient value
        :rtype: `float`

        """
        col = self.find_col_as_needed(col)
        return glpk.get_obj_coef(self._prob, col)

    def get_obj_const(self):  # variant of get_obj_coef
        """Retrieve obj. constant term

        :returns: the constant value
        :rtype: `float`

        """
        return glpk.get_obj_coef(self._prob, 0)

    def get_num_nz(self):
        """Retrieve number of constraint coefficients

        :returns: the number of (non-zero) constraint coefficients
        :rtype: `int`

        """
        return glpk.get_num_nz(self._prob)

    def get_mat_row(self, row, names_preferred=False):
        """Retrieve row of the constraint matrix

        :param row: the index or name of the row
        :type row: `int` or `str`
        :param names_preferred: whether to return row and column
            names or indices
        :type names_preferred: `bool`
        :returns: the row
        :rtype coeffs: |Mapping| from column names (`str` strings) to
            coefficient values (|Real|)

        """
        row = self.find_row_as_needed(row)
        cdef int n = self.get_num_cols()
        cdef int* cols =  <int*>glpk.alloc(1+n, sizeof(int))
        cdef double* vals =  <double*>glpk.alloc(1+n, sizeof(double))
        cdef int k
        try:
            k = glpk.get_mat_row(self._prob, row, cols, vals)
            coeffs = {self.get_col_name_if(cols[i], names_preferred): vals[i]
                      for i in range(1, 1+k)}
        finally:
            glpk.free(cols)
            glpk.free(vals)
        return coeffs

    def get_mat_col(self, col, names_preferred=False):
        """Retrieve column of the constraint matrix

        :param col: the index or name of the column
        :type col: `int` or `str`
        :param names_preferred: whether to return row and column
            names or indices
        :type names_preferred: `bool`
        :returns: the column
        :rtype coeffs: |Mapping| from row names (`str` strings) to
            coefficient values (|Real|)

        """
        col = self.find_col_as_needed(col)
        cdef int m = self.get_num_rows()
        cdef int* rows =  <int*>glpk.alloc(1+m, sizeof(int))
        cdef double* vals =  <double*>glpk.alloc(1+m, sizeof(double))
        cdef int k
        try:
            k = glpk.get_mat_col(self._prob, col, rows, vals)
            coeffs = {self.get_row_name_if(rows[i], names_preferred): vals[i]
                      for i in range(1, 1+k)}
        finally:
            glpk.free(rows)
            glpk.free(vals)
        return coeffs

    def find_row(self, str name):
        """Find row by its name

        :param name: the name of the row
        :type name: `str`
        :returns: the row index
        :rtype: `int`

        """
        cdef int row = glpk.find_row(self._prob, name2chars(name))
        if row is 0:
            raise ValueError("'" + name + "' is not a row name.")
        return row

    def find_row_as_needed(self, row):
        """

        :param row: the index or name of the row
        :type row: `int` or `str`
        :returns: the row index
        :rtype: `int`

        """
        if isinstance(row, int):
            return row
        elif isinstance(row, str):
            return self.find_row(row)
        else:
            raise TypeError("'row' must be a number ('int') or a name " +
                            "(str 'str'), not '" + type(row).__name__ +
                            "'.")

    def find_col(self, str name):
        """Find column by its name

        :param name: the name of the column
        :type name: `str`
        :returns: the column index
        :rtype: `int`

        """
        cdef int col = glpk.find_col(self._prob, name2chars(name))
        if col is 0:
            raise ValueError("'" + name + "' is not a column name.")
        return col

    def find_col_as_needed(self, col):
        """

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the column index
        :rtype: `int`

        """
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
        """Set (change) row scale factor

        :param row: the index or name of the row
        :type row: `int` or `str`
        :param sf: the scale factor
        :type sf: |Real|

        """
        row = self.find_row_as_needed(row)
        glpk.set_rii(self._prob, row, sf)

    def set_sjj(self, col, double sf):
        """Set (change) column scale factor

        :param col: the index or name of the column
        :type col: `int` or `str`
        :param sf: the scale factor
        :type sf: |Real|

        """
        col = self.find_col_as_needed(col)
        glpk.set_sjj(self._prob, col, sf)

    def get_rii(self, row):
        """Retrieve row scale factor

        :param row: the index or name of the row
        :type row: `int` or `str`
        :returns: the scale factor
        :rtype: `float`

        """
        row = self.find_row_as_needed(row)
        return glpk.get_rii(self._prob, row)

    def get_sjj(self, col):
        """Retrieve column scale factor

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the scale factor
        :rtype: `float`

        """
        col = self.find_col_as_needed(col)
        return glpk.get_sjj(self._prob, col)

    def scale_prob(self, *algorithms):
        """Scale problem data

        :param algorithms: the algorithms to apply, one or more from `'auto'`,
            `'skip'`, `'geometric'`, `'equilibration'`, `'round'`

        """
        if len(algorithms) is not 0:
            glpk.scale_prob(self._prob, sum(str2scalopt[algorithm]
                                            for algorithm in algorithms))

    def unscale_prob(self):
        """Unscale problem data"""
        glpk.unscale_prob(self._prob)

    def set_row_stat(self, row, str status):
        """Set (change) row status

        :param row: the index or name of the row
        :type row: `int` or `str`

        """
        row = self.find_row_as_needed(row)
        _statuscheck(self.get_row_type(row), status)
        glpk.set_row_stat(self._prob, row, str2varstat[status])

    def set_col_stat(self, col, str status):
        """Set (change) column status

        :param col: the index or name of the column
        :type col: `int` or `str`

        """
        col = self.find_col_as_needed(col)
        _statuscheck(self.get_col_type(col), status)
        glpk.set_col_stat(self._prob, col, str2varstat[status])

    def std_basis(self):
        """Construct standard initial LP basis"""
        glpk.std_basis(self._prob)

    def adv_basis(self):
        """Construct advanced initial LP basis"""
        glpk.adv_basis(self._prob, 0)

    def cpx_basis(self):
        """Construct Bixby's initial LP basis"""
        glpk.cpx_basis(self._prob)

    def simplex(self, SimplexControls controls):
        """Solve LP problem with the simplex method"""
        if ((controls.meth is not 'dual') and
            ((controls.obj_ll > -double_MAX) or
             (controls.obj_ul < +double_MAX))):
                raise ValueError("Objective function limits only with dual " +
                                 "simplex.")
        cdef int retcode = glpk.simplex(self._prob, &controls._smcp)
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
        cdef int retcode = glpk.exact(self._prob, &controls._smcp)
        if retcode is not 0:
            raise smretcode2error[retcode]
        return self.get_status()

    def get_status(self):
        """Retrieve generic status of basic solution"""
        return solstat2str[glpk.get_status(self._prob)]

    def get_prim_stat(self):
        """Retrieve status of primal basic solution"""
        return solstat2str[glpk.get_prim_stat(self._prob)]

    def get_dual_stat(self):
        """Retrieve status of dual basic solution"""
        return solstat2str[glpk.get_dual_stat(self._prob)]

    def get_obj_val(self):
        """Retrieve objective value (basic solution)

        :returns: the objective value of the basic solution
        :rtype: `float`

        """
        return glpk.get_obj_val(self._prob)

    def get_row_stat(self, row):
        """Retrieve row status

        :param row: the index or name of the row
        :type row: `int` or `str`
        :returns: the row status, either `'basic'`, `'lower'`, `'upper'`,
            `'free'`, or `'fixed'`
        :rtype: `str`

        """
        row = self.find_row_as_needed(row)
        return varstat2str[glpk.get_row_stat(self._prob, row)]

    def get_row_prim(self, row):
        """Retrieve row primal value (basic solution)

        :param row: the index or name of the row
        :type row: `int` or `str`
        :returns: the solution value
        :rtype: `float`

        """
        row = self.find_row_as_needed(row)
        return glpk.get_row_prim(self._prob, row)

    def get_row_dual(self, row):
        """Retrieve row dual value (basic solution)

        :param row: the index or name of the row
        :type row: `int` or `str`
        :returns: the solution value
        :rtype: `float`

        """
        row = self.find_row_as_needed(row)
        return glpk.get_row_dual(self._prob, row)

    def get_col_stat(self, col):
        """Retrieve column status

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the column status, either `'basic'`, `'lower'`, `'upper'`,
            `'free'`, or `'fixed'`
        :rtype: `str`

        """
        col = self.find_col_as_needed(col)
        return varstat2str[glpk.get_col_stat(self._prob, col)]

    def get_col_prim(self, col):
        """Retrieve column primal value (basic solution)

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the solution value
        :rtype: `float`

        """
        col = self.find_col_as_needed(col)
        return glpk.get_col_prim(self._prob, col)

    def get_col_dual(self, col):
        """retrieve column dual value (basic solution)

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the solution value
        :rtype: `float`

        """
        col = self.find_col_as_needed(col)
        return glpk.get_col_dual(self._prob, col)

    def get_unbnd_ray(self, names_preferred=False):
        """Determine variable causing unboundedness

        :param names_preferred: whether to return the row or column
            name or index
        :type names_preferred: `bool`

        """
        return self.get_row_or_col_name_if(glpk.get_unbnd_ray(self._prob),
                                           names_preferred)

    def interior(self, IPointControls controls):
        """Solve LP problem with the interior-point method"""
        cdef int retcode = glpk.interior(self._prob, &controls._iptcp)
        if retcode is not 0:
            raise iptretcode2error[retcode]
        return self.ipt_status()

    def ipt_status(self):
        """Retrieve status of interior-point solution"""
        return solstat2str[glpk.ipt_status(self._prob)]

    def ipt_obj_val(self):
        """Retrieve objective value (interior point)

        :returns: the objective value of the interior point solution
        :rtype: `float`

        """
        return glpk.ipt_obj_val(self._prob)

    def ipt_row_prim(self, row):
        """Retrieve row primal value (interior point)

        :param row: the index or name of the row
        :type row: `int` or `str`
        :returns: the solution value
        :rtype: `float`

        """
        row = self.find_row_as_needed(row)
        return glpk.ipt_row_prim(self._prob, row)

    def ipt_row_dual(self, row):
        """Retrieve row dual value (interior point)

        :param row: the index or name of the row
        :type row: `int` or `str`
        :returns: the solution value
        :rtype: `float`

        """
        row = self.find_row_as_needed(row)
        return glpk.ipt_row_dual(self._prob, row)

    def ipt_col_prim(self, col):
        """Retrieve column primal value (interior point)

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the solution value
        :rtype: `float`

        """
        col = self.find_col_as_needed(col)
        return glpk.ipt_col_prim(self._prob, col)

    def ipt_col_dual(self, col):
        """Retrieve column dual value (interior point)

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the solution value
        :rtype: `float`

        """
        col = self.find_col_as_needed(col)
        return glpk.ipt_col_dual(self._prob, col)

    def set_col_kind(self, col, str kind):
        """Set (change) column kind

        :param col: the index or name of the column
        :type col: `int` or `str`
        :param kind: the column kind, either `'continuous'`, `'integer'`,
            or `'binary'`
        :type kind: `str`

        """
        col = self.find_col_as_needed(col)
        glpk.set_col_kind(self._prob, col, str2varkind[kind])

    def get_col_kind(self, col):
        """Retrieve column kind

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the column kind, either `'continuous'`, `'integer'`,
            or `'binary'`
        :rtype: `str`

        """
        col = self.find_col_as_needed(col)
        return varkind2str[glpk.get_col_kind(self._prob, col)]

    def get_num_int(self):
        """Retrieve number of integer columns

        :returns: the number of integer columns
        :rtype: `int`

        """
        return glpk.get_num_int(self._prob)

    def get_num_bin(self):
        """Retrieve number of binary columns

        :returns: the number of binary columns
        :rtype: `int`

        """
        return glpk.get_num_bin(self._prob)

    def intopt(self, IntOptControls controls):
        """Solve MIP problem with the branch-and-bound method"""
        cdef int retcode = glpk.intopt(self._prob, &controls._iocp)
        if retcode is not 0:
            raise ioretcode2error[retcode]
        return self.mip_status()

    def mip_status(self):
        """Retrieve status of MIP solution"""
        return solstat2str[glpk.mip_status(self._prob)]

    def mip_obj_val(self):
        """Retrieve objective value (MIP solution)

        :returns: the objective value of the MIP solution
        :rtype: `float`

        """
        return glpk.mip_obj_val(self._prob)

    def mip_row_val(self, row):
        """Retrieve row value (MIP solution)

        :param row: the index or name of the row
        :type row: `int` or `str`
        :returns: the solution value
        :rtype: `float`

        """
        row = self.find_row_as_needed(row)
        return glpk.mip_row_val(self._prob, row)

    def mip_col_val(self, col):
        """Retrieve column value (MIP solution)

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: the solution value
        :rtype: `float`

        """
        col = self.find_col_as_needed(col)
        return glpk.mip_col_val(self._prob, col)

    def check_kkt(self, str solver, str condition, bint dual=False,
                  names_preferred=False):
        """Check feasibility/optimality conditions

        :param names_preferred: whether to return the row or column
            name or index
        :type names_preferred: `bool`

        """
        if dual and (solver is 'intopt'):
            raise ValueError("Dual conditions cannot be checked for the " +
                             "integer optimization solution.")
        cdef int sol = str2solind[solver]
        cdef int cond = pair2condind[(dual, condition)]
        cdef double ae_max
        cdef int ae_ind
        cdef double re_max
        cdef int re_ind
        glpk.check_kkt(self._prob, sol, cond,
                       &ae_max, &ae_ind, &re_max, &re_ind)
        if condition is 'equalities':
            if not dual:
                ae_id = 'row', self.get_row_name_if(ae_ind, names_preferred)
                re_id = 'row', self.get_row_name_if(re_ind, names_preferred)
            else:
                ae_id = 'col', self.get_col_name_if(ae_ind, names_preferred)
                re_id = 'col', self.get_col_name_if(re_ind, names_preferred)
        elif condition is 'bounds':
            ae_id = self.get_row_or_col_name_if(ae_ind, names_preferred)
            re_id = self.get_row_or_col_name_if(re_ind, names_preferred)
        else:
            raise ValueError("Condition is either 'equalities' or 'bounds'.")
        return {'abs': (ae_max, ae_id), 'rel': (re_max, re_id)}

    def print_sol(self, str fname):
        """Write basic solution in printable format

        :param fname: file name
        :type fname: `str`

        """
        retcode = glpk.print_sol(self._prob, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing printable basic solution file")

    def read_sol(self, str fname):
        """Read basic solution from text file

        :param fname: file name
        :type fname: `str`

        """
        retcode = glpk.read_sol(self._prob, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading basic solution file")

    def write_sol(self, str fname):
        """Write basic solution to text file

        :param fname: file name
        :type fname: `str`

        """
        retcode = glpk.write_sol(self._prob, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing basic solution file")

    def print_ranges(self, row_or_cols, str fname):
        """Print sensitivity analysis report

        :param fname: file name
        :type fname: `str`

        """
        if not isinstance(row_or_cols, Sequence):
            raise TypeError("'row_or_cols' must be a 'Sequence', not " +
                            type(row_or_cols).__name__ + ".")
        cdef int k = len(row_or_cols)
        cdef int* inds = <int*>glpk.alloc(1+k, sizeof(int))
        try:
            for i, row_or_col in enumerate(row_or_cols, start=1):
                inds[i] = self.find_row_or_col_as_needed(row_or_col)
            glpk.print_ranges(self._prob, k, inds, 0, str2chars(fname))
        finally:
            glpk.free(inds)

    def print_ipt(self, str fname):
        """Write interior-point solution in printable format

        :param fname: file name
        :type fname: `str`

        """
        retcode = glpk.print_ipt(self._prob, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing printable interior point " +
                               "solution file")

    def read_ipt(self, str fname):
        """Read interior-point solution from text file

        :param fname: file name
        :type fname: `str`

        """
        retcode = glpk.read_ipt(self._prob, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading interior point solution file")

    def write_ipt(self, str fname):
        """Write interior-point solution to text file

        :param fname: file name
        :type fname: `str`

        """
        retcode = glpk.write_ipt(self._prob, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing interior point solution file")

    def print_mip(self, str fname):
        """Write MIP solution in printable format

        :param fname: file name
        :type fname: `str`

        """
        retcode = glpk.print_mip(self._prob, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing printable integer " +
                               "optimization solution file")

    def read_mip(self, str fname):
        """Read MIP solution from text file

        :param fname: file name
        :type fname: `str`

        """
        retcode = glpk.read_mip(self._prob, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading integer optimization solution " +
                               "file")

    def write_mip(self, str fname):
        """Write MIP solution to text file

        :param fname: file name
        :type fname: `str`

        """
        retcode = glpk.write_mip(self._prob, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing integer optimization solution " +
                               "file")

    def bf_exists(self):
        """Check if LP basis factorization exists

        :rtype: `bool`

        """
        return glpk.bf_exists(self._prob)

    def factorize(self):
        """Compute LP basis factorization"""
        cdef int retcode = glpk.factorize(self._prob)
        if retcode is not 0:
            raise smretcode2error[retcode]

    def bf_updated(self):
        """Check if LP basis factorization has been updated

        :rtype: `bool`

        """
        return glpk.bf_updated(self._prob)

    def get_bfcp(self):
        """Retrieve LP basis factorization control parameters

        :returns: basis factorization control parameter object
        :rtype: `.FactorizationControls`

        """
        return FactorizationControls(self)

    def set_bfcp(self, FactorizationControls controls):
        """Change LP basis factorization control parameters"""
        glpk.set_bfcp(self._prob, &controls._bfcp)

    def get_bhead(self, int k, names_preferred=False):
        """Retrieve LP basis header information

        :param names_preferred: whether to return the row or column
            name or index
        :type names_preferred: `bool`

        """
        return self.get_row_or_col_name_if(glpk.get_bhead(self._prob, k),
                                           names_preferred)

    def get_row_bind(self, row):
        """Retrieve row index in the basis header

        :param row: the index or name of the row
        :type row: `int` or `str`
        :returns: basis header index
        :rtype: `int`

        """
        row = self.find_row_as_needed(row)
        return glpk.get_row_bind(self._prob, row)

    def get_col_bind(self, col):
        """Retrieve column index in the basis header

        :param col: the index or name of the column
        :type col: `int` or `str`
        :returns: basis header index
        :rtype: `int`

        """
        col = self.find_col_as_needed(col)
        return glpk.get_col_bind(self._prob, col)

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
        glpk.ftran(self._prob, rhs_pre_x_post)
        return (rhs_pre_x_post[i] for i in range(1, 1+m))

    def btran(self, tuple rhs):
        """Perform backward transformation (solve system B'x = b)"""
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
        glpk.btran(self._prob, rhs_pre_x_post)
        return (rhs_pre_x_post[i] for i in range(1, 1+m))

    def warm_up(self):
        """“Warm up” LP basis"""
        cdef int retcode = glpk.warm_up(self._prob)
        if retcode is not 0:
            raise smretcode2error[retcode]

    def eval_tab_row(self, ind, names_preferred=False):
        """Compute row of the simplex tableau

        :param names_preferred: whether to return row and column
            names or indices
        :type names_preferred: `bool`

        """
        ind = self.find_row_or_col_as_needed(ind)
        cdef int n = self.get_num_cols()
        cdef int* inds = <int*>glpk.alloc(1+n, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+n, sizeof(double))
        cdef int k
        try:
            k = glpk.eval_tab_row(self._prob, ind, inds, vals)
            return {self.get_row_or_col_name_if(inds[i], names_preferred):
                        vals[i] for i in range(1, 1+k)}
        finally:
            glpk.free(inds)
            glpk.free(vals)

    def eval_tab_col(self, ind, names_preferred=False):
        """Compute column of the simplex tableau

        :param names_preferred: whether to return row and column
            names or indices
        :type names_preferred: `bool`

        """
        ind = self.find_row_or_col_as_needed(ind)
        cdef int m = self.get_num_rows()
        cdef int* inds = <int*>glpk.alloc(1+m, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+m, sizeof(double))
        cdef int k
        try:
            k = glpk.eval_tab_col(self._prob, ind, inds, vals)
            return {self.get_row_or_col_name_if(inds[i], names_preferred):
                        vals[i] for i in range(1, 1+k)}
        finally:
            glpk.free(inds)
            glpk.free(vals)

    def transform_row(self, coeffs, names_preferred=False):
        """Transform explicitly specified row

        :param coeffs: |Mapping| from column names (`str` strings) to
            coefficient values (|Real|)
        :param names_preferred: whether to return row and column
            names or indices
        :type names_preferred: `bool`
        :returns: transformed row
        :rtype: |Mapping| from row and column names (`str` strings) to
            coefficient values (|Real|)

        """
        _coeffscheck(coeffs)
        cdef int n = self.get_num_cols()
        cdef int* inds = <int*>glpk.alloc(1+n, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+n, sizeof(double))
        cdef int k
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                inds[i] = self.find_col_as_needed(item[0])
                vals[i] = item[1]
            k = glpk.transform_row(self._prob, len(coeffs), inds, vals)
            return {self.get_row_or_col_name_if(inds[i], names_preferred):
                        vals[i] for i in range(1, 1+k)}
        finally:
            glpk.free(inds)
            glpk.free(vals)

    def transform_col(self, coeffs, names_preferred=False):
        """Transform explicitly specified column

        :param coeffs: |Mapping| from row names (`str` strings) to
            coefficient values (|Real|)
        :param names_preferred: whether to return row and column
            names or indices
        :type names_preferred: `bool`
        :returns: transformed column
        :rtype: |Mapping| from row and column names (`str` strings) to
            coefficient values (|Real|)

        """
        _coeffscheck(coeffs)
        cdef int m = self.get_num_cols()
        cdef int* inds = <int*>glpk.alloc(1+m, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+m, sizeof(double))
        cdef int k
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                inds[i] = self.find_row_as_needed(item[0])
                vals[i] = item[1]
            k = glpk.transform_col(self._prob, len(coeffs), inds, vals)
            return {self.get_row_or_col_name_if(inds[i], names_preferred):
                        vals[i] for i in range(1, 1+k)}
        finally:
            glpk.free(inds)
            glpk.free(vals)

    def prim_rtest(self, coeffs, int direction, double eps,
                   names_preferred=False):
        """Perform primal ratio test

        :param names_preferred: whether to return row and column
            names or indices
        :type names_preferred: `bool`

        """
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
            j = glpk.prim_rtest(self._prob, len(coeffs), inds, vals,
                                direction, eps)
            return self.get_row_or_col_name_if(inds[j], names_preferred)
        finally:
            glpk.free(inds)
            glpk.free(vals)

    def dual_rtest(self, coeffs, int direction, double eps,
                   names_preferred=False):
        """Perform dual ratio test

        :param names_preferred: whether to return row and column
            names or indices
        :type names_preferred: `bool`

        """
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
            j = glpk.dual_rtest(self._prob, len(coeffs), inds, vals,
                                direction, eps)
            return self.get_row_or_col_name_if(inds[j], names_preferred)
        finally:
            glpk.free(inds)
            glpk.free(vals)

    def analyze_bound(self, ind, names_preferred=False):
        """Analyze active bound of non-basic variable

        :param names_preferred: whether to return row and column
            names or indices
        :type names_preferred: `bool`

        """
        ind = self.find_row_or_col_as_needed(ind)
        cdef double min_bnd
        cdef int min_bnd_k
        cdef double max_bnd
        cdef int max_bnd_k
        glpk.analyze_bound(self._prob, ind,
                           &min_bnd, &min_bnd_k, &max_bnd, &max_bnd_k)
        if min_bnd > -double_MAX:
            minimal = min_bnd, self.get_row_or_col_name_if(min_bnd_k,
                                                           names_preferred)
        else:
            minimal = (-float('inf'), None)
        if max_bnd < +double_MAX:
            maximal = max_bnd, self.get_row_or_col_name_if(max_bnd_k,
                                                           names_preferred)
        else:
            maximal = (+float('inf'), None)
        return {'minimal': minimal, 'maximal': maximal}

    def analyze_coef(self, ind, names_preferred=False):
        """Analyze objective coefficient at basic variable

        :param names_preferred: whether to return row and column
            names or indices
        :type names_preferred: `bool`

        """
        ind = self.find_row_or_col_as_needed(ind)
        cdef double min_coef
        cdef int min_coef_k
        cdef double val_min_coef,
        cdef double max_coef
        cdef int max_coef_k
        cdef double val_max_coef
        glpk.analyze_coef(self._prob, ind,
                          &min_coef, &min_coef_k, &val_min_coef,
                          &max_coef, &max_coef_k, &val_max_coef)
        if val_min_coef <= -double_MAX:
            minval = -float('inf')
        elif val_min_coef >= double_MAX:
            minval = +float('inf')
        if val_max_coef <= -double_MAX:
            maxval = -float('inf')
        elif val_max_coef >= double_MAX:
            maxval = +float('inf')
        if min_coef > -double_MAX:
            minimal = (min_coef,
                       self.get_row_or_col_name_if(min_coef_k,
                                                   names_preferred),
                       minval)
        else:
            minimal = -float('inf'), None, minval
        if max_coef < +double_MAX:
            maximal = (max_coef,
                       self.get_row_or_col_name_if(max_coef_k,
                                                   names_preferred),
                       maxval)
        else:
            maximal = +float('inf'), None, maxval
        return {'minimal': minimal, 'maximal': maximal}

    @classmethod
    def read_mps(cls, str format, str fname):
        """Read problem data in MPS format

        :param format: the MPS file format, either `'fixed'` or `'free'`
        :type format: `str`
        :param fname: file name
        :type fname: `str`
        :returns: the problem read
        :rtype: `.Problem`

        """
        problem = cls()
        cdef glpk.Prob* _prob = <glpk.Prob*>PyCapsule_GetPointer(
                                                    problem._prob_ptr(), NULL)
        retcode = glpk.read_mps(_prob, str2mpsfmt[format], NULL,
                                str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading MPS file.")
        return problem

    def write_mps(self, str format, str fname):
        """Write problem data in MPS format

        :param format: the MPS file format, either `'fixed'` or `'free'`
        :type format: `str`
        :param fname: file name
        :type fname: `str`

        """
        retcode = glpk.write_mps(self._prob, str2mpsfmt[format], NULL,
                                 str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing MPS file.")

    @classmethod
    def read_lp(cls, str fname):
        """Read problem data in CPLEX LP format

        :param fname: file name
        :type fname: `str`
        :returns: the problem read
        :rtype: `.Problem`

        """
        problem = cls()
        cdef glpk.Prob* _prob = <glpk.Prob*>PyCapsule_GetPointer(
                                                    problem._prob_ptr(), NULL)
        retcode = glpk.read_lp(_prob, NULL, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading LP file.")
        return problem

    def write_lp(self, str fname):
        """Write problem data in CPLEX LP format

        :param fname: file name
        :type fname: `str`

        """
        retcode = glpk.write_lp(self._prob, NULL, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing LP file.")

    @classmethod
    def read_prob(cls, str fname):
        """Read problem data in GLPK format

        :param fname: file name
        :type fname: `str`
        :returns: the problem read
        :rtype: `.Problem`

        """
        problem = cls()
        cdef glpk.Prob* _prob = <glpk.Prob*>PyCapsule_GetPointer(
                                                    problem._prob_ptr(), NULL)
        retcode = glpk.read_prob(_prob, 0, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading GLPK file.")
        return problem

    def write_prob(self, str fname):
        """Write problem data in GLPK format

        :param fname: file name
        :type fname: `str`

        """
        retcode = glpk.write_prob(self._prob, 0, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing GLPK file.")

    @classmethod
    def read_cnfsat(cls, str fname):
        """Read CNF-SAT problem data in DIMACS format

        This class method reads the CNF-SAT problem data from a text file in
        DIMACS format and automatically translates the data to corresponding
        0-1 programming problem instance :eq:`1.5`–:eq:`1.6`.

        :param fname: file name
        :type fname: `str`
        :returns: the problem read
        :rtype: `.Problem`

        .. note::

            If the filename ends with the suffix ‘.gz’, the file is assumed to
            be compressed, in which case the routine decompresses it “on the
            fly”.

        .. doctest:: read_cnfsat

            >>> p = Problem.read_cnfsat('examples/sample.cnf')
            >>> p.get_num_cols() # the number of variables
            4
            >>> all(p.get_col_kind(j) == 'binary' for j in range(1, 5))
            True
            >>> p.get_num_rows() # the number of clauses
            3
            >>> [p.get_mat_row(i) for i in range(1,4)]
            [{1: 1.0, 2: 1.0}, {2: -1.0, 3: 1.0, 4: -1.0}, {1: -1.0, 4: 1.0}]

        """
        problem = cls()
        cdef glpk.Prob* _prob = <glpk.Prob*>PyCapsule_GetPointer(
                                                    problem._prob_ptr(), NULL)
        retcode = glpk.read_cnfsat(_prob, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading CNF-SAT file.")
        return problem

    def check_cnfsat(self):
        """Check for CNF-SAT problem instance

        This method checks if the specified problem object P contains a 0-1
        programming problem instance in the format :eq:`1.5`–:eq:`1.6` and
        therefore encodes a CNF-SAT problem instance.

        :rtype: `bool`

        .. doctest:: check_cnfsat

            >>> p = Problem.read_cnfsat('examples/sample.cnf')
            >>> p.check_cnfsat()
            True
            >>> clause = p.get_mat_row(1)
            >>> clause[1] = 2
            >>> p.set_mat_row(1, clause)
            >>> p.check_cnfsat()
            False

        """
        return not bool(glpk.check_cnfsat(self._prob))

    def write_cnfsat(self, str fname):
        """Write CNF-SAT problem data in DIMACS format

        This method automatically translates the specified 0-1 programming
        problem instance :eq:`1.5`–:eq:`1.6` to a CNF-SAT problem instance and
        writes the problem data to a text file in DIMACS format.

        :param fname: file name
        :type fname: `str`

        .. note::

            If the filename ends with suffix ‘.gz’, the file is assumed to be
            compressed, in which case the routine performs automatic
            compression on writing that file.

        """
        retcode = glpk.write_cnfsat(self._prob, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error writing CNF-SAT file.")

    def minisat1(self):
        """Solve CNF-SAT problem with MiniSat solver

        This method is a driver to MiniSat_, a CNF-SAT solver developed by
        Niklas Eén and Niklas Sörensson, Chalmers University of Technology,
        Sweden.

        .. _MiniSat: http://minisat.se/

        :returns: solution status, either `'optimal'` (satistfiable)
            or `'no feasible'` (unsatisfiable)
        :rtype: `str`

        .. note::

            It is assumed that the specified problem is a 0-1 programming
            problem instance in the format :eq:`1.5`–:eq:`1.6` and therefore
            encodes a CNF-SAT problem instance.

        .. doctest:: minisat1

            >>> p = Problem.read_cnfsat('examples/sample.cnf')
            >>> p.minisat1()
            'optimal'
            >>> {j: bool(p.mip_col_val(j)) for j in range(1, 5)}
            {1: False, 2: True, 3: False, 4: False}

        """
        cdef int retcode = glpk.minisat1(self._prob)
        if retcode is not 0:
            raise ioretcode2error[retcode]
        return self.mip_status()

    def intfeas1(self, bint use_bound, int obj_bound):
        """Solve integer feasibility problem

        This method is a tentative implementation of an integer feasibility
        solver based on a CNF-SAT solver (currently it is MiniSat; see
        `.minisat1`).

        :param use_bound: whether to require a solution with an objective
            function value not worse than the bound given in `obj_bound`
        :type use_bound: `bool`
        :param obj_bound: specifies an upper (in case of minimization) or lower
            (in case of maximization) bound to the objective function
        :type obj_bound: `int`
        :returns: solution status, either `'feasible'` or `'no feasible'`
        :rtype: `str`

        .. note::

            The integer programming problem should satisfy to the following
            requirements:

            #. All variables (columns) should be either `'binary'`
               (cf. `Problem.get_col_kind`) or `'fixed'`
               (cf. `Problem.get_col_type`) at integer values

            #. All constraint and objective coefficients should be integer
               numbers in the range :math:`[-2^{31}, +2^{31}-1]`.

            Though there are no special requirements to the constraints,
            currently this method is efficient mainly for problems, where most
            constraints (rows) fall into the following three classes:

            #. Covering inequalities: :math:`\sum_{j}t_j\geq 1`, where
               :math:`t_j=x_j` or :math:`t_j=1-x_j`, :math:`x_j` is a
               binary variable.

            #. Packing inequalities: :math:`\sum_{j}t_j\leq 1`.

            #. Partitioning equalities (SOS1 constraints):
               :math:`\sum_{j}t_j=1`.

        .. doctest:: intfeas1

            >>> # The Queens Problem is to place as many queens as possible on
            ... # the 8x8 chess board in a way that they do not fight each
            ... # other.
            >>> p = Problem()
            >>> p.add_cols(64) # a variable for each square on the board
            >>> square2col = lambda b_col, b_row: 8 * (b_col - 1) + b_row
            >>> for col in range(1, 65):
            ...   # add variable encoding presence of a queen in a square
            ...   p.set_col_kind(col, 'binary')
            ...   # objective is to place as many queens as possible
            ...   p.set_obj_coef(col, 1)
            >>> for index in range(1, 9):
            ...   # at most one queen can be placed in each row
            ...   row = p.add_rows(1)
            ...   p.set_mat_row(row, {square2col(b_col, index): 1
            ...                       for b_col in range(1, 9)})
            ...   p.set_row_bnds(row, None, 1)
            ...   # at most one queen can be placed in each column
            ...   row = p.add_rows(1)
            ...   p.set_mat_row(row, {square2col(index, b_row): 1
            ...                       for b_row in range(1, 9)})
            ...   p.set_row_bnds(row, None, 1)
            ...   # at most one queen can be placed in each diagonal
            ...   row = p.add_rows(1)
            ...   p.set_mat_row(row, {square2col(index - b_row + 1, b_row): 1
            ...                       for b_row in range(1, index+1)})
            ...   p.set_row_bnds(row, None, 1)
            ...   # at most one queen can be placed in each cross diagonal
            ...   row = p.add_rows(1)
            ...   p.set_mat_row(row, {square2col(8 - diff, index - diff): 1
            ...                       for diff in range(0, index)})
            ...   p.set_row_bnds(row, None, 1)
            >>> best = 0
            >>> p.intfeas1(use_bound=True, obj_bound=best)

/* solve the problem */
solve;

/* and print its optimal solution */
for {i in 1..n}
{  for {j in 1..n} printf " %s", if x[i,j] then "Q" else ".";
   printf("\n");
}

        """
        cdef int retcode = glpk.intfeas1(self._prob, use_bound, obj_bound)
        if retcode is not 0:
            raise ioretcode2error[retcode]
        return self.mip_status()
