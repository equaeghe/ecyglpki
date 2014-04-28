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
        if lb <= -DBL_MAX:
            lb = -DBL_MAX
            lower = None
    elif lower is None:
        lb = -DBL_MAX
    else:
        raise TypeError("Lower bound must be 'None' or 'Real', not " +
                        type(lower).__name__ + ".")
    if isinstance(upper, numbers.Real):
        ub = upper
        if ub >= +DBL_MAX:
            ub = +DBL_MAX
            upper = None
    elif upper is None:
        ub = +DBL_MAX
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

cdef _statuscheck(unicode vartypestr, unicode status) except NULL:
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


# MPS file format
cdef str2mpsfmt = {
    'fixed': glpk.MPS_DECK,
    'free': glpk.MPS_FILE
    }


# problem coefficients
cdef _coeffscheck(coeffs) except NULL:
    if not isinstance(coeffs, collections.abc.Mapping):
        raise TypeError("Coefficients must be passed in a 'Mapping', not " +
                        type(coeffs).__name__)
    if not all([isinstance(value, numbers.Real) for value in coeffs.values()]):
        raise TypeError("Coefficient values must be 'Real'.")


cdef class Problem:
    """A GLPK problem"""

    ### Object definition, creation, setup, and cleanup ###

    cdef glpk.ProbObj* _problem

    def __cinit__(self):
        self._problem = glpk.create_prob()
        glpk.create_index(self._problem)

    def __dealloc__(self):
        glpk.delete_prob(self._problem)

    ### Translated GLPK functions ###

    def set_prob_name(self, unicode name):
        """Assign (change) problem name"""
        glpk.set_prob_name(self._problem, Name(name).to_chars())

    def set_obj_name(self, unicode name):
        """Assign (change) objective function name"""
        glpk.set_obj_name(self._problem, Name(name).to_chars())

    def set_obj_dir(self, str direction)
        """Set (change) optimization direction flag"""
        glpk.set_obj_dir(self._problem, str2optdir(direction))

    def add_rows(self, *names)
        """Add new rows to problem object

        :param names: the names (unicode strings) of the rows to add

        """
        cdef int first
        cdef int n = len(names)
        if n is not 0:
            first = glpk.add_rows(self._problem, n)
            for row, name in enumerate(names, start=first):
                glpk.set_row_name(self._problem, row, Name(name).to_chars())

    def add_cols(self, *names)
        """Add new columns to problem object

        :param names: the names (unicode strings) of the columns to add

        """
        cdef int first
        cdef int n = len(names)
        if n is not 0:
            first = glpk.add_cols(self._problem, n)
            for col, name in enumerate(names, start=first):
                glpk.set_col_name(self._problem, col, Name(name).to_chars())

    def set_row_name(self, unicode old_name, unicode new_name):
        """Change row name"""
        glpk.set_row_name(self._problem, self._find_row(old_name),
                          Name(new_name).to_chars())

    def set_col_name(self, unicode old_name, unicode new_name):
        """Change column name"""
        glpk.set_col_name(self._problem, self._find_col(old_name),
                          Name(new_name).to_chars())

    def set_row_bnds(self, unicode name, lower, upper):
        """Set (change) row bounds"""
        cdef int vartype
        cdef double lb
        cdef double ub
        vartype, lb, ub = _bounds(lower, upper)
        glpk.set_row_bnds(self._problem, self._find_row(name), vartype, lb, ub)

    def set_col_bnds(self, unicode name, lower, upper):
        """Set (change) column bounds"""
        cdef int vartype
        cdef double lb
        cdef double ub
        vartype, lb, ub = _bounds(lower, upper)
        glpk.set_col_bnds(self._problem, self._find_col(name), vartype, lb, ub)

    def set_obj_coef(self, unicode name, double coeff):
        """Set (change) obj. coefficient or constant term"""
        glpk.set_obj_coef(self._problem, self._find_col(name), coeff)

    def set_mat_row(self, unicode name, coeffs):
        """Set (replace) row of the constraint matrix

        :param coeffs: |Mapping| from column names (unicode strings) to
            coefficient values (|Real|).

        """
        _coeffscheck(coeffs)
        if not all(isinstance(name, unicode) for name in coeffs.keys()):
            raise TypeError("Coefficient keys must be (unicode) strings.")
        cdef int k = len(coeffs)
        cdef const int* cols =  <int*>glpk.alloc(1+k, sizeof(int))
        cdef const double* vals =  <double*>glpk.alloc(1+k, sizeof(double))
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                cols[i] = self._find_col(item[0])
                vals[i] = item[1]
            glpk.set_mat_row(self._problem, self._find_row(name),
                             k, cols, vals)
        finally:
            glpk.free(cols)
            glpk.free(vals)

    def clear_mat_row(self, unicode name):  # variant of set_mat_row
        """Clear row of the constraint matrix"""
        glpk.set_mat_row(self._problem, self._find_row(name), 0, NULL, NULL)

    def set_mat_col(self, unicode name, coeffs):
        """Set (replace) column of the constraint matrix

        :param coeffs: |Mapping| from row names (unicode strings) to
            coefficient values (|Real|).

        """
        _coeffscheck(coeffs)
        if not all(isinstance(name, unicode) for name in coeffs.keys()):
            raise TypeError("Coefficient keys must be (unicode) strings.")
        cdef int k = len(coeffs)
        cdef const int* rows =  <int*>glpk.alloc(1+k, sizeof(int))
        cdef const double* vals =  <double*>glpk.alloc(1+k, sizeof(double))
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                rows[i] = self._find_row(item[0])
                vals[i] = item[1]
            glpk.set_mat_col(self._problem, self._find_col(name),
                             k, rows, vals)
        finally:
            glpk.free(cols)
            glpk.free(vals)

    def clear_mat_col(self, unicode name):  # variant of set_mat_col
        """Clear column of the constraint matrix"""
        glpk.set_mat_col(self._problem, self._find_col(name), 0, NULL, NULL)

    def load_matrix(self, coeffs):
        """Load (replace) the whole constraint matrix

        :param coeffs: |Mapping| from row and column name (unicode string)
            pairs (length-2 `tuple`) to coefficient values (|Real|).

        """
        _coeffscheck(coeffs)
        if not all(isinstance(key, tuple) and (len(key) is 2)
                   for key in coeffs.keys()):
            raise TypeError("Coefficient keys must be pairs, " +
                            "i.e., length-2 tuples.")
        if not all(isinstance(name, unicode)
                   for name in itertools.chain.from_iterable(coeffs.keys())):
            raise TypeError("Coefficient key components must be " +
                            "(unicode) strings.")
        cdef int k = len(coeffs)
        cdef int* rows = <int*>glpk.alloc(1+k, sizeof(int))
        cdef int* cols = <int*>glpk.alloc(1+k, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+k, sizeof(double))
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                rows[i] = self._find_row(item[0][0])
                cols[i] = self._find_col(item[0][1])
                vals[i] = item[1]
            glpk.load_matrix(self._problem, k, rows, cols, vals)
        finally:
            glpk.free(rows)
            glpk.free(cols)
            glpk.free(vals)

    def clear_matrix(self):  # variant of load_matrix
        """Clear the whole constraint matrix"""
        glpk.load_matrix(self._problem, 0, NULL, NULL, NULL)

    def del_rows(self, *names):
        """Delete specified rows from problem object

        :param names: the names (unicode strings) of the rows to delete

        """
        cdef int n = len(names)
        cdef const int* rows =  <int*>glpk.alloc(1+n, sizeof(int))
        try:
            if n is not 0:
                for i, name in enumerate(names, start=1):
                    rows[i] = self._find_row(name)
                del_rows(self._problem, n, rows)
        finally:
            glpk.free(rows)

    def del_cols(self, *names):
        """Delete specified columns from problem object

        :param names: the names (unicode strings) of the columns to delete

        """
        cdef int n = len(names)
        cdef const int* cols =  <int*>glpk.alloc(1+n, sizeof(int))
        try:
            if n is not 0:
                for i, name in enumerate(names, start=1):
                    cols[i] = self._find_col(name)
                del_cols(self._problem, n, cols)
        finally:
            glpk.free(cols)

    @classmethod
    def copy_prob(cls, ProbObj* source):
        """Copy problem object content"""
        problem = cls()
        glpk.copy_prob(problem, source, True)
                                    # always copy names, as they function as
                                    # unique identifiers for rows and columns
        return problem

    def erase_prob(self):
        """Erase problem object content"""
        glpk.erase_prob(self._problem)

    def get_prob_name(self):
        """Retrieve problem name"""
        return Name._from_chars(glpk.get_prob_name(self._problem))

    def get_obj_name(self):
        """Retrieve objective function name"""
        return Name._from_chars(glpk.get_obj_name(self._problem))

    def get_obj_dir(self):
        """Retrieve optimization direction flag"""
        return optdir2str[glpk.get_obj_dir(self._problem)]

    def get_num_rows(self):
        """Retrieve number of rows"""
        return glpk.get_num_rows(self._problem)

    def get_num_cols(self):
        """Retrieve number of columns"""
        return glpk.get_num_cols(self._problem)

    cdef unicode _get_row_name(self, int row):
        """Retrieve row name"""
        return Name._from_chars(glpk.get_row_name(self._problem, row))

    cdef unicode _get_col_name(self, int col):
        """Retrieve column name"""
        return Name._from_chars(glpk.get_col_name(self._problem, col))

    cdef _get_row_or_col_name(self, int ind):  # variant of _get_row/col_name
        """Retrieve row or column name"""
        cdef unicode row_or_col
        cdef unicode name
        cdef int m = self.get_num_rows()
        if ind > m:
            row_or_col = 'row'
            name = self._get_row_name(ind)
        else:
            row_or_col = 'col'
            name = self._get_col_name(ind - m)
        return row_or_col, name

    def get_row_type(self, unicode name):
        """Retrieve row type"""
        return vartype2str[glpk.get_row_type(self._problem,
                                             self._find_row(name))]

    def get_row_lb(self, unicode name):
        """Retrieve row lower bound"""
        cdef double lb = glpk.get_row_lb(self._problem, self._find_row(name))
        return -float('inf') if lb == -DBL_MAX else lb

    def get_row_ub(self, unicode name):
        """Retrieve row upper bound"""
        cdef double ub = glpk.get_row_ub(self._problem, self._find_row(name))
        return -float('inf') if ub == -DBL_MAX else ub

    def get_col_type(self, unicode name):
        """Retrieve column type"""
        return vartype2str[glpk.get_col_type(self._problem,
                                             self._find_col(name))]

    def get_col_lb(self, unicode name):
        """Retrieve column lower bound"""
        cdef double lb = glpk.get_col_lb(self._problem, self._find_col(name))
        return -float('inf') if lb == -DBL_MAX else lb

    def get_col_ub(self, unicode name):
        """Retrieve column upper bound"""
        cdef double ub = glpk.get_col_ub(self._problem, self._find_col(name))
        return -float('inf') if ub == -DBL_MAX else ub

    def get_obj_coef(self, unicode name):
        """Retrieve obj. coefficient"""
        return glpk.get_obj_coef(self._problem, self._find_col(name))

    def get_obj_const(self):  # variant of get_obj_coef
        """Retrieve obj. constant term"""
        return glpk.get_obj_coef(self._problem, 0)

    def get_num_nz(self):
        """Retrieve number of constraint coefficients"""
        return glpk.get_num_nz(self._problem)

    def get_mat_row(self, unicode name):
        """Retrieve row of the constraint matrix"""
        cdef int row = self._find_row(name)
        cdef int k = glpk.get_mat_row(self._problem, row, NULL, NULL)
        cdef int* cols =  <int*>glpk.alloc(1+k, sizeof(int))
        cdef double* vals =  <double*>glpk.alloc(1+k, sizeof(double))
        try:
            glpk.get_mat_row(self._problem, row, cols, vals)
            coeffs = {self._get_col_name(cols[i]): vals[i]
                      for i in range(1, 1+k)}
        finally:
            glpk.free(cols)
            glpk.free(vals)
        return coeffs

    def get_mat_col(self, unicode name):
        """Retrieve column of the constraint matrix"""
        cdef int col = self._find_col(name)
        cdef int k = glpk.get_mat_col(self._problem, col, NULL, NULL)
        cdef int* rows =  <int*>glpk.alloc(1+k, sizeof(int))
        cdef double* vals =  <double*>glpk.alloc(1+k, sizeof(double))
        try:
            glpk.get_mat_col(self._problem, col, rows, vals)
            coeffs = {self._get_row_name(rows[i]): vals[i]
                      for i in range(1, 1+k)}
        finally:
            glpk.free(rows)
            glpk.free(vals)
        return coeffs

    cdef int _find_row(self, unicode name) except 0:
        """Find row by its name"""
        cdef int row = glpk.find_row(self._problem, Name(name)._to_chars())
        if row is 0:
            raise ValueError("'" + name + "' is not a row name.")
        else:
            return row

    cdef int _find_col(self, unicode name) except 0:
        """Find column by its name"""
        cdef int col = glpk.find_col(self._problem, Name(name)._to_chars())
        if col is 0:
            raise ValueError("'" + name + "' is not a column name.")
        else:
            return col

    def set_rii(self, unicode name, double sf):
        """Set (change) row scale factor"""
        glpk.set_rii(self._problem, self._find_row(name), sf)

    def set_sjj(self, unicode name, double sf):
        """Set (change) column scale factor"""
        glpk.set_sjj(self._problem, self._find_col(name), sf)

    def get_rii(self, unicode name):
        """Retrieve row scale factor"""
        return glpk.get_rii(self._problem, self._find_row(name))

    def get_sjj(self, unicode name):
        """Retrieve column scale factor"""
        return glpk.get_sjj(self._problem, self._find_col(name))

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

    def set_row_stat(self, unicode name, unicode status):
        """Set (change) row status"""
        _statuscheck(self.get_row_type(name), status)
        glpk.set_row_stat(self._problem, self._find_row(name),
                          str2varstat[status])

    def set_col_stat(self, unicode name, unicode status):
        """Set (change) column status"""
        _statuscheck(self.get_col_type(name), status)
        glpk.set_col_stat(self._problem, self._find_col(name),
                          str2varstat[status])

    def std_basis(self):
        """Construct standard initial LP basis"""
        glpk.std_basis(self._problem)

    def adv_basis(self):
        """Construct advanced initial LP basis"""
        glpk.adv_basis(self._problem, 0)

    def cpx_basis(self):
        """Construct Bixby's initial LP basis"""
        glpk.cpx_basis(self._problem)

        """solve LP problem with the simplex method; returns retcode"""
    int simplex(self._problem, const SimplexCP* cp)

        """solve LP problem in exact arithmetic; returns retcode"""
    int exact(self._problem, const SimplexCP* cp)

        """initialize simplex method control parameters"""
    void init_smcp "glp_init_smcp" (SimplexCP* cp)

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

    def get_row_stat(self, unicode name):
        """Retrieve row status"""
        return varstat2str[glpk.get_row_stat(self._problem,
                                             self._find_row(name))]

    def get_row_prim(self, unicode name):
        """Retrieve row primal value (basic solution)"""
        return glpk.get_row_prim(self._problem, self._find_row(name))

    def get_row_dual(self, unicode name):
        """Retrieve row dual value (basic solution)"""
        return glpk.get_row_dual(self._problem, self._find_row(name))

    def get_col_stat(self, unicode name):
        """Retrieve column status"""
        return varstat2str[glpk.get_col_stat(self._problem,
                                             self._find_col(name))]

    def get_col_prim(self, unicode name):
        """Retrieve column primal value (basic solution)"""
        return glpk.get_col_prim(self._problem, self._find_col(name))

    def get_col_dual(self, unicode name):
        """retrieve column dual value (basic solution)"""
        return glpk.get_col_dual(self._problem, self._find_col(name))

    def get_unbnd_ray(self):
        """Determine variable causing unboundedness"""
        return self._get_row_or_col_name(glpk.get_unbnd_ray(self._problem))

        """solve LP problem with the interior-point method; returns retcode"""
    int interior(self._problem, const IPointCP* cp)

        """initialize interior-point solver control parameters"""
    void init_iptcp "glp_init_iptcp" (IPointCP* cp)

    def ipt_status(self):
        """Retrieve status of interior-point solution"""
        return solstat2str[glpk.ipt_status(self._problem)]

    def ipt_obj_val(self):
        """Retrieve objective value (interior point)"""
        return glpk.ipt_obj_val(self._problem)

    def ipt_row_prim(self, unicode name):
        """Retrieve row primal value (interior point)"""
        return glpk.ipt_row_prim(self._problem, self._find_row(name))

    def ipt_row_dual(self, unicode name):
        """Retrieve row dual value (interior point)"""
        return glpk.ipt_row_dual(self._problem, self._find_row(name))

    def ipt_col_prim(self, unicode name):
        """Retrieve column primal value (interior point)"""
        return glpk.ipt_col_prim(self._problem, self._find_col(name))

    def ipt_col_dual(self, unicode name):
        """Retrieve column dual value (interior point)"""
        return glpk.ipt_col_dual(self._problem, self._find_col(name))

    def set_col_kind(self, unicode name, unicode kind):
        """Set (change) column kind"""
        return glpk.set_col_kind(self._problem, self._find_col(name),
                                 str2varkind[kind])

    def get_col_kind(self, unicode name):
        """Retrieve column kind; returns varkind"""
        return varkind2str[glpk.get_col_kind(self._problem,
                                             self._find_col(name))]

    def get_num_int(self):
        """Retrieve number of integer columns"""
        return glpk.get_num_int(self._problem)

    def get_num_bin(self):
        """Retrieve number of binary columns"""
        return glpk.get_num_bin(self._problem)

        """solve MIP problem with the branch-and-bound method; returns retcode"""
    int intopt(self._problem, const IntOptCP* cp)

        """initialize integer optimizer control parameters"""
    void init_iocp "glp_init_iocp" (IntOptCP* cp)

    def mip_status(self):
        """Retrieve status of MIP solution"""
        return solstat2str[glpk.mip_status(self._problem)]

    def mip_obj_val(self):
        """Retrieve objective value (MIP solution)"""
        return glpk.mip_obj_val(self._problem)

    def mip_row_val(self, unicode name):
        """Retrieve row value (MIP solution)"""
        return glpk.mip_row_val(self._problem, self._find_row(name))

    def mip_col_val(self, unicode name):
        """Retrieve column value (MIP solution)"""
        return glpk.mip_col_val(self._problem, self._find_col(name))

        """check feasibility/optimality conditions"""
    void check_kkt(self._problem, int sol, int cond,
                   double* ae_max, int* ae_ind, double* re_max, int* re_ind)

    def print_sol(self, unicode fname):
        """Write basic solution in printable format"""
        retcode = glpk.print_sol(self._problem, fname.encode())
        if retcode is not 0:
            raise RuntimeError("Error writing printable basic solution file")

        """read basic solution from text file"""
    int read_sol(self._problem, const char* fname)

    def write_sol(self, unicode fname):
        """Write basic solution to text file"""
        retcode = glpk.write_sol(self._problem, fname.encode())
        if retcode is not 0:
            raise RuntimeError("Error writing basic solution file")

        """print sensitivity analysis report"""
    int print_ranges(self._problem, int length, const int indlist[], 0,
                     const char* fname)

    def print_ipt(self, unicode fname):
        """Write interior-point solution in printable format"""
        retcode = glpk.print_ipt(self._problem, fname.encode())
        if retcode is not 0:
            raise RuntimeError("Error writing printable interior point " +
                               "solution file")

        """read interior-point solution from text file"""
    int read_ipt(self._problem, const char* fname)

    def write_ipt(self, unicode fname):
        """Write interior-point solution to text file"""
        retcode = glpk.write_ipt(self._problem, fname.encode())
        if retcode is not 0:
            raise RuntimeError("Error writing interior point solution file")

    def print_mip(self, unicode fname):
        """Write MIP solution in printable format"""
        retcode = glpk.print_mip(self._problem, fname.encode())
        if retcode is not 0:
            raise RuntimeError("Error writing printable integer " +
                               "optimization solution file")

        """read MIP solution from text file"""
    int read_mip(self._problem, const char* fname)

    def write_mip(self, unicode fname):
        """Write MIP solution to text file"""
        retcode = glpk.write_mip(self._problem, fname.encode())
        if retcode is not 0:
            raise RuntimeError("Error writing integer optimization solution " +
                               "file")

    def bf_exists(self):
        """Check if LP basis factorization exists"""
        return glpk.bf_exists(self._problem)

    def factorize(self._problem):
        """Compute LP basis factorization"""
        cdef int retcode = glpk.factorize(self._problem)
        if retcode is not 0:
            raise smretcode2error[retcode]

    def bf_updated(self):
        """Check if LP basis factorization has been updated"""
        return glpk.bf_updated(self._problem)

        """retrieve LP basis factorization control parameters"""
    void get_bfcp(self._problem, BasFacCP* cp)

        """change LP basis factorization control parameters"""
    void set_bfcp(self._problem, const BasFacCP* cp)

    def get_bhead(self, int k):
        """Retrieve LP basis header information"""
        return self._get_row_or_col_name(glpk.get_bhead(self._problem, int k))

    def get_row_bind(self, unicode name):
        """Retrieve row index in the basis header"""
        return glpk.get_row_bind(self._problem, self._find_row(name))

    def get_col_bind(self, unicode name):
        """Retrieve column index in the basis header"""
        return glpk.get_col_bind(self._problem, self._find_col(name))

        """perform forward transformation (solve system B*x = b)"""
    void ftran(self._problem, double rhs_pre_x_post[])

        """perform backward transformation (solve system B'*x = b)"""
    void btran(self._problem, double rhs_pre_x_post[])

    def warm_up(self._problem):
        """“Warm up” LP basis"""
        cdef int retcode = glpk.warm_up(self._problem)
        if retcode is not 0:
            raise smretcode2error[retcode]

        """compute row of the simplex tableau"""
    int eval_tab_row(self._problem, int k, int ind[], double val[])

        """compute column of the simplex tableau"""
    int eval_tab_col(self._problem, int k, int ind[], double val[])

        """transform explicitly specified row"""
    int transform_row(self._problem, int length, int ind[], double val[])

        """transform explicitly specified column"""
    int transform_col(self._problem, int length, int ind[], double val[])

        """perform primal ratio test"""
    int prim_rtest(self._problem, int length,
                   const int ind[], const double val[],
                   int direction, double eps)

        """perform dual ratio test"""
    int dual_rtest(self._problem, int length,
                   const int ind[], const double val[],
                   int direction, double eps)

        """analyze active bound of non-basic variable"""
    void analyze_bound(self._problem, int k, double* min_bnd, int* min_bnd_k,
                                             double* max_bnd, int* max_bnd_k)

        """analyze objective coefficient at basic variable"""
    void analyze_coef(self._problem, int k, double* min_coef, int* min_coef_k,
                                            double* val_min_coef,
                                            double* max_coef, int* max_coef_k,
                                            double* val_max_coef)

        """read problem data in MPS format"""
    int read_mps(self._problem, int mpsfmt, NULL, const char* fname)

    def write_mps(self, unicode mpsfmtstr, unicode fname):
        """Write problem data in MPS format"""
        retcode = glpk.write_mps(self._problem,
                                 str2mpsfmt[mpsfmtstr], NULL, fname.encode())
        if retcode is not 0:
            raise RuntimeError("Error writing MPS file.")

        """read problem data in CPLEX LP format"""
    int read_lp(self._problem, NULL, const char* fname)

    def write_lp(self, unicode fname):
        """Write problem data in CPLEX LP format"""
        retcode = glpk.write_lp(self._problem, NULL, fname.encode())
        if retcode is not 0:
            raise RuntimeError("Error writing LP file.")

        """read problem data in GLPK format"""
    int read_prob(self._problem, 0, const char* fname)

    def write_prob(self, unicode fname):
        """Write problem data in GLPK format"""
        retcode = glpk.write_prob(self._problem, 0, fname.encode())
        if retcode is not 0:
            raise RuntimeError("Error writing GLPK file.")

        """read CNF-SAT problem data in DIMACS format"""
    int read_cnfsat(self._problem, const char* fname)

    def check_cnfsat(self):
        """Check for CNF-SAT problem instance"""
        return not bool(glpk.check_cnfsat(self._problem))

    def write_cnfsat(self, unicode fname):
        """Write CNF-SAT problem data in DIMACS format"""
        retcode = glpk.write_cnfsat(self._problem, fname.encode())
        if retcode is not 0:
            raise RuntimeError("Error writing CNF-SAT file.")

    def minisat1(self):
        """Solve CNF-SAT problem with MiniSat solver"""
        cdef int retcode = glpk.minisat1(self._problem)
        if retcode is not 0:
            raise ioretcode2error[retcode]

    def intfeas1(self, bool use_bound, int obj_bound):
        """Solve integer feasibility problem"""
        cdef int retcode = glpk.intfeas1(self._problem, use_bound, obj_bound)
        if retcode is not 0:
            raise ioretcode2error[retcode]
