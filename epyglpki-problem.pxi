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

# optimization directions
cdef str2optdir = {
    'minimize': glpk.MIN,
    'maximize': glpk.MAX
    }
cdef optdir2str = {optdir: string for string, optdir in str2optdir.items()}


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

    def get_row_type(self, unicode name):
        """Retrieve row type"""
        return vartype2str[glpk.get_row_type(self._problem,
                                             self._find_row(name))]

        """retrieve row lower bound"""
    double get_row_lb(self._problem, self._find_row(name))

        """retrieve row upper bound"""
    double get_row_ub(self._problem, self._find_row(name))

    def get_col_type(self, unicode name):
        """Retrieve column type"""
        return vartype2str[glpk.get_col_type(self._problem,
                                             self._find_col(name))]

        """retrieve column lower bound"""
    double get_col_lb(self._problem, self._find_col(name))

        """retrieve column upper bound"""
    double get_col_ub(self._problem, self._find_col(name))

    def get_obj_coef(self, unicode name):
        """Retrieve obj. coefficient"""
        return glpk.get_obj_coef(self._problem, self._find_col(name))

    def get_obj_const(self):  # variant of get_obj_coef
        """Retrieve obj. constant term"""
        return glpk.get_obj_coef(self._problem, 0)

    def get_num_nz(self):
        """Retrieve number of constraint coefficients"""
        return glpk.get_num_nz(self._problem)

        """retrieve row of the constraint matrix"""
    int get_mat_row(self._problem, self._find_row(name), int ind[], double val[])

        """retrieve column of the constraint matrix"""
    int get_mat_col(self._problem, self._find_col(name), int ind[], double val[])

    cdef int _find_row(self, unicode name):
        """Find row by its name"""
        return glpk.find_row(self._problem, Name(name)._to_chars())

    cdef int _find_col(self, unicode name):
        """Find column by its name"""
        return glpk.find_col(self._problem, Name(name)._to_chars())

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

        """scale problem data"""
    void scale_prob(self._problem, int scalopt)

    def unscale_prob(self):
        """Unscale problem data"""
        glpk.unscale_prob(self._problem)

        """set (change) row status"""
    void set_row_stat(self._problem, self._find_row(name), int varstat)

        """set (change) column status"""
    void set_col_stat(self._problem, self._find_col(name), int varstat)

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

        """retrieve generic status of basic solution; returns solstat"""
    int get_status(self._problem)

        """retrieve status of primal basic solution; returns solstat"""
    int get_prim_stat(self._problem)

        """retrieve status of dual basic solution; returns solstat"""
    int get_dual_stat(self._problem)

        """retrieve objective value (basic solution)"""
    double get_obj_val(self._problem)

        """retrieve row status; returns varstat"""
    int get_row_stat(self._problem, self._find_row(name))

        """retrieve row primal value (basic solution)"""
    double get_row_prim(self._problem, self._find_row(name))

        """retrieve row dual value (basic solution)"""
    double get_row_dual(self._problem, self._find_row(name))

        """retrieve column status; returns varstat"""
    int get_col_stat(self._problem, self._find_col(name))

        """retrieve column primal value (basic solution)"""
    double get_col_prim(self._problem, self._find_col(name))

        """retrieve column dual value (basic solution)"""
    double get_col_dual(self._problem, self._find_col(name))

        """determine variable causing unboundedness"""
    int get_unbnd_ray(self._problem)

        """solve LP problem with the interior-point method; returns retcode"""
    int interior(self._problem, const IPointCP* cp)

        """initialize interior-point solver control parameters"""
    void init_iptcp "glp_init_iptcp" (IPointCP* cp)

        """retrieve status of interior-point solution; returns solstat"""
    int ipt_status(self._problem)

        """retrieve objective value (interior point)"""
    double ipt_obj_val(self._problem)

        """retrieve row primal value (interior point)"""
    double ipt_row_prim(self._problem, self._find_row(name))

        """retrieve row dual value (interior point)"""
    double ipt_row_dual(self._problem, self._find_row(name))

        """retrieve column primal value (interior point)"""
    double ipt_col_prim(self._problem, self._find_col(name))

        """retrieve column dual value (interior point)"""
    double ipt_col_dual(self._problem, self._find_col(name))

        """set (change) column kind"""
    void set_col_kind(self._problem, self._find_col(name), int varkind)

        """retrieve column kind; returns varkind"""
    int get_col_kind(self._problem, self._find_col(name))

        """retrieve number of integer columns"""
    int get_num_int(self._problem)

        """retrieve number of binary columns"""
    int get_num_bin(self._problem)

        """solve MIP problem with the branch-and-bound method; returns retcode"""
    int intopt(self._problem, const IntOptCP* cp)

        """initialize integer optimizer control parameters"""
    void init_iocp "glp_init_iocp" (IntOptCP* cp)

        """retrieve status of MIP solution; returns solstat"""
    int mip_status(self._problem)

        """retrieve objective value (MIP solution)"""
    double mip_obj_val(self._problem)

        """retrieve row value (MIP solution)"""
    double mip_row_val(self._problem, self._find_row(name))

        """retrieve column value (MIP solution)"""
    double mip_col_val(self._problem, self._find_col(name))

        """check feasibility/optimality conditions"""
    void check_kkt(self._problem, int sol, int cond,
                   double* ae_max, int* ae_ind, double* re_max, int* re_ind)

        """write basic solution in printable format"""
    int print_sol(self._problem, const char* fname)

        """read basic solution from text file"""
    int read_sol(self._problem, const char* fname)

        """write basic solution to text file"""
    int write_sol(self._problem, const char* fname)

        """print sensitivity analysis report"""
    int print_ranges(self._problem, int length, const int indlist[], 0,
                     const char* fname)

        """write interior-point solution in printable format"""
    int print_ipt(self._problem, const char* fname)

        """read interior-point solution from text file"""
    int read_ipt(self._problem, const char* fname)

        """write interior-point solution to text file"""
    int write_ipt(self._problem, const char* fname)

        """write MIP solution in printable format"""
    int print_mip(self._problem, const char* fname)

        """read MIP solution from text file"""
    int read_mip(self._problem, const char* fname)

        """write MIP solution to text file"""
    int write_mip(self._problem, const char* fname)

        """check if LP basis factorization exists"""
    bint bf_exists(self._problem)

        """compute LP basis factorization; returns retcode"""
    int factorize(self._problem)

        """check if LP basis factorization has been updated"""
    bint bf_updated(self._problem)

        """retrieve LP basis factorization control parameters"""
    void get_bfcp(self._problem, BasFacCP* cp)

        """change LP basis factorization control parameters"""
    void set_bfcp(self._problem, const BasFacCP* cp)

        """retrieve LP basis header information"""
    int get_bhead(self._problem, int k)

        """retrieve row index in the basis header"""
    int get_row_bind(self._problem, self._find_row(name))

        """retrieve column index in the basis header"""
    int get_col_bind(self._problem, self._find_col(name))

        """perform forward transformation (solve system B*x = b)"""
    void ftran(self._problem, double rhs_pre_x_post[])

        """perform backward transformation (solve system B'*x = b)"""
    void btran(self._problem, double rhs_pre_x_post[])

        """"warm up" LP basis; returns retcode"""
    int warm_up(self._problem)

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

        """write problem data in MPS format"""
    int write_mps(self._problem, int mpsfmt, NULL, const char* fname)

        """read problem data in CPLEX LP format"""
    int read_lp(self._problem, NULL, const char* fname)

        """write problem data in CPLEX LP format"""
    int write_lp(self._problem, NULL, const char* fname)

        """read problem data in GLPK format"""
    int read_prob(self._problem, 0, const char* fname)

        """write problem data in GLPK format"""
    int write_prob(self._problem, 0, const char* fname)

        """read CNF-SAT problem data in DIMACS format"""
    int read_cnfsat(self._problem, const char* fname)

        """check for CNF-SAT problem instance"""
    int check_cnfsat(self._problem)

        """write CNF-SAT problem data in DIMACS format"""
    int write_cnfsat(self._problem, const char* fname)

        """solve CNF-SAT problem with MiniSat solver; returns retcode"""
    int minisat1(self._problem)

        """solve integer feasibility problem; returns retcode"""
    int intfeas1(self._problem, bint use_bound, int obj_bound)
