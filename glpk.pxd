# glpk.pxd: Cython bindings for GLPK

###############################################################################
#
#  This code is part of ecyglpki (a Cython GLPK interface).
#
#  Copyright (C) 2014 Erik Quaeghebeur. All rights reserved.
#
#  ecyglpki is free software: you can redistribute it and/or modify it under
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


ctypedef void* CBInfo
ctypedef void (*CBFunc)(Tree*, CBInfo)


cdef extern from "glpk.h":
    #  library version numbers:
    enum: MAJOR_VERSION "GLP_MAJOR_VERSION"
    enum: MINOR_VERSION "GLP_MINOR_VERSION"

    #  LP/MIP problem object
    cdef struct Prob "glp_prob":
        pass

    #  optimization direction flag (argument name is 'optdir'):
    enum: MIN "GLP_MIN"  #  minimization
    enum: MAX "GLP_MAX"  #  maximization

    #  kind of structural variable (argument name is 'varkind'):
    enum: CV "GLP_CV"  #  continuous variable
    enum: IV "GLP_IV"  #  integer variable
    enum: BV "GLP_BV"  #  binary variable

    #  type of auxiliary/structural variable (argument name is 'vartype'):
    enum: FR "GLP_FR"  #  free (unbounded) variable
    enum: LO "GLP_LO"  #  variable with lower bound
    enum: UP "GLP_UP"  #  variable with upper bound
    enum: DB "GLP_DB"  #  double-bounded variable
    enum: FX "GLP_FX"  #  fixed variable

    #  status of auxiliary/structural variable (argument name is 'varstat'):
    enum: BS "GLP_BS"  #  basic variable
    enum: NL "GLP_NL"  #  non-basic variable on lower bound
    enum: NU "GLP_NU"  #  non-basic variable on upper bound
    enum: NF "GLP_NF"  #  non-basic free (unbounded) variable
    enum: NS "GLP_NS"  #  non-basic fixed variable

    #  scaling options (argument name is 'scalopt'):
    enum: SF_GM "GLP_SF_GM"      #  perform geometric mean scaling
    enum: SF_EQ "GLP_SF_EQ"      #  perform equilibration scaling
    enum: SF_2N "GLP_SF_2N"      #  round scale factors to power of two
    enum: SF_SKIP "GLP_SF_SKIP"  #  skip if problem is well scaled
    enum: SF_AUTO "GLP_SF_AUTO"  #  choose scaling options automatically

    #  solution indicator (argument name is 'solind'):
    enum: SOL "GLP_SOL"  #  basic solution
    enum: IPT "GLP_IPT"  #  interior-point solution
    enum: MIP "GLP_MIP"  #  mixed integer solution

    #  solution status (argument name is 'solstat'):
    enum: UNDEF "GLP_UNDEF"    #  solution is undefined
    enum: FEAS "GLP_FEAS"      #  solution is feasible
    enum: INFEAS "GLP_INFEAS"  #  solution is infeasible
    enum: NOFEAS "GLP_NOFEAS"  #  no feasible solution exists
    enum: OPT "GLP_OPT"        #  solution is optimal
    enum: UNBND "GLP_UNBND"    #  solution is unbounded

    #  factorization type (argument name is 'type'):
    enum: BF_LUF "GLP_BF_LUF"  #  plain LU-factorization
    enum: BF_BTF "GLP_BF_BTF"  #  block triangular LU factorization
    enum: BF_FT "GLP_BF_FT"    #  Forrest-Tomlin (LUF only)
    enum: BF_BG "GLP_BF_BG"    #  Schur compl. + Bartels-Golub
    enum: BF_GR "GLP_BF_GR"    #  Schur compl. + Givens rotation

    #  basis factorization control parameters
    ctypedef struct BfCp "glp_bfcp":
        int type        #  factorization type
        double piv_tol  #  sgf_piv_tol
        int piv_lim     #  sgf_piv_lim
        bint suhl       #  sgf_suhl
        double eps_tol  #  sgf_eps_tol
        int nfs_max     #  fhvint.nfs_max
        int nrs_max     #  scfint.nn_max

    #  message level (argument name is 'msg_lev'):
    enum: MSG_OFF "GLP_MSG_OFF"  #  no output
    enum: MSG_ERR "GLP_MSG_ERR"  #  warning and error messages only
    enum: MSG_ON "GLP_MSG_ON"    #  normal output
    enum: MSG_ALL "GLP_MSG_ALL"  #  full output
    enum: MSG_DBG "GLP_MSG_DBG"  #  debug output

    #  simplex method option (argument name is 'meth'):
    enum: PRIMAL "GLP_PRIMAL"  #  use primal simplex
    enum: DUALP "GLP_DUALP"    #  use dual if it fails, use primal
    enum: DUAL "GLP_DUAL"      #  use dual simplex

    #  pricing technique (argument name is 'pricing'):
    enum: PT_STD "GLP_PT_STD"  #  standard (Dantzig rule)
    enum: PT_PSE "GLP_PT_PSE"  #  projected steepest edge

    #  ratio test technique (argument name is 'r_test'):
    enum: RT_STD "GLP_RT_STD"  #  standard (textbook)
    enum: RT_HAR "GLP_RT_HAR"  #  two-pass Harris' ratio test

    #  simplex method control parameters
    ctypedef struct SmCp "glp_smcp":
        int msg_lev     #  message level
        int meth        #  simplex method option
        int pricing     #  pricing technique
        int r_test      #  ratio test technique
        double tol_bnd  #  spx.tol_bnd
        double tol_dj   #  spx.tol_dj
        double tol_piv  #  spx.tol_piv
        double obj_ll   #  spx.obj_ll
        double obj_ul   #  spx.obj_ul
        int it_lim      #  spx.it_lim
        int tm_lim      #  spx.tm_lim (milliseconds)
        int out_frq     #  spx.out_frq
        int out_dly     #  spx.out_dly (milliseconds)
        bint presolve   #  enable/disable using LP presolver

    #  ordering algorithm (argument name is 'ord_alg'):
    enum: ORD_NONE "GLP_ORD_NONE"      #  natural (original) ordering
    enum: ORD_QMD "GLP_ORD_QMD"        #  quotient minimum degree (QMD)
    enum: ORD_AMD "GLP_ORD_AMD"        #  approx. minimum degree (AMD)
    enum: ORD_SYMAMD "GLP_ORD_SYMAMD"  #  approx. minimum degree (SYMAMD)

    #  interior-point solver control parameters
    ctypedef struct IPtCp "glp_iptcp":
        int msg_lev  #  message level
        int ord_alg  #  ordering algorithm

    #  branch-and-bound tree
    cdef struct Tree "glp_tree":
        pass

    #  branching technique (argument name is 'br_tech'):
    enum: BR_FFV "GLP_BR_FFV"  #  first fractional variable
    enum: BR_LFV "GLP_BR_LFV"  #  last fractional variable
    enum: BR_MFV "GLP_BR_MFV"  #  most fractional variable
    enum: BR_DTH "GLP_BR_DTH"  #  heuristic by Driebeck and Tomlin
    enum: BR_PCH "GLP_BR_PCH"  #  hybrid pseudocost heuristic

    #  backtracking technique (argument name is 'bt_tech'):
    enum: BT_DFS "GLP_BT_DFS"  #  depth first search
    enum: BT_BFS "GLP_BT_BFS"  #  breadth first search
    enum: BT_BLB "GLP_BT_BLB"  #  best local bound
    enum: BT_BPH "GLP_BT_BPH"  #  best projection heuristic

    #  preprocessing technique (argument name is 'pp_tech'):
    enum: PP_NONE "GLP_PP_NONE"  #  disable preprocessing
    enum: PP_ROOT "GLP_PP_ROOT"  #  preprocessing only on root level
    enum: PP_ALL "GLP_PP_ALL"    #  preprocessing on all levels

    #  integer optimizer control parameters
    ctypedef struct IoCp "glp_iocp":
        int msg_lev     #  message level
        int br_tech     #  branching technique
        int bt_tech     #  backtracking technique
        double tol_int  #  mip.tol_int
        double tol_obj  #  mip.tol_obj
        int tm_lim      #  mip.tm_lim (milliseconds)
        int out_frq     #  mip.out_frq (milliseconds)
        int out_dly     #  mip.out_dly (milliseconds)
        CBFunc cb_func  #  mip.cb_func
        CBInfo cb_info  #  mip.cb_info
        int cb_size     #  mip.cb_size
        int pp_tech     #  preprocessing technique
        double mip_gap  #  relative MIP gap tolerance
        bint mir_cuts   #  MIR cuts
        bint gmi_cuts   #  Gomory's cuts
        bint cov_cuts   #  cover cuts
        bint clq_cuts   #  clique cuts
        bint presolve   #  enable/disable using MIP presolver
        bint binarize   #  try to binarize integer variables
        bint fp_heur    #  feasibility pump heuristic
        bint ps_heur    #  proximity search heuristic
        int ps_tm_lim   #  proxy time limit (milliseconds)
        #bint use_sol    #  use existing solution (experimental/undocumented)
        #const char* save_sol
                        #  filename to save every new solution
                        # (experimental/undocumented)

    #  row origin flag (argument name is 'origin'):
    enum: RF_REG "GLP_RF_REG"    #  regular constraint
    enum: RF_LAZY "GLP_RF_LAZY"  #  "lazy" constraint
    enum: RF_CUT "GLP_RF_CUT"    #  cutting plane constraint

    #  row class descriptor (argument name is 'glp_klass'):
    enum: RF_GMI "GLP_RF_GMI"  #  Gomory's mixed integer cut
    enum: RF_MIR "GLP_RF_MIR"  #  mixed integer rounding cut
    enum: RF_COV "GLP_RF_COV"  #  mixed cover cut
    enum: RF_CLQ "GLP_RF_CLQ"  #  clique cut

    #  additional row attributes
    ctypedef struct Attr "glp_attr":
        int level   #  subproblem level at which the row was added
        int origin  #  row origin flag
        int klass   #  row class descriptor

# We use the Cython bint type instead of te following:
#
#    #  enable/disable flag:
#    enum: ON "GLP_ON"   #  enable something
#    enum: OFF "GLP_OFF" #  disable something
#
# So GLP_ON corresponds to True and GLP_OFF to False.

    #  reason codes (argument name is 'reascode'):
    enum: IROWGEN "GLP_IROWGEN"  #  request for row generation
    enum: IBINGO "GLP_IBINGO"    #  better integer solution found
    enum: IHEUR "GLP_IHEUR"      #  request for heuristic solution
    enum: ICUTGEN "GLP_ICUTGEN"  #  request for cut generation
    enum: IBRANCH "GLP_IBRANCH"  #  request for branching
    enum: ISELECT "GLP_ISELECT"  #  request for subproblem selection
    enum: IPREPRO "GLP_IPREPRO"  #  request for preprocessing

    #  branch selection indicator (argument name is 'branch'):
    enum: NO_BRNCH "GLP_NO_BRNCH"  #  select no branch
    enum: DN_BRNCH "GLP_DN_BRNCH"  #  select down-branch
    enum: UP_BRNCH "GLP_UP_BRNCH"  #  select up-branch

    #  return codes (argument name is 'retcode'):
    enum: EBADB "GLP_EBADB"      #  invalid basis
    enum: ESING "GLP_ESING"      #  singular matrix
    enum: ECOND "GLP_ECOND"      #  ill-conditioned matrix
    enum: EBOUND "GLP_EBOUND"    #  invalid bounds
    enum: EFAIL "GLP_EFAIL"      #  solver failed
    enum: EOBJLL "GLP_EOBJLL"    #  objective lower limit reached
    enum: EOBJUL "GLP_EOBJUL"    #  objective upper limit reached
    enum: EITLIM "GLP_EITLIM"    #  iteration limit exceeded
    enum: ETMLIM "GLP_ETMLIM"    #  time limit exceeded
    enum: ENOPFS "GLP_ENOPFS"    #  no primal feasible solution
    enum: ENODFS "GLP_ENODFS"    #  no dual feasible solution
    enum: EROOT "GLP_EROOT"      #  root LP optimum not provided
    enum: ESTOP "GLP_ESTOP"      #  search terminated by application
    enum: EMIPGAP "GLP_EMIPGAP"  #  relative mip gap tolerance reached
    enum: ENOFEAS "GLP_ENOFEAS"  #  no primal/dual feasible solution
    enum: ENOCVG "GLP_ENOCVG"    #  no convergence
    enum: EINSTAB "GLP_EINSTAB"  #  numerical instability
    enum: EDATA "GLP_EDATA"      #  invalid data
    enum: ERANGE "GLP_ERANGE"    #  result out of range

    #  condition indicator:
    enum: KKT_PE "GLP_KKT_PE"  #  primal equalities
    enum: KKT_PB "GLP_KKT_PB"  #  primal bounds
    enum: KKT_DE "GLP_KKT_DE"  #  dual equalities
    enum: KKT_DB "GLP_KKT_DB"  #  dual bounds
    enum: KKT_CS "GLP_KKT_CS"  #  complementary slackness

    #  MPS file format (argument name is 'mpsfmt'):
    enum: MPS_DECK "GLP_MPS_DECK"  #  fixed (ancient)
    enum: MPS_FILE "GLP_MPS_FILE"  #  free (modern)

    #  MPS format control parameters
    ctypedef struct MPSCp "glp_mpscp":
      int blank       #  character code to replace blanks in symbolic names
      char* obj_name  #  objective row name
      double tol_mps  #  zero tolerance for MPS data

    #  CPLEX LP format control parameters
    ctypedef struct CPXCp "glp_cpxcp":
        pass

    #  MathProg translator workspace
    cdef struct Tran "glp_tran":
        pass

    #  create problem object
    Prob* create_prob "glp_create_prob" ()

    #  assign (change) problem name
    void set_prob_name "glp_set_prob_name" (Prob* prob, const char* name)

    #  assign (change) objective function name
    void set_obj_name "glp_set_obj_name" (Prob* prob, const char* name)

    #  set (change) optimization direction flag
    void set_obj_dir "glp_set_obj_dir" (Prob* prob, int optdir)

    #  add new rows to problem object
    int add_rows "glp_add_rows" (Prob* prob, int rows)

    #  add new columns to problem object
    int add_cols "glp_add_cols" (Prob* prob, int cols)

    #  assign (change) row name
    void set_row_name "glp_set_row_name" (Prob* prob,
                                          int row, const char* name)

    #  assign (change) column name
    void set_col_name "glp_set_col_name" (Prob* prob,
                                          int col, const char* name)

    #  set (change) row bounds
    void set_row_bnds "glp_set_row_bnds" (Prob* prob, int row, int vartype,
                                          double lb, double ub)

    #  set (change) column bounds
    void set_col_bnds "glp_set_col_bnds" (Prob* prob, int col, int vartype,
                                          double lb, double ub)

    #  set (change) obj. coefficient or constant term
    void set_obj_coef "glp_set_obj_coef" (Prob* prob, int col, double coef)

    #  set (replace) row of the constraint matrix
    void set_mat_row "glp_set_mat_row" (Prob* prob, int row, int length,
                                        const int ind[], const double val[])

    #  set (replace) column of the constraint matrix
    void set_mat_col "glp_set_mat_col" (Prob* prob, int col, int length,
                                        const int ind[], const double val[])

    #  load (replace) the whole constraint matrix
    void load_matrix "glp_load_matrix" (Prob* prob, int elements,
                                        const int row_ind[],
                                        const int col_ind[],
                                        const double val[])

    #  check for duplicate elements in sparse matrix
    int check_dup "glp_check_dup" (int rows, int cols, int elements,
                                   const int row_ind[], const int col_ind[])

    #  sort elements of the constraint matrix
    void sort_matrix "glp_sort_matrix" (Prob* prob)

    #  delete specified rows from problem object
    void del_rows "glp_del_rows" (Prob* prob, int rows, const int row_ind[])

    #  delete specified columns from problem object
    void del_cols "glp_del_cols" (Prob* prob, int cols, const int col_ind[])

    #  copy problem object content
    void copy_prob "glp_copy_prob" (Prob* prob, Prob* source, bint copy_names)

    #  erase problem object content
    void erase_prob "glp_erase_prob" (Prob* prob)

    #  delete problem object
    void delete_prob "glp_delete_prob" (Prob* prob)

    #  retrieve problem name
    const char* get_prob_name "glp_get_prob_name" (Prob* prob)

    #  retrieve objective function name
    const char* get_obj_name "glp_get_obj_name" (Prob* prob)

    #  retrieve optimization direction flag; returns optdir
    int get_obj_dir "glp_get_obj_dir" (Prob* prob)

    #  retrieve number of rows
    int get_num_rows "glp_get_num_rows" (Prob* prob)

    #  retrieve number of columns
    int get_num_cols "glp_get_num_cols" (Prob* prob)

    #  retrieve row name
    const char* get_row_name "glp_get_row_name" (Prob* prob, int row)

    #  retrieve column name
    const char* get_col_name "glp_get_col_name" (Prob* prob, int col)

    #  retrieve row type; returns vartype
    int get_row_type "glp_get_row_type" (Prob* prob, int row)

    #  retrieve row lower bound
    double get_row_lb "glp_get_row_lb" (Prob* prob, int row)

    #  retrieve row upper bound
    double get_row_ub "glp_get_row_ub" (Prob* prob, int row)

    #  retrieve column type; returns vartype
    int get_col_type "glp_get_col_type" (Prob* prob, int col)

    #  retrieve column lower bound
    double get_col_lb "glp_get_col_lb" (Prob* prob, int col)

    #  retrieve column upper bound
    double get_col_ub "glp_get_col_ub" (Prob* prob, int col)

    #  retrieve obj. coefficient or constant term
    double get_obj_coef "glp_get_obj_coef" (Prob* prob, int col)

    #  retrieve number of constraint coefficients
    int get_num_nz "glp_get_num_nz" (Prob* prob)

    #  retrieve row of the constraint matrix
    int get_mat_row "glp_get_mat_row" (Prob* prob, int row,
                                       int ind[], double val[])

    #  retrieve column of the constraint matrix
    int get_mat_col "glp_get_mat_col" (Prob* prob, int col,
                                       int ind[], double val[])

    #  create the name index
    void create_index "glp_create_index" (Prob* prob)

    #  find row by its name
    int find_row "glp_find_row" (Prob* prob, const char* name)

    #  find column by its name
    int find_col "glp_find_col" (Prob* prob, const char* name)

    #  delete the name index
    void delete_index "glp_delete_index" (Prob* prob)

    #  set (change) row scale factor
    void set_rii "glp_set_rii" (Prob* prob, int row, double sf)

    #  set (change) column scale factor
    void set_sjj "glp_set_sjj" (Prob* prob, int col, double sf)

    #  retrieve row scale factor
    double get_rii "glp_get_rii" (Prob* prob, int row)

    #  retrieve column scale factor
    double get_sjj "glp_get_sjj" (Prob* prob, int col)

    #  scale problem data
    void scale_prob "glp_scale_prob" (Prob* prob, int scalopt)

    #  unscale problem data
    void unscale_prob "glp_unscale_prob" (Prob* prob)

    #  set (change) row status
    void set_row_stat "glp_set_row_stat" (Prob* prob, int row, int varstat)

    #  set (change) column status
    void set_col_stat "glp_set_col_stat" (Prob* prob, int col, int varstat)

    #  construct standard initial LP basis
    void std_basis "glp_std_basis" (Prob* prob)

    #  construct advanced initial LP basis
    void adv_basis "glp_adv_basis" (Prob* prob,
                                    int flags)  #  flags must be 0!

    #  construct Bixby's initial LP basis
    void cpx_basis "glp_cpx_basis" (Prob* prob)

    #  solve LP problem with the simplex method; returns retcode
    int simplex "glp_simplex" (Prob* prob, const SmCp* cp)

    #  solve LP problem in exact arithmetic; returns retcode
    int exact "glp_exact" (Prob* prob, const SmCp* cp)

    #  initialize simplex method control parameters
    void init_smcp "glp_init_smcp" (SmCp* cp)

    #  retrieve generic status of basic solution; returns solstat
    int get_status "glp_get_status" (Prob* prob)

    #  retrieve status of primal basic solution; returns solstat
    int get_prim_stat "glp_get_prim_stat" (Prob* prob)

    #  retrieve status of dual basic solution; returns solstat
    int get_dual_stat "glp_get_dual_stat" (Prob* prob)

    #  retrieve objective value (basic solution)
    double get_obj_val "glp_get_obj_val" (Prob* prob)

    #  retrieve row status; returns varstat
    int get_row_stat "glp_get_row_stat" (Prob* prob, int row)

    #  retrieve row primal value (basic solution)
    double get_row_prim "glp_get_row_prim" (Prob* prob, int row)

    #  retrieve row dual value (basic solution)
    double get_row_dual "glp_get_row_dual" (Prob* prob, int row)

    #  retrieve column status; returns varstat
    int get_col_stat "glp_get_col_stat" (Prob* prob, int col)

    #  retrieve column primal value (basic solution)
    double get_col_prim "glp_get_col_prim" (Prob* prob, int col)

    #  retrieve column dual value (basic solution)
    double get_col_dual "glp_get_col_dual" (Prob* prob, int col)

    #  determine variable causing unboundedness
    int get_unbnd_ray "glp_get_unbnd_ray" (Prob* prob)

# Undocumented
#
#    # get simplex solver iteration count
#    int get_it_cnt "glp_get_it_cnt" (Prob* prob)
#
#    # set simplex solver iteration count
#    void set_it_cnt "glp_set_it_cnt" (Prob* prob, int it_cnt)

    #  solve LP problem with the interior-point method; returns retcode
    int interior "glp_interior" (Prob* prob, const IPtCp* cp)

    #  initialize interior-point solver control parameters
    void init_iptcp "glp_init_iptcp" (IPtCp* cp)

    #  retrieve status of interior-point solution; returns solstat
    int ipt_status "glp_ipt_status" (Prob* prob)

    #  retrieve objective value (interior point)
    double ipt_obj_val "glp_ipt_obj_val" (Prob* prob)

    #  retrieve row primal value (interior point)
    double ipt_row_prim "glp_ipt_row_prim" (Prob* prob, int row)

    #  retrieve row dual value (interior point)
    double ipt_row_dual "glp_ipt_row_dual" (Prob* prob, int row)

    #  retrieve column primal value (interior point)
    double ipt_col_prim "glp_ipt_col_prim" (Prob* prob, int col)

    #  retrieve column dual value (interior point)
    double ipt_col_dual "glp_ipt_col_dual" (Prob* prob, int col)

    #  set (change) column kind
    void set_col_kind "glp_set_col_kind" (Prob* prob, int col, int varkind)

    #  retrieve column kind; returns varkind
    int get_col_kind "glp_get_col_kind" (Prob* prob, int col)

    #  retrieve number of integer columns
    int get_num_int "glp_get_num_int" (Prob* prob)

    #  retrieve number of binary columns
    int get_num_bin "glp_get_num_bin" (Prob* prob)

    #  solve MIP problem with the branch-and-bound method; returns retcode
    int intopt "glp_intopt" (Prob* prob, const IoCp* cp)

    #  initialize integer optimizer control parameters
    void init_iocp "glp_init_iocp" (IoCp* cp)

    #  retrieve status of MIP solution; returns solstat
    int mip_status "glp_mip_status" (Prob* prob)

    #  retrieve objective value (MIP solution)
    double mip_obj_val "glp_mip_obj_val" (Prob* prob)

    #  retrieve row value (MIP solution)
    double mip_row_val "glp_mip_row_val" (Prob* prob, int row)

    #  retrieve column value (MIP solution)
    double mip_col_val "glp_mip_col_val" (Prob* prob, int col)

    #  check feasibility/optimality conditions
    void check_kkt "glp_check_kkt" (Prob* prob, int sol, int cond,
                                    double* ae_max, int* ae_ind,
                                    double* re_max, int* re_ind)

    #  write basic solution in printable format
    int print_sol "glp_print_sol" (Prob* prob, const char* fname)

    #  read basic solution from text file
    int read_sol "glp_read_sol" (Prob* prob, const char* fname)

    #  write basic solution to text file
    int write_sol "glp_write_sol" (Prob* prob, const char* fname)

    #  print sensitivity analysis report
    int print_ranges "glp_print_ranges" (Prob* prob,
                                         int length, const int indlist[],
                                         int flags,  #  flags must be 0!
                                         const char* fname)

    #  write interior-point solution in printable format
    int print_ipt "glp_print_ipt" (Prob* prob, const char* fname)

    #  read interior-point solution from text file
    int read_ipt "glp_read_ipt" (Prob* prob, const char* fname)

    #  write interior-point solution to text file
    int write_ipt "glp_write_ipt" (Prob* prob, const char* fname)

    #  write MIP solution in printable format
    int print_mip "glp_print_mip" (Prob* prob, const char* fname)

    #  read MIP solution from text file
    int read_mip "glp_read_mip" (Prob* prob, const char* fname)

    #  write MIP solution to text file
    int write_mip "glp_write_mip" (Prob* prob, const char* fname)

    #  check if LP basis factorization exists
    bint bf_exists "glp_bf_exists" (Prob* prob)

    #  compute LP basis factorization; returns retcode
    int factorize "glp_factorize" (Prob* prob)

    #  check if LP basis factorization has been updated
    bint bf_updated "glp_bf_updated" (Prob* prob)

    #  retrieve LP basis factorization control parameters
    void get_bfcp "glp_get_bfcp" (Prob* prob, BfCp* cp)

    #  change LP basis factorization control parameters
    void set_bfcp "glp_set_bfcp" (Prob* prob, const BfCp* cp)

    #  retrieve LP basis header information
    int get_bhead "glp_get_bhead" (Prob* prob, int k)

    #  retrieve row index in the basis header
    int get_row_bind "glp_get_row_bind" (Prob* prob, int row)

    #  retrieve column index in the basis header
    int get_col_bind "glp_get_col_bind" (Prob* prob, int col)

    #  perform forward transformation (solve system B*x = b)
    void ftran "glp_ftran" (Prob* prob, double rhs_pre_x_post[])

    #  perform backward transformation (solve system B'*x = b)
    void btran "glp_btran" (Prob* prob, double rhs_pre_x_post[])

    #  "warm up" LP basis; returns retcode
    int warm_up "glp_warm_up" (Prob* prob)

    #  compute row of the simplex tableau
    int eval_tab_row "glp_eval_tab_row" (Prob* prob, int k,
                                         int ind[], double val[])

    #  compute column of the simplex tableau
    int eval_tab_col "glp_eval_tab_col" (Prob* prob, int k,
                                         int ind[], double val[])

    #  transform explicitly specified row
    int transform_row "glp_transform_row" (Prob* prob, int length,
                                           int ind[], double val[])

    #  transform explicitly specified column
    int transform_col "glp_transform_col" (Prob* prob, int length,
                                           int ind[], double val[])

    #  perform primal ratio test
    int prim_rtest "glp_prim_rtest" (Prob* prob, int length,
                                     const int ind[], const double val[],
                                     int direction, double eps)

    #  perform dual ratio test
    int dual_rtest "glp_dual_rtest" (Prob* prob, int length,
                                     const int ind[], const double val[],
                                     int direction, double eps)

    #  analyze active bound of non-basic variable
    void analyze_bound "glp_analyze_bound" (Prob* prob, int k,
                                            double* min_bnd, int* min_bnd_k,
                                            double* max_bnd, int* max_bnd_k)

    #  analyze objective coefficient at basic variable
    void analyze_coef "glp_analyze_coef" (Prob* prob, int k,
                                          double* min_coef, int* min_coef_k,
                                          double* val_min_coef,
                                          double* max_coef, int* max_coef_k,
                                          double* val_max_coef)

    #  determine reason for calling the callback routine; returns reascode
    int ios_reason "glp_ios_reason" (Tree* tree)

    #  access the problem object
    Prob* ios_get_prob "glp_ios_get_prob" (Tree* tree)

    #  determine size of the branch-and-bound tree
    void ios_tree_size "glp_ios_tree_size" (Tree* tree,
                                            int* a_cnt, int* n_cnt, int* t_cnt)

    #  determine current active subproblem
    int ios_curr_node "glp_ios_curr_node" (Tree* tree)

    #  determine next active subproblem
    int ios_next_node "glp_ios_next_node" (Tree* tree, int node)

    #  determine previous active subproblem
    int ios_prev_node "glp_ios_prev_node" (Tree* tree, int node)

    #  determine parent subproblem
    int ios_up_node "glp_ios_up_node" (Tree* tree, int node)

    #  determine subproblem level
    int ios_node_level "glp_ios_node_level" (Tree* tree, int node)

    #  determine subproblem local bound
    double ios_node_bound "glp_ios_node_bound" (Tree* tree, int node)

    #  find active subproblem with best local bound
    int ios_best_node "glp_ios_best_node" (Tree* tree)

    #  compute relative MIP gap
    double ios_mip_gap "glp_ios_mip_gap" (Tree* tree)

    #  access subproblem application-specific data
    CBInfo ios_node_data "glp_ios_node_data" (Tree* tree, int node)

    #  retrieve additional row attributes
    void ios_row_attr "glp_ios_row_attr" (Tree* tree, int row, Attr* attr)

    #  determine current size of the cut pool
    int ios_pool_size "glp_ios_pool_size" (Tree* tree)

    #  add row (constraint) to the cut pool
    int ios_add_row "glp_ios_add_row" (Tree* tree, const char* name, int klass,
                                       int flags,  #  flags must be 0!
                                       int length,
                                       const int ind[], const double val[],
                                       int vartype, double rhs)

    #  remove row (constraint) from the cut pool
    void ios_del_row "glp_ios_del_row" (Tree* tree, int row)

    #  remove all rows (constraints) from the cut pool
    void ios_clear_pool "glp_ios_clear_pool" (Tree* tree)

    #  check if can branch upon specified variable
    bint ios_can_branch "glp_ios_can_branch" (Tree* tree, int col)

    #  choose variable to branch upon
    void ios_branch_upon "glp_ios_branch_upon" (Tree* tree, int col,
                                                int branch)

    #  select subproblem to continue the search
    void ios_select_node "glp_ios_select_node" (Tree* tree, int node)

    #  provide solution found by heuristic
    int ios_heur_sol "glp_ios_heur_sol" (Tree* tree, const double heur_sol[])

    #  terminate the solution process
    void ios_terminate "glp_ios_terminate" (Tree* tree)

# Not used and (therefore) undocumented.
#
#    #  initialize MPS format control parameters
#    void init_mpscp "glp_init_mpscp" (MPSCp* cp)

    #  read problem data in MPS format
    int read_mps "glp_read_mps" (Prob* prob, int mpsfmt,
                                 const MPSCp* cp,  #  cp must be NULL!
                                 const char* fname)

    #  write problem data in MPS format
    int write_mps "glp_write_mps" (Prob* prob, int mpsfmt,
                                   const MPSCp* cp,  #  cp must be NULL!
                                   const char* fname)

# Not used and (therefore) undocumented.
#
#    #  initialize CPLEX LP format control parameters
#    void init_cpxcp "glp_init_cpxcp" (CPXCp* cp)

    #  read problem data in CPLEX LP format
    int read_lp "glp_read_lp" (Prob* prob,
                               const CPXCp* cp,  #  cp must be NULL!
                               const char* fname)

    #  write problem data in CPLEX LP format
    int write_lp "glp_write_lp" (Prob* prob,
                                 const CPXCp* cp,  #  cp must be NULL!
                                 const char* fname)

    #  read problem data in GLPK format
    int read_prob "glp_read_prob" (Prob* prob,
                                   int flags,  #  flags must be 0!
                                   const char* fname)

    #  write problem data in GLPK format
    int write_prob "glp_write_prob" (Prob* prob,
                                     int flags,  #  flags must be 0!
                                     const char* fname)

    #  allocate the MathProg translator workspace
    Tran* mpl_alloc_wksp "glp_mpl_alloc_wksp" ()

    #  read and translate model section
    int mpl_read_model "glp_mpl_read_model" (Tran* wksp,
                                             const char* fname, bint skip_data)

    #  read and translate data section
    int mpl_read_data "glp_mpl_read_data" (Tran* wksp, const char* fname)

    #  generate the model
    int mpl_generate "glp_mpl_generate" (Tran* wksp, const char* fname)

    #  build LP/MIP problem instance from the model
    void mpl_build_prob "glp_mpl_build_prob" (Tran* wksp, Prob* prob)

    #  postsolve the model
    int mpl_postsolve "glp_mpl_postsolve" (Tran* wksp, Prob* prob,
                                           int solind)

    #  free the MathProg translator workspace
    void mpl_free_wksp "glp_mpl_free_wksp" (Tran* wksp)

    #  stand-alone LP/MIP solver (main function of glpsol;
    # can be used to avoid having to call glpsol in a separate process)
    int main "glp_main" (int argc, const char* argv[])

    #  read CNF-SAT problem data in DIMACS format
    int read_cnfsat "glp_read_cnfsat" (Prob* prob, const char* fname)

    #  check for CNF-SAT problem instance
    int check_cnfsat "glp_check_cnfsat" (Prob* prob)

    #  write CNF-SAT problem data in DIMACS format
    int write_cnfsat "glp_write_cnfsat" (Prob* prob, const char* fname)

    #  solve CNF-SAT problem with MiniSat solver; returns retcode
    int minisat1 "glp_minisat1" (Prob* prob)

    #  solve integer feasibility problem; returns retcode
    int intfeas1 "glp_intfeas1" (Prob* prob, bint use_bound, int obj_bound)

    #  initialization return codes (argument name is 'initretcode'):
    enum: INIT_OK   #  initialization successful
    enum: INIT_UNN  #  already initialized
    enum: INIT_OOM  #  insufficient memory
    enum: INIT_NOK  #  unsupported programming model

    #  initialize GLPK environment; returns initretcode
    int init_env "glp_init_env" ()

    #  determine library version
    const char* version "glp_version" ()

    # Free return codes (argument name is 'freeretcode'):
    enum: FREE_OK   #  termination successful
    enum: FREE_UNN  #  environment is inactive (was not initialized)

    #  free GLPK environment; returns freeretcode
    int free_env "glp_free_env" ()

    # write string on terminal
    void puts "glp_puts" (const char *s)

# Wrapping va_list type args nontrivial
# (cf. https://github.com/cython/cython/wiki/FAQ#wiki-how-do-i-use-variable-args)
#
#    #  write formatted output on terminal
#    void printf "glp_printf" (const char* fmt, ...)
#
#    #  write formatted output on terminal
#    void vprintf "glp_vprintf" (const char* fmt, va_list arg)

    #  enable/disable terminal output
    bint term_out "glp_term_out" (bint flag)

    #  install hook to intercept terminal output
    void term_hook "glp_term_hook" (int (*func)(void* info,
                                                const char* output),
                                    void* info)

    #  start copying terminal output to text file
    int open_tee "glp_open_tee" (const char* fname)

    #  stop copying terminal output to text file
    int close_tee "glp_close_tee" ()

# Currently not fully  wrapped (how to deal with second macro definition?)
#
#    ctypedef void (*_glp_errfunc)(const char* fmt, ...)
#
#    #  display fatal error message and terminate execution
##define glp_error glp_error_(__FILE__, __LINE__)
#    glp_errfunc glp_error_ "error_" (const char* file, int line)
#
#    #  check for logical condition
##define glp_assert(expr) \
#      ((void)((expr) || (glp_assert_(#expr, __FILE__, __LINE__), 1)))
#    void assert_ "glp_assert_" (const char* expr, const char* file, int line)

    #  install hook to intercept abnormal termination
    void error_hook "glp_error_hook" (void (*func)(void* info), void* info)

    #  allocate memory block
    void* alloc "glp_alloc" (int n, int size)

    #  reallocate memory block
    void* realloc "glp_realloc" (void *ptr, int n, int size)

    #  free (deallocate) memory block
    void free "glp_free" (void* ptr)

    #  set memory usage limit
    void mem_limit "glp_mem_limit" (int limit)

    #  get memory usage information
    void mem_usage "glp_mem_usage" (int* count, int* cpeak,
                                    size_t* total, size_t* tpeak)

    #  graph descriptor
    cdef struct Graph "glp_graph":
        void* pool   #  DMP *pool; memory pool to store graph components
        char* name   #  graph name (1 to 255 chars)
                     #   NULL means no name is assigned to the graph
        int nv_max   #  length of the vertex list (enlarged automatically)
        int nv       #  number of vertices in the graph, 0 <= nv <= nv_max
        int na       #  number of arcs in the graph, na >= 0
        Vertex** v   #  glp_vertex *v[1+nv_max]
                     #   v[i], 1 <= i <= nv, is a pointer to i-th vertex
        void* index  #  AVL *index
                     #   vertex index to find vertices by their names
                     #   NULL means the index does not exist
        int v_size   #  size of data associated with each vertex
                     #   (0 to 256 bytes)
        int a_size   #  size of data associated with each arc (0 to 256 bytes)

    #  vertex descriptor
    cdef struct Vertex "glp_vertex":
      int i          #  vertex ordinal number, 1 <= i <= nv
      char* name     #  vertex name (1 to 255 chars)
                     #   NULL means no name is assigned to the vertex
      void* entry    #  AVLNODE *entry
                     #   pointer to corresponding entry in the vertex index
                     #   NULL means that either the index does not exist
                     #   or the vertex has no name assigned
      void* data     #  pointer to data associated with the vertex
      void* temp     #  working pointer
      Arc* inc "in"  #  pointer to the (unordered) list of incoming arcs
      Arc* out       #  pointer to the (unordered) list of outgoing arcs

    #  arc descriptor
    cdef struct Arc "glp_arc":
      Vertex* tail  #  pointer to the tail endpoint
      Vertex* head  #  pointer to the head endpoint
      void* data    #  pointer to data associated with the arc
      void* temp    #  working pointer
      Arc* t_prev   #  pointer to previous arc having the same tail endpoint
      Arc* t_next   #  pointer to next arc having the same tail endpoint
      Arc* h_prev   #  pointer to previous arc having the same head endpoint
      Arc* h_next   #  pointer to next arc having the same head endpoint

    #  create graph
    Graph* create_graph "glp_create_graph" (int vertex_data_size,
                                            int arc_data_size)

    #  assign (change) graph name
    void set_graph_name "glp_set_graph_name" (Graph* graph, const char* name)

    #  add new vertices to graph
    int add_vertices "glp_add_vertices" (Graph* graph, int vertices)

    #  assign (change) vertex name
    void set_vertex_name "glp_set_vertex_name" (Graph* graph, int vertex,
                                                const char* name)

    #  add new arc to graph
    Arc* add_arc "glp_add_arc" (Graph* graph, int tail, int head)

    #  delete vertices from graph
    void del_vertices "glp_del_vertices" (Graph* graph, int vertices,
                                          const int ind[])

    #  delete arc from graph
    void del_arc "glp_del_arc" (Graph* graph, Arc* arc)

    #  erase graph content
    void erase_graph "glp_erase_graph" (Graph* graph,
                                        int vertex_data_size,
                                        int arc_data_size)

    #  delete graph
    void delete_graph "glp_delete_graph" (Graph* graph)

    #  create vertex name index
    void create_v_index "glp_create_v_index" (Graph* graph)

    #  find vertex by its name
    int find_vertex "glp_find_vertex" (Graph* graph, const char* name)

    #  delete vertex name index
    void delete_v_index "glp_delete_v_index" (Graph* graph)

    #  read graph from plain text file
    int read_graph "glp_read_graph" (Graph* graph, const char* fname)

    #  write graph to plain text file
    int write_graph "glp_write_graph" (Graph* graph, const char* fname)

    #  convert minimum cost flow problem to LP
    void mincost_lp "glp_mincost_lp" (Prob* prob, Graph* graph,
                                      bint copy_names, int v_rhs,
                                      int a_low, int a_cap, int a_cost)

    #  find minimum-cost flow with out-of-kilter algorithm; returns retcode
    int mincost_okalg "glp_mincost_okalg" (Graph* graph, int v_rhs,
                                           int a_low, int a_cap, int a_cost,
                                           double* sol, int a_x, int v_pi)

    # find minimum-cost flow with Bertsekas-Tseng relaxation method
    int mincost_relax4 "glp_mincost_relax4" (Graph* graph, int v_rhs,
                                             int a_low, int a_cap, int a_cost,
                                             int crash, double* sol,
                                             int a_x, int a_rc)

    #  convert maximum flow problem to LP
    void maxflow_lp "glp_maxflow_lp" (Prob* prob, Graph* graph,
                                      bint copy_names,
                                      int source, int sink, int a_cap)

    #  find maximal flow with Ford-Fulkerson algorithm; returns retcode
    int maxflow_ffalg "glp_maxflow_ffalg" (Graph* graph, int source, int sink,
                                           int a_cap, double* sol, int a_x,
                                           int v_cut)

    #  check correctness of assignment problem data
    int check_asnprob "glp_check_asnprob" (Graph* graph, int v_set)

    #  assignment problem formulation (argument name 'asnform'):
    enum: ASN_MIN "GLP_ASN_MIN"  #  perfect matching (minimization)
    enum: ASN_MAX "GLP_ASN_MAX"  #  perfect matching (maximization)
    enum: ASN_MMP "GLP_ASN_MMP"  #  maximum matching

    #  convert assignment problem to LP
    int asnprob_lp "glp_asnprob_lp" (Prob* prob, int asnform,
                                     Graph* graph, bint copy_names,
                                     int v_set, int a_cost)

    #  solve assignment problem with out-of-kilter algorithm; returns retcode
    int asnprob_okalg "glp_asnprob_okalg" (int asnform, Graph* graph,
                                           int v_set, int a_cost,
                                           double* sol, int a_x)

    #  find bipartite matching of maximum cardinality
    int asnprob_hall "glp_asnprob_hall" (Graph* graph, int v_set, int a_x)

    #  solve critical path problem
    double cpp "glp_cpp" (Graph* graph, int v_t, int v_es, int v_ls)

    #  read min-cost flow problem data in DIMACS format
    int read_mincost "glp_read_mincost" (Graph* graph, int v_rhs,
                                         int a_low, int a_cap, int a_cost,
                                         const char* fname)

    #  write min-cost flow problem data in DIMACS format
    int write_mincost "glp_write_mincost" (Graph* graph, int v_rhs,
                                           int a_low, int a_cap, int a_cost,
                                           const char* fname)

    #  read maximum flow problem data in DIMACS format
    int read_maxflow "glp_read_maxflow" (Graph* graph, int* source, int* sink,
                                         int a_cap, const char* fname)

    #  write maximum flow problem data in DIMACS format
    int write_maxflow "glp_write_maxflow" (Graph* graph, int source, int sink,
                                           int a_cap, const char* fname)

    #  read assignment problem data in DIMACS format
    int read_asnprob "glp_read_asnprob" (Graph* graph,
                                         int v_set, int a_cost,
                                         const char* fname)

    #  write assignment problem data in DIMACS format
    int write_asnprob "glp_write_asnprob" (Graph* graph,
                                           int v_set, int a_cost,
                                           const char* fname)

    #  read graph in DIMACS clique/coloring format
    int read_ccdata "glp_read_ccdata" (Graph* graph,
                                       int v_wgt, const char* fname)

    #  write graph in DIMACS clique/coloring format
    int write_ccdata "glp_write_ccdata" (Graph* graph,
                                         int v_wgt, const char* fname)

    #  Klingman's network problem generator
    int netgen "glp_netgen" (Graph* graph,
                             int v_rhs, int a_cap, int a_cost,
                             const int param[1+15])

    #  Klingman's standard network problem instance
    void netgen_prob "glp_netgen_prob" (int nprob, int param[1+15])

    #  grid-like network problem generator
    int gridgen "glp_gridgen" (Graph* graph,
                               int v_rhs, int a_cap, int a_cost,
                               const int parm[1+14])

    #  Goldfarb's maximum flow problem generator
    int rmfgen "glp_rmfgen" (Graph* graph, int* source, int* sink,
                             int a_cap, const int param[1+5])

    #  find all weakly connected components of graph
    int weak_comp "glp_weak_comp" (Graph* graph, int v_num)

    #  find all strongly connected components of graph
    int strong_comp "glp_strong_comp" (Graph* graph, int v_num)

    #  topological sorting of acyclic digraph
    int top_sort "glp_top_sort" (Graph* graph, int v_num)

    #  find maximum weight clique with exact algorithm; returns retcode
    int wclique_exact "glp_wclique_exact" (Graph* graph, int v_wgt,
                                           double* sol, int v_set)
