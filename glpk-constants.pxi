# glpk-constants.pxi: Cython/Python interface for GLPK constants

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


from libc.limits cimport INT_MAX
from libc.float cimport DBL_MAX

# GENERAL
# MPS file format
cdef str2mpsfmt = {
    'fixed': glpk.MPS_DECK,
    'free': glpk.MPS_FILE
    }

# scaling options
cdef str2scalopt = {
    'geometric': glpk.SF_GM,
    'equilibration': glpk.SF_EQ,
    'round': glpk.SF_2N,
    'skip': glpk.SF_SKIP,
    'auto': glpk.SF_AUTO
    }

# variable types
cdef pair2vartype = {
    (False, False): glpk.FR,
    (True, False): glpk.LO,
    (False, True): glpk.UP,
    (True, True): glpk.DB
    }

# variable kinds
cdef str2varkind = {
    'continuous': glpk.CV,
    'integer': glpk.IV,
    'binary': glpk.BV
    }
cdef varkind2str = {varkind: string for string, varkind in str2varkind.items()}

# optimization directions
cdef str2optdir = {
    'minimize': glpk.MIN,
    'maximize': glpk.MAX
    }
cdef optdir2str = {optdir: string for string, optdir in str2optdir.items()}

# message levels
cdef str2msglev = {
    'no': glpk.MSG_OFF,
    'warnerror': glpk.MSG_ERR,
    'normal': glpk.MSG_ON,
    'full': glpk.MSG_ALL
    }
cdef msglev2str = {msg_lev: string for string, msg_lev in str2msglev.items()}

# solution statuses
cdef solstat2str = {
    glpk.UNDEF: "undefined",
    glpk.OPT: "optimal",
    glpk.INFEAS: "infeasible",
    glpk.NOFEAS: "no feasible",
    glpk.FEAS: "feasible",
    glpk.UNBND: "unbounded",
    }


# SIMPLEX-SPECIFIC
# simplex method
cdef str2meth = {
    'primal': glpk.PRIMAL,
    'dual': glpk.DUAL,
    'dual_fail_primal': glpk.DUALP
    }
cdef meth2str = {meth: string for string, meth in str2meth.items()}

# pricing strategy
cdef str2pricing = {
    'Dantzig': glpk.PT_STD,
    'steepest': glpk.PT_PSE
    }
cdef pricing2str = {pricing: string for string, pricing in str2pricing.items()}

# ratio test type
cdef str2rtest = {
    'standard': glpk.RT_STD,
    'Harris': glpk.RT_HAR
    }
cdef rtest2str = {r_test: string for string, r_test in str2rtest.items()}

# basis factorization approach
cdef str2bftype = {
    'Forrest-Tomlin': glpk.BF_FT,
    'Bartels-Golub': glpk.BF_BG,
    'Givens': glpk.BF_GR
    }
cdef bftype2str = {bftype: string for string, bftype in str2bftype.items()}

# variable status
cdef str2varstat = {
    'basic': glpk.BS,
    'lower': glpk.NL,
    'upper': glpk.NU,
    'free': glpk.NF,
    'fixed': glpk.NS
    }
cdef varstat2str = {varstat: string for string, varstat in str2varstat.items()}

# return codes (errors)
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

# return codes (message)
cdef smretcode2str = {
    glpk.EOBJLL: "Objective function has reached its lower limit.",
    glpk.EOBJUL: "Objective function has reached its upper limit.",
    }


# INTERIOR POINT-SPECIFIC
# ordering algorithm
cdef str2ordalg = {
    'orig': glpk.ORD_NONE,
    'qmd': glpk.ORD_QMD,
    'amd': glpk.ORD_AMD,
    'symamd': glpk.ORD_SYMAMD
    }
cdef ordalg2str = {ord_alg: string for string, ord_alg in str2ordalg.items()}

# return codes (errors)
cdef iptretcode2error = {
    glpk.EFAIL: ValueError("The problem has no rows/columns."),
    glpk.ENOCVG: ArithmeticError("Very slow convergence or divergence."),
    glpk.EITLIM: StopIteration("Iteration limit exceeded."),
    glpk.EINSTAB: ArithmeticError("Numerical instability " +
                                  "on solving Newtonian system.")
    }


# INTEGER OPTIMIZATION-SPECIFIC
# branching technique
cdef str2brtech = {
    'first_fracvar': glpk.BR_FFV,
    'last_fracvar': glpk.BR_LFV,
    'most_fracvar': glpk.BR_MFV,
    'Driebeck-Tomlin': glpk.BR_DTH, 
    'hybrid_peudocost': glpk.BR_PCH
    }
cdef brtech2str = {br_tech: string for string, br_tech in str2brtech.items()}

# backtracking technique
cdef str2bttech = {
    'depth': glpk.BT_DFS,
    'breadth': glpk.BT_BFS,
    'bound': glpk.BT_BLB,
    'projection': glpk.BT_BPH
    }
cdef bttech2str = {bt_tech: string for string, bt_tech in str2bttech.items()}

# preprocessing technique
cdef str2pptech = {
    'none': glpk.PP_NONE,
    'root': glpk.PP_ROOT,
    'all': glpk.PP_ALL
    }
cdef pptech2str = {pp_tech: string for string, pp_tech in str2pptech.items()}

# return codes (errors)
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
