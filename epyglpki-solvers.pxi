# epyglpki-solvers.pxi: Cython/Python interface for GLPK solvers

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


cdef class _Solver(_Component):

    cdef _value(self, varstraint,
                double (*variable_func)(glpk.ProbObj*, int),
                double (*constraint_func)(glpk.ProbObj*, int)):
        if isinstance(varstraint, Variable):
            col = self._program._col(varstraint)
            return variable_func(self._problem, col)
        elif isinstance(varstraint, Constraint):
            row = self._program._row(varstraint)
            return constraint_func(self._problem, row)
        else:
            raise TypeError("varstraint must be a Variable or Constraint")

    def _error(self, soltype, solver):
        cdef double* ae_max = <double*>glpk.alloc(1, sizeof(double))
        cdef int* ae_ind = <int*>glpk.alloc(1, sizeof(int))
        cdef double* re_max = <double*>glpk.alloc(1, sizeof(double))
        cdef int* re_ind = <int*>glpk.alloc(1, sizeof(int))
        try:
            if soltype is 'primal':
                eqtype = glpk.KKT_PE
                bndtype = glpk.KKT_PB
            elif soltype is 'dual':
                eqtype = glpk.KKT_DE
                bndtype = glpk.KKT_DB
            else:
                raise ValueError("soltype should be 'primal' or 'dual'.")
            error = {}
            # equalities
            glpk.check_kkt(self._problem, solver, eqtype,
                           ae_max, ae_ind, re_max, re_ind)
            a_max = ae_max[0]
            a_ind = ae_ind[0]
            r_max = re_max[0]
            r_ind = re_ind[0]
            if eqtype is glpk.KKT_PE:
                a_varstraint = self._program._constraint(a_ind)
                r_varstraint = self._program._constraint(r_ind)
            elif eqtype is glpk.KKT_DE:
                a_varstraint = self._program._variable(a_ind)
                r_varstraint = self._program._variable(r_ind)
            error['equalities'] = {'abs': (a_max, a_varstraint),
                                   'rel': (r_max, r_varstraint)}
            # bounds
            glpk.check_kkt(self._problem, solver, bndtype,
                           ae_max, ae_ind, re_max, re_ind)
            a_max = ae_max[0]
            a_ind = ae_ind[0]
            r_max = re_max[0]
            r_ind = re_ind[0]
            a_varstraint = self._program._varstraint(a_ind)
            r_varstraint = self._program._varstraint(r_ind)
            error['bounds'] = {'abs': (a_max, a_varstraint),
                               'rel': (r_max, r_varstraint)}
            return error
        finally:
            glpk.free(ae_max)
            glpk.free(ae_ind)
            glpk.free(re_max)
            glpk.free(ae_ind)

    cdef _faccess(self,
                  int (*faccess_function)(glpk.ProbObj*, const char*),
                  fname, error_msg):
        retcode = faccess_function(self._problem, name2chars(fname))
        if retcode is not 0:
            raise RuntimeError(error_msg + " '" + fname + "'.")

    cdef _read(self,
               int (*faccess_function)(glpk.ProbObj*, const char*), fname):
        self._faccess(faccess_function, fname, "Error reading file")

    cdef _write(self,
                int (*faccess_function)(glpk.ProbObj*, const char*), fname):
        self._faccess(faccess_function, fname, "Error writing file")


include "epyglpki-simplex.pxi"


include "epyglpki-ipoint.pxi"


include "epyglpki-intopt.pxi"
