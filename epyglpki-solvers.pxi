# epyglpki-solvers.pxi: Cython/Python interface for GLPK solvers

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


cdef class _Solver(_ProgramComponent):

    cdef _faccess(self,
                  int (*faccess_function)(glpk.ProbObj*, const char*),
                  fname, error_msg):
        cdef char* chars
        fname = name2chars(fname)
        chars = fname
        retcode = faccess_function(self._problem, chars)
        if retcode is not 0:
            raise RuntimeError(error_msg + " '" + fname + "'.")

    cdef _read(self,
               int (*faccess_function)(glpk.ProbObj*, const char*), fname):
        self._faccess(faccess_function, fname, "Error reading file")

    cdef _write(self,
                int (*faccess_function)(glpk.ProbObj*, const char*), fname):
        self._faccess(faccess_function, fname, "Error writing file")


cdef class _LPSolver(_Solver):

    cdef _solution(self, varstraints,
                   double (*primal_func)(glpk.ProbObj*, int),
                   double (*dual_func)(glpk.ProbObj*, int), values):
        solution = {}
        if values == 'primal':
            for varstraint in varstraints:
                val = primal_func(self._problem,
                                  self._program._ind(varstraint))
                if val != 0.0:
                    solution[varstraint] = val
        elif values == 'dual':
            for varstraint in varstraints:
                val = dual_func(self._problem, self._program._ind(varstraint))
                if val != 0.0:
                    solution[varstraint] = val
        else:
            raise ValueError("Values must be either 'primal' or 'dual'.")
        return solution


cdef class SimplexSolver(_LPSolver):
    """The problem's simplex solver"""

    cdef glpk.SimplexCP _smcp
    cdef glpk.BasFacCP _bfcp

    def __cinit__(self, program):
        glpk.init_smcp(&self._smcp)
        glpk.get_bfcp(self._problem, &self._bfcp)

    def controls(self, defaults=False, **controls):
        if defaults:
            glpk.init_smcp(&self._smcp)
            glpk.set_bfcp(self._problem, NULL)
            glpk.get_bfcp(self._problem, &self._bfcp)
        for control, val in controls.items():
            # smcp enumerated parameters
            if control is 'msg_lev':
                self._smcp.msg_lev = str2msglev[val]
            elif control is 'meth':
                self._smcp.meth = str2meth[val]
            elif control is 'pricing':
                self._smcp.meth = str2pricing[val]
            elif control is 'r_test':
                self._smcp.r_test = str2rtest[val]
            # bfcp enumerated parameters
            elif control is 'type':
                self._bfcp.type = str2bftype[val]
            # double parameters
            elif control in {
                # smcp
                'tol_bnd', 'tol_dj', 'tol_piv', 'obj_ll', 'obj_ul',
                # bfcp
                'piv_tol', 'eps_tol', 'max_gro', 'upd_tol'
                }:
                if not isinstance(val, numbers.Real):
                    raise TypeError("'" + control + "' value must be real.")
                # smcp
                elif control is 'tol_bnd':
                    self._smcp.tol_bnd = val
                elif control is 'tol_dj':
                    self._smcp.tol_dj = val
                elif control is 'tol_piv':
                    self._smcp.tol_piv = val
                elif control is 'obj_ll':
                    self._smcp.obj_ll = val
                elif control is 'obj_ul':
                    self._smcp.obj_ul = val
                # bfcp
                elif control is 'piv_tol':
                    self._bfcp.piv_tol = val
                elif control is 'eps_tol':
                    self._bfcp.eps_tol = val
                elif control is 'max_gro':
                    self._bfcp.max_gro = val
                elif control is 'upd_tol':
                    self._bfcp.upd_tol = val
            # int parameters
            elif control in {
                # smcp
                'it_lim', 'tm_lim', 'out_frq', 'out_dly',
                # bfcp
                'lu_size', 'piv_lim', 'nfs_max', 'nrs_max', 'rs_size'
                }:
                if not isinstance(val, numbers.Integral):
                    raise TypeError("'" + control + "' value must be integer.")
                # smcp
                elif control is 'it_lim':
                    self._smcp.it_lim = val
                elif control is 'tm_lim':
                    self._smcp.tm_lim = val
                elif control is 'out_frq':
                    self._smcp.out_frq = val
                elif control is 'out_dly':
                    self._smcp.out_dly = val
                # bfcp
                elif control is 'lu_size':
                    self._bfcp.lu_size = val
                elif control is 'piv_lim':
                    self._bfcp.piv_lim = val
                elif control is 'nfs_max':
                    self._bfcp.nfs_max = val
                elif control is 'nrs_max':
                    self._bfcp.nrs_max = val
                elif control is 'rs_size':
                    self._bfcp.rs_size = val
            # bint parameters
            elif control in {
                # smcp
                'presolve',
                # bfcp
                'suhl'
                }:
                if not isinstance(val, bool):
                    raise TypeError("'" + control + "' value must be Boolean.")
                # smcp
                elif control is 'presolve':
                    self._smcp.presolve = val
                # bfcp
                elif control is 'suhl':
                    self._bfcp.suhl = val
            else:
                raise ValueError("Non-existing control: " + repr(control))
        glpk.set_bfcp(self._problem, &self._bfcp)
        scontrols = {}
        scontrols = self._smcp
        scontrols['msg_lev'] = msglev2str[scontrols['msg_lev']]
        scontrols['meth'] = meth2str[scontrols['meth']]
        scontrols['pricing'] = pricing2str[scontrols['pricing']]
        scontrols['r_test'] = rtest2str[scontrols['r_test']]
        fcontrols = {}
        fcontrols = self._bfcp
        fcontrols['type'] = bftype2str[fcontrols['type']]
        controls = {}
        controls.update(scontrols)
        controls.update(fcontrols)
        return controls

    def solve(self, exact=False):
        if exact:
            if meth2str[self._smcp.meth] is not 'primal':
                raise ValueError("Only primal simplex with exact arithmetic.")
            retcode = glpk.simplex_exact(self._problem, &self._smcp)
        else:
            if ((meth2str[self._smcp.meth] is not 'dual') and
                ((self._smcp.obj_ll > -DBL_MAX) or
                 (self._smcp.obj_ul < +DBL_MAX))):
                 raise ValueError("Objective function limits only with " +
                                  "dual simplex.")
            retcode = glpk.simplex(self._problem, &self._smcp)
        if retcode is 0:
            return self.status()
        elif retcode in {glpk.EOBJLL, glpk.EOBJUL}:
            return smretcode2str[retcode]
        else:
            raise smretcode2error[retcode]

    def status(self, detailed=False):
        """Return the current solution status

        :param detailed: whether to give a detailed solution status
        :type detailed: :class:`bool`
        :returns: the current solution status

            * in case `detailed` is :data:`False`, either :data:`'undefined'`,
              :data:`'optimal`, :data:`'infeasible'`, :data:`'no feasible'`,
              :data:`'feasible'`, or :data:`'unbounded'`
            * in case `detailed` is :data:`True`, a pair of statuses is given,
              one for the primal solution and one for the dual solution, either
              :data:`'undefined'`, :data:`'infeasible'`, :data:`'no feasible'`,
              or :data:`'feasible'`

        :rtype: :class:`str` or length-2 :class:`tuple` of :class:`str`

        .. todo::

            Add doctest

        """
        status = solstat2str[glpk.sm_status(self._problem)]
        if detailed:
            return (status, (solstat2str[glpk.sm_prim_stat(self._problem)],
                             solstat2str[glpk.sm_dual_stat(self._problem)]))
        else:
            return status

    def objective(self):
        """Return the objective value for the current solution

        :returns: the objective value for the current solution
        :rtype: :class:`float`

        .. todo::

            Add doctest

        """
        return glpk.sm_obj_val(self._problem)

    def variables(self, values='primal'):
        return self._solution(self._program._variables,
                              glpk.sm_col_prim, glpk.sm_col_dual, values)

    def constraints(self, values='primal'):
        return self._solution(self._program._constraints,
                              glpk.sm_row_prim, glpk.sm_row_dual, values)

    def unboundedness(self):
        ind = glpk.sm_unbnd_ray(self._problem)
        constraints = len(self._program._constraints)
        if ind is 0:
            return (None, '')
        elif ind <= constraints:
            varstraint = self._program._constraints[ind]
            varstat = glpk.get_row_stat(self._problem, ind)
        else:
            ind -= constraints
            varstraint = self._program._variables[ind]
            varstat = glpk.get_col_stat(self._problem, ind)
        nature = 'dual' if varstat is glpk.BS else 'primal'
        return (varstraint, nature)

    def basis(self, basis=None, warmup=False):
        if basis is 'standard':
            glpk.std_basis(self._problem)
        elif basis is 'advanced':
            glpk.adv_basis(self._problem, 0)
        elif basis is 'Bixby':
            glpk.cpx_basis(self._problem)
        elif basis is not None:
            for varstraint, string in basis.items():
                varstat = str2varstat[string]
                if isinstance(varstraint, Variable):
                    col = self._program._ind(varstraint)
                    glpk.set_col_stat(self._problem, col, varstat)
                elif isinstance(varstraint, Constraint):
                    row = self._program._ind(varstraint)
                    glpk.set_row_stat(self._problem, row, varstat)
                else:
                    raise TypeError("Only 'Variable' and 'Constraint' " +
                                    "can have a status.")
        if warmup:
            retcode = glpk.warm_up(self._problem)
            if retcode is not 0:
                raise smretcode2error[retcode]
        basis = {}
        for col, variable in enumerate(self._program._variables, start=1):
            varstat = glpk.get_col_stat(self._problem, col)
            basis[variable] = varstat2str[varstat]
        for row, constraint in enumerate(self._program._constraints, start=1):
            varstat = glpk.get_row_stat(self._problem, row)
            basis[constraint] = varstat2str[varstat]
        return basis

    def print_solution(self, fname):
        """Write the solution to a file in a readable format

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.print_sol, fname)

    def read_solution(self, fname):
        """Read the solution from a file

        :param fname: the name of the file to read from
            (written by :meth:`SimplexSolver.write_solution`)
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error reading the file

        .. todo::

            Add doctest

        """
        self._read(glpk.read_sol, fname)

    def write_solution(self, fname):
        """Write the solution to a file

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.write_sol, fname)

    def print_ranges(self, varstraints, fname):
        if self.status() is not 'optimal':
            raise Exception("Solution must be optimal.")
        if not isinstance(varstraints, collections.abc.Sequence):
            raise TypeError("'varstraints' must be a sequence.")
        length = len(varstraints)
        cdef char* chars
        fname = name2chars(fname)
        chars = fname
        cdef int* indlist = <int*>glpk.calloc(1+length, sizeof(int))
        try:
            rows = len(self._program._constraints)
            for pos, varstraint in enumerate(varstraints, start=1):
                ind = self._program._ind(varstraint)
                if isinstance(varstraint, Variable):
                    ind += rows
                indlist[pos] = ind
            if not glpk.bf_exists(self._problem):
                retcode = glpk.factorize(self._problem)
                if retcode is not 0:
                    raise smretcode2error[retcode]
            glpk.print_ranges(self._problem, length, indlist, 0, chars)
        finally:
            glpk.free(indlist)


cdef class IPointSolver(_LPSolver):
    """The problem's interior point solver"""

    cdef glpk.IPointCP _iptcp

    def __cinit__(self, program):
        glpk.init_iptcp(&self._iptcp)

    def controls(self, defaults=False, **controls):
        if defaults:
            glpk.init_iptcp(&self._iptcp)
        for control, val in controls.items():
            if control is 'msg_lev':
                self._iptcp.msg_lev = str2msglev[val]
            elif control is 'ord_alg':
                self._iptcp.ord_alg = str2ordalg[val]
            else:
                raise ValueError("Non-existing control: " + repr(control))
        controls = {}
        controls = self._iptcp
        controls['msg_lev'] = msglev2str[controls['msg_lev']]
        controls['ord_alg'] = ordalg2str[controls['ord_alg']]
        return controls

    def solve(self):
        """Solve the linear program

        :returns: solution status; see :meth:`IPointSolver.status` for details
        :rtype: :class:`str`
        :raises ValueError: if the problem has no rows/columns
        :raises ArithmeticError: if there occurs very slow convergence or
            divergence
        :raises StopIteration: if the iteration limit is exceeded
        :raises ArithmeticError: if numerical instability occurs on solving
            the Newtonian system

        .. todo::

            Add doctest

        """
        retcode = glpk.interior(self._problem, &self._iptcp)
        if retcode is 0:
            return self.status()
        else:
            raise iptretcode2error[retcode]

    def status(self):
        """Return the current solution status

        :returns: the current solution status, either :data:`'undefined'`,
            :data:`'optimal`, :data:`'infeasible'`, or :data:`'no feasible'`
        :rtype: :class:`str`

        .. todo::

            Add doctest

        """
        return solstat2str[glpk.ipt_status(self._problem)]

    def objective(self):
        """Return the objective value for the current solution

        :returns: the objective value for the current solution
        :rtype: :class:`float`

        .. todo::

            Add doctest

        """
        return glpk.ipt_obj_val(self._problem)

    def variables(self, values='primal'):
        return self._solution(self._program._variables,
                              glpk.ipt_col_prim, glpk.ipt_col_dual, values)

    def constraints(self, values='primal'):
        return self._solution(self._program._constraints,
                              glpk.ipt_row_prim, glpk.ipt_row_dual, values)

    def print_solution(self, fname):
        """Write the solution to a file in a readable format

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.print_ipt, fname)

    def read_solution(self, fname):
        """Read the solution from a file

        :param fname: the name of the file to read from
            (written by :meth:`IPointSolver.write_solution`)
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error reading the file

        .. todo::

            Add doctest

        """
        self._read(glpk.read_ipt, fname)

    def write_solution(self, fname):
        """Write the solution to a file

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.write_ipt, fname)


cdef class IntOptSolver(_Solver):
    """The problem's integer optimization solver"""

    cdef glpk.IntOptCP _iocp

    def __cinit__(self, program):
        glpk.init_iocp(&self._iocp)

    def controls(self, defaults=False, **controls):
        if defaults:
            glpk.init_iocp(&self._iocp)
        for control, val in controls.items():
            if control is 'msg_lev':
                self._iocp.msg_lev = str2msglev[val]
            elif control is 'br_tech':
                self._iocp.br_tech = str2brtech[val]
            elif control is 'bt_tech':
                self._iocp.bt_tech = str2bttech[val]
            elif control is 'pp_tech':
                self._iocp.pp_tech = str2pptech[val]
            elif control in {'tol_int', 'tol_obj', 'mip_gap'}:
                if not isinstance(val, numbers.Real):
                    raise TypeError("'" + control + "' value must be real.")
                elif control is 'tol_int':
                    self._iocp.tol_int = val
                elif control is 'tol_obj':
                    self._iocp.tol_obj = val
                elif control is 'mip_gap':
                    self._iocp.mip_gap = val
            elif control in {'tm_lim', 'out_frq', 'out_dly', 'cb_size'}:
                if not isinstance(val, numbers.Integral):
                    raise TypeError("'" + control + "' value must be integer.")
                elif control is 'tm_lim':
                    self._iocp.tm_lim = val
                elif control is 'out_frq':
                    self._iocp.out_frq = val
                elif control is 'out_dly':
                    self._iocp.out_dly = val
                elif control is 'cb_size':
                    self._iocp.cb_size = val
            elif control in {'mir_cuts', 'gmi_cuts', 'cov_cuts', 'clq_cuts',
                             'presolve', 'binarize', 'fp_heur'}:
                if not isinstance(val, bool):
                    raise TypeError("'" + control + "' value must be Boolean.")
                elif control is 'mir_cuts':
                    self._iocp.mir_cuts = val
                elif control is 'gmi_cuts':
                    self._iocp.gmi_cuts = val
                elif control is 'cov_cuts':
                    self._iocp.cov_cuts = val
                elif control is 'clq_cuts':
                    self._iocp.clq_cuts = val
                elif control is 'presolve':
                    self._iocp.presolve = val
                elif control is 'binarize':
                    self._iocp.binarize = val
                elif control is 'fp_heur':
                    self._iocp.fp_heur = val
            else:
                raise ValueError("Non-existing control: " + repr(control))
        controls = {}
        controls = self._iocp
        controls['msg_lev'] = msglev2str[controls['msg_lev']]
        controls['br_tech'] = brtech2str[controls['br_tech']]
        controls['bt_tech'] = bttech2str[controls['bt_tech']]
        controls['pp_tech'] = pptech2str[controls['pp_tech']]
        return controls

    def solve(self, solver='branchcut', obj_bound=False):
        if solver is 'branchcut':
            retcode = glpk.intopt(self._problem, &self._iocp)
        elif solver is 'intfeas1':
            if obj_bound is False:
                retcode = glpk.intfeas1(self._problem, False, 0)
            elif isinstance(obj_bound, numbers.Integral):
                retcode = glpk.intfeas1(self._problem, True, obj_bound)
            else:
                raise TypeError("'obj_bound' must be an integer.")
        else:
            raise ValueError("The only available solvers are 'branchcut' " +
                             "and 'intfeas1'.")
        if retcode is 0:
            return self.status()
        else:
            raise ioretcode2error[retcode]

    def status(self):
        """Return the current solution status

        :returns: the current solution status, either :data:`'undefined'`,
            :data:`'optimal`, :data:`'no feasible'`, or :data:`'feasible'`
        :rtype: :class:`str`

        .. todo::

            Add doctest

        """
        return solstat2str[glpk.mip_status(self._problem)]

    def objective(self):
        """Return the objective value for the current solution

        :returns: the objective value for the current solution
        :rtype: :class:`float`

        .. todo::

            Add doctest

        """
        return glpk.mip_obj_val(self._problem)

    def variables(self):
        """Returns the values of the variables for the current solution

        :returns: the nonzero values of the variables for the current
            solution
        :rtype: :class:`dict` from :class:`Variable` to :class:`float` or 
            :class:`int`

        .. todo::

            Add doctest

        """
        solution = {}
        for col, variable in enumerate(self._program._variables, start=1):
            val = glpk.mip_col_val(self._problem, col)
            if variable.kind() in {'integer', 'binary'}:
                val = int(val)
            if val != 0:
                solution[variable] = val
        return solution

    def constraints(self):
        """Returns the values of the constraints for the current solution

        :returns: the nonzero values of the constraints for the current
            solution
        :rtype: :class:`dict` from :class:`Constraint` to :class:`float`

        .. todo::

            Add doctest

        """
        solution = {}
        for row, constraint in enumerate(self._program._constraints, start=1):
            val = glpk.mip_row_val(self._problem, row)
            if val != 0:
                solution[constraint] = val
        return solution

    def print_solution(self, fname):
        """Write the solution to a file in a readable format

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.print_mip, fname)

    def read_solution(self, fname):
        """Read the solution from a file

        :param fname: the name of the file to read from
            (written by :meth:`IntOptSolver.write_solution`)
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error reading the file

        .. todo::

            Add doctest

        """
        self._read(glpk.read_mip, fname)

    def write_solution(self, fname):
        """Write the solution to a file

        :param fname: the name of the file to write to
        :type fname: :class:`str`
        :raises RuntimeError: if there is an error writing the file

        .. todo::

            Add doctest

        """
        self._write(glpk.write_mip, fname)
