# ecyglpki-tree.pxi: Cython interface for GLPK search trees

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
#  ecyglpki is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
#  details.
#
#  You should have received a copy of the GNU General Public License
#  along with ecyglpki. If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################


# reason codes
cdef reascode2str = {
    glpk.IROWGEN: 'rowgen',
    glpk.IBINGO: 'improved',
    glpk.IHEUR: 'heur_sol',
    glpk.ICUTGEN: 'cutgen',
    glpk.IBRANCH: 'branch',
    glpk.ISELECT: 'subproblem',
    glpk.IPREPRO: 'preprocess'
}

# row origin flag
cdef origin2str = {
    glpk.RF_REG: 'regular',
    glpk.RF_LAZY: 'lazy',
    glpk.RF_CUT: 'cut'
}
# row class descriptor
cdef klass2str = {
    glpk.RF_GMI: 'Gomory',
    glpk.RF_MIR: 'rounding',
    glpk.RF_COV: 'cover',
    glpk.RF_CLQ: 'clique'
}
cdef str2klass = {string: klass for klass, string in klass2str.items()}


#  branch selection indicator
cdef str2branchdir = {
    glpk.NO_BRNCH: 'no',
    glpk.DN_BRNCH: 'down',
    glpk.UP_BRNCH: 'up'
}


cdef class Tree:
    """A GLPK search tree"""

    ### Object definition and setup ###

    cdef glpk.Tree* _tree
    cdef readonly Problem problem

    def __cinit__(self):
        problem = Problem(PyCapsule_New(glpk.ios_get_prob(self._tree),
                                        NULL, NULL))

    ### Translated GLPK functions ###

    def ios_reason(self):
        """Determine reason for calling the callback routine"""
        return reascode2str[glpk.ios_reason(self._tree)]

    def ios_tree_size(self):
        """Determine size of the branch-and-bound tree"""
        cdef int a_cnt
        cdef int n_cnt
        cdef int t_cnt
        glpk.ios_tree_size(self._tree, &a_cnt, &n_cnt, &t_cnt)
        return {'active': a_cnt, 'current': n_cnt, 'total': t_cnt}

    def ios_curr_node(self):
        """Determine current active subproblem"""
        node = glpk.ios_curr_node(self._tree)
        if node is 0:
            raise Exception("There is no current active subproblem.")
        return node

    def ios_next_node(self, int node):
        """Determine next active subproblem"""
        if node is 0:
            return self.ios_first_node()
        else:
            node = glpk.ios_next_node(self._tree, node)
            if node is 0:
                raise Exception("There is no next active subproblem.")
            return node

    def ios_first_node(self):  # variant of ios_next_node
        """Determine first active subproblem"""
        node = glpk.ios_next_node(self._tree, 0)
        if node is 0:
            raise Exception("The search tree is empty.")
        return node

    def ios_prev_node(self, int node):
        """Determine previous active subproblem"""
        if node is 0:
            return self.ios_last_node()
        else:
            node = glpk.ios_prev_node(self._tree, node)
            if node is 0:
                raise Exception("There is no previous active subproblem.")
            return node

    def ios_last_node(self):  # variant of ios_prev_node
        """Determine last active subproblem"""
        node = glpk.ios_next_node(self._tree, 0)
        if node is 0:
            raise Exception("The search tree is empty.")
        return node

    def ios_up_node(self, int node):
        """Determine parent subproblem"""
        node = glpk.ios_up_node(self._tree, node)
        if node is 0:
            raise Exception("This is the search tree root; it has no parent.")
        return node

    def ios_node_level(self, int node):
        """Determine subproblem level"""
        return glpk.ios_node_level(self._tree, node)

    def ios_node_bound(self, int node):
        """Determine subproblem local bound"""
        return glpk.ios_node_bound(self._tree, node)

    def ios_best_node(self):
        """Find active subproblem with best local bound"""
        node = glpk.ios_best_node(self._tree)
        if node is 0:
            raise Exception("The search tree is empty.")
        return node

    def ios_mip_gap(self):
        """Compute relative MIP gap"""
        return glpk.ios_mip_gap(self._tree)

    def ios_node_data(self, int node):  # TODO: use Python object for node data?
        """Access subproblem application-specific data"""
        cdef glpk.CBInfo info = glpk.ios_node_data(self._tree, node)
        if info is not NULL:
            return CallbackInfo(PyCapsule_New(info, NULL, NULL))
        else:
            return None

    def ios_row_attr(self, row):
        """Retrieve additional row attributes"""
        cdef glpk.Attr* attr
        row = self.problem.find_row_as_needed(row)
        glpk.ios_row_attr(self._tree, row, attr)
        return {'level': attr.level,
                'origin': origin2str[attr.origin],
                'class': klass2str[attr.klass]}

    def ios_pool_size(self):
        """Determine current size of the cut pool"""
        return glpk.ios_pool_size(self._tree)

    def ios_add_row(self, str name, str rowclass,
                    coeffs, str vartype, double rhs):
        """Add row (constraint) to the cut pool"""
        _coeffscheck(coeffs)
        cdef int k = len(coeffs)
        cdef int* cols = <int*>glpk.alloc(1+k, sizeof(int))
        cdef double* vals = <double*>glpk.alloc(1+k, sizeof(double))
        try:
            for i, item in enumerate(coeffs.items(), start=1):
                cols[i] = self.problem.find_col_as_needed(item[0])
                vals[i] = item[1]
            return glpk.ios_add_row(self._tree, name2chars(name),
                                    str2klass[rowclass], 0,
                                    k, cols, vals, str2vartype[vartype], rhs)
        finally:
            glpk.free(cols)
            glpk.free(vals)

    def ios_del_row(self, int row):  # row is row in cut pool, not in problem object!
        """Remove row (constraint) from the cut pool"""
        glpk.ios_del_row(self._tree, row)

    def ios_clear_pool(self):
        """Remove all rows (constraints) from the cut pool"""
        glpk.ios_clear_pool(self._tree)

    def ios_can_branch(self, col):
        """Check if can branch upon specified variable"""
        col = self.problem.find_col_as_needed(col)
        return glpk.ios_can_branch(self._tree, col)

    def ios_branch_upon(self, col, str branchdir):
        """Choose variable to branch upon"""
        col = self.problem.find_col_as_needed(col)
        glpk.ios_branch_upon(self._tree, col, str2branchdir[branchdir])

    def ios_select_node(self, int node):
        """Select subproblem to continue the search"""
        glpk.ios_select_node(self._tree, node)

    def ios_heur_sol(self, coeffs):
        """Provide solution found by heuristic"""
        _coeffscheck(coeffs)
        cdef int n = self.problem.get_num_cols()
        cdef double* heur_sol = <double*>glpk.alloc(1+n, sizeof(double))
        try:
            for i in range(1, 1+n):
                heur_sol[i] = 0
            for col, val in coeffs:
                heur_sol[self.problem.find_col_as_needed(col)] = val
            if glpk.ios_heur_sol(self._tree, heur_sol) is not 0:
                raise ValueError("Solution rejected.")
        finally:
            glpk.free(heur_sol)

    def ios_terminate(self):
        """Terminate the solution process"""
        glpk.ios_terminate(self._tree)
