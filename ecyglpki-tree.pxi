# ecyglpki-tree.pxi: Cython interface for GLPK search trees

###############################################################################
#
#  This code is part of ecyglpki (a Cython GLPK interface).
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
        return glpk.ios_curr_node(self._tree)  # TODO: 0 retval is special

    def ios_next_node(self, int node):
        """Determine next active subproblem"""
        return glpk.ios_next_node(self._tree, int node)  # TODO: 0 retval is special

    def ios_prev_node(self, int node):
        """Determine previous active subproblem"""
        return glpk.ios_prev_node(self._tree, int node)  # TODO: 0 retval is special

    def ios_up_node(self, int node):
        """Determine parent subproblem"""
        return glpk.ios_up_node(self._tree, int node)  # TODO: 0 retval is special

    def ios_node_level(self, int node):
        """Determine subproblem level"""
        return glpk.ios_node_level(self._tree, int node)  # TODO: 0 retval is special

    def ios_node_bound(self, int node):
        """Determine subproblem local bound"""
        return glpk.ios_node_bound(self._tree, int node)  # TODO: 0 retval is special

    def ios_best_node(self):
        """Find active subproblem with best local bound"""
        return glpk.ios_best_node(self._tree)  # TODO: 0 retval is special

    def ios_mip_gap(self):
        """Compute relative MIP gap"""
        return glpk.ios_mip_gap(self._tree)

    def ios_node_data(self, int node):
        """Access subproblem application-specific data"""
    void* glpk.ios_node_data(self._tree, int node)  # TODO

    def ios_row_attr(self, int row):  # TODO: can row be a name?
        """Retrieve additional row attributes"""
        cdef glpk.RowAttr* attr
        glpk.ios_row_attr(self._tree, row, attr)
        return {'level': attr.level,
                'origin': origin2str[attr.origin],
                'class': klass2str[attr.klass]}

    def ios_pool_size(self):
        """Determine current size of the cut pool"""
        return glpk.ios_pool_size(self._tree)

    def ios_add_row(self, str name, str rowclass):  # TODO: is name useful?
        """Add row (constraint) to the cut pool"""
        return glpk.ios_add_row(self._tree, name2chars(name),
                                str2klass[rowclass], 0,
                                int length, const int ind[], const double val[],  # TODO
                                int vartype, double rhs)  # TODO

    def ios_del_row(self, int row):
        """Remove row (constraint) from the cut pool"""
        glpk.ios_del_row(self._tree, int row)

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

    def ios_heur_sol(self, int node):
        """Select subproblem to continue the search"""
        glpk.ios_heur_sol(self._tree, int node)

    def ios_heur_sol(self, solution):  # TODO
        """Provide solution found by heuristic"""
        if glpk.ios_heur_sol(self._tree, const double heur_sol[]) is not 0:  # TODO
            raise ValueError("Solution rejected.")

    def ios_terminate(self):
        """Terminate the solution process"""
        glpk.ios_terminate(self._tree)
