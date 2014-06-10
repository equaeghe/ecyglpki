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


cdef class Tree:
    """A GLPK search tree"""

    ### Object definition ###

    cdef glpk.Tree* _tree

    ### Translated GLPK functions ###

    def ios_reason(self):
        """Determine reason for calling the callback routine"""
        return reascode2str[glpk.ios_reason(self._tree)]

    def ios_get_prob(self):
        """Access the problem object"""
        return glpk.ios_get_prob(self._tree)
            # TODO: move problem.prefer_names to Problem methods
            # to make Problem self-contained
    ProbObj* ios_get_prob "glp_ios_get_prob" (Tree* tree)

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
    void* ios_node_data "glp_ios_node_data" (Tree* tree, int node)

    #  retrieve additional row attributes
    void ios_row_attr "glp_ios_row_attr" (Tree* tree, int row, RowAttr* attr)

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
