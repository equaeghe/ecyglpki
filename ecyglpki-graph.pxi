# ecyglpki-graph.pxi: Cython interface for GLPK graph algorithms

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



    cdef struct Graph "glp_graph":
        void* pool   #  DMP *pool; memory pool to store graph components
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


cdef class Arc:
    """A GLPK graph arc"""

    cdef glpk.Arc* _arc

    def __cinit__(self, arc_ptr):
        self._arc = <glpk.Arc*>PyCapsule_GetPointer(arc_ptr(), NULL)

    def _arc_ptr(self):
        """Encapsulate the pointer to the arc object

        The arc object pointer `self._arc` cannot be passed as such as
        an argument to other functions. Therefore we encapsulate it in a
        capsule that can be passed. It has to be unencapsulated after
        reception.

        """
        return PyCapsule_New(self._arc, NULL, NULL)


cdef class Graph:
    """A GLPK graph"""

    ### Object definition, creation, setup, and cleanup ###

    cdef glpk.Graph* _graph

    def __cinit__(self, int vertex_data_size, int arc_data_size,
                  graph_ptr=None):
        if graph_ptr is None:
            self._graph = glpk.create_graph(vertex_data_size, arc_data_size)
            glpk.create_v_index(self._graph)
        else:
            self._graph = <glpk.Graph*>PyCapsule_GetPointer(graph_ptr(), NULL)

    def _graph_ptr(self):
        """Encapsulate the pointer to the graph object

        The graph object pointer `self._graph` cannot be passed as such as
        an argument to other functions. Therefore we encapsulate it in a
        capsule that can be passed. It has to be unencapsulated after
        reception.

        """
        return PyCapsule_New(self._graph, NULL, NULL)

    def __dealloc__(self):
        glpk.delete_v_index(self._graph)
        glpk.delete_graph(self._graph)

    ### Translated GLPK functions ###

    def set_graph_name(self, str name):
        """Assign (change) graph name"""
        glpk.set_graph_name(self._graph, name2chars(name))

    def add_vertices(self, int vertices):
        """Add new vertices to graph"""
        return glpk.add_vertices(self._graph, vertices)

    def add_named_vertices(self, *names):  # variant of add_vertices
        """Add new vertices to graph

        :param names: the names (str strings) of the vertices to add

        """
        cdef int number = len(names)
        if number is 0:
            return
        cdef int first = self.add_vertices(number)
        for vertex, name in enumerate(names, start=first):
            glpk.set_vertex_name(self._graph, vertex, name2chars(name))

    def set_vertex_name(self, vertex, str name):
        """Assign (change) vertex name"""
        vertex = self.find_vertex_as_needed(vertex)
        return set_vertex_name(self._graph, vertex, name2chars(name))

    def add_arc(self, tail, head):
        """Add new arc to graph"""
        tail = self.find_vertex_as_needed(tail)
        head = self.find_vertex_as_needed(head)
        cdef glpk.Arc* arc = glpk.add_arc(self._graph, tail, head)
        return Arc(PyCapsule_New(arc, NULL, NULL))

    def del_vertices(self, *vertices):
        """Delete vertices from graph"""
        cdef int n = len(vertices)
        if n is 0:
            return
        cdef int* inds = <int*>glpk.alloc(1+n, sizeof(int))
        try:
            for i, ind in enumerate(inds, start=1):
                inds[i] = self.find_vertex_as_needed(ind)
            glpk.del_vertices(self._graph, n, inds)
        finally:
            glpk.free(inds)

    def del_arc(self, Arc arc):
        """Delete arc from graph"""
        cdef glpk.Arc* arc = <glpk.Arc*>PyCapsule_GetPointer(arc._arc_ptr(),
                                                             NULL)
        glpk.del_arc(self._graph, arc)

    def erase_graph(self, int vertex_data_size, int arc_data_size)
        """Erase graph content"""
        glpk.erase_graph(self._graph, vertex_data_size, arc_data_size)

    def find_vertex(self, str name):
        """Find vertex by its name"""
        cdef int vertex = glpk.find_vertex(self._graph, name2chars(name))
        if vertex is 0:
            raise ValueError("'" + name + "' is not a vertex name.")
        else:
            return vertex

    def find_vertex_as_needed(self, vertex):
        if isinstance(vertex, int):
            return vertex
        elif isinstance(vertex, str):
            return self.find_vertex(vertex)
        else:
            raise TypeError("'vertex' must be a number ('int') or a name " +
                            "(str 'str'), not '" + type(vertex).__name__ +
                            "'.")

    #  read graph from plain text file
    int read_graph(self._graph, const char* fname)

    #  write graph to plain text file
    int write_graph(self._graph, const char* fname)

    #  convert minimum cost flow problem to LP
    void mincost_lp(Prob* prob, self._graph, bint copy_names, int v_rhs, int a_low, int a_cap, int a_cost)

    #  find minimum-cost flow with out-of-kilter algorithm; returns retcode
    int mincost_okalg(self._graph, int v_rhs, int a_low, int a_cap, int a_cost, double* sol, int a_x, int v_pi)

    # find minimum-cost flow with Bertsekas-Tseng relaxation method
    int mincost_relax4(self._graph, int v_rhs, int a_low, int a_cap, int a_cost, int crash, double* sol, int a_x, int a_rc)

    #  convert maximum flow problem to LP
    void maxflow_lp(Prob* prob, self._graph, bint copy_names, int source, int sink, int a_cap)

    #  find maximal flow with Ford-Fulkerson algorithm; returns retcode
    int maxflow_ffalg(self._graph, int source, int sink, int a_cap, double* sol, int a_x, int v_cut)

    #  check correctness of assignment problem data
    int check_asnprob(self._graph, int v_set)

    #  assignment problem formulation (argument name 'asnform'):
    enum: ASN_MIN "GLP_ASN_MIN"  #  perfect matching (minimization)
    enum: ASN_MAX "GLP_ASN_MAX"  #  perfect matching (maximization)
    enum: ASN_MMP "GLP_ASN_MMP"  #  maximum matching

    #  convert assignment problem to LP
    int asnprob_lp(Prob* prob, int asnform, self._graph, bint copy_names, int v_set, int a_cost)

    #  solve assignment problem with out-of-kilter algorithm; returns retcode
    int asnprob_okalg(int asnform, self._graph, int v_set, int a_cost, double* sol, int a_x)

    #  find bipartite matching of maximum cardinality
    int asnprob_hall(self._graph, int v_set, int a_x)

    #  solve critical path problem
    double cpp(self._graph, int v_t, int v_es, int v_ls)

    #  read min-cost flow problem data in DIMACS format
    int read_mincost(self._graph, int v_rhs, int a_low, int a_cap, int a_cost, const char* fname)

    #  write min-cost flow problem data in DIMACS format
    int write_mincost(self._graph, int v_rhs, int a_low, int a_cap, int a_cost, const char* fname)

    #  read maximum flow problem data in DIMACS format
    int read_maxflow(self._graph, int* source, int* sink, int a_cap, const char* fname)

    #  write maximum flow problem data in DIMACS format
    int write_maxflow(self._graph, int source, int sink, int a_cap, const char* fname)

    #  read assignment problem data in DIMACS format
    int read_asnprob(self._graph, int v_set, int a_cost, const char* fname)

    #  write assignment problem data in DIMACS format
    int write_asnprob(self._graph, int v_set, int a_cost, const char* fname)

    #  read graph in DIMACS clique/coloring format
    int read_ccdata(self._graph, int v_wgt, const char* fname)

    #  write graph in DIMACS clique/coloring format
    int write_ccdata(self._graph, int v_wgt, const char* fname)

    #  Klingman's network problem generator
    int netgen(self._graph, int v_rhs, int a_cap, int a_cost, const int param[1+15])

    #  Klingman's standard network problem instance
    void netgen_prob(int nprob, int param[1+15])

    #  grid-like network problem generator
    int gridgen(self._graph, int v_rhs, int a_cap, int a_cost, const int parm[1+14])

    #  Goldfarb's maximum flow problem generator
    int rmfgen(self._graph, int* source, int* sink, int a_cap, const int param[1+5])

    #  find all weakly connected components of graph
    int weak_comp(self._graph, int v_num)

    #  find all strongly connected components of graph
    int strong_comp(self._graph, int v_num)

    #  topological sorting of acyclic digraph
    int top_sort(self._graph, int v_num)

    #  find maximum weight clique with exact algorithm; returns retcode
    int wclique_exact(self._graph, int v_wgt, double* sol, int v_set)
