# ecyglpki-graph.pxi: Cython interface for GLPK graph algorithms

###############################################################################
#
#  This code is part of ecyglpki (a Cython GLPK interface).
#
#  Copyright ⓒ 2014 Erik Quaeghebeur. All rights reserved.
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


cdef class Vertex:
    """A GLPK graph vertex"""

    cdef glpk.Vertex* _vertex

    def __cinit__(self, vertex_ptr):
        self._vertex = <glpk.Vertex*>PyCapsule_GetPointer(vertex_ptr(), NULL)

    def _vertex_ptr(self):
        """Encapsulate the pointer to the arc object

        The arc object pointer `self._vertex` cannot be passed as such as
        an argument to other functions. Therefore we encapsulate it in a
        capsule that can be passed. It has to be unencapsulated after
        reception.

        """
        return PyCapsule_New(self._vertex, NULL, NULL)

    ### Selected struct elements become properties ###

    property i:
        """Vertex ordinal number, an `int`"""
        def __get__(self):
            return self._vertex.i

    property name:
        """Vertex name, a `str`"""
        def __get__(self):
            return chars2name(self._vertex.name)

    property inc:
        """First incoming arc, an `.Arc`"""
        def __get__(self):
            cdef glpk.Arc* _arc = self._vertex.inc
            return Arc(PyCapsule_New(_arc, NULL, NULL))

    property out:
        """First outgoing arc, an `.Arc`"""
        def __get__(self):
            cdef glpk.Arc* _arc = self._vertex.out
            return Arc(PyCapsule_New(_arc, NULL, NULL))


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

    ### Selected struct elements become properties ###

    property tail:
        """Tail vertex, a `.Vertex`"""
        def __get__(self):
            cdef glpk.Vertex* _vertex = self._arc.tail
            return Vertex(PyCapsule_New(_vertex, NULL, NULL))

    property head:
        """Head vertex, a `.Vertex`"""
        def __get__(self):
            cdef glpk.Vertex* _vertex = self._arc.head
            return Vertex(PyCapsule_New(_vertex, NULL, NULL))

    property t_prev:
        """Previous arc having the same tail endpoint, an `.Arc`"""
        def __get__(self):
            cdef glpk.Arc* _arc = self._arc.t_prev
            return Arc(PyCapsule_New(_arc, NULL, NULL))

    property t_next:
        """Next arc having the same tail endpoint, an `.Arc`"""
        def __get__(self):
            cdef glpk.Arc* _arc = self._arc.t_next
            return Arc(PyCapsule_New(_arc, NULL, NULL))

    property h_prev:
        """Previous arc having the same head endpoint, an `.Arc`"""
        def __get__(self):
            cdef glpk.Arc* _arc = self._arc.h_prev
            return Arc(PyCapsule_New(_arc, NULL, NULL))

    property h_next:
        """Next arc having the same head endpoint, an `.Arc`"""
        def __get__(self):
            cdef glpk.Arc* _arc = self._arc.h_next
            return Arc(PyCapsule_New(_arc, NULL, NULL))


#  assignment problem formulation
cdef str2asnform = {
    'perfect_maximize': glpk.ASN_MIN,
    'perfect_minimize': glpk.ASN_MAX,
    'maximum': glpk.ASN_MMP
    }


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

    ### Selected struct elements become properties ###

    property nv:
        """The number of vertices in the graph, an `int`"""
        def __get__(self):
            return self._graph.nv

    property na:
        """The number of arcs in the graph, an `int`"""
        def __get__(self):
            return self._graph.na

    property v_size:
        """Size of data associated with each vertex, an `int`"""
        def __get__(self):
            return self._graph.v_size

    property a_size:
        """size of data associated with each arc, an `int`"""
        def __get__(self):
            return self._graph.a_size

    ### Translated GLPK functions ###

    def set_graph_name(self, str name):
        """Assign (change) graph name

        :param name: the name of the graph, it may not exceed 255 bytes
            *encoded as UTF-8*
        :type name: `str`

        """
        glpk.set_graph_name(self._graph, name2chars(name))

    def add_vertices(self, int vertices):
        """Add new vertices to graph"""
        return glpk.add_vertices(self._graph, vertices)

    def add_named_vertices(self, *names):  # variant of add_vertices
        """Add new vertices to graph

        :param names: the names of the vertices to add, none may exceed 255
            bytes *encoded as UTF-8*
        :type names: `tuple` of `str`

        """
        cdef int number = len(names)
        if number is 0:
            return
        cdef int first = self.add_vertices(number)
        for vertex, name in enumerate(names, start=first):
            glpk.set_vertex_name(self._graph, vertex, name2chars(name))

    def set_vertex_name(self, vertex, str name):
        """Assign (change) vertex name

        :param name: the name of the vertex, it may not exceed 255 bytes
            *encoded as UTF-8*
        :type name: `str`

        """
        vertex = self.find_vertex_as_needed(vertex)
        glpk.set_vertex_name(self._graph, vertex, name2chars(name))

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
            for i, vertex in enumerate(vertices, start=1):
                inds[i] = self.find_vertex_as_needed(vertex)
            glpk.del_vertices(self._graph, n, inds)
        finally:
            glpk.free(inds)

    def del_arc(self, Arc arc):
        """Delete arc from graph"""
        cdef glpk.Arc* _arc = <glpk.Arc*>PyCapsule_GetPointer(arc._arc_ptr(),
                                                             NULL)
        glpk.del_arc(self._graph, _arc)

    def erase_graph(self, int vertex_data_size, int arc_data_size):
        """Erase graph content"""
        glpk.erase_graph(self._graph, vertex_data_size, arc_data_size)

    def find_vertex(self, str name):
        """Find vertex by its name

        :param name: the name of the vertex
        :type name: `str`

        """
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

    def get_vertex_object(self, vertex):
        """Return Vertex object"""
        cdef int ind = self.find_vertex_as_needed(vertex)
        if (ind > self.nv) or (ind < 1):
            raise ValueError("Incorrect vertex index.")
        cdef glpk.Vertex* _vertex = self._graph.v[ind]
        return Vertex(PyCapsule_New(_vertex, NULL, NULL))

    @classmethod
    def read_graph(cls, int vertex_data_size, int arc_data_size, str fname):
        """Read graph from plain text file"""
        graph = cls(vertex_data_size, arc_data_size)
        cdef glpk.Graph* _graph = <glpk.Graph*>PyCapsule_GetPointer(
                                                    graph._graph_ptr(), NULL)
        retcode = glpk.read_graph(_graph, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading graph from plain text file")
        return graph

    def write_graph(self, fname=None):
        """Write graph to plain text file"""
        fn = tempFile() if fname is None else fname
        retcode = glpk.write_graph(self._graph, str2chars(fn))
        if retcode is not 0:
            raise RuntimeError("Error writing graph plain text file")
        if fname is None:
            return popFile(fn)

    def mincost_lp(self, bint copy_names,
                   int v_rhs, int a_low, int a_cap, int a_cost):
        """Convert minimum cost flow problem to LP"""
        problem = Problem()
        cdef glpk.Prob* prob = <glpk.Prob*>PyCapsule_GetPointer(
                                                    problem._prob_ptr(), NULL)
        glpk.mincost_lp(prob, self._graph, copy_names,
                        v_rhs, a_low, a_cap, a_cost)
        return problem

    def mincost_okalg(self, int v_rhs, int a_low, int a_cap, int a_cost,
                      int a_x, int v_pi):
        """Find minimum-cost flow with out-of-kilter algorithm"""
        cdef double sol
        retcode = glpk.mincost_okalg(self._graph, v_rhs, a_low, a_cap, a_cost,
                                     &sol, a_x, v_pi)
        if retcode is not 0:
            raise ioretcode2error[retcode]
        return sol

    def mincost_relax4(self, int v_rhs, int a_low, int a_cap, int a_cost,
                       int crash, int a_x, int a_rc):
        """Find minimum-cost flow with Bertsekas-Tseng relaxation method"""
        cdef double sol
        retcode = glpk.mincost_relax4(self._graph, v_rhs, a_low, a_cap, a_cost,
                                      crash, &sol, a_x, a_rc)
        if retcode is not 0:
            raise ioretcode2error[retcode]
        return sol

    def maxflow_lp(self, bint copy_names, source, sink, int a_cap):
        """Convert maximum flow problem to LP"""
        source = self.find_vertex_as_needed(source)
        sink = self.find_vertex_as_needed(sink)
        problem = Problem()
        cdef glpk.Prob* prob = <glpk.Prob*>PyCapsule_GetPointer(
                                                    problem._prob_ptr(), NULL)
        glpk.maxflow_lp(prob, self._graph, copy_names, source, sink, a_cap)
        return problem

    def maxflow_ffalg(self, source, sink, int a_cap, int a_x, int v_cut):
        """Find maximal flow with Ford-Fulkerson algorithm"""
        source = self.find_vertex_as_needed(source)
        sink = self.find_vertex_as_needed(sink)
        cdef double sol
        retcode = glpk.maxflow_ffalg(self._graph, source, sink,
                                     a_cap, &sol, a_x, v_cut)
        return sol

    def check_asnprob(self, int v_set):
        """Check correctness of assignment problem data"""
        retval = glpk.check_asnprob(self._graph, v_set)
        if retval is not 0:
            raise ValueError("Assignment problem data is incorrect, code "
                             + retval + '.')

    def asnprob_lp(self, str asnform, bint copy_names, int v_set, int a_cost):
        """Convert assignment problem to LP"""
        problem = Problem()
        cdef glpk.Prob* prob = <glpk.Prob*>PyCapsule_GetPointer(
                                                    problem._prob_ptr(), NULL)
        retval = glpk.asnprob_lp(prob, str2asnform[asnform], self._graph,
                                 copy_names, v_set, a_cost)
        if retval is not 0:
            raise ValueError("Assignment problem data is incorrect, code "
                             + retval + '.')
        return problem

    def asnprob_okalg(self, str asnform, int v_set, int a_cost, int a_x):
        """Solve assignment problem with out-of-kilter algorithm"""
        cdef double sol
        retcode = glpk.asnprob_okalg(str2asnform[asnform], self._graph,
                                     v_set, a_cost, &sol, a_x)
        if retcode is not 0:
            raise ioretcode2error[retcode]
        return sol

    def asnprob_hall(self, int v_set, int a_x):
        """Find bipartite matching of maximum cardinality"""
        cardinality = glpk.asnprob_hall(self._graph, v_set, a_x)
        if cardinality < 0:
            raise ValueError("The specified graph is incorrect")
        return cardinality

    def cpp(self, int v_t, int v_es, int v_ls):
        """Solve critical path problem"""
        return glpk.cpp(self._graph, v_t, v_es, v_ls)

    @classmethod
    def read_mincost(cls, int vertex_data_size, int arc_data_size,
                     int v_rhs, int a_low, int a_cap, int a_cost, str fname):
        """Read min-cost flow problem data in DIMACS format"""
        graph = cls(vertex_data_size, arc_data_size)
        cdef glpk.Graph* _graph = <glpk.Graph*>PyCapsule_GetPointer(
                                                    graph._graph_ptr(), NULL)
        retcode = glpk.read_mincost(_graph, v_rhs, a_low, a_cap, a_cost,
                                    str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading min-cost flow problem DIMACS " +
                               "file")
        return graph

    def write_mincost(self, int v_rhs, int a_low, int a_cap, int a_cost,
                      fname=None):
        """Write min-cost flow problem data in DIMACS format"""
        fn = tempFile() if fname is None else fname
        retcode = glpk.write_mincost(self._graph, v_rhs, a_low, a_cap, a_cost,
                                     str2chars(fn))
        if retcode is not 0:
            raise RuntimeError("Error writing min-cost flow problem DIMACS " +
                               "file")
        if fname is None:
            return popFile(fn)

    @classmethod
    def read_maxflow(cls, int vertex_data_size, int arc_data_size,
                     int a_cap, str fname):
        """Read maximum flow problem data in DIMACS format"""
        graph = cls(vertex_data_size, arc_data_size)
        cdef glpk.Graph* _graph = <glpk.Graph*>PyCapsule_GetPointer(
                                                    graph._graph_ptr(), NULL)
        cdef int source
        cdef int sink
        retcode = glpk.read_maxflow(_graph, &source, &sink, a_cap,
                                    str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading maximum flow problem DIMACS " +
                               "file")
        return (graph, source, sink)

    def write_maxflow(self, source, sink, int a_cap, fname=None):
        """Write maximum flow problem data in DIMACS format"""
        source = self.find_vertex_as_needed(source)
        sink = self.find_vertex_as_needed(sink)
        fn = tempFile() if fname is None else fname
        retcode = glpk.write_maxflow(self._graph, source, sink, a_cap,
                                     str2chars(fn))
        if retcode is not 0:
            raise RuntimeError("Error writing maximum flow problem DIMACS " +
                               "file")
        if fname is None:
            return popFile(fn)

    @classmethod
    def read_asnprob(cls, int vertex_data_size, int arc_data_size,
                     int v_set, int a_cost, str fname):
        """Read assignment problem data in DIMACS format"""
        graph = cls(vertex_data_size, arc_data_size)
        cdef glpk.Graph* _graph = <glpk.Graph*>PyCapsule_GetPointer(
                                                    graph._graph_ptr(), NULL)
        retcode = glpk.read_asnprob(_graph, v_set, a_cost, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error assignment problem DIMACS file")
        return graph

    def write_asnprob(self, int v_set, int a_cost, fname=None):
        """Write assignment problem data in DIMACS format"""
        fn = tempFile() if fname is None else fname
        retcode = glpk.write_asnprob(self._graph, v_set, a_cost,
                                     str2chars(fn))
        if retcode is not 0:
            raise RuntimeError("Error writing assignment problem DIMACS file")
        if fname is None:
            return popFile(fn)

    @classmethod
    def read_ccdata(cls, int vertex_data_size, int arc_data_size,
                    int v_wgt, str fname):
        """Read graph in DIMACS clique/coloring format"""
        graph = cls(vertex_data_size, arc_data_size)
        cdef glpk.Graph* _graph = <glpk.Graph*>PyCapsule_GetPointer(
                                                    graph._graph_ptr(), NULL)
        retcode = glpk.read_ccdata(_graph, v_wgt, str2chars(fname))
        if retcode is not 0:
            raise RuntimeError("Error reading graph DIMACS clique/coloring " +
                               "file")
        return graph

    def write_ccdata(self, int v_wgt, fname=None):
        """Write graph in DIMACS clique/coloring format"""
        fn = tempFile() if fname is None else fname
        retcode = glpk.write_ccdata(self._graph, v_wgt, str2chars(fn))
        if retcode is not 0:
            raise RuntimeError("Error writing graph DIMACS clique/coloring " +
                               "file")
        if fname is None:
            return popFile(fn)

    @classmethod
    def netgen(cls, int vertex_data_size, int arc_data_size,
               int v_rhs, int a_cap, int a_cost, **parameters):
        """Klingman's network problem generator"""
        graph = cls(vertex_data_size, arc_data_size)
        cdef glpk.Graph* _graph = <glpk.Graph*>PyCapsule_GetPointer(
                                                    graph._graph_ptr(), NULL)
        cdef int params[1+15]
        cdef int nprob = parameters['nprob']  # 8-digit problem id number
        if (nprob > 100) and (nprob <= 150):
            glpk.netgen_prob(nprob, params)
        else:
            params[2] = nprob
        for parameter, value in parameters.items():
            if parameter is 'iseed':
                params[1] = value  # 8-digit positive random number seed
            elif parameter is 'nodes':
                params[3] = value  # total number of nodes
            elif parameter is 'nsorc':
                params[4] = value
                # total number of source nodes (including transshipment nodes)
            elif parameter is 'nsink':
                params[5] = value
                # total number of sink nodes (including transshipment nodes)
            elif parameter is 'iarcs':
                params[6] = value  # number of arcs
            elif parameter is 'mincst':
                params[7] = value  # minimum cost for arcs
            elif parameter is 'maxcst':
                params[8] = value  # maximum cost for arcs
            elif parameter is 'itsup':
                params[9] = value  # total supply
            elif parameter is 'ntsorc':
                params[10] = value  # number of transshipment source nodes
            elif parameter is 'ntsink':
                params[11] = value  # number of transshipment sink nodes
            elif parameter is 'iphic':
                params[12] = value
                # percentage of skeleton arcs to be given the maximum cost
            elif parameter is 'ipcap':
                params[13] = value  # percentage of arcs to be capacitated
            elif parameter is 'mincap':
                params[14] = value  # minimum upper bound for capacitated arcs
            elif parameter is 'maxcap':
                params[15] = value  # maximum upper bound for capacitated arcs
        retval = glpk.netgen(_graph, v_rhs, a_cap, a_cost, params)
        if retval is not 0:
            raise ValueError("Network generator parameters are inconsistent")
        return graph

    @classmethod
    def gridgen(cls, int vertex_data_size, int arc_data_size,
                int v_rhs, int a_cap, int a_cost, **parameters):
        """Grid-like network problem generator"""
        graph = cls(vertex_data_size, arc_data_size)
        cdef glpk.Graph* _graph = <glpk.Graph*>PyCapsule_GetPointer(
                                                    graph._graph_ptr(), NULL)
        cdef int params[1+14]
        for parameter, value in parameters.items():
            if parameter is 'two_way':
                params[1] = value
                # two-ways arcs indicator
                #  1 — if links in both direction should be generated
                #  0 — otherwise
            if parameter is 'iseed':
                params[2] = value  # random number seed (a positive integer)
            if parameter is 'nodes':
                params[3] = value
                # number of nodes (the number of nodes generated might be
                # slightly different to make the network a grid)
            if parameter is 'width':
                params[4] = value  # grid width
            if parameter is 'sources':
                params[5] = value  # number of sources
            if parameter is 'sinks':
                params[6] = value  # number of sinks
            if parameter is 'degree':
                params[7] = value  # average degree
            if parameter is 'flow':
                params[8] = value  # total flow
            if parameter is 'dist_costs':
                params[9] = value
                # distribution of arc costs: 1 — uniform, 2 — exponential
            if parameter is 'cost_lb':
                params[10] = value
                # lower bound for arc cost (uniform), 100λ (exponential)
            if parameter is 'cost_ub':
                params[11] = value
                # upper bound for arc cost (uniform), not used (exponential)
            if parameter is 'dist_caps':
                params[12] = value
                # distribution of arc capacities: 1 — uniform, 2 — exponential
            if parameter is 'caps_lb':
                params[13] = value
                # lower bound for arc capacity (uniform), 100λ (exponential)
            if parameter is 'caps_ub':
                params[14] = value
                # upper bound for arc capacity (uniform), not used (exponential)
        retval = glpk.gridgen(_graph, v_rhs, a_cap, a_cost, params)
        if retval is not 0:
            raise ValueError("Network generator parameters are inconsistent")
        return graph

    @classmethod
    def rmfgen(cls, int vertex_data_size, int arc_data_size,
               int a_cap, parameters):
        """Goldfarb's maximum flow problem generator"""
        graph = cls(vertex_data_size, arc_data_size)
        cdef glpk.Graph* _graph = <glpk.Graph*>PyCapsule_GetPointer(
                                                    graph._graph_ptr(), NULL)
        cdef int params[1+5]
        for parameter, value in parameters.items():
            if parameter is 'seed':
                params[1] = value  # random number seed (a positive integer)
            if parameter is 'a':
                params[2] = value  # frame size
            if parameter is 'b':
                params[3] = value  # depth
            if parameter is 'c1':
                params[4] = value  # minimal arc capacity
            if parameter is 'c2':
                params[5] = value  # maximal arc capacity
        cdef int source
        cdef int sink
        retval = glpk.rmfgen(_graph, &source, &sink, a_cap, params)
        if retval is not 0:
            raise ValueError("Network generator parameters are inconsistent")
        return (graph, source, sink)

    def weak_comp(self, int v_num):
        """Find all weakly connected components of graph"""
        return glpk.weak_comp(self._graph, v_num)

    def strong_comp(self, int v_num):
        """Find all strongly connected components of graph"""
        return glpk.strong_comp(self._graph, v_num)

    def top_sort(self, int v_num):
        """Topological sorting of acyclic digraph"""
        return glpk.top_sort(self._graph, v_num)

    def wclique_exact(self, int v_wgt, int v_set):
        """Find maximum weight clique with exact algorithm"""
        cdef double sol
        retcode = glpk.wclique_exact(self._graph, v_wgt, &sol, v_set)
        if retcode is not 0:
            raise ioretcode2error[retcode]
        return sol
