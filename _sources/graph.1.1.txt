.. module:: ecyglpki

.. testsetup:: *

    from ecyglpki import Graph, Vertex, Arc

Graph program object
--------------------

.. autoclass:: Graph
    :members: nv, na, v_size, a_size

.. autoclass:: Vertex
    :members: i, name, inc, out

.. autoclass:: Arc
    :members: tail, head, t_prev, t_next, h_prev, h_next
