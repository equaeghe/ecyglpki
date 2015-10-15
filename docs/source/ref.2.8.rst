.. module:: ecyglpki

.. testsetup:: *

    from ecyglpki import Problem, SimplexControls

Simplex method routines
=======================

.. automethod:: Problem.simplex

.. automethod:: Problem.exact

.. automethod:: Problem.get_status

.. automethod:: Problem.get_prim_stat

.. automethod:: Problem.get_dual_stat

.. automethod:: Problem.get_obj_val

.. automethod:: Problem.get_row_stat

.. automethod:: Problem.get_row_prim

.. automethod:: Problem.get_row_dual

.. automethod:: Problem.get_col_stat

.. automethod:: Problem.get_col_prim

.. automethod:: Problem.get_col_dual

.. automethod:: Problem.get_unbnd_ray

Simplex controls
----------------

.. autoclass:: SimplexControls
    :members:
    :undoc-members:
