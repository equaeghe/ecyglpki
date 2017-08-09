.. module:: ecyglpki

.. testsetup:: *

    from ecyglpki import Problem, IntOptControls

Mixed integer programming routines
==================================

.. automethod:: Problem.set_col_kind

.. automethod:: Problem.get_col_kind

.. automethod:: Problem.get_num_int

.. automethod:: Problem.get_num_bin

.. automethod:: Problem.intopt

.. automethod:: Problem.mip_status

.. automethod:: Problem.mip_obj_val

.. automethod:: Problem.mip_row_val

.. automethod:: Problem.mip_col_val

Integer optimization controls
-----------------------------

.. autoclass:: IntOptControls
    :members:
    :undoc-members:
