.. module:: ecyglpki

.. testsetup:: *

    from ecyglpki import Problem, FactorizationControls

LP basis routines
-----------------

.. automethod:: Problem.bf_exists

.. automethod:: Problem.factorize

.. automethod:: Problem.bf_updated

.. automethod:: Problem.get_bfcp

.. automethod:: Problem.set_bfcp

.. automethod:: Problem.get_bhead

.. automethod:: Problem.get_row_bind

.. automethod:: Problem.get_col_bind

.. automethod:: Problem.ftran

.. automethod:: Problem.btran

.. automethod:: Problem.warm_up


Basis factorization controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FactorizationControls
    :members:
    :undoc-members:
