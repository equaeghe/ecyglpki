# epyglpki-solvers.pxi: Cython/Python interface for GLPK objectives

###############################################################################
#
#  This code is part of epyglpki (a Cython/Python GLPK interface).
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


cdef class Objective(_Component):
    """The problem's objective

    .. doctest:: Objective

        >>> p = MILProgram()
        >>> o = p.objective
        >>> isinstance(o, Objective)
        True

    """

    property direction:
        """The objective direction, either `'minimize'` or `'maximize'`

        .. doctest:: Objective

            >>> o.direction  # the GLPK default
            'minimize'
            >>> o.direction = 'maximize'
            >>> o.direction
            'maximize'

        """
        def __get__(self):
            return optdir2str[glpk.get_obj_dir(self._problem)]
        def __set__(self, direction):
            if direction in str2optdir:
                glpk.set_obj_dir(self._problem, str2optdir[direction])
            else:
                raise ValueError("Direction must be 'minimize' or 'maximize'.")

    property constant:
        """The objective function constant, a |Real| number

        .. doctest:: Objective

            >>> o.constant  # the GLPK default
            0.0
            >>> o.constant = 3
            >>> o.constant
            3.0
            >>> del o.constant
            >>> o.constant
            0.0

        """
        def __get__(self):
            return glpk.get_obj_coef(self._problem, 0)
        def __set__(self, constant):
            if isinstance(constant, numbers.Real):
                glpk.set_obj_coef(self._problem, 0, constant)
            else:
                raise TypeError("Objective constant must be a real number.")
        def __del__(self):
            glpk.set_obj_coef(self._problem, 0, 0.0)

    property name:
        """The objective function name, a `str` of ≤255 bytes UTF-8 encoded

        .. doctest:: Objective

            >>> o.name  # the GLPK default
            ''
            >>> o.name = 'σκοπός'
            >>> o.name
            'σκοπός'
            >>> del o.name  # clear name
            >>> o.name
            ''

        """
        def __get__(self):
            cdef const char* chars = glpk.get_obj_name(self._problem)
            return '' if chars is NULL else chars.decode()
        def __set__(self, name):
            glpk.set_obj_name(self._problem, name2chars(name))
        def __del__(self):
            glpk.set_obj_name(self._problem, NULL)

    property simplex:
        """The objective value produced by the simplex solver, a |Real| number

        .. doctest:: Objective

            >>> o.simplex
            0.0

        """
        def __get__(self):
            return glpk.sm_obj_val(self._problem)

    property ipoint:
        """The objective value produced by the interior point solver, a |Real| number

        .. doctest:: Objective

            >>> o.ipoint
            0.0

        """
        def __get__(self):
            return glpk.ipt_obj_val(self._problem)

    property intopt:
        """The objective value produced by the integer optimization solver, a |Real| number

        .. doctest:: Objective

            >>> o.intopt
            0.0

        """
        def __get__(self):
            return glpk.mip_obj_val(self._problem)
