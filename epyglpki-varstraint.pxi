# epyglpki-solvers.pxi: Cython/Python interface for GLPK variables/constraints

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


cdef class Variable(_Component):

    cdef str _alias  # unique identifier; invariant after initialization
    cdef str _name
        #  always identical to glpk.get_col_name(self._problem, self._col)

    def __cinit__(self, program):
        glpk.add_cols(self._problem, 1)
        self._name = self._alias = self._program._generate_alias()
        col = glpk.get_col_num(self._problem)
        glpk.set_col_name(self._problem, col, name2chars(self._name))

    def __hash__(self):
        return hash(self._alias)

    property _col:
        """Return the column index"""
        def __get__(self):
            try:
                return glpk.find_col(self._problem, name2chars(self._name))
            except ValueError:
                raise IndexError("This is possibly a zombie; " +
                                 "kill it using 'del'.")

    property _ind:
        """Return the variable index

        (GLPK sometimes indexes variables after constraints.)

        """
        def __get__(self):
            return glpk.get_row_num(self._problem) + self._col

    property name:
        """The variable name, a `str` of ≤255 bytes UTF-8 encoded

        .. doctest:: Variable

            >>> x.name  # the GLPK default
            ''
            >>> x.name = 'Stake'
            >>> x.name
            'Stake'
            >>> del x.name  # clear name
            >>> x.name
            ''

        """
        def __get__(self):
            return '' if self._name is self._alias else self._name
        def __set__(self, name):
            self._name = self._alias if name is '' else name
            glpk.set_col_name(self._problem, self._col, name2chars(self._name))
        def __del__(self):
            self.name = self._alias

    property kind:
        """The variable kind, either `'continuous'`, `'integer'`, or `'binary'`

        .. doctest:: Variable

            >>> x.kind  # the GLPK default
            'continuous'
            >>> x.kind = 'integer'
            >>> x.kind
            'integer'

        .. note::

            A variable has `'binary'` kind if and only if it is an integer
            variable with lower bound zero and upper bound one:

            .. doctest:: Variable

                >>> x.kind
                'integer'
                >>> x.bounds(lower=0, upper=1)
                (0.0, 1.0)
                >>> x.kind
                'binary'
                >>> x.bounds(upper=3)
                (0.0, 3.0)
                >>> x.kind
                'integer'
                >>> x.kind = 'binary'
                >>> x.bounds()
                (0.0, 1.0)

        """
        def __get__(self):
            return varkind2str[glpk.get_col_kind(self._problem, self._col)]
        def __set__(self, kind):
            if kind in str2varkind:
                glpk.set_col_kind(self._problem, self._col, str2varkind[kind])
            else:
                raise ValueError("Kind must be 'continuous', 'integer', " +
                                 "or 'binary'.")

