# epyglpki-solvers.pxi: Cython/Python interface for GLPK variable/constraint lists

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


cdef class _Varstraints(_Component):

    cdef list _varstraints

    def __cinit__(self, program):
        self._varstraints = []

    def __len__(self):
        return len(self._varstraints)

    def __iter__(self):
        return self._varstraints.__iter__()

    def __contains__(self, varstraint):
        try:
            varstraint._ind
        except IndexError:
            return False
        else:
            return True

    def __getitem__(self, name):
        if isinstance(name, str):
            ind = self._find_ind(name)
        else:
            raise TypeError("Name must be a string; "
                            + str(name) + " is " + type(name).__name__ + '.')
        return self._varstraints[ind-1]  # GLPK indices start at 1

    def _from_ind(self, ind):
        if isinstance(ind, numbers.Integral):
            return self._varstraints[ind-1]  # GLPK indices start at 1
        else:
            raise TypeError("Indices must be Integral.")

    def __delitem__(self, names):
        inds = [self[name]._ind for name in names]
        for ind in inds:
            del self._varstraints[ind-1]  # GLPK indices start at 1
        self._del_inds(inds)

    def _add(self, varstraint, attributes):
        self._varstraints.append(varstraint)
        for attribute, value in attributes.items():
            setattr(varstraint, attribute, value)


cdef class Variables(_Varstraints):
    """The problem's collection of variables

    .. doctest:: Variables

        >>> p = MILProgram()
        >>> vs = p.variables
        >>> x = vs.add(name='ξ')
        >>> y = vs.add(name='υ')
        >>> len(vs)
        2
        >>> {vs['ξ'], vs['υ']} == {z for z in vs} == {x, y}
        True
        >>> del vs['υ']
        >>> len(vs)
        1

    """

    def _find_ind(self, name):
        return glpk.find_col(self._problem, name2chars(name))

    def _del_inds(self, inds):
        k = len(inds)
        cdef int* cinds =  <int*>glpk.alloc(1+k, sizeof(int))
        for i, ind in enumerate(inds, start=1):
            cinds[i] = ind
        glpk.del_cols(self._problem, k, cinds)

    def _link(self):
        variable = Variable(self._program)
        self._add(variable, {})

    def add(self, **attributes):
        """Add a new variable to the problem

        :param attributes: zero or more named parameters from the list of
            writable `.Variable` attributes
        :returns: the new variable
        :rtype: `.Variable`

        .. doctest:: Variables

            >>> x = vs.add(name='Veranderlijke', kind='binary', weight=3)
            >>> x.name
            'Veranderlijke'
            >>> x.kind
            'binary'
            >>> x.weight
            3.0

        """
        glpk.add_cols(self._problem, 1)
        variable = Variable(self._program)
        self._add(variable, attributes)
        return variable


cdef class Constraints(_Varstraints):
    """The problem's collection of constraints

    .. doctest:: Constraints

        >>> p = MILProgram()
        >>> cs = p.constraints
        >>> c = cs.add(name='δ')
        >>> d = cs.add(name='ε')
        >>> len(cs)
        2
        >>> {cs['δ'], cs['ε']} == {e for e in cs} == {c, d}
        True
        >>> del cs['ε']
        >>> len(cs)
        1

    """

    def _find_ind(self, name):
        return glpk.find_row(self._problem, name2chars(name))

    def _del_inds(self, inds):
        k = len(inds)
        cdef int* cinds =  <int*>glpk.alloc(1+k, sizeof(int))
        for i, ind in enumerate(inds, start=1):
            cinds[i] = ind
        glpk.del_rows(self._problem, k, cinds)

    def _link(self):
        constraint = Constraint(self._program)
        self._add(constraint, {})

    def add(self, **attributes):
        """Add a new constraint to the problem

        :param attributes: zero or more named parameters from the list of
            writable `.Constraint` attributes
        :returns: the new constraint
        :rtype: `.Constraint`

        .. doctest:: Constraints

            >>> c = cs.add(name='Capacity', scale=.5)
            >>> c.name
            'Capacity'
            >>> c.scale
            0.5

        """
        glpk.add_rows(self._problem, 1)
        constraint = Constraint(self._program)
        self._add(constraint, attributes)
        return constraint
