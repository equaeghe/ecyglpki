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


cdef class _Varstraints(_Component)

    cdef list _varstraints

    def __cinit__(self, program):
        self._varstraints = []

    def __len__(self):
        return len(self._varstraints)

    def __iter__(self):
        return self._varstraints.__iter__()

    def __contains__(self, varstraint):
        return varstraint._ind is not None

    def _getoneitem(self, name):
        if not isinstance(name, str):
            raise TypeError("Names are strings; "
                            + str(name) + " is " + type(name).__name__ + '.')
        ind = self._find_ind(name)
        if ind is None:
            raise KeyError("Unknown name: " + name)
        return self._varstraints[ind-1]  # GLPK indices start at 1

    def __getitem__(self, names*):
        n = len(names)
        if n is 0:
            raise SyntaxError("At least one name argument is required")
        elif n is 1:
            return self._getoneitem(self, names[0])
        else:
            for name in names:
                yield self._getoneitem(self, name)

    def __delitem__(self, names*):
        nameinds = {name: self[name]._ind for name in names
                                          if self[name]._ind is not None}
        numinds = len(nameinds)
        cdef int inds[1+numinds]
        for ind, nameind in enumerate(nameinds.items(), start=1):
                                      # GLPK indices start at 1
            inds[ind] = nameind[1]
            del self._varstraints[nameind[0]]
        self._del_cols(numinds, inds)

    def _add(self, varstraint, attributes):
        self._varstraints.append(varstraint)
        for attribute, value in attributes:
            setattr(varstraint, attribute, value)


cdef class Variables(_Varstraints):

    cdef _find_ind(self, name):
        return glpk.find_col(self._problem, name2chars(name))

    cdef _del_inds(self, int numinds, const char* inds):
        glpk.del_cols(self._problem, numinds, inds)

    def add(self, attributes**):
        """Add a new variable to the problem

        :param attributes: zero or more named parameters from the list of
            `.Variable` attributes
        :returns: the new variable
        :rtype: `.Variable`

        """
        variable = Variable()
        self._add(variable, attributes)
        return variable


cdef class Constraints(_Varstraints):

    cdef _find_ind(self, name):
        return glpk.find_row(self._problem, name2chars(name))

    cdef _del_inds(self, int numinds, const char* inds):
        glpk.del_rows(self._problem, numinds, inds)

    def add(self, attributes**):
        """Add a new constraint to the problem

        :param attributes: zero or more named parameters from the list of
            `.Constraint` attributes
        :returns: the new constraint
        :rtype: `.Constraint`

        """
        constraint = Constraint()
        self._add(constraint, attributes)
        return constraint
