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

    def _getoneitem(self, arg):
        if isinstance(name, numbers.Integral):
            ind = arg
        if isinstance(name, str):
            ind = self._find_ind(arg)
            if ind is None:
                raise KeyError("Unknown name: " + arg)
        else:
            raise TypeError("Arguments are are integers or strings; "
                            + str(arg) + " is " + type(arg).__name__ + '.')
        return self._varstraints[ind-1]  # GLPK indices start at 1

    def __getitem__(self, args*):
        n = len(args)
        if n is 0:
            raise SyntaxError("At least one argument is required")
        elif n is 1:
            return self._getoneitem(self, args[0])
        else:
            for arg in args:
                yield self._getoneitem(self, arg)

    def __delitem__(self, args*):
        arginds = {arg: self[arg]._ind for arg in args
                                       if self[arg]._ind is not None}
        numinds = len(arginds)
        cdef int inds[1+numinds]
        for ind, argind in enumerate(arginds.items(), start=1):
                                     # GLPK indices start at 1
            inds[ind] = argind[1]
            del self._varstraints[argind[0]]
        self._del_cols(numinds, inds)

    def _add(self, varstraint, attributes):
        self._varstraints.append(varstraint)
        for attribute, value in attributes:
            setattr(varstraint, attribute, value)


cdef class Variables(_Varstraints):

    cdef _find_ind(self, name):
        return glpk.find_col(self._problem, name2chars(name))

    cdef _del_inds(self, int numinds, const int* inds):
        glpk.del_cols(self._problem, numinds, inds)

    def _link(self):
        variable = Variable()
        self._add(variable, {})

    def add(self, attributes**):
        """Add a new variable to the problem

        :param attributes: zero or more named parameters from the list of
            `.Variable` attributes
        :returns: the new variable
        :rtype: `.Variable`

        """
        glpk.add_cols(self._problem, 1)
        variable = Variable()
        self._add(variable, attributes)
        return variable


cdef class Constraints(_Varstraints):

    cdef _find_ind(self, name):
        return glpk.find_row(self._problem, name2chars(name))

    cdef _del_inds(self, int numinds, const int* inds):
        glpk.del_rows(self._problem, numinds, inds)

    def _link(self):
        constraint = Constraint()
        self._add(constraint, {})

    def add(self, attributes**):
        """Add a new constraint to the problem

        :param attributes: zero or more named parameters from the list of
            `.Constraint` attributes
        :returns: the new constraint
        :rtype: `.Constraint`

        """
        glpk.add_rows(self._problem, 1)
        constraint = Constraint()
        self._add(constraint, attributes)
        return constraint
