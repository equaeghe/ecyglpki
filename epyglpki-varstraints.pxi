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


cdef class Variables(_Component):

    cdef list _variables

    def __cinit__(self, program):
        self._variables = []

    def __len__(self):
        return len(self._variables)

    def __iter__(self):
        return self._variables.__iter__()

    def __contains__(self, variable):
        return variable._col is not None

    def _getoneitem(self, name):
        if not isinstance(name, str):
            raise TypeError("Variable names are strings; "
                            + str(name) + " is " + type(name).__name__ + '.')
        col = glpk.find_col(self._problem, name2chars(name))
        if col is None:
            raise KeyError("Unknown Variable name: " + name)
        return self._variables[col-1]  # GLPK indices start at 1

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
        namecols = {name: self[name]._col for name in names
                                          if self[name]._col is not None}
        numcols = len(namecols)
        cdef int inds[1+numcols]
        for ind, namecol in enumerate(namecols.items(), start=1):
                                      # GLPK indices start at 1
            inds[ind] = namecol[1]
            del self._variables[namecol[0]]
        glpk.del_cols(self._problem, numcols, inds)

    def add(self, attributes**):
        """Add a new variable to the problem

        :param attributes: zero or more named parameters from the list of
            `.Variable` attributes
        :returns: the new variable
        :rtype: `.Variable`

        """
        variable = Variable()
        self._variables.append(variable)
        for attribute, value in attributes:
            setattr(variable, attribute, value)
        return variable


cdef class Constraints(_Component):

    cdef list _constraints

    def __cinit__(self, program):
        self._constraints = []

    def __len__(self):
        return len(self._constraints)

    def __iter__(self):
        return self._constraints.__iter__()

    def __contains__(self, constraint):
        return constraint._row is not None

    def _getoneitem(self, name):
        if not isinstance(name, str):
            raise TypeError("Constraint names are strings; "
                            + str(name) + " is " + type(name).__name__ + '.')
        row = glpk.find_row(self._problem, name2chars(name))
        if row is None:
            raise KeyError("Unknown Constraint name: " + name)
        return self._constraints[row-1]  # GLPK indices start at 1

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
        namerows = {name: self[name]._row for name in names
                                          if self[name]._row is not None}
        numrows = len(namerows)
        cdef int inds[1+numrows]
        for ind, namerow in enumerate(namerows.items(), start=1):
                                      # GLPK indices start at 1
            inds[ind] = namerow[1]
            del self._constraints[namerow[0]]
        glpk.del_rows(self._problem, numrows, inds)

    def add(self, attributes**):
        """Add a new constraint to the problem

        :param attributes: zero or more named parameters from the list of
            `.Constraint` attributes
        :returns: the new constraint
        :rtype: `.Constraint`

        """
        constraint = Constraint()
        self._constraints.append(constraint)
        for attribute, value in attributes:
            setattr(constraint, attribute, value)
        return constraint
