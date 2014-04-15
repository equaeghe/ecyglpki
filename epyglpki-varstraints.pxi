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

    def __getitem__(self, name):
        col = glpk.find_col(self._problem, name2chars(name))
        return None if col is None else self._variables[col-1]
                                        # GLPK indices start at 1

    def __delitem__(self, name):
        self._variables[name].remove()

    def add(self, attributes**):
        variable = Variable()
        self._variables.append(variable)
        for attribute, value in attributes:
            setattr(variable, attribute, value)
