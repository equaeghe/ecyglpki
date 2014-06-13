# ecyglpki-iocp.pxi: Cython interface for GLPK integer optimization solver controls

###############################################################################
#
#  This code is part of ecyglpki (a Cython GLPK interface).
#
#  Copyright (C) 2014 Erik Quaeghebeur. All rights reserved.
#
#  ecyglpki is free software: you can redistribute it and/or modify it under
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


cdef class CallbackFunction:
    """The callback function wrapper"""

    cdef glpk.CBFunc _cb_func

    def __cinit__(self, cbfunc_ptr):
        self._cb_func = <glpk.CBFunc>PyCapsule_GetPointer(cbfunc_ptr(), NULL)


cdef class CallbackInfo:
    """The callback function info reference"""

    cdef glpk.CBInfo _cb_info

    def __cinit__(self, cbinfo_ptr):
        self._cb_info = <glpk.CBInfo>PyCapsule_GetPointer(cbinfo_ptr(), NULL)



# branching technique
cdef str2brtech = {
    'first_fracvar': glpk.BR_FFV,
    'last_fracvar': glpk.BR_LFV,
    'most_fracvar': glpk.BR_MFV,
    'Driebeek-Tomlin': glpk.BR_DTH,
    'hybrid_peudocost': glpk.BR_PCH
    }
cdef brtech2str = {br_tech: string for string, br_tech in str2brtech.items()}

# backtracking technique
cdef str2bttech = {
    'depth': glpk.BT_DFS,
    'breadth': glpk.BT_BFS,
    'bound': glpk.BT_BLB,
    'projection': glpk.BT_BPH
    }
cdef bttech2str = {bt_tech: string for string, bt_tech in str2bttech.items()}

# preprocessing technique
cdef str2pptech = {
    'none': glpk.PP_NONE,
    'root': glpk.PP_ROOT,
    'all': glpk.PP_ALL
    }
cdef pptech2str = {pp_tech: string for string, pp_tech in str2pptech.items()}


cdef class IntOptControls:
    """The integer optimization solver control parameter object

    .. doctest:: IntOptControls

        >>> r = IntOptControls()

    """

    cdef glpk.IntOptCP _iocp

    def __cinit__(self):
        glpk.init_iocp(&self._iocp)

    property msg_lev:
        """The message level, a `str`

        The possible values are

        * `'no'`: no output
        * `'warnerror'`: warnings and errors only
        * `'normal'`: normal output
        * `'full'`: normal output and informational messages

        .. doctest:: IntOptControls

            >>> r.msg_lev  # the GLPK default
            'full'
            >>> r.msg_lev = 'no'
            >>> r.msg_lev
            'no'

        """
        def __get__(self):
            return msglev2str[self._iocp.msg_lev]
        def __set__(self, value):
            self._iocp.msg_lev = str2msglev[value]

    property out_frq:
        """Output frequency [ms] of informational messages, an `int`"""
        def __get__(self):
            return self._iocp.out_frq
        def __set__(self, value):
            self._iocp.out_frq = int(value)

    property out_dly:
        """Output delay [ms] of current LP relaxation solution, an `int`"""
        def __get__(self):
            return self._iocp.out_dly
        def __set__(self, value):
            self._iocp.out_dly = int(value)

    property cb_func:
        """Callback routine"""
        def __get__(self):
            cdef glpk.CBFunc callback = self._iocp.cb_func
            if callback is not NULL:
                return CallbackFunction(PyCapsule_New(callback, NULL, NULL))
            else:
                return None
        def __set__(self, CallbackFunction callback):
            self._iocp.cb_func = callback._cb_func

    property cb_info:  # TODO: use Python object for node data?
        """Transit pointer passed to the routine cb_func"""
        def __get__(self):
            cdef glpk.CBInfo info = self._iocp.cb_info
            if info is not NULL:
                return CallbackInfo(PyCapsule_New(info, NULL, NULL))
            else:
                return None
        def __set__(self, CallbackInfo info):
            self._iocp.cb_info = info._cb_info

    property cb_size:  # TODO: use Python object for node data?
        """Number of extra bytes allocated for each search tree node, an `int`

        Up to 256 bytes can be allocated for each node of the branch-and-bound
        tree to store application-specific data. On creating a node these bytes
        are initialized by binary zeros.

        .. doctest:: IntOptControls

            >>> r.cb_size  # the GLPK default
            0
            >>> r.cb_size = 128
            >>> r.cb_size
            128

        """
        def __get__(self):
            return self._iocp.cb_size
        def __set__(self, value):
            value = int(value)
            if (value < 0) or (value > 256):
                raise ValueError("'cb_size' must be an int between 0 and 256.")
            self._iocp.cb_size = value

    property tm_lim:
        """Time limit [ms], an `int`"""
        def __get__(self):
            return self._iocp.tm_lim
        def __set__(self, value):
            self._iocp.tm_lim = int(value)

    property br_tech:
        """The branching technique, a `str`

        The possible values are

        * `'first_fracvar'`: first fractional variable
        * `'last_fracvar'`: last fractional variable
        * `'most_fracvar'`: most fractional variable
        * `'Driebeek-Tomlin'`: heuristic by Driebeek_ & Tomlin
        * `'hybrid_peudocost'`: hybrid pseudocost heuristic

        .. _Driebeek: http://dx.doi.org/10.1287/ijoc.4.3.267

        """
        def __get__(self):
            return brtech2str[self._iocp.br_tech]
        def __set__(self, value):
            self._iocp.br_tech = str2brtech[value]

    property bt_tech:
        """The backtracking technique, a `str`

        The possible values are

        * `'depth'`: depth first search
        * `'breadth'`: breadth first search
        * `'bound'`: best local bound
        * `'projection'`: best projection heuristic

        """
        def __get__(self):
            return bttech2str[self._iocp.bt_tech]
        def __set__(self, value):
            self._iocp.bt_tech = str2bttech[value]

    property pp_tech:
        """The preprocessing technique, a `str`

        The possible values are

        * `'none'`: disable preprocessing
        * `'root'`: preprocessing only on the root level
        * `'all'`: preprocessing on all levels

        """
        def __get__(self):
            return pptech2str[self._iocp.pp_tech]
        def __set__(self, value):
            self._iocp.pp_tech = str2pptech[value]

    property fp_heur:
        """Whether to apply the `feasibility pump`_ heuristic, a `bool`

        .. _feasibility pump: http://dx.doi.org/10.1007/s10107-004-0570-3

        """
        def __get__(self):
            return self._iocp.fp_heur
        def __set__(self, value):
            self._iocp.fp_heur = bool(value)

    property ps_heur:
        """Whether to apply the `proximity search`_ heuristic, a `bool`

        .. _proximity search: http://www.dei.unipd.it/~fisch/papers/proximity_search.pdf

        """
        def __get__(self):
            return self._iocp.ps_heur
        def __set__(self, value):
            self._iocp.ps_heur = bool(value)

    property ps_tm_lim:
        """Time limit [ms] for the proximity earch heuristic, an `int`"""
        def __get__(self):
            return self._iocp.ps_tm_lim
        def __set__(self, value):
            self._iocp.ps_tm_lim = int(value)

    property gmi_cuts:
        """Whether to generate Gomory’s mixed integer cuts, a `bool`"""
        def __get__(self):
            return self._iocp.gmi_cuts
        def __set__(self, value):
            self._iocp.gmi_cuts = bool(value)

    property mir_cuts:
        """Whether to generate mixed integer rounding cuts, a `bool`"""
        def __get__(self):
            return self._iocp.mir_cuts
        def __set__(self, value):
            self._iocp.mir_cuts = bool(value)

    property cov_cuts:
        """Whether to generate mixed cover cuts, a `bool`"""
        def __get__(self):
            return self._iocp.cov_cuts
        def __set__(self, value):
            self._iocp.cov_cuts = bool(value)

    property clq_cuts:
        """Whether to generate generate clique cuts, a `bool`"""
        def __get__(self):
            return self._iocp.clq_cuts
        def __set__(self, value):
            self._iocp.clq_cuts = bool(value)

    property tol_int:
        """Abs. tolerance for LP solution integer feasibility, a |Real| number

        This is the absolute tolerance used to check if the optimal solution to
        the current LP relaxation is integer feasible.

        """
        def __get__(self):
            return self._iocp.tol_int
        def __set__(self, value):
            self._iocp.tol_int = float(value)

    property tol_obj:
        """Rel. tolerance of LP objective optimality, a |Real| number

        This is the relative tolerance used to check if the objective value in
        the optimal solution to the current LP relaxation is not better than in
        the best known integer feasible solution.

        """
        def __get__(self):
            return self._iocp.tol_obj
        def __set__(self, value):
            self._iocp.tol_obj = float(value)

    property mip_gap:
        """The relative MIP-gap tolerance, a |Real| number

        The search stops once the relative MIP-gap falls below this value.

        """
        def __get__(self):
            return self._iocp.mip_gap
        def __set__(self, value):
            self._iocp.mip_gap = float(value)

    property presolve:
        """Whether to use the MIP presolver, a `bool`

        Using the MIP presolver may simplify the problem

        """
        def __get__(self):
            return self._iocp.presolve
        def __set__(self, value):
            self._iocp.presolve = bool(value)

    property binarize:
        """Whether to binarize integer variables, a `bool`

        This option is only used if *presolve* is `True`.

        """
        def __get__(self):
            return self._iocp.binarize
        def __set__(self, value):
            self._iocp.binarize = bool(value)
