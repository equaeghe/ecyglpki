import ecyglpki
import collections

pspace = ('W', 'D', 'L')
"""The set of possible football match outcomes (Win, Draw, or Lose)"""

Gamble = collections.namedtuple('Gamble', pspace)
"""Gamble on the outcome of a football match

With each possible match outcome, the gamble assigns a payoff (a |Real|);
it corresponds to winnings if it is positive, to losses if it is negative.

"""


def lp_setup(lp, accdss):
    """Set up the basic linear feasibility problem defining the gambling pool

    A number of players will form a gambling pool:

        * Each player states which gambles he finds *acceptable*, i.e., which
          have an expected payoff that is nonnegative
        * If there is a bet between the players, a gamble is assigned to each
          of them; from the player's perspective, the assigned gamble should
          be acceptable, meaning that is a positive linear combination of the
          gambles that that player has stated to be acceptable
        * The stake is assumed to be one, i.e., the assigned gamble's lowest
          payoff is -1.
        * There is a bet between the players if the gambles assigned to the
          players sum up to one for each possible match outcome and the
          assigned gambles do not all have identically zero payoff
        * An important quantity for each player is the ‘lower prevision’ of the
          gamble assigned to him: the greatest lower bound on the player's
          expected payoff for the assigned gamble that can be deduced from the
          player's stated acceptable gambles
        * The objective is to maximize the sum of the lower previsions,
          possibly under additional constraints

    :param lp: the problem object to set up
    :type lp: `ecyglpki.Problem`
    :param accdss: the gambles accepted by the players
    :type accdss: `dict` of `str` (player id) to `list` of `Gamble`
    :returns: the set up problem object
    :rtype: `ecyglpki.Problem`

    """
    for player, accds in accdss.items():  # per-player setup

        for outcome in pspace:
            player_outcome = player + outcome
            # constraints to ensure the assigned gamble is acceptable,
            # i.e., nonnegative for each outcome
            lp.add_named_rows(player_outcome)
            lp.set_row_bnds(player_outcome, None, 0)
            # variables for the gamble to be assigned
            lp.add_named_cols(player_outcome)
            lp.set_col_bnds(player_outcome, -1, None)  # stake of 1
            lp.set_mat_col(player_outcome, {player_outcome: -1})

        # variables for the assigned gambles' lower previsions
        player_lpr = player + 'lpr'
        lp.add_named_cols(player_lpr)
        lp.set_col_bnds(player_lpr, 0, None)  # ensure nonnegative lprs
        lp.set_mat_col(player_lpr, {player + outcome: 1 for outcome in pspace})
        lp.set_obj_coef(player_lpr, 1)  # objective is the sum of lprs

        # variables for the accepted gambles' coefficients that define the
        # assigned gamble as a positive linear combination of acceptable
        # gambles
        for accd in accds:
            j = lp.add_cols(1)
            lp.set_col_bnds(j, 0, None)  # ensure nonnegative coeffs
            lp.set_mat_col(j, {player + outcome: getattr(accd, outcome)
                               for outcome in pspace})

        # remove names from database that are not needed anymore
        for outcome in pspace:
            lp.set_row_name(player + outcome, '')

    # global setup: constraints that ensure the assigned gambles sum up to zero
    for outcome in pspace:
        lp.add_named_rows(outcome)
        lp.set_row_bnds(outcome, 0, 0)
        lp.set_mat_row(outcome, {player + outcome: 1
                                 for player in accdss.keys()})

    return lp


def lp_assign(accdss):
    """Assign gambles to all players that maximize the sum of lower previsions

    :param accdss: the gambles accepted by the players
    :type accdss: `dict` of `str` (player id) to `list` of `Gamble`
    :returns: the gamble assigned to the players
        and the corresponding lower prevision value
    :rtype: `dict` of `str` (player id) to
        {`'gamble'`: `Gamble`, `'lpr'`: `float`}

    >>> Agambles = [Gamble(.5, .25, -1), Gamble(1, -1, -1)]
    >>> Bgambles = [Gamble(-1, -1, 2), Gamble(1, .5, -.25)]
    >>> Cgambles = [Gamble(-1, 0, -1)]
    >>> accdss = {'Alice': Agambles, 'Bob': Bgambles, 'Cthulhu': Cgambles}
    >>> solution = lp_assign(accdss)
    >>> assert solution == (
    ...     {'Alice': {'gamble': Gamble(W=2.0, D=-1.0, L=-1.0), 'lpr': 0.5},
    ...      'Bob': {'gamble': Gamble(W=-1.0, D=-1.0, L=2.0), 'lpr': 0.0},
    ...      'Cthulhu': {'gamble': Gamble(W=-1.0, D=2.0, L=-1.0), 'lpr': 2.0}}
    ... )

    """
    lp = ecyglpki.Problem()  # construct the linear program
    lp_setup(lp, accdss)

    # solve the linear program
    lp.set_obj_dir('maximize')
    smcp = ecyglpki.SimplexControls() # (default) simplex control parameters
    smcp.presolve = True
    status = lp.simplex(smcp)  # solve using control parameters given
    if status != 'optimal':
        raise RuntimeError('Error while solving...: ' + status)
    smcp.presolve = False
    status = lp.exact(smcp)  # now solve exactly using control parameters given
    if status != 'optimal':
        raise RuntimeError('Error while solving...: ' + status)

    return {player: {
                'gamble': Gamble(*(lp.get_col_prim(player + outcome)
                                   for outcome in pspace)),
                'lpr': lp.get_col_prim(player + 'lpr')
            } for player in accdss.keys()}


def milp_assign(accdss):
    """Assign gambles to some players that maximize the sum of lower previsions

    As compared to `lp_assign`, we now try to make the bet fair in the sense
    that for each player the lower prevision for the assigned gamble is equal
    to the other player's lower prevision, or the gamble assigned to the player
    is zero for all outcomes.

    :param accdss: the gambles accepted by the players
    :type accdss: `dict` of `str` (player id) to `list` of `Gamble`
    :returns: the gamble assigned to the players
        and the corresponding lower prevision value
    :rtype: `dict` of `str` (player id) to
        {`'gamble'`: `Gamble`, `'lpr'`: `float`}

    >>> Agambles = [Gamble(.5, .25, -1), Gamble(1, -1, -1)]
    >>> Bgambles = [Gamble(-1, -1, 2), Gamble(1, .5, -.25)]
    >>> Cgambles = [Gamble(-1, 0, -1)]
    >>> accdss = {'Alice': Agambles, 'Bob': Bgambles, 'Cthulhu': Cgambles}
    >>> solution = milp_assign(accdss)
    >>> for player, data in solution.items():  # round floats for doctest
    ...     solution[player]['lpr'] = round(data['lpr'], 2)
    ...     solution[player]['gamble'] = Gamble(round(data['gamble'].W, 2),
    ...                                         round(data['gamble'].D, 2),
    ...                                         round(data['gamble'].L, 2))
    >>> assert solution == (
    ...     {'Alice': {'gamble': Gamble(W=1.4, D=0.0, L=-1.0), 'lpr': 0.4},
    ...      'Bob': {'gamble': Gamble(W=-0.4, D=-0.4, L=2.0), 'lpr': 0.4},
    ...      'Cthulhu': {'gamble': Gamble(W=-1.0, D=0.4, L=-1.0), 'lpr': 0.4}}
    ... )

    """
    milp = ecyglpki.Problem()  # construct the linear program
    lp_setup(milp, accdss)

    maxval = len(accdss) - 1  # the largest possible payoff in the pool:
                              # stake (==1) times number of other players

    # the unique lower prevision for included players
    milp.add_named_cols('lpr')
    milp.set_col_bnds('lpr', 0, None)

    for player in accdss.keys():
        player_b = player + 'b'
        # the binary variable to select whether a player is included or not
        milp.add_named_cols(player_b)
        milp.set_col_kind(player_b, 'binary')
        # add constraints to ensure that a player's lpr
        # is either equal to the global one or zero
        i = milp.add_rows(3)
        # player_lpr <= lpr
        milp.set_row_bnds(i, None, 0)
        milp.set_mat_row(i, {player + 'lpr': 1, 'lpr': -1})
        # player_lpr <= maxval * player_b
        milp.set_row_bnds(i+1, None, 0)
        milp.set_mat_row(i+1, {player + 'lpr': 1, player_b: -maxval})
        # lpr - player_lpr <= maxval * (1 - player_b)
        milp.set_row_bnds(i+2, None, maxval)
        milp.set_mat_row(i+2, {'lpr': 1, player + 'lpr': -1, player_b: maxval})
        # -1 * player_b <= player_outcome <= maxval * player_b
        for outcome in pspace:
            i = milp.add_rows(2)
            milp.set_row_bnds(i, None, 0)
            milp.set_mat_row(i, {player + outcome: 1, player_b: -maxval})
            milp.set_row_bnds(i+1, 0, None)
            milp.set_mat_row(i+1, {player + outcome: 1, player_b: 1})

    # solve the mixed integer linear program
    milp.set_obj_dir('maximize')
    iocp = ecyglpki.IntOptControls() # (default) int. opt. control parameters
    iocp.presolve = True
    status = milp.intopt(iocp)
    if status != 'optimal':
        raise RuntimeError('Error while solving...: ' + status)
    # fix the binary variables to their computed value and find exact solution
    for player in accdss.keys():
        player_b = player + 'b'
        player_b_val = milp.mip_col_val(player_b)
        milp.set_col_bnds(player_b, player_b_val, player_b_val)
    smcp = ecyglpki.SimplexControls() # (default) simplex control parameters
    smcp.presolve = True
    status = milp.simplex(smcp)  # solve
    if status != 'optimal':
        raise RuntimeError('Error while solving...: ' + status)
    smcp.presolve = False
    status = milp.exact(smcp)  # now solve exactly
    if status != 'optimal':
        raise RuntimeError('Error while solving...: ' + status)

    return {player: {
                'gamble': Gamble(*(milp.get_col_prim(player + outcome)
                                   for outcome in pspace)),
                'lpr': milp.get_col_prim(player + 'lpr')
            } for player in accdss.keys()}
