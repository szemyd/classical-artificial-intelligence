###############################################################################
#     YOU CAN MODIFY THIS FILE, BUT CHANGES WILL NOT APPLY DURING GRADING     #
###############################################################################
import logging
import pickle
import random

import time
import math

from isolation import DebugState

logger = logging.getLogger(__name__)


class BasePlayer:
    def __init__(self, player_id):
        self.player_id = player_id
        self.timer = None
        self.queue = None
        self.context = None
        self.data = None

    def get_action(self, state):
        """ Implement a function that calls self.queue.put(ACTION) within the allowed time limit 

        See RandomPlayer and GreedyPlayer for examples.
        """
        raise NotImplementedError


class DataPlayer(BasePlayer):
    def __init__(self, player_id):
        super().__init__(player_id)
        try:
            with open("data.pickle", "rb") as f:
                self.data = pickle.load(f)
        except (IOError, TypeError) as e:
            logger.info(str(e))
            self.data = None


class RandomPlayer(BasePlayer):
    def get_action(self, state):
        """ Randomly select a move from the available legal moves.

        Parameters
        ----------
        state : `isolation.Isolation`
            An instance of `isolation.Isolation` encoding the current state of the
            game (e.g., player locations and blocked cells)
        """
        # print("OPP state, ", state)
        # debug_board = DebugState.from_state(state)
        # print(debug_board)
        # print("my loc: ",state.locs[self.player_id%2])
        


        self.queue.put(random.choice(state.actions()))


class GreedyPlayer(BasePlayer):
    """ Player that chooses next move to maximize heuristic score. This is
    equivalent to a minimax search agent with a search depth of one.
    """
    def score(self, state):
        own_loc = state.locs[self.player_id]
        own_liberties = state.liberties(own_loc)
        return len(own_liberties)

    def get_action(self, state):
        """Select the move from the available legal moves with the highest
        heuristic score.

        Parameters
        ----------
        state : `isolation.Isolation`
            An instance of `isolation.Isolation` encoding the current state of the
            game (e.g., player locations and blocked cells)
        """
        self.queue.put(max(state.actions(), key=lambda x: self.score(state.result(x))))


class MinimaxPlayer(BasePlayer):
    """ Implement an agent using any combination of techniques discussed
    in lecture (or that you find online on your own) that can beat
    sample_players.GreedyPlayer in >80% of "fair" matches (see tournament.py
    or readme for definition of fair matches).

    Implementing get_action() is the only required method, but you can add any
    other methods you want to perform minimax/alpha-beta/monte-carlo tree search,
    etc.

    **********************************************************************
    NOTE: The test cases will NOT be run on a machine with GPU access, or
          be suitable for using any other machine learning techniques.
    **********************************************************************
    """
    def get_action(self, state):
        """ Choose an action available in the current state

        See RandomPlayer and GreedyPlayer for examples.

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        **********************************************************************
        NOTE: since the caller is responsible for cutting off search, calling
              get_action() from your own code will create an infinite loop!
              See (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # randomly select a move as player 1 or 2 on an empty board, otherwise
        # return the optimal minimax move at a fixed search depth of 3 plies
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.minimax(state, depth=3))

    def minimax(self, state, depth):

        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)




class CustomOpponent:
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def __init__(self, player_id):
        self.player_id = player_id
        self.start_time = None
        self.max_time = 140  # in miliseconds
        self.depth_limit = 5
        self.timertest_off = True

        self.my_moves_prev = 0
        self.opp_moves_prev = 0


    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        #

        # For Debugging

        # print('In get_action(), state received:')
        # debug_board = DebugState.from_state(state)
        # print(debug_board)

        # With iterative deepening
        # for depth in range(1, self.depth_limit + 1):
        #   self.queue.put(self.minimax(state, depth))

        # With iterative deepening & Alpha-Beta Pruning
        for depth in range(1, self.depth_limit + 1):
            self.queue.put(self.alpha_beta_search(state, depth))

        # Without iterative deepening
        # self.queue.put(self.minimax(state, self.depth_limit))

        # With random choice
        # self.queue.put(random.choice(state.actions()))

    def iterative_deepening(self, state, depth_limit):

        self.start_time = time.time()
        if state.terminal_test() or len(state.actions()) < 1:
            print("terminal test was true")
            return state.locs[self.player_id]

        best_move = None
        for depth in range(1, depth_limit + 1):
            # print("==========")
            # print("DEPTH = ", depth)
            if self.timertest():
                return best_move
            best_move = self.minimax(state, depth)

        print(best_move)
        return best_move

    def timertest(self):
        if self.timertest_off:
            return False
        if (time.time() - self.start_time) * 1000 > self.max_time:
            return True  # in milliseconds
        else:
            return False

    ## MINIMAX ##
    def minimax(self, state, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.

        You can ignore the special case of calling this function
        from a terminal state.
        """
        best_score = float("-inf")
        best_move = random.choice(state.actions())
        for a in state.actions():
            if self.timertest():
                return best_move
            # call has been updated with a depth limit
            v = self.min_value(state.result(a), depth - 1)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move

    def min_value(self, state, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(self.player_id)

        if depth <= 0: ## self.timertest():
            return self.score(state, self.player_id)

        v = float("inf")
        for a in state.actions():
            v = min(v, self.max_value(state.result(a), depth - 1))
        return v

    def max_value(self, state, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(self.player_id)

        if depth <= 0: ## self.timertest():
            return self.score(state, self.player_id)

        v = float("-inf")
        for a in state.actions():
            v = max(v, self.min_value(state.result(a), depth - 1))
        return v

    ## ALPHA-BETA ##
    def alpha_beta_search(self, state, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.

        You can ignore the special case of calling this function
        from a terminal state.
        """
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = random.choice(state.actions())
        for a in state.actions():
            v = self.min_valueAB(state.result(a), depth - 1, alpha, beta)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move

    def min_valueAB(self, state, depth, alpha, beta):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(self.player_id)

        if depth <= 0: ## self.timertest():
            return self.score(state, self.player_id)

        v = float("inf")
        for a in state.actions():
            v = min(v, self.max_valueAB(
                state.result(a), depth - 1, alpha, beta))

            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def max_valueAB(self, state, depth, alpha, beta):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if state.terminal_test():
            return state.utility(self.player_id)

        if depth <= 0: ## self.timertest():
            return self.score(state, self.player_id)

        v = float("-inf")
        for a in state.actions():
            v = max(v, self.min_valueAB(
                state.result(a), depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)

        return v


    ## HEURISTICS ##
    def score(self, state, player):
        # return self.num_moves(state, player)
        return self.liberty_difference(state, player)

    def liberty_difference(self, state, player):
        own_loc = state.locs[player]
        opp_loc = state.locs[1 - player]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    def num_moves(self, state, player):
        player_location = state.locs[player]
        return len(state.liberties(player_location))
    
