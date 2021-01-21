
from sample_players import DataPlayer

from isolation import DebugState

import random
import time
import math

_WIDTH = 11
_HEIGHT = 9
_BOARDSIZE= _WIDTH * _HEIGHT


class CustomPlayer2(DataPlayer):
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

        # print("NEW GAME")

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

        if depth <= 0:  # or self.timertest():
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

        if depth <= 0:  # self.timertest():
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

        if depth <= 0:  # self.timertest():
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

        if depth <= 0:  # self.timertest():
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
        return self.weighted_relu(state, player)

    def liberty_difference(self, state, player):
        own_loc = state.locs[player]
        opp_loc = state.locs[1 - player]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    def num_moves(self, state, player):
        player_location = state.locs[player]
        return len(state.liberties(player_location))

    def movesDelta(self, state, player):
        # This function returns the difference between the amount of change
        # in number of available actions for player minus the opponent.
        # If this number is positive that means the player is moving into positions with bigger 'optionality',
        # if it's negative that means the opponent has a better move tendency

        myMoves = self.num_moves(state, player)
        oppMoves = self.num_moves(state, abs(player - 1))

        dMyMoves = myMoves - self.my_moves_prev
        dOpponentMoves = oppMoves - self.opp_moves_prev

        self.my_moves_prev = myMoves
        self.opp_moves_prev = oppMoves

        return dMyMoves - dOpponentMoves

    def liberties_of_liberties(self, state, player):  # Takes too long
        own_loc = state.locs[player]
        opp_loc = state.locs[1 - player]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)

        my_next_step_liberties = 0
        opp_next_step_liberties = 0
        for loc in own_liberties:
            my_next_step_liberties += len(state.liberties(loc))
        for loc in opp_liberties:
            opp_next_step_liberties += len(state.liberties(loc))

        return my_next_step_liberties - opp_next_step_liberties

    def weighted_self(self, state, player):
        return math.pow(self.num_moves(state, player), 2) - self.num_moves(state, player - 1)

    def weighted_opp(self, state, player):
        return self.num_moves(state, player) - math.pow(self.num_moves(state, player - 1), 2)

    def weighted_linear_function(self, state, player):

        # this goes from 0 -> as 1 as the game progresses
        progress = state.ply_count/_BOARDSIZE

        w1, w2 = progress, 1 - progress
 
        return w1 * self.weighted_self(state, player) + w2 * self.weighted_opp(state, player) 

    def weighted_relu(self, state, player):

        # this goes from 0 -> as 1 as the game progresses
        progress = state.ply_count/_BOARDSIZE

        w1, w2 = self.rectified((progress * 2) - 1 ), self.rectified(1 - (progress * 4)) 
 
        return w1 * self.weighted_self(state, player) + w2 * self.weighted_opp(state, player) + self.num_moves(state, player)

    def weighted_binary(self, state, player):

        # this goes from 0 -> as 1 as the game progresses
        progress = state.ply_count/_BOARDSIZE

        return self.weighted_opp(state, player) if progress <= 0.5 else self.weighted_self(state,player)

    def rectified(self, num):
        return max(0.0, num)


class CustomPlayer(DataPlayer):
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
        self.max_time = 140  # in miliseconds
        self.iteration_limit = 10000

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

        ## For Debugging

        # print('In get_action(), state received:')
        # debug_board = DebugState.from_state(state)
        # print(debug_board)

        ## Monte Carlo ##

        root = state

        


        for depth in range(self.iteration_limit):
            leaf = self.select(state) ## selecting a leaf node which has no children yet or which has 
            if(leaf != state): ## if state is not terminal (we have found a valid child leaf)
                child = self.expand(leaf) ## should return a state
                result = self.simulate(child) ## should return a result
                self.backpropagate(result, child) ## updates the parent nodes according to the result
            self.queue.put(self.choose_best(result))

        def select(self, state):

            if state.terminal_test(): return state

            if len(state.children) > 0:
                return self.select(max(state.children, key=uct))

            else: 
                state["children"] = state.actions()
                return random.choice(state.children)


        def expand(self, leaf):
            pass

        def simulate(self, child):
            pass
        
        def choose_best(self, result):
            pass

        def backpropagate(self, result, child):
            pass

        def uct(self, state):
            pass