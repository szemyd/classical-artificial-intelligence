
from sample_players import DataPlayer

from isolation import DebugState

import random
import time
import math

# import pickle

_WIDTH = 11
_HEIGHT = 9
_BOARDSIZE = _WIDTH * _HEIGHT

def writeToCsv(myCsvRow):
    with open('depth_diagnostics.csv', 'a') as fd:
        fd.write(myCsvRow)

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
        self.start_time = None
        self.max_time = 140  # in miliseconds
        self.depth_limit = _BOARDSIZE
        self.timertest_off = False

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

        print('In get_action(), state received:')
        debug_board = DebugState.from_state(state)
        print(debug_board)

        # With iterative deepening
        # for depth in range(1, self.depth_limit + 1):
        #   self.queue.put(self.minimax(state, depth))
        # self.start_time = time.time()
        # With iterative deepening & Alpha-Beta Pruning
        for depth in range(1, self.depth_limit + 1):
            # if self.timertest():
                # print("\t", depth)
                # writeToCsv(
                #         'Player Weighted Self' + 
                #     "," + str(depth)+ 
                #     '\n')
            if len(state.actions()) > 0: 
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
            if self.timertest():
                print("max DEPTH = ", depth)
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
        return self.weighted_binary(state, player)

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

        w1, w2 = self.rectified(
            (progress * 2) - 1), self.rectified(1 - (progress * 4))

        return w1 * self.weighted_self(state, player) + w2 * self.weighted_opp(state, player) + self.num_moves(state, player)

    def weighted_binary(self, state, player):

        # this goes from 0 -> as 1 as the game progresses
        progress = state.ply_count/_BOARDSIZE

        return self.weighted_opp(state, player) if progress <= 0.5 else self.weighted_self(state, player)

    def rectified(self, num):
        return max(0.0, num)

branch = '├'
pipe = '|'
end = '└'
dash = '─'

class TreeNode():
    def __init__(self, state, parent, origin_action, player_layer):
        self.parent = parent 
        self.state = state
        self.children = []
        
        self.uct = 0
        self.win_count = 2
        self.sim_count = 2

        self.origin_action = origin_action
        self.player_layer = player_layer

        self.max_depth = 1

        # self.simulation_num = 0

    def __str__(self):
        return "My State is: " + str(self.state) + " \n \t children are: " + str(len(self.children)) + " \n \t UCT: " + str(self.uct)

    def init_children(self):
        self.children = [TreeNode(self.state.result(x), self, x, abs(self.player_layer - 1)) for x in self.state.actions()]

    def visualize(self, depth):
        spaces = ''
        for _ in range(depth):
            spaces += '\t'
        
        print(spaces + str(self.win_count) + '/' + str(self.sim_count))
        print(spaces + '[ ' + str(round(self.uct, 4)) + ' ]')
       
        if len(self.children) < 1 or depth > self.max_depth: 
            return
            
        else: 
            for child in self.children:
                child.visualize(depth + 1)



class MonteCarloPlayer(DataPlayer):

    def __init__(self, player_id):
        self.player_id = player_id
        self.max_time = 240  # in miliseconds
        self.iteration_limit = 10000
        self.exploration_weight = math.sqrt(2)  # usually it is sqrt of 2
        self.start_time = 0

        print("Player ID: ", self.player_id)

    def timertest(self):
        if (time.time() - self.start_time) * 1000 > self.max_time:
            return True  # in milliseconds
        else:
            return False

    def get_action(self, state):

        prev_tree = TreeNode(state, None, None, abs(self.player_id - 1))

        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else: 
            for _ in range(self.iteration_limit):
                best_state = self.montecarlo(state, prev_tree)

                if hasattr(best_state,'origin_action'): self.queue.put(best_state.origin_action)
                else: self.queue.put(random.choice(state.actions()))


    def montecarlo(self, state, prev_tree):
        
        def run_search(t):

            leaf = select(t)
            if leaf is False:
                return None

            result = simulate(leaf.state)
            backpropagate(result, leaf)

            best_next_state = choose_best(t)

            # print(best_next_state.origin_action)
            # if self.timertest(): 
            # # if visualize: 
            #     t.visualize(0)
                # print(best_next_state.origin_action)
            # self.context = best_next_state

            return best_next_state



        def select(t):
            if t.state.terminal_test():
                return False

            if len(t.children) > 0:
                if t.parent: t.uct = uct(t)
                for child in t.children:
                    child.uct = uct(child)
                best_child = max(t.children, key=lambda x: x.uct)
                return select(best_child)

            else:
                t.init_children()
                random_child = random.choice(t.children)
                random_child.uct = uct(random_child)
                return random_child
          

        def simulate(state):
            if state.terminal_test():
                return state.utility(self.player_id)
            else:
                return simulate(state.result(random.choice(state.actions())))        


        def backpropagate(result, leaf):
            leaf.sim_count += 1

            conv_result = 0
            if result > 0: conv_result = 1

            if result == 0: leaf.win_count += 1
            elif leaf.state.player() == self.player_id:
                    leaf.win_count += conv_result
            else: leaf.win_count += 1 - conv_result

            if leaf.parent == None:
                return
            else: return backpropagate(result, leaf.parent)
        

        def uct(state):
            log_n = math.log(state.parent.sim_count, 10) 
            explore_term = self.exploration_weight * math.sqrt(log_n / state.sim_count)
            exploit_term = (state.win_count / state.sim_count)

            return exploit_term + explore_term

        def choose_best(state):
            return max(state.children, key=lambda x: x.sim_count)

        def find_node(prev_tree):
            for opponent_child in prev_tree.children:
                if opponent_child.state.locs == state.locs:
                    return opponent_child

        def create_tree():
            return TreeNode(state, None, None, abs(self.player_id - 1))
        
        def process_tree(prev_tree):
            t = None
            if prev_tree is not None:  t = find_node(prev_tree)
            else: t = create_tree()

            if t is None: 
                # print("Didn't find node!")
                t = create_tree()
            
            return t

        # return run_search(process_tree(prev_tree))
        return run_search(prev_tree)





        # def calc_ucts(leaf):
        #     if leaf.parent == None:
        #         return
        #     else: 
        #         leaf.uct = uct(leaf)
        #         calc_ucts(leaf.parent)

        # def calc_ucts_top_down(leaf):
        #     if len(leaf.children) < 1:
        #         return
        #     else:
        #         if leaf.parent: 
        #             leaf.uct = uct(leaf)
        #         for child in leaf.children:
        #             calc_ucts_top_down(child)





    ### SOLUTION that uses Isolation() class ### 
    ## This solution appends values to the Isolation() class, the state object
    ## that is passed to get_action(). This solution might have errors, as
    ## I have changed the approach before fully debugging this.

    # def select(self, leaf, parent):  # Leaf is a state
    #     leaf.parent = parent

    #     if leaf.terminal_test():
    #         return leaf

    #     if hasattr(leaf, 'children') and len(leaf.children) > 0:
    #         self.calc_children_ucts(leaf)
    #         return self.select(max(leaf.children, key=lambda x: x.uct), leaf)

    #     else:
    #         leaf.children = [leaf.result(state) for state in leaf.actions()]
    #         for child in leaf.children:
    #             child.sim_count = 0
    #             child.win_count = 0
    #         randomly_selected = random.choice(leaf.children)
    #         randomly_selected.parent = leaf
    #         return randomly_selected

    # def simulate(self, leaf):
    #     if leaf.terminal_test():
    #         return leaf.utility(self.player_id)
    #     else:
    #         return self.simulate(leaf.result(random.choice(leaf.actions())))

    # def choose_best(self, state):
    #     return max(state.children, key=lambda x: x.sim_count)

    # def backpropagate(self, result, leaf):
    #     if hasattr(leaf, 'sim_count'):
    #         leaf.sim_count += 1
    #     else:
    #         leaf.sim_count = 1

    #     if leaf.player() == self.player_id:
    #         if result > 0:
    #             if hasattr(leaf, 'win_count'):
    #                 leaf.win_count += 1
    #             else:
    #                 leaf.win_count = 1

    #     if leaf.parent == None:
    #         return None

    #     self.backpropagate(result, leaf.parent)

    # def calc_children_ucts(self, state):
    #     for child in state.children:
    #         child.uct = self.uct(child)

    # def uct(self, state):
    #     if state.sim_count == 0:
    #         return 0
    #     else:
    #         log_n = math.log(state.parent.sim_count)
    #         return (state.win_count / state.sim_count) + (self.exploration_weight * math.sqrt(log_n / state.sim_count))
