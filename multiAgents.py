# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from time import time
from collections import deque
from pprint import pprint

from pacman import GameState

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def find_nearest_food(start_point, GameMap, num_rows, num_cols):
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    visited = set()
    queue = deque([((start_point[1], start_point[0]), 0)])  # (point, distance)

    while queue:
        current_point, distance = queue.popleft()
        row, col = current_point
        # Check if the current point contains food

        if row < num_rows and col < num_cols and GameMap[row][col] == '.':
            return distance

        # Explore the neighbors in all four directions
        for direction in directions:
            new_row = row + direction[0]
            new_col = col + direction[1]

            # Check if the new position is within the map boundaries and not a wall
            if 0 <= new_row < num_rows and 0 <= new_col < num_cols and GameMap[new_row][new_col] != '%' and \
                    GameMap[new_row][new_col] != 'G':
                new_point = (new_row, new_col)

                # Check if the new point has not been visited before
                if new_point not in visited:
                    visited.add(new_point)
                    queue.append((new_point, distance + 1))

    # If no food is found, return 100 to indicate that there is no reachable food
    return 100


def find_nearest_ghost(start_point, GameMap, num_rows, num_cols, max_move=3):
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    visited = {(start_point[1], start_point[0])}  # Set of tuples (row, col)
    queue = deque([(start_point[1], start_point[0], 0)])  # (row, col, distance)

    while queue:
        row, col, distance = queue.popleft()

        if GameMap[row][col] == 'G' or distance > max_move:
            return distance

        for direction in directions:
            new_row = row + direction[0]
            new_col = col + direction[1]
            new_point = (new_row, new_col)

            if 0 <= new_row < num_rows and 0 <= new_col < num_cols \
                    and GameMap[new_row][new_col] != '%' and new_point not in visited:
                visited.add(new_point)
                queue.append((new_row, new_col, distance + 1))

    return 100

def find_nearest_thing(start_point, GameMap, num_rows, num_cols):
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    visited = {(start_point[1], start_point[0])}  # Set of tuples (row, col)
    queue = deque([(start_point[1], start_point[0], 0)])  # (row, col, distance)
    foodDist = 100
    ghostDist = 5
    capsuleDist = 100
    ghost_flag = food_flag = capsule_flag = False

    while queue:
        row, col, distance = queue.popleft()

        if GameMap[row][col] == '.':
            foodDist = min(distance, foodDist)
            food_flag = True
        if GameMap[row][col] == 'o':
            capsuleDist = min(distance, capsuleDist)
            capsule_flag = True
        if GameMap[row][col] == 'G':
            ghostDist = min(distance, ghostDist)
            ghost_flag = True
            continue

        if food_flag and (ghost_flag or distance > 3) and (capsule_flag or distance > 10):
            return foodDist, ghostDist, capsuleDist

        for direction in directions:
            new_row = row + direction[0]
            new_col = col + direction[1]
            new_point = (new_row, new_col)

            if 0 <= new_row < num_rows and 0 <= new_col < num_cols and GameMap[new_row][new_col] != '%' \
                    and new_point not in visited:
                visited.add(new_point)
                queue.append((new_row, new_col, distance + 1))

    return foodDist, ghostDist, capsuleDist
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        board = successorGameState.__str__().splitlines()[:-1][::-1]
        num_rows, num_cols = len(board), len(board[0])
        score = successorGameState.getScore()
        
        #get distance from successor to the closest food
        minimum_distance = find_nearest_food(newPos, board, num_rows, num_cols)
        score += 1 / (minimum_distance * 100)
        #avoding the ghost in near places
        nearest_ghost = find_nearest_ghost(newPos, board, num_rows, num_cols)
        newGhostStates = successorGameState.getGhostStates()
        minScaredTimes = min([ghostState.scaredTimer for ghostState in newGhostStates])
        if minScaredTimes == 0 and nearest_ghost > 0:
            score -= (15 / nearest_ghost)
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn= 'e_utility', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def mini_max(self, game_state, depth, agent_num):
        if game_state.isWin() or game_state.isLose() or (agent_num == 0 and depth == self.depth):
            return self.evaluationFunction(game_state)

        num_agents = game_state.getNumAgents()
        legalMoves = game_state.getLegalActions(agent_num)
        next_agent = (agent_num + 1) % num_agents
        best_action = Directions.STOP

        if agent_num == 0:
            return self.maximize(game_state, depth, agent_num, legalMoves, next_agent)
        else:
            return self.minimize(game_state, depth, agent_num, legalMoves, next_agent)

    def maximize(self, game_state, depth, agent_num, legal_moves, next_agent):
        max_eval_score = float('-inf')
        best_action = Directions.STOP

        for move in legal_moves:
            successorGameState = game_state.generateSuccessor(agent_num, move)
            score = self.mini_max(successorGameState, depth, next_agent)
            if score > max_eval_score:
                max_eval_score = score
                best_action = move

        return best_action if depth == 0 else max_eval_score

    def minimize(self, game_state, depth, agent_num, legal_moves, next_agent):
        min_eval_score = float('inf')

        for move in legal_moves:
            successorGameState = game_state.generateSuccessor(agent_num, move)
            score = self.mini_max(successorGameState, depth + (next_agent == 0), next_agent)
            min_eval_score = min(min_eval_score, score)

        return min_eval_score


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.mini_max(gameState, 0 ,0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def mini_max(self, game_state, depth, agent_num, alpha, beta):
        if game_state.isWin() or game_state.isLose() or (agent_num == 0 and depth == self.depth):
            return self.evaluationFunction(game_state)

        num_agents = game_state.getNumAgents()
        legalMoves = game_state.getLegalActions(agent_num)
        next_agent = agent_num + 1
        best_action = Directions.STOP

        if agent_num == 0:
            max_eval_score = float('-inf')
            for move in legalMoves:
                successorGameState = game_state.generateSuccessor(agent_num, move)
                score = self.mini_max(successorGameState, depth, next_agent, alpha, beta)
                if score > max_eval_score:
                    max_eval_score = score
                    best_action = move
                alpha = max(alpha, max_eval_score)
                if beta <= max_eval_score:
                    break
            if depth == 0:
                return best_action
            else:
                return max_eval_score

        else:
            min_eval_score = float('inf')
            if agent_num + 1 == num_agents:
                depth += 1
                next_agent = 0

            for move in legalMoves:
                successorGameState = game_state.generateSuccessor(agent_num, move)
                score = self.mini_max(successorGameState, depth, next_agent, alpha, beta)
                min_eval_score = min(min_eval_score, score)
                beta = min(beta, min_eval_score)
                if min_eval_score <= alpha:
                    break
            return min_eval_score
            
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.mini_max(gameState, 0, 0, float('-inf'), float('inf'))
class ExpectimaxAgent(MultiAgentSearchAgent):
    def expecti_max(self, game_state, depth, agent_num):
        if game_state.isWin() or game_state.isLose() or (agent_num == 0 and depth == self.depth):
            return self.evaluationFunction(game_state)

        num_agents = game_state.getNumAgents()
        legalMoves = game_state.getLegalActions(agent_num)
        next_agent = agent_num + 1
        best_action = Directions.STOP

        if agent_num == 0:
            max_eval_score = float('-inf')
            for move in legalMoves:
                successorGameState = game_state.generateSuccessor(agent_num, move)
                score = self.expecti_max(successorGameState, depth, next_agent)
                if score > max_eval_score:
                    max_eval_score = score
                    best_action = move
            if depth == 0:
                return best_action
            else:
                return max_eval_score

        else:
            if agent_num + 1 == num_agents:
                depth += 1
                next_agent = 0
            ind = sum_eval_score = 0
            for move in legalMoves:
                ind += 1
                successorGameState = game_state.generateSuccessor(agent_num, move)
                score = self.expecti_max(successorGameState, depth, next_agent)
                sum_eval_score += score
            return sum_eval_score / len(legalMoves)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.expecti_max(gameState, 0, 0)

class ExpectimaxAlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax_alpha_beta(self, game_state, depth, agent_num, alpha, beta):
        if game_state.isWin() or game_state.isLose() or (agent_num == 0 and depth == self.depth):
            return self.evaluationFunction(game_state)

        num_agents = game_state.getNumAgents()
        legal_moves = game_state.getLegalActions(agent_num)
        next_agent = (agent_num + 1) % num_agents

        if agent_num == 0:  # Pacman's turn (MAX)
            return self.maximize(game_state, depth, agent_num, legal_moves, next_agent, alpha, beta)
        else:  # Ghosts' turn (EXP)
            return self.expect(game_state, depth, agent_num, legal_moves, next_agent, alpha, beta)

    def maximize(self, game_state, depth, agent_num, legal_moves, next_agent, alpha, beta):
        max_eval_score = float('-inf')
        best_action = Directions.STOP

        for move in legal_moves:
            successorGameState = game_state.generateSuccessor(agent_num, move)
            score = self.expectimax_alpha_beta(successorGameState, depth, next_agent, alpha, beta)
            if score > max_eval_score:
                max_eval_score = score
                best_action = move
            alpha = max(alpha, max_eval_score)
            if max_eval_score >= beta:
                break  # Beta cutoff

        return best_action if depth == 0 else max_eval_score

    def expect(self, game_state, depth, agent_num, legal_moves, next_agent, alpha, beta):
        sum_eval_score = 0
        num_actions = len(legal_moves)

        for move in legal_moves:
            successorGameState = game_state.generateSuccessor(agent_num, move)
            score = self.expectimax_alpha_beta(successorGameState, depth + (next_agent == 0), next_agent, alpha, beta)
            sum_eval_score += score

        return sum_eval_score / num_actions

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax_alpha_beta(gameState, 0, 0, float('-inf'), float('inf'))



def e_utility(currentGameState):
    newPos = currentGameState.getPacmanPosition()
    gameMap = currentGameState.__str__().splitlines()[:-1][::-1]
    num_rows, num_cols = len(gameMap), len(gameMap[0])

    score = currentGameState.getScore()

    if currentGameState.isWin() or currentGameState.isLose():
        return score

    nearest_food, nearest_ghost, nearest_capsule = find_nearest_thing(newPos, gameMap, num_rows, num_cols)
    score += (1 / (nearest_food * 1000))

    newGhostStates = currentGameState.getGhostStates()
    minScaredTimes = min([ghostState.scaredTimer for ghostState in newGhostStates])

    if minScaredTimes == 0 and 3 > nearest_ghost > 0:
        score -= (5 / nearest_ghost)

    if minScaredTimes:
        score += (1 / (nearest_ghost * 100) ** 2)

    if nearest_capsule != 100:
        score += (1 / (nearest_capsule * 10))

    return score

# Abbreviation
better = e_utility
