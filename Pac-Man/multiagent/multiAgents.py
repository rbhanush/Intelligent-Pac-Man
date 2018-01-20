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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostLocations = currentGameState.getGhostPositions()
        score= successorGameState.getScore()
        if self.isDangerZone(ghostLocations,newPos):
            score -= 100
        score -=self.closestDist(newFood.asList(),newPos)
        if successorGameState.getNumFood() < currentGameState.getNumFood():
            score += 100
        return score

    def isDangerZone(self, ghostLocations, successorLocation):
        for each_location in ghostLocations:
            if (each_location[0]+1,each_location[1]) == successorLocation:
                return True
            elif (each_location[0]-1,each_location[1]) == successorLocation:
                return True
            elif (each_location[0],each_location[1]+1) == successorLocation:
                return True
            elif (each_location[0],each_location[1]-1) == successorLocation:
                return True
            elif each_location == successorLocation:
                return True

    def closestDist(self,distList,newPos):
        closestDist = 100
        for each_pos in distList:
            thisdist = util.manhattanDistance(each_pos, newPos)
            if (thisdist < closestDist):
                closestDist = thisdist
        return closestDist

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

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
        """
        return self.value(gameState)

    def isEndGame(self,gameState):
        return gameState.isWin() or gameState.isLose()

    def value(self,gameState):
        if self.isEndGame(gameState):
            return gameState.getScore()
        best_action = Directions.STOP
        best_value = float("-inf")
        value = best_value
        for each_action in gameState.getLegalActions():
            temp_value = value
            value = max(self.minValue(gameState.generateSuccessor(0,each_action), 0, 1),value)
            if value > temp_value:
                best_action = each_action
        return best_action

    def minValue(self,gameState, currDepth, ghostIndex):
        if self.isEndGame(gameState):
            return gameState.getScore()
        value = float("inf")
        for each_action in gameState.getLegalActions(ghostIndex):
            if ghostIndex==gameState.getNumAgents()-1:
                if currDepth==self.depth-1:
                    value = min(self.evaluationFunction(gameState.generateSuccessor(ghostIndex,each_action)),value)
                else:
                    value = min(self.maxValue(gameState.generateSuccessor(ghostIndex,each_action),currDepth+1),value)
            else:
                value = min(self.minValue(gameState.generateSuccessor(ghostIndex,each_action),currDepth,ghostIndex+1),value)
        return value

    def maxValue(self,gameState,currDepth):
        if self.isEndGame(gameState):
            return gameState.getScore()
        value=float("-inf")
        for each_action in gameState.getLegalActions():
            value = max(self.minValue(gameState.generateSuccessor(0,each_action),currDepth,1),value)
        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.value(gameState)

    def isEndGame(self, gameState):
        return gameState.isWin() or gameState.isLose()

    def value(self, gameState):
        if self.isEndGame(gameState):
            return gameState.getScore()
        best_action = Directions.STOP
        best_value = float("-inf")
        value = best_value
        alpha = float("-inf")
        beta = float("inf")
        for each_action in gameState.getLegalActions():
            temp_value = value
            value = max(self.minValue(gameState.generateSuccessor(0, each_action), 0, 1, alpha, beta), value)
            if value > temp_value:
                best_action = each_action
            alpha = max(alpha,value)
            if value>beta:
                break
        return best_action

    def minValue(self, gameState, currDepth, ghostIndex, alpha, beta):
        if self.isEndGame(gameState):
            return gameState.getScore()
        value = float("inf")
        for each_action in gameState.getLegalActions(ghostIndex):
            if ghostIndex == gameState.getNumAgents() - 1:
                if currDepth == self.depth - 1:
                    value = min(self.evaluationFunction(gameState.generateSuccessor(ghostIndex, each_action)),
                                value)
                else:
                    value = min(self.maxValue(gameState.generateSuccessor(ghostIndex, each_action), currDepth + 1,alpha, beta),
                                value)
            else:
                value = min(
                    self.minValue(gameState.generateSuccessor(ghostIndex, each_action), currDepth, ghostIndex + 1,alpha, beta),
                    value)
            beta = min(beta,value)
            if value < alpha:
                return value
        return value


    def maxValue(self, gameState, currDepth, alpha, beta):
        if self.isEndGame(gameState):
            return gameState.getScore()
        value = float("-inf")
        for each_action in gameState.getLegalActions():
            value = max(self.minValue(gameState.generateSuccessor(0, each_action), currDepth, 1,alpha, beta), value)
            alpha = max(alpha,value)
            if value > beta:
                return value
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.value(gameState)

    def isEndGame(self, gameState):
        return gameState.isWin() or gameState.isLose()

    def value(self, gameState):
        if self.isEndGame(gameState):
            return gameState.getScore()
        best_action = Directions.STOP
        best_value = float("-inf")
        value = best_value
        for each_action in gameState.getLegalActions():
            temp_value = value
            value = max(self.minValue(gameState.generateSuccessor(0, each_action), 0, 1), value)
            if value > temp_value:
                best_action = each_action
        return best_action

    def minValue(self, gameState, currDepth, ghostIndex):
        if gameState.isLose():
            return gameState.getScore()
        value = 0

        for each_action in gameState.getLegalActions(ghostIndex):
            uni_prob = 1.0 / len(gameState.getLegalActions(ghostIndex))
            if ghostIndex == gameState.getNumAgents() - 1:
                if currDepth == self.depth - 1:
                    successor_value = self.evaluationFunction(gameState.generateSuccessor(ghostIndex, each_action))
                    value+= uni_prob*successor_value
                else:
                    successor_value = self.maxValue(gameState.generateSuccessor(ghostIndex, each_action), currDepth + 1)
                    value += uni_prob * successor_value
            else:
                successor_value = self.minValue(gameState.generateSuccessor(ghostIndex, each_action), currDepth, ghostIndex + 1)
                value += uni_prob * successor_value
        return value


    def maxValue(self, gameState, currDepth):
        if self.isEndGame(gameState):
            return gameState.getScore()
        value = float("-inf")
        for each_action in gameState.getLegalActions():
            value = max(self.minValue(gameState.generateSuccessor(0, each_action), currDepth, 1), value)
        return value


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      
      The features I have used is :
      - 2 * closest ghost distance 
      -40 * reciprocal of average distance of food from position
      -number of food at a radius of 3
      -state score
      
      I have used manhattan distance as the metric for all these features
      
      score is sum of all these features
      
      I 
    """
    def closestDist(distList, newPos):
        if len(distList)==0:
            return (1,0,1,1)
        closestDist = 100
        cloesetPos = ""
        sum =1
        for each_pos in distList:
            thisdist = util.manhattanDistance(each_pos, newPos)
            sum +=thisdist
            if (thisdist < closestDist):
                closestDist = thisdist
                cloesetPos = each_pos
        return (closestDist if closestDist else 1 , cloesetPos,sum,float(sum)/len(distList))

    def numOfFood(distList,newPos,radius):
        num=1
        for each in distList:
            if util.manhattanDistance(each,newPos) <= radius:
                num+=1
        return num

    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood().asList()
    currentGhostPositions = currentGameState.getGhostPositions()
    score = currentGameState.getScore()

    averageFoodDistance = closestDist(currentFood,currentPos)[3]
    closestGhostDistance = closestDist(currentGhostPositions,currentPos)[0]
    numOfFood = numOfFood(currentFood, currentPos, 3)
    closestGDValue = closestGhostDistance*2
    avgFD = 40*(1.0/averageFoodDistance)
    newScore = score + avgFD + closestGDValue + numOfFood
    return newScore



# Abbreviation
better = betterEvaluationFunction

