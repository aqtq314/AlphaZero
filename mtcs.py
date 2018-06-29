import enum
import collections
import typing
import math
import re
import numpy as np
import keras


isDebug = False

class Player(enum.Enum):
    P1 = 1
    P2 = 2

    def toStateValue(self):
        return 1 if self == Player.P1 else -1

    def opposite(self):
        return Player.P2 if self == Player.P1 else Player.P1


class GameBoardBase:
    def copyTo(self, other): raise NotImplementedError()
    def getCurrState(self): raise NotImplementedError()
    def getLegalNextMoves(self): raise NotImplementedError()
    def parseMove(self, inputStr): raise NotImplementedError()
    def doMove(self, move): raise NotImplementedError()
    def getNextPlayer(self): raise NotImplementedError()
    def prettyPrint(self, policy = None, value = None): raise NotImplementedError()

TrainingParams = collections.namedtuple(
    "TrainingParams",
    ["bundleCount", "gamesPerBundle", "batchSize", "epochsPerStage", "challengesPerStage"])
assert issubclass(TrainingParams, tuple)

class GameBase:
    def getGameName(): raise NotImplementedError()
    def getTrainingParams(): raise NotImplementedError()
    def mtcsSimCount(): raise NotImplementedError()
    def getPolicyCount(): raise NotImplementedError()
    def toPolicyIndex(move): raise NotImplementedError()
    def toMove(pi): raise NotImplementedError()
    def createNewGameBoard(): raise NotImplementedError()
    def createNewPredictionModel(): raise NotImplementedError()
    def augmentTrainingData(states, policies, values): raise NotImplementedError()


def softmaxWithTemperature(policy, tau):
    assert isinstance(policy, np.ndarray)
    policy = np.log(policy + 1e-10) / tau
    policy = np.exp(policy - np.max(policy))
    return policy / np.sum(policy)

def weightedRandom(distribution):
    assert isinstance(distribution, np.ndarray)
    return np.random.choice(distribution.size, p = distribution)


class TreeNode:
    def __init__(self, parent, p):
        assert isinstance(parent, TreeNode) or (parent is None)
        self.parent = parent
        self.p = p
        self.n = 0
        self.w = 0
        self.q = 0
        self.children = None

    def getU(self, cpuct):
        sumN = self.n if self.parent == None else self.parent.n
        if sumN == 0: sumN = 1
        return cpuct * self.p * math.sqrt(sumN) / (1 + self.n)

    def getConfidenceBound(self, cpuct):
        return self.q + self.getU(cpuct)

    def expandChildren(self, board, game, childPolicies, isExplore):
        assert isinstance(board, GameBoardBase)
        assert issubclass(game, GameBase)
        assert isinstance(childPolicies, np.ndarray)

        policies = np.array([*(childPolicies[game.toPolicyIndex(move)] for move in board.getLegalNextMoves())])
        policies = policies / np.sum(policies)

        if isExplore:
            noiseWeight = 0.25
            noise = np.random.dirichlet((0.03, 0.97), size=policies.shape)[:, 0]
            policies = policies * (1.0 - noiseWeight) + noise * noiseWeight
            policies = policies / np.sum(policies)

        self.children = dict(zip(board.getLegalNextMoves(), (TreeNode(self, policy) for policy in policies)))

    def addNewPlay(self, nodeValue):
        self.n += 1
        self.w += nodeValue
        self.q = self.w / self.n


def mtcs(root, prevPlayer, board, iterBoard, model, game, isExplore):
    assert isinstance(root, TreeNode)
    assert isinstance(board, GameBoardBase)
    assert isinstance(iterBoard, GameBoardBase)
    assert isinstance(model, keras.models.Model)
    assert issubclass(game, GameBase)

    if root.children == None:
        policy, value = model.predict(np.array([board.getCurrState()]))
        root.expandChildren(board, game, policy[0, :], isExplore)

    #children = [*(root.children[m] for m in board.getLegalNextMoves())]
    #print(board.prettyPrint(policy[0, :], float(value)))

    rootPlayer = board.getNextPlayer()
    rootValue = root.w if rootPlayer == prevPlayer else -root.w
    cpuct = np.sqrt(game.mtcsSimCount()) / 8
    for _ in range(game.mtcsSimCount()):
        board.copyTo(iterBoard)
        iterNode = root
        currPlayers = []
        while bool(iterNode.children):
            move = max(iterBoard.getLegalNextMoves(),
                key = lambda move: iterNode.children[move].getConfidenceBound(cpuct))
            iterNode = iterNode.children[move]
            currPlayer = iterBoard.getNextPlayer()
            currPlayers.append(currPlayer)
            isGameEnded, winner = iterBoard.doMove(move)

        nodeValue = 0
        if isGameEnded:
            if winner == None:         nodeValue = 0
            elif winner == currPlayer: nodeValue = 1
            else:                      nodeValue = -1
        else:
            policy, value = model.predict(np.array([iterBoard.getCurrState()]))
            iterNode.expandChildren(iterBoard, game, policy[0, :], isExplore)
            nodeValue = -value[0, 0]

        #while iterNode != None:
            #iterNode.addNewPlay(nodeValue)
            #nodeValue = -nodeValue
            #iterNode = iterNode.parent
        for player in reversed(currPlayers):
            iterNodeValue = nodeValue if player == currPlayer else -nodeValue
            iterNode.addNewPlay(iterNodeValue)
            iterNode = iterNode.parent
        assert iterNode == root
        root.n += 1
        rootValue += nodeValue if rootPlayer == currPlayer else -nodeValue

        #print(iterBoard.prettyPrint(policy[0, :], float(value)))
        #print("NodeValue:", nodeValue if rootPlayer == currPlayer else -nodeValue)

        #if _ == 8:
        #    policy = np.zeros((game.getPolicyCount()), dtype = np.float32)
        #    for move in board.getLegalNextMoves():
        #        policy[game.toPolicyIndex(move)] = root.children[move].n
        #    policy = policy / np.sum(policy)

    policy = np.zeros((game.getPolicyCount()), dtype = np.float32)
    for move in board.getLegalNextMoves():
        policy[game.toPolicyIndex(move)] = root.children[move].n
    policy = policy / np.sum(policy)

    value = rootValue / root.n

    return (policy, value)

def selfPlay(model, game, temperature = 0.001):
    assert isinstance(model, keras.models.Model)
    assert issubclass(game, GameBase)

    states = []
    policies = []
    values = []
    players = []

    root = TreeNode(None, 1)
    board     = game.createNewGameBoard()
    iterBoard = game.createNewGameBoard()
    assert isinstance(board    , GameBoardBase)
    assert isinstance(iterBoard, GameBoardBase)

    prevPlayer = Player.P1
    isGameEnded, winner = False, None
    while not isGameEnded:
        policy, value = mtcs(root, prevPlayer, board, iterBoard, model, game, True)

        states.append(board.getCurrState())
        policies.append(policy)
        values.append(value)
        players.append(board.getNextPlayer())

        prevPlayer = board.getNextPlayer()
        policy = softmaxWithTemperature(policy, temperature)
        move = game.toMove(weightedRandom(policy))
        isGameEnded, winner = board.doMove(move)
        root = root.children[move]
        root.parent = None

    finalValues = []
    for player in players:
        if winner is None:      finalValues.append(0)
        elif winner == player:  finalValues.append(1)
        else:                   finalValues.append(-1)

    return states, policies, finalValues

def play(p1Model, p2Model, game, verbose = False, tempurature = 0.05):
    assert (p1Model is None) or isinstance(p1Model, keras.models.Model)
    assert (p2Model is None) or isinstance(p2Model, keras.models.Model)
    assert issubclass(game, GameBase)

    board     = game.createNewGameBoard()
    iterBoard = game.createNewGameBoard()
    assert isinstance(board    , GameBoardBase)
    assert isinstance(iterBoard, GameBoardBase)

    playerLookup = {
       Player.P1: (p1Model, p1Model and TreeNode(None, 1)),
       Player.P2: (p2Model, p2Model and TreeNode(None, 1))}

    prevPlayer = Player.P1
    isGameEnded, winner = False, None
    while not isGameEnded:
        player = board.getNextPlayer()
        assert isinstance(player, Player)

        model, root = playerLookup[player]
        move = None
        if model is None:
            print(board.prettyPrint())
            while move is None:
                inputLine = input(">>> Your move: ")
                parsedMove = board.parseMove(inputLine)
                if parsedMove is None: continue
                if parsedMove not in board.getLegalNextMoves(): continue
                move = parsedMove

        else:
            policy, value = mtcs(root, prevPlayer, board, iterBoard, model, game, False)
            if verbose:
                print(board.prettyPrint(policy, value))

            policy = softmaxWithTemperature(policy, tempurature)
            move = game.toMove(weightedRandom(policy))

            root = root.children[move]
            root.parent = None
            playerLookup[player] = model, root

        player = player.opposite()
        model, root = playerLookup[player]
        if model is not None:
            if root.children == None:
                policy, _ = model.predict(np.array([board.getCurrState()]))
                root.expandChildren(board, game, policy[0, :], False)

            root = root.children[move]
            root.parent = None
            playerLookup[player] = model, root

        prevPlayer = board.getNextPlayer()
        isGameEnded, winner = board.doMove(move)

    p1Won = None if winner is None else (winner == Player.P1)
    if verbose:
        print(board.prettyPrint())
        print("******************** Winner is %s" % winner)
        print()

    return p1Won


