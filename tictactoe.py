import typing
import math
import re
import numpy as np
import keras
import mtcs


sideCount = 3
maxMoves = sideCount * sideCount

def toPolicyIndex(i, j): return i * sideCount + j
def toMove(pi): return pi // sideCount, pi % sideCount

class TicTacToeBoard(mtcs.GameBoardBase):
    def __init__(self):
        self.moves = np.zeros((sideCount, sideCount), dtype = int)
        self.moveCount = 0
        self.nextPlayer = mtcs.Player.P1

    def copyTo(self, other):
        assert isinstance(other, TicTacToeBoard)
        np.copyto(other.moves, self.moves)
        other.moveCount = self.moveCount
        other.nextPlayer = self.nextPlayer

    def getCurrState(self):
        p1Index = 0 if self.nextPlayer == mtcs.Player.P1 else 1
        p2Index = 1 - p1Index

        tensor = np.zeros((sideCount, sideCount, 2), dtype = np.float32)
        tensor[:, :, p1Index] = self.moves * (self.moves > 0)
        tensor[:, :, p2Index] = -self.moves * (self.moves < 0)
        return tensor

    def getLegalNextMoves(self):
        for i in range(sideCount):
            for j in range(sideCount):
                if self.moves[i, j] == 0:
                    yield i, j

    def parseMove(self, inputStr):
        return re.match("""^\s*\d+\s*,\s*\d+\s*$""", inputStr) and eval(inputStr)

    def doMove(self, move):
        i, j = move
        nextPlayerValue = self.nextPlayer.toStateValue()
        self.moves[i, j] = nextPlayerValue
        playerWon = all([self.moves[i, j] == nextPlayerValue for i in range(sideCount)])
        playerWon |= all([self.moves[i, j] == nextPlayerValue for j in range(sideCount)])
        playerWon |= ((i == j) and
            all([self.moves[i, i] == nextPlayerValue for i in range(sideCount)]))
        playerWon |= ((i + j == sideCount - 1) and
            all([self.moves[i, sideCount - 1 - i] == nextPlayerValue for i in range(sideCount)]))

        self.moveCount += 1

        if playerWon:
            return True, self.nextPlayer
        elif self.moveCount == maxMoves:
            return True, None

        self.nextPlayer = self.nextPlayer.opposite()
        return False, None

    def getNextPlayer(self): return self.nextPlayer

    def prettyPrint(self, policy = None, value = None):
        assert (policy is None) or isinstance(policy, np.ndarray)

        firstLine = "******************** Next is "
        if self.nextPlayer == mtcs.Player.P1: firstLine += "O"
        else:                                 firstLine += "X"

        if value is not None:
            firstLine += ", Value = %f" % value

        lines = [firstLine]

        moveStrs = ["  .   ", "  O   ", "  X   "]
        for i in range(sideCount):
            line = ""
            for j in range(sideCount):
                move = self.moves[i, j]
                if (move == 0) and (policy is not None):
                    line += "(%3d) " % (round(policy[toPolicyIndex(i, j)] * 999.4))
                else:
                    line += moveStrs[move]
            lines.append(line)

        return "\r\n".join(lines)

def createNewModel():
    input = keras.layers.Input(shape = (sideCount, sideCount, 2), dtype = np.float32)

    l2const = 1e-4
    layer = input
    layer = keras.layers.Flatten()(layer)
    layer = keras.layers.Dense(128, kernel_regularizer = keras.regularizers.l2(l2const))(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Activation("relu")(layer)
    for _ in range(8 if not mtcs.isDebug else 4):
        res = layer
        layer = keras.layers.Dense(128, kernel_regularizer = keras.regularizers.l2(l2const))(layer)
        layer = keras.layers.Activation("relu")(layer)
        layer = keras.layers.Dense(128, kernel_regularizer = keras.regularizers.l2(l2const))(layer)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.Add()([layer, res])
        layer = keras.layers.Activation("relu")(layer)

    vhead = layer
    vhead = keras.layers.Dense(36, kernel_regularizer = keras.regularizers.l2(l2const))(vhead)
    vhead = keras.layers.Activation("relu")(vhead)
    vhead = keras.layers.Dense(1)(vhead)
    vhead = keras.layers.Activation("tanh", name = "vh")(vhead)

    phead = layer
    phead = keras.layers.Dense(64, kernel_regularizer = keras.regularizers.l2(l2const))(phead)
    phead = keras.layers.BatchNormalization()(phead)
    phead = keras.layers.Activation("relu")(phead)
    phead = keras.layers.Dense(9)(phead)
    phead = keras.layers.Activation("softmax", name = "ph")(phead)

    model = keras.models.Model(inputs = [input], outputs = [phead, vhead])
    model.compile(
        optimizer = keras.optimizers.Adadelta(),
        loss = [keras.losses.categorical_crossentropy, keras.losses.mean_squared_error],
        loss_weights = [0.5, 0.5],
        metrics=["accuracy"])
    return model

def augmentTrainingData(states, policies, values):
    assert isinstance(states, np.ndarray)
    assert isinstance(policies, np.ndarray)
    assert isinstance(values, np.ndarray)

    states = np.concatenate([
        states,                states[:, ::-1, ::-1, :],
        states[:, ::-1, :, :], states[:, :, ::-1, :]])
    states = np.concatenate([
        states, np.rot90(states, axes = (1, 2))])

    policies = np.reshape(policies, (len(policies), sideCount, sideCount))
    policies = np.concatenate([
        policies,             policies[:, ::-1, ::-1],
        policies[:, ::-1, :], policies[:, :, ::-1]])
    policies = np.concatenate([
        policies, np.rot90(policies, axes = (1, 2))])
    policies = np.reshape(policies, (len(policies), maxMoves))

    values = np.concatenate([values, values, values, values])
    values = np.concatenate([values, values])

    return states, policies, values

trainingParams = mtcs.TrainingParams(
    bundleCount         = 2   if not mtcs.isDebug else 2,    # Delete npz if modified
    gamesPerBundle      = 36  if not mtcs.isDebug else 8,
    batchSize           = 32  if not mtcs.isDebug else 16,
    epochsPerStage      = 32  if not mtcs.isDebug else 12,
    challengesPerStage  = 10  if not mtcs.isDebug else 10)

class TicTacToe(mtcs.GameBase):
    def getGameName(): return "tictactoe"
    def getTrainingParams(): return trainingParams
    def mtcsSimCount(): return 128 if not mtcs.isDebug else 32
    def getPolicyCount(): return maxMoves
    def toPolicyIndex(move): return toPolicyIndex(*move)
    def toMove(pi): return toMove(pi)
    def createNewGameBoard(): return TicTacToeBoard()
    def createNewPredictionModel(): return createNewModel()
    def augmentTrainingData(states, policies, values): return augmentTrainingData(states, policies, values)


