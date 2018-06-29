import typing
import math
import re
import numpy as np
import keras
import mtcs


rowCount, colCount = 6, 7
maxMoves = rowCount * colCount

#def toPolicyIndex(i, j): return i * colCount + j
#def toCoordinates(ci): return ci // colCount, ci % colCount

class Connect4Board(mtcs.GameBoardBase):
    def __init__(self):
        self.moves = np.zeros((rowCount, colCount), dtype = int)
        self.moveCount = 0
        self.nextPlayer = mtcs.Player.P1

    def copyTo(self, other):
        assert isinstance(other, Connect4Board)
        np.copyto(other.moves, self.moves)
        other.moveCount = self.moveCount
        other.nextPlayer = self.nextPlayer

    def getCurrState(self):
        p1Index = 0 if self.nextPlayer == mtcs.Player.P1 else 1
        p2Index = 1 - p1Index

        tensor = np.zeros((rowCount, colCount, 2), dtype = np.float32)
        tensor[:, :, p1Index] = self.moves * (self.moves > 0)
        tensor[:, :, p2Index] = -self.moves * (self.moves < 0)
        return tensor

    def getLegalNextMoves(self):
        colNotFull = np.sum(np.abs(self.moves), axis = 0) < rowCount
        return (i for i in range(colCount) if colNotFull[i])

    def parseMove(self, inputStr):
        try:
            return int(inputStr)
        except ValueError:
            return None

    def doMove(self, move):
        j = move
        i = rowCount - 1
        while self.moves[i, j] != 0:
            i -= 1

        nextPlayerValue = self.nextPlayer.toStateValue()
        self.moves[i, j] = nextPlayerValue

        playerWon = False
        for (di, dj) in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            consCount = 0
            for inc in range(-3, 4):
                i2, j2 = i + di * inc, j + dj * inc
                if i2 < 0 or i2 >= rowCount: continue
                if j2 < 0 or j2 >= colCount: continue
                if self.moves[i2, j2] != nextPlayerValue:
                    consCount = 0
                else:
                    consCount += 1
                    if consCount == 4:
                        playerWon = True
                        break
            if playerWon:
                break

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

        if policy is not None:
            lines.append(" ".join((("%3d" % (int(p * 999.99)) if p > 0 else "   ") for p in policy)))

        moveStrs = [" . ", " O ", " X "]
        for i in range(rowCount):
            lines.append(" ".join(moveStrs[self.moves[i, j]] for j in range(colCount)))

        return "\r\n".join(lines)

def createNewModel():
    input = keras.layers.Input((rowCount, colCount, 2), dtype = np.float32)

    l2const = 1e-4
    layer = input
    layer = keras.layers.ZeroPadding2D((2, 2))(layer)
    layer = keras.layers.Conv2D(96, (4, 4), padding = "valid", kernel_regularizer = keras.regularizers.l2(l2const))(layer)
    layer = keras.layers.Activation("relu")(layer)
    layer = keras.layers.Conv2D(96, (2, 2), padding = "valid", kernel_regularizer = keras.regularizers.l2(l2const))(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Activation("relu")(layer)
    for _ in range(16 if not mtcs.isDebug else 6):
        res = layer
        layer = keras.layers.ZeroPadding2D((2, 2))(layer)
        layer = keras.layers.Conv2D(96, (4, 4), padding = "valid", kernel_regularizer = keras.regularizers.l2(l2const))(layer)
        layer = keras.layers.Activation("relu")(layer)
        layer = keras.layers.Conv2D(96, (2, 2), padding = "valid", kernel_regularizer = keras.regularizers.l2(l2const))(layer)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.Add()([layer, res])
        layer = keras.layers.Activation("relu")(layer)

    vhead = layer
    vhead = keras.layers.Conv2D(1, (1, 1), kernel_regularizer = keras.regularizers.l2(l2const))(vhead)
    vhead = keras.layers.BatchNormalization()(vhead)
    vhead = keras.layers.Activation("relu")(vhead)
    vhead = keras.layers.Flatten()(vhead)
    vhead = keras.layers.Dense(32)(vhead)
    vhead = keras.layers.Activation("relu")(vhead)
    vhead = keras.layers.Dense(1)(vhead)
    vhead = keras.layers.Activation("tanh", name = "vh")(vhead)

    phead = layer
    phead = keras.layers.Conv2D(7, (6, 1), padding = "valid", kernel_regularizer = keras.regularizers.l2(l2const))(phead)
    phead = keras.layers.Activation("relu")(phead)
    phead = keras.layers.Conv2D(1, (1, 1), padding = "valid", kernel_regularizer = keras.regularizers.l2(l2const))(phead)
    phead = keras.layers.BatchNormalization()(phead)
    phead = keras.layers.Activation("relu")(phead)
    phead = keras.layers.Flatten()(phead)
    phead = keras.layers.Dense(7)(phead)
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

    states = np.concatenate([states, states[:, :, ::-1, :]])

    policies = np.concatenate([policies, policies[:, ::-1]])

    values = np.concatenate([values, values])

    return states, policies, values

trainingParams = mtcs.TrainingParams(
    bundleCount         = 3   if not mtcs.isDebug else 1,    # Delete npz if modified
    gamesPerBundle      = 60  if not mtcs.isDebug else 1,
    batchSize           = 32  if not mtcs.isDebug else 16,
    epochsPerStage      = 256 if not mtcs.isDebug else 12,
    challengesPerStage  = 16  if not mtcs.isDebug else 6)

class Connect4(mtcs.GameBase):
    def getGameName(): return "connect4"
    def getTrainingParams(): return trainingParams
    def mtcsSimCount(): return 240 if not mtcs.isDebug else 120
    def getPolicyCount(): return colCount
    def toPolicyIndex(move): return move
    def toMove(pi): return pi
    def createNewGameBoard(): return Connect4Board()
    def createNewPredictionModel(): return createNewModel()
    def augmentTrainingData(states, policies, values): return augmentTrainingData(states, policies, values)


