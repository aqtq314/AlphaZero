import typing
import math
import re
import numpy as np
import keras
import mtcs


sideCount = 8
policyCount = sideCount * sideCount
directions = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if not (i == 0 and j == 0)]
def isCoordsInRange(i, j):
    return i >= 0 and i < sideCount and j >= 0 and j < sideCount;

def toPolicyIndex(i, j): return i * sideCount + j
def toMove(pi): return pi // sideCount, pi % sideCount

class ReversiBoard(mtcs.GameBoardBase):
    def __init__(self):
        self.moves = np.zeros((sideCount, sideCount), dtype = int)
        self.moves[3:5, 3:5] = np.array([[-1, 1], [1, -1]])
        self.contourCells = set([(2, 2), (2, 3), (2, 4), (2, 5), (3, 2), (3, 5), (4, 2), (4, 5), (5, 2), (5, 3), (5, 4), (5, 5)])
        self.legalNextMoves = [(2, 3), (3, 2), (4, 5), (5, 4)]
        self.nextPlayer = mtcs.Player.P1

    def copyTo(self, other):
        assert isinstance(other, ReversiBoard)
        np.copyto(other.moves, self.moves)
        other.contourCells.clear()
        other.contourCells |= self.contourCells
        other.legalNextMoves.clear()
        other.legalNextMoves.extend(self.legalNextMoves)
        other.nextPlayer = self.nextPlayer

    def getCurrState(self):
        p1Index = 0 if self.nextPlayer == mtcs.Player.P1 else 1
        p2Index = 1 - p1Index

        tensor = np.zeros((sideCount, sideCount, 2), dtype = np.float32)
        tensor[:, :, p1Index] = self.moves * (self.moves > 0)
        tensor[:, :, p2Index] = -self.moves * (self.moves < 0)
        return tensor

    def getCoordsInDirection(self, start, direction):
        i, j = start
        di, dj = direction
        for inc in range(1, 8):
            i2, j2 = i + inc * di, j + inc * dj
            if not isCoordsInRange(i2, j2): break
            yield i2, j2

    def getLegalNextMoves(self):
        return self.legalNextMoves

    def parseMove(self, inputStr):
        return re.match("""^\s*\d+\s*,\s*\d+\s*$""", inputStr) and eval(inputStr)

    def doMove(self, move):
        i, j = move
        nextPlayerValue = self.nextPlayer.toStateValue()

        self.moves[i, j] = nextPlayerValue
        for direction in directions:
            met = False
            revCoords = []
            for revCoord in self.getCoordsInDirection(move, direction):
                revValue = self.moves[revCoord]
                if revValue == 0:
                    break
                elif revValue == nextPlayerValue:
                    met = True
                    break
                else:
                    revCoords.append(revCoord)
            if met:
                for revCoord in revCoords:
                    self.moves[revCoord] = nextPlayerValue

        self.contourCells.remove(move)
        for (di, dj) in directions:
            i2, j2 = i + di, j + dj
            if isCoordsInRange(i2, j2) and self.moves[i2, j2] == 0:
                self.contourCells.add((i2, j2))

        generateNextMovesCount = 0
        self.legalNextMoves.clear()
        while True:
            self.nextPlayer = self.nextPlayer.opposite()
            nextPlayerValue = -nextPlayerValue
            for cellCoord in self.contourCells:
                for direction in directions:
                    met = False
                    revCount = 0
                    for revCoord in self.getCoordsInDirection(cellCoord, direction):
                        revValue = self.moves[revCoord]
                        if revValue == 0:
                            break
                        elif revValue == nextPlayerValue:
                            met = True
                            break
                        else:
                            revCount += 1
                    if met and revCount > 0:
                        self.legalNextMoves.append(cellCoord)
                        break

            if len(self.legalNextMoves) > 0:
                return False, None

            elif generateNextMovesCount == 0:
                generateNextMovesCount += 1

            else:
                p1Count = np.sum(self.moves > 0)
                p2Count = np.sum(self.moves < 0)
                if p1Count == p2Count:
                    return True, None
                elif p1Count > p2Count:
                    return True, mtcs.Player.P1
                else:
                    return True, mtcs.Player.P2

    def getNextPlayer(self): return self.nextPlayer

    def prettyPrint(self, policy = None, value = None):
        assert (policy is None) or isinstance(policy, np.ndarray)

        firstLine = "******************************** Next is "
        if self.nextPlayer == mtcs.Player.P1: firstLine += "O"
        else:                                 firstLine += "X"

        if value is not None:
            firstLine += ", Value = %f" % value

        lines = [firstLine]

        moveStrs = [" .  ", " O  ", " X  "]
        for i in range(sideCount):
            line = ""
            for j in range(sideCount):
                moveValue = self.moves[i, j]
                if moveValue == 0 and policy is not None and policy[toPolicyIndex(i, j)] > 0:
                    line += "%3d " % (int(policy[toPolicyIndex(i, j)] * 999.99))
                else:
                    line += moveStrs[moveValue]
            lines.append(line)

        return "\r\n".join(lines)

def createNewModel():
    input = keras.layers.Input(shape = (sideCount, sideCount, 2), dtype = np.float32)

    l2const = 1e-4
    layer = input
    layer = keras.layers.Conv2D(128, (3, 3), padding = "same", kernel_regularizer = keras.regularizers.l2(l2const))(layer)
    layer = keras.layers.BatchNormalization()(layer)
    layer = keras.layers.Activation("relu")(layer)
    for _ in range(18 if not mtcs.isDebug else 6):
        res = layer
        layer = keras.layers.Conv2D(128, (3, 3), padding = "same", kernel_regularizer = keras.regularizers.l2(l2const))(layer)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.Activation("relu")(layer)
        layer = keras.layers.Conv2D(128, (3, 3), padding = "same", kernel_regularizer = keras.regularizers.l2(l2const))(layer)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.Add()([layer, res])
        layer = keras.layers.Activation("relu")(layer)

    vhead = layer
    vhead = keras.layers.Conv2D(1, (1, 1), kernel_regularizer = keras.regularizers.l2(l2const))(vhead)
    vhead = keras.layers.BatchNormalization()(vhead)
    vhead = keras.layers.Activation("relu")(vhead)
    vhead = keras.layers.Flatten()(vhead)
    vhead = keras.layers.Dense(64, kernel_regularizer = keras.regularizers.l2(l2const))(vhead)
    vhead = keras.layers.Activation("relu")(vhead)
    vhead = keras.layers.Dense(1)(vhead)
    vhead = keras.layers.Activation("tanh", name = "vh")(vhead)

    phead = layer
    phead = keras.layers.Conv2D(2, (1, 1), kernel_regularizer = keras.regularizers.l2(l2const))(phead)
    phead = keras.layers.BatchNormalization()(phead)
    phead = keras.layers.Activation("relu")(phead)
    phead = keras.layers.Flatten()(phead)
    phead = keras.layers.Dense(policyCount)(phead)
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
    policies = np.reshape(policies, (len(policies), policyCount))

    values = np.concatenate([values, values, values, values])
    values = np.concatenate([values, values])

    return states, policies, values

trainingParams = mtcs.TrainingParams(
    bundleCount         = 3   if not mtcs.isDebug else 1,    # Delete npz if modified
    gamesPerBundle      = 48  if not mtcs.isDebug else 1,
    batchSize           = 32  if not mtcs.isDebug else 16,
    epochsPerStage      = 64  if not mtcs.isDebug else 4,
    challengesPerStage  = 16  if not mtcs.isDebug else 6)

class Reversi(mtcs.GameBase):
    def getGameName(): return "reversi"
    def getTrainingParams(): return trainingParams
    def mtcsSimCount(): return 288 if not mtcs.isDebug else 120
    def getPolicyCount(): return policyCount
    def toPolicyIndex(move): return toPolicyIndex(*move)
    def toMove(pi): return toMove(pi)
    def createNewGameBoard(): return ReversiBoard()
    def createNewPredictionModel(): return createNewModel()
    def augmentTrainingData(states, policies, values): return augmentTrainingData(states, policies, values)


