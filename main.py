import sys
import os
import shutil


isDebug = False
if sys.platform == "win32":
    isDebug = True

argv = sys.argv
if isDebug:
    #gameName = "c4"
    ##argv = ["main.py", gameName, "train", "8"]
    #argv = ["main.py", gameName, "test"]
    pass

#isDebug = False #################
scriptFileName = os.path.basename(argv[0])
currDir = os.path.dirname(os.path.realpath(__file__))

print()
print("Usage:")
print("    python %s <game> train [<timeInHours>]" % scriptFileName)
print("    python %s <game> test [(c|h)(c|h)]" % scriptFileName)
print()

isTrain = None
trainHours = 65535
p1Human = None
p2Human = None
if argv[2] == "train":
    isTrain = True
    if len(argv) >= 4:
        trainHours = float(argv[3])

elif argv[2] == "test":
    isTrain = False
    if len(argv) < 4:
        players = input(">>> Players (c|h)(c|h): ")
        players = "".join(players.split())  # remove whitespace
        argv.append(players)
    def parseIsHuman(playerChar):
        if   playerChar.lower() == "c": return False
        elif playerChar.lower() == "h": return True
        else: raise ValueError("Invalid player %s" % playerChar)
    p1Human = parseIsHuman(argv[3][0])
    p2Human = parseIsHuman(argv[3][1])

else:
    raise ValueError("Invalid mode %s" % argv[2])


import time
import collections
import numpy as np
import keras
import mtcs
mtcs.isDebug = isDebug

cudaVisibleDevices = "0" if isTrain else "1"
if argv[1] in ("ttt", "tictactoe"):
    from tictactoe import TicTacToe as game
elif argv[1] in ("c4", "connect4"):
    from connect4 import Connect4 as game
elif argv[1] in ("rev", "reversi"):
    from reversi import Reversi as game
    cudaVisibleDevices = "1" if isTrain else "0"
else:
    raise KeyError("Game %s not found" % argv[1])

assert issubclass(game, mtcs.GameBase)
gameName = game.getGameName()
selfPlayDataPath        = os.path.join(currDir, "_" + gameName + ".selfPlay.npz")
backupSelfPlayDataPath  = os.path.join(currDir, "_" + gameName + ".selfPlay.backup.npz")
modelPath               = os.path.join(currDir, "_" + gameName + ".model.h5")
backupModelPath         = os.path.join(currDir, "_" + gameName + ".model.backup.h5")
bestModelPath           = os.path.join(currDir, "_" + gameName + ".bestModel.h5")
backupBestModelPath     = os.path.join(currDir, "_" + gameName + ".bestModel.backup.h5")
if sys.platform == "linux":
    os.environ["CUDA_VISIBLE_DEVICES"] = cudaVisibleDevices

def formatTime(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

class Perf:
    def __init__(self, text, length):
        self.text = text
        self.length = length
        self.startTime = time.perf_counter()
        print("%s: 0 / %g" % (text, length), end = "")

    def progress(self, value, text = None):
        print("\r%s: %g / %g, " % (self.text, value, self.length), end = "")

        if text is not None:
            print("%s, " % text, end = "")

        etr = "nan" if value <= 0 else formatTime((time.perf_counter() - self.startTime) / value * (self.length - value))
        print("ETR = %s" % etr, end = "")

    def finish(self, text = None):
        print("\r%s completed, " % self.text, end = "")

        if text is not None:
            print("%s, " % text, end = "")

        elapsed = formatTime(time.perf_counter() - self.startTime)
        print("Elapsed = %s" % elapsed)


if isTrain:
    model = None
    if os.path.isfile(modelPath):
        print("Loading existing model")
        loadedModel = keras.models.load_model(modelPath)
        model = game.createNewPredictionModel()
        model.set_weights(loadedModel.get_weights())

    else:
        print("Creating new model")
        model = game.createNewPredictionModel()

    print("****************************************")
    print("Start training for %g hours" % trainHours)
    params = game.getTrainingParams()    # stepsPerEpoch is bundleCount * gamesPerBundle * avgStatesPerGame / batchSize
    endTime = time.perf_counter() + trainHours * 3600
    epochCount = 0
    stageCount = 0

    def createBundle(model, gamesPerBundle):
        statesBatch = []; policiesBatch = []; valuesBatch = []
        perf = Perf("Create self-play data bundle", gamesPerBundle)
        for i in range(gamesPerBundle):
            states, policies, values = mtcs.selfPlay(model, game, 1)
            statesBatch.extend(states)
            policiesBatch.extend(policies)
            valuesBatch.extend(values)
            perf.progress(i + 1)
        perf.finish()
        return (statesBatch, policiesBatch, valuesBatch)

    bestModel = None    # bestModel is not compiled
    if os.path.isfile(bestModelPath):
        print("Loading existing best model")
        bestModel = keras.models.load_model(bestModelPath)
    else:
        print("Duplicating new best model")
        bestModel = keras.models.clone_model(model)
        bestModel.set_weights(model.get_weights())

    selfPlayBundleQueue = collections.deque()
    if os.path.isfile(selfPlayDataPath):
        print("Loading existing %d bundles" % params.bundleCount)
        npz = np.load(selfPlayDataPath)
        for i in range(params.bundleCount):
            selfPlayBundleQueue.append((npz["s%d" % i], npz["p%d" % i], npz["v%d" % i]))
        if ("stageCount" not in npz) and (game.getGameName() == "tictactoe"):
            stageCount = 100
        else:
            stageCount = int(npz["stageCount"])

    else:
        print("Initializing %d bundles" % params.bundleCount)
        for _ in range(params.bundleCount):
            selfPlayBundleQueue.append(createBundle(bestModel, params.gamesPerBundle // 2))

    while time.perf_counter() < endTime:
        print("****************************************")
        print("Training stage %d, saving progress ... " % stageCount, end = "")
        npzEntries = {"stageCount": stageCount}
        for i in range(len(selfPlayBundleQueue)):
            states, policies, values = selfPlayBundleQueue[i]
            npzEntries.update({("s%d" % i): states, ("p%d" % i): policies, ("v%d" % i): values})
        try:
            np.savez(selfPlayDataPath, **npzEntries)
            shutil.copyfile(selfPlayDataPath, backupSelfPlayDataPath)
        except BaseException as err:
            print(err)
        print("done", flush = True)

        print("Preparing training set ... ", end = "")
        states   = np.array([state  for (states, policies, values) in selfPlayBundleQueue for state  in states])
        policies = np.array([policy for (states, policies, values) in selfPlayBundleQueue for policy in policies])
        values   = np.array([value  for (states, policies, values) in selfPlayBundleQueue for value  in values])
        print("(%d, %d, %d) samples prepared" % (len(states), len(policies), len(values)))
        states, policies, values = game.augmentTrainingData(states, policies, values)
        print(" ... augmented to (%d, %d, %d) samples" % (len(states), len(policies), len(values)))

        print("Train model, %d epochs" % params.epochsPerStage)
        class TrainingCallback(keras.callbacks.Callback):
            def formatLogs(self, logs = None):
                if logs is not None: self.logs = logs
                outstr = ", ".join(["%s: %0.3f" % pair for pair in sorted(self.logs.items())])
                return outstr
            def set_params(self, params):
                self.metrics = params["metrics"]
                self.epochs = params["epochs"]
                self.samples = params["samples"]
                self.perf = Perf("Train", self.epochs - epochCount)
                return super().set_params(params)
            def on_epoch_end(self, epoch, logs = None):
                self.perf.progress(epoch + 1 - epochCount, self.formatLogs(logs))
                return super().on_epoch_end(epoch, logs)
            def on_train_end(self, logs = None):
                self.perf.finish(self.formatLogs())
                return super().on_train_end(logs)
        model.fit(
            states, [policies, values], batch_size = params.batchSize,
            epochs = epochCount + params.epochsPerStage, initial_epoch = epochCount,
            callbacks = [TrainingCallback()], verbose = 0)
        epochCount += params.epochsPerStage
        try:
            model.save(modelPath, overwrite=True)
            shutil.copyfile(modelPath, backupModelPath)
        except BaseException as err:
            print(err)

        print("Challenge current best model")
        perf = Perf("Challenge", params.challengesPerStage)
        winCount = 0; loseCount = 0; drawCount = 0
        for i in range(params.challengesPerStage // 2):
            result = mtcs.play(model, bestModel, game)
            if result is None: drawCount += 1
            elif result:       winCount  += 1
            else:              loseCount += 1
            perf.progress(i + 1, "%d won, %d lost, %d draws" % (winCount, loseCount, drawCount))
        for i in range(params.challengesPerStage // 2, params.challengesPerStage):
            result = mtcs.play(bestModel, model, game)
            if result is None: drawCount += 1
            elif not result:   winCount  += 1
            else:              loseCount += 1
            perf.progress(i + 1, "%d won, %d lost, %d draws" % (winCount, loseCount, drawCount))
        perf.finish("%d won, %d lost, %d draws" % (winCount, loseCount, drawCount))

        challengeSucceeded = (winCount > 0) and (winCount / (winCount + loseCount) >= 0.55)
        if challengeSucceeded:
            print("Challenge succeeded")
            bestModel.set_weights(model.get_weights())
        else:
            print("Challenge failed")

        if challengeSucceeded or not os.path.isfile(bestModelPath):
            try:
                bestModel.save(bestModelPath, overwrite=True)
                shutil.copyfile(bestModelPath, backupBestModelPath)
            except BaseException as err:
                print(err)

        print("Update new self play bundle")
        selfPlayBundleQueue.pop()
        selfPlayBundleQueue.appendleft(createBundle(bestModel, params.gamesPerBundle))

        stageCount += 1

else:
    model = None
    modelPath = bestModelPath
    if os.path.isfile(modelPath):
        print("Loading existing best model")
        model = keras.models.load_model(modelPath)

    else:
        print("Creating new model")
        model = game.createNewPredictionModel()

    p1 = None if p1Human else model
    p2 = None if p2Human else model

    playAgain = True
    while playAgain:
        print("****************************************")
        print("Game start")
        mtcs.play(p1, p2, game, True)

        print("****************************************")
        playAgain = None
        while playAgain is None:
            inputLine = input(">>> Play again? ([y]/n) ").strip().lower()
            if inputLine in ("", "y", "yes"):
                playAgain = True
            elif inputLine in ("n", "no"):
                playAgain = False


        



