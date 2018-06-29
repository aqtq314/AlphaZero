# AlphaZero
A Keras implementation of Google's AlphaZero project on tic-tac-toe, connect-four and reversi. One can add other games by just providing rules and hyperparameters for training.

## Dependencies
python >= 3.5
numpy
Keras with TensorFlow backend

## Usage
```
python main.py <game> train [<timeInHours>]
python main.py <game> test [(c|h)(c|h)]
```

`<game>` can be one of `tictactoe`, `connect4`, `reversi` or their short forms `ttt`, `c4`, `rev`

`<timeInHours>`: training time in hours

`(c|h)(c|h)`: `c` for AI, `h` for human, in their respective play orders. For example, `ch` means AI plays first and human plays second.

## Test Mode

The human player will need to input the coordinates for each move on board, first row then column, separated by commas. For example, `0,0` indicates top-left corner, and `7,0` indicates bottom-left corner if on Reversi. For connect-four, the user will input column index.

The AI player will load model from file named `_<game>.bestModel.h5` in the same directory as the script file.
