# SudokuSolverUsingCNN
A simple CNN implementation using Keras to solve sudoku 

## Setup
We train a model with only 3 convolutions of 32, 16 and 1 filters each. The purpose of the last convolution is to predict the soluced sudoku.
The training is made with an Adam optimizer and it try to minimize an MSE Loss function, however it is also evaluated with the mae.
The training base is randomly create and soluced with a classical algorithme which takes quite a longer time than the CNN one. It represents most of the algorithme time execution and can surely be improved (even though generating and solving something like 30 000 sudokus need obviously a bit of time)
The model is trained with 3 differents difficulties which are each composed of 10 000 samples and the weights are resets when the difficulty change.

## Performances
An interesting approach is to check if the difficulty impact the training duration and the stability of the learning.
However we show that both difficulties converges in a reasonnable amount of time (and epochs) which is quite a good result.
Nevertheless, MSE might not be a really good metrics evaluation. We propose to create a new one similar to accuracy. The aim of that metric is to consider float predictions as integer and evaluate the percentage of them correctly placed among the grid by the model.
