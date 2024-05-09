import random
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def is_valid(grid, row, col, num):
        # Vérifier la ligne
        if num in grid[row]:
            return False

        # Vérifier la colonne
        if num in [grid[i][col] for i in range(9)]:
            return False

        # Vérifier le carré 3x3
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if grid[i][j] == num:
                    return False

        return True

def solve_sudoku(grid):
        empty_cell = find_empty_cell(grid)
        if not empty_cell:
            return True

        row, col = empty_cell

        for num in range(1, 10):
            if is_valid(grid, row, col, num):
                grid[row][col] = num
                if solve_sudoku(grid):
                    return True
                grid[row][col] = 0

        return False

def find_empty_cell(grid):
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    return (i, j)
        return None

def generate_sudoku(difficulty):
        grid = np.zeros((9, 9))
        solve_sudoku(grid)

        # Déterminer le nombre de cellules à vider en fonction de la difficulté
        if difficulty == "easy":
            empty_cells = random.randint(40, 50)
        elif difficulty == "medium":
            empty_cells = random.randint(50, 60)
        elif difficulty == "hard":
            empty_cells = random.randint(60, 70)
        else:
            raise ValueError("La difficulté doit être 'easy', 'medium' ou 'hard'.")

        for _ in range(empty_cells):
            row, col = random.randint(0, 8), random.randint(0, 8)
            if grid[row][col] != 0:
                grid[row][col] = 0
        return grid

def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))

print("definitions terminés")

"""
model = keras.Sequential()
model.add(keras.Input(shape=(9, 9, 1)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same'))

model.compile(optimizer='Adam', loss='mse', metrics=['mae', 'mse'])
"""
print("modele compilé")

def generate_base(nb) :
    X = []
    y = []
    difficulties = ["easy", "medium", 'hard']
    for j in range(len(difficulties)) :
        diff = difficulties[j]
        Xi = []
        yi = []
        for i in range(nb):
            base = generate_sudoku(diff)
            Xi.append(base)
            solution = base.copy()
            solve_sudoku(solution)
            yi.append(solution)
        X.append(Xi)
        y.append(yi)
    return X, y

X, y = generate_base(10000)
X = np.array(X)
X = np.expand_dims(X, axis=-1)
y = np.array(y)
y = np.expand_dims(y, axis=-1)

print("base générée")

history_list = []
for i in range(len(X)):
    X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], test_size=0.2, random_state=42)
    model = keras.Sequential()
    model.add(keras.Input(shape=(9, 9, 1)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same'))
    model.compile(optimizer='Adam', loss='mse', metrics=['mae', 'mse'])
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))
    history_list.append(history)
    name = "model"+str(i)+'.h5'
    model.save(name)



for i in range(len(history_list)) :
    plt.plot(history_list[i].history['mse'], label='Model'+str(i))
    

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MSE par difficulté')
plt.legend()
plt.ylim(0, 10)
plt.savefig('mse.png')  # Enregistrer la figure au format PNG
plt.close()

for i in range(len(history_list)) :
    plt.plot(history_list[i].history['mae'], label='Model'+str(i))
    

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MAE par difficulté')
plt.legend()
plt.ylim(0, 10)

plt.savefig('mae.png')  # Enregistrer la figure au format PNG
plt.close()


