import numpy as np

def rotate_board(board, k):
    return np.rot90(board.reshape(5, 5), k).flatten()

def rotate_action(action, k, size=5):
    i, j = divmod(action, size)
    for _ in range(k):
        i, j = j, size - 1 - i
    return i * size + j
