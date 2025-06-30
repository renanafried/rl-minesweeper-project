import numpy as np

def rotate_board(board, k):
    # Rotate a flattened 5x5 board k times counter-clockwise
    return np.rot90(board.reshape(5, 5), k).flatten()

def rotate_action(action, k, size=5):
    # Rotate an action index (flattened position) k times on a size x size board
    i, j = divmod(action, size)
    for _ in range(k):
        i, j = j, size - 1 - i  # Rotate 90 degrees counter-clockwise
    return i * size + j
