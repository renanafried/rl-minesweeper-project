import numpy as np

def get_safe_moves(board):
    """
    Given the current visible board (–2 = unopened, ≥0 = opened numbers, –3 = flagged),
    return all moves (as (i,j) tuples) that are provably safe:
      – any unopened neighbor of a 0-cell,
      – any unopened neighbor when all mines around a number-cell have already been flagged.
    """
    H, W = board.shape
    safe_moves = set()

    def neighbors(i, j):
        # Return all valid neighbors (excluding the cell itself)
        return [
            (ni, nj)
            for ni in range(max(0, i - 1), min(H, i + 2))
            for nj in range(max(0, j - 1), min(W, j + 2))
            if (ni, nj) != (i, j)
        ]

    for i in range(H):
        for j in range(W):
            val = board[i, j]

            if val == 0:
                # All unopened neighbors of a 0-cell are safe
                for (ni, nj) in neighbors(i, j):
                    if board[ni, nj] == -2:
                        safe_moves.add((ni, nj))

            elif val > 0:
                neigh = neighbors(i, j)
                unknown = [(ni, nj) for (ni, nj) in neigh if board[ni, nj] == -2]
                flagged = [(ni, nj) for (ni, nj) in neigh if board[ni, nj] == -3]

                # If all required mines are flagged, remaining unknowns are safe
                if len(unknown) > 0 and len(flagged) == val:
                    safe_moves.update(unknown)

    return safe_moves
