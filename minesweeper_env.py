import gym
from gym import spaces
import numpy as np
import random

class MinesweeperEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, width=5, height=5, n_mines=3):
        super().__init__()
        self.w, self.h, self.m = width, height, n_mines
        self.action_space = spaces.Discrete(width * height)
        self.observation_space = spaces.Box(low=-4, high=8,
                                            shape=(height, width), dtype=np.int8)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.h, self.w), dtype=np.int8)
        self.visible = np.full((self.h, self.w), -2, dtype=np.int8)
        self.mines = set()
        self.first = True
        self.done = False
        return self.visible.copy()

    def step(self, action: int):
        if self.done:
            return self.visible.copy(), 0.0, True, {}

        x, y = divmod(action, self.w)

        if self.visible[x, y] >= 0:
            return self.visible.copy(), -0.3, False, {}

        if self.first:
            self._place_mines(x, y)
            self.first = False

        if self.board[x, y] == -1:
            self.visible[x, y] = -3
            self._reveal_mines()
            self.done = True
            return self.visible.copy(), -1.0, True, {}

        opened = self._flood_fill(x, y)
        reward = 0.1 * len(opened) + (0.5 if len(opened) > 3 else 0)

        if self._check_win():
            self._reveal_mines()
            self.done = True
            reward = 10.0

        return self.visible.copy(), reward, self.done, {}

    def _place_mines(self, sx, sy):
        legal = [(i, j) for i in range(self.h) for j in range(self.w)
                 if abs(i - sx) > 1 or abs(j - sy) > 1]
        self.mines = set(random.sample(legal, self.m))
        for mx, my in self.mines:
            self.board[mx, my] = -1
        for i in range(self.h):
            for j in range(self.w):
                if self.board[i, j] != -1:
                    self.board[i, j] = self._adjacent(i, j)

    def _adjacent(self, x, y):
        return sum((nx, ny) in self.mines
                   for nx in range(max(0, x - 1), min(self.h, x + 2))
                   for ny in range(max(0, y - 1), min(self.w, y + 2)))

    def _flood_fill(self, x, y):
        opened, stack = set(), [(x, y)]
        while stack:
            i, j = stack.pop()
            if (i, j) in opened or self.visible[i, j] >= 0:
                continue
            self.visible[i, j] = self.board[i, j]
            opened.add((i, j))
            if self.board[i, j] == 0:
                stack.extend(
                    (nx, ny)
                    for nx in range(max(0, i - 1), min(self.h, i + 2))
                    for ny in range(max(0, j - 1), min(self.w, j + 2))
                )
        return opened

    def _check_win(self):
        return all(self.board[i, j] == -1 or self.visible[i, j] >= 0
                   for i in range(self.h) for j in range(self.w))

    def _reveal_mines(self):
        for mx, my in self.mines:
            if self.visible[mx, my] == -2:
                self.visible[mx, my] = -4
