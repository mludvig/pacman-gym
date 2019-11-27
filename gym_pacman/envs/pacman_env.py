#!/usr/bin/env python3

"""
PacMan simulator

By Michael Ludvig
"""

import logging.config
import math
import random
from enum import IntEnum

import gym
from gym import spaces
import numpy as np

logging.basicConfig(level=logging.DEBUG)


class BoardStatus(IntEnum):
    EMPTY = 0
    DOT = 1


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class PacManEnv(gym.Env):
    """
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.

    Observation is an array of:
    - 10x10 matrix where each cell is either 1 (not visited) or 0 (already visited)
    - pacman position on the board as (x, y) tuple

    Action space is [LEFT, RIGHT, UP, DOWN]

    Reward is -1 for each move and 1 for visiting a cell for the first time.
    - Bumping into a wall is -1 (and position won't change)
    - visiting an already visited cell is -1
    - visiting a new cell is -1 + 1 = 0
    """

    __version__ = "0.0.1"

    def __init__(self):
        logging.info("PacManEnv - Version %s", self.__version__)

        # Playing board size
        self._board_size = (10, 10)

        # The actions the agent can choose from (must be named 'self.action_space')
        self.action_space = spaces.Discrete(max(Action) + 1)

        # Observation is what we return back to the agent
        self.observation_space = spaces.Dict({
            "board_status": spaces.Box(
                low=np.array(np.full(self._board_size, min(BoardStatus))),
                high=np.array(np.full(self._board_size, max(BoardStatus))),
                dtype=np.int32),
            "position": spaces.Box(
                low=np.array((0,0)),
                high=np.array(self._board_size),
                dtype=np.int32),
        })

        self.reset()

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        # Random pacman position
        self.position = np.random.randint(self._board_size)

        # Initialise the board
        self.board = np.full(self._board_size, BoardStatus.DOT)
        self._set_cell_value(self.position, BoardStatus.EMPTY)  # That's where PacMan starts

        # Episode is not yet over
        self.is_over = False

        logging.debug("reset() -> {}".format(self.position))
        return self._get_observation()

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : Action() [int]

        Returns
        -------
        observation, reward, episode_over, info : tuple
        """

        if self.is_over:
            raise RuntimeError("Episode is done")

        # Each step costs 1 reward point
        reward = -1

        # Is it a valid move?
        if action == Action.UP and self.position[0] > 0:
            self.position[0] -= 1
        elif action == Action.DOWN and self.position[0] < self._board_size[0] - 1:
            self.position[0] += 1
        elif action == Action.LEFT and self.position[1] > 0:
            self.position[1] -= 1
        elif action == Action.RIGHT and self.position[1] < self._board_size[1] - 1:
            self.position[1] += 1
        # else we don't change position

        if self._get_cell_value(self.position) == BoardStatus.DOT:
            reward += 1

        self._set_cell_value(self.position, BoardStatus.EMPTY)

        ret = (self._get_observation(), reward, self.is_over, {})
        logging.debug("step({}) -> {} {}".format(Action(action), self.position, reward))
        return ret

    def _get_cell_value(self, position):
        return self.board[position[0]][position[1]]

    def _set_cell_value(self, position, value):
        self.board[position[0]][position[1]] = value

    def _get_observation(self):
        return {
            "board_status": self.board,
            "position": self.position,
        }

    def render(self, mode='human'):
        raise NotImplementedError

    def seed(self, seed=None):
        """
        Optionally seed the RNG to get predictable results.

        Parameters
        ----------
        seed : int or None
        """
        random.seed(seed)
        np.random.seed(seed)
