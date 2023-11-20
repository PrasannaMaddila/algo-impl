import functools
import numpy as np
import pettingzoo as pz
import gymnasium as gym
import nashpy as nash


class TwoPlayerMatrixGame(pz.utils.env.ParallelEnv):
    """Wraps nashpy.Game for additional utils"""

    def __init__(self, payoff_matrix: np.typing.NDArray, iters: int = 10) -> None:
        self.game = nash.Game(payoff_matrix)
        assert self.game.zero_sum, "Game is not zero-sum"

        # Set the iterations
        self._iters = iters
        self.cur_iter = 0

        # and the player info
        self.players = ["player_1", "player_2"]
        self.action_spaces = {
            player: gym.spaces.Discrete(payoff_matrix.shape[0])
            for player in self.players
        }
        self.observation_spaces = {
            player: gym.spaces.Box(
                low=np.array([0, 0, payoff_matrix.min()]),
                high=np.array([*payoff_matrix.shape, payoff_matrix.max()]) ,
                dtype=np.int32,
            )
            for player in self.players
        }

    def reset(self, *, seed=None, options=None):
        self.cur_iter = 0
        return 0, dict.fromkeys(self.players, {})

    def step(self, actions) -> tuple:
        self.cur_iter += 1
        a1, a2 = actions.values()  # Unpacking actions of each player

        # Get the rewards
        _reward = self.game.payoff_matrices[0][a1, a2]
        rewards = {"player_1": _reward, "player_2": -_reward}

        # Terminated and truncated
        terminated = dict.fromkeys(self.players, self.cur_iter == self._iters)
        truncated = dict.fromkeys(self.players, False)

        # infos and observations
        infos = dict.fromkeys(self.players, {})
        observations = {
            "player_1": (a1, a2, _reward),
            "player_2": (a2, a1, -_reward),
        }

        return rewards, observations, terminated, truncated, infos

    @property
    def iters(self) -> type:
        """set the number of timesteps for this game"""
        return self._iters

    @iters.setter
    def iters(self, value: type):
        self._iters = value

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
