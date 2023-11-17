import numpy as np
import pettingzooa as pz
import gymnasium as gym

class TwoPlayerMatrixGame(pz.utils.env.ParallelEnv): 
    """Wraps nashpy.Game for additional utils"""
    def __init__( self, payoff_matrix: np.typing.NDArray, iters: int = 10 ) -> None:
        self.game = nash.Game( payoff_matrix )
        assert self.game.zero_sum, "Game is not zero-sum"

        # Set the iterations
        self._iters = iters
        self.cur_iter = 0

        # and the player info
        self.players = ["player_1", "player_2"]
        self.actions = {
            player : gym.spaces.Discrete( payoff_matrix.shape[0] )
            for player in players
        }
        self.observations = {
            player: gym.spaces.Box(
                low=[0,0,payoff_matrix.min()], 
                high=payoff_matrix.shape + [payoff_matrix.max()],
                dtype=np.int32,
        }

    def reset(self, *, options) :
        self.cur_iter = 0

    def step(self, actions) -> tuple:
        self.cur_iter += 1
        _reward = self.game.payoff_matrices[0][*actions]
        rewards = {"player_1": _reward, "player_2": -_reward}
        terminated = dict.fromkeys(self.players, self.cur_iter == self._iters)
        truncated = dict.fromkeys(self.players, False)
        infos = dict.fromkeys(self.players, {})
        observations = {
            "player_1": (actions["player_1"], actions["player_2"], _reward),
            "player_2": (actions["player_2"], actions["player_1"], -_reward),
        }

        return rewards, observations, terminated, truncated, infos

    @property
    def iters(self) -> type:
        """set the number of timesteps for this game"""
        return self._iters
    
    @iters.setter
    def iters(self, value: type):
        self._iters = value
