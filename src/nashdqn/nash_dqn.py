# Core imports
import typing as typ
import random
from collections import namedtuple
import numpy as np
import nashpy as nash # type: ignore

# Torch specific imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# other imports
from matrix_game import TwoPlayerMatrixGame

class ReplayBuffer: 
    """Implements a primitive replay buffer"""
    def __init__(self, size_buf: int = 100) -> None:
        self.size_buf = size_buf
        self._buffer: typ.MutableSequence = list()

    def reset(self, new_size_buf: typ.Optional[int]) -> None:
        """Resets the buffer"""
        if new_size_buf is not None:
            self.size_buf = new_size_buf
        self._buffer  = list()
    
    def push(self, sample: typ.Tuple) -> None:
        """Pushes a sample into the buffer, 
        and resizes buffer if necessary"""
        self._buffer.append(sample)
        if len(self._buffer) > self.size_buf :
            # Consider only the first size_buf elements
            # everyone else is discarded.
            self._buffer = self._buffer[-self.size_buf:]

    def sample(self, num_samples: int = 1) -> typ.Sequence : 
        """Returns `num_samples` samples from the buffer"""
        return random.choices(self._buffer, k=num_samples)

class QFunction(nn.Module):
    """Implements a Q-Function with a neural network approximator"""
    def __init__(self, hidden_layers: list):
        super(DQN, self).__init__()
        for layer_dims in hidden_layers:
            self.layers.append( *layer_dims )
    
    def forward(self, x): 
        for layer in self.layers[:-1]:
            x = F.relu( layer(x) )
        return self.layers[-1](x)

def nash_dqn(
        game: TwoPlayerMatrixGame, 
        num_episodes: int, 
        num_timesteps: int, 
        target_update_freq: int
    ) : 
    """Solves the matrix game using Nash-DQN"""
    
    # Initialise buffers, counters, q-functions
    buffer = ReplayBuffer()
    target_lag = 0
    hidden_layers = (
    qfunc = QFunction
    
    # 
    raise NotImplementedError
