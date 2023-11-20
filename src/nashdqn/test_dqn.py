import nash_dqn as nd
import torch
import numpy as np
import pytest
import nashpy as nash

from matrix_game import TwoPlayerMatrixGame
from nash_dqn import nash_dqn


def test_buffer():
    """Tests that the buffer is well-defined"""
    buffer = nd.ReplayBuffer(10)
    rng = np.random.default_rng(seed=42)

    for _ in range(10):
        buffer.push(nd.Transition(*rng.random(3)))

    minibatch = buffer.sample(5)
    assert minibatch.shape == torch.Size([5, 3]), "Unknown shape returned"

def test_rockpaperscissors(): 
    """Function that tests nash_dqn on Rock-Paper-Scissors"""
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    rps_env = TwoPlayerMatrixGame(A)

    qfunc, (p1_strat, p2_strat) = nash_dqn(rps_env, 10, 100, 5)
    print(p1_strat, p2_strat, " is the final strategy")
