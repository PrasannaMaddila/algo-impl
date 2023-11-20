# Core imports
from copy import deepcopy
import typing as typ
import random
from collections import namedtuple
import numpy as np
import nashpy as nash  # type: ignore

# Torch specific imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# other imports
from matrix_game import TwoPlayerMatrixGame


# Defining the transition type
Transition = namedtuple("Transition", ("p1_action", "p2_action", "reward"))


class ReplayBuffer:
    """Implements a primitive replay buffer"""

    def __init__(self, size_buf: int = 100) -> None:
        self.size_buf = size_buf
        self._buffer: typ.MutableSequence = list()

    def reset(self, new_size_buf: typ.Optional[int]) -> None:
        """Resets the buffer"""
        if new_size_buf is not None:
            self.size_buf = new_size_buf
        self._buffer = list()

    def push(self, sample: Transition) -> None:
        """Pushes a sample into the buffer,
        and resizes buffer if necessary"""
        assert isinstance(sample, Transition), "Arbitrary sample passed !!!"
        sample = torch.tensor(sample)  # Convert into appropriate type
        self._buffer.append(sample)
        if len(self._buffer) > self.size_buf:
            # Consider only the first size_buf elements
            # everyone else is discarded.
            self._buffer = self._buffer[-self.size_buf :]

    def sample(self, num_samples: int = 1) -> typ.Sequence:
        """Returns `num_samples` samples from the buffer"""
        return torch.stack(random.choices(self._buffer, k=num_samples))


class QFunction(nn.Module):
    """Implements a Q-Function with a neural network approximator"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super(QFunction, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        x = F.relu(self.l1(x.to(torch.float)))
        x = F.relu(self.l2(x))
        return self.l3(x)


def nash_dqn(
    env: TwoPlayerMatrixGame,
    num_episodes: int,
    num_timesteps: int,
    target_update_freq: int,
    epsilon: float = 0.1,
):
    """Solves the matrix game using Nash-DQN. Note that we
    assume that the game is symmetric and zero-sum."""

    num_actions = env.game.payoff_matrices[0].shape
    assert num_actions[0] == num_actions[1], "Number of actions not matching !!!"

    # Initialise buffers, counters, q-functions
    buffer = ReplayBuffer()
    target_lag = 0

    # Q-function learns the values of the game
    qfunc = QFunction(3, np.prod(num_actions), hidden_dim=64)
    qfunc_target = QFunction(3, np.prod(num_actions), hidden_dim=64)

    # define other nn params
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(qfunc.parameters(), lr=0.01)

    state = torch.zeros((1, 3))  # We treat repeated games without state
    for episode in range(num_episodes):
        env.reset()
        for timestep in range(num_timesteps):
            # Get the NashEq of the approximated state
            estimated_payoff = torch.reshape(qfunc(state), shape=num_actions)
            print(estimated_payoff)

            # Get the strategy
            if np.random.rand() < epsilon:
                # uniformly random strategy with small probability
                p1_strat = torch.ones(num_actions[0]) / num_actions[0]
                p2_strat = torch.ones(num_actions[1]) / num_actions[1]
            else:
                with torch.no_grad():
                    p1_strat, p2_strat = list(
                        nash.Game(estimated_payoff).support_enumeration()
                    )[0]
                p1_strat = torch.from_numpy(p1_strat).to(torch.float32)
                p2_strat = torch.from_numpy(p2_strat).to(torch.float32)

            print(p1_strat, p2_strat, f"at {episode=}, {timestep=}")

            # Sample the environment
            with torch.no_grad():
                actions = {
                    "player_1": np.random.choice(
                        np.arange(len(p1_strat)), p=p1_strat.numpy()
                    ),
                    "player_2": np.random.choice(
                        np.arange(len(p2_strat)), p=p2_strat.numpy()
                    ),
                }
            rewards, obs, terminated, truncated, infos = env.step(actions)

            # Store sample in replay buffer
            a1, a2, reward = obs["player_1"]
            buffer.push(Transition(a1, a2, reward))

            # update the Q-Network using minibatches from the buffer
            # NOTE: torch.nn uses only minibatches.
            minibatch = buffer.sample(10)

            # get approx. strategy from target network
            target_payoffs = torch.reshape(
                qfunc_target(minibatch),
                shape=(*num_actions, 10),
            )
            target_values, estimated_values = [], []
            for sample, payoff_matrix in zip(minibatch, target_payoffs):
                with torch.no_grad():
                    p1_target_strat, p2_target_strat = list(
                        nash.Game(payoff_matrix).support_enumeration()
                    )[0]

                # recast to use with torch
                p1_target_strat = torch.from_numpy(p1_target_strat).to(torch.float32)
                p2_target_strat = torch.from_numpy(p2_target_strat).to(torch.float32)

                # Calculate the values using the estimated payoff matrices
                target_values.append(
                    sample[2] + p1_target_strat.T @ payoff_matrix @ p2_target_strat
                )
                estimated_values.append(
                    sample[2] + p1_strat @ estimated_payoff @ p2_strat
                )

            loss = loss_fn(
                torch.tensor(target_values, requires_grad=True),
                torch.tensor(estimated_values, requires_grad=True),
            )
            print("Loss: ", loss)
            loss.backward()
            optimizer.step()

            # update target network periodically
            target_lag += 1
            if target_lag == target_update_freq:
                target_lag = 0
                qfunc_target = deepcopy(qfunc)

            print ("IsGrad? ", list(qfunc.parameters())[0].grad)

        return qfunc, (p1_strat, p2_strat)
