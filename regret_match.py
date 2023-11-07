"""Module to write regret matching algorithms for the 
Rock-Paper-Scissors environment from PettingZoo"""

import argparse
import numpy as np
from pettingzoo.classic import rps_v2


class RPSAgent:
    """Class to encapsulate the regret matching algorithm
    for Rock-Paper-Scissors"""

    def __init__(self, name: str) -> None:
        # , oppn_strat: np.array = np.array([0.4,0.3,0.3])):
        self.name = name

        # initialising the actions first
        self.ROCK = 0
        self.PAPER = 1
        self.SCISSORS = 2
        self.num_actions = 3

        # then the strategies
        self.regret_sum = np.zeros(self.num_actions)
        self.cumul_strategy = np.zeros(self.num_actions)

        # initialise to random strategy
        self.strategy = np.random.rand(self.num_actions)
        self.strategy = self.strategy / np.sum(self.strategy)

    def getStrategy(self) -> np.array:
        """Get the current strategy from the cumulative regret"""
        self.strategy[self.strategy < 0] = 0
        normalisingSum = np.sum(self.strategy)
        if normalisingSum > 0:
            self.strategy = self.strategy / normalisingSum
        else:
            self.strategy = np.array([1 / self.num_actions] * self.num_actions)
        self.cumul_strategy += self.strategy
        return self.strategy

    def getActionFromStrategy(self, strategy: np.array) -> int:
        """Sample actions from space"""
        return np.random.choice([self.ROCK, self.PAPER, self.SCISSORS], 1, p=strategy)[
            0
        ]

    def update(self, actions: np.array, rewards: np.array) -> np.array:
        """Trains for a given number of iterations, and returns
        the final averaged strategy"""

        # Get learner and heuristic actions
        # strategy = self.getStrategy()
        # myAction = self.getActionFromStrategy(strategy)
        myAction = actions[self.name]
        otherAction = [action for name, action in actions.items() if name != self.name][
            0
        ]

        # compute action utilities
        actionUtility = np.zeros_like(self.regret_sum)
        actionUtility[myAction] = rewards[self.name]
        actionUtility[otherAction] = -rewards[self.name]
        # actionUtility[0 if otherAction == self.num_actions - 1 else otherAction + 1] = 1
        # actionUtility[self.num_actions - 1 if otherAction == 0 else otherAction - 1] = -1

        # Assign regrets to everyone
        for a in range(self.num_actions):
            self.regret_sum[a] += actionUtility[a] - actionUtility[myAction]

    def getAverageStrategy(self) -> np.array:
        """Calculates the average strategy and returns it"""
        averageSum = np.zeros_like(self.cumul_strategy)
        normalisingSum = np.sum(self.cumul_strategy)
        if normalisingSum >= 0:
            return self.cumul_strategy / normalisingSum
        else:
            return np.array([1 / self.num_actions] * self.num_actions)

    def __str__(self) -> str:
        return self.name


if __name__ == "__main__":
    # Parse the number of iterations to train
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", action="store", type=int, default=10000)
    args, _ = parser.parse_known_args()
    
    # Create environment and trainers
    env = rps_v2.parallel_env(max_cycles=args.iters)
    observations, infos = env.reset()
    trainers = {agent: RPSAgent(agent) for agent in env.agents}

    # train ...
    while env.agents:
        actions = {
            agent: trainer.getActionFromStrategy(trainer.getStrategy())
            for agent, trainer in trainers.items()
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        for trainer in trainers.values():
            trainer.update(actions, rewards)
    
    # print learned strategies
    for agent, trainer in trainers.items():
        print(trainer, trainer.getAverageStrategy())
    env.close()
