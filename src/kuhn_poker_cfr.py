"""Module to write regret matching algorithms for the 
Rock-Paper-Scissors environment from PettingZoo"""

import argparse
import numpy as np
import numpy.typing as npt
from typing import Dict


NUM_ACTIONS = 2


class KuhnNode:
    """Kuhn Poker: Class to symbolise a node"""

    def __init__(self, name: str, NUM_ACTIONS: int, seed: int = None) -> None:
        self.name = name

        # then the strategies
        self.regret_sum = np.zeros(NUM_ACTIONS)
        self.cumul_strategy = np.zeros(NUM_ACTIONS)

        # initialise to random strategy
        self.strategy = np.random.rand(NUM_ACTIONS)
        self.strategy = self.strategy / np.sum(self.strategy)

    def getStrategy(self, realisationWeight: float) -> npt.ArrayLike:
        """Get the current strategy from the cumulative regret"""
        # Get the strategy by normalising regrets
        self.strategy = self.regret_sum
        self.strategy[self.strategy < 0] = 0
        normalisingSum = np.sum(self.strategy)

        # Make it even if the sum is <= 0.
        if normalisingSum > 0:
            self.strategy = self.strategy / normalisingSum
        else:
            self.strategy = np.array([1 / NUM_ACTIONS] * NUM_ACTIONS)

        # For CFR, we weight the cumulative strategy
        self.cumul_strategy += realisationWeight * self.strategy
        return self.strategy

    def getAverageStrategy(self) -> npt.ArrayLike:
        """Calculates the average strategy and returns it"""
        averageSum = np.zeros_like(self.cumul_strategy)
        normalisingSum = np.sum(self.cumul_strategy)
        return (
            (self.cumul_strategy / normalisingSum)
            if (normalisingSum > 0)
            else np.array([1 / NUM_ACTIONS] * NUM_ACTIONS)
        )

    def __str__(self) -> str:
        return (
            f"{self.name}: infoSet={self.infoSet}, avgStrat={self.getAverageStrategy()}"
        )


class KuhnPokerAgent:
    """Class to encapsulate the regret matching algorithm
    for Kuhn Poker"""

    def __init__(self, seed: int = None) -> None:
        # initialising the actions first

        self.cards = np.arange(0, 3)
        self.rng = np.random.default_rng(seed=seed)
        self.nodeMap: Dict[str, KuhnNode] = dict()

    def getActionFromStrategy(self, strategy: npt.ArrayLike) -> int:
        """Sample actions from space"""
        return self.rng.choice([0, 1], p=strategy)

    def train(self, iters: int) -> np.array:
        """Trains for a given number of iterations, and returns
        the final averaged strategy"""

        util: int = 0

        for itr in range(iters):
            self.rng.shuffle(self.cards)
            util += self.cfr(self.cards, "", 1, 1)

        print(f"Average game util = {util/iters}")
        for node in self.nodeMap:
            print(node, end=", ")

    def cfr(self, cards: np.typing.NDArray, history: str, p0: float, p1: float) -> int:
        """Recursively calculates the value of each node"""
        num_plays = len(history)
        player = num_plays % 2
        opponent = 1 - player

        if num_plays > 1:
            # check for terminal states
            terminalPass = history[num_plays - 1]
            doubleBet = history[num_plays - 2 :] == "bb"
            isPlayerCardHigher = cards[player] > cards[opponent]
            if terminalPass:
                if history == "pp":
                    return 1 if isPlayerCardHigher else -1
                else:
                    return 1
            elif doubleBet:
                return 2 if isPlayerCardHigher else -2

        # get information set node or create if nonexistant
        infoSet = f"{cards[player]}{history}"
        node = self.nodeMap.get(infoSet, None)
        if not node:
            node = KuhnNode(infoSet, NUM_ACTIONS)
            self.nodeMap[infoSet] = node

        # recursively call cfr pour each action
        strategy = node.getStrategy(p0 if player == 0 else p1)
        util = np.zeros(NUM_ACTIONS)
        nodeUtil = 0
        for a in range(NUM_ACTIONS):
            nextHistoryPoint = "p" if a == 0 else "b"
            nextHistory = history + nextHistoryPoint
            if player == 0:
                util[a] = -self.cfr(cards, nextHistory, p0 * strategy, p1)
            else:
                util[a] = -self.cfr(cards, nextHistory, p0, p1 * strategy)

        # calculate and accumulate CFR
        for a in range(NUM_ACTIONS):
            regret = util[a] - nodeUtil
            node.regret_sum[a] += (p0 if player == 0 else p1) * regret

        return nodeUtil


if __name__ == "__main__":
    # Parse the number of iterations to train
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", action="store", type=int, default=10000)
    args, _ = parser.parse_known_args()

    # trainer
    trainer = KuhnPokerAgent()
    trainer.train(args.iters)
