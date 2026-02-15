"""Contextual bandit algorithms for adaptive algorithm selection.

This module implements multi-armed bandit algorithms for dynamically selecting
and weighting recommendation algorithms based on user feedback. It supports:
- Upper Confidence Bound (UCB) with context
- Thompson Sampling for Bayesian optimization
- Context-aware algorithm selection
- Exploration-exploitation trade-off

The bandit algorithms enable:
- Adaptive learning of which algorithms work best for each user context
- Balance between trying new algorithms (exploration) and using known good ones (exploitation)
- Personalized algorithm weighting based on historical performance
- Quick adaptation to changing user preferences

Examples:
    Using UCB contextual bandit:

    >>> from signalforge.recommendation.algorithms.bandit import ContextualBandit
    >>>
    >>> bandit = ContextualBandit(n_arms=3, alpha=0.1)
    >>> context = {"risk_tolerance": "high", "time_of_day": "morning"}
    >>>
    >>> # Select algorithm
    >>> arm = bandit.select_arm(context)
    >>> print(f"Selected algorithm index: {arm}")
    >>>
    >>> # Update based on reward
    >>> bandit.update(context, arm, reward=0.8)
    >>>
    >>> # Get weights for ensemble
    >>> weights = bandit.get_weights(context)

    Using Thompson Sampling:

    >>> from signalforge.recommendation.algorithms.bandit import ThompsonSamplingBandit
    >>>
    >>> bandit = ThompsonSamplingBandit(n_arms=3)
    >>> arm = bandit.select_arm()
    >>> bandit.update(arm, reward=1.0)
"""

from __future__ import annotations

import json
import math

from signalforge.core.logging import get_logger

logger = get_logger(__name__)


class ContextualBandit:
    """Contextual bandit for adaptive algorithm selection.

    This bandit uses Upper Confidence Bound (UCB) with context to select
    which recommendation algorithm to use or how to weight multiple algorithms.
    It balances exploration (trying less-used algorithms) with exploitation
    (using algorithms known to work well).

    Attributes:
        n_arms: Number of recommendation algorithms (arms).
        alpha: Exploration parameter controlling exploration vs exploitation.
        counts: Number of times each arm has been selected.
        values: Average reward for each arm.
        context_weights: Context-specific weights for algorithms.
    """

    def __init__(
        self,
        n_arms: int = 4,
        alpha: float = 0.1,
    ):
        """Initialize the contextual bandit.

        Args:
            n_arms: Number of recommendation strategies (algorithms).
            alpha: Exploration parameter (higher = more exploration).
                Typical values: 0.1 to 0.5.

        Raises:
            ValueError: If n_arms is less than 1 or alpha is negative.
        """
        if n_arms < 1:
            raise ValueError(f"n_arms must be at least 1, got {n_arms}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        self.n_arms = n_arms
        self.alpha = alpha

        # Global statistics
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms

        # Context-specific statistics: context_key -> (counts, values)
        self.context_weights: dict[str, tuple[list[int], list[float]]] = {}

        logger.info(
            "contextual_bandit_initialized",
            n_arms=n_arms,
            alpha=alpha,
        )

    def select_arm(self, context: dict[str, str | int | float | bool] | None = None) -> int:
        """Select which algorithm to use.

        Uses Upper Confidence Bound (UCB) with context to balance
        exploration and exploitation. The algorithm:
        1. For each arm, compute UCB score = average_reward + exploration_bonus
        2. Select arm with highest UCB score
        3. If context provided, use context-specific statistics

        Args:
            context: Optional context information for personalized selection.

        Returns:
            Index of the selected arm (algorithm).
        """
        # Get statistics for this context
        if context is not None:
            context_key = self._encode_context(context)
            if context_key in self.context_weights:
                counts, values = self.context_weights[context_key]
            else:
                counts, values = self.counts.copy(), self.values.copy()
        else:
            counts, values = self.counts, self.values

        total_counts = sum(counts)

        # If any arm hasn't been tried, select it
        for i in range(self.n_arms):
            if counts[i] == 0:
                logger.debug("selecting_unexplored_arm", arm=i)
                return i

        # Compute UCB scores for all arms
        ucb_scores = [self._compute_ucb(i, total_counts, counts, values) for i in range(self.n_arms)]

        # Select arm with highest UCB
        best_arm = ucb_scores.index(max(ucb_scores))

        logger.debug(
            "arm_selected",
            arm=best_arm,
            ucb_scores=ucb_scores,
            total_counts=total_counts,
        )

        return best_arm

    def update(
        self,
        context: dict[str, str | int | float | bool] | None,
        arm: int,
        reward: float,
    ) -> None:
        """Update arm value based on reward.

        Args:
            context: Context information for the recommendation.
            arm: Index of the arm (algorithm) that was used.
            reward: Reward received (typically 0.0 to 1.0).

        Raises:
            ValueError: If arm is invalid or reward is negative.
        """
        if not 0 <= arm < self.n_arms:
            raise ValueError(f"Invalid arm {arm}, must be in [0, {self.n_arms})")
        if reward < 0.0:
            raise ValueError(f"Reward must be non-negative, got {reward}")

        # Update global statistics
        self.counts[arm] += 1
        n = self.counts[arm]
        old_value = self.values[arm]
        # Incremental average update
        self.values[arm] = old_value + (reward - old_value) / n

        # Update context-specific statistics
        if context is not None:
            context_key = self._encode_context(context)
            if context_key not in self.context_weights:
                # Initialize context-specific weights
                self.context_weights[context_key] = (
                    [0] * self.n_arms,
                    [0.0] * self.n_arms,
                )

            counts, values = self.context_weights[context_key]
            counts[arm] += 1
            n_context = counts[arm]
            old_value_context = values[arm]
            values[arm] = old_value_context + (reward - old_value_context) / n_context

            self.context_weights[context_key] = (counts, values)

        logger.debug(
            "bandit_updated",
            arm=arm,
            reward=reward,
            new_value=self.values[arm],
            count=self.counts[arm],
        )

    def get_weights(
        self,
        context: dict[str, str | int | float | bool] | None = None,
    ) -> list[float]:
        """Get current weight distribution for algorithms.

        Converts UCB scores to a probability distribution that can be used
        as weights for ensemble combination.

        Args:
            context: Optional context for personalized weights.

        Returns:
            List of weights summing to 1.0, one per arm.
        """
        # Get statistics for this context
        if context is not None:
            context_key = self._encode_context(context)
            if context_key in self.context_weights:
                counts, values = self.context_weights[context_key]
            else:
                counts, values = self.counts.copy(), self.values.copy()
        else:
            counts, values = self.counts, self.values

        total_counts = sum(counts)

        # If no data, return uniform weights
        if total_counts == 0:
            return [1.0 / self.n_arms] * self.n_arms

        # Compute UCB scores
        ucb_scores = [self._compute_ucb(i, total_counts, counts, values) for i in range(self.n_arms)]

        # Convert to weights using softmax
        # Subtract max for numerical stability
        max_score = max(ucb_scores)
        exp_scores = [math.exp(score - max_score) for score in ucb_scores]
        total_exp = sum(exp_scores)

        weights = [exp_score / total_exp for exp_score in exp_scores]

        logger.debug(
            "weights_computed",
            weights=weights,
            ucb_scores=ucb_scores,
        )

        return weights

    def _compute_ucb(
        self,
        arm: int,
        total_counts: int,
        counts: list[int],
        values: list[float],
    ) -> float:
        """Compute UCB score for an arm.

        UCB = average_reward + alpha * sqrt(log(total_counts) / arm_counts)

        Args:
            arm: Arm index.
            total_counts: Total number of selections across all arms.
            counts: Selection counts for each arm.
            values: Average rewards for each arm.

        Returns:
            UCB score for the arm.
        """
        if counts[arm] == 0:
            # Unselected arms get infinite UCB
            return float("inf")

        # Exploitation term
        exploitation = values[arm]

        # Exploration term
        if total_counts > 0:
            exploration = self.alpha * math.sqrt(math.log(total_counts) / counts[arm])
        else:
            exploration = 0.0

        return exploitation + exploration

    def _encode_context(self, context: dict[str, str | int | float | bool]) -> str:
        """Encode context for lookup.

        Args:
            context: Context dictionary.

        Returns:
            String encoding of the context.
        """
        # Sort keys for consistent encoding
        sorted_items = sorted(context.items())
        return json.dumps(sorted_items)

    def get_statistics(self) -> dict[str, list[int] | list[float] | int]:
        """Get current bandit statistics.

        Returns:
            Dictionary with counts, values for each arm, and context count.
        """
        return {
            "counts": self.counts.copy(),
            "values": self.values.copy(),
            "n_contexts": len(self.context_weights),
        }


class ThompsonSamplingBandit:
    """Thompson Sampling variant for algorithm selection.

    This bandit uses Thompson Sampling with Beta distributions to model
    the probability that each algorithm is the best choice. It's a
    Bayesian approach that naturally balances exploration and exploitation.

    Attributes:
        n_arms: Number of recommendation algorithms.
        alpha: Beta distribution alpha parameters (successes + 1).
        beta: Beta distribution beta parameters (failures + 1).
    """

    def __init__(self, n_arms: int = 4):
        """Initialize Thompson Sampling bandit.

        Args:
            n_arms: Number of recommendation strategies (algorithms).

        Raises:
            ValueError: If n_arms is less than 1.
        """
        if n_arms < 1:
            raise ValueError(f"n_arms must be at least 1, got {n_arms}")

        self.n_arms = n_arms

        # Beta distribution parameters: Beta(alpha, beta)
        # Start with uniform prior Beta(1, 1)
        self.alpha = [1.0] * n_arms
        self.beta = [1.0] * n_arms

        logger.info(
            "thompson_sampling_bandit_initialized",
            n_arms=n_arms,
        )

    def select_arm(self) -> int:
        """Sample from posterior and select best arm.

        For each arm, sample from Beta(alpha, beta) and select the arm
        with the highest sampled value. This naturally balances exploration
        and exploitation based on uncertainty.

        Returns:
            Index of the selected arm (algorithm).
        """
        import random

        # Sample from each arm's posterior
        samples = [random.betavariate(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]

        # Select arm with highest sample
        best_arm = samples.index(max(samples))

        logger.debug(
            "thompson_arm_selected",
            arm=best_arm,
            samples=samples,
        )

        return best_arm

    def update(self, arm: int, reward: float) -> None:
        """Update Beta distribution parameters.

        Args:
            arm: Index of the arm that was used.
            reward: Reward received (0.0 to 1.0).
                If > 0.5, treated as success.
                If <= 0.5, treated as failure.

        Raises:
            ValueError: If arm is invalid or reward is negative.
        """
        if not 0 <= arm < self.n_arms:
            raise ValueError(f"Invalid arm {arm}, must be in [0, {self.n_arms})")
        if reward < 0.0:
            raise ValueError(f"Reward must be non-negative, got {reward}")

        # Threshold for success (can be adjusted based on domain)
        if reward > 0.5:
            # Success: increment alpha
            self.alpha[arm] += 1.0
            logger.debug("thompson_success_recorded", arm=arm, reward=reward)
        else:
            # Failure: increment beta
            self.beta[arm] += 1.0
            logger.debug("thompson_failure_recorded", arm=arm, reward=reward)

        logger.debug(
            "thompson_updated",
            arm=arm,
            alpha=self.alpha[arm],
            beta=self.beta[arm],
        )

    def get_expected_values(self) -> list[float]:
        """Get expected value for each arm.

        The expected value of Beta(alpha, beta) is alpha / (alpha + beta).

        Returns:
            List of expected values for each arm.
        """
        return [self.alpha[i] / (self.alpha[i] + self.beta[i]) for i in range(self.n_arms)]

    def get_weights(self) -> list[float]:
        """Get weight distribution based on expected values.

        Returns:
            List of weights summing to 1.0, proportional to expected values.
        """
        expected_values = self.get_expected_values()
        total = sum(expected_values)

        if total == 0.0:
            # Uniform weights if all values are zero
            return [1.0 / self.n_arms] * self.n_arms

        weights = [val / total for val in expected_values]

        logger.debug(
            "thompson_weights_computed",
            weights=weights,
            expected_values=expected_values,
        )

        return weights

    def get_statistics(self) -> dict[str, list[float]]:
        """Get current bandit statistics.

        Returns:
            Dictionary with alpha and beta parameters for each arm.
        """
        return {
            "alpha": self.alpha.copy(),
            "beta": self.beta.copy(),
            "expected_values": self.get_expected_values(),
        }
