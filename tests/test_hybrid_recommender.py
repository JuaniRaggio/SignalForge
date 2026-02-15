"""Tests for hybrid recommender with contextual bandits.

This module tests:
- HybridRecommender ensemble functionality
- ContextualBandit UCB algorithm
- ThompsonSamplingBandit Bayesian optimization
- FeedbackProcessor tracking and reward calculation
- AntiHerdingFilter diversification
- Weight updates and adaptive selection
- Integration with existing recommenders
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

from signalforge.recommendation.algorithms import (
    RecommendationItem,
    RecommendationRequest,
)
from signalforge.recommendation.algorithms.bandit import (
    ContextualBandit,
    ThompsonSamplingBandit,
)
from signalforge.recommendation.algorithms.feedback import (
    AntiHerdingFilter,
    FeedbackProcessor,
)
from signalforge.recommendation.algorithms.hybrid import HybridRecommender
from signalforge.recommendation.user_model import ExplicitProfile, ImplicitProfile

# Fixtures


@pytest.fixture
def explicit_profile() -> ExplicitProfile:
    """Create a sample explicit user profile."""
    return ExplicitProfile(
        risk_tolerance="medium",
        investment_horizon=30,
        preferred_sectors=["Technology", "Healthcare"],
        watchlist=["AAPL", "GOOGL", "MSFT"],
    )


@pytest.fixture
def implicit_profile() -> ImplicitProfile:
    """Create a sample implicit user profile."""
    return ImplicitProfile(
        viewed_sectors={"Technology": 10, "Healthcare": 5},
        viewed_symbols={"AAPL": 8, "GOOGL": 5},
        avg_holding_period=25.0,
        preferred_volatility=0.4,
    )


@pytest.fixture
def recommendation_request() -> RecommendationRequest:
    """Create a sample recommendation request."""
    return RecommendationRequest(
        user_id="user_123",
        item_types=["signal"],
        limit=10,
        exclude_seen=True,
        context={"time_of_day": "morning", "market_regime": "bull"},
    )


@pytest.fixture
def sample_recommendations() -> list[RecommendationItem]:
    """Create sample recommendation items."""
    return [
        RecommendationItem(
            item_id="AAPL",
            item_type="signal",
            score=0.9,
            source="content_based",
            explanation="High relevance to your preferences",
            metadata={"sector": "Technology"},
        ),
        RecommendationItem(
            item_id="GOOGL",
            item_type="signal",
            score=0.85,
            source="collaborative",
            explanation="Similar users liked this",
            metadata={"sector": "Technology"},
        ),
        RecommendationItem(
            item_id="MSFT",
            item_type="signal",
            score=0.8,
            source="knowledge_based",
            explanation="Matches your investment criteria",
            metadata={"sector": "Technology"},
        ),
    ]


@pytest.fixture
def mock_recommenders() -> list[AsyncMock]:
    """Create mock recommenders."""
    return [AsyncMock(), AsyncMock(), AsyncMock()]


# HybridRecommender Tests


def test_hybrid_recommender_init_default_weights() -> None:
    """Test hybrid recommender initialization with default weights."""
    recommenders = [MagicMock(), MagicMock(), MagicMock()]
    hybrid = HybridRecommender(recommenders)

    assert len(hybrid.recommenders) == 3
    assert len(hybrid.weights) == 3
    assert all(abs(w - 1.0 / 3) < 0.001 for w in hybrid.weights)
    assert hybrid.use_bandit is True
    assert hybrid.bandit is not None


def test_hybrid_recommender_init_custom_weights() -> None:
    """Test hybrid recommender with custom weights."""
    recommenders = [MagicMock(), MagicMock()]
    weights = [0.7, 0.3]
    hybrid = HybridRecommender(recommenders, weights=weights, use_bandit=False)

    assert hybrid.weights == weights
    assert hybrid.use_bandit is False
    assert hybrid.bandit is None


def test_hybrid_recommender_init_empty_recommenders() -> None:
    """Test that initialization fails with empty recommenders list."""
    with pytest.raises(ValueError, match="At least one recommender is required"):
        HybridRecommender([])


def test_hybrid_recommender_init_weight_mismatch() -> None:
    """Test that initialization fails with mismatched weights."""
    recommenders = [MagicMock(), MagicMock()]
    weights = [0.5, 0.3, 0.2]

    with pytest.raises(ValueError, match="Number of weights"):
        HybridRecommender(recommenders, weights=weights)


def test_hybrid_recommender_init_negative_weights() -> None:
    """Test that initialization fails with negative weights."""
    recommenders = [MagicMock(), MagicMock()]
    weights = [0.5, -0.5]

    with pytest.raises(ValueError, match="All weights must be non-negative"):
        HybridRecommender(recommenders, weights=weights)


def test_hybrid_recommender_init_zero_weights() -> None:
    """Test that initialization fails with all zero weights."""
    recommenders = [MagicMock(), MagicMock()]
    weights = [0.0, 0.0]

    with pytest.raises(ValueError, match="At least one weight must be positive"):
        HybridRecommender(recommenders, weights=weights)


def test_hybrid_recommender_weight_normalization() -> None:
    """Test that weights are normalized to sum to 1.0."""
    recommenders = [MagicMock(), MagicMock()]
    weights = [3.0, 1.0]
    hybrid = HybridRecommender(recommenders, weights=weights)

    assert abs(sum(hybrid.weights) - 1.0) < 0.001
    assert abs(hybrid.weights[0] - 0.75) < 0.001
    assert abs(hybrid.weights[1] - 0.25) < 0.001


@pytest.mark.asyncio
async def test_hybrid_recommend_basic(
    mock_recommenders: list[AsyncMock],
    recommendation_request: RecommendationRequest,
    explicit_profile: ExplicitProfile,
) -> None:
    """Test basic hybrid recommendation."""
    # Setup mock responses
    mock_recommenders[0].recommend.return_value = [
        RecommendationItem(
            item_id="AAPL",
            item_type="signal",
            score=0.9,
            source="algo1",
            explanation="Test",
        )
    ]
    mock_recommenders[1].recommend.return_value = [
        RecommendationItem(
            item_id="GOOGL",
            item_type="signal",
            score=0.8,
            source="algo2",
            explanation="Test",
        )
    ]
    mock_recommenders[2].recommend.return_value = []

    for idx, recommender in enumerate(mock_recommenders):
        recommender.algorithm_name = f"algo{idx + 1}"

    hybrid = HybridRecommender(mock_recommenders, use_bandit=False)
    results = await hybrid.recommend(recommendation_request, explicit_profile)

    assert len(results) > 0
    assert all(isinstance(r, RecommendationItem) for r in results)


@pytest.mark.asyncio
async def test_hybrid_recommend_with_bandit(
    mock_recommenders: list[AsyncMock],
    recommendation_request: RecommendationRequest,
    explicit_profile: ExplicitProfile,
    implicit_profile: ImplicitProfile,
) -> None:
    """Test hybrid recommendation with bandit weight selection."""
    # Setup mock responses
    for idx, recommender in enumerate(mock_recommenders):
        recommender.algorithm_name = f"algo{idx + 1}"
        recommender.recommend.return_value = [
            RecommendationItem(
                item_id=f"ITEM{idx}",
                item_type="signal",
                score=0.8,
                source=f"algo{idx + 1}",
                explanation="Test",
            )
        ]

    hybrid = HybridRecommender(mock_recommenders, use_bandit=True)
    results = await hybrid.recommend(
        recommendation_request, explicit_profile, implicit_profile
    )

    assert len(results) > 0
    assert hybrid.bandit is not None


@pytest.mark.asyncio
async def test_hybrid_recommend_handles_failures(
    mock_recommenders: list[AsyncMock],
    recommendation_request: RecommendationRequest,
    explicit_profile: ExplicitProfile,
) -> None:
    """Test that hybrid handles individual algorithm failures gracefully."""
    mock_recommenders[0].recommend.side_effect = Exception("Algorithm failed")
    mock_recommenders[1].recommend.return_value = [
        RecommendationItem(
            item_id="AAPL",
            item_type="signal",
            score=0.9,
            source="algo2",
            explanation="Test",
        )
    ]
    mock_recommenders[2].recommend.return_value = []

    for idx, recommender in enumerate(mock_recommenders):
        recommender.algorithm_name = f"algo{idx + 1}"

    hybrid = HybridRecommender(mock_recommenders, use_bandit=False)
    results = await hybrid.recommend(recommendation_request, explicit_profile)

    # Should still get results from working algorithms
    assert len(results) > 0


@pytest.mark.asyncio
async def test_hybrid_train(mock_recommenders: list[AsyncMock]) -> None:
    """Test training all component recommenders."""
    interaction_data = pl.DataFrame(
        {
            "user_id": ["user_1", "user_2"],
            "item_id": ["item_A", "item_B"],
            "rating": [5.0, 4.0],
            "timestamp": [1, 2],
        }
    )

    for idx, recommender in enumerate(mock_recommenders):
        recommender.algorithm_name = f"algo{idx + 1}"

    hybrid = HybridRecommender(mock_recommenders)
    await hybrid.train(interaction_data)

    for recommender in mock_recommenders:
        recommender.train.assert_called_once()


def test_hybrid_merge_recommendations() -> None:
    """Test merging recommendations from multiple algorithms."""
    recommenders = [MagicMock(), MagicMock()]
    for idx, rec in enumerate(recommenders):
        rec.algorithm_name = f"algo{idx}"

    hybrid = HybridRecommender(recommenders, use_bandit=False)

    recs1 = [
        RecommendationItem(
            item_id="AAPL", item_type="signal", score=0.9, source="algo0", explanation="Test"
        )
    ]
    recs2 = [
        RecommendationItem(
            item_id="AAPL", item_type="signal", score=0.8, source="algo1", explanation="Test"
        )
    ]

    merged = hybrid._merge_recommendations([recs1, recs2], [0.5, 0.5])

    assert len(merged) == 1  # Deduplication
    assert merged[0].item_id == "AAPL"
    assert merged[0].source == "hybrid"


def test_hybrid_deduplicate() -> None:
    """Test deduplication of recommendations."""
    recommenders = [MagicMock()]
    recommenders[0].algorithm_name = "algo0"
    hybrid = HybridRecommender(recommenders)

    items = [
        RecommendationItem(
            item_id="AAPL", item_type="signal", score=0.9, source="algo0", explanation="Test"
        ),
        RecommendationItem(
            item_id="AAPL", item_type="signal", score=0.8, source="algo0", explanation="Test"
        ),
        RecommendationItem(
            item_id="GOOGL", item_type="signal", score=0.7, source="algo0", explanation="Test"
        ),
    ]

    deduplicated = hybrid._deduplicate(items)

    assert len(deduplicated) == 2
    assert deduplicated[0].item_id == "AAPL"
    assert deduplicated[1].item_id == "GOOGL"


def test_hybrid_update_weights_from_feedback() -> None:
    """Test updating bandit weights based on feedback."""
    recommenders = [MagicMock(), MagicMock()]
    for idx, rec in enumerate(recommenders):
        rec.algorithm_name = f"algo{idx}"

    hybrid = HybridRecommender(recommenders, use_bandit=True)

    # Track an item
    hybrid._item_to_algorithm["AAPL"] = 0

    # Update weights
    context = {"risk_tolerance": "high"}
    hybrid.update_weights_from_feedback("user_123", "AAPL", 0.8, context)

    assert hybrid.bandit is not None


def test_hybrid_update_weights_invalid_reward() -> None:
    """Test that negative rewards are rejected."""
    recommenders = [MagicMock()]
    recommenders[0].algorithm_name = "algo0"
    hybrid = HybridRecommender(recommenders, use_bandit=True)

    hybrid._item_to_algorithm["AAPL"] = 0

    with pytest.raises(ValueError, match="Reward must be non-negative"):
        hybrid.update_weights_from_feedback("user_123", "AAPL", -0.5, {})


def test_hybrid_get_algorithm_weights() -> None:
    """Test getting current algorithm weights."""
    recommenders = [MagicMock(), MagicMock(), MagicMock()]
    for idx, rec in enumerate(recommenders):
        rec.algorithm_name = f"algo{idx}"

    hybrid = HybridRecommender(recommenders, use_bandit=False)
    weights = hybrid.get_algorithm_weights()

    assert len(weights) == 3
    assert "algo0" in weights
    assert "algo1" in weights
    assert "algo2" in weights


# ContextualBandit Tests


def test_contextual_bandit_init() -> None:
    """Test contextual bandit initialization."""
    bandit = ContextualBandit(n_arms=4, alpha=0.2)

    assert bandit.n_arms == 4
    assert bandit.alpha == 0.2
    assert len(bandit.counts) == 4
    assert len(bandit.values) == 4
    assert all(c == 0 for c in bandit.counts)
    assert all(v == 0.0 for v in bandit.values)


def test_contextual_bandit_init_invalid_arms() -> None:
    """Test that invalid number of arms raises error."""
    with pytest.raises(ValueError, match="n_arms must be at least 1"):
        ContextualBandit(n_arms=0)


def test_contextual_bandit_init_negative_alpha() -> None:
    """Test that negative alpha raises error."""
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        ContextualBandit(n_arms=3, alpha=-0.1)


def test_contextual_bandit_select_unexplored() -> None:
    """Test that unexplored arms are selected first."""
    bandit = ContextualBandit(n_arms=3)

    # Should select each arm once before exploring
    selected = set()
    for _ in range(3):
        arm = bandit.select_arm()
        selected.add(arm)
        bandit.update(None, arm, 0.5)

    assert len(selected) == 3


def test_contextual_bandit_select_with_context() -> None:
    """Test arm selection with context."""
    bandit = ContextualBandit(n_arms=3)
    context = {"risk_tolerance": "high", "time_of_day": "morning"}

    arm = bandit.select_arm(context)

    assert 0 <= arm < 3


def test_contextual_bandit_update() -> None:
    """Test updating bandit with rewards."""
    bandit = ContextualBandit(n_arms=3)

    # Update arm 0
    bandit.update(None, 0, 1.0)

    assert bandit.counts[0] == 1
    assert bandit.values[0] == 1.0


def test_contextual_bandit_update_running_average() -> None:
    """Test that values are updated as running average."""
    bandit = ContextualBandit(n_arms=2)

    bandit.update(None, 0, 1.0)
    bandit.update(None, 0, 0.5)

    assert bandit.counts[0] == 2
    assert abs(bandit.values[0] - 0.75) < 0.001  # (1.0 + 0.5) / 2


def test_contextual_bandit_update_invalid_arm() -> None:
    """Test that invalid arm index raises error."""
    bandit = ContextualBandit(n_arms=3)

    with pytest.raises(ValueError, match="Invalid arm"):
        bandit.update(None, 5, 0.5)


def test_contextual_bandit_update_negative_reward() -> None:
    """Test that negative reward raises error."""
    bandit = ContextualBandit(n_arms=3)

    with pytest.raises(ValueError, match="Reward must be non-negative"):
        bandit.update(None, 0, -0.5)


def test_contextual_bandit_get_weights() -> None:
    """Test getting weight distribution."""
    bandit = ContextualBandit(n_arms=3)

    # Update with different rewards
    bandit.update(None, 0, 1.0)
    bandit.update(None, 1, 0.5)
    bandit.update(None, 2, 0.3)

    weights = bandit.get_weights()

    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 0.001
    assert weights[0] >= weights[1] >= weights[2]  # Higher reward = higher weight


def test_contextual_bandit_get_weights_no_data() -> None:
    """Test getting weights with no data returns uniform distribution."""
    bandit = ContextualBandit(n_arms=3)

    weights = bandit.get_weights()

    assert len(weights) == 3
    assert all(abs(w - 1.0 / 3) < 0.001 for w in weights)


def test_contextual_bandit_context_specific_weights() -> None:
    """Test that context-specific statistics are maintained."""
    bandit = ContextualBandit(n_arms=2)

    context1 = {"risk": "high"}
    context2 = {"risk": "low"}

    # Different rewards for different contexts
    bandit.update(context1, 0, 1.0)
    bandit.update(context2, 1, 1.0)

    # Should have context-specific statistics
    assert len(bandit.context_weights) == 2


def test_contextual_bandit_get_statistics() -> None:
    """Test getting bandit statistics."""
    bandit = ContextualBandit(n_arms=3)
    bandit.update(None, 0, 1.0)
    bandit.update(None, 1, 0.5)

    stats = bandit.get_statistics()

    assert "counts" in stats
    assert "values" in stats
    assert len(stats["counts"]) == 3
    assert stats["counts"][0] == 1
    assert stats["counts"][1] == 1


# ThompsonSamplingBandit Tests


def test_thompson_bandit_init() -> None:
    """Test Thompson sampling bandit initialization."""
    bandit = ThompsonSamplingBandit(n_arms=4)

    assert bandit.n_arms == 4
    assert len(bandit.alpha) == 4
    assert len(bandit.beta) == 4
    assert all(a == 1.0 for a in bandit.alpha)
    assert all(b == 1.0 for b in bandit.beta)


def test_thompson_bandit_init_invalid_arms() -> None:
    """Test that invalid number of arms raises error."""
    with pytest.raises(ValueError, match="n_arms must be at least 1"):
        ThompsonSamplingBandit(n_arms=0)


def test_thompson_bandit_select_arm() -> None:
    """Test arm selection."""
    bandit = ThompsonSamplingBandit(n_arms=3)

    arm = bandit.select_arm()

    assert 0 <= arm < 3


def test_thompson_bandit_update_success() -> None:
    """Test updating with successful outcome."""
    bandit = ThompsonSamplingBandit(n_arms=3)

    initial_alpha = bandit.alpha[0]
    bandit.update(0, 0.8)  # Success (> 0.5)

    assert bandit.alpha[0] == initial_alpha + 1.0
    assert bandit.beta[0] == 1.0  # Beta unchanged


def test_thompson_bandit_update_failure() -> None:
    """Test updating with failed outcome."""
    bandit = ThompsonSamplingBandit(n_arms=3)

    initial_beta = bandit.beta[0]
    bandit.update(0, 0.3)  # Failure (<= 0.5)

    assert bandit.alpha[0] == 1.0  # Alpha unchanged
    assert bandit.beta[0] == initial_beta + 1.0


def test_thompson_bandit_update_invalid_arm() -> None:
    """Test that invalid arm raises error."""
    bandit = ThompsonSamplingBandit(n_arms=3)

    with pytest.raises(ValueError, match="Invalid arm"):
        bandit.update(5, 0.5)


def test_thompson_bandit_update_negative_reward() -> None:
    """Test that negative reward raises error."""
    bandit = ThompsonSamplingBandit(n_arms=3)

    with pytest.raises(ValueError, match="Reward must be non-negative"):
        bandit.update(0, -0.5)


def test_thompson_bandit_get_expected_values() -> None:
    """Test getting expected values."""
    bandit = ThompsonSamplingBandit(n_arms=3)

    # Update to create differences
    bandit.update(0, 1.0)  # Success
    bandit.update(1, 0.3)  # Failure

    expected = bandit.get_expected_values()

    assert len(expected) == 3
    assert expected[0] > expected[1]  # Arm 0 should have higher expected value


def test_thompson_bandit_get_weights() -> None:
    """Test getting weight distribution."""
    bandit = ThompsonSamplingBandit(n_arms=3)

    weights = bandit.get_weights()

    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 0.001


def test_thompson_bandit_get_statistics() -> None:
    """Test getting bandit statistics."""
    bandit = ThompsonSamplingBandit(n_arms=3)
    bandit.update(0, 1.0)

    stats = bandit.get_statistics()

    assert "alpha" in stats
    assert "beta" in stats
    assert "expected_values" in stats


# FeedbackProcessor Tests


@pytest.mark.asyncio
async def test_feedback_processor_init() -> None:
    """Test feedback processor initialization."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    assert processor.session is mock_session
    assert processor.get_impression_count() == 0


@pytest.mark.asyncio
async def test_feedback_record_impression() -> None:
    """Test recording an impression."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    impression_id = await processor.record_impression(
        user_id="user_123",
        item_id="AAPL",
        algorithm="content_based",
        position=1,
        context={"time": "morning"},
    )

    assert impression_id is not None
    assert processor.get_impression_count() == 1


@pytest.mark.asyncio
async def test_feedback_record_impression_invalid_position() -> None:
    """Test that invalid position raises error."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    with pytest.raises(ValueError, match="Position must be positive"):
        await processor.record_impression(
            user_id="user_123",
            item_id="AAPL",
            algorithm="content_based",
            position=0,
            context={},
        )


@pytest.mark.asyncio
async def test_feedback_record_click() -> None:
    """Test recording a click."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    impression_id = await processor.record_impression(
        user_id="user_123",
        item_id="AAPL",
        algorithm="content_based",
        position=1,
        context={},
    )

    await processor.record_click(impression_id, datetime.now())

    assert processor.get_click_count() == 1


@pytest.mark.asyncio
async def test_feedback_record_click_invalid_impression() -> None:
    """Test that invalid impression ID raises error."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    with pytest.raises(ValueError, match="Impression .* not found"):
        await processor.record_click("invalid_id", datetime.now())


@pytest.mark.asyncio
async def test_feedback_record_outcome() -> None:
    """Test recording an outcome."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    impression_id = await processor.record_impression(
        user_id="user_123",
        item_id="AAPL",
        algorithm="content_based",
        position=1,
        context={},
    )

    await processor.record_outcome(impression_id, "trade", outcome_value=100.0)

    assert processor.get_outcome_count() == 1


@pytest.mark.asyncio
async def test_feedback_record_outcome_invalid_impression() -> None:
    """Test that invalid impression raises error."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    with pytest.raises(ValueError, match="Impression .* not found"):
        await processor.record_outcome("invalid_id", "trade")


@pytest.mark.asyncio
async def test_feedback_record_outcome_invalid_type() -> None:
    """Test that invalid outcome type raises error."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    impression_id = await processor.record_impression(
        user_id="user_123", item_id="AAPL", algorithm="test", position=1, context={}
    )

    with pytest.raises(ValueError, match="Invalid outcome_type"):
        await processor.record_outcome(impression_id, "invalid_type")


@pytest.mark.asyncio
async def test_feedback_calculate_reward_no_interaction() -> None:
    """Test reward calculation with no interaction."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    impression_id = await processor.record_impression(
        user_id="user_123", item_id="AAPL", algorithm="test", position=1, context={}
    )

    reward = await processor.calculate_reward(impression_id)

    assert reward == 0.0


@pytest.mark.asyncio
async def test_feedback_calculate_reward_click_only() -> None:
    """Test reward calculation with click only."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    impression_id = await processor.record_impression(
        user_id="user_123", item_id="AAPL", algorithm="test", position=1, context={}
    )
    await processor.record_click(impression_id, datetime.now())

    reward = await processor.calculate_reward(impression_id)

    assert reward == 0.3


@pytest.mark.asyncio
async def test_feedback_calculate_reward_watchlist() -> None:
    """Test reward calculation for watchlist add."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    impression_id = await processor.record_impression(
        user_id="user_123", item_id="AAPL", algorithm="test", position=1, context={}
    )
    await processor.record_click(impression_id, datetime.now())
    await processor.record_outcome(impression_id, "watchlist_add")

    reward = await processor.calculate_reward(impression_id)

    assert reward == 0.5


@pytest.mark.asyncio
async def test_feedback_calculate_reward_profitable_trade() -> None:
    """Test reward calculation for profitable trade."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    impression_id = await processor.record_impression(
        user_id="user_123", item_id="AAPL", algorithm="test", position=1, context={}
    )
    await processor.record_outcome(impression_id, "trade", outcome_value=150.0)

    reward = await processor.calculate_reward(impression_id)

    assert reward == 1.0


@pytest.mark.asyncio
async def test_feedback_calculate_reward_losing_trade() -> None:
    """Test reward calculation for losing trade."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    impression_id = await processor.record_impression(
        user_id="user_123", item_id="AAPL", algorithm="test", position=1, context={}
    )
    await processor.record_outcome(impression_id, "trade", outcome_value=-50.0)

    reward = await processor.calculate_reward(impression_id)

    assert reward == 0.2


@pytest.mark.asyncio
async def test_feedback_get_algorithm_performance() -> None:
    """Test getting algorithm performance metrics."""
    mock_session = MagicMock()
    processor = FeedbackProcessor(mock_session)

    # Create several impressions
    for i in range(3):
        impression_id = await processor.record_impression(
            user_id=f"user_{i}",
            item_id=f"ITEM{i}",
            algorithm="algo1",
            position=1,
            context={},
        )
        if i < 2:
            await processor.record_click(impression_id, datetime.now())

    performance = await processor.get_algorithm_performance(days=30)

    assert "algo1" in performance
    assert performance["algo1"]["impressions"] == 3.0
    assert performance["algo1"]["clicks"] == 2.0
    assert abs(performance["algo1"]["ctr"] - 2.0 / 3) < 0.001


# AntiHerdingFilter Tests


def test_anti_herding_filter_init() -> None:
    """Test anti-herding filter initialization."""
    filter = AntiHerdingFilter(herding_threshold=0.3)

    assert filter.herding_threshold == 0.3


def test_anti_herding_filter_init_invalid_threshold() -> None:
    """Test that invalid threshold raises error."""
    with pytest.raises(ValueError, match="herding_threshold must be in"):
        AntiHerdingFilter(herding_threshold=1.5)


def test_anti_herding_filter_no_popular_items(
    sample_recommendations: list[RecommendationItem],
) -> None:
    """Test filter with no popular items."""
    filter = AntiHerdingFilter()

    filtered = filter.filter(sample_recommendations, [])

    assert len(filtered) == len(sample_recommendations)


def test_anti_herding_filter_empty_recommendations() -> None:
    """Test filter with empty recommendations."""
    filter = AntiHerdingFilter()

    filtered = filter.filter([], ["AAPL", "GOOGL"])

    assert len(filtered) == 0


def test_anti_herding_filter_applies_penalty() -> None:
    """Test that penalty is applied to popular items."""
    filter = AntiHerdingFilter(herding_threshold=0.2)

    recommendations = [
        RecommendationItem(
            item_id="AAPL",
            item_type="signal",
            score=0.9,
            source="test",
            explanation="Test",
        )
    ]

    # Make AAPL very popular (above threshold)
    popular = ["AAPL"] * 50 + ["GOOGL"] * 10

    filtered = filter.filter(recommendations, popular)

    # Score should be reduced
    assert filtered[0].score < 0.9
    assert "anti_herding_penalty" in filtered[0].metadata


def test_anti_herding_filter_below_threshold() -> None:
    """Test that items below threshold are not penalized."""
    filter = AntiHerdingFilter(herding_threshold=0.5)

    recommendations = [
        RecommendationItem(
            item_id="AAPL",
            item_type="signal",
            score=0.9,
            source="test",
            explanation="Test",
        )
    ]

    # AAPL is popular but below threshold
    popular = ["AAPL"] * 20 + ["GOOGL"] * 30 + ["MSFT"] * 50

    filtered = filter.filter(recommendations, popular)

    # Score should not be changed
    assert filtered[0].score == 0.9


def test_anti_herding_calculate_penalty() -> None:
    """Test penalty calculation."""
    filter = AntiHerdingFilter(herding_threshold=0.3)

    popular = ["AAPL"] * 60 + ["GOOGL"] * 40

    penalty = filter._calculate_popularity_penalty("AAPL", popular)

    assert penalty > 0.0
    assert penalty <= 0.5  # Max penalty is 50%


def test_anti_herding_get_popularity_stats() -> None:
    """Test getting popularity statistics."""
    filter = AntiHerdingFilter()

    popular = ["AAPL"] * 30 + ["GOOGL"] * 20 + ["MSFT"] * 10

    stats = filter.get_popularity_stats(popular)

    assert stats["total_items"] == 60
    assert stats["unique_items"] == 3
    assert stats["max_frequency"] == 30
    assert abs(stats["max_popularity"] - 0.5) < 0.001


def test_anti_herding_get_popularity_stats_empty() -> None:
    """Test getting popularity statistics with empty list."""
    filter = AntiHerdingFilter()

    stats = filter.get_popularity_stats([])

    assert stats["total_items"] == 0
    assert stats["unique_items"] == 0
