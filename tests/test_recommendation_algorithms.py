"""Tests for recommendation algorithms.

This module tests all recommendation algorithm implementations including:
- Collaborative filtering (user-based and item-based)
- Content-based filtering
- Knowledge-based recommendations
- Base functionality (filtering, diversity)
"""

from __future__ import annotations

import time

import polars as pl
import pytest

from signalforge.recommendation.algorithms import (
    CollaborativeFilteringRecommender,
    ContentBasedRecommender,
    KnowledgeBasedRecommender,
    RecommendationItem,
    RecommendationRequest,
)
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
        viewed_sectors={"Technology": 10, "Healthcare": 5, "Finance": 2},
        viewed_symbols={"AAPL": 8, "GOOGL": 5, "MSFT": 3},
        avg_holding_period=25.0,
        preferred_volatility=0.4,
    )


@pytest.fixture
def recommendation_request() -> RecommendationRequest:
    """Create a sample recommendation request."""
    return RecommendationRequest(
        user_id="user_123",
        item_types=["signal", "stock"],
        limit=10,
        exclude_seen=True,
        context={"event_driven": True},
    )


@pytest.fixture
def interaction_data() -> pl.DataFrame:
    """Create sample interaction data for collaborative filtering."""
    data = {
        "user_id": ["user_1", "user_1", "user_1", "user_2", "user_2", "user_2", "user_3", "user_3"],
        "item_id": ["item_A", "item_B", "item_C", "item_A", "item_B", "item_D", "item_C", "item_D"],
        "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 5.0, 4.0],
        "timestamp": [1, 2, 3, 4, 5, 6, 7, 8],
    }
    return pl.DataFrame(data)


@pytest.fixture
def item_features_data() -> pl.DataFrame:
    """Create sample item features for content-based filtering."""
    current_time = int(time.time())
    data = {
        "item_id": ["item_A", "item_B", "item_C", "item_D"],
        "item_type": ["signal", "stock", "signal", "stock"],
        "sector": ["Technology", "Healthcare", "Technology", "Finance"],
        "market_cap": [1e12, 5e11, 2e12, 3e11],
        "volatility": [0.3, 0.2, 0.5, 0.4],
        "signal_type": ["momentum", "value", "momentum", "growth"],
        "timestamp": [current_time - 3600, current_time - 7200, current_time - 1800, current_time - 10800],
    }
    return pl.DataFrame(data)


@pytest.fixture
def knowledge_base_data() -> pl.DataFrame:
    """Create sample data for knowledge-based recommender."""
    data = {
        "item_id": ["item_A", "item_B", "item_C", "item_D"],
        "item_type": ["signal", "stock", "signal", "stock"],
        "sector": ["Technology", "Healthcare", "Technology", "Finance"],
        "risk_level": ["medium", "low", "high", "medium"],
        "volatility": [0.3, 0.2, 0.6, 0.4],
        "market_cap": [1e12, 5e11, 2e12, 3e11],
        "upcoming_events": [["earnings"], [], ["FOMC"], ["earnings", "split"]],
    }
    return pl.DataFrame(data)


# Tests for RecommendationRequest


def test_recommendation_request_creation() -> None:
    """Test creation of recommendation request."""
    request = RecommendationRequest(
        user_id="user_123",
        item_types=["signal"],
        limit=5,
        exclude_seen=False,
    )
    assert request.user_id == "user_123"
    assert request.item_types == ["signal"]
    assert request.limit == 5
    assert request.exclude_seen is False


def test_recommendation_request_defaults() -> None:
    """Test default values in recommendation request."""
    request = RecommendationRequest(user_id="user_123")
    assert request.item_types is None
    assert request.limit == 10
    assert request.exclude_seen is True
    assert request.context is None


def test_recommendation_request_validation() -> None:
    """Test validation of recommendation request."""
    with pytest.raises(ValueError):
        RecommendationRequest(user_id="", limit=5)

    with pytest.raises(ValueError):
        RecommendationRequest(user_id="user_123", limit=0)

    with pytest.raises(ValueError):
        RecommendationRequest(user_id="user_123", limit=200)


# Tests for RecommendationItem


def test_recommendation_item_creation() -> None:
    """Test creation of recommendation item."""
    item = RecommendationItem(
        item_id="item_123",
        item_type="signal",
        score=0.85,
        source="collaborative",
        explanation="Users similar to you liked this",
        metadata={"sector": "Technology"},
    )
    assert item.item_id == "item_123"
    assert item.item_type == "signal"
    assert item.score == 0.85
    assert item.source == "collaborative"


def test_recommendation_item_score_validation() -> None:
    """Test score validation in recommendation item."""
    with pytest.raises(ValueError):
        RecommendationItem(
            item_id="item_123",
            item_type="signal",
            score=1.5,
            source="test",
            explanation="test",
        )

    with pytest.raises(ValueError):
        RecommendationItem(
            item_id="item_123",
            item_type="signal",
            score=-0.1,
            source="test",
            explanation="test",
        )


def test_recommendation_item_empty_string_validation() -> None:
    """Test empty string validation in recommendation item."""
    with pytest.raises(ValueError):
        RecommendationItem(
            item_id="",
            item_type="signal",
            score=0.5,
            source="test",
            explanation="test",
        )


# Tests for CollaborativeFilteringRecommender


def test_collaborative_recommender_initialization() -> None:
    """Test initialization of collaborative filtering recommender."""
    recommender = CollaborativeFilteringRecommender(method="user_based", n_neighbors=10)
    assert recommender.method == "user_based"
    assert recommender.n_neighbors == 10
    assert recommender.algorithm_name == "collaborative"


def test_collaborative_recommender_invalid_method() -> None:
    """Test invalid method raises error."""
    with pytest.raises(ValueError):
        CollaborativeFilteringRecommender(method="invalid_method")  # type: ignore[arg-type]


def test_collaborative_recommender_invalid_neighbors() -> None:
    """Test invalid n_neighbors raises error."""
    with pytest.raises(ValueError):
        CollaborativeFilteringRecommender(n_neighbors=0)

    with pytest.raises(ValueError):
        CollaborativeFilteringRecommender(n_neighbors=-5)


@pytest.mark.asyncio
async def test_collaborative_training(interaction_data: pl.DataFrame) -> None:
    """Test training collaborative filtering model."""
    recommender = CollaborativeFilteringRecommender()
    await recommender.train(interaction_data)

    assert len(recommender.user_items) == 3
    assert len(recommender.item_users) == 4
    assert "user_1" in recommender.user_items
    assert "item_A" in recommender.item_users


@pytest.mark.asyncio
async def test_collaborative_training_invalid_data() -> None:
    """Test training with invalid data raises error."""
    recommender = CollaborativeFilteringRecommender()
    invalid_data = pl.DataFrame({"user_id": ["user_1"], "wrong_column": ["value"]})

    with pytest.raises(ValueError):
        await recommender.train(invalid_data)


@pytest.mark.asyncio
async def test_collaborative_user_similarity(interaction_data: pl.DataFrame) -> None:
    """Test user similarity computation."""
    # Use min_common_items=2 to allow similarity with 2 common items
    recommender = CollaborativeFilteringRecommender(min_common_items=2)
    await recommender.train(interaction_data)

    user1_items = recommender.user_items["user_1"]
    user2_items = recommender.user_items["user_2"]

    similarity = recommender._compute_user_similarity(user1_items, user2_items)
    assert 0.0 <= similarity <= 1.0
    assert similarity > 0.0  # They have common items (item_A and item_B)


@pytest.mark.asyncio
async def test_collaborative_item_similarity(interaction_data: pl.DataFrame) -> None:
    """Test item similarity computation."""
    recommender = CollaborativeFilteringRecommender()
    await recommender.train(interaction_data)

    item_a_users = recommender.item_users["item_A"]
    item_b_users = recommender.item_users["item_B"]

    similarity = recommender._compute_item_similarity(item_a_users, item_b_users)
    assert 0.0 <= similarity <= 1.0


@pytest.mark.asyncio
async def test_collaborative_find_neighbors(interaction_data: pl.DataFrame) -> None:
    """Test finding nearest neighbors."""
    recommender = CollaborativeFilteringRecommender(n_neighbors=2)
    await recommender.train(interaction_data)

    neighbors = recommender._find_neighbors("user_1", k=2)
    assert len(neighbors) <= 2
    assert all(sim >= 0.0 for _, sim in neighbors)


@pytest.mark.asyncio
async def test_collaborative_user_based_recommendation(
    interaction_data: pl.DataFrame,
    recommendation_request: RecommendationRequest,
    explicit_profile: ExplicitProfile,
) -> None:
    """Test user-based collaborative filtering recommendations."""
    recommender = CollaborativeFilteringRecommender(method="user_based")
    await recommender.train(interaction_data)

    # Update request for existing user
    request = RecommendationRequest(user_id="user_1", limit=5)
    recommendations = await recommender.recommend(request, explicit_profile)

    # user_1 has rated A, B, C, so should recommend D
    assert isinstance(recommendations, list)


@pytest.mark.asyncio
async def test_collaborative_item_based_recommendation(
    interaction_data: pl.DataFrame,
    recommendation_request: RecommendationRequest,
    explicit_profile: ExplicitProfile,
) -> None:
    """Test item-based collaborative filtering recommendations."""
    recommender = CollaborativeFilteringRecommender(method="item_based")
    await recommender.train(interaction_data)

    request = RecommendationRequest(user_id="user_1", limit=5)
    recommendations = await recommender.recommend(request, explicit_profile)

    assert isinstance(recommendations, list)


@pytest.mark.asyncio
async def test_collaborative_unknown_user(
    interaction_data: pl.DataFrame,
    explicit_profile: ExplicitProfile,
) -> None:
    """Test recommendation for unknown user."""
    recommender = CollaborativeFilteringRecommender()
    await recommender.train(interaction_data)

    request = RecommendationRequest(user_id="unknown_user", limit=5)
    recommendations = await recommender.recommend(request, explicit_profile)

    assert recommendations == []


# Tests for ContentBasedRecommender


def test_content_based_recommender_initialization() -> None:
    """Test initialization of content-based recommender."""
    recommender = ContentBasedRecommender()
    assert recommender.algorithm_name == "content_based"
    assert "sector" in recommender.feature_weights
    assert sum(recommender.feature_weights.values()) == pytest.approx(1.0, rel=1e-2)


def test_content_based_custom_weights() -> None:
    """Test custom feature weights."""
    weights = {
        "sector": 0.4,
        "market_cap": 0.1,
        "volatility": 0.2,
        "signal_type": 0.1,
        "recency": 0.2,
    }
    recommender = ContentBasedRecommender(feature_weights=weights)
    assert recommender.feature_weights["sector"] == 0.4


@pytest.mark.asyncio
async def test_content_based_training(item_features_data: pl.DataFrame) -> None:
    """Test training content-based model."""
    recommender = ContentBasedRecommender()
    await recommender.train(item_features_data)

    assert len(recommender.item_features) == 4
    assert "item_A" in recommender.item_features
    assert recommender.item_features["item_A"]["sector"] == "Technology"


@pytest.mark.asyncio
async def test_content_based_training_invalid_data() -> None:
    """Test training with invalid data."""
    recommender = ContentBasedRecommender()
    invalid_data = pl.DataFrame({"wrong_column": ["value"]})

    with pytest.raises(ValueError):
        await recommender.train(invalid_data)


@pytest.mark.asyncio
async def test_content_based_recommendation(
    item_features_data: pl.DataFrame,
    recommendation_request: RecommendationRequest,
    explicit_profile: ExplicitProfile,
    implicit_profile: ImplicitProfile,
) -> None:
    """Test content-based recommendations."""
    recommender = ContentBasedRecommender()
    await recommender.train(item_features_data)

    recommendations = await recommender.recommend(
        recommendation_request, explicit_profile, implicit_profile
    )

    assert isinstance(recommendations, list)
    assert len(recommendations) <= recommendation_request.limit
    assert all(isinstance(item, RecommendationItem) for item in recommendations)


@pytest.mark.asyncio
async def test_content_based_sector_alignment(
    item_features_data: pl.DataFrame,
    explicit_profile: ExplicitProfile,
) -> None:
    """Test sector alignment scoring."""
    recommender = ContentBasedRecommender()
    await recommender.train(item_features_data)

    # Technology is in preferred sectors
    score1 = recommender._get_sector_alignment("Technology", ["Technology", "Healthcare"])
    assert score1 == 1.0

    # Finance is not
    score2 = recommender._get_sector_alignment("Finance", ["Technology", "Healthcare"])
    assert score2 == 0.0


@pytest.mark.asyncio
async def test_content_based_volatility_match() -> None:
    """Test volatility matching scoring."""
    recommender = ContentBasedRecommender()

    # Perfect match
    score1 = recommender._get_volatility_match(0.5, 0.5)
    assert score1 == 1.0

    # Partial match
    score2 = recommender._get_volatility_match(0.3, 0.5)
    assert 0.0 < score2 < 1.0

    # No match
    score3 = recommender._get_volatility_match(0.0, 1.0)
    assert score3 == 0.0


@pytest.mark.asyncio
async def test_content_based_recency_score() -> None:
    """Test recency scoring."""
    recommender = ContentBasedRecommender()

    # Recent timestamp
    recent_time = int(time.time()) - 3600  # 1 hour ago
    score1 = recommender._get_recency_score(recent_time)
    assert score1 > 0.8

    # Old timestamp
    old_time = int(time.time()) - (30 * 24 * 3600)  # 30 days ago
    score2 = recommender._get_recency_score(old_time)
    assert score2 < 0.5


# Tests for KnowledgeBasedRecommender


def test_knowledge_based_recommender_initialization() -> None:
    """Test initialization of knowledge-based recommender."""
    recommender = KnowledgeBasedRecommender()
    assert recommender.algorithm_name == "knowledge_based"
    assert len(recommender.rules) > 0


@pytest.mark.asyncio
async def test_knowledge_based_training(knowledge_base_data: pl.DataFrame) -> None:
    """Test updating item database."""
    recommender = KnowledgeBasedRecommender()
    await recommender.train(knowledge_base_data)

    assert len(recommender.item_database) == 4
    assert "item_A" in recommender.item_database


@pytest.mark.asyncio
async def test_knowledge_based_training_invalid_data() -> None:
    """Test training with invalid data."""
    recommender = KnowledgeBasedRecommender()
    invalid_data = pl.DataFrame({"wrong_column": ["value"]})

    with pytest.raises(ValueError):
        await recommender.train(invalid_data)


@pytest.mark.asyncio
async def test_knowledge_based_recommendation(
    knowledge_base_data: pl.DataFrame,
    recommendation_request: RecommendationRequest,
    explicit_profile: ExplicitProfile,
) -> None:
    """Test knowledge-based recommendations."""
    recommender = KnowledgeBasedRecommender()
    await recommender.train(knowledge_base_data)

    recommendations = await recommender.recommend(recommendation_request, explicit_profile)

    assert isinstance(recommendations, list)
    assert len(recommendations) <= recommendation_request.limit


@pytest.mark.asyncio
async def test_knowledge_based_risk_filtering(
    knowledge_base_data: pl.DataFrame,
    explicit_profile: ExplicitProfile,
) -> None:
    """Test risk tolerance filtering."""
    recommender = KnowledgeBasedRecommender()
    await recommender.train(knowledge_base_data)

    # Low risk profile should filter high risk items
    low_risk_profile = ExplicitProfile(
        risk_tolerance="low",
        investment_horizon=30,
        preferred_sectors=["Technology"],
        watchlist=["AAPL"],
    )

    request = RecommendationRequest(user_id="user_123", limit=10)
    recommendations = await recommender.recommend(request, low_risk_profile)

    # Should not include item_C which has high risk
    recommendation_ids = {r.item_id for r in recommendations}
    assert "item_C" not in recommendation_ids


@pytest.mark.asyncio
async def test_knowledge_based_event_boosting(
    knowledge_base_data: pl.DataFrame,
    explicit_profile: ExplicitProfile,
) -> None:
    """Test event-driven recommendation boosting."""
    recommender = KnowledgeBasedRecommender()
    await recommender.train(knowledge_base_data)

    # Request with event context
    request = RecommendationRequest(
        user_id="user_123",
        limit=10,
        context={"event_driven": True},
    )

    recommendations = await recommender.recommend(request, explicit_profile)

    # Items with events should have boosted scores
    assert isinstance(recommendations, list)


@pytest.mark.asyncio
async def test_knowledge_based_diversification(
    knowledge_base_data: pl.DataFrame,
) -> None:
    """Test diversification rules."""
    recommender = KnowledgeBasedRecommender()
    await recommender.train(knowledge_base_data)

    # Profile with heavy tech concentration
    tech_heavy_profile = ExplicitProfile(
        risk_tolerance="medium",
        investment_horizon=30,
        preferred_sectors=["Technology"],
        watchlist=["item_A", "item_C", "TECH1", "TECH2", "TECH3"],
    )

    request = RecommendationRequest(user_id="user_123", limit=10)
    recommendations = await recommender.recommend(request, tech_heavy_profile)

    # Should exist (diversification applied)
    assert isinstance(recommendations, list)


# Tests for BaseRecommender functionality


def test_filter_seen_items() -> None:
    """Test filtering of seen items."""
    recommender = ContentBasedRecommender()

    items = [
        RecommendationItem(
            item_id="item_1", item_type="signal", score=0.9, source="test", explanation="test"
        ),
        RecommendationItem(
            item_id="item_2", item_type="signal", score=0.8, source="test", explanation="test"
        ),
        RecommendationItem(
            item_id="item_3", item_type="signal", score=0.7, source="test", explanation="test"
        ),
    ]

    seen_ids = {"item_2"}
    filtered = recommender.filter_seen(items, seen_ids)

    assert len(filtered) == 2
    assert "item_2" not in {item.item_id for item in filtered}


def test_filter_seen_empty_set() -> None:
    """Test filtering with empty seen set."""
    recommender = ContentBasedRecommender()

    items = [
        RecommendationItem(
            item_id="item_1", item_type="signal", score=0.9, source="test", explanation="test"
        ),
    ]

    filtered = recommender.filter_seen(items, set())
    assert len(filtered) == 1


def test_apply_diversity() -> None:
    """Test diversity application."""
    recommender = ContentBasedRecommender()

    items = [
        RecommendationItem(
            item_id="item_1",
            item_type="signal",
            score=0.9,
            source="test",
            explanation="test",
            metadata={"sector": "Tech"},
        ),
        RecommendationItem(
            item_id="item_2",
            item_type="signal",
            score=0.85,
            source="test",
            explanation="test",
            metadata={"sector": "Tech"},
        ),
        RecommendationItem(
            item_id="item_3",
            item_type="stock",
            score=0.8,
            source="test",
            explanation="test",
            metadata={"sector": "Healthcare"},
        ),
    ]

    diversified = recommender.apply_diversity(items, diversity_factor=0.5)
    assert len(diversified) == 3


def test_apply_diversity_invalid_factor() -> None:
    """Test diversity with invalid factor."""
    recommender = ContentBasedRecommender()

    items = [
        RecommendationItem(
            item_id="item_1", item_type="signal", score=0.9, source="test", explanation="test"
        ),
    ]

    with pytest.raises(ValueError):
        recommender.apply_diversity(items, diversity_factor=1.5)

    with pytest.raises(ValueError):
        recommender.apply_diversity(items, diversity_factor=-0.1)


def test_apply_diversity_zero_factor() -> None:
    """Test diversity with zero factor (no diversity)."""
    recommender = ContentBasedRecommender()

    items = [
        RecommendationItem(
            item_id="item_1", item_type="signal", score=0.9, source="test", explanation="test"
        ),
        RecommendationItem(
            item_id="item_2", item_type="signal", score=0.8, source="test", explanation="test"
        ),
    ]

    diversified = recommender.apply_diversity(items, diversity_factor=0.0)
    assert diversified == items


def test_item_similarity() -> None:
    """Test item similarity calculation."""
    recommender = ContentBasedRecommender()

    item1 = RecommendationItem(
        item_id="item_1",
        item_type="signal",
        score=0.9,
        source="test",
        explanation="test",
        metadata={"sector": "Tech", "risk": "medium"},
    )

    item2 = RecommendationItem(
        item_id="item_2",
        item_type="signal",
        score=0.8,
        source="test",
        explanation="test",
        metadata={"sector": "Tech", "risk": "medium"},
    )

    item3 = RecommendationItem(
        item_id="item_3",
        item_type="stock",
        score=0.7,
        source="test",
        explanation="test",
        metadata={"sector": "Healthcare", "risk": "low"},
    )

    # Same type and metadata
    similarity1 = recommender._item_similarity(item1, item2)
    assert similarity1 > 0.5

    # Different type and metadata
    similarity2 = recommender._item_similarity(item1, item3)
    assert similarity2 < similarity1

    # Same item
    similarity3 = recommender._item_similarity(item1, item1)
    assert similarity3 == 1.0


def test_calculate_diversity_score() -> None:
    """Test diversity score calculation."""
    recommender = ContentBasedRecommender()

    candidate = RecommendationItem(
        item_id="candidate",
        item_type="signal",
        score=0.8,
        source="test",
        explanation="test",
        metadata={"sector": "Tech"},
    )

    selected = [
        RecommendationItem(
            item_id="item_1",
            item_type="signal",
            score=0.9,
            source="test",
            explanation="test",
            metadata={"sector": "Tech"},
        ),
    ]

    diversity_score = recommender._calculate_diversity_score(candidate, selected)
    assert 0.0 <= diversity_score <= 1.0


def test_calculate_diversity_score_empty_selected() -> None:
    """Test diversity score with empty selected list."""
    recommender = ContentBasedRecommender()

    candidate = RecommendationItem(
        item_id="candidate",
        item_type="signal",
        score=0.8,
        source="test",
        explanation="test",
    )

    diversity_score = recommender._calculate_diversity_score(candidate, [])
    assert diversity_score == 1.0
