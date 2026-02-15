"""Tests for recommendation engine module.

This module tests the complete recommendation system including item modeling,
user profiling, ranking, and feed generation.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from signalforge.recommendation import (
    ExplicitProfile,
    FeedConfig,
    FeedGenerator,
    FeedItem,
    ImplicitProfile,
    ItemModel,
    RankedSignal,
    RankingConfig,
    RankingEngine,
    SignalFeatures,
    SignalItem,
    UserModel,
    UserProfile,
)


class TestSignalFeatures:
    """Tests for SignalFeatures dataclass."""

    def test_valid_signal_features(self) -> None:
        """Test creation of valid signal features."""
        features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )

        assert features.symbol == "AAPL"
        assert features.sector == "Technology"
        assert features.volatility == 0.25
        assert features.expected_return == 0.05
        assert features.holding_period == 5
        assert features.risk_level == "medium"
        assert features.sentiment_score == 0.7
        assert features.regime == "bull"

    def test_empty_symbol(self) -> None:
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            SignalFeatures(
                symbol="",
                sector="Technology",
                volatility=0.25,
                expected_return=0.05,
                holding_period=5,
                risk_level="medium",
                sentiment_score=0.7,
                regime="bull",
            )

    def test_negative_volatility(self) -> None:
        """Test that negative volatility raises ValueError."""
        with pytest.raises(ValueError, match="Volatility must be non-negative"):
            SignalFeatures(
                symbol="AAPL",
                sector="Technology",
                volatility=-0.1,
                expected_return=0.05,
                holding_period=5,
                risk_level="medium",
                sentiment_score=0.7,
                regime="bull",
            )

    def test_non_positive_holding_period(self) -> None:
        """Test that non-positive holding period raises ValueError."""
        with pytest.raises(ValueError, match="Holding period must be positive"):
            SignalFeatures(
                symbol="AAPL",
                sector="Technology",
                volatility=0.25,
                expected_return=0.05,
                holding_period=0,
                risk_level="medium",
                sentiment_score=0.7,
                regime="bull",
            )

    def test_invalid_sentiment_score(self) -> None:
        """Test that sentiment score outside [-1, 1] raises ValueError."""
        with pytest.raises(ValueError, match="Sentiment score must be between"):
            SignalFeatures(
                symbol="AAPL",
                sector="Technology",
                volatility=0.25,
                expected_return=0.05,
                holding_period=5,
                risk_level="medium",
                sentiment_score=1.5,
                regime="bull",
            )

    def test_invalid_risk_level(self) -> None:
        """Test that invalid risk level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid risk level"):
            SignalFeatures(
                symbol="AAPL",
                sector="Technology",
                volatility=0.25,
                expected_return=0.05,
                holding_period=5,
                risk_level="invalid",  # type: ignore[arg-type]
                sentiment_score=0.7,
                regime="bull",
            )

    def test_invalid_regime(self) -> None:
        """Test that invalid regime raises ValueError."""
        with pytest.raises(ValueError, match="Invalid regime"):
            SignalFeatures(
                symbol="AAPL",
                sector="Technology",
                volatility=0.25,
                expected_return=0.05,
                holding_period=5,
                risk_level="medium",
                sentiment_score=0.7,
                regime="invalid",  # type: ignore[arg-type]
            )


class TestSignalItem:
    """Tests for SignalItem dataclass."""

    def test_valid_signal_item(self) -> None:
        """Test creation of valid signal item."""
        features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )

        embedding = np.random.randn(32)
        created_at = datetime.utcnow()

        item = SignalItem(
            signal_id="sig_001", features=features, embedding=embedding, created_at=created_at
        )

        assert item.signal_id == "sig_001"
        assert item.features == features
        assert len(item.embedding) == 32
        assert item.created_at == created_at

    def test_empty_signal_id(self) -> None:
        """Test that empty signal_id raises ValueError."""
        features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )

        with pytest.raises(ValueError, match="Signal ID cannot be empty"):
            SignalItem(
                signal_id="",
                features=features,
                embedding=np.array([1.0, 2.0]),
                created_at=datetime.utcnow(),
            )

    def test_empty_embedding(self) -> None:
        """Test that empty embedding raises ValueError."""
        features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )

        with pytest.raises(ValueError, match="Embedding cannot be empty"):
            SignalItem(
                signal_id="sig_001",
                features=features,
                embedding=np.array([]),
                created_at=datetime.utcnow(),
            )

    def test_invalid_embedding_values(self) -> None:
        """Test that NaN or Inf in embedding raises ValueError."""
        features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )

        with pytest.raises(ValueError, match="contains invalid values"):
            SignalItem(
                signal_id="sig_001",
                features=features,
                embedding=np.array([1.0, np.nan, 3.0]),
                created_at=datetime.utcnow(),
            )


class TestItemModel:
    """Tests for ItemModel class."""

    def test_item_model_initialization(self) -> None:
        """Test ItemModel initialization."""
        model = ItemModel(embedding_dim=64)
        assert model.embedding_dim == 64

    def test_invalid_embedding_dim(self) -> None:
        """Test that non-positive embedding dimension raises ValueError."""
        with pytest.raises(ValueError, match="Embedding dimension must be positive"):
            ItemModel(embedding_dim=0)

    def test_extract_features(self) -> None:
        """Test feature extraction from signal data."""
        model = ItemModel()
        signal_data = {
            "symbol": "AAPL",
            "sector": "Technology",
            "volatility": 0.25,
            "expected_return": 0.05,
            "holding_period": 5,
            "risk_level": "medium",
            "sentiment_score": 0.7,
            "regime": "bull",
        }

        features = model.extract_features(signal_data)

        assert features.symbol == "AAPL"
        assert features.sector == "Technology"
        assert features.volatility == 0.25

    def test_extract_features_missing_field(self) -> None:
        """Test that missing required field raises KeyError."""
        model = ItemModel()
        signal_data = {
            "symbol": "AAPL",
            "sector": "Technology",
            # Missing other fields
        }

        with pytest.raises(KeyError, match="Missing required fields"):
            model.extract_features(signal_data)

    def test_create_embedding(self) -> None:
        """Test embedding creation from features."""
        model = ItemModel(embedding_dim=32)
        features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )

        embedding = model.create_embedding(features)

        assert len(embedding) == 32
        assert np.all(np.isfinite(embedding))
        # Check L2 normalization
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6

    def test_create_signal_item(self) -> None:
        """Test complete signal item creation."""
        model = ItemModel(embedding_dim=32)
        features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )

        signal_item = model.create_signal_item("sig_001", features)

        assert signal_item.signal_id == "sig_001"
        assert signal_item.features == features
        assert len(signal_item.embedding) == 32

    def test_calculate_similarity(self) -> None:
        """Test similarity calculation between signal items."""
        model = ItemModel(embedding_dim=32)

        features1 = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )

        features2 = SignalFeatures(
            symbol="GOOGL",
            sector="Technology",
            volatility=0.30,
            expected_return=0.06,
            holding_period=7,
            risk_level="medium",
            sentiment_score=0.6,
            regime="bull",
        )

        item1 = model.create_signal_item("sig_001", features1)
        item2 = model.create_signal_item("sig_002", features2)

        similarity = model.calculate_similarity(item1, item2)

        assert -1.0 <= similarity <= 1.0
        # Similar tech stocks should have high similarity
        assert similarity > 0.5

    def test_calculate_similarity_dimension_mismatch(self) -> None:
        """Test that dimension mismatch raises ValueError."""
        model1 = ItemModel(embedding_dim=32)
        model2 = ItemModel(embedding_dim=64)

        features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )

        item1 = model1.create_signal_item("sig_001", features)
        item2 = model2.create_signal_item("sig_002", features)

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            model1.calculate_similarity(item1, item2)

    def test_find_similar(self) -> None:
        """Test finding similar signals."""
        model = ItemModel(embedding_dim=32)

        # Create query signal
        query_features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )
        query_item = model.create_signal_item("sig_query", query_features)

        # Create candidate signals
        candidates = []
        for i in range(10):
            features = SignalFeatures(
                symbol=f"STOCK{i}",
                sector="Technology" if i < 5 else "Healthcare",
                volatility=0.2 + i * 0.01,
                expected_return=0.03 + i * 0.01,
                holding_period=5 + i,
                risk_level="medium",
                sentiment_score=0.5 + i * 0.02,
                regime="bull",
            )
            candidates.append(model.create_signal_item(f"sig_{i}", features))

        similar = model.find_similar(query_item, candidates, top_k=5)

        assert len(similar) <= 5
        assert all(isinstance(item, SignalItem) for item, _ in similar)
        assert all(isinstance(score, float) for _, score in similar)
        # Scores should be in descending order
        scores = [score for _, score in similar]
        assert scores == sorted(scores, reverse=True)

    def test_find_similar_empty_candidates(self) -> None:
        """Test that empty candidates list raises ValueError."""
        model = ItemModel()
        features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )
        item = model.create_signal_item("sig_001", features)

        with pytest.raises(ValueError, match="Candidates list cannot be empty"):
            model.find_similar(item, [], top_k=5)

    def test_find_similar_invalid_top_k(self) -> None:
        """Test that non-positive top_k raises ValueError."""
        model = ItemModel()
        features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )
        item = model.create_signal_item("sig_001", features)
        candidates = [item]

        with pytest.raises(ValueError, match="top_k must be positive"):
            model.find_similar(item, candidates, top_k=0)


class TestExplicitProfile:
    """Tests for ExplicitProfile dataclass."""

    def test_valid_explicit_profile(self) -> None:
        """Test creation of valid explicit profile."""
        profile = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology", "Healthcare"],
            watchlist=["AAPL", "GOOGL"],
        )

        assert profile.risk_tolerance == "medium"
        assert profile.investment_horizon == 30
        assert len(profile.preferred_sectors) == 2
        assert len(profile.watchlist) == 2

    def test_invalid_risk_tolerance(self) -> None:
        """Test that invalid risk tolerance raises ValueError."""
        with pytest.raises(ValueError, match="Invalid risk tolerance"):
            ExplicitProfile(
                risk_tolerance="invalid",  # type: ignore[arg-type]
                investment_horizon=30,
                preferred_sectors=["Technology"],
                watchlist=["AAPL"],
            )

    def test_non_positive_investment_horizon(self) -> None:
        """Test that non-positive investment horizon raises ValueError."""
        with pytest.raises(ValueError, match="Investment horizon must be positive"):
            ExplicitProfile(
                risk_tolerance="medium",
                investment_horizon=0,
                preferred_sectors=["Technology"],
                watchlist=["AAPL"],
            )


class TestImplicitProfile:
    """Tests for ImplicitProfile dataclass."""

    def test_valid_implicit_profile(self) -> None:
        """Test creation of valid implicit profile."""
        profile = ImplicitProfile(
            viewed_sectors={"Technology": 10, "Healthcare": 5},
            viewed_symbols={"AAPL": 15, "GOOGL": 8},
            avg_holding_period=30.0,
            preferred_volatility=0.25,
        )

        assert profile.viewed_sectors["Technology"] == 10
        assert profile.viewed_symbols["AAPL"] == 15
        assert profile.avg_holding_period == 30.0
        assert profile.preferred_volatility == 0.25

    def test_negative_avg_holding_period(self) -> None:
        """Test that negative avg holding period raises ValueError."""
        with pytest.raises(ValueError, match="Average holding period cannot be negative"):
            ImplicitProfile(
                viewed_sectors={},
                viewed_symbols={},
                avg_holding_period=-10.0,
                preferred_volatility=0.25,
            )

    def test_negative_view_count(self) -> None:
        """Test that negative view counts raise ValueError."""
        with pytest.raises(ValueError, match="View count for sector.*cannot be negative"):
            ImplicitProfile(
                viewed_sectors={"Technology": -5},
                viewed_symbols={},
                avg_holding_period=30.0,
                preferred_volatility=0.25,
            )


class TestUserProfile:
    """Tests for UserProfile dataclass."""

    def test_valid_user_profile(self) -> None:
        """Test creation of valid user profile."""
        explicit = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )

        implicit = ImplicitProfile(
            viewed_sectors={"Technology": 10},
            viewed_symbols={"AAPL": 15},
            avg_holding_period=30.0,
            preferred_volatility=0.25,
        )

        embedding = np.random.randn(32)
        embedding = embedding / np.linalg.norm(embedding)

        profile = UserProfile(
            user_id="user_123", explicit=explicit, implicit=implicit, combined_embedding=embedding
        )

        assert profile.user_id == "user_123"
        assert profile.explicit == explicit
        assert profile.implicit == implicit
        assert len(profile.combined_embedding) == 32

    def test_empty_user_id(self) -> None:
        """Test that empty user_id raises ValueError."""
        explicit = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )
        implicit = ImplicitProfile()
        embedding = np.random.randn(32)

        with pytest.raises(ValueError, match="User ID cannot be empty"):
            UserProfile(
                user_id="", explicit=explicit, implicit=implicit, combined_embedding=embedding
            )


class TestUserModel:
    """Tests for UserModel class."""

    def test_user_model_initialization(self) -> None:
        """Test UserModel initialization."""
        model = UserModel(embedding_dim=64)
        assert model.embedding_dim == 64

    def test_from_preferences(self) -> None:
        """Test creating explicit profile from preferences."""
        model = UserModel()
        preferences = {
            "risk_tolerance": "medium",
            "investment_horizon": 30,
            "preferred_sectors": ["Technology", "Healthcare"],
            "watchlist": ["AAPL", "GOOGL"],
        }

        profile = model.from_preferences(preferences)

        assert profile.risk_tolerance == "medium"
        assert profile.investment_horizon == 30
        assert len(profile.preferred_sectors) == 2
        assert len(profile.watchlist) == 2

    def test_from_preferences_missing_field(self) -> None:
        """Test that missing required field raises KeyError."""
        model = UserModel()
        preferences = {
            "risk_tolerance": "medium",
            # Missing other fields
        }

        with pytest.raises(KeyError, match="Missing required fields"):
            model.from_preferences(preferences)

    def test_from_activity(self) -> None:
        """Test creating implicit profile from activity."""
        model = UserModel()
        activities = [
            {"action": "view", "symbol": "AAPL", "sector": "Technology", "volatility": 0.25},
            {"action": "view", "symbol": "GOOGL", "sector": "Technology", "volatility": 0.30},
            {"action": "click", "symbol": "MSFT", "sector": "Technology", "volatility": 0.20},
        ]

        profile = model.from_activity(activities)

        assert profile.viewed_sectors["Technology"] == 3
        assert profile.viewed_symbols["AAPL"] == 1
        assert profile.preferred_volatility > 0

    def test_from_activity_empty_list(self) -> None:
        """Test that empty activity list returns empty profile."""
        model = UserModel()
        profile = model.from_activity([])

        assert len(profile.viewed_sectors) == 0
        assert len(profile.viewed_symbols) == 0

    def test_combine_profiles(self) -> None:
        """Test combining explicit and implicit profiles."""
        model = UserModel(embedding_dim=32)

        explicit = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )

        implicit = ImplicitProfile(
            viewed_sectors={"Technology": 10},
            viewed_symbols={"AAPL": 15},
            avg_holding_period=30.0,
            preferred_volatility=0.25,
        )

        combined = model.combine_profiles(explicit, implicit, weight=0.7)

        assert len(combined) == 32
        assert np.all(np.isfinite(combined))
        # Check L2 normalization
        norm = np.linalg.norm(combined)
        assert abs(norm - 1.0) < 1e-6

    def test_combine_profiles_invalid_weight(self) -> None:
        """Test that invalid weight raises ValueError."""
        model = UserModel()
        explicit = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )
        implicit = ImplicitProfile()

        with pytest.raises(ValueError, match="Weight must be between 0.0 and 1.0"):
            model.combine_profiles(explicit, implicit, weight=1.5)

    def test_update_implicit(self) -> None:
        """Test updating implicit profile with new activity."""
        model = UserModel()

        profile = ImplicitProfile(
            viewed_sectors={"Technology": 5},
            viewed_symbols={"AAPL": 3},
            avg_holding_period=30.0,
            preferred_volatility=0.25,
        )

        activity = {"symbol": "GOOGL", "sector": "Technology", "volatility": 0.30}

        updated = model.update_implicit(profile, activity)

        assert updated.viewed_sectors["Technology"] == 6
        assert updated.viewed_symbols["GOOGL"] == 1
        assert updated.viewed_symbols["AAPL"] == 3

    def test_create_user_profile(self) -> None:
        """Test creating complete user profile."""
        model = UserModel(embedding_dim=32)

        explicit = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )

        implicit = ImplicitProfile(
            viewed_sectors={"Technology": 10},
            viewed_symbols={"AAPL": 15},
            avg_holding_period=30.0,
            preferred_volatility=0.25,
        )

        profile = model.create_user_profile("user_123", explicit, implicit, weight=0.7)

        assert profile.user_id == "user_123"
        assert profile.explicit == explicit
        assert profile.implicit == implicit
        assert len(profile.combined_embedding) == 32


class TestRankingConfig:
    """Tests for RankingConfig dataclass."""

    def test_valid_ranking_config(self) -> None:
        """Test creation of valid ranking config."""
        config = RankingConfig(
            relevance_weight=0.5,
            confidence_weight=0.3,
            recency_weight=0.2,
            diversity_penalty=0.1,
        )

        assert config.relevance_weight == 0.5
        assert config.confidence_weight == 0.3
        assert config.recency_weight == 0.2
        assert config.diversity_penalty == 0.1

    def test_invalid_weights(self) -> None:
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError, match="relevance_weight must be between"):
            RankingConfig(relevance_weight=1.5)

        with pytest.raises(ValueError, match="confidence_weight must be between"):
            RankingConfig(confidence_weight=-0.1)


class TestRankedSignal:
    """Tests for RankedSignal dataclass."""

    def test_valid_ranked_signal(self) -> None:
        """Test creation of valid ranked signal."""
        ranked = RankedSignal(
            signal_id="sig_001",
            score=0.85,
            relevance=0.9,
            confidence=0.8,
            recency=0.95,
            explanation="High relevance to your preferences",
        )

        assert ranked.signal_id == "sig_001"
        assert ranked.score == 0.85
        assert ranked.relevance == 0.9

    def test_invalid_score(self) -> None:
        """Test that invalid score raises ValueError."""
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            RankedSignal(
                signal_id="sig_001",
                score=1.5,
                relevance=0.9,
                confidence=0.8,
                recency=0.95,
                explanation="Test",
            )


class TestRankingEngine:
    """Tests for RankingEngine class."""

    def test_ranking_engine_initialization(self) -> None:
        """Test RankingEngine initialization."""
        config = RankingConfig()
        engine = RankingEngine(config)
        assert engine.config == config

    def test_calculate_relevance(self) -> None:
        """Test relevance calculation."""
        engine = RankingEngine()

        user_model = UserModel(embedding_dim=32)
        explicit = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )
        implicit = ImplicitProfile()
        user_profile = user_model.create_user_profile("user_123", explicit, implicit)

        item_model = ItemModel(embedding_dim=32)
        features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )
        signal_item = item_model.create_signal_item("sig_001", features)

        relevance = engine.calculate_relevance(user_profile, signal_item)

        assert 0.0 <= relevance <= 1.0
        # Should be high since signal matches user preferences
        assert relevance > 0.5

    def test_calculate_recency(self) -> None:
        """Test recency calculation."""
        engine = RankingEngine()

        item_model = ItemModel()
        features = SignalFeatures(
            symbol="AAPL",
            sector="Technology",
            volatility=0.25,
            expected_return=0.05,
            holding_period=5,
            risk_level="medium",
            sentiment_score=0.7,
            regime="bull",
        )

        # Recent signal
        recent_item = item_model.create_signal_item("sig_001", features)
        recent_recency = engine.calculate_recency(recent_item, max_age_hours=48)
        assert 0.9 <= recent_recency <= 1.0

        # Old signal
        old_item = SignalItem(
            signal_id="sig_002",
            features=features,
            embedding=np.random.randn(32),
            created_at=datetime.utcnow() - timedelta(hours=50),
        )
        old_recency = engine.calculate_recency(old_item, max_age_hours=48)
        assert old_recency == 0.0

    def test_filter_by_risk_tolerance(self) -> None:
        """Test filtering signals by risk tolerance."""
        engine = RankingEngine()
        item_model = ItemModel()

        signals = []
        for risk in ["low", "medium", "high"]:
            features = SignalFeatures(
                symbol=f"{risk.upper()}_STOCK",
                sector="Technology",
                volatility=0.25,
                expected_return=0.05,
                holding_period=5,
                risk_level=risk,  # type: ignore[arg-type]
                sentiment_score=0.7,
                regime="bull",
            )
            signals.append(item_model.create_signal_item(f"sig_{risk}", features))

        # Low tolerance should only get low risk
        low_filtered = engine.filter_by_risk_tolerance(signals, "low")
        assert len(low_filtered) == 1
        assert low_filtered[0].features.risk_level == "low"

        # Medium tolerance should get low and medium
        medium_filtered = engine.filter_by_risk_tolerance(signals, "medium")
        assert len(medium_filtered) == 2

        # High tolerance should get all
        high_filtered = engine.filter_by_risk_tolerance(signals, "high")
        assert len(high_filtered) == 3

    def test_rank_signals(self) -> None:
        """Test ranking signals for a user."""
        engine = RankingEngine()

        user_model = UserModel(embedding_dim=32)
        explicit = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )
        implicit = ImplicitProfile()
        user_profile = user_model.create_user_profile("user_123", explicit, implicit)

        item_model = ItemModel(embedding_dim=32)
        signals = []
        for i in range(5):
            features = SignalFeatures(
                symbol=f"STOCK{i}",
                sector="Technology",
                volatility=0.2 + i * 0.01,
                expected_return=0.03 + i * 0.01,
                holding_period=5 + i,
                risk_level="medium",
                sentiment_score=0.5 + i * 0.05,
                regime="bull",
            )
            signals.append(item_model.create_signal_item(f"sig_{i}", features))

        confidence_scores = {f"sig_{i}": 0.7 + i * 0.05 for i in range(5)}
        ranked = engine.rank_signals(user_profile, signals, confidence_scores)

        assert len(ranked) == 5
        # Scores should be in descending order
        scores = [r.score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_signals_empty_list(self) -> None:
        """Test that empty signals list returns empty ranking."""
        engine = RankingEngine()
        user_model = UserModel()
        explicit = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )
        implicit = ImplicitProfile()
        user_profile = user_model.create_user_profile("user_123", explicit, implicit)

        ranked = engine.rank_signals(user_profile, [])
        assert len(ranked) == 0


class TestFeedConfig:
    """Tests for FeedConfig dataclass."""

    def test_valid_feed_config(self) -> None:
        """Test creation of valid feed config."""
        config = FeedConfig(max_signals_per_day=10, min_confidence=0.5, include_watchlist_boost=True)

        assert config.max_signals_per_day == 10
        assert config.min_confidence == 0.5
        assert config.include_watchlist_boost is True

    def test_invalid_max_signals(self) -> None:
        """Test that non-positive max_signals raises ValueError."""
        with pytest.raises(ValueError, match="max_signals_per_day must be positive"):
            FeedConfig(max_signals_per_day=0)

    def test_invalid_min_confidence(self) -> None:
        """Test that invalid min_confidence raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence must be between"):
            FeedConfig(min_confidence=1.5)


class TestFeedGenerator:
    """Tests for FeedGenerator class."""

    def test_feed_generator_initialization(self) -> None:
        """Test FeedGenerator initialization."""
        config = FeedConfig()
        engine = RankingEngine()
        feed_gen = FeedGenerator(config, engine)

        assert feed_gen.config == config
        assert feed_gen.ranking_engine == engine

    def test_generate_feed(self) -> None:
        """Test generating a personalized feed."""
        config = FeedConfig(max_signals_per_day=5, min_confidence=0.5)
        engine = RankingEngine()
        feed_gen = FeedGenerator(config, engine)

        user_model = UserModel(embedding_dim=32)
        explicit = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )
        implicit = ImplicitProfile()
        user_profile = user_model.create_user_profile("user_123", explicit, implicit)

        item_model = ItemModel(embedding_dim=32)
        signals = []
        for i in range(10):
            features = SignalFeatures(
                symbol=f"STOCK{i}",
                sector="Technology" if i < 7 else "Healthcare",
                volatility=0.2 + i * 0.01,
                expected_return=0.03 + i * 0.01,
                holding_period=5 + i,
                risk_level="medium",
                sentiment_score=0.5 + i * 0.02,
                regime="bull",
            )
            signals.append(item_model.create_signal_item(f"sig_{i}", features))

        confidence_scores = {f"sig_{i}": 0.6 + i * 0.02 for i in range(10)}
        feed = feed_gen.generate_feed(user_profile, signals, confidence_scores)

        assert len(feed) <= 5
        # All items should have positions
        for item in feed:
            assert item.position > 0
            assert item.reason

    def test_generate_feed_empty_signals(self) -> None:
        """Test that empty signals list returns empty feed."""
        config = FeedConfig()
        engine = RankingEngine()
        feed_gen = FeedGenerator(config, engine)

        user_model = UserModel()
        explicit = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )
        implicit = ImplicitProfile()
        user_profile = user_model.create_user_profile("user_123", explicit, implicit)

        feed = feed_gen.generate_feed(user_profile, [])
        assert len(feed) == 0

    def test_ensure_diversity(self) -> None:
        """Test diversity enforcement in feed."""
        config = FeedConfig()
        engine = RankingEngine()
        feed_gen = FeedGenerator(config, engine)

        item_model = ItemModel()

        # Create feed items with same sector
        feed_items = []
        for i in range(5):
            features = SignalFeatures(
                symbol=f"STOCK{i}",
                sector="Technology",
                volatility=0.25,
                expected_return=0.05,
                holding_period=5,
                risk_level="medium",
                sentiment_score=0.7,
                regime="bull",
            )
            signal = item_model.create_signal_item(f"sig_{i}", features)
            ranked = RankedSignal(
                signal_id=f"sig_{i}",
                score=0.8 - i * 0.1,
                relevance=0.8,
                confidence=0.7,
                recency=0.9,
                explanation="Test",
            )
            feed_items.append(FeedItem(signal=signal, rank=ranked, position=i + 1, reason="Test"))

        # This should limit to 3 per sector
        diversified = feed_gen.ensure_diversity(feed_items, max_per_sector=3)
        assert len(diversified) <= 3

    def test_boost_watchlist(self) -> None:
        """Test watchlist boosting."""
        config = FeedConfig()
        engine = RankingEngine()
        feed_gen = FeedGenerator(config, engine)

        ranked_signals = [
            RankedSignal(
                signal_id="sig_AAPL",
                score=0.7,
                relevance=0.8,
                confidence=0.7,
                recency=0.9,
                explanation="Test",
            ),
            RankedSignal(
                signal_id="sig_OTHER",
                score=0.8,
                relevance=0.8,
                confidence=0.7,
                recency=0.9,
                explanation="Test",
            ),
        ]

        watchlist = ["AAPL"]
        boosted = feed_gen.boost_watchlist(ranked_signals, watchlist, boost=1.5)

        # AAPL signal should be boosted and potentially move up in ranking
        assert len(boosted) == 2
        # Check that scores are re-sorted
        assert boosted[0].score >= boosted[1].score
