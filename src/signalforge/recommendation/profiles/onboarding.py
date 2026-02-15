"""Onboarding quiz for new users."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from signalforge.recommendation.profiles.schemas import (
    ExplicitProfile,
    InvestmentHorizon,
    OnboardingAnswers,
    RiskTolerance,
    UserType,
)

logger = structlog.get_logger(__name__)


class OnboardingQuiz:
    """Onboarding quiz for new users."""

    QUESTIONS: list[dict[str, Any]] = [
        {
            "id": "time_following_markets",
            "question": "How much time do you dedicate to following financial markets?",
            "options": ["rarely", "daily", "constantly"],
        },
        {
            "id": "recent_trading_activity",
            "question": "How would you describe your recent trading activity?",
            "options": ["none", "occasional", "frequent"],
        },
        {
            "id": "financial_knowledge",
            "question": "How would you rate your financial knowledge?",
            "options": ["beginner", "intermediate", "advanced"],
        },
        {
            "id": "preference_style",
            "question": "Do you prefer guided recommendations or raw data?",
            "options": ["guidance", "raw_data", "both"],
        },
        {
            "id": "api_interest",
            "question": "Are you interested in API access for automation?",
            "options": ["yes", "no"],
        },
    ]

    def __init__(self) -> None:
        """Initialize onboarding quiz."""
        logger.info("onboarding_quiz_initialized")

    def get_questions(self) -> list[dict[str, Any]]:
        """Get quiz questions.

        Returns:
            List of question dictionaries.
        """
        return self.QUESTIONS

    def process_answers(self, answers: OnboardingAnswers) -> UserType:
        """Classify user type from answers.

        Args:
            answers: User's quiz answers.

        Returns:
            Classified user type.
        """
        score = 0

        # Score based on time following markets
        if answers.time_following_markets == "constantly":
            score += 3
        elif answers.time_following_markets == "daily":
            score += 2
        else:  # rarely
            score += 1

        # Score based on trading activity
        if answers.recent_trading_activity == "frequent":
            score += 3
        elif answers.recent_trading_activity == "occasional":
            score += 2
        else:  # none
            score += 1

        # Score based on knowledge
        if answers.financial_knowledge == "advanced":
            score += 3
        elif answers.financial_knowledge == "intermediate":
            score += 2
        else:  # beginner
            score += 1

        # API interest indicates professional/institutional
        if answers.api_interest:
            score += 2

        # Classify based on total score
        if score >= 10:
            user_type = UserType.PROFESSIONAL
        elif score >= 7:
            user_type = UserType.ACTIVE
        else:
            user_type = UserType.CASUAL

        # Special case: API interest + advanced knowledge = professional
        if answers.api_interest and answers.financial_knowledge == "advanced":
            user_type = UserType.PROFESSIONAL

        logger.info(
            "user_type_classified",
            user_type=user_type.value,
            score=score,
        )

        return user_type

    def generate_initial_profile(
        self,
        user_id: str,
        answers: OnboardingAnswers,
    ) -> ExplicitProfile:
        """Generate initial profile from quiz answers.

        Args:
            user_id: User identifier.
            answers: User's quiz answers.

        Returns:
            Initial explicit profile.
        """
        user_type = self.process_answers(answers)

        # Determine risk tolerance
        risk_tolerance = self.suggest_risk_tolerance(answers)

        # Determine investment horizon
        horizon_map = {
            UserType.CASUAL: InvestmentHorizon.LONG_TERM,
            UserType.ACTIVE: InvestmentHorizon.POSITION,
            UserType.PROFESSIONAL: InvestmentHorizon.SWING,
            UserType.INSTITUTIONAL: InvestmentHorizon.POSITION,
        }
        investment_horizon = horizon_map[user_type]

        # Suggest sectors
        preferred_sectors = self.suggest_sectors(user_type)

        profile = ExplicitProfile(
            user_id=user_id,
            portfolio=[],
            watchlist=[],
            preferred_sectors=preferred_sectors,
            risk_tolerance=risk_tolerance,
            investment_horizon=investment_horizon,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        logger.info(
            "initial_profile_generated",
            user_id=user_id,
            user_type=user_type.value,
        )

        return profile

    def suggest_sectors(self, user_type: UserType) -> list[str]:
        """Suggest sectors based on user type.

        Args:
            user_type: Classified user type.

        Returns:
            List of recommended sectors.
        """
        sector_suggestions = {
            UserType.CASUAL: [
                "Technology",
                "Healthcare",
                "Consumer",
            ],
            UserType.ACTIVE: [
                "Technology",
                "Financials",
                "Healthcare",
                "Energy",
            ],
            UserType.PROFESSIONAL: [
                "Technology",
                "Financials",
                "Healthcare",
                "Energy",
                "Industrial",
                "Materials",
            ],
            UserType.INSTITUTIONAL: [
                "Technology",
                "Financials",
                "Healthcare",
                "Energy",
                "Industrial",
                "Materials",
                "Utilities",
                "Consumer",
            ],
        }

        sectors = sector_suggestions.get(user_type, ["Technology", "Healthcare"])

        logger.info(
            "sectors_suggested",
            user_type=user_type.value,
            count=len(sectors),
        )

        return sectors

    def suggest_risk_tolerance(self, answers: OnboardingAnswers) -> RiskTolerance:
        """Infer risk tolerance from answers.

        Args:
            answers: User's quiz answers.

        Returns:
            Suggested risk tolerance level.
        """
        # Conservative defaults
        if answers.financial_knowledge == "beginner":
            return RiskTolerance.CONSERVATIVE

        # Aggressive for experienced active traders
        if (
            answers.financial_knowledge == "advanced"
            and answers.recent_trading_activity == "frequent"
        ):
            return RiskTolerance.AGGRESSIVE

        # Active traders with moderate knowledge
        if answers.recent_trading_activity in ["occasional", "frequent"]:
            return RiskTolerance.MODERATE

        # Default to moderate
        return RiskTolerance.MODERATE
