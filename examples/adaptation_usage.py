"""Example usage of the Output Adaptation Layer."""

from uuid import uuid4

from signalforge.adaptation import create_adaptation_service
from signalforge.models.user import ExperienceLevel, InvestmentHorizon, RiskTolerance, User


def main() -> None:
    """Demonstrate adaptation layer usage."""
    service = create_adaptation_service()

    sample_signal = {
        "symbol": "AAPL",
        "type": "momentum",
        "signal": {
            "type": "buy",
            "direction": "bullish",
            "strength": "strong",
            "confidence": 0.85,
            "price_target": 180.50,
            "stop_loss": 165.25,
            "risk_reward": 3.2,
            "rationale": "Strong momentum with RSI oversold",
        },
        "metrics": {
            "price": 172.45,
            "change_percent": 0.025,
            "rsi": 68.5,
            "macd": 1.25,
            "volume": 45_000_000,
            "volatility": 0.28,
            "sharpe_ratio": 1.85,
        },
        "recommendations": [
            {
                "action": "buy",
                "rationale": "Strong upward momentum",
                "risk_level": "medium",
            }
        ],
    }

    casual_user = User(
        id=uuid4(),
        email="casual@example.com",
        hashed_password="hashed",
        username="casual_investor",
        experience_level=ExperienceLevel.CASUAL,
        risk_tolerance=RiskTolerance.LOW,
        investment_horizon=InvestmentHorizon.LONG,
    )

    quant_user = User(
        id=uuid4(),
        email="quant@example.com",
        hashed_password="hashed",
        username="quant_pro",
        experience_level=ExperienceLevel.QUANT,
        risk_tolerance=RiskTolerance.HIGH,
        investment_horizon=InvestmentHorizon.SHORT,
    )

    print("=" * 80)
    print("CASUAL USER OUTPUT (Simple Guidance)")
    print("=" * 80)
    casual_output = service.adapt_signal(sample_signal, casual_user)
    print(f"Simple Summary: {casual_output.get('simple_summary')}")
    print(f"\nWhat to Do: {casual_output.get('what_to_do')}")
    print(f"\nKey Points: {casual_output.get('key_points')}")

    print("\n" + "=" * 80)
    print("QUANT USER OUTPUT (Raw Data)")
    print("=" * 80)
    quant_output = service.adapt_signal(sample_signal, quant_user)
    print(f"Signal Type: {quant_output.get('signal', {}).get('type')}")
    print(f"Confidence: {quant_output.get('signal', {}).get('confidence')}")
    print(f"Full Metrics: {quant_output.get('metrics')}")

    print("\n" + "=" * 80)
    print("NEWS ADAPTATION EXAMPLE")
    print("=" * 80)
    news_article = {
        "title": "Tech Sector Rallies",
        "summary": "The RSI shows strong momentum in tech sector with volatility increasing.",
        "timestamp": "2024-01-01T10:00:00",
    }

    casual_news = service.adapt_news(news_article, casual_user)
    print(f"Casual User Summary: {casual_news.get('summary')}")


if __name__ == "__main__":
    main()
