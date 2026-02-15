"""Tests for sector intelligence modules."""

from datetime import datetime

import pytest

from signalforge.nlp.sectors import (
    ConsumerAnalyzer,
    ConsumerSignal,
    EnergyAnalyzer,
    EnergySignal,
    FinancialAnalyzer,
    FinancialSignal,
    HealthcareAnalyzer,
    HealthcareSignal,
    SectorAnalyzerFactory,
    SignalStrength,
    TechnologyAnalyzer,
    TechnologySignal,
)


# Technology Sector Tests
class TestTechnologyAnalyzer:
    """Tests for technology sector analyzer."""

    @pytest.fixture
    def analyzer(self) -> TechnologyAnalyzer:
        """Create technology analyzer instance."""
        return TechnologyAnalyzer()

    def test_get_sector_keywords(self, analyzer: TechnologyAnalyzer) -> None:
        """Test retrieval of technology keywords."""
        keywords = analyzer.get_sector_keywords()
        assert len(keywords) > 0
        assert "launch" in keywords
        assert "AI" in keywords
        assert "chip" in keywords

    def test_product_launch_detection(self, analyzer: TechnologyAnalyzer) -> None:
        """Test product launch detection."""
        text = "Apple to launch new iPhone with advanced AI features next month"
        phase, confidence = analyzer.detect_product_cycle(text)
        assert phase == "launch"
        assert confidence > 0.5

    def test_product_growth_detection(self, analyzer: TechnologyAnalyzer) -> None:
        """Test product growth phase detection."""
        text = "iPhone sales growing rapidly with increasing demand and market penetration"
        phase, confidence = analyzer.detect_product_cycle(text)
        assert phase == "growth"
        assert confidence > 0.5

    def test_product_decline_detection(self, analyzer: TechnologyAnalyzer) -> None:
        """Test product decline phase detection."""
        text = "Product is declining and being phased out due to obsolete technology"
        phase, confidence = analyzer.detect_product_cycle(text)
        assert phase == "decline"
        assert confidence > 0.5

    def test_ai_relevance_high(self, analyzer: TechnologyAnalyzer) -> None:
        """Test high AI relevance scoring."""
        text = "Company announces breakthrough in AI, machine learning, and GPT models"
        score = analyzer.score_ai_relevance(text)
        assert score >= 0.5

    def test_ai_relevance_low(self, analyzer: TechnologyAnalyzer) -> None:
        """Test low AI relevance scoring."""
        text = "Company reports quarterly earnings with revenue growth"
        score = analyzer.score_ai_relevance(text)
        assert score < 0.3

    def test_chip_shortage_detection(self, analyzer: TechnologyAnalyzer) -> None:
        """Test chip shortage detection."""
        text = "Semiconductor shortage continues with supply constraints affecting GPU production"
        indicator = analyzer.analyze_chip_demand(text)
        assert indicator == "shortage"

    def test_chip_high_demand(self, analyzer: TechnologyAnalyzer) -> None:
        """Test high chip demand detection."""
        text = "Strong demand for processors drives capacity expansion at semiconductor fabs"
        indicator = analyzer.analyze_chip_demand(text)
        assert indicator == "high_demand"

    def test_analyze_technology_signals(self, analyzer: TechnologyAnalyzer) -> None:
        """Test comprehensive signal extraction."""
        text = "NVIDIA launches new AI chip with strong demand from data centers. The GPU shows breakthrough performance in machine learning workloads."
        signals = analyzer.analyze(text, ["NVDA"])

        assert len(signals) > 0
        assert all(isinstance(s, TechnologySignal) for s in signals)
        assert any(s.signal_type == "product_cycle" for s in signals)
        assert any(s.signal_type == "ai_relevance" for s in signals)

    def test_extract_entities(self, analyzer: TechnologyAnalyzer) -> None:
        """Test entity extraction."""
        text = 'NVIDIA Corp announces "H100" GPU under Section 301'
        entities = analyzer.extract_entities(text)

        assert "companies" in entities
        assert "products" in entities
        assert "regulations" in entities


# Healthcare Sector Tests
class TestHealthcareAnalyzer:
    """Tests for healthcare sector analyzer."""

    @pytest.fixture
    def analyzer(self) -> HealthcareAnalyzer:
        """Create healthcare analyzer instance."""
        return HealthcareAnalyzer()

    def test_get_sector_keywords(self, analyzer: HealthcareAnalyzer) -> None:
        """Test retrieval of healthcare keywords."""
        keywords = analyzer.get_sector_keywords()
        assert len(keywords) > 0
        assert any("FDA" in kw for kw in keywords)
        assert any("phase" in kw.lower() for kw in keywords)

    def test_drug_phase_detection_phase3(self, analyzer: HealthcareAnalyzer) -> None:
        """Test phase 3 drug detection."""
        text = "Company initiates Phase 3 pivotal trial for new cancer drug"
        result = analyzer.detect_drug_phase(text)
        assert result is not None
        phase, confidence = result
        assert phase == "phase3"
        assert confidence > 0.5

    def test_drug_phase_detection_preclinical(self, analyzer: HealthcareAnalyzer) -> None:
        """Test preclinical phase detection."""
        text = "Preclinical animal study shows promising results in vitro"
        result = analyzer.detect_drug_phase(text)
        assert result is not None
        phase, confidence = result
        assert phase == "preclinical"

    def test_fda_approval_detection(self, analyzer: HealthcareAnalyzer) -> None:
        """Test FDA approval detection."""
        text = "FDA approved the new drug for market authorization"
        result = analyzer.detect_fda_action(text)
        assert result is not None
        action, strength = result
        assert "approval" in action.lower()
        assert strength == SignalStrength.STRONG_BULLISH

    def test_fda_rejection_detection(self, analyzer: HealthcareAnalyzer) -> None:
        """Test FDA rejection detection."""
        text = "FDA rejected the application with a complete response letter (CRL)"
        result = analyzer.detect_fda_action(text)
        assert result is not None
        action, strength = result
        assert "rejection" in action.lower()
        assert strength == SignalStrength.STRONG_BEARISH

    def test_priority_review_detection(self, analyzer: HealthcareAnalyzer) -> None:
        """Test priority review detection."""
        text = "Drug receives priority review designation from FDA"
        result = analyzer.detect_fda_action(text)
        assert result is not None
        action, strength = result
        assert action == "priority_review"
        assert strength == SignalStrength.BULLISH

    def test_patent_expiration_detection(self, analyzer: HealthcareAnalyzer) -> None:
        """Test patent expiration detection."""
        text = "Company faces patent expiration and loss of exclusivity next year"
        status = analyzer.analyze_patent_status(text)
        assert status == "expiring"

    def test_patent_granted_detection(self, analyzer: HealthcareAnalyzer) -> None:
        """Test patent granted detection."""
        text = "New patent granted for innovative drug formulation"
        status = analyzer.analyze_patent_status(text)
        assert status == "granted"

    def test_analyze_healthcare_signals(self, analyzer: HealthcareAnalyzer) -> None:
        """Test comprehensive healthcare signal extraction."""
        text = "Pfizer receives FDA approval for Phase 3 drug. The pivotal trial showed statistically significant results with strong efficacy."
        signals = analyzer.analyze(text, ["PFE"])

        assert len(signals) > 0
        assert all(isinstance(s, HealthcareSignal) for s in signals)
        assert any(s.signal_type == "fda_action" for s in signals)

    def test_generic_competition_detection(self, analyzer: HealthcareAnalyzer) -> None:
        """Test generic competition detection."""
        text = "Generic competition expected to impact branded drug sales"
        status = analyzer.analyze_patent_status(text)
        assert status == "generic_competition"


# Energy Sector Tests
class TestEnergyAnalyzer:
    """Tests for energy sector analyzer."""

    @pytest.fixture
    def analyzer(self) -> EnergyAnalyzer:
        """Create energy analyzer instance."""
        return EnergyAnalyzer()

    def test_get_sector_keywords(self, analyzer: EnergyAnalyzer) -> None:
        """Test retrieval of energy keywords."""
        keywords = analyzer.get_sector_keywords()
        assert len(keywords) > 0
        assert "oil" in keywords
        assert "solar" in keywords

    def test_commodity_detection_oil(self, analyzer: EnergyAnalyzer) -> None:
        """Test oil commodity detection."""
        text = "Oil prices surge to $100 per barrel on supply concerns"
        result = analyzer.detect_commodity_signals(text)
        assert result is not None
        commodity, sensitivity = result
        assert commodity == "oil"
        assert sensitivity > 0.6

    def test_commodity_price_sensitivity(self, analyzer: EnergyAnalyzer) -> None:
        """Test price sensitivity calculation."""
        text = "Crude oil prices spike dramatically on geopolitical tensions"
        result = analyzer.detect_commodity_signals(text)
        assert result is not None
        commodity, sensitivity = result
        assert sensitivity >= 0.7

    def test_renewable_energy_detection(self, analyzer: EnergyAnalyzer) -> None:
        """Test renewable energy signal detection."""
        text = "Company expands solar and wind energy portfolio with clean energy investments"
        signals = analyzer.analyze(text, ["ENPH"])
        renewable_signals = [s for s in signals if s.signal_type == "renewable_transition"]
        assert len(renewable_signals) > 0

    def test_esg_positive_impact(self, analyzer: EnergyAnalyzer) -> None:
        """Test positive ESG impact detection."""
        text = "Company commits to net zero emissions and carbon neutral operations through sustainable practices"
        esg_score = analyzer.analyze_esg_impact(text)
        assert esg_score is not None
        assert esg_score > 0.0

    def test_esg_negative_impact(self, analyzer: EnergyAnalyzer) -> None:
        """Test negative ESG impact detection."""
        text = "Environmental violation leads to pollution concerns and increased emissions"
        esg_score = analyzer.analyze_esg_impact(text)
        assert esg_score is not None
        assert esg_score < 0.0

    def test_regulatory_signal_detection(self, analyzer: EnergyAnalyzer) -> None:
        """Test regulatory signal detection."""
        text = "New EPA regulation mandates stricter emissions standards"
        has_regulatory = analyzer.detect_regulatory_signal(text)
        assert has_regulatory is True

    def test_analyze_energy_signals(self, analyzer: EnergyAnalyzer) -> None:
        """Test comprehensive energy signal extraction."""
        text = "Oil prices rise while company invests in renewable energy. Strong ESG commitment to reduce carbon emissions."
        signals = analyzer.analyze(text, ["XOM"])

        assert len(signals) > 0
        assert all(isinstance(s, EnergySignal) for s in signals)

    def test_no_esg_content(self, analyzer: EnergyAnalyzer) -> None:
        """Test when no ESG content is present."""
        text = "Company reports quarterly earnings"
        esg_score = analyzer.analyze_esg_impact(text)
        assert esg_score is None


# Financial Sector Tests
class TestFinancialAnalyzer:
    """Tests for financial sector analyzer."""

    @pytest.fixture
    def analyzer(self) -> FinancialAnalyzer:
        """Create financial analyzer instance."""
        return FinancialAnalyzer()

    def test_get_sector_keywords(self, analyzer: FinancialAnalyzer) -> None:
        """Test retrieval of financial keywords."""
        keywords = analyzer.get_sector_keywords()
        assert len(keywords) > 0
        assert any("rate" in kw.lower() for kw in keywords)
        assert "Fed" in keywords

    def test_rate_sensitivity_high(self, analyzer: FinancialAnalyzer) -> None:
        """Test high interest rate sensitivity."""
        text = "Fed announces 75 basis points rate hike at FOMC meeting"
        sensitivity = analyzer.analyze_rate_sensitivity(text)
        assert sensitivity is not None
        assert sensitivity > 0.6

    def test_rate_sensitivity_bps(self, analyzer: FinancialAnalyzer) -> None:
        """Test rate sensitivity with basis points."""
        text = "Interest rate increased by 50 basis points"
        sensitivity = analyzer.analyze_rate_sensitivity(text)
        assert sensitivity is not None
        assert sensitivity >= 0.5

    def test_credit_quality_deteriorating(self, analyzer: FinancialAnalyzer) -> None:
        """Test deteriorating credit quality detection."""
        text = "Rising defaults and increasing NPL provisions signal credit deterioration"
        indicator = analyzer.analyze_credit_quality(text)
        assert indicator == "deteriorating"

    def test_credit_quality_improving(self, analyzer: FinancialAnalyzer) -> None:
        """Test improving credit quality detection."""
        text = "Declining defaults and lower provisions show credit quality improvement"
        indicator = analyzer.analyze_credit_quality(text)
        assert indicator == "improving"

    def test_stress_test_pass(self, analyzer: FinancialAnalyzer) -> None:
        """Test stress test pass detection."""
        text = "Bank passes Federal Reserve stress test with adequate capital"
        impact = analyzer.analyze_regulatory_impact(text)
        assert impact == "stress_test_pass"

    def test_stress_test_fail(self, analyzer: FinancialAnalyzer) -> None:
        """Test stress test fail detection."""
        text = "Bank fails stress test due to inadequate capital buffers"
        impact = analyzer.analyze_regulatory_impact(text)
        assert impact == "stress_test_fail"

    def test_capital_requirement_increase(self, analyzer: FinancialAnalyzer) -> None:
        """Test capital requirement increase detection."""
        text = "Basel III regulations require higher tier 1 capital requirements"
        impact = analyzer.analyze_regulatory_impact(text)
        assert impact == "capital_requirement_increase"

    def test_analyze_financial_signals(self, analyzer: FinancialAnalyzer) -> None:
        """Test comprehensive financial signal extraction."""
        text = "JPMorgan benefits from Fed rate hike of 50 bps. Bank passes stress test and shows strong loan growth."
        signals = analyzer.analyze(text, ["JPM"])

        assert len(signals) > 0
        assert all(isinstance(s, FinancialSignal) for s in signals)

    def test_no_rate_content(self, analyzer: FinancialAnalyzer) -> None:
        """Test when no rate content is present."""
        text = "Company announces new product"
        sensitivity = analyzer.analyze_rate_sensitivity(text)
        assert sensitivity is None


# Consumer Sector Tests
class TestConsumerAnalyzer:
    """Tests for consumer sector analyzer."""

    @pytest.fixture
    def analyzer(self) -> ConsumerAnalyzer:
        """Create consumer analyzer instance."""
        return ConsumerAnalyzer()

    def test_get_sector_keywords(self, analyzer: ConsumerAnalyzer) -> None:
        """Test retrieval of consumer keywords."""
        keywords = analyzer.get_sector_keywords()
        assert len(keywords) > 0
        assert "consumer confidence" in keywords
        assert "brand" in keywords

    def test_consumer_sentiment_positive(self, analyzer: ConsumerAnalyzer) -> None:
        """Test positive consumer sentiment detection."""
        text = "Strong demand and robust sales growth driven by increased consumer spending"
        sentiment = analyzer.analyze_consumer_sentiment(text)
        assert sentiment is not None
        assert sentiment > 0.3

    def test_consumer_sentiment_negative(self, analyzer: ConsumerAnalyzer) -> None:
        """Test negative consumer sentiment detection."""
        text = "Weak demand and soft sales as consumer confidence falling leads to decreased spending"
        sentiment = analyzer.analyze_consumer_sentiment(text)
        assert sentiment is not None
        assert sentiment < -0.3

    def test_holiday_season_detection(self, analyzer: ConsumerAnalyzer) -> None:
        """Test holiday season detection."""
        text = "Retailers prepare for Q4 holiday shopping season and Christmas sales"
        factor = analyzer.detect_seasonal_factor(text)
        assert factor == "holiday_season"

    def test_black_friday_detection(self, analyzer: ConsumerAnalyzer) -> None:
        """Test Black Friday detection."""
        text = "Black Friday and Cyber Monday drive record sales"
        factor = analyzer.detect_seasonal_factor(text)
        assert factor == "black_friday"

    def test_back_to_school_detection(self, analyzer: ConsumerAnalyzer) -> None:
        """Test back-to-school detection."""
        text = "Back-to-school shopping season boosts retail sales"
        factor = analyzer.detect_seasonal_factor(text)
        assert factor == "back_to_school"

    def test_brand_momentum_positive(self, analyzer: ConsumerAnalyzer) -> None:
        """Test positive brand momentum detection."""
        text = "Brand strength and growing market share demonstrate strong brand equity increase"
        momentum = analyzer.analyze_brand_momentum(text)
        assert momentum == "strong_positive"

    def test_brand_momentum_negative(self, analyzer: ConsumerAnalyzer) -> None:
        """Test negative brand momentum detection."""
        text = "Brand erosion and losing market share signal declining loyalty"
        momentum = analyzer.analyze_brand_momentum(text)
        assert momentum == "negative"

    def test_pricing_power_detection(self, analyzer: ConsumerAnalyzer) -> None:
        """Test pricing power detection."""
        text = "Company demonstrates pricing power with price increase and margin expansion"
        has_pricing_power = analyzer.detect_pricing_power(text)
        assert has_pricing_power is True

    def test_analyze_consumer_signals(self, analyzer: ConsumerAnalyzer) -> None:
        """Test comprehensive consumer signal extraction."""
        text = "Nike shows strong brand momentum with holiday season sales. Consumer confidence rising drives robust demand."
        signals = analyzer.analyze(text, ["NKE"])

        assert len(signals) > 0
        assert all(isinstance(s, ConsumerSignal) for s in signals)


# Factory Tests
class TestSectorAnalyzerFactory:
    """Tests for sector analyzer factory."""

    def test_get_analyzer_technology(self) -> None:
        """Test getting technology analyzer."""
        analyzer = SectorAnalyzerFactory.get_analyzer("technology")
        assert isinstance(analyzer, TechnologyAnalyzer)

    def test_get_analyzer_healthcare(self) -> None:
        """Test getting healthcare analyzer."""
        analyzer = SectorAnalyzerFactory.get_analyzer("healthcare")
        assert isinstance(analyzer, HealthcareAnalyzer)

    def test_get_analyzer_energy(self) -> None:
        """Test getting energy analyzer."""
        analyzer = SectorAnalyzerFactory.get_analyzer("energy")
        assert isinstance(analyzer, EnergyAnalyzer)

    def test_get_analyzer_financial(self) -> None:
        """Test getting financial analyzer."""
        analyzer = SectorAnalyzerFactory.get_analyzer("financial")
        assert isinstance(analyzer, FinancialAnalyzer)

    def test_get_analyzer_consumer(self) -> None:
        """Test getting consumer analyzer."""
        analyzer = SectorAnalyzerFactory.get_analyzer("consumer")
        assert isinstance(analyzer, ConsumerAnalyzer)

    def test_get_analyzer_case_insensitive(self) -> None:
        """Test case insensitive sector lookup."""
        analyzer = SectorAnalyzerFactory.get_analyzer("TECHNOLOGY")
        assert isinstance(analyzer, TechnologyAnalyzer)

    def test_get_analyzer_invalid_sector(self) -> None:
        """Test error on invalid sector."""
        with pytest.raises(ValueError, match="Unsupported sector"):
            SectorAnalyzerFactory.get_analyzer("invalid_sector")

    def test_get_all_analyzers(self) -> None:
        """Test getting all analyzers."""
        analyzers = SectorAnalyzerFactory.get_all_analyzers()
        assert len(analyzers) == 5
        assert "technology" in analyzers
        assert "healthcare" in analyzers
        assert "energy" in analyzers
        assert "financial" in analyzers
        assert "consumer" in analyzers

    def test_get_supported_sectors(self) -> None:
        """Test getting supported sectors list."""
        sectors = SectorAnalyzerFactory.get_supported_sectors()
        assert len(sectors) == 5
        assert "technology" in sectors
        assert "healthcare" in sectors
        assert "energy" in sectors
        assert "financial" in sectors
        assert "consumer" in sectors

    def test_register_custom_analyzer(self) -> None:
        """Test registering custom analyzer."""
        class CustomAnalyzer(TechnologyAnalyzer):
            sector_name = "custom"

        SectorAnalyzerFactory.register_analyzer("custom", CustomAnalyzer)
        analyzer = SectorAnalyzerFactory.get_analyzer("custom")
        assert isinstance(analyzer, CustomAnalyzer)

        # Cleanup
        SectorAnalyzerFactory._analyzers.pop("custom", None)


# Signal Schema Tests
class TestSignalSchemas:
    """Tests for signal schemas."""

    def test_signal_strength_values(self) -> None:
        """Test signal strength enum values."""
        assert SignalStrength.STRONG_BULLISH == "strong_bullish"
        assert SignalStrength.BULLISH == "bullish"
        assert SignalStrength.NEUTRAL == "neutral"
        assert SignalStrength.BEARISH == "bearish"
        assert SignalStrength.STRONG_BEARISH == "strong_bearish"

    def test_technology_signal_creation(self) -> None:
        """Test technology signal creation."""
        signal = TechnologySignal(
            sector="technology",
            signal_type="product_cycle",
            strength=SignalStrength.BULLISH,
            confidence=0.8,
            description="Product launch detected",
            affected_symbols=["AAPL"],
            source_text="Apple launches new product",
            product_cycle_phase="launch",
            ai_relevance=0.5,
        )
        assert signal.sector == "technology"
        assert signal.product_cycle_phase == "launch"
        assert signal.ai_relevance == 0.5

    def test_healthcare_signal_creation(self) -> None:
        """Test healthcare signal creation."""
        signal = HealthcareSignal(
            sector="healthcare",
            signal_type="fda_action",
            strength=SignalStrength.STRONG_BULLISH,
            confidence=0.9,
            description="FDA approval",
            affected_symbols=["PFE"],
            source_text="FDA approves drug",
            drug_phase="phase3",
            fda_action="approval",
        )
        assert signal.sector == "healthcare"
        assert signal.fda_action == "approval"

    def test_signal_timestamp_auto_generated(self) -> None:
        """Test signal timestamp is auto-generated."""
        signal = TechnologySignal(
            sector="technology",
            signal_type="test",
            strength=SignalStrength.NEUTRAL,
            confidence=0.5,
            description="Test",
            affected_symbols=[],
            source_text="Test",
        )
        assert isinstance(signal.timestamp, datetime)
