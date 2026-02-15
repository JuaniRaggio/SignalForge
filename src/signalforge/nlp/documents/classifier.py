"""Document classifier for financial documents."""

import re
from collections import defaultdict

import structlog

from .schemas import ClassificationResult, DocumentType

logger = structlog.get_logger(__name__)


class DocumentClassifier:
    """Classify financial documents by type."""

    def __init__(self) -> None:
        """Initialize the classifier with patterns."""
        self._patterns = self._load_patterns()
        self._sector_keywords = self._load_sector_keywords()

    def classify(
        self, text: str, metadata: dict[str, str] | None = None
    ) -> ClassificationResult:
        """
        Classify document type.

        Steps:
        1. Check metadata hints (filename, source)
        2. Analyze keywords and patterns
        3. Detect financial symbols (tickers)
        4. Detect sectors

        Args:
            text: Document text content
            metadata: Optional metadata dictionary

        Returns:
            ClassificationResult with type, confidence, symbols, and sectors
        """
        metadata = metadata or {}
        text_lower = text.lower()

        # Score each document type based on pattern matches
        scores: dict[DocumentType, float] = defaultdict(float)

        # Check metadata hints first
        if metadata:
            metadata_type = self._classify_from_metadata(metadata)
            if metadata_type:
                scores[metadata_type] += 0.3

        # Analyze text patterns
        for doc_type, patterns in self._patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern.lower(), text_lower))
                scores[doc_type] += matches * 0.1

        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}

        # Determine primary type
        if not scores:
            primary_type = DocumentType.UNKNOWN
            confidence = 0.0
        else:
            primary_type = max(scores.items(), key=lambda x: x[1])[0]
            confidence = self._calculate_confidence(scores)

        # Get secondary types (excluding primary)
        secondary_types = [
            (doc_type, score)
            for doc_type, score in sorted(scores.items(), key=lambda x: -x[1])
            if doc_type != primary_type and score > 0.1
        ][:3]

        # Extract symbols and sectors
        symbols = self._extract_symbols(text)
        sectors = self._detect_sectors(text, symbols)

        logger.info(
            "classified_document",
            document_type=primary_type.value,
            confidence=confidence,
            symbols_count=len(symbols),
            sectors_count=len(sectors),
        )

        return ClassificationResult(
            document_type=primary_type,
            confidence=confidence,
            secondary_types=secondary_types,
            detected_symbols=symbols,
            detected_sectors=sectors,
        )

    def _load_patterns(self) -> dict[DocumentType, list[str]]:
        """Load classification patterns for each document type."""
        return {
            DocumentType.EARNINGS_REPORT: [
                r"quarterly results",
                r"earnings per share",
                r"\beps\b",
                r"revenue growth",
                r"q[1-4]\s+\d{4}\s+earnings",
                r"fiscal quarter",
                r"diluted earnings",
                r"operating income",
                r"net income",
                r"earnings call",
                r"guidance",
            ],
            DocumentType.SEC_FILING: [
                r"form\s+10-k",
                r"form\s+10-q",
                r"form\s+8-k",
                r"form\s+s-1",
                r"form\s+4",
                r"securities and exchange commission",
                r"sec\s+filing",
                r"edgar",
                r"pursuant to",
                r"proxy statement",
            ],
            DocumentType.ANALYST_REPORT: [
                r"price target",
                r"rating\s*:\s*(buy|sell|hold)",
                r"target price",
                r"initiat(e|ing)\s+coverage",
                r"maintain\s+(buy|sell|hold)",
                r"upgrade",
                r"downgrade",
                r"overweight",
                r"underweight",
                r"outperform",
                r"underperform",
                r"recommendation",
            ],
            DocumentType.MARKET_REPORT: [
                r"market update",
                r"market commentary",
                r"market overview",
                r"market outlook",
                r"market conditions",
                r"trading session",
                r"market close",
                r"market open",
                r"indices",
                r"dow jones",
                r"s&p 500",
                r"nasdaq",
            ],
            DocumentType.SECTOR_ANALYSIS: [
                r"sector\s+analysis",
                r"industry\s+outlook",
                r"sector\s+performance",
                r"industry\s+trends",
                r"competitive\s+landscape",
                r"sector\s+review",
                r"industry\s+report",
                r"market\s+share",
            ],
            DocumentType.NEWS_ARTICLE: [
                r"breaking news",
                r"reuters",
                r"bloomberg",
                r"associated press",
                r"ap news",
                r"cnbc",
                r"wall street journal",
                r"financial times",
                r"reported today",
                r"according to sources",
            ],
            DocumentType.PRESS_RELEASE: [
                r"press release",
                r"for immediate release",
                r"announces",
                r"announcement",
                r"contact:",
                r"media contact",
                r"investor relations",
                r"pr\s+newswire",
                r"business wire",
                r"globe newswire",
            ],
            DocumentType.RESEARCH_NOTE: [
                r"research note",
                r"research report",
                r"research update",
                r"investment thesis",
                r"valuation analysis",
                r"dcf\s+model",
                r"discounted cash flow",
                r"comp(arable)?s?\s+analysis",
                r"peer\s+comparison",
            ],
        }

    def _classify_from_metadata(self, metadata: dict[str, str]) -> DocumentType | None:
        """Classify based on metadata hints."""
        filename = metadata.get("filename", "").lower()
        source = metadata.get("source", "").lower()

        # Check filename patterns
        if "10-k" in filename or "10k" in filename:
            return DocumentType.SEC_FILING
        if "10-q" in filename or "10q" in filename:
            return DocumentType.SEC_FILING
        if "8-k" in filename or "8k" in filename:
            return DocumentType.SEC_FILING
        if "earnings" in filename:
            return DocumentType.EARNINGS_REPORT
        if "press" in filename or "pr_" in filename:
            return DocumentType.PRESS_RELEASE

        # Check source patterns
        if "edgar" in source or "sec" in source:
            return DocumentType.SEC_FILING
        if "analyst" in source:
            return DocumentType.ANALYST_REPORT
        if "news" in source or "rss" in source:
            return DocumentType.NEWS_ARTICLE

        return None

    def _extract_symbols(self, text: str) -> list[str]:
        """Extract stock symbols from text."""
        symbols = set()

        # Pattern 1: $SYMBOL format
        dollar_symbols = re.findall(r"\$([A-Z]{1,5})\b", text)
        symbols.update(dollar_symbols)

        # Pattern 2: EXCHANGE:SYMBOL format
        exchange_symbols = re.findall(
            r"(?:NYSE|NASDAQ|AMEX|TSX|LSE):([A-Z]{1,5})\b", text
        )
        symbols.update(exchange_symbols)

        # Pattern 3: (SYMBOL) format in context
        paren_symbols = re.findall(
            r"\(([A-Z]{1,5})\)(?:\s+(?:stock|shares|ticker))?", text
        )
        symbols.update(paren_symbols)

        # Pattern 4: ticker: SYMBOL format
        ticker_symbols = re.findall(r"ticker\s*:\s*([A-Z]{1,5})\b", text, re.IGNORECASE)
        symbols.update([s.upper() for s in ticker_symbols])

        # Pattern 5: Symbol mentioned with stock/shares context
        context_symbols = re.findall(
            r"\b([A-Z]{2,5})\s+(?:stock|shares|equity|securities)\b", text
        )
        symbols.update(context_symbols)

        # Filter out common false positives
        false_positives = {
            "US",
            "USA",
            "UK",
            "EU",
            "CEO",
            "CFO",
            "CTO",
            "COO",
            "VP",
            "EVP",
            "SVP",
            "SEC",
            "GAAP",
            "IPO",
            "ETF",
            "LLC",
            "INC",
            "LTD",
            "CORP",
            "NYSE",
            "AMEX",
            "TSX",
            "LSE",
            "ESG",
            "ROE",
            "ROI",
            "EBIT",
            "EPS",
            "PE",
            "PEG",
            "YOY",
            "QOQ",
            "MOM",
            "YTD",
            "MTD",
            "GDP",
            "CPI",
            "PPI",
        }

        symbols = {s for s in symbols if s not in false_positives and len(s) <= 5}

        return sorted(symbols)

    def _detect_sectors(self, text: str, symbols: list[str]) -> list[str]:  # noqa: ARG002
        """Detect sectors mentioned or implied."""
        text_lower = text.lower()
        detected_sectors = set()

        for sector, keywords in self._sector_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_sectors.add(sector)
                    break

        return sorted(detected_sectors)

    def _load_sector_keywords(self) -> dict[str, list[str]]:
        """Load sector detection keywords."""
        return {
            "Technology": [
                "software",
                "hardware",
                "semiconductor",
                "cloud computing",
                "artificial intelligence",
                "machine learning",
                "cybersecurity",
                "tech",
                "saas",
                "platform",
            ],
            "Healthcare": [
                "pharmaceutical",
                "biotech",
                "medical device",
                "healthcare",
                "clinical trial",
                "fda approval",
                "drug",
                "hospital",
                "health insurance",
            ],
            "Financial Services": [
                "bank",
                "insurance",
                "investment",
                "asset management",
                "wealth management",
                "fintech",
                "payment processing",
                "credit card",
                "lending",
            ],
            "Energy": [
                "oil",
                "gas",
                "renewable energy",
                "solar",
                "wind power",
                "energy",
                "petroleum",
                "utility",
                "power generation",
            ],
            "Consumer Discretionary": [
                "retail",
                "e-commerce",
                "automotive",
                "luxury goods",
                "restaurant",
                "hotel",
                "entertainment",
                "media",
                "consumer spending",
            ],
            "Consumer Staples": [
                "food",
                "beverage",
                "household products",
                "personal care",
                "tobacco",
                "grocery",
                "consumer goods",
            ],
            "Industrials": [
                "aerospace",
                "defense",
                "manufacturing",
                "construction",
                "machinery",
                "transportation",
                "logistics",
                "industrial",
            ],
            "Materials": [
                "chemicals",
                "metals",
                "mining",
                "paper",
                "packaging",
                "steel",
                "aluminum",
                "commodities",
            ],
            "Real Estate": [
                "real estate",
                "reit",
                "property",
                "commercial real estate",
                "residential real estate",
                "mortgage",
            ],
            "Communication Services": [
                "telecommunications",
                "telecom",
                "wireless",
                "broadband",
                "internet service",
                "media content",
                "streaming",
            ],
            "Utilities": [
                "electric utility",
                "water utility",
                "gas utility",
                "utility company",
                "power distribution",
            ],
        }

    def _calculate_confidence(self, scores: dict[DocumentType, float]) -> float:
        """Calculate classification confidence."""
        if not scores:
            return 0.0

        sorted_scores = sorted(scores.values(), reverse=True)

        if len(sorted_scores) == 1:
            return min(sorted_scores[0], 1.0)

        # Confidence based on gap between top and second choice
        top_score = sorted_scores[0]
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        # Calculate confidence as combination of absolute score and separation
        confidence = (top_score * 0.7) + ((top_score - second_score) * 0.3)

        return min(confidence, 1.0)
