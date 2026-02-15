"""Financial terminology glossary for user education."""

import re
from collections.abc import Callable

import structlog

logger = structlog.get_logger(__name__)


class Glossary:
    """
    Financial terms glossary with definitions for non-professional users.

    Provides term lookup, text scanning, and inline tooltip injection.
    """

    def __init__(self, terms: dict[str, str]) -> None:
        """
        Initialize glossary with financial terms.

        Args:
            terms: Dictionary mapping term to definition
        """
        self._terms = {k.lower(): v for k, v in terms.items()}
        self._logger = logger.bind(component="glossary", term_count=len(self._terms))
        self._logger.info("glossary_initialized")

    def get_definition(self, term: str) -> str | None:
        """
        Get definition for a financial term.

        Args:
            term: Term to look up (case-insensitive)

        Returns:
            Definition string or None if not found
        """
        definition = self._terms.get(term.lower())
        if definition:
            self._logger.debug("term_found", term=term)
        else:
            self._logger.debug("term_not_found", term=term)
        return definition

    def find_terms_in_text(self, text: str) -> list[str]:
        """
        Find all glossary terms present in text.

        Args:
            text: Text to scan for terms

        Returns:
            List of found terms (deduplicated, case-preserved from glossary)
        """
        text_lower = text.lower()
        found_terms = []

        for term in self._terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower):
                found_terms.append(term)

        self._logger.debug(
            "terms_found_in_text",
            text_length=len(text),
            terms_found=len(found_terms),
        )
        return found_terms

    def inject_tooltips(self, text: str) -> str:
        """
        Inject inline definitions into text as tooltips.

        Replaces first occurrence of each term with: "term (definition)"

        Args:
            text: Original text

        Returns:
            Text with inline tooltips added
        """
        result = text
        found_terms = self.find_terms_in_text(text)
        already_replaced: set[str] = set()

        for term in found_terms:
            if term in already_replaced:
                continue

            definition = self._terms[term]
            pattern = r'\b(' + re.escape(term) + r')\b'

            def make_replacer(
                current_term: str, current_def: str
            ) -> Callable[[re.Match[str]], str]:
                def replace_first(match: re.Match[str]) -> str:
                    if current_term not in already_replaced:
                        already_replaced.add(current_term)
                        return f"{match.group(1)} ({current_def})"
                    return match.group(1)

                return replace_first

            result = re.sub(
                pattern, make_replacer(term, definition), result, flags=re.IGNORECASE, count=1
            )

        self._logger.debug(
            "tooltips_injected",
            original_length=len(text),
            modified_length=len(result),
            terms_replaced=len(already_replaced),
        )
        return result

    def get_all_terms(self) -> list[str]:
        """
        Get list of all available glossary terms.

        Returns:
            Sorted list of all terms
        """
        return sorted(self._terms.keys())


def create_default_glossary() -> Glossary:
    """
    Create glossary with default financial terms.

    Returns:
        Glossary instance with 50+ financial terms
    """
    terms = {
        "EPS": "Earnings Per Share, company profit divided by outstanding shares",
        "P/E": "Price-to-Earnings ratio, stock price divided by earnings per share",
        "P/E Ratio": "Price-to-Earnings ratio, stock price divided by earnings per share",
        "EBITDA": "Earnings Before Interest, Taxes, Depreciation, and Amortization",
        "VIX": "Volatility Index, measures market expectation of near-term volatility",
        "RSI": "Relative Strength Index, momentum indicator ranging from 0 to 100",
        "MACD": "Moving Average Convergence Divergence, trend-following momentum indicator",
        "Moving Average": "Average price over a specific time period",
        "SMA": "Simple Moving Average, arithmetic mean of prices over time",
        "EMA": "Exponential Moving Average, weighted average favoring recent prices",
        "Volatility": "Measure of price variation over time",
        "Drawdown": "Peak-to-trough decline during a specific period",
        "Max Drawdown": "Largest peak-to-trough decline over entire period",
        "Sharpe Ratio": "Risk-adjusted return metric, excess return per unit of volatility",
        "Alpha": "Excess return compared to benchmark index",
        "Beta": "Measure of volatility relative to the market",
        "Market Cap": "Total market value of company's outstanding shares",
        "Volume": "Number of shares traded during a time period",
        "Liquidity": "Ease of buying or selling without affecting price",
        "Bid": "Highest price a buyer is willing to pay",
        "Ask": "Lowest price a seller is willing to accept",
        "Spread": "Difference between bid and ask prices",
        "Bull Market": "Market condition with rising prices",
        "Bear Market": "Market condition with falling prices",
        "Support": "Price level where buying pressure prevents further decline",
        "Resistance": "Price level where selling pressure prevents further rise",
        "Dividend": "Portion of earnings distributed to shareholders",
        "Dividend Yield": "Annual dividend divided by stock price",
        "ROE": "Return on Equity, net income divided by shareholder equity",
        "ROA": "Return on Assets, net income divided by total assets",
        "Debt-to-Equity": "Company's total debt divided by shareholder equity",
        "Current Ratio": "Current assets divided by current liabilities",
        "Quick Ratio": "Liquid assets divided by current liabilities",
        "Free Cash Flow": "Cash generated after capital expenditures",
        "Operating Margin": "Operating income divided by revenue",
        "Profit Margin": "Net income divided by revenue",
        "Revenue": "Total income from sales before expenses",
        "Gross Profit": "Revenue minus cost of goods sold",
        "Net Income": "Profit after all expenses and taxes",
        "Earnings": "Company's profit, same as net income",
        "Guidance": "Company's forecast of future performance",
        "Consensus": "Average of analyst estimates",
        "Upgrade": "Analyst raises rating or price target",
        "Downgrade": "Analyst lowers rating or price target",
        "IPO": "Initial Public Offering, first sale of stock to public",
        "Index": "Basket of securities representing a market segment",
        "ETF": "Exchange-Traded Fund, security tracking an index or sector",
        "Mutual Fund": "Investment vehicle pooling money to buy securities",
        "Portfolio": "Collection of investments held by an individual or institution",
        "Diversification": "Spreading investments to reduce risk",
        "Correlation": "Statistical measure of how two securities move together",
        "Standard Deviation": "Measure of dispersion from average value",
        "Variance": "Square of standard deviation, measure of volatility",
        "Bollinger Bands": "Volatility indicator with upper and lower bands",
        "Fibonacci Retracement": "Support and resistance levels based on Fibonacci ratios",
        "Candlestick": "Price chart showing open, high, low, close",
        "Trend": "General direction of market or security price",
        "Breakout": "Price movement beyond established support or resistance",
        "Rally": "Sustained increase in prices",
        "Correction": "Decline of 10% or more from recent high",
        "Recession": "Economic decline lasting at least two quarters",
        "Inflation": "Rate at which general price level increases",
        "Interest Rate": "Cost of borrowing money, usually annual percentage",
        "Federal Reserve": "Central bank of the United States",
        "Yield Curve": "Graph showing yields of bonds with different maturities",
        "Credit Rating": "Assessment of creditworthiness of borrower",
        "Default": "Failure to meet debt obligations",
        "Hedge": "Investment to reduce risk of adverse price movements",
        "Derivative": "Security whose value derives from underlying asset",
        "Option": "Contract giving right to buy or sell at specific price",
        "Call Option": "Right to buy security at specific price",
        "Put Option": "Right to sell security at specific price",
        "Strike Price": "Price at which option can be exercised",
        "Expiration": "Date when option or contract becomes invalid",
    }

    return Glossary(terms)
