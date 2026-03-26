"""
Numeric Hallucination Detection for Financial AI

This module addresses a critical research gap: LLMs confidently generate
financial numbers that may be completely fabricated. This is especially
dangerous in financial applications where incorrect numbers can lead to
costly decisions.

Key Features:
1. Cross-reference LLM outputs against XBRL ground truth
2. Detect numeric deviations beyond acceptable thresholds
3. Flag unverifiable claims
4. Generate verification reports with source citations

Research Contribution:
- First system to ground-truth verify LLM-generated financial numbers
- Novel numeric alignment scoring for financial text
- Provenance-based verification chain
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Status of numeric verification."""
    VERIFIED = "verified"           # Number matches source within tolerance
    DEVIATION = "deviation"         # Number differs from source
    UNVERIFIABLE = "unverifiable"   # No source data found
    PARTIAL = "partial"             # Some numbers verified, some not
    HALLUCINATION = "hallucination" # Number contradicts source significantly


class DeviationType(Enum):
    """Type of numeric deviation detected."""
    EXACT_MATCH = "exact_match"
    WITHIN_TOLERANCE = "within_tolerance"
    MINOR_DEVIATION = "minor_deviation"     # 1-5% off
    MODERATE_DEVIATION = "moderate_deviation"  # 5-15% off
    MAJOR_DEVIATION = "major_deviation"     # 15-50% off
    CRITICAL_DEVIATION = "critical_deviation"  # >50% off
    ORDER_OF_MAGNITUDE = "order_of_magnitude"  # Wrong by 10x or more


@dataclass
class NumericClaim:
    """A numeric claim extracted from LLM output."""
    value: float
    raw_text: str
    metric_type: Optional[str] = None  # e.g., "revenue", "net_income"
    company: Optional[str] = None
    period: Optional[str] = None  # e.g., "2023", "Q2 2023"
    unit: str = "USD"
    scale: str = "units"  # "units", "millions", "billions"
    position: int = 0  # Character position in text
    
    @property
    def normalized_value(self) -> float:
        """Get value normalized to base units (dollars)."""
        multipliers = {
            'units': 1,
            'thousands': 1e3,
            'millions': 1e6,
            'billions': 1e9,
            'trillions': 1e12
        }
        return self.value * multipliers.get(self.scale, 1)


@dataclass
class VerificationResult:
    """Result of verifying a numeric claim."""
    claim: NumericClaim
    status: VerificationStatus
    deviation_type: Optional[DeviationType] = None
    source_value: Optional[float] = None
    source_document: Optional[str] = None
    source_date: Optional[str] = None
    deviation_percent: Optional[float] = None
    confidence: float = 0.0
    explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'claimed_value': self.claim.value,
            'claimed_text': self.claim.raw_text,
            'status': self.status.value,
            'deviation_type': self.deviation_type.value if self.deviation_type else None,
            'source_value': self.source_value,
            'source_document': self.source_document,
            'deviation_percent': self.deviation_percent,
            'confidence': self.confidence,
            'explanation': self.explanation
        }


@dataclass
class HallucinationReport:
    """Comprehensive hallucination detection report."""
    text_analyzed: str
    total_claims: int
    verified_count: int
    deviation_count: int
    hallucination_count: int
    unverifiable_count: int
    
    results: List[VerificationResult] = field(default_factory=list)
    overall_status: VerificationStatus = VerificationStatus.PARTIAL
    trust_score: float = 0.0  # 0-1, higher is more trustworthy
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def format_report(self) -> str:
        """Generate human-readable report."""
        status_emoji = {
            VerificationStatus.VERIFIED: "✅",
            VerificationStatus.DEVIATION: "⚠️",
            VerificationStatus.HALLUCINATION: "❌",
            VerificationStatus.UNVERIFIABLE: "❓",
            VerificationStatus.PARTIAL: "⚡"
        }
        
        lines = [
            "═" * 50,
            "📊 NUMERIC HALLUCINATION DETECTION REPORT",
            "═" * 50,
            "",
            f"Trust Score: {self.trust_score:.0%}",
            f"Overall Status: {status_emoji.get(self.overall_status, '•')} {self.overall_status.value.upper()}",
            "",
            f"Claims Analyzed: {self.total_claims}",
            f"  ✅ Verified: {self.verified_count}",
            f"  ⚠️ Deviations: {self.deviation_count}",
            f"  ❌ Hallucinations: {self.hallucination_count}",
            f"  ❓ Unverifiable: {self.unverifiable_count}",
            "",
            "─" * 50,
            "DETAILED RESULTS:",
            "─" * 50,
        ]
        
        for i, result in enumerate(self.results, 1):
            emoji = status_emoji.get(result.status, "•")
            lines.append(f"\n{i}. {emoji} \"{result.claim.raw_text}\"")
            lines.append(f"   Claimed: {result.claim.value:,.2f}")
            
            if result.source_value is not None:
                lines.append(f"   Source:  {result.source_value:,.2f}")
                if result.deviation_percent is not None:
                    lines.append(f"   Deviation: {result.deviation_percent:+.1f}%")
            
            if result.source_document:
                lines.append(f"   Source: {result.source_document}")
            
            if result.explanation:
                lines.append(f"   Note: {result.explanation}")
        
        lines.extend([
            "",
            "═" * 50,
            f"Report generated: {self.timestamp}"
        ])
        
        return "\n".join(lines)


class NumericHallucinationDetector:
    """
    Detects hallucinated financial numbers by cross-referencing against XBRL data.
    
    This is a novel research contribution: no existing financial RAG system
    automatically verifies that LLM-generated numbers match source documents.
    """
    
    # Patterns for extracting numeric claims
    NUMERIC_PATTERNS = [
        # $X.XX billion/million
        r'\$\s*([\d,]+\.?\d*)\s*(billion|million|trillion|B|M|T)',
        # X.XX billion/million dollars
        r'([\d,]+\.?\d*)\s*(billion|million|trillion)\s+dollars?',
        # $X,XXX,XXX
        r'\$\s*([\d,]+\.?\d*)',
        # XX.X% (percentages)
        r'([\d,]+\.?\d*)\s*%',
        # Plain numbers in context
        r'(?:revenue|income|profit|earnings|EPS|cash\s*flow|assets|liabilities|equity)\s+(?:of|was|were|is|are)?\s*\$?\s*([\d,]+\.?\d*)',
    ]
    
    # Metric type detection patterns
    METRIC_PATTERNS = {
        'revenue': r'revenue|sales|top\s*line',
        'net_income': r'net\s*income|net\s*profit|earnings|bottom\s*line',
        'gross_profit': r'gross\s*profit|gross\s*margin',
        'operating_income': r'operating\s*income|operating\s*profit|EBIT',
        'eps': r'EPS|earnings\s*per\s*share',
        'assets': r'total\s*assets|assets',
        'liabilities': r'total\s*liabilities|liabilities|debt',
        'equity': r'shareholders?\s*equity|stockholders?\s*equity|book\s*value',
        'cash_flow': r'cash\s*flow|operating\s*cash|free\s*cash',
        'market_cap': r'market\s*cap|market\s*value|valuation',
    }
    
    # Acceptable deviation thresholds
    TOLERANCE_THRESHOLDS = {
        'exact': 0.001,      # 0.1%
        'minor': 0.05,       # 5%
        'moderate': 0.15,    # 15%
        'major': 0.50,       # 50%
    }
    
    def __init__(self, xbrl_dir: str = "data/raw/xbrl_structured"):
        """
        Initialize the hallucination detector.
        
        Args:
            xbrl_dir: Directory containing XBRL ground truth data
        """
        self.xbrl_dir = Path(xbrl_dir)
        self.ground_truth_cache: Dict[str, Dict] = {}
        self._load_ground_truth()
        logger.info("Initialized Numeric Hallucination Detector")
    
    def _load_ground_truth(self):
        """Load XBRL data as ground truth for verification."""
        if not self.xbrl_dir.exists():
            logger.warning(f"XBRL directory not found: {self.xbrl_dir}")
            return
        
        for json_file in self.xbrl_dir.glob("*_facts.json"):
            try:
                # Extract ticker by removing "_facts" suffix
                ticker = json_file.stem.replace('_facts', '').upper()
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.ground_truth_cache[ticker] = data
                logger.debug(f"Loaded ground truth for {ticker}")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Loaded ground truth for {len(self.ground_truth_cache)} companies")
    
    def detect_hallucinations(
        self,
        text: str,
        ticker: str = None,
        period: str = None
    ) -> HallucinationReport:
        """
        Analyze text for numeric hallucinations.
        
        Args:
            text: Text containing numeric claims (e.g., LLM output)
            ticker: Company ticker to verify against
            period: Time period for verification (e.g., "2023")
            
        Returns:
            HallucinationReport with verification results
        """
        # Extract numeric claims from text
        claims = self._extract_numeric_claims(text, ticker, period)
        
        # Verify each claim
        results = []
        for claim in claims:
            result = self._verify_claim(claim)
            results.append(result)
        
        # Aggregate results
        verified_count = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
        deviation_count = sum(1 for r in results if r.status == VerificationStatus.DEVIATION)
        hallucination_count = sum(1 for r in results if r.status == VerificationStatus.HALLUCINATION)
        unverifiable_count = sum(1 for r in results if r.status == VerificationStatus.UNVERIFIABLE)
        
        # Calculate trust score
        if len(results) > 0:
            trust_score = (verified_count + 0.5 * deviation_count) / len(results)
        else:
            trust_score = 1.0  # No claims to verify
        
        # Determine overall status
        if hallucination_count > 0:
            overall_status = VerificationStatus.HALLUCINATION
        elif deviation_count > 0:
            overall_status = VerificationStatus.DEVIATION
        elif unverifiable_count == len(results):
            overall_status = VerificationStatus.UNVERIFIABLE
        elif verified_count == len(results):
            overall_status = VerificationStatus.VERIFIED
        else:
            overall_status = VerificationStatus.PARTIAL
        
        return HallucinationReport(
            text_analyzed=text[:500] + "..." if len(text) > 500 else text,
            total_claims=len(claims),
            verified_count=verified_count,
            deviation_count=deviation_count,
            hallucination_count=hallucination_count,
            unverifiable_count=unverifiable_count,
            results=results,
            overall_status=overall_status,
            trust_score=trust_score
        )
    
    def _extract_numeric_claims(
        self,
        text: str,
        ticker: str = None,
        period: str = None
    ) -> List[NumericClaim]:
        """Extract numeric claims from text."""
        claims = []
        seen_positions = set()
        
        # Extract ticker from text if not provided
        if not ticker:
            ticker_match = re.search(r'\b(AAPL|MSFT|GOOGL|AMZN|META|JPM|GS|XOM|CVX)\b', text.upper())
            if ticker_match:
                ticker = ticker_match.group(1)
        
        # Extract year/period from text if not provided
        if not period:
            year_match = re.search(r'\b(20\d{2}|FY\d{2,4}|Q[1-4]\s*20\d{2})\b', text)
            if year_match:
                period = year_match.group(1)
        
        for pattern in self.NUMERIC_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pos = match.start()
                
                # Skip if we've already captured a claim at this position
                if any(abs(pos - p) < 10 for p in seen_positions):
                    continue
                
                seen_positions.add(pos)
                
                # Extract the numeric value
                groups = match.groups()
                value_str = groups[0] if groups else match.group()
                value_str = value_str.replace(',', '')
                
                try:
                    value = float(value_str)
                except ValueError:
                    continue
                
                # Determine scale from match and surrounding context
                scale = 'units'
                full_match = match.group().lower()
                
                # Look at text after the match for scale indicators (e.g., "500 billion")
                end_pos = pos + len(match.group())
                following_text = text[end_pos:end_pos + 20].lower().strip()
                
                # Check both the match and following text for scale indicators
                combined_text = full_match + ' ' + following_text
                
                if 'trillion' in combined_text or ' t ' in combined_text or combined_text.endswith('t'):
                    scale = 'trillions'
                elif 'billion' in combined_text or ' b ' in combined_text or combined_text.endswith('b'):
                    scale = 'billions'
                elif 'million' in combined_text or ' m ' in combined_text or combined_text.endswith('m'):
                    scale = 'millions'
                
                # Also check for scale in capture groups (e.g., pattern like "([\d,]+)\s*(billion)")
                if len(groups) > 1 and groups[1]:
                    scale_word = groups[1].lower()
                    if 'trillion' in scale_word or scale_word == 't':
                        scale = 'trillions'
                    elif 'billion' in scale_word or scale_word == 'b':
                        scale = 'billions'
                    elif 'million' in scale_word or scale_word == 'm':
                        scale = 'millions'
                
                # Detect metric type from context (look at surrounding text)
                context_start = max(0, pos - 50)
                context_end = min(len(text), pos + 50)
                context = text[context_start:context_end].lower()
                
                metric_type = None
                for mtype, pattern in self.METRIC_PATTERNS.items():
                    if re.search(pattern, context, re.IGNORECASE):
                        metric_type = mtype
                        break
                
                claims.append(NumericClaim(
                    value=value,
                    raw_text=match.group().strip(),
                    metric_type=metric_type,
                    company=ticker,
                    period=period,
                    scale=scale,
                    position=pos
                ))
        
        return claims
    
    def _verify_claim(self, claim: NumericClaim) -> VerificationResult:
        """Verify a single numeric claim against ground truth."""
        
        # Check if we have ground truth for this company
        if not claim.company or claim.company not in self.ground_truth_cache:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                explanation=f"No ground truth data available for {claim.company or 'unknown company'}"
            )
        
        ground_truth = self.ground_truth_cache[claim.company]
        
        # Try to find matching metric in ground truth
        source_value = self._find_source_value(claim, ground_truth)
        
        if source_value is None:
            return VerificationResult(
                claim=claim,
                status=VerificationStatus.UNVERIFIABLE,
                explanation=f"Could not find {claim.metric_type or 'metric'} in source data"
            )
        
        # Calculate deviation
        claimed_normalized = claim.normalized_value
        deviation = abs(claimed_normalized - source_value) / abs(source_value) if source_value != 0 else 1.0
        deviation_percent = (claimed_normalized - source_value) / abs(source_value) * 100 if source_value != 0 else 100
        
        # Determine deviation type
        if deviation <= self.TOLERANCE_THRESHOLDS['exact']:
            deviation_type = DeviationType.EXACT_MATCH
            status = VerificationStatus.VERIFIED
        elif deviation <= self.TOLERANCE_THRESHOLDS['minor']:
            deviation_type = DeviationType.WITHIN_TOLERANCE
            status = VerificationStatus.VERIFIED
        elif deviation <= self.TOLERANCE_THRESHOLDS['moderate']:
            deviation_type = DeviationType.MINOR_DEVIATION
            status = VerificationStatus.DEVIATION
        elif deviation <= self.TOLERANCE_THRESHOLDS['major']:
            deviation_type = DeviationType.MODERATE_DEVIATION
            status = VerificationStatus.DEVIATION
        elif deviation <= 1.0:
            deviation_type = DeviationType.MAJOR_DEVIATION
            status = VerificationStatus.HALLUCINATION
        else:
            deviation_type = DeviationType.ORDER_OF_MAGNITUDE
            status = VerificationStatus.HALLUCINATION
        
        # Generate explanation
        if status == VerificationStatus.VERIFIED:
            explanation = f"Value matches source within {deviation*100:.1f}% tolerance"
        elif status == VerificationStatus.DEVIATION:
            explanation = f"Value deviates {deviation_percent:+.1f}% from source"
        else:
            explanation = f"HALLUCINATION DETECTED: Claimed {claimed_normalized:,.0f} but source shows {source_value:,.0f}"
        
        return VerificationResult(
            claim=claim,
            status=status,
            deviation_type=deviation_type,
            source_value=source_value,
            source_document=f"{claim.company} XBRL Data",
            source_date=claim.period,
            deviation_percent=deviation_percent,
            confidence=1 - min(deviation, 1.0),
            explanation=explanation
        )
    
    def _find_source_value(
        self,
        claim: NumericClaim,
        ground_truth: Dict
    ) -> Optional[float]:
        """Find the source value for a claim in ground truth data."""
        
        # Map claim metric types to XBRL field names
        metric_mapping = {
            'revenue': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet'],
            'net_income': ['NetIncomeLoss', 'ProfitLoss'],
            'gross_profit': ['GrossProfit'],
            'operating_income': ['OperatingIncomeLoss'],
            'eps': ['EarningsPerShareDiluted', 'EarningsPerShareBasic'],
            'assets': ['Assets'],
            'liabilities': ['Liabilities'],
            'equity': ['StockholdersEquity', 'CommonStockholdersEquity'],
            'cash_flow': ['NetCashProvidedByUsedInOperatingActivities'],
        }
        
        # Get potential field names
        if claim.metric_type:
            field_names = metric_mapping.get(claim.metric_type, [])
        else:
            # Try all fields
            field_names = [fn for fns in metric_mapping.values() for fn in fns]
        
        # Navigate XBRL structure: facts -> us-gaap -> metric -> units -> USD -> [values]
        us_gaap = ground_truth.get('facts', {}).get('us-gaap', {})
        
        # Search for the value
        for field_name in field_names:
            if field_name in us_gaap:
                metric_data = us_gaap[field_name]
                
                # Get USD values
                units = metric_data.get('units', {})
                usd_values = units.get('USD', [])
                
                if not usd_values:
                    continue
                
                # Filter for annual (10-K) values
                annual_values = [v for v in usd_values if v.get('form') == '10-K' and 'fy' in v]
                
                if not annual_values:
                    annual_values = usd_values
                
                # Try to find matching period
                if claim.period:
                    year_match = re.search(r'20\d{2}', claim.period)
                    if year_match:
                        target_year = int(year_match.group())
                        for v in annual_values:
                            if v.get('fy') == target_year:
                                return float(v.get('val', 0))
                
                # Return most recent annual value
                if annual_values:
                    # Sort by fiscal year
                    sorted_vals = sorted(annual_values, key=lambda x: x.get('fy', 0), reverse=True)
                    return float(sorted_vals[0].get('val', 0))
        
        return None
    
    def verify_llm_output(
        self,
        llm_output: str,
        ticker: str,
        query_context: str = None
    ) -> Tuple[str, HallucinationReport]:
        """
        Verify LLM output and annotate with verification markers.
        
        Args:
            llm_output: Raw LLM output text
            ticker: Company ticker for verification
            query_context: Original query for context
            
        Returns:
            Tuple of (annotated_output, report)
        """
        report = self.detect_hallucinations(llm_output, ticker)
        
        # Annotate the output
        annotated = llm_output
        offset = 0
        
        for result in sorted(report.results, key=lambda r: r.claim.position):
            pos = result.claim.position + offset
            raw_text = result.claim.raw_text
            
            if result.status == VerificationStatus.VERIFIED:
                marker = " ✓"
            elif result.status == VerificationStatus.DEVIATION:
                marker = f" ⚠️[{result.deviation_percent:+.0f}%]"
            elif result.status == VerificationStatus.HALLUCINATION:
                marker = f" ❌[Source: {result.source_value:,.0f}]" if result.source_value else " ❌[HALLUCINATED]"
            else:
                marker = " ❓"
            
            # Insert marker after the number
            end_pos = pos + len(raw_text)
            annotated = annotated[:end_pos] + marker + annotated[end_pos:]
            offset += len(marker)
        
        return annotated, report


class NumericGroundingValidator:
    """
    Validates that generated text is properly grounded in source numbers.
    
    This extends hallucination detection to provide corrective suggestions.
    """
    
    def __init__(self, detector: NumericHallucinationDetector):
        self.detector = detector
    
    def validate_and_correct(
        self,
        text: str,
        ticker: str
    ) -> Tuple[str, List[str]]:
        """
        Validate text and provide corrected version if hallucinations found.
        
        Args:
            text: Text to validate
            ticker: Company ticker
            
        Returns:
            Tuple of (corrected_text, list_of_corrections_made)
        """
        report = self.detector.detect_hallucinations(text, ticker)
        
        if report.overall_status == VerificationStatus.VERIFIED:
            return text, []
        
        corrections = []
        corrected = text
        offset = 0
        
        for result in sorted(report.results, key=lambda r: r.claim.position):
            if result.status in [VerificationStatus.HALLUCINATION, VerificationStatus.DEVIATION]:
                if result.source_value is not None:
                    pos = result.claim.position + offset
                    old_text = result.claim.raw_text
                    
                    # Format the corrected value
                    source_val = result.source_value
                    if source_val >= 1e12:
                        new_text = f"${source_val/1e12:.2f} trillion"
                    elif source_val >= 1e9:
                        new_text = f"${source_val/1e9:.2f} billion"
                    elif source_val >= 1e6:
                        new_text = f"${source_val/1e6:.2f} million"
                    else:
                        new_text = f"${source_val:,.2f}"
                    
                    # Replace in text
                    end_pos = pos + len(old_text)
                    corrected = corrected[:pos] + new_text + corrected[end_pos:]
                    offset += len(new_text) - len(old_text)
                    
                    corrections.append(
                        f"Corrected '{old_text}' to '{new_text}' "
                        f"(source: {result.source_document})"
                    )
        
        return corrected, corrections
