"""
Look-Ahead Bias Detector for TemporalGuard-RAG

Dedicated module for detecting and preventing look-ahead bias in
financial RAG system outputs and document retrieval.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import re
import logging
import json
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiasType(Enum):
    """Types of look-ahead bias."""
    FUTURE_DATE_REFERENCE = "future_date_reference"
    UNAVAILABLE_FILING = "unavailable_filing"
    FORWARD_LOOKING_LANGUAGE = "forward_looking_language"
    ANACHRONISTIC_EVENT = "anachronistic_event"
    FILING_LAG_VIOLATION = "filing_lag_violation"


@dataclass
class BiasDetection:
    """Single bias detection result."""
    bias_type: BiasType
    severity: str  # 'high', 'medium', 'low'
    description: str
    evidence: str
    location: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class BiasReport:
    """Complete bias analysis report."""
    has_bias: bool
    bias_count: int
    detections: List[BiasDetection] = field(default_factory=list)
    analysis_date: str = ""
    query: str = ""
    overall_risk: str = "low"  # 'high', 'medium', 'low'
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_detection(self, detection: BiasDetection):
        """Add a bias detection."""
        self.detections.append(detection)
        self.bias_count = len(self.detections)
        self.has_bias = self.bias_count > 0
        self._update_risk_level()
        
    def _update_risk_level(self):
        """Update overall risk level."""
        if not self.detections:
            self.overall_risk = "low"
        elif any(d.severity == 'high' for d in self.detections):
            self.overall_risk = "high"
        elif any(d.severity == 'medium' for d in self.detections):
            self.overall_risk = "medium"
        else:
            self.overall_risk = "low"
            
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'has_bias': self.has_bias,
            'bias_count': self.bias_count,
            'overall_risk': self.overall_risk,
            'analysis_date': self.analysis_date,
            'query': self.query,
            'timestamp': self.timestamp,
            'detections': [
                {
                    'type': d.bias_type.value,
                    'severity': d.severity,
                    'description': d.description,
                    'evidence': d.evidence,
                    'location': d.location,
                    'recommendation': d.recommendation
                }
                for d in self.detections
            ]
        }


class LookAheadBiasDetector:
    """
    Comprehensive look-ahead bias detection for financial RAG systems.
    
    Detection Categories:
    1. Future Date References - References to dates after analysis point
    2. Filing Lag Violations - Using documents not yet available
    3. Forward-Looking Language - Predictions presented as facts
    4. Anachronistic Events - Referencing events that haven't occurred
    """
    
    # Filing lag constants (in days)
    FILING_LAGS = {
        "10-K": 60,
        "10-Q": 40,
        "8-K": 4,
        "earnings_call": 2,
        "press_release": 1
    }
    
    # Quarter end dates
    QUARTER_ENDS = {
        1: (3, 31),
        2: (6, 30),  
        3: (9, 30),
        4: (12, 31)
    }
    
    # Forward-looking language patterns
    FORWARD_LOOKING_PATTERNS = [
        r'\bwill\s+(?:be|have|increase|decrease|grow|decline)\b',
        r'\bis\s+expected\s+to\b',
        r'\bare\s+expected\s+to\b',
        r'\bforecast(?:s|ed|ing)?\b',
        r'\bproject(?:s|ed|ion|ions)?\b',
        r'\banticipate(?:s|d)?\b',
        r'\bguidance\s+(?:for|of)\b',
        r'\boutlook\s+(?:for|on)\b',
        r'\bestimate(?:s|d)?\b.*\d{4}',
        r'\bprediction(?:s)?\b',
        r'\bby\s+(?:the\s+)?end\s+of\s+\d{4}\b',
        r'\bin\s+(?:the\s+)?coming\s+(?:year|quarter|months)\b',
    ]
    
    # Major market events for anachronism detection
    MAJOR_EVENTS = {
        "2020": ["COVID-19 pandemic impact", "market crash March 2020"],
        "2021": ["inflation concerns", "supply chain disruptions"],
        "2022": ["Fed rate hikes", "tech selloff", "crypto winter"],
        "2023": ["banking crisis", "AI boom", "recession fears"],
        "2024": ["election year", "AI expansion"],
    }
    
    def __init__(self):
        """Initialize bias detector."""
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.FORWARD_LOOKING_PATTERNS
        ]
        
    def detect_bias(self, 
                    text: str, 
                    analysis_date: str,
                    query: Optional[str] = None) -> BiasReport:
        """
        Comprehensive bias detection on text.
        
        Args:
            text: Text to analyze (query, answer, or combined)
            analysis_date: Point-in-time date (YYYYMMDD)
            query: Original query (for context)
            
        Returns:
            BiasReport with all detections
        """
        report = BiasReport(
            has_bias=False,
            bias_count=0,
            analysis_date=analysis_date,
            query=query or ""
        )
        
        try:
            analysis_dt = datetime.strptime(analysis_date, '%Y%m%d')
        except ValueError:
            logger.warning(f"Could not parse analysis date: {analysis_date}")
            return report
            
        # Run all detection methods
        self._detect_future_dates(text, analysis_dt, report)
        self._detect_forward_looking_language(text, analysis_dt, report)
        self._detect_filing_lag_violations(text, analysis_dt, report)
        self._detect_anachronistic_events(text, analysis_dt, report)
        
        return report
        
    def _detect_future_dates(self, text: str, 
                              analysis_date: datetime, 
                              report: BiasReport):
        """Detect references to future dates."""
        analysis_year = analysis_date.year
        analysis_month = analysis_date.month
        
        # Detect year references
        year_pattern = r'\b(20[2-9]\d)\b'
        years_found = re.findall(year_pattern, text)
        
        for year_str in years_found:
            year = int(year_str)
            if year > analysis_year:
                report.add_detection(BiasDetection(
                    bias_type=BiasType.FUTURE_DATE_REFERENCE,
                    severity='high',
                    description=f"Reference to future year {year} (analysis year: {analysis_year})",
                    evidence=f"Found year reference: {year_str}",
                    recommendation=f"Remove or replace reference to {year}"
                ))
                
        # Detect quarter references
        quarter_pattern = r'\bQ([1-4])\s*(20[2-9]\d)\b'
        quarters_found = re.findall(quarter_pattern, text, re.IGNORECASE)
        
        for quarter, year_str in quarters_found:
            quarter = int(quarter)
            year = int(year_str)
            
            # Calculate quarter end date
            month, day = self.QUARTER_ENDS[quarter]
            quarter_end = datetime(year, month, day)
            
            # Add filing lag to determine availability
            filing_available = quarter_end + timedelta(days=self.FILING_LAGS["10-Q"])
            
            if filing_available > analysis_date:
                report.add_detection(BiasDetection(
                    bias_type=BiasType.FILING_LAG_VIOLATION,
                    severity='high',
                    description=f"Q{quarter} {year} data would not be available at analysis date",
                    evidence=f"Quarter ends {quarter_end.strftime('%Y-%m-%d')}, filing available ~{filing_available.strftime('%Y-%m-%d')}",
                    recommendation=f"Use earlier quarter data or adjust analysis date"
                ))
                
        # Detect month-year references
        month_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s*(20[2-9]\d)\b'
        months_found = re.findall(month_pattern, text, re.IGNORECASE)
        
        month_mapping = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for month_str, year_str in months_found:
            month = month_mapping[month_str.lower()]
            year = int(year_str)
            
            reference_date = datetime(year, month, 1)
            
            if reference_date > analysis_date:
                report.add_detection(BiasDetection(
                    bias_type=BiasType.FUTURE_DATE_REFERENCE,
                    severity='medium',
                    description=f"Reference to future month: {month_str} {year}",
                    evidence=f"Analysis date is {analysis_date.strftime('%B %Y')}",
                    recommendation="Use only historical date references"
                ))
                
    def _detect_forward_looking_language(self, text: str,
                                          analysis_date: datetime,
                                          report: BiasReport):
        """Detect forward-looking language patterns."""
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            
            for match in matches:
                # Get surrounding context
                full_match = pattern.search(text)
                if full_match:
                    start = max(0, full_match.start() - 20)
                    end = min(len(text), full_match.end() + 20)
                    context = text[start:end]
                    
                    report.add_detection(BiasDetection(
                        bias_type=BiasType.FORWARD_LOOKING_LANGUAGE,
                        severity='medium',
                        description="Forward-looking language detected",
                        evidence=f"...{context}...",
                        location=f"Position {full_match.start()}",
                        recommendation="Replace with historical facts or clearly label as projection"
                    ))
                    break  # One detection per pattern
                    
    def _detect_filing_lag_violations(self, text: str,
                                       analysis_date: datetime,
                                       report: BiasReport):
        """Detect references to documents that wouldn't be available."""
        # Detect 10-K references
        tenk_pattern = r'\b10-K\b.*?\b(20[2-9]\d)\b'
        tenk_matches = re.findall(tenk_pattern, text, re.IGNORECASE)
        
        for year_str in tenk_matches:
            year = int(year_str)
            # 10-K for fiscal year X typically filed February-March of year X+1
            typical_filing = datetime(year + 1, 3, 1)
            
            if typical_filing > analysis_date:
                report.add_detection(BiasDetection(
                    bias_type=BiasType.UNAVAILABLE_FILING,
                    severity='high',
                    description=f"10-K for fiscal {year} likely not filed by analysis date",
                    evidence=f"10-K typically filed March {year+1}, analysis date is {analysis_date.strftime('%Y-%m-%d')}",
                    recommendation=f"Use 10-K for fiscal {year-1} instead"
                ))
                
        # Detect 10-Q references
        tenq_pattern = r'\b10-Q\b.*?\bQ([1-4])\s*(20[2-9]\d)\b'
        tenq_matches = re.findall(tenq_pattern, text, re.IGNORECASE)
        
        for quarter, year_str in tenq_matches:
            quarter = int(quarter)
            year = int(year_str)
            
            month, day = self.QUARTER_ENDS[quarter]
            quarter_end = datetime(year, month, day)
            filing_available = quarter_end + timedelta(days=self.FILING_LAGS["10-Q"])
            
            if filing_available > analysis_date:
                report.add_detection(BiasDetection(
                    bias_type=BiasType.UNAVAILABLE_FILING,
                    severity='high',
                    description=f"10-Q for Q{quarter} {year} not filed by analysis date",
                    evidence=f"Q{quarter} ends {quarter_end.strftime('%Y-%m-%d')}, 10-Q available ~{filing_available.strftime('%Y-%m-%d')}",
                    recommendation="Use earlier quarter filing"
                ))
                
    def _detect_anachronistic_events(self, text: str,
                                      analysis_date: datetime,
                                      report: BiasReport):
        """Detect references to events that hadn't occurred yet."""
        analysis_year = str(analysis_date.year)
        
        # Check for references to events in future years
        for year, events in self.MAJOR_EVENTS.items():
            if int(year) > analysis_date.year:
                for event in events:
                    # Check for event keywords
                    event_words = event.lower().split()
                    text_lower = text.lower()
                    
                    # Simple keyword matching
                    if any(word in text_lower for word in event_words if len(word) > 4):
                        report.add_detection(BiasDetection(
                            bias_type=BiasType.ANACHRONISTIC_EVENT,
                            severity='medium',
                            description=f"Possible reference to {year} event: {event}",
                            evidence=f"Keywords from '{event}' found in text",
                            recommendation="Verify this event occurred before analysis date"
                        ))
                        
    def check_document_availability(self,
                                    filing_type: str,
                                    filing_period_end: str,
                                    analysis_date: str) -> Tuple[bool, str]:
        """
        Check if a specific document would be available.
        
        Args:
            filing_type: Type of filing ('10-K', '10-Q', '8-K')
            filing_period_end: End date of the filing period
            analysis_date: Analysis point-in-time date
            
        Returns:
            Tuple of (is_available, explanation)
        """
        try:
            period_end = datetime.strptime(filing_period_end, '%Y%m%d')
            analysis_dt = datetime.strptime(analysis_date, '%Y%m%d')
        except ValueError as e:
            return False, f"Date parsing error: {e}"
            
        lag = self.FILING_LAGS.get(filing_type, 30)
        available_date = period_end + timedelta(days=lag)
        
        is_available = available_date <= analysis_dt
        
        if is_available:
            days_available = (analysis_dt - available_date).days
            explanation = f"✅ Available - Filed ~{available_date.strftime('%Y-%m-%d')}, {days_available} days before analysis"
        else:
            days_until = (available_date - analysis_dt).days
            explanation = f"❌ Not Available - Won't be filed until ~{available_date.strftime('%Y-%m-%d')} ({days_until} days after analysis)"
            
        return is_available, explanation
        
    def validate_query(self, query: str, analysis_date: str) -> BiasReport:
        """
        Validate a query for temporal consistency before processing.
        
        Args:
            query: User query
            analysis_date: Point-in-time date
            
        Returns:
            BiasReport
        """
        report = self.detect_bias(query, analysis_date, query)
        
        if report.has_bias:
            logger.warning(f"Query has {report.bias_count} potential bias issues")
            
        return report
        
    def validate_answer(self, 
                        answer: str,
                        query: str,
                        analysis_date: str,
                        source_documents: Optional[List[Dict]] = None) -> BiasReport:
        """
        Validate an answer for look-ahead bias.
        
        Args:
            answer: Generated answer
            query: Original query
            analysis_date: Point-in-time date
            source_documents: Optional list of source document metadata
            
        Returns:
            BiasReport
        """
        # Check answer text
        report = self.detect_bias(answer, analysis_date, query)
        
        # Check source documents if provided
        if source_documents:
            try:
                analysis_dt = datetime.strptime(analysis_date, '%Y%m%d')
            except ValueError:
                return report
                
            for doc in source_documents:
                filing_date = doc.get('filing_date', '')
                if filing_date:
                    try:
                        doc_date = datetime.strptime(filing_date, '%Y%m%d')
                        if doc_date > analysis_dt:
                            report.add_detection(BiasDetection(
                                bias_type=BiasType.UNAVAILABLE_FILING,
                                severity='high',
                                description=f"Source document filed after analysis date",
                                evidence=f"Document filed {filing_date}, analysis date {analysis_date}",
                                location=f"Document: {doc.get('id', 'unknown')}",
                                recommendation="Exclude this document from analysis"
                            ))
                    except ValueError:
                        continue
                        
        return report
        
    def generate_summary(self, report: BiasReport) -> str:
        """Generate human-readable summary of bias report."""
        lines = [
            "=" * 60,
            "LOOK-AHEAD BIAS ANALYSIS REPORT",
            "=" * 60,
            f"Analysis Date: {report.analysis_date}",
            f"Query: {report.query[:50]}..." if len(report.query) > 50 else f"Query: {report.query}",
            "",
            f"STATUS: {'⚠️ BIAS DETECTED' if report.has_bias else '✅ NO BIAS DETECTED'}",
            f"Risk Level: {report.overall_risk.upper()}",
            f"Issues Found: {report.bias_count}",
            ""
        ]
        
        if report.detections:
            lines.append("DETECTIONS:")
            lines.append("-" * 40)
            
            for i, d in enumerate(report.detections, 1):
                severity_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[d.severity]
                lines.extend([
                    f"\n{i}. {severity_icon} [{d.severity.upper()}] {d.bias_type.value}",
                    f"   Description: {d.description}",
                    f"   Evidence: {d.evidence}",
                ])
                if d.recommendation:
                    lines.append(f"   Recommendation: {d.recommendation}")
                    
        lines.extend([
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)


# Usage
if __name__ == "__main__":
    detector = LookAheadBiasDetector()
    
    # Test text with potential bias
    test_text = """
    Apple's revenue in Q2 2024 is expected to exceed $100 billion based on 
    the 10-K for fiscal 2024. The company's guidance for 2025 suggests 
    continued growth. According to the Q3 2023 filing, margins improved.
    """
    
    # Analysis as of October 2023
    report = detector.detect_bias(test_text, "20231001")
    
    # Print summary
    print(detector.generate_summary(report))
    
    # Test document availability
    available, explanation = detector.check_document_availability(
        filing_type="10-Q",
        filing_period_end="20230930",
        analysis_date="20231101"
    )
    print(f"\nDocument Availability Check:")
    print(explanation)
