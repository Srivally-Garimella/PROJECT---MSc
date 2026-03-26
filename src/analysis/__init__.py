"""
Financial Analysis Module for TemporalGuard-RAG

Provides comprehensive financial analysis capabilities including:
- Projections and forecasting
- Historical analysis
- Financial ratios and metrics
- Valuation models
- Data loading from XBRL and market sources
- Uncertainty quantification (NOVEL)
- Numeric hallucination detection (NOVEL)
"""

from .formulas import FinancialFormulas
from .projections import ProjectionEngine
from .historical import HistoricalAnalyzer
from .data_loader import FinancialDataLoader, StockDataLoader
from .uncertainty import UncertaintyQuantifier, EnsembleProjector, UncertaintyEstimate, ConfidenceLevel
from .hallucination_detector import (
    NumericHallucinationDetector, 
    NumericGroundingValidator,
    HallucinationReport,
    VerificationStatus
)

__all__ = [
    # Core analysis
    'FinancialFormulas',
    'ProjectionEngine',
    'HistoricalAnalyzer',
    'FinancialDataLoader',
    'StockDataLoader',
    # Novel contributions
    'UncertaintyQuantifier',
    'EnsembleProjector',
    'UncertaintyEstimate',
    'ConfidenceLevel',
    'NumericHallucinationDetector',
    'NumericGroundingValidator',
    'HallucinationReport',
    'VerificationStatus'
]
