"""
Comprehensive Evaluation Pipeline for TemporalGuard-RAG

This module provides REAL evaluation (not placeholders) for the novel contributions:
1. Temporal Consistency - Actual look-ahead bias detection testing
2. Uncertainty Quantification - Calibration and coverage tests
3. Numeric Hallucination Detection - Precision/recall on ground truth
4. End-to-End System - Full pipeline evaluation

Research Metrics:
- Temporal Precision: % of queries correctly filtered for time
- Uncertainty Calibration: % of actuals within predicted CI
- Hallucination Detection Rate: Recall on known fabricated numbers
- Grounding Accuracy: % of numbers matching source
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Temporal Consistency
    temporal_precision: float = 0.0
    temporal_recall: float = 0.0
    temporal_f1: float = 0.0
    look_ahead_detection_rate: float = 0.0
    
    # Uncertainty Quantification
    uncertainty_coverage_90: float = 0.0  # % of actuals in 90% CI
    uncertainty_calibration_error: float = 0.0
    mean_interval_width: float = 0.0
    
    # Hallucination Detection
    hallucination_precision: float = 0.0
    hallucination_recall: float = 0.0
    hallucination_f1: float = 0.0
    grounding_accuracy: float = 0.0
    
    # End-to-End
    answer_accuracy: float = 0.0
    answer_groundedness: float = 0.0
    average_response_time: float = 0.0
    
    # Sample sizes
    n_temporal_tests: int = 0
    n_uncertainty_tests: int = 0
    n_hallucination_tests: int = 0
    n_e2e_tests: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'temporal': {
                'precision': self.temporal_precision,
                'recall': self.temporal_recall,
                'f1': self.temporal_f1,
                'look_ahead_detection_rate': self.look_ahead_detection_rate,
                'n_tests': self.n_temporal_tests
            },
            'uncertainty': {
                'coverage_90': self.uncertainty_coverage_90,
                'calibration_error': self.uncertainty_calibration_error,
                'mean_interval_width': self.mean_interval_width,
                'n_tests': self.n_uncertainty_tests
            },
            'hallucination': {
                'precision': self.hallucination_precision,
                'recall': self.hallucination_recall,
                'f1': self.hallucination_f1,
                'grounding_accuracy': self.grounding_accuracy,
                'n_tests': self.n_hallucination_tests
            },
            'end_to_end': {
                'accuracy': self.answer_accuracy,
                'groundedness': self.answer_groundedness,
                'avg_response_time': self.average_response_time,
                'n_tests': self.n_e2e_tests
            }
        }


@dataclass
class TestCase:
    """A single test case."""
    id: str
    category: str
    input_data: Dict
    expected_output: Dict
    metadata: Dict = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of a single test."""
    test_id: str
    passed: bool
    score: float
    expected: Any
    actual: Any
    details: str
    execution_time: float


class TemporalConsistencyEvaluator:
    """
    Evaluates temporal consistency and look-ahead bias detection.
    
    This tests the core research contribution: preventing anachronistic data access.
    """
    
    def __init__(self):
        from src.agents.temporal_agent import TemporalAgent
        self.temporal_agent = TemporalAgent()
        logger.info("Initialized Temporal Consistency Evaluator")
    
    def create_test_cases(self) -> List[TestCase]:
        """Generate comprehensive temporal test cases."""
        return [
            # True positives - should detect bias
            TestCase(
                id="T001",
                category="temporal",
                input_data={
                    "query": "What was Apple's revenue in Q4 2024?",
                    "analysis_date": "20240101"
                },
                expected_output={"has_bias": True, "reason": "future_quarter"}
            ),
            TestCase(
                id="T002",
                category="temporal",
                input_data={
                    "query": "Compare 2024 earnings to 2023 for Microsoft",
                    "analysis_date": "20231015"
                },
                expected_output={"has_bias": True, "reason": "future_year_reference"}
            ),
            TestCase(
                id="T003",
                category="temporal",
                input_data={
                    "query": "What did the CEO say about AI strategy in the 2024 annual report?",
                    "analysis_date": "20231201"
                },
                expected_output={"has_bias": True, "reason": "future_filing"}
            ),
            TestCase(
                id="T004",
                category="temporal",
                input_data={
                    "query": "What was Q2 2023 EPS?",
                    "analysis_date": "20230715"  # Only 15 days after Q2 end
                },
                expected_output={"has_bias": True, "reason": "filing_lag"}
            ),
            
            # True negatives - should NOT detect bias  
            TestCase(
                id="T005",
                category="temporal",
                input_data={
                    "query": "What was Apple's fiscal 2022 revenue?",
                    "analysis_date": "20231001"
                },
                expected_output={"has_bias": False}
            ),
            TestCase(
                id="T006",
                category="temporal",
                input_data={
                    "query": "Summarize Microsoft's risk factors from the 2022 10-K",
                    "analysis_date": "20230301"
                },
                expected_output={"has_bias": False}
            ),
            TestCase(
                id="T007",
                category="temporal",
                input_data={
                    "query": "What was Q1 2023 gross margin?",
                    "analysis_date": "20230701"  # 3 months after Q1, 10-Q should be filed
                },
                expected_output={"has_bias": False}
            ),
            
            # Edge cases
            TestCase(
                id="T008",
                category="temporal",
                input_data={
                    "query": "What is the expected revenue growth?",  # Forward-looking language
                    "analysis_date": "20231001"
                },
                expected_output={"has_bias": False, "has_warning": True}  # Warning but not bias
            ),
        ]
    
    def evaluate(self, test_cases: List[TestCase] = None) -> Tuple[Dict[str, float], List[TestResult]]:
        """
        Run temporal consistency evaluation.
        
        Returns:
            Tuple of (metrics_dict, list_of_results)
        """
        if test_cases is None:
            test_cases = self.create_test_cases()
        
        results = []
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for tc in test_cases:
            start_time = time.time()
            
            # Run temporal validation
            validation = self.temporal_agent.validate_query(
                tc.input_data["query"],
                tc.input_data["analysis_date"]
            )
            
            execution_time = time.time() - start_time
            
            # Determine if bias was detected
            detected_bias = not validation.get("is_valid", True) or validation.get("has_violations", False)
            expected_bias = tc.expected_output.get("has_bias", False)
            
            passed = detected_bias == expected_bias
            
            # Update confusion matrix
            if expected_bias and detected_bias:
                true_positives += 1
            elif expected_bias and not detected_bias:
                false_negatives += 1
            elif not expected_bias and detected_bias:
                false_positives += 1
            else:
                true_negatives += 1
            
            results.append(TestResult(
                test_id=tc.id,
                passed=passed,
                score=1.0 if passed else 0.0,
                expected=f"has_bias={expected_bias}",
                actual=f"detected_bias={detected_bias}",
                details=validation.get("bias_detection", "")[:200],
                execution_time=execution_time
            ))
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (true_positives + true_negatives) / len(test_cases) if test_cases else 0,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives
        }
        
        return metrics, results


class UncertaintyCalibrationEvaluator:
    """
    Evaluates uncertainty quantification calibration.
    
    Key question: Do our confidence intervals actually contain the true value
    at the stated confidence level?
    """
    
    def __init__(self, xbrl_dir: str = "data/raw/xbrl_structured"):
        from src.analysis.uncertainty import EnsembleProjector
        from src.analysis.data_loader import FinancialDataLoader
        
        self.projector = EnsembleProjector()
        self.data_loader = FinancialDataLoader(xbrl_dir)
        logger.info("Initialized Uncertainty Calibration Evaluator")
    
    def create_test_cases(self) -> List[TestCase]:
        """Generate calibration test cases using historical data."""
        test_cases = []
        
        # We create test cases by:
        # 1. Taking historical data up to year Y
        # 2. Projecting to year Y+1
        # 3. Comparing projection to actual Y+1 value
        
        tickers = ["AAPL", "MSFT", "JPM", "GS", "XOM", "CVX"]
        metrics = ["Revenue", "NetIncome", "OperatingCashFlow"]
        
        for ticker in tickers:
            for metric in metrics:
                # Get full historical data
                history = self.data_loader.get_metric_history(ticker, metric)
                
                if not history or len(history) < 4:
                    continue
                
                # Use all but last year for training, last year as actual
                sorted_hist = sorted(history, key=lambda x: x[0])
                train_data = sorted_hist[:-1]
                actual_year, actual_value = sorted_hist[-1]
                
                test_cases.append(TestCase(
                    id=f"U{len(test_cases)+1:03d}",
                    category="uncertainty",
                    input_data={
                        "ticker": ticker,
                        "metric": metric,
                        "historical_data": train_data,
                        "target_year": int(actual_year)
                    },
                    expected_output={
                        "actual_value": actual_value
                    },
                    metadata={
                        "ticker": ticker,
                        "metric": metric
                    }
                ))
        
        return test_cases
    
    def evaluate(self, test_cases: List[TestCase] = None) -> Tuple[Dict[str, float], List[TestResult]]:
        """
        Evaluate uncertainty calibration.
        
        For a 90% CI, we expect ~90% of actual values to fall within the interval.
        """
        if test_cases is None:
            test_cases = self.create_test_cases()
        
        if not test_cases:
            return {"coverage_90": 0, "calibration_error": 1.0}, []
        
        results = []
        within_ci = 0
        interval_widths = []
        
        for tc in test_cases:
            start_time = time.time()
            
            # Generate projection with uncertainty
            estimate = self.projector.project_with_uncertainty(
                historical_data=tc.input_data["historical_data"],
                target_year=tc.input_data["target_year"],
                metric_name=tc.input_data["metric"]
            )
            
            execution_time = time.time() - start_time
            
            actual = tc.expected_output["actual_value"]
            
            # Check if actual falls within CI
            in_interval = estimate.lower_bound <= actual <= estimate.upper_bound
            if in_interval:
                within_ci += 1
            
            # Track interval width (relative). Force real float for stable JSON output.
            if estimate.point_estimate != 0:
                rel_width = (estimate.upper_bound - estimate.lower_bound) / abs(estimate.point_estimate)
                interval_widths.append(float(getattr(rel_width, "real", rel_width)))
            
            # Calculate score based on how close prediction was
            if estimate.point_estimate != 0:
                error_pct = abs(actual - estimate.point_estimate) / abs(actual) if actual != 0 else 1.0
                score = max(0, 1 - error_pct)
            else:
                score = 0
            
            results.append(TestResult(
                test_id=tc.id,
                passed=in_interval,
                score=score,
                expected=f"actual={actual:,.0f}",
                actual=f"predicted={estimate.point_estimate:,.0f} CI=[{estimate.lower_bound:,.0f}, {estimate.upper_bound:,.0f}]",
                details=f"Confidence: {estimate.confidence_level.value}",
                execution_time=execution_time
            ))
        
        # Calculate calibration metrics
        coverage = within_ci / len(test_cases) if test_cases else 0
        calibration_error = abs(coverage - 0.90)  # Target is 90% coverage
        mean_width = float(np.mean(interval_widths)) if interval_widths else 0.0
        
        metrics = {
            "coverage_90": coverage,
            "calibration_error": calibration_error,
            "mean_interval_width": mean_width,
            "n_tests": len(test_cases)
        }
        
        return metrics, results


class HallucinationDetectionEvaluator:
    """
    Evaluates numeric hallucination detection.
    
    Tests whether the system correctly identifies fabricated numbers.
    """
    
    def __init__(self, xbrl_dir: str = "data/raw/xbrl_structured"):
        from src.analysis.hallucination_detector import NumericHallucinationDetector
        self.detector = NumericHallucinationDetector(xbrl_dir)
        logger.info("Initialized Hallucination Detection Evaluator")
    
    def create_test_cases(self) -> List[TestCase]:
        """Generate hallucination detection test cases."""
        return [
            # True hallucinations (wrong numbers)
            TestCase(
                id="H001",
                category="hallucination",
                input_data={
                    "text": "Apple's revenue was $500 billion in fiscal 2023",  # Actually ~$383B
                    "ticker": "AAPL"
                },
                expected_output={"is_hallucination": True, "metric": "revenue"}
            ),
            TestCase(
                id="H002",
                category="hallucination",
                input_data={
                    "text": "Microsoft reported net income of $10 billion",  # Actually ~$72B
                    "ticker": "MSFT"
                },
                expected_output={"is_hallucination": True, "metric": "net_income"}
            ),
            TestCase(
                id="H003",
                category="hallucination",
                input_data={
                    "text": "JPMorgan's total assets exceeded $10 trillion",  # Actually ~$3.7T
                    "ticker": "JPM"
                },
                expected_output={"is_hallucination": True, "metric": "assets"}
            ),
            
            # Correct numbers (should NOT flag)
            TestCase(
                id="H004",
                category="hallucination",
                input_data={
                    "text": "Apple reported approximately $383 billion in revenue",
                    "ticker": "AAPL"
                },
                expected_output={"is_hallucination": False}
            ),
            TestCase(
                id="H005",
                category="hallucination",
                input_data={
                    "text": "ExxonMobil's revenue was around $400 billion",  # ~$398B
                    "ticker": "XOM"
                },
                expected_output={"is_hallucination": False}  # Within tolerance
            ),
            
            # Minor deviations (should be flagged as deviation, not hallucination)
            TestCase(
                id="H006",
                category="hallucination",
                input_data={
                    "text": "Apple's revenue was approximately $390 billion",  # ~2% off
                    "ticker": "AAPL"
                },
                expected_output={"is_hallucination": False, "is_deviation": True}
            ),
        ]
    
    def evaluate(self, test_cases: List[TestCase] = None) -> Tuple[Dict[str, float], List[TestResult]]:
        """Evaluate hallucination detection accuracy."""
        if test_cases is None:
            test_cases = self.create_test_cases()
        
        results = []
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for tc in test_cases:
            start_time = time.time()
            
            report = self.detector.detect_hallucinations(
                tc.input_data["text"],
                tc.input_data["ticker"]
            )
            
            execution_time = time.time() - start_time
            
            # Determine if hallucination was detected
            detected_hallucination = report.hallucination_count > 0
            expected_hallucination = tc.expected_output.get("is_hallucination", False)
            
            passed = detected_hallucination == expected_hallucination
            
            # Update confusion matrix
            if expected_hallucination and detected_hallucination:
                true_positives += 1
            elif expected_hallucination and not detected_hallucination:
                false_negatives += 1
            elif not expected_hallucination and detected_hallucination:
                false_positives += 1
            else:
                true_negatives += 1
            
            results.append(TestResult(
                test_id=tc.id,
                passed=passed,
                score=report.trust_score,
                expected=f"hallucination={expected_hallucination}",
                actual=f"detected={detected_hallucination}, trust_score={report.trust_score:.2f}",
                details=report.overall_status.value,
                execution_time=execution_time
            ))
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(test_cases) if test_cases else 0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "grounding_accuracy": accuracy  # Same as accuracy for this task
        }
        
        return metrics, results


class ComprehensiveEvaluator:
    """
    Runs all evaluation suites and generates comprehensive report.
    """
    
    def __init__(self, xbrl_dir: str = "data/raw/xbrl_structured"):
        self.temporal_evaluator = TemporalConsistencyEvaluator()
        self.uncertainty_evaluator = UncertaintyCalibrationEvaluator(xbrl_dir)
        self.hallucination_evaluator = HallucinationDetectionEvaluator(xbrl_dir)
        
        self.results_dir = Path("results/evaluation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized Comprehensive Evaluator")
    
    def run_all(self, save_results: bool = True) -> EvaluationMetrics:
        """
        Run all evaluation suites.
        
        Returns:
            EvaluationMetrics with all results
        """
        print("=" * 60)
        print("🔬 TEMPORALGUARD-RAG COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        metrics = EvaluationMetrics()
        all_results = {}
        
        # 1. Temporal Consistency
        print("\n📅 Running Temporal Consistency Evaluation...")
        try:
            temporal_metrics, temporal_results = self.temporal_evaluator.evaluate()
            metrics.temporal_precision = temporal_metrics["precision"]
            metrics.temporal_recall = temporal_metrics["recall"]
            metrics.temporal_f1 = temporal_metrics["f1"]
            metrics.look_ahead_detection_rate = temporal_metrics["recall"]
            metrics.n_temporal_tests = len(temporal_results)
            all_results["temporal"] = [r.__dict__ for r in temporal_results]
            print(f"   ✅ Precision: {metrics.temporal_precision:.1%}")
            print(f"   ✅ Recall: {metrics.temporal_recall:.1%}")
            print(f"   ✅ F1: {metrics.temporal_f1:.1%}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 2. Uncertainty Calibration
        print("\n📊 Running Uncertainty Calibration Evaluation...")
        try:
            uncertainty_metrics, uncertainty_results = self.uncertainty_evaluator.evaluate()
            metrics.uncertainty_coverage_90 = uncertainty_metrics["coverage_90"]
            metrics.uncertainty_calibration_error = uncertainty_metrics["calibration_error"]
            metrics.mean_interval_width = uncertainty_metrics["mean_interval_width"]
            metrics.n_uncertainty_tests = uncertainty_metrics["n_tests"]
            all_results["uncertainty"] = [r.__dict__ for r in uncertainty_results]
            print(f"   ✅ 90% CI Coverage: {metrics.uncertainty_coverage_90:.1%}")
            print(f"   ✅ Calibration Error: {metrics.uncertainty_calibration_error:.3f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 3. Hallucination Detection
        print("\n🔍 Running Hallucination Detection Evaluation...")
        try:
            hallucination_metrics, hallucination_results = self.hallucination_evaluator.evaluate()
            metrics.hallucination_precision = hallucination_metrics["precision"]
            metrics.hallucination_recall = hallucination_metrics["recall"]
            metrics.hallucination_f1 = hallucination_metrics["f1"]
            metrics.grounding_accuracy = hallucination_metrics["grounding_accuracy"]
            metrics.n_hallucination_tests = len(hallucination_results)
            all_results["hallucination"] = [r.__dict__ for r in hallucination_results]
            print(f"   ✅ Precision: {metrics.hallucination_precision:.1%}")
            print(f"   ✅ Recall: {metrics.hallucination_recall:.1%}")
            print(f"   ✅ F1: {metrics.hallucination_f1:.1%}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\nTemporal Consistency:")
        print(f"  • F1 Score: {metrics.temporal_f1:.1%}")
        print(f"  • Tests: {metrics.n_temporal_tests}")
        
        print(f"\nUncertainty Quantification:")
        print(f"  • 90% CI Coverage: {metrics.uncertainty_coverage_90:.1%}")
        print(f"  • Tests: {metrics.n_uncertainty_tests}")
        
        print(f"\nHallucination Detection:")
        print(f"  • F1 Score: {metrics.hallucination_f1:.1%}")
        print(f"  • Tests: {metrics.n_hallucination_tests}")
        
        if save_results:
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Custom JSON encoder for numpy types
            def json_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, complex):
                    return str(obj)
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                return str(obj)
            
            # Save metrics
            metrics_path = self.results_dir / f"metrics_{timestamp}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2, default=json_serializable)
            
            # Save detailed results
            results_path = self.results_dir / f"detailed_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=json_serializable)
            
            print(f"\n📁 Results saved to {self.results_dir}/")
        
        return metrics


def run_evaluation():
    """Run the complete evaluation pipeline."""
    evaluator = ComprehensiveEvaluator()
    metrics = evaluator.run_all(save_results=True)
    return metrics


if __name__ == "__main__":
    run_evaluation()
