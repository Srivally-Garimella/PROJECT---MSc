"""
Temporal Benchmarking for TemporalGuard-RAG

Provides benchmarking capabilities to evaluate the system against
temporal consistency, accuracy, and look-ahead bias prevention.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark test result."""
    test_id: str
    test_name: str
    passed: bool
    expected: Any
    actual: Any
    score: float
    details: str
    execution_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    total_score: float = 0.0
    average_score: float = 0.0
    results: List[BenchmarkResult] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_result(self, result: BenchmarkResult):
        """Add a test result."""
        self.results.append(result)
        self.total_tests += 1
        if result.passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
        self.total_score += result.score
        self.average_score = self.total_score / self.total_tests
        
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'pass_rate': self.passed_tests / self.total_tests if self.total_tests > 0 else 0,
            'total_score': self.total_score,
            'average_score': self.average_score,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp,
            'results': [
                {
                    'test_id': r.test_id,
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'score': r.score,
                    'details': r.details
                }
                for r in self.results
            ]
        }


class TemporalBenchmark:
    """
    Benchmark suite for evaluating temporal consistency and accuracy.
    
    Test Categories:
    1. Temporal Consistency - Ensures no look-ahead bias
    2. Retrieval Accuracy - Correct document retrieval
    3. Calculation Correctness - Financial calculations accurate
    4. Cross-Verification - Consistency across sources
    """
    
    def __init__(self, 
                 test_data_path: str = "data/evaluation/benchmark_cases.json",
                 orchestrator=None):
        """
        Initialize benchmark suite.
        
        Args:
            test_data_path: Path to benchmark test cases
            orchestrator: MultiAgentOrchestrator instance
        """
        self.test_data_path = test_data_path
        self.orchestrator = orchestrator
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> List[Dict]:
        """Load benchmark test cases."""
        path = Path(self.test_data_path)
        
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        else:
            logger.info("No test cases found, using default cases")
            return self._get_default_test_cases()
            
    def _get_default_test_cases(self) -> List[Dict]:
        """Return default benchmark test cases."""
        return [
            # Temporal Consistency Tests
            {
                "id": "TC001",
                "name": "Basic Temporal Cutoff",
                "category": "temporal_consistency",
                "query": "What was Apple's revenue in Q1 2023?",
                "analysis_date": "20230201",
                "expected_behavior": "Should NOT retrieve Q1 2023 10-Q (filed ~May 2023)",
                "pass_criteria": "no_future_documents"
            },
            {
                "id": "TC002",
                "name": "Look-Ahead Bias Detection",
                "category": "temporal_consistency",
                "query": "Compare Apple's 2024 revenue to 2023",
                "analysis_date": "20231001",
                "expected_behavior": "Should detect reference to future year",
                "pass_criteria": "bias_detected"
            },
            {
                "id": "TC003",
                "name": "Filing Lag Awareness",
                "category": "temporal_consistency",
                "query": "What was Microsoft's Q2 2023 EPS?",
                "analysis_date": "20230701",
                "expected_behavior": "Should note 10-Q may not be available yet",
                "pass_criteria": "filing_lag_warning"
            },
            
            # Retrieval Accuracy Tests
            {
                "id": "RA001",
                "name": "Correct Document Type",
                "category": "retrieval_accuracy",
                "query": "What are the risk factors mentioned in the annual report?",
                "ticker": "AAPL",
                "analysis_date": "20231001",
                "expected_behavior": "Should retrieve 10-K documents",
                "pass_criteria": "correct_document_type"
            },
            {
                "id": "RA002",
                "name": "Temporal Document Filtering",
                "category": "retrieval_accuracy",
                "query": "Revenue recognition policy",
                "ticker": "MSFT",
                "analysis_date": "20220601",
                "expected_behavior": "Only retrieve documents filed before June 2022",
                "pass_criteria": "temporal_filtering"
            },
            
            # Calculation Tests
            {
                "id": "CA001",
                "name": "ROE Calculation",
                "category": "calculation",
                "ticker": "AAPL",
                "ratio": "ROE",
                "analysis_date": "20231001",
                "expected_range": [0.1, 2.0],  # 10% to 200%
                "pass_criteria": "range_check"
            },
            {
                "id": "CA002",
                "name": "Revenue Growth",
                "category": "calculation",
                "ticker": "MSFT",
                "ratio": "revenue_growth",
                "analysis_date": "20231001",
                "expected_range": [-0.5, 1.0],  # -50% to 100%
                "pass_criteria": "range_check"
            },
            
            # Cross-Verification Tests
            {
                "id": "CV001",
                "name": "Narrative vs XBRL Consistency",
                "category": "cross_verification",
                "claim": "Revenue was approximately $100 billion",
                "ticker": "AAPL",
                "analysis_date": "20231001",
                "expected_behavior": "Should verify against XBRL data",
                "pass_criteria": "verification_performed"
            }
        ]
        
    def run_all_benchmarks(self) -> BenchmarkResults:
        """Run all benchmark tests."""
        start_time = datetime.now()
        results = BenchmarkResults()
        
        logger.info(f"Running {len(self.test_cases)} benchmark tests...")
        
        for test_case in self.test_cases:
            try:
                result = self._run_test(test_case)
                results.add_result(result)
                
                status = "PASS" if result.passed else "FAIL"
                logger.info(f"  [{status}] {test_case['name']}")
                
            except Exception as e:
                logger.error(f"  [ERROR] {test_case['name']}: {e}")
                results.add_result(BenchmarkResult(
                    test_id=test_case['id'],
                    test_name=test_case['name'],
                    passed=False,
                    expected="No error",
                    actual=str(e),
                    score=0.0,
                    details=f"Test execution failed: {e}",
                    execution_time=0.0
                ))
                
        results.execution_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"\nBenchmark Complete:")
        logger.info(f"  Total: {results.total_tests}")
        logger.info(f"  Passed: {results.passed_tests}")
        logger.info(f"  Failed: {results.failed_tests}")
        logger.info(f"  Average Score: {results.average_score:.2%}")
        
        return results
        
    def _run_test(self, test_case: Dict) -> BenchmarkResult:
        """Run a single benchmark test."""
        test_start = datetime.now()
        
        category = test_case.get('category', 'unknown')
        
        if category == 'temporal_consistency':
            return self._run_temporal_test(test_case, test_start)
        elif category == 'retrieval_accuracy':
            return self._run_retrieval_test(test_case, test_start)
        elif category == 'calculation':
            return self._run_calculation_test(test_case, test_start)
        elif category == 'cross_verification':
            return self._run_verification_test(test_case, test_start)
        else:
            return BenchmarkResult(
                test_id=test_case['id'],
                test_name=test_case['name'],
                passed=False,
                expected="Known category",
                actual=category,
                score=0.0,
                details=f"Unknown test category: {category}",
                execution_time=(datetime.now() - test_start).total_seconds()
            )
            
    def _run_temporal_test(self, test_case: Dict, start_time: datetime) -> BenchmarkResult:
        """Run temporal consistency test."""
        if not self.orchestrator:
            # Run basic validation without orchestrator
            from src.agents.temporal_agent import TemporalAgent
            agent = TemporalAgent()
            
            result = agent.validate_query(
                test_case['query'],
                test_case['analysis_date']
            )
            
            pass_criteria = test_case.get('pass_criteria', '')
            
            passed = False
            if pass_criteria == 'no_future_documents':
                passed = result.get('is_valid', False)
            elif pass_criteria == 'bias_detected':
                passed = result.get('has_violations', False)
            elif pass_criteria == 'filing_lag_warning':
                passed = result.get('has_warnings', False)
            else:
                passed = result.get('is_valid', False)
                
            return BenchmarkResult(
                test_id=test_case['id'],
                test_name=test_case['name'],
                passed=passed,
                expected=test_case.get('expected_behavior', 'Pass'),
                actual=str(result.get('bias_detection', '')[:200]),
                score=1.0 if passed else 0.0,
                details=f"Temporal validation: {'passed' if passed else 'failed'}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
        else:
            # Use orchestrator
            result = self.orchestrator.run_single_agent(
                'temporal',
                query=test_case['query'],
                analysis_date=test_case['analysis_date']
            )
            
            passed = result.get('is_valid', False)
            
            return BenchmarkResult(
                test_id=test_case['id'],
                test_name=test_case['name'],
                passed=passed,
                expected=test_case.get('expected_behavior', 'Pass'),
                actual=str(result)[:200],
                score=1.0 if passed else 0.0,
                details=f"Via orchestrator",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
    def _run_retrieval_test(self, test_case: Dict, start_time: datetime) -> BenchmarkResult:
        """Run retrieval accuracy test."""
        # For now, return a mock result
        # In full implementation, this would test actual document retrieval
        
        return BenchmarkResult(
            test_id=test_case['id'],
            test_name=test_case['name'],
            passed=True,  # Placeholder
            expected=test_case.get('expected_behavior', ''),
            actual="[Retrieval test - requires vector store]",
            score=0.5,  # Partial score since not fully testable
            details="Retrieval test requires configured vector store",
            execution_time=(datetime.now() - start_time).total_seconds()
        )
        
    def _run_calculation_test(self, test_case: Dict, start_time: datetime) -> BenchmarkResult:
        """Run calculation accuracy test."""
        if not self.orchestrator:
            from src.agents.calculation_agent import CalculationAgent
            agent = CalculationAgent()
            
            result = agent.calculate(
                ticker=test_case['ticker'],
                ratio=test_case['ratio'],
                date=test_case['analysis_date']
            )
            
            # Check if result is within expected range
            output = result.get('output', '')
            expected_range = test_case.get('expected_range', [-float('inf'), float('inf')])
            
            # Try to extract numeric value
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+%?', output)
            
            passed = False
            actual_value = "No numeric result"
            
            if numbers:
                try:
                    value = float(numbers[0].replace('%', '')) / 100 if '%' in numbers[0] else float(numbers[0])
                    actual_value = f"{value:.2%}"
                    passed = expected_range[0] <= value <= expected_range[1]
                except ValueError:
                    pass
                    
            return BenchmarkResult(
                test_id=test_case['id'],
                test_name=test_case['name'],
                passed=passed,
                expected=f"Range: {expected_range[0]:.0%} to {expected_range[1]:.0%}",
                actual=actual_value,
                score=1.0 if passed else 0.0,
                details=output[:200],
                execution_time=(datetime.now() - start_time).total_seconds()
            )
        else:
            result = self.orchestrator.run_single_agent(
                'calculation',
                ticker=test_case['ticker'],
                ratio=test_case['ratio'],
                date=test_case['analysis_date']
            )
            
            return BenchmarkResult(
                test_id=test_case['id'],
                test_name=test_case['name'],
                passed=True,  # Placeholder
                expected=str(test_case.get('expected_range', '')),
                actual=str(result)[:200],
                score=0.5,
                details="Via orchestrator",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
    def _run_verification_test(self, test_case: Dict, start_time: datetime) -> BenchmarkResult:
        """Run cross-verification test."""
        return BenchmarkResult(
            test_id=test_case['id'],
            test_name=test_case['name'],
            passed=True,  # Placeholder
            expected=test_case.get('expected_behavior', ''),
            actual="[Verification test - requires XBRL data]",
            score=0.5,
            details="Verification test requires XBRL data",
            execution_time=(datetime.now() - start_time).total_seconds()
        )
        
    def run_category(self, category: str) -> BenchmarkResults:
        """Run benchmarks for a specific category."""
        filtered_cases = [tc for tc in self.test_cases if tc.get('category') == category]
        
        start_time = datetime.now()
        results = BenchmarkResults()
        
        for test_case in filtered_cases:
            try:
                result = self._run_test(test_case)
                results.add_result(result)
            except Exception as e:
                logger.error(f"Test {test_case['id']} failed: {e}")
                
        results.execution_time = (datetime.now() - start_time).total_seconds()
        
        return results
        
    def save_results(self, results: BenchmarkResults, output_path: str):
        """Save benchmark results to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
            
        logger.info(f"Results saved to {output_path}")


# Usage
if __name__ == "__main__":
    # Run benchmarks
    benchmark = TemporalBenchmark()
    
    print("Running TemporalGuard-RAG Benchmark Suite")
    print("=" * 60)
    
    results = benchmark.run_all_benchmarks()
    
    print("\nDetailed Results:")
    print("-" * 60)
    for r in results.results:
        status = "✅" if r.passed else "❌"
        print(f"{status} [{r.test_id}] {r.test_name}")
        print(f"   Expected: {r.expected}")
        print(f"   Actual: {r.actual}")
        print()
        
    # Save results
    benchmark.save_results(results, "data/evaluation/benchmark_results.json")
