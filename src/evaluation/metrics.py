"""
Evaluation Metrics for TemporalGuard-RAG

Comprehensive metrics for evaluating RAG system performance
including retrieval quality, answer accuracy, and temporal consistency.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import json
import logging
import math
import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Single metric result."""
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for financial RAG systems.
    
    Metrics Categories:
    1. Retrieval Metrics - Precision, Recall, MRR, NDCG
    2. Answer Quality Metrics - Accuracy, Completeness, Groundedness
    3. Temporal Metrics - Bias detection rate, Temporal precision
    4. Financial Metrics - Calculation accuracy, Source attribution
    """
    
    def __init__(self):
        """Initialize evaluation metrics calculator."""
        self.results_history = []
        
    # ═══════════════════════════════════════════════════════════════
    # RETRIEVAL METRICS
    # ═══════════════════════════════════════════════════════════════
    
    def precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs (ground truth)
            k: Number of top results to consider
            
        Returns:
            Precision@K score
        """
        if k <= 0 or not retrieved:
            return 0.0
            
        retrieved_at_k = retrieved[:k]
        relevant_set = set(relevant)
        
        relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant_set)
        
        return relevant_retrieved / k
        
    def recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs (ground truth)
            k: Number of top results to consider
            
        Returns:
            Recall@K score
        """
        if not relevant:
            return 0.0
            
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        relevant_retrieved = len(retrieved_at_k & relevant_set)
        
        return relevant_retrieved / len(relevant_set)
        
    def mean_reciprocal_rank(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            
        Returns:
            MRR score
        """
        relevant_set = set(relevant)
        
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant_set:
                return 1.0 / rank
                
        return 0.0
        
    def ndcg_at_k(self, retrieved: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            retrieved: List of retrieved document IDs
            relevance_scores: Dict mapping doc IDs to relevance scores
            k: Number of top results
            
        Returns:
            NDCG@K score
        """
        def dcg(scores: List[float]) -> float:
            return sum(
                (2 ** score - 1) / math.log2(rank + 2)
                for rank, score in enumerate(scores)
            )
            
        # Get relevance scores for retrieved docs
        retrieved_scores = [
            relevance_scores.get(doc, 0.0) 
            for doc in retrieved[:k]
        ]
        
        # Calculate ideal DCG
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        
        dcg_value = dcg(retrieved_scores)
        idcg_value = dcg(ideal_scores)
        
        if idcg_value == 0:
            return 0.0
            
        return dcg_value / idcg_value
        
    def retrieval_metrics(self, 
                          retrieved: List[str], 
                          relevant: List[str],
                          relevance_scores: Optional[Dict[str, float]] = None,
                          k: int = 5) -> Dict[str, float]:
        """
        Calculate all retrieval metrics.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            relevance_scores: Optional dict of relevance scores for NDCG
            k: K value for Precision/Recall/NDCG
            
        Returns:
            Dict of metric names to scores
        """
        if relevance_scores is None:
            relevance_scores = {doc: 1.0 for doc in relevant}
            
        return {
            f'precision_at_{k}': self.precision_at_k(retrieved, relevant, k),
            f'recall_at_{k}': self.recall_at_k(retrieved, relevant, k),
            'mrr': self.mean_reciprocal_rank(retrieved, relevant),
            f'ndcg_at_{k}': self.ndcg_at_k(retrieved, relevance_scores, k)
        }
        
    # ═══════════════════════════════════════════════════════════════
    # ANSWER QUALITY METRICS
    # ═══════════════════════════════════════════════════════════════
    
    def answer_accuracy(self, 
                        predicted_answer: str, 
                        ground_truth: str,
                        tolerance: float = 0.05) -> float:
        """
        Calculate answer accuracy, especially for numerical answers.
        
        Args:
            predicted_answer: Model's predicted answer
            ground_truth: Known correct answer
            tolerance: Acceptable percentage difference for numerical answers
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        # Try to extract and compare numbers
        pred_numbers = re.findall(r'[\d,]+\.?\d*', predicted_answer.replace(',', ''))
        truth_numbers = re.findall(r'[\d,]+\.?\d*', ground_truth.replace(',', ''))
        
        if pred_numbers and truth_numbers:
            try:
                pred_val = float(pred_numbers[0])
                truth_val = float(truth_numbers[0])
                
                if truth_val == 0:
                    return 1.0 if pred_val == 0 else 0.0
                    
                relative_error = abs(pred_val - truth_val) / abs(truth_val)
                
                if relative_error <= tolerance:
                    return 1.0
                elif relative_error <= tolerance * 2:
                    return 0.75
                elif relative_error <= tolerance * 5:
                    return 0.5
                else:
                    return 0.0
                    
            except ValueError:
                pass
                
        # Fall back to text comparison
        pred_clean = predicted_answer.lower().strip()
        truth_clean = ground_truth.lower().strip()
        
        if pred_clean == truth_clean:
            return 1.0
        elif truth_clean in pred_clean or pred_clean in truth_clean:
            return 0.8
        else:
            # Calculate word overlap
            pred_words = set(pred_clean.split())
            truth_words = set(truth_clean.split())
            
            if not truth_words:
                return 0.0
                
            overlap = len(pred_words & truth_words) / len(truth_words)
            return overlap * 0.5  # Partial credit
            
    def answer_completeness(self, 
                            answer: str, 
                            required_elements: List[str]) -> float:
        """
        Calculate answer completeness based on required elements.
        
        Args:
            answer: Model's answer
            required_elements: List of elements that should be in answer
            
        Returns:
            Completeness score (0.0 to 1.0)
        """
        if not required_elements:
            return 1.0
            
        answer_lower = answer.lower()
        
        found = sum(
            1 for element in required_elements
            if element.lower() in answer_lower
        )
        
        return found / len(required_elements)
        
    def groundedness_score(self, 
                           answer: str, 
                           source_documents: List[str]) -> float:
        """
        Calculate how well the answer is grounded in source documents.
        
        Args:
            answer: Model's answer
            source_documents: List of source document texts
            
        Returns:
            Groundedness score (0.0 to 1.0)
        """
        if not answer or not source_documents:
            return 0.0
            
        # Combine all source texts
        source_text = ' '.join(source_documents).lower()
        
        # Extract key phrases from answer (simple approach)
        answer_words = answer.lower().split()
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'as', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'between', 'under', 'and', 'but', 'or',
                      'if', 'then', 'than', 'this', 'that', 'these', 'those', 'it'}
        
        content_words = [w for w in answer_words if w not in stop_words and len(w) > 2]
        
        if not content_words:
            return 0.5  # Can't determine
            
        # Check how many content words appear in sources
        grounded = sum(1 for word in content_words if word in source_text)
        
        return grounded / len(content_words)
        
    # ═══════════════════════════════════════════════════════════════
    # TEMPORAL CONSISTENCY METRICS
    # ═══════════════════════════════════════════════════════════════
    
    def temporal_precision(self,
                           retrieved_dates: List[str],
                           cutoff_date: str) -> float:
        """
        Calculate temporal precision - fraction of retrieved docs before cutoff.
        
        Args:
            retrieved_dates: List of filing dates for retrieved documents
            cutoff_date: Analysis date cutoff
            
        Returns:
            Temporal precision score (0.0 to 1.0)
        """
        if not retrieved_dates:
            return 1.0  # No documents = no violations
            
        try:
            cutoff = datetime.strptime(cutoff_date, '%Y%m%d')
        except ValueError:
            return 0.5  # Can't parse cutoff
            
        valid_count = 0
        for date_str in retrieved_dates:
            try:
                doc_date = datetime.strptime(date_str, '%Y%m%d')
                if doc_date <= cutoff:
                    valid_count += 1
            except ValueError:
                continue
                
        return valid_count / len(retrieved_dates)
        
    def bias_detection_rate(self,
                            test_cases: List[Dict],
                            detector_results: List[bool]) -> float:
        """
        Calculate the rate at which look-ahead bias is correctly detected.
        
        Args:
            test_cases: List of test cases with 'has_bias' field
            detector_results: List of boolean detector outputs (True = bias detected)
            
        Returns:
            Detection rate (accuracy)
        """
        if len(test_cases) != len(detector_results):
            raise ValueError("Test cases and results must have same length")
            
        if not test_cases:
            return 0.0
            
        correct = sum(
            1 for test, result in zip(test_cases, detector_results)
            if test.get('has_bias', False) == result
        )
        
        return correct / len(test_cases)
        
    def temporal_leakage_score(self,
                               answer: str,
                               analysis_date: str) -> float:
        """
        Detect potential temporal information leakage in answer.
        
        Args:
            answer: Model's answer text
            analysis_date: Point-in-time date
            
        Returns:
            Leakage score (0.0 = no leakage, 1.0 = severe leakage)
        """
        try:
            analysis_dt = datetime.strptime(analysis_date, '%Y%m%d')
            analysis_year = analysis_dt.year
        except ValueError:
            return 0.0
            
        # Find year references in answer
        years = re.findall(r'20\d{2}', answer)
        
        future_refs = sum(1 for y in years if int(y) > analysis_year)
        
        if not years:
            return 0.0
            
        leakage = future_refs / len(years)
        
        # Check for future-oriented language
        future_indicators = [
            'will be', 'is expected', 'forecast', 'projection',
            'anticipated', 'outlook for', 'guidance'
        ]
        
        answer_lower = answer.lower()
        future_language = sum(1 for ind in future_indicators if ind in answer_lower)
        
        if future_language > 0:
            leakage = min(1.0, leakage + 0.2 * future_language)
            
        return leakage
        
    # ═══════════════════════════════════════════════════════════════
    # FINANCIAL CALCULATION METRICS
    # ═══════════════════════════════════════════════════════════════
    
    def calculation_accuracy(self,
                             calculated_value: float,
                             ground_truth: float,
                             metric_type: str = 'ratio') -> float:
        """
        Calculate accuracy of financial calculations.
        
        Args:
            calculated_value: Model's calculated value
            ground_truth: Known correct value
            metric_type: Type of metric ('ratio', 'percentage', 'currency')
            
        Returns:
            Accuracy score
        """
        if ground_truth == 0:
            return 1.0 if calculated_value == 0 else 0.0
            
        # Set tolerance based on metric type
        tolerances = {
            'ratio': 0.01,       # 1% tolerance for ratios
            'percentage': 0.02,  # 2% for percentages
            'currency': 0.001,   # 0.1% for currency (should be more precise)
            'growth': 0.05       # 5% for growth rates
        }
        
        tolerance = tolerances.get(metric_type, 0.02)
        
        relative_error = abs(calculated_value - ground_truth) / abs(ground_truth)
        
        if relative_error <= tolerance:
            return 1.0
        elif relative_error <= tolerance * 2:
            return 0.8
        elif relative_error <= tolerance * 5:
            return 0.5
        elif relative_error <= tolerance * 10:
            return 0.2
        else:
            return 0.0
            
    def source_attribution_score(self,
                                 answer: str,
                                 expected_sources: List[str]) -> float:
        """
        Evaluate quality of source attribution in answer.
        
        Args:
            answer: Model's answer
            expected_sources: List of source identifiers that should be cited
            
        Returns:
            Attribution score
        """
        if not expected_sources:
            return 1.0
            
        answer_lower = answer.lower()
        
        # Check for source citations
        cited = sum(
            1 for source in expected_sources
            if source.lower() in answer_lower
        )
        
        base_score = cited / len(expected_sources)
        
        # Bonus for proper citation format
        citation_patterns = [
            r'10-[KQ]',           # SEC filing types
            r'Q[1-4]\s*20\d{2}',  # Quarter references
            r'fiscal\s*\d{4}',   # Fiscal year
            r'dated\s',          # Date attribution
            r'according to',     # Source attribution
        ]
        
        bonus = sum(
            0.1 for pattern in citation_patterns
            if re.search(pattern, answer, re.IGNORECASE)
        )
        
        return min(1.0, base_score + bonus)
        
    # ═══════════════════════════════════════════════════════════════
    # AGGREGATE METRICS
    # ═══════════════════════════════════════════════════════════════
    
    def comprehensive_evaluation(self,
                                 prediction: Dict[str, Any],
                                 ground_truth: Dict[str, Any]) -> Dict[str, MetricResult]:
        """
        Run comprehensive evaluation on a prediction.
        
        Args:
            prediction: Prediction dictionary with answer, retrieved_docs, etc.
            ground_truth: Ground truth dictionary
            
        Returns:
            Dict of metric name to MetricResult
        """
        results = {}
        
        # Retrieval metrics
        if 'retrieved_docs' in prediction and 'relevant_docs' in ground_truth:
            retrieval = self.retrieval_metrics(
                prediction['retrieved_docs'],
                ground_truth['relevant_docs'],
                k=5
            )
            for name, score in retrieval.items():
                results[name] = MetricResult(name, score)
                
        # Answer accuracy
        if 'answer' in prediction and 'answer' in ground_truth:
            accuracy = self.answer_accuracy(
                prediction['answer'],
                ground_truth['answer']
            )
            results['answer_accuracy'] = MetricResult('answer_accuracy', accuracy)
            
        # Groundedness
        if 'answer' in prediction and 'source_texts' in prediction:
            grounded = self.groundedness_score(
                prediction['answer'],
                prediction['source_texts']
            )
            results['groundedness'] = MetricResult('groundedness', grounded)
            
        # Temporal precision
        if 'doc_dates' in prediction and 'analysis_date' in prediction:
            temporal = self.temporal_precision(
                prediction['doc_dates'],
                prediction['analysis_date']
            )
            results['temporal_precision'] = MetricResult('temporal_precision', temporal)
            
        # Temporal leakage
        if 'answer' in prediction and 'analysis_date' in prediction:
            leakage = self.temporal_leakage_score(
                prediction['answer'],
                prediction['analysis_date']
            )
            results['temporal_leakage'] = MetricResult('temporal_leakage', 1.0 - leakage)
            
        return results
        
    def summary_statistics(self, 
                           evaluation_results: List[Dict[str, MetricResult]]) -> Dict:
        """
        Calculate summary statistics across multiple evaluations.
        
        Args:
            evaluation_results: List of evaluation result dicts
            
        Returns:
            Summary statistics
        """
        if not evaluation_results:
            return {}
            
        # Aggregate scores by metric
        metric_scores = {}
        for result in evaluation_results:
            for metric_name, metric_result in result.items():
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                metric_scores[metric_name].append(metric_result.score)
                
        # Calculate statistics
        summary = {}
        for metric_name, scores in metric_scores.items():
            summary[metric_name] = {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'count': len(scores)
            }
            
        return summary


# Usage
if __name__ == "__main__":
    metrics = EvaluationMetrics()
    
    # Test retrieval metrics
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = ['doc1', 'doc3', 'doc5', 'doc7']
    
    retrieval_results = metrics.retrieval_metrics(retrieved, relevant, k=5)
    print("Retrieval Metrics:")
    for name, score in retrieval_results.items():
        print(f"  {name}: {score:.3f}")
        
    # Test answer accuracy
    accuracy = metrics.answer_accuracy(
        "$94.8 billion in revenue",
        "$95.1 billion"
    )
    print(f"\nAnswer Accuracy: {accuracy:.3f}")
    
    # Test temporal leakage
    leakage = metrics.temporal_leakage_score(
        "Apple's revenue in 2024 is expected to exceed 2023 levels",
        "20231001"
    )
    print(f"Temporal Leakage Score: {leakage:.3f}")
