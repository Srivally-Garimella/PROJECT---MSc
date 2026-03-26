"""
Adversarial Filter for TemporalGuard-RAG

Detects and filters potentially poisoned or adversarial documents.
Implements multiple security checks for RAG system integrity.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdversarialFilter:
    """
    Adversarial attack detection and filtering for RAG systems.
    
    Detects:
    - Embedding poisoning attacks
    - Outlier/anomalous documents
    - Potential data injection
    - Suspicious retrieval patterns
    """
    
    def __init__(self, 
                 embeddings_path: str = "data/processed/embeddings/embeddings.npy",
                 contamination: float = 0.01):
        """
        Initialize Adversarial Filter.
        
        Args:
            embeddings_path: Path to embeddings file
            contamination: Expected proportion of poisoned embeddings (0.01 = 1%)
        """
        self.embeddings_path = Path(embeddings_path)
        self.contamination = contamination
        
        self.embeddings = None
        self.detector = None
        self.pca = None
        self.anomaly_scores = None
        self.suspicious_indices = set()
        
        if self.embeddings_path.exists():
            self._load_and_train()
        else:
            logger.warning(f"Embeddings not found at {embeddings_path}")
            
    def _load_and_train(self):
        """Load embeddings and train anomaly detector."""
        logger.info(f"Loading embeddings from {self.embeddings_path}...")
        self.embeddings = np.load(self.embeddings_path)
        
        logger.info(f"Training anomaly detector on {len(self.embeddings)} embeddings...")
        
        # Train Isolation Forest for anomaly detection
        self.detector = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.detector.fit(self.embeddings)
        
        # Get anomaly scores
        self.anomaly_scores = self.detector.score_samples(self.embeddings)
        
        # Identify suspicious embeddings
        predictions = self.detector.predict(self.embeddings)
        self.suspicious_indices = set(np.where(predictions == -1)[0])
        
        # PCA for visualization and additional analysis
        n_components = min(50, self.embeddings.shape[1])
        self.pca = PCA(n_components=n_components)
        self.reduced_embeddings = self.pca.fit_transform(self.embeddings)
        
        logger.info(f"Detected {len(self.suspicious_indices)} suspicious embeddings ({len(self.suspicious_indices)/len(self.embeddings)*100:.2f}%)")
        
    def detect_poisoned_embeddings(self, threshold: float = -0.5) -> Tuple[List[int], np.ndarray]:
        """
        Detect potentially poisoned embeddings.
        
        Args:
            threshold: Anomaly score threshold (lower = more suspicious)
            
        Returns:
            Tuple of (suspicious indices list, anomaly scores array)
        """
        if self.anomaly_scores is None:
            logger.error("Anomaly detector not trained")
            return [], np.array([])
            
        suspicious = np.where(self.anomaly_scores < threshold)[0]
        
        logger.info(f"Detected {len(suspicious)} embeddings below threshold {threshold}")
        
        return suspicious.tolist(), self.anomaly_scores
        
    def analyze_embedding_clusters(self, 
                                   eps: float = 3.0,
                                   min_samples: int = 10) -> Dict:
        """
        Analyze embedding clusters to identify outliers.
        
        Args:
            eps: DBSCAN neighborhood radius
            min_samples: Minimum samples for core point
            
        Returns:
            Dictionary with cluster analysis results
        """
        if self.reduced_embeddings is None:
            logger.error("Embeddings not loaded")
            return {}
            
        # Cluster using DBSCAN
        logger.info("Performing cluster analysis...")
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(self.reduced_embeddings)
        
        # Find outliers (label = -1 in DBSCAN)
        outliers = np.where(labels == -1)[0]
        
        # Analyze cluster distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels.astype(int).tolist(), counts.tolist()))
        
        results = {
            'total_embeddings': len(self.embeddings),
            'n_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0),
            'n_outliers': len(outliers),
            'outlier_percentage': len(outliers) / len(self.embeddings) * 100,
            'cluster_sizes': cluster_sizes,
            'outlier_indices': outliers.tolist()[:100]  # First 100
        }
        
        logger.info(f"Cluster analysis: {results['n_clusters']} clusters, {results['n_outliers']} outliers")
        
        return results
        
    def is_suspicious(self, index: int) -> bool:
        """
        Check if a specific embedding index is suspicious.
        
        Args:
            index: Embedding index
            
        Returns:
            True if suspicious
        """
        return index in self.suspicious_indices
        
    def filter_suspicious_results(self, 
                                  retrieved_indices: List[int],
                                  retrieved_scores: List[float] = None) -> Dict:
        """
        Filter suspicious documents from retrieval results.
        
        Args:
            retrieved_indices: Indices of retrieved documents
            retrieved_scores: Optional retrieval scores
            
        Returns:
            Dictionary with filtered results
        """
        clean_indices = []
        suspicious = []
        
        for i, idx in enumerate(retrieved_indices):
            if idx in self.suspicious_indices:
                suspicious.append({
                    'index': idx,
                    'position': i,
                    'anomaly_score': float(self.anomaly_scores[idx]) if self.anomaly_scores is not None else None,
                    'retrieval_score': retrieved_scores[i] if retrieved_scores else None
                })
            else:
                clean_indices.append(idx)
                
        return {
            'clean_indices': clean_indices,
            'suspicious_removed': suspicious,
            'n_removed': len(suspicious),
            'n_remaining': len(clean_indices),
            'warning': len(suspicious) > 0
        }
        
    def verify_retrieval_integrity(self,
                                   query_embedding: np.ndarray,
                                   retrieved_embeddings: np.ndarray,
                                   similarity_threshold: float = 0.3) -> Dict:
        """
        Verify integrity of retrieval results.
        
        Checks:
        - Similarity scores are reasonable
        - No obviously injected documents
        - Embedding distribution is normal
        
        Args:
            query_embedding: Query embedding vector
            retrieved_embeddings: Retrieved document embeddings
            similarity_threshold: Minimum expected similarity
            
        Returns:
            Integrity check results
        """
        # Calculate similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        similarities = []
        for emb in retrieved_embeddings:
            emb_norm = emb / np.linalg.norm(emb)
            sim = np.dot(query_norm, emb_norm)
            similarities.append(float(sim))
            
        # Check for suspiciously high similarities (potential injection)
        very_high_sim = [s for s in similarities if s > 0.95]
        
        # Check for suspiciously low similarities (noise)
        very_low_sim = [s for s in similarities if s < similarity_threshold]
        
        # Statistical analysis
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        integrity_issues = []
        
        if len(very_high_sim) > 0:
            integrity_issues.append({
                'type': 'potential_injection',
                'description': f'{len(very_high_sim)} results with suspiciously high similarity (>0.95)',
                'severity': 'high'
            })
            
        if len(very_low_sim) > len(similarities) // 2:
            integrity_issues.append({
                'type': 'poor_retrieval',
                'description': f'{len(very_low_sim)} results below similarity threshold',
                'severity': 'medium'
            })
            
        return {
            'similarities': similarities,
            'mean_similarity': float(mean_sim),
            'std_similarity': float(std_sim),
            'integrity_valid': len(integrity_issues) == 0,
            'issues': integrity_issues
        }
        
    def generate_security_report(self, 
                                output_path: str = "results/security_report.json") -> Dict:
        """
        Generate comprehensive security report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Security report dictionary
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'embeddings_analyzed': len(self.embeddings) if self.embeddings is not None else 0,
            'contamination_threshold': self.contamination,
            'anomaly_detection': {},
            'cluster_analysis': {},
            'recommendations': []
        }
        
        # Anomaly detection results
        if self.anomaly_scores is not None:
            report['anomaly_detection'] = {
                'suspicious_count': len(self.suspicious_indices),
                'suspicious_percentage': len(self.suspicious_indices) / len(self.embeddings) * 100,
                'mean_anomaly_score': float(np.mean(self.anomaly_scores)),
                'std_anomaly_score': float(np.std(self.anomaly_scores)),
                'min_anomaly_score': float(np.min(self.anomaly_scores)),
                'suspicious_indices_sample': list(self.suspicious_indices)[:20]
            }
            
        # Cluster analysis
        cluster_results = self.analyze_embedding_clusters()
        report['cluster_analysis'] = cluster_results
        
        # Generate recommendations
        if report['anomaly_detection'].get('suspicious_percentage', 0) > 5:
            report['recommendations'].append({
                'priority': 'high',
                'recommendation': 'High proportion of suspicious embeddings detected. Review data sources.'
            })
            
        if cluster_results.get('n_outliers', 0) > len(self.embeddings) * 0.1:
            report['recommendations'].append({
                'priority': 'medium',
                'recommendation': 'Significant cluster outliers found. Consider data quality review.'
            })
            
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Security report saved to {output_path}")
        
        return report
        
    def update_detector(self, new_embeddings: np.ndarray):
        """
        Update detector with new embeddings.
        
        Args:
            new_embeddings: New embeddings to incorporate
        """
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        # Retrain detector
        self._load_and_train()


# Usage
if __name__ == "__main__":
    # Initialize filter
    adversarial_filter = AdversarialFilter()
    
    if adversarial_filter.embeddings is not None:
        # Generate security report
        report = adversarial_filter.generate_security_report()
        
        print("\nSecurity Report Summary:")
        print(f"  Embeddings analyzed: {report['embeddings_analyzed']}")
        print(f"  Suspicious embeddings: {report['anomaly_detection'].get('suspicious_count', 0)}")
        print(f"  Cluster outliers: {report['cluster_analysis'].get('n_outliers', 0)}")
        print(f"  Recommendations: {len(report['recommendations'])}")
    else:
        print("Embeddings not found. Run preprocessing pipeline first.")
