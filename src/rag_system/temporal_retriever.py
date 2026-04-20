"""
Temporal Retriever for TemporalGuard-RAG

Advanced retrieval system with temporal reasoning and point-in-time enforcement.
Prevents look-ahead bias through strict temporal constraints.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

from .vector_store import TemporalVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalRetriever:
    """
    Temporal-aware retrieval system with look-ahead bias prevention.
    
    Features:
    - Point-in-time (PiT) query enforcement
    - Filing date lag handling
    - Temporal consistency validation
    - Multi-period retrieval
    """
    
    # SEC filing deadlines by filer status (days after period end)
    # Reference: https://www.sec.gov/corpfin/form-10-k-10-q-filing-deadlines
    FILER_DEADLINES = {
        'Large Accelerated': {'10-K': 60, '10-Q': 40, '8-K': 4},
        'Accelerated': {'10-K': 75, '10-Q': 40, '8-K': 4},
        'Non-accelerated': {'10-K': 90, '10-Q': 45, '8-K': 4},
        'Default': {'10-K': 60, '10-Q': 45, '8-K': 4}
    }
    
    # Typical filing lag (conservative estimate)
    DEFAULT_FILING_LAG = 45
    
    def __init__(self, vector_store: TemporalVectorStore = None):
        """
        Initialize Temporal Retriever.
        
        Args:
            vector_store: TemporalVectorStore instance
        """
        if vector_store is None:
            vector_store = TemporalVectorStore()
            
        self.vector_store = vector_store
        self.metadata_dir = Path("data/raw/xbrl_structured")
        
        logger.info("Initialized Temporal Retriever with dynamic lag support")
        
    def _get_filer_status(self, ticker: str = None) -> str:
        """
        Determine filer status for a ticker.
        
        Looks for metadata in data/raw/xbrl_structured/ticker_metadata.json
        """
        if not ticker:
            return 'Default'
            
        ticker = ticker.upper()
        metadata_path = self.metadata_dir / "ticker_metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get(ticker, {}).get('filer_status', 'Large Accelerated')
            except Exception as e:
                logger.warning(f"Error reading metadata for {ticker}: {e}")
                
        # Fallback to Large Accelerated for most S&P 500 companies
        return 'Large Accelerated'
        
    def calculate_pit_cutoff(self, 
                            query_date: str,
                            filing_type: str = None) -> str:
        """
        Calculate point-in-time cutoff date for a query.
        
        The cutoff date is the latest filing date that would have been
        available at the query_date.
        
        Args:
            query_date: The as-of date for the query (YYYYMMDD or YYYY-MM-DD)
            filing_type: Optional filing type to adjust for specific deadlines
            
        Returns:
            Cutoff date string (YYYYMMDD format)
        """
        # Parse query date
        if '-' in query_date:
            query_dt = datetime.strptime(query_date, '%Y-%m-%d')
        else:
            query_dt = datetime.strptime(query_date, '%Y%m%d')
            
        # For PiT analysis, we can only use documents that were filed
        # ON or BEFORE the query date. The cutoff equals the query date.
        cutoff_dt = query_dt
        
        # Return in standard format
        return cutoff_dt.strftime('%Y%m%d')
        
    def calculate_available_period(self,
                                   query_date: str,
                                   filing_type: str = '10-Q') -> Dict:
        """
        Calculate what fiscal period information would be available at query date.
        
        Args:
            query_date: The as-of date for the query
            filing_type: Type of filing to consider
            
        Returns:
            Dictionary with available period information
        """
        if '-' in query_date:
            query_dt = datetime.strptime(query_date, '%Y-%m-%d')
        else:
            query_dt = datetime.strptime(query_date, '%Y%m%d')
            
        # Get filer status and appropriate lag
        from .vector_store import TemporalVectorStore
        ticker = getattr(self, '_current_ticker', None)
        status = self._get_filer_status(ticker)
        
        deadlines = self.FILER_DEADLINES.get(status, self.FILER_DEADLINES['Default'])
        filing_lag = deadlines.get(filing_type, self.DEFAULT_FILING_LAG)
        
        if ticker:
            logger.info(f"Applying {status} deadlines for {ticker}: {filing_type}={filing_lag}d")
        else:
            logger.debug(f"Applying default deadlines: {filing_type}={filing_lag}d")
        
        # Calculate the period end that would be available
        # A filing made on query_date would be for a period ending ~lag days before
        period_end = query_dt - timedelta(days=filing_lag)
        
        # Determine quarter
        quarter = (period_end.month - 1) // 3 + 1
        fiscal_year = period_end.year
        
        # If we're in Q1 and the period end is in Q4 of previous year
        if quarter == 4 and period_end.month <= 3:
            # This might be the 10-K filing
            pass
            
        return {
            'query_date': query_date,
            'estimated_period_end': period_end.strftime('%Y-%m-%d'),
            'estimated_fiscal_year': fiscal_year,
            'estimated_fiscal_quarter': f'Q{quarter}',
            'filing_type': filing_type,
            'filing_lag_days': filing_lag
        }
        
    def retrieve_pit(self,
                    query: str,
                    query_date: str,
                    ticker: str = None,
                    filing_type: str = None,
                    n_results: int = 5) -> Dict:
        """
        Retrieve documents using point-in-time constraints.
        
        This is the main retrieval method that enforces temporal integrity.
        
        Args:
            query: Search query
            query_date: The as-of date for the query (YYYYMMDD or YYYY-MM-DD)
            ticker: Optional company filter
            filing_type: Optional filing type filter
            n_results: Number of results to return
            
        Returns:
            Dictionary with results and temporal metadata
        """
        # Set current ticker for lag calculation
        self._current_ticker = ticker
        
        # Calculate PiT cutoff
        cutoff_date = self.calculate_pit_cutoff(query_date, filing_type)
        
        logger.info(f"PiT retrieval: query_date={query_date}, cutoff={cutoff_date}")
        
        # Execute temporal search
        results = self.vector_store.temporal_search(
            query=query,
            cutoff_date=cutoff_date,
            ticker=ticker,
            filing_type=filing_type,
            n_results=n_results
        )
        
        # Add temporal context to results
        results['temporal_context'] = {
            'query_date': query_date,
            'pit_cutoff': cutoff_date,
            'available_period': self.calculate_available_period(query_date, filing_type or '10-Q'),
            'temporal_enforcement': 'strict'
        }
        
        # Validate no look-ahead bias
        violations = self._check_temporal_violations(results, cutoff_date)
        results['temporal_violations'] = violations
        
        return results
        
    def retrieve_comparative(self,
                            query: str,
                            query_date: str,
                            ticker: str,
                            periods: List[str] = None,
                            n_results_per_period: int = 3) -> Dict:
        """
        Retrieve documents for comparative analysis across periods.
        
        Args:
            query: Search query
            query_date: As-of date for retrieval
            ticker: Company ticker
            periods: List of fiscal periods to compare (e.g., ['FY', 'Q1', 'Q2'])
            n_results_per_period: Results per period
            
        Returns:
            Dictionary with results organized by period
        """
        if periods is None:
            periods = ['FY']  # Default to annual reports
            
        cutoff_date = self.calculate_pit_cutoff(query_date)
        
        comparative_results = {
            'query': query,
            'ticker': ticker,
            'query_date': query_date,
            'pit_cutoff': cutoff_date,
            'periods': {}
        }
        
        for period in periods:
            results = self.vector_store.temporal_search(
                query=query,
                cutoff_date=cutoff_date,
                ticker=ticker,
                n_results=n_results_per_period
            )
            
            # Filter by period (post-processing)
            period_results = {
                'documents': [],
                'metadatas': []
            }
            
            for i, meta in enumerate(results.get('metadatas', [[]])[0]):
                if meta.get('fiscal_period') == period:
                    period_results['documents'].append(results['documents'][0][i])
                    period_results['metadatas'].append(meta)
                    
            comparative_results['periods'][period] = period_results
            
        return comparative_results
        
    def _check_temporal_violations(self, results: Dict, cutoff_date: str) -> List[Dict]:
        """
        Check for any temporal violations in search results.
        
        Args:
            results: Search results dictionary
            cutoff_date: The cutoff date that should have been enforced
            
        Returns:
            List of violation records
        """
        violations = []
        
        for i, meta in enumerate(results.get('metadatas', [[]])[0]):
            filing_date = meta.get('filing_date', '')
            
            if filing_date and filing_date > cutoff_date:
                violations.append({
                    'index': i,
                    'filing_date': filing_date,
                    'cutoff_date': cutoff_date,
                    'ticker': meta.get('ticker'),
                    'violation_type': 'look_ahead_bias'
                })
                
        if violations:
            logger.warning(f"⚠️ Temporal violations detected: {len(violations)}")
            
        return violations
        
    def get_latest_filing_before(self,
                                 ticker: str,
                                 cutoff_date: str,
                                 filing_type: str = '10-K') -> Optional[Dict]:
        """
        Get the most recent filing for a company before a cutoff date.
        
        Args:
            ticker: Company ticker
            cutoff_date: Maximum filing date (YYYYMMDD)
            filing_type: Type of filing to retrieve
            
        Returns:
            Most recent filing or None
        """
        results = self.vector_store.temporal_search(
            query="company overview financial performance",
            cutoff_date=cutoff_date,
            ticker=ticker,
            filing_type=filing_type,
            n_results=10
        )
        
        if not results['documents'][0]:
            return None
            
        # Find the one with latest filing date (still before cutoff)
        latest_idx = 0
        latest_date = ''
        
        for i, meta in enumerate(results['metadatas'][0]):
            if meta.get('filing_date', '') > latest_date:
                latest_date = meta['filing_date']
                latest_idx = i
                
        return {
            'document': results['documents'][0][latest_idx],
            'metadata': results['metadatas'][0][latest_idx],
            'filing_date': latest_date
        }
        
    def retrieve_for_question(self,
                             question: str,
                             query_date: str,
                             context: Dict = None) -> Dict:
        """
        Intelligent retrieval based on question analysis.
        
        Analyzes the question to determine optimal retrieval strategy.
        
        Args:
            question: Natural language question
            query_date: As-of date
            context: Optional context with ticker, period hints
            
        Returns:
            Retrieval results with strategy explanation
        """
        # Extract hints from question
        ticker = context.get('ticker') if context else None
        
        # Detect question type
        question_lower = question.lower()
        
        # Detect if asking about specific metric
        metric_keywords = ['revenue', 'profit', 'income', 'assets', 'liabilities', 
                          'margin', 'growth', 'eps', 'earnings']
        is_metric_question = any(kw in question_lower for kw in metric_keywords)
        
        # Detect if asking about risks
        is_risk_question = any(kw in question_lower for kw in ['risk', 'challenge', 'threat'])
        
        # Detect comparative question
        is_comparative = any(kw in question_lower for kw in ['compare', 'versus', 'vs', 'change'])
        
        # Select retrieval strategy
        strategy = {
            'type': 'standard',
            'filing_type': None,
            'n_results': 5
        }
        
        if is_metric_question:
            strategy['type'] = 'metric_focused'
            strategy['n_results'] = 10  # More results for verification
            
        if is_risk_question:
            strategy['type'] = 'risk_focused'
            strategy['filing_type'] = '10-K'  # Annual reports have detailed risk factors
            
        if is_comparative:
            strategy['type'] = 'comparative'
            strategy['n_results'] = 15
            
        # Execute retrieval
        results = self.retrieve_pit(
            query=question,
            query_date=query_date,
            ticker=ticker,
            filing_type=strategy['filing_type'],
            n_results=strategy['n_results']
        )
        
        results['retrieval_strategy'] = strategy
        
        return results


# Usage
if __name__ == "__main__":
    # Initialize retriever
    retriever = TemporalRetriever()
    
    # Test PiT calculation
    print("Testing Point-in-Time calculations...")
    
    query_date = "2023-06-30"
    cutoff = retriever.calculate_pit_cutoff(query_date)
    print(f"Query date: {query_date} -> PiT cutoff: {cutoff}")
    
    available = retriever.calculate_available_period(query_date)
    print(f"Available period info: {json.dumps(available, indent=2)}")
    
    # Test retrieval if data exists
    stats = retriever.vector_store.get_collection_stats()
    
    if stats['total_documents'] > 0:
        print(f"\nTesting retrieval with {stats['total_documents']} documents...")
        
        results = retriever.retrieve_pit(
            query="What are the major business risks?",
            query_date="2023-06-30",
            ticker="AAPL",
            n_results=3
        )
        
        print(f"\nResults found: {results['search_metadata']['n_results_returned']}")
        print(f"Temporal violations: {len(results['temporal_violations'])}")
    else:
        print("\nNo documents in vector store. Run preprocessing pipeline first.")
