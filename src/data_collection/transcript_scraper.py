"""
Earnings Transcript Scraper for TemporalGuard-RAG

Downloads and processes earnings call transcripts.
Supports multiple data sources including Kaggle and Financial Modeling Prep.
"""

import requests
import json
from pathlib import Path
from datetime import datetime
import time
import logging
from typing import Optional, List
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptScraper:
    """
    Earnings call transcript collector.
    
    Supports:
    - Financial Modeling Prep API
    - Manual loading from Kaggle datasets
    - Custom transcript sources
    """
    
    def __init__(self, output_dir: str = "data/raw/earnings_transcripts",
                 api_key: Optional[str] = None):
        """
        Initialize Transcript Scraper.
        
        Args:
            output_dir: Directory to save transcripts
            api_key: Financial Modeling Prep API key (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to get API key from environment
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        
        self.fmp_base_url = "https://financialmodelingprep.com/api/v3"
        
        logger.info(f"Initialized Transcript Scraper")
        logger.info(f"API key configured: {bool(self.api_key)}")
        
    def download_transcript_fmp(self, ticker: str, year: int, quarter: int) -> dict:
        """
        Download earnings transcript from Financial Modeling Prep.
        
        Args:
            ticker: Stock ticker symbol
            year: Fiscal year
            quarter: Fiscal quarter (1-4)
            
        Returns:
            Dictionary containing transcript data
        """
        if not self.api_key:
            logger.error("FMP API key not configured")
            return None
            
        try:
            url = f"{self.fmp_base_url}/earning_call_transcript/{ticker}"
            params = {
                "year": year,
                "quarter": quarter,
                "apikey": self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data:
                transcript_data = data[0] if isinstance(data, list) else data
                
                # Add metadata
                transcript_data['_metadata'] = {
                    'ticker': ticker,
                    'year': year,
                    'quarter': quarter,
                    'download_date': datetime.now().isoformat(),
                    'source': 'FMP'
                }
                
                # Save transcript
                filename = f"{ticker}_Q{quarter}_{year}.json"
                output_path = self.output_dir / filename
                
                with open(output_path, 'w') as f:
                    json.dump(transcript_data, f, indent=2)
                    
                logger.info(f"Downloaded transcript: {filename}")
                return transcript_data
                
            return None
            
        except Exception as e:
            logger.error(f"Error downloading transcript for {ticker} Q{quarter} {year}: {e}")
            return None
            
    def download_all_transcripts(self, tickers: list, years: List[int] = None) -> dict:
        """
        Download transcripts for multiple companies.
        
        Args:
            tickers: List of stock ticker symbols
            years: List of years to download (default: 2020-2024)
            
        Returns:
            Dictionary with download results
        """
        if years is None:
            current_year = datetime.now().year
            years = list(range(2020, current_year + 1))
            
        quarters = [1, 2, 3, 4]
        
        results = {}
        
        total_requests = len(tickers) * len(years) * len(quarters)
        current = 0
        
        logger.info(f"Downloading transcripts for {len(tickers)} companies, {len(years)} years")
        
        for ticker in tickers:
            results[ticker] = {"downloaded": [], "failed": []}
            
            for year in years:
                for quarter in quarters:
                    current += 1
                    
                    if current % 10 == 0:
                        logger.info(f"Progress: {current}/{total_requests}")
                        
                    transcript = self.download_transcript_fmp(ticker, year, quarter)
                    
                    if transcript:
                        results[ticker]["downloaded"].append(f"Q{quarter}_{year}")
                    else:
                        results[ticker]["failed"].append(f"Q{quarter}_{year}")
                        
                    # Rate limiting for free tier
                    time.sleep(0.5)
                    
        # Save summary
        summary_path = self.output_dir / "download_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'download_date': datetime.now().isoformat(),
                'tickers': tickers,
                'years': years,
                'results': results
            }, f, indent=2)
            
        return results
        
    def load_kaggle_transcripts(self, kaggle_dir: str) -> dict:
        """
        Load transcripts from a Kaggle dataset.
        
        Expects transcripts in JSON or TXT format.
        
        Args:
            kaggle_dir: Path to extracted Kaggle dataset
            
        Returns:
            Dictionary with loaded transcripts
        """
        kaggle_path = Path(kaggle_dir)
        
        if not kaggle_path.exists():
            logger.error(f"Kaggle directory not found: {kaggle_dir}")
            return {}
            
        results = {}
        file_count = 0
        
        # Look for JSON files
        for json_file in kaggle_path.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Try to extract ticker from filename or content
                ticker = self._extract_ticker(json_file.stem, data)
                
                if ticker:
                    # Copy to our output directory
                    output_path = self.output_dir / f"kaggle_{ticker}_{json_file.stem}.json"
                    
                    with open(output_path, 'w') as f:
                        json.dump(data, f, indent=2)
                        
                    results[str(json_file)] = {
                        'ticker': ticker,
                        'output_path': str(output_path),
                        'status': 'success'
                    }
                    file_count += 1
                    
            except Exception as e:
                results[str(json_file)] = {'status': 'error', 'error': str(e)}
                
        # Look for TXT files
        for txt_file in kaggle_path.rglob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                ticker = self._extract_ticker(txt_file.stem, {})
                
                if ticker:
                    output_path = self.output_dir / f"kaggle_{ticker}_{txt_file.stem}.json"
                    
                    with open(output_path, 'w') as f:
                        json.dump({
                            'content': content,
                            '_metadata': {
                                'source': 'kaggle',
                                'original_file': str(txt_file),
                                'ticker': ticker
                            }
                        }, f, indent=2)
                        
                    file_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing {txt_file}: {e}")
                
        logger.info(f"Loaded {file_count} transcripts from Kaggle")
        
        return results
        
    def _extract_ticker(self, filename: str, data: dict) -> Optional[str]:
        """Extract ticker symbol from filename or content."""
        # Common ticker patterns
        import re
        
        # Try to find ticker in filename
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        matches = re.findall(ticker_pattern, filename.upper())
        
        # Filter out common non-ticker words
        non_tickers = {'Q1', 'Q2', 'Q3', 'Q4', 'FY', 'CY', 'THE', 'AND', 'FOR', 'NOT'}
        valid_matches = [m for m in matches if m not in non_tickers and len(m) >= 2]
        
        if valid_matches:
            return valid_matches[0]
            
        # Try to get from data
        if isinstance(data, dict):
            return data.get('symbol') or data.get('ticker') or data.get('company_ticker')
            
        return None
        
    def parse_transcript(self, transcript_path: str) -> dict:
        """
        Parse a transcript into structured sections.
        
        Args:
            transcript_path: Path to transcript file
            
        Returns:
            Dictionary with parsed sections
        """
        with open(transcript_path, 'r') as f:
            data = json.load(f)
            
        content = data.get('content', '') or data.get('transcript', '')
        
        # Parse common sections
        sections = {
            'prepared_remarks': [],
            'qa_session': [],
            'speakers': set()
        }
        
        # Simple section detection
        lines = content.split('\n')
        current_section = 'prepared_remarks'
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
                
            # Detect Q&A section
            if any(keyword in line.lower() for keyword in ['question-and-answer', 'q&a', 'questions and answers']):
                current_section = 'qa_session'
                continue
                
            # Detect speaker
            if ':' in line and len(line.split(':')[0]) < 50:
                potential_speaker = line.split(':')[0].strip()
                sections['speakers'].add(potential_speaker)
                
            sections[current_section].append(line)
            
        sections['speakers'] = list(sections['speakers'])
        
        return sections
        
    def get_transcript(self, ticker: str, year: int, quarter: int) -> Optional[dict]:
        """
        Get a specific transcript.
        
        Args:
            ticker: Stock ticker symbol
            year: Fiscal year
            quarter: Fiscal quarter
            
        Returns:
            Transcript data if found
        """
        # Look for exact match first
        exact_path = self.output_dir / f"{ticker}_Q{quarter}_{year}.json"
        
        if exact_path.exists():
            with open(exact_path, 'r') as f:
                return json.load(f)
                
        # Look for kaggle version
        for path in self.output_dir.glob(f"*{ticker}*{year}*.json"):
            if f"Q{quarter}" in str(path) or str(quarter) in str(path):
                with open(path, 'r') as f:
                    return json.load(f)
                    
        return None


# Usage
if __name__ == "__main__":
    # Initialize scraper
    scraper = TranscriptScraper()
    
    # Check if API key is available
    if scraper.api_key:
        print("API key found, downloading transcripts...")
        
        # Demo download
        transcript = scraper.download_transcript_fmp("AAPL", 2023, 2)
        if transcript:
            print(f"Downloaded transcript for AAPL Q2 2023")
    else:
        print("No FMP API key found.")
        print("To use the FMP API, set the FMP_API_KEY environment variable.")
        print("\nAlternatively, download transcripts from Kaggle:")
        print("  https://www.kaggle.com/datasets/rpradyumna/earning-call-transcripts")
        print("\nThen use: scraper.load_kaggle_transcripts('path/to/kaggle/data')")
