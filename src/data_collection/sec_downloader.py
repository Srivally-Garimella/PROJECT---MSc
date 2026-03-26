"""
SEC EDGAR Data Downloader for TemporalGuard-RAG

Downloads 10-K and 10-Q filings from SEC EDGAR with temporal metadata
for look-ahead bias prevention testing.
"""

from sec_edgar_downloader import Downloader
import json
from pathlib import Path
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SECDataCollector:
    """
    SEC EDGAR filing downloader with temporal metadata tracking.
    
    This class downloads SEC filings while preserving temporal information
    critical for point-in-time analysis and look-ahead bias prevention.
    """
    
    def __init__(self, company_email: str = "your.email@example.com", 
                 output_dir: str = "data/raw/sec_filings"):
        """
        Initialize SEC Data Collector.
        
        Args:
            company_email: Email for SEC EDGAR User-Agent (required by SEC)
            output_dir: Directory to save downloaded filings
        """
        self.company_email = company_email
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SEC EDGAR downloader
        self.dl = Downloader("TemporalGuardRAG", company_email, str(self.output_dir))
        
        logger.info(f"Initialized SEC Data Collector with output dir: {self.output_dir}")
        
    def download_filings(self, tickers: list, filing_type: str = "10-K", 
                        years: int = 5, delay: float = 0.5) -> list:
        """
        Download SEC filings with temporal metadata.
        
        Args:
            tickers: List of stock ticker symbols
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
            years: Number of years of filings to download
            delay: Delay between requests (SEC rate limiting)
            
        Returns:
            List of metadata dictionaries for downloaded filings
        """
        metadata = []
        failed_downloads = []
        
        logger.info(f"Starting download of {filing_type} filings for {len(tickers)} companies")
        
        for i, ticker in enumerate(tickers):
            logger.info(f"[{i+1}/{len(tickers)}] Downloading {filing_type} for {ticker}...")
            
            try:
                # Download filings
                self.dl.get(filing_type, ticker, limit=years)
                
                # Record metadata with timestamps
                filing_metadata = {
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "download_date": datetime.now().isoformat(),
                    "download_timestamp": int(datetime.now().timestamp()),
                    "years_requested": years,
                    "status": "success"
                }
                metadata.append(filing_metadata)
                
                logger.info(f"Successfully downloaded {filing_type} for {ticker}")
                
                # Rate limiting - be nice to SEC servers
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error downloading {filing_type} for {ticker}: {e}")
                failed_downloads.append({
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
        # Save metadata
        metadata_path = self.output_dir / "download_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "download_summary": {
                    "total_tickers": len(tickers),
                    "successful": len(metadata),
                    "failed": len(failed_downloads),
                    "filing_type": filing_type,
                    "years": years,
                    "download_date": datetime.now().isoformat()
                },
                "successful_downloads": metadata,
                "failed_downloads": failed_downloads
            }, f, indent=2)
            
        logger.info(f"Download complete. Metadata saved to {metadata_path}")
        logger.info(f"Success: {len(metadata)}, Failed: {len(failed_downloads)}")
        
        return metadata
    
    def download_all_filings(self, tickers: list, years: int = 5) -> dict:
        """
        Download both 10-K and 10-Q filings for given tickers.
        
        Args:
            tickers: List of stock ticker symbols
            years: Number of years of filings to download
            
        Returns:
            Dictionary with metadata for all filing types
        """
        results = {}
        
        # Download annual reports (10-K)
        logger.info("Downloading 10-K (Annual Reports)...")
        results["10-K"] = self.download_filings(tickers, "10-K", years)
        
        # Download quarterly reports (10-Q)
        logger.info("Downloading 10-Q (Quarterly Reports)...")
        results["10-Q"] = self.download_filings(tickers, "10-Q", years * 4)
        
        return results
    
    def get_filing_paths(self, ticker: str = None, filing_type: str = None) -> list:
        """
        Get paths to downloaded filings.
        
        Args:
            ticker: Filter by specific ticker (optional)
            filing_type: Filter by filing type (optional)
            
        Returns:
            List of file paths
        """
        search_pattern = "**/*.html"
        
        paths = []
        for path in self.output_dir.rglob(search_pattern):
            # Apply filters if specified
            if ticker and ticker.upper() not in str(path).upper():
                continue
            if filing_type and filing_type not in str(path):
                continue
            paths.append(path)
            
        return sorted(paths)


def load_company_list(filepath: str = "data/company_list.json") -> list:
    """Load company tickers from JSON file."""
    with open(filepath, 'r') as f:
        companies = json.load(f)
    
    # Flatten all tickers from all sectors
    all_tickers = []
    for sector, tickers in companies.items():
        all_tickers.extend(tickers)
    
    return all_tickers


# Usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get email from environment or use default
    email = os.getenv("SEC_EMAIL", "your.email@example.com")
    
    # Initialize collector
    collector = SECDataCollector(company_email=email)
    
    # Create sample company list if it doesn't exist
    company_list_path = Path("data/company_list.json")
    if not company_list_path.exists():
        company_list_path.parent.mkdir(parents=True, exist_ok=True)
        sample_companies = {
            "tech": ["AAPL", "MSFT", "NVDA", "META", "GOOGL"],
            "finance": ["JPM", "GS", "BAC", "WFC"],
            "energy": ["XOM", "CVX"],
            "healthcare": ["JNJ", "UNH"],
            "consumer": ["AMZN", "WMT"]
        }
        with open(company_list_path, 'w') as f:
            json.dump(sample_companies, f, indent=2)
        print(f"Created sample company list at {company_list_path}")
    
    # Load companies and download
    try:
        all_tickers = load_company_list(str(company_list_path))
        print(f"Loaded {len(all_tickers)} tickers: {all_tickers}")
        
        # Download filings
        collector.download_all_filings(all_tickers, years=5)
        
    except FileNotFoundError:
        print("Company list not found. Please create data/company_list.json")
        print("Example format:")
        print(json.dumps({
            "tech": ["AAPL", "MSFT"],
            "finance": ["JPM", "GS"]
        }, indent=2))
