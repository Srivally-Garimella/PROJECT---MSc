"""
XBRL Data Parser for TemporalGuard-RAG

Extracts structured financial data from SEC XBRL filings.
This provides verified numerical data for the Calculation Agent.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XBRLCollector:
    """
    SEC XBRL structured data collector.
    
    Downloads and parses XBRL financial facts from SEC's API,
    providing exact numerical data for verification and calculations.
    """
    
    def __init__(self, output_dir: str = "data/raw/xbrl_structured",
                 user_agent: str = "TemporalGuardRAG your.email@example.com"):
        """
        Initialize XBRL Collector.
        
        Args:
            output_dir: Directory to save XBRL data
            user_agent: User-Agent string for SEC requests
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        self.company_facts_url = "https://data.sec.gov/api/xbrl/companyfacts"
        self.submissions_url = "https://data.sec.gov/submissions"
        
        self.headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
        }
        
        # Headers specifically for data.sec.gov
        self.data_headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        
        # Headers specifically for www.sec.gov
        self.www_headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        
        # CIK cache to avoid repeated lookups
        self.cik_cache = {}
        
        logger.info(f"Initialized XBRL Collector with output dir: {self.output_dir}")
        
    def get_company_cik(self, ticker: str) -> str:
        """
        Get CIK (Central Index Key) number for a company.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CIK number as string (zero-padded to 10 digits)
        """
        if ticker in self.cik_cache:
            return self.cik_cache[ticker]
            
        try:
            # Use SEC's ticker to CIK mapping
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(tickers_url, headers=self.www_headers)
            response.raise_for_status()
            
            tickers_data = response.json()
            
            # Search for ticker
            for entry in tickers_data.values():
                if entry.get('ticker', '').upper() == ticker.upper():
                    cik = str(entry['cik_str']).zfill(10)
                    self.cik_cache[ticker] = cik
                    return cik
                    
            logger.warning(f"CIK not found for ticker: {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {e}")
            return None
            
    def download_xbrl_facts(self, ticker: str, cik: str = None) -> dict:
        """
        Download structured financial data from SEC's companyfacts API.
        
        Args:
            ticker: Stock ticker symbol
            cik: CIK number (optional, will be looked up if not provided)
            
        Returns:
            Dictionary containing XBRL financial facts
        """
        if cik is None:
            cik = self.get_company_cik(ticker)
            
        if cik is None:
            logger.error(f"Cannot download XBRL facts without CIK for {ticker}")
            return None
            
        try:
            # SEC companyfacts API endpoint
            url = f"{self.company_facts_url}/CIK{cik}.json"
            
            time.sleep(0.2)  # Rate limiting
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Add metadata
            data['_metadata'] = {
                'ticker': ticker,
                'cik': cik,
                'download_date': datetime.now().isoformat(),
                'source_url': url
            }
            
            # Save raw data
            output_path = self.output_dir / f"{ticker}_facts.json"
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Downloaded XBRL facts for {ticker} to {output_path}")
            
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"No XBRL data available for {ticker}")
            else:
                logger.error(f"HTTP error downloading XBRL for {ticker}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Error downloading XBRL for {ticker}: {e}")
            return None
            
    def extract_key_metrics(self, xbrl_data: dict, ticker: str = None) -> pd.DataFrame:
        """
        Extract key financial metrics from XBRL data.
        
        Args:
            xbrl_data: Raw XBRL data dictionary
            ticker: Ticker symbol for reference
            
        Returns:
            DataFrame with key metrics over time
        """
        if xbrl_data is None:
            return pd.DataFrame()
            
        # Common XBRL concepts for financial metrics
        metric_mapping = {
            # Revenue
            'Revenues': 'Revenue',
            'RevenueFromContractWithCustomerExcludingAssessedTax': 'Revenue',
            'SalesRevenueNet': 'Revenue',
            
            # Assets
            'Assets': 'TotalAssets',
            'AssetsCurrent': 'CurrentAssets',
            
            # Liabilities
            'Liabilities': 'TotalLiabilities',
            'LiabilitiesCurrent': 'CurrentLiabilities',
            
            # Equity
            'StockholdersEquity': 'Equity',
            'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest': 'Equity',
            
            # Net Income
            'NetIncomeLoss': 'NetIncome',
            'ProfitLoss': 'NetIncome',
            
            # Operating Income
            'OperatingIncomeLoss': 'OperatingIncome',
            
            # EPS
            'EarningsPerShareBasic': 'EPS_Basic',
            'EarningsPerShareDiluted': 'EPS_Diluted'
        }
        
        records = []
        
        try:
            facts = xbrl_data.get('facts', {})
            us_gaap = facts.get('us-gaap', {})
            
            for xbrl_concept, metric_name in metric_mapping.items():
                if xbrl_concept in us_gaap:
                    concept_data = us_gaap[xbrl_concept]
                    units = concept_data.get('units', {})
                    
                    # Handle USD values
                    usd_data = units.get('USD', [])
                    for entry in usd_data:
                        record = {
                            'ticker': ticker or xbrl_data.get('_metadata', {}).get('ticker'),
                            'metric': metric_name,
                            'value': entry.get('val'),
                            'end_date': entry.get('end'),
                            'start_date': entry.get('start'),
                            'fiscal_year': entry.get('fy'),
                            'fiscal_period': entry.get('fp'),
                            'form': entry.get('form'),
                            'filed_date': entry.get('filed'),
                            'xbrl_concept': xbrl_concept
                        }
                        records.append(record)
                        
                    # Handle per-share values
                    share_data = units.get('USD/shares', [])
                    for entry in share_data:
                        record = {
                            'ticker': ticker,
                            'metric': metric_name,
                            'value': entry.get('val'),
                            'end_date': entry.get('end'),
                            'start_date': entry.get('start'),
                            'fiscal_year': entry.get('fy'),
                            'fiscal_period': entry.get('fp'),
                            'form': entry.get('form'),
                            'filed_date': entry.get('filed'),
                            'xbrl_concept': xbrl_concept
                        }
                        records.append(record)
                        
        except Exception as e:
            logger.error(f"Error extracting metrics: {e}")
            
        df = pd.DataFrame(records)
        
        if not df.empty:
            # Convert dates
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
            df['filed_date'] = pd.to_datetime(df['filed_date'], errors='coerce')
            
            # Sort by date
            df = df.sort_values(['metric', 'end_date'], ascending=[True, False])
            
        return df
        
    def download_all_companies(self, tickers: list) -> dict:
        """
        Download XBRL data for multiple companies.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping tickers to their XBRL data
        """
        results = {}
        
        logger.info(f"Downloading XBRL data for {len(tickers)} companies...")
        
        for i, ticker in enumerate(tickers):
            logger.info(f"[{i+1}/{len(tickers)}] Processing {ticker}...")
            
            data = self.download_xbrl_facts(ticker)
            
            if data:
                results[ticker] = {
                    'raw_data_path': str(self.output_dir / f"{ticker}_facts.json"),
                    'status': 'success'
                }
                
                # Also extract and save key metrics
                metrics_df = self.extract_key_metrics(data, ticker)
                if not metrics_df.empty:
                    metrics_path = self.output_dir / f"{ticker}_metrics.csv"
                    metrics_df.to_csv(metrics_path, index=False)
                    results[ticker]['metrics_path'] = str(metrics_path)
            else:
                results[ticker] = {'status': 'failed'}
                
            time.sleep(0.5)  # Rate limiting
            
        # Save summary
        summary_path = self.output_dir / "xbrl_download_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'download_date': datetime.now().isoformat(),
                'total_companies': len(tickers),
                'successful': sum(1 for r in results.values() if r['status'] == 'success'),
                'results': results
            }, f, indent=2)
            
        logger.info(f"XBRL download complete. Summary saved to {summary_path}")
        
        return results
        
    def get_metric_at_date(self, ticker: str, metric: str, as_of_date: str) -> dict:
        """
        Get a specific metric value as of a given date (point-in-time).
        
        Args:
            ticker: Stock ticker symbol
            metric: Metric name (e.g., 'Revenue', 'NetIncome')
            as_of_date: Date string (YYYY-MM-DD format)
            
        Returns:
            Dictionary with metric value and metadata
        """
        metrics_path = self.output_dir / f"{ticker}_metrics.csv"
        
        if not metrics_path.exists():
            logger.error(f"No metrics data found for {ticker}")
            return None
            
        df = pd.read_csv(metrics_path)
        df['filed_date'] = pd.to_datetime(df['filed_date'])
        
        as_of_datetime = pd.to_datetime(as_of_date)
        
        # Filter to metric and filed before as_of_date (point-in-time)
        filtered = df[
            (df['metric'] == metric) & 
            (df['filed_date'] <= as_of_datetime)
        ]
        
        if filtered.empty:
            logger.warning(f"No {metric} data found for {ticker} before {as_of_date}")
            return None
            
        # Get most recent filing before as_of_date
        latest = filtered.sort_values('filed_date', ascending=False).iloc[0]
        
        return {
            'ticker': ticker,
            'metric': metric,
            'value': latest['value'],
            'end_date': str(latest['end_date']),
            'filed_date': str(latest['filed_date']),
            'as_of_date': as_of_date,
            'fiscal_year': latest['fiscal_year'],
            'fiscal_period': latest['fiscal_period']
        }


# Usage
if __name__ == "__main__":
    collector = XBRLCollector()
    
    # Load company list
    company_list_path = Path("data/company_list.json")
    
    if company_list_path.exists():
        with open(company_list_path) as f:
            companies = json.load(f)
        all_tickers = [t for sector in companies.values() for t in sector]
        
        # Download XBRL data
        collector.download_all_companies(all_tickers)
    else:
        # Demo with single company
        print("Company list not found. Running demo with AAPL...")
        data = collector.download_xbrl_facts("AAPL")
        
        if data:
            metrics = collector.extract_key_metrics(data, "AAPL")
            print("\nExtracted Metrics:")
            print(metrics.head(20))
