"""
Stock Price Data Collector for TemporalGuard-RAG

Downloads historical stock prices from Yahoo Finance for temporal validation
and look-ahead bias testing.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """
    Stock price data collector using Yahoo Finance.
    
    Provides historical price data for:
    - Temporal validation
    - Look-ahead bias detection
    - Performance correlation analysis
    """
    
    def __init__(self, output_dir: str = "data/raw/stock_prices",
                 start_date: str = "2018-01-01"):
        """
        Initialize Stock Data Collector.
        
        Args:
            output_dir: Directory to save stock price data
            start_date: Start date for historical data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_date = start_date
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Initialized Stock Data Collector")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        
    def download_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        Download historical stock data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Downloading stock data for {ticker}...")
            
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(start=self.start_date, end=self.end_date)
            
            if hist.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
                
            # Add ticker column
            hist['Ticker'] = ticker
            
            # Reset index to make Date a column
            hist = hist.reset_index()
            
            # Save to CSV
            output_path = self.output_dir / f"{ticker}.csv"
            hist.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(hist)} records for {ticker}")
            
            return hist
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            return pd.DataFrame()
            
    def download_multiple_stocks(self, tickers: list) -> dict:
        """
        Download historical stock data for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary with download metadata
        """
        all_data = {}
        
        logger.info(f"Downloading stock data for {len(tickers)} companies...")
        
        for i, ticker in enumerate(tickers):
            logger.info(f"[{i+1}/{len(tickers)}] Processing {ticker}...")
            
            df = self.download_stock_data(ticker)
            
            if not df.empty:
                all_data[ticker] = {
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "records": len(df),
                    "path": str(self.output_dir / f"{ticker}.csv"),
                    "first_date": str(df['Date'].min()),
                    "last_date": str(df['Date'].max()),
                    "status": "success"
                }
            else:
                all_data[ticker] = {"status": "failed"}
                
        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "download_date": datetime.now().isoformat(),
                "total_tickers": len(tickers),
                "successful": sum(1 for d in all_data.values() if d.get("status") == "success"),
                "date_range": {
                    "start": self.start_date,
                    "end": self.end_date
                },
                "tickers": all_data
            }, f, indent=2)
            
        logger.info(f"Stock data download complete. Metadata saved to {metadata_path}")
            
        return all_data
        
    def get_price_at_date(self, ticker: str, target_date: str) -> dict:
        """
        Get stock price at a specific date.
        
        Args:
            ticker: Stock ticker symbol
            target_date: Date string (YYYY-MM-DD format)
            
        Returns:
            Dictionary with price data
        """
        data_path = self.output_dir / f"{ticker}.csv"
        
        if not data_path.exists():
            logger.error(f"No data found for {ticker}")
            return None
            
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        target_datetime = pd.to_datetime(target_date)
        
        # Find closest trading day on or before target date
        before_data = df[df['Date'] <= target_datetime]
        
        if before_data.empty:
            logger.warning(f"No data found for {ticker} before {target_date}")
            return None
            
        closest = before_data.iloc[-1]
        
        return {
            'ticker': ticker,
            'requested_date': target_date,
            'actual_date': str(closest['Date'].date()),
            'open': closest['Open'],
            'high': closest['High'],
            'low': closest['Low'],
            'close': closest['Close'],
            'volume': closest['Volume']
        }
        
    def get_returns_after_filing(self, ticker: str, filing_date: str, 
                                  days: list = [1, 5, 20, 60]) -> dict:
        """
        Calculate stock returns after a filing date.
        
        Useful for analyzing market reaction to filings.
        
        Args:
            ticker: Stock ticker symbol
            filing_date: Date of SEC filing
            days: List of day periods for return calculation
            
        Returns:
            Dictionary with return data
        """
        data_path = self.output_dir / f"{ticker}.csv"
        
        if not data_path.exists():
            return None
            
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        filing_datetime = pd.to_datetime(filing_date)
        
        # Get price at filing
        filing_data = df[df['Date'] >= filing_datetime]
        if filing_data.empty:
            return None
            
        filing_price = filing_data.iloc[0]['Close']
        filing_actual_date = filing_data.iloc[0]['Date']
        
        returns = {
            'ticker': ticker,
            'filing_date': filing_date,
            'actual_start_date': str(filing_actual_date.date()),
            'start_price': filing_price,
            'returns': {}
        }
        
        for day_count in days:
            future_date = filing_actual_date + timedelta(days=day_count)
            future_data = df[df['Date'] <= future_date]
            
            if not future_data.empty:
                end_price = future_data.iloc[-1]['Close']
                pct_return = ((end_price - filing_price) / filing_price) * 100
                
                returns['returns'][f'{day_count}d'] = {
                    'end_date': str(future_data.iloc[-1]['Date'].date()),
                    'end_price': end_price,
                    'return_pct': round(pct_return, 2)
                }
                
        return returns
        
    def calculate_volatility(self, ticker: str, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling volatility for a stock.
        
        Args:
            ticker: Stock ticker symbol
            window: Rolling window size in days
            
        Returns:
            DataFrame with volatility data
        """
        data_path = self.output_dir / f"{ticker}.csv"
        
        if not data_path.exists():
            return pd.DataFrame()
            
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Calculate rolling volatility (annualized)
        df['Volatility'] = df['Daily_Return'].rolling(window=window).std() * (252 ** 0.5)
        
        return df[['Date', 'Ticker', 'Close', 'Daily_Return', 'Volatility']]


# Usage
if __name__ == "__main__":
    collector = StockDataCollector()
    
    # Load company list
    company_list_path = Path("data/company_list.json")
    
    if company_list_path.exists():
        with open(company_list_path) as f:
            companies = json.load(f)
        all_tickers = [t for sector in companies.values() for t in sector]
        
        # Download stock data
        collector.download_multiple_stocks(all_tickers)
    else:
        # Demo with sample tickers
        print("Company list not found. Running demo with sample tickers...")
        demo_tickers = ["AAPL", "MSFT", "GOOGL"]
        collector.download_multiple_stocks(demo_tickers)
        
        # Test point-in-time lookup
        price = collector.get_price_at_date("AAPL", "2023-06-30")
        print(f"\nAAPL price at 2023-06-30: {price}")
        
        # Test returns calculation
        returns = collector.get_returns_after_filing("AAPL", "2023-01-27")
        print(f"\nReturns after filing: {returns}")
