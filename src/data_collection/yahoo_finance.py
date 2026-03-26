"""
Yahoo Finance Data Collector

Fetches financial statements for ANY publicly traded company globally:
- Balance Sheet
- Income Statement (P&L)
- Cash Flow Statement
- Key Metrics & Ratios

Supports: NYSE, NASDAQ, LSE, TSE, HKEx, NSE, BSE, and 50+ global exchanges
"""

import yfinance as yf
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "yahoo_finance"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class YahooFinanceCollector:
    """
    Collect financial statements from Yahoo Finance.
    Works for global stocks - just use the right ticker format:
    
    Examples:
        US: AAPL, MSFT, GOOGL
        UK: HSBA.L, BP.L, VOD.L
        India: RELIANCE.NS, TCS.NS, INFY.NS (NSE) or .BO for BSE
        Japan: 7203.T (Toyota), 6758.T (Sony)
        Hong Kong: 0700.HK (Tencent), 9988.HK (Alibaba)
        Germany: SAP.DE, BMW.DE
        France: MC.PA (LVMH), OR.PA (L'Oreal)
    """
    
    def __init__(self, output_dir: Path = DATA_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_company_data(self, ticker: str) -> Dict:
        """
        Fetch all available financial data for a company.
        
        Args:
            ticker: Stock ticker (e.g., "AAPL", "RELIANCE.NS", "7203.T")
            
        Returns:
            Dict with company info and financial statements
        """
        logger.info(f"Fetching data for {ticker}")
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or info.get('regularMarketPrice') is None:
                logger.warning(f"No data found for {ticker}")
                return None
            
            data = {
                "ticker": ticker,
                "fetched_at": datetime.now().isoformat(),
                "company_info": self._extract_company_info(info),
                "financials": {
                    "income_statement": self._df_to_dict(stock.income_stmt),
                    "balance_sheet": self._df_to_dict(stock.balance_sheet),
                    "cash_flow": self._df_to_dict(stock.cashflow),
                    "quarterly_income": self._df_to_dict(stock.quarterly_income_stmt),
                    "quarterly_balance": self._df_to_dict(stock.quarterly_balance_sheet),
                    "quarterly_cashflow": self._df_to_dict(stock.quarterly_cashflow),
                },
                "metrics": self._extract_metrics(info),
                "historical_prices": self._get_price_history(stock),
            }
            
            # Save to file
            self._save_data(ticker, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None
    
    def _extract_company_info(self, info: Dict) -> Dict:
        """Extract basic company information."""
        return {
            "name": info.get("longName", info.get("shortName", "")),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "country": info.get("country", ""),
            "exchange": info.get("exchange", ""),
            "currency": info.get("currency", ""),
            "market_cap": info.get("marketCap"),
            "employees": info.get("fullTimeEmployees"),
            "website": info.get("website", ""),
            "description": info.get("longBusinessSummary", ""),
        }
    
    def _extract_metrics(self, info: Dict) -> Dict:
        """Extract key financial metrics and ratios."""
        return {
            # Valuation
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "ev_to_revenue": info.get("enterpriseToRevenue"),
            
            # Profitability
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "gross_margin": info.get("grossMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            
            # Growth
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            
            # Dividends
            "dividend_yield": info.get("dividendYield"),
            "dividend_rate": info.get("dividendRate"),
            "payout_ratio": info.get("payoutRatio"),
            
            # Financial Health
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "debt_to_equity": info.get("debtToEquity"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "free_cash_flow": info.get("freeCashflow"),
            "operating_cash_flow": info.get("operatingCashflow"),
            
            # Per Share
            "eps_trailing": info.get("trailingEps"),
            "eps_forward": info.get("forwardEps"),
            "book_value": info.get("bookValue"),
            "revenue_per_share": info.get("revenuePerShare"),
            
            # Other
            "beta": info.get("beta"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "50_day_avg": info.get("fiftyDayAverage"),
            "200_day_avg": info.get("twoHundredDayAverage"),
        }
    
    def _df_to_dict(self, df: pd.DataFrame) -> Dict:
        """Convert pandas DataFrame to JSON-serializable dict."""
        if df is None or df.empty:
            return {}
        
        result = {}
        for col in df.columns:
            # Convert Timestamp to string
            col_key = col.strftime("%Y-%m-%d") if hasattr(col, 'strftime') else str(col)
            result[col_key] = {}
            for idx in df.index:
                val = df.loc[idx, col]
                # Handle NaN and convert numpy types
                if pd.isna(val):
                    result[col_key][str(idx)] = None
                elif hasattr(val, 'item'):
                    result[col_key][str(idx)] = val.item()
                else:
                    result[col_key][str(idx)] = val
        return result
    
    def _get_price_history(self, stock, period: str = "5y") -> List[Dict]:
        """Get historical price data."""
        try:
            hist = stock.history(period=period)
            if hist.empty:
                return []
            
            # Sample monthly for storage efficiency
            hist_monthly = hist.resample('ME').last()  # ME = Month End
            
            prices = []
            for date, row in hist_monthly.iterrows():
                prices.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(row["Open"], 2) if pd.notna(row["Open"]) else None,
                    "high": round(row["High"], 2) if pd.notna(row["High"]) else None,
                    "low": round(row["Low"], 2) if pd.notna(row["Low"]) else None,
                    "close": round(row["Close"], 2) if pd.notna(row["Close"]) else None,
                    "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else None,
                })
            return prices
        except Exception as e:
            logger.warning(f"Could not get price history: {e}")
            return []
    
    def _save_data(self, ticker: str, data: Dict):
        """Save data to JSON file."""
        # Sanitize ticker for filename
        safe_ticker = ticker.replace(".", "_").replace(":", "_")
        filepath = self.output_dir / f"{safe_ticker}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved {ticker} data to {filepath}")
    
    def fetch_multiple(self, tickers: List[str], delay: float = 0.5) -> Dict[str, bool]:
        """
        Fetch data for multiple companies.
        
        Args:
            tickers: List of ticker symbols
            delay: Delay between requests (seconds)
            
        Returns:
            Dict mapping ticker to success status
        """
        import time
        
        results = {}
        total = len(tickers)
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{total}] {ticker}...", end=" ", flush=True)
            
            data = self.fetch_company_data(ticker)
            success = data is not None
            results[ticker] = success
            
            print("✅" if success else "❌")
            
            if i < total:
                time.sleep(delay)
        
        return results


# Global ticker examples for reference
GLOBAL_TICKERS = {
    "US": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"],
    "UK": ["HSBA.L", "BP.L", "SHEL.L", "AZN.L", "ULVR.L", "RIO.L", "GSK.L", "DGE.L"],
    "India_NSE": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HINDUNILVR.NS"],
    "India_BSE": ["RELIANCE.BO", "TCS.BO", "INFY.BO", "HDFCBANK.BO"],
    "Japan": ["7203.T", "6758.T", "9984.T", "6861.T", "8306.T"],  # Toyota, Sony, SoftBank, Keyence, MUFG
    "Hong_Kong": ["0700.HK", "9988.HK", "1299.HK", "0005.HK"],  # Tencent, Alibaba, AIA, HSBC
    "Germany": ["SAP.DE", "SIE.DE", "ALV.DE", "BAS.DE", "BMW.DE"],  # SAP, Siemens, Allianz, BASF, BMW
    "France": ["MC.PA", "OR.PA", "SAN.PA", "AIR.PA", "TTE.PA"],  # LVMH, L'Oreal, Sanofi, Airbus, TotalEnergies
    "China_Shanghai": ["600519.SS", "601318.SS"],  # Kweichow Moutai, Ping An (limited data)
    "Australia": ["BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX"],
    "Canada": ["RY.TO", "TD.TO", "ENB.TO", "CNR.TO", "BMO.TO"],
}


if __name__ == "__main__":
    # Example usage
    collector = YahooFinanceCollector()
    
    # Test with a few global stocks
    test_tickers = [
        "AAPL",          # Apple (US)
        "RELIANCE.NS",   # Reliance Industries (India)
        "7203.T",        # Toyota (Japan)
        "HSBA.L",        # HSBC (UK)
    ]
    
    print("🌍 Testing Yahoo Finance global data collection\n")
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"Fetching: {ticker}")
        print('='*60)
        
        data = collector.fetch_company_data(ticker)
        
        if data:
            info = data["company_info"]
            print(f"  Company: {info['name']}")
            print(f"  Country: {info['country']}")
            print(f"  Sector: {info['sector']}")
            print(f"  Market Cap: ${info['market_cap']:,}" if info['market_cap'] else "  Market Cap: N/A")
            
            # Show available statements
            fin = data["financials"]
            print(f"\n  📊 Financial Statements:")
            print(f"     Income Statement: {len(fin['income_statement'])} periods")
            print(f"     Balance Sheet: {len(fin['balance_sheet'])} periods")
            print(f"     Cash Flow: {len(fin['cash_flow'])} periods")
