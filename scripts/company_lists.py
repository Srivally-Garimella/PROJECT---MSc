# Company Lists for TemporalGuard-RAG
# Includes function to fetch ALL SEC-registered companies

import requests
from typing import List, Dict

def get_all_sec_companies() -> List[Dict]:
    """
    Fetch ALL companies registered with SEC (~10,000+ tickers).
    Returns list of {"cik": "...", "ticker": "...", "title": "..."}
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "YourEmail@example.com"}  # CHANGE THIS!
    
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    
    companies = []
    for entry in data.values():
        companies.append({
            "cik": str(entry["cik_str"]).zfill(10),
            "ticker": entry["ticker"],
            "title": entry["title"]
        })
    return companies

def get_all_tickers() -> List[str]:
    """Get just the ticker symbols for all SEC companies (~10,000+)."""
    return [c["ticker"] for c in get_all_sec_companies()]


# ============================================================
# For GLOBAL companies (non-US), you'd need these data sources:
# - Yahoo Finance API (global coverage, limited financials)
# - Alpha Vantage (global, 500 calls/day free)
# - EOD Historical Data (paid, comprehensive global)
# - IEX Cloud (US-focused, some international)
# ============================================================

# Top 100 by market cap (as of 2024)
SP500_TOP_100 = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", 
    "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "ACN", "INTC",
    "IBM", "QCOM", "TXN", "NOW", "INTU", "AMAT", "MU", "ADI",
    
    # Finance
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP",
    "SPGI", "SCHW", "CB", "MMC", "PGR", "CME", "ICE", "AON",
    
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "MDT", "ISRG", "GILD", "CVS", "ELV", "CI", "SYK",
    
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT",
    "HD", "LOW", "TJX", "AMZN", "BKNG", "MAR", "HLT",
    
    # Industrial
    "CAT", "BA", "HON", "GE", "RTX", "LMT", "UPS", "DE", "UNP",
    "MMM", "EMR", "ITW", "ETN", "PH", "ROK",
    
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    
    # Communication
    "DIS", "CMCSA", "NFLX", "TMUS", "VZ", "T", "CHTR",
    
    # Real Estate
    "PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL",
    
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
    
    # Materials
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DD",
]

# Sector-specific lists for focused analysis
TECH_SECTOR = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", 
               "ORCL", "CRM", "AMD", "ADBE", "CSCO", "ACN", "INTC", "IBM"]

FINANCE_SECTOR = ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
                  "SPGI", "MMC", "PGR", "CME", "ICE", "AON", "V", "MA"]

HEALTHCARE_SECTOR = ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
                     "DHR", "BMY", "AMGN", "MDT", "ISRG", "GILD", "CVS", "ELV"]

ENERGY_SECTOR = ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
                 "HAL", "BKR", "FANG", "DVN", "HES", "MRO", "APA"]
