"""
Batch Company Adder for TemporalGuard-RAG

Add companies from US (SEC XBRL) or GLOBAL (Yahoo Finance)
Run: python scripts/add_companies.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.xbrl_parser import XBRLCollector
from src.data_collection.yahoo_finance import YahooFinanceCollector, GLOBAL_TICKERS
from src.preprocessing.temporal_chunker import TemporalChunker
from src.rag_system.vector_store import TemporalVectorStore
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# ADD YOUR COMPANIES HERE
# ═══════════════════════════════════════════════════════════════

COMPANIES = [
    # Tech Giants
    "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    
    # Finance
    "BAC", "WFC", "C", "MS", "BLK",
    
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK",
    
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST",
    
    # Industrial
    "CAT", "BA", "HON", "GE", "MMM",
    
    # Add more tickers here...
]

# ═══════════════════════════════════════════════════════════════


def download_xbrl_data(tickers: list, delay: float = 0.5):
    """
    Download XBRL data for multiple companies.
    
    Args:
        tickers: List of stock ticker symbols
        delay: Seconds between requests (SEC rate limit)
    """
    collector = XBRLCollector(
        user_agent="TemporalGuardRAG research@university.edu"  # Change this!
    )
    
    success = []
    failed = []
    
    print(f"\n📥 Downloading XBRL data for {len(tickers)} companies...\n")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ")
        
        try:
            result = collector.fetch_company_facts(ticker)
            if result:
                print("✅")
                success.append(ticker)
            else:
                print("❌ No data")
                failed.append(ticker)
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            failed.append(ticker)
        
        # SEC rate limit: 10 requests/second max
        time.sleep(delay)
    
    print(f"\n{'='*50}")
    print(f"✅ Success: {len(success)} companies")
    print(f"❌ Failed: {len(failed)} companies")
    if failed:
        print(f"   Failed tickers: {', '.join(failed)}")
    
    return success, failed


def download_yahoo_data(tickers: list, delay: float = 0.3):
    """
    Download financial statements from Yahoo Finance (GLOBAL coverage).
    
    Args:
        tickers: List of stock ticker symbols (any exchange)
        delay: Seconds between requests
    """
    collector = YahooFinanceCollector()
    
    success = []
    failed = []
    
    print(f"\n🌍 Downloading Yahoo Finance data for {len(tickers)} companies...\n")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ")
        
        try:
            result = collector.fetch_company_data(ticker)
            if result:
                print("✅")
                success.append(ticker)
            else:
                print("❌ No data")
                failed.append(ticker)
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            failed.append(ticker)
        
        time.sleep(delay)
    
    print(f"\n{'='*50}")
    print(f"✅ Success: {len(success)} companies")
    print(f"❌ Failed: {len(failed)} companies")
    if failed:
        print(f"   Failed tickers: {', '.join(failed[:20])}")
    
    return success, failed


def rebuild_vector_store():
    """Rebuild the vector store with new documents."""
    print("\n🔄 Rebuilding vector store...")
    
    # This would reprocess all documents
    # For now, just inform user
    print("""
To rebuild the vector store with new companies:

1. Run the chunker:
   python -c "from src.preprocessing.temporal_chunker import TemporalChunker; TemporalChunker().process_all()"

2. Rebuild embeddings:
   python -c "from src.rag_system.vector_store import TemporalVectorStore; vs = TemporalVectorStore(); vs.rebuild()"

Or simply restart the app - it will detect new data.
""")


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          TemporalGuard-RAG: Batch Company Adder              ║
║                   US + GLOBAL Coverage                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("📊 DATA SOURCE:")
    print("  1. SEC XBRL (US companies - detailed financials)")
    print("  2. Yahoo Finance (GLOBAL - any exchange)")
    
    source = input("\nSelect data source (1-2): ").strip()
    
    if source == "1":
        # SEC XBRL options
        print("\n📁 US COMPANIES (SEC XBRL):")
        print("  1. Custom list (edit COMPANIES in script)")
        print("  2. S&P 500 Top 100")
        print("  3. ALL SEC companies (~10,000)")
        print("  4. By sector (tech/finance/healthcare/energy)")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            tickers = COMPANIES
        elif choice == "2":
            from scripts.company_lists import SP500_TOP_100
            tickers = SP500_TOP_100
        elif choice == "3":
            from scripts.company_lists import get_all_tickers
            print("\n⚠️  WARNING: ~10,000 companies, 2-3 hours download time")
            confirm = input("   Continue? (type 'yes'): ").strip()
            if confirm != 'yes':
                return
            tickers = get_all_tickers()
        elif choice == "4":
            from scripts.company_lists import TECH_SECTOR, FINANCE_SECTOR, HEALTHCARE_SECTOR, ENERGY_SECTOR
            sector = input("Enter sector (tech/finance/healthcare/energy): ").strip().lower()
            sectors = {"tech": TECH_SECTOR, "finance": FINANCE_SECTOR, 
                      "healthcare": HEALTHCARE_SECTOR, "energy": ENERGY_SECTOR}
            tickers = sectors.get(sector, [])
            if not tickers:
                print("Unknown sector")
                return
        else:
            print("Invalid")
            return
        
        print(f"\nCompanies: {len(tickers)}")
        response = input("Proceed? (y/n): ").strip().lower()
        if response != 'y':
            return
        
        download_xbrl_data(tickers)
        
    elif source == "2":
        # Yahoo Finance options
        print("\n🌍 GLOBAL COMPANIES (Yahoo Finance):")
        print("  1. Enter tickers manually")
        print("  2. Pre-made: US Top 10")
        print("  3. Pre-made: UK Top Stocks")
        print("  4. Pre-made: India (NSE)")
        print("  5. Pre-made: Japan")
        print("  6. Pre-made: Europe (Germany, France)")
        print("  7. Pre-made: All regions sample")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            print("\nEnter tickers separated by comma (e.g., AAPL, RELIANCE.NS, 7203.T):")
            print("Ticker formats:")
            print("  US: AAPL, MSFT")
            print("  UK: HSBA.L, BP.L") 
            print("  India NSE: RELIANCE.NS, TCS.NS")
            print("  Japan: 7203.T (Toyota)")
            print("  Hong Kong: 0700.HK (Tencent)")
            print("  Germany: SAP.DE")
            tickers_input = input("\nTickers: ").strip()
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        elif choice == "2":
            tickers = GLOBAL_TICKERS["US"]
        elif choice == "3":
            tickers = GLOBAL_TICKERS["UK"]
        elif choice == "4":
            tickers = GLOBAL_TICKERS["India_NSE"]
        elif choice == "5":
            tickers = GLOBAL_TICKERS["Japan"]
        elif choice == "6":
            tickers = GLOBAL_TICKERS["Germany"] + GLOBAL_TICKERS["France"]
        elif choice == "7":
            # Sample from all regions
            tickers = []
            for region, stocks in GLOBAL_TICKERS.items():
                tickers.extend(stocks[:3])  # Top 3 from each
        else:
            print("Invalid")
            return
        
        if not tickers:
            print("No tickers specified")
            return
        
        print(f"\nCompanies: {len(tickers)}")
        print(f"Tickers: {', '.join(tickers[:15])}{'...' if len(tickers) > 15 else ''}")
        response = input("Proceed? (y/n): ").strip().lower()
        if response != 'y':
            return
        
        download_yahoo_data(tickers)
    else:
        print("Invalid source")
        return
    
    rebuild_vector_store()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
