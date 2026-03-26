"""
Temporal-Aware Chunker for TemporalGuard-RAG

Creates document chunks while preserving temporal metadata critical
for look-ahead bias prevention and point-in-time analysis.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import hashlib
from datetime import datetime
import json
from pathlib import Path
import re
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalChunker:
    """
    Temporal-aware document chunker for financial documents.
    
    Key Features:
    - Preserves temporal metadata with each chunk
    - Creates cryptographic chunk fingerprints
    - Tracks provenance for audit trails
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize Temporal Chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len
        )
        
        logger.info(f"Initialized Temporal Chunker (size={chunk_size}, overlap={chunk_overlap})")
        
    def parse_sec_filing(self, filepath: str) -> Dict:
        """
        Extract text and filing metadata from SEC HTML filing.
        
        Args:
            filepath: Path to SEC filing HTML file
            
        Returns:
            Dictionary with text and metadata
        """
        filepath = Path(filepath)
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract filing date from document
            filing_date = self._extract_filing_date(soup, content)
            
            # Extract fiscal period
            fiscal_info = self._extract_fiscal_period(soup, content)
            
            # Extract text content
            text = self._extract_text(soup)
            
            # Parse company info from path
            # Expected path: sec-edgar-filings/TICKER/FILING_TYPE/...
            parts = filepath.parts
            ticker = None
            filing_type = None
            
            for i, part in enumerate(parts):
                if part == "sec-edgar-filings" and i + 2 < len(parts):
                    ticker = parts[i + 1]
                    filing_type = parts[i + 2]
                    break
                    
            # Fallback: try to extract from content
            if ticker is None:
                ticker = self._extract_ticker(soup, content)
                
            if filing_type is None:
                filing_type = self._extract_filing_type(content)
                
            return {
                'text': text,
                'ticker': ticker,
                'filing_type': filing_type,
                'filing_date': filing_date,
                'fiscal_year': fiscal_info.get('fiscal_year'),
                'fiscal_period': fiscal_info.get('fiscal_period'),
                'source_path': str(filepath),
                'file_size': filepath.stat().st_size,
                'parse_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")
            return None
            
    def _extract_filing_date(self, soup: BeautifulSoup, content: str) -> Optional[str]:
        """Extract filing date from SEC document."""
        
        # Method 1: Look for acceptance-datetime tag
        acceptance_tag = soup.find('acceptance-datetime')
        if acceptance_tag:
            date_str = acceptance_tag.text.strip()[:8]  # YYYYMMDD
            return date_str
            
        # Method 2: Look for FILED AS OF DATE
        filed_match = re.search(r'FILED AS OF DATE:\s*(\d{8})', content)
        if filed_match:
            return filed_match.group(1)
            
        # Method 3: Look for CONFORMED PERIOD OF REPORT
        period_match = re.search(r'CONFORMED PERIOD OF REPORT:\s*(\d{8})', content)
        if period_match:
            return period_match.group(1)
            
        # Method 4: Look for date patterns in content
        date_patterns = [
            r'Filed:\s*(\w+\s+\d{1,2},\s+\d{4})',
            r'Date:\s*(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content[:5000])
            if match:
                try:
                    date_str = match.group(1)
                    # Convert to YYYYMMDD format
                    for fmt in ['%B %d, %Y', '%m/%d/%Y', '%Y-%m-%d']:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            return dt.strftime('%Y%m%d')
                        except ValueError:
                            continue
                except:
                    pass
                    
        return None
        
    def _extract_fiscal_period(self, soup: BeautifulSoup, content: str) -> Dict:
        """Extract fiscal year and period information."""
        
        fiscal_info = {
            'fiscal_year': None,
            'fiscal_period': None
        }
        
        # Look for fiscal year end
        fy_match = re.search(r'FISCAL YEAR END:\s*(\d{4})', content)
        if fy_match:
            fiscal_info['fiscal_year'] = int(fy_match.group(1))
            
        # Determine fiscal period from filing type
        if '10-K' in content[:10000]:
            fiscal_info['fiscal_period'] = 'FY'
        elif '10-Q' in content[:10000]:
            # Try to determine quarter
            q_match = re.search(r'Q([1-4])|QUARTER\s*([1-4])', content[:10000], re.IGNORECASE)
            if q_match:
                quarter = q_match.group(1) or q_match.group(2)
                fiscal_info['fiscal_period'] = f'Q{quarter}'
            else:
                fiscal_info['fiscal_period'] = 'Q'
                
        return fiscal_info
        
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from HTML."""
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text
        
    def _extract_ticker(self, soup: BeautifulSoup, content: str) -> Optional[str]:
        """Extract ticker symbol from document."""
        
        # Look for trading symbol
        ticker_match = re.search(r'Trading Symbol:\s*([A-Z]{1,5})', content, re.IGNORECASE)
        if ticker_match:
            return ticker_match.group(1).upper()
            
        # Look for common exchange listings
        exchange_match = re.search(r'(NYSE|NASDAQ):\s*([A-Z]{1,5})', content, re.IGNORECASE)
        if exchange_match:
            return exchange_match.group(2).upper()
            
        return None
        
    def _extract_filing_type(self, content: str) -> Optional[str]:
        """Extract filing type from document."""
        
        type_match = re.search(r'FORM\s+(10-[KQ]|8-K|20-F)', content[:5000])
        if type_match:
            return type_match.group(1)
            
        for filing_type in ['10-K', '10-Q', '8-K', '20-F']:
            if filing_type in content[:5000]:
                return filing_type
                
        return None
        
    def create_temporal_chunks(self, document_dict: Dict) -> List[Dict]:
        """
        Chunk document while preserving temporal metadata.
        
        Args:
            document_dict: Dictionary from parse_sec_filing
            
        Returns:
            List of chunk dictionaries with temporal metadata
        """
        if document_dict is None or 'text' not in document_dict:
            return []
            
        # Split text into chunks
        chunks = self.splitter.split_text(document_dict['text'])
        
        temporal_chunks = []
        
        for i, chunk_text in enumerate(chunks):
            # Create unique chunk ID using hash
            chunk_id = hashlib.sha256(
                f"{document_dict['source_path']}_{i}".encode()
            ).hexdigest()[:16]
            
            # Create chunk hash for integrity verification
            chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
            
            chunk = {
                # Identifiers
                'chunk_id': chunk_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
                
                # Content
                'text': chunk_text,
                'text_length': len(chunk_text),
                
                # Temporal metadata (CRITICAL for look-ahead bias prevention!)
                'ticker': document_dict.get('ticker'),
                'filing_type': document_dict.get('filing_type'),
                'filing_date': document_dict.get('filing_date'),
                'fiscal_year': document_dict.get('fiscal_year'),
                'fiscal_period': document_dict.get('fiscal_period'),
                
                # Provenance
                'source_path': document_dict.get('source_path'),
                'processing_date': datetime.now().isoformat(),
                'chunk_hash': chunk_hash,
                
                # Configuration used
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
            
            temporal_chunks.append(chunk)
            
        return temporal_chunks
        
    def process_all_filings(self, 
                           input_dir: str = "data/raw/sec_filings",
                           output_dir: str = "data/processed/chunks") -> List[Dict]:
        """
        Process all SEC filings into temporal chunks.
        
        Args:
            input_dir: Directory containing SEC filings
            output_dir: Directory to save processed chunks
            
        Returns:
            List of all chunks
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all SEC filing files (HTML, HTM, or TXT)
        filings = (list(input_path.rglob("*.html")) + 
                   list(input_path.rglob("*.htm")) + 
                   list(input_path.rglob("*.txt")))
        
        logger.info(f"Found {len(filings)} filings to process")
        
        all_chunks = []
        processed_count = 0
        error_count = 0
        
        for i, filepath in enumerate(filings):
            if i % 10 == 0:
                logger.info(f"Processing {i+1}/{len(filings)} filings...")
                
            try:
                # Parse filing
                doc = self.parse_sec_filing(filepath)
                
                if doc is None:
                    error_count += 1
                    continue
                    
                # Create temporal chunks
                chunks = self.create_temporal_chunks(doc)
                all_chunks.extend(chunks)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                error_count += 1
                
        # Save chunks to JSONL
        output_file = output_path / "temporal_chunks.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                
        logger.info(f"✅ Created {len(all_chunks)} temporal chunks")
        logger.info(f"   Processed: {processed_count} filings")
        logger.info(f"   Errors: {error_count} filings")
        logger.info(f"   Output: {output_file}")
        
        # Save processing metadata
        metadata = {
            'total_chunks': len(all_chunks),
            'total_filings': len(filings),
            'processed_filings': processed_count,
            'error_filings': error_count,
            'processing_date': datetime.now().isoformat(),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'input_dir': str(input_path),
            'output_file': str(output_file)
        }
        
        metadata_file = output_path / "processing_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Generate chunk statistics
        self._generate_statistics(all_chunks, output_path)
            
        return all_chunks
        
    def _generate_statistics(self, chunks: List[Dict], output_path: Path):
        """Generate statistics about processed chunks."""
        
        stats = {
            'total_chunks': len(chunks),
            'by_ticker': {},
            'by_filing_type': {},
            'by_year': {},
            'temporal_coverage': {
                'earliest_filing': None,
                'latest_filing': None
            }
        }
        
        filing_dates = []
        
        for chunk in chunks:
            ticker = chunk.get('ticker', 'unknown')
            filing_type = chunk.get('filing_type', 'unknown')
            filing_date = chunk.get('filing_date')
            
            # Count by ticker
            stats['by_ticker'][ticker] = stats['by_ticker'].get(ticker, 0) + 1
            
            # Count by filing type
            stats['by_filing_type'][filing_type] = stats['by_filing_type'].get(filing_type, 0) + 1
            
            # Count by year
            if filing_date and len(filing_date) >= 4:
                year = filing_date[:4]
                stats['by_year'][year] = stats['by_year'].get(year, 0) + 1
                filing_dates.append(filing_date)
                
        if filing_dates:
            stats['temporal_coverage']['earliest_filing'] = min(filing_dates)
            stats['temporal_coverage']['latest_filing'] = max(filing_dates)
            
        stats_file = output_path / "chunk_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        logger.info(f"Statistics saved to {stats_file}")

    def process_yahoo_finance_data(self,
                                   input_dir: str = "data/raw/yahoo_finance",
                                   output_dir: str = "data/processed/chunks") -> List[Dict]:
        """
        Process Yahoo Finance JSON files into temporal chunks.
        
        Creates chunks for:
        - Income Statement line items
        - Balance Sheet line items
        - Cash Flow Statement line items
        - Key metrics and ratios
        
        Args:
            input_dir: Directory containing Yahoo Finance JSON files
            output_dir: Output directory for chunks
            
        Returns:
            List of all chunks created
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all Yahoo Finance JSON files
        json_files = list(input_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} Yahoo Finance files to process")
        
        all_chunks = []
        
        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                ticker = data.get('ticker', filepath.stem)
                company_info = data.get('company_info', {})
                financials = data.get('financials', {})
                metrics = data.get('metrics', {})
                
                # Process each financial statement type
                for statement_type, statement_data in financials.items():
                    if not statement_data:
                        continue
                    
                    chunks = self._create_financial_statement_chunks(
                        ticker=ticker,
                        company_name=company_info.get('name', ticker),
                        country=company_info.get('country', ''),
                        currency=company_info.get('currency', 'USD'),
                        statement_type=statement_type,
                        statement_data=statement_data,
                        source_path=str(filepath)
                    )
                    all_chunks.extend(chunks)
                
                # Process metrics
                if metrics:
                    metrics_chunks = self._create_metrics_chunks(
                        ticker=ticker,
                        company_name=company_info.get('name', ticker),
                        country=company_info.get('country', ''),
                        currency=company_info.get('currency', 'USD'),
                        metrics=metrics,
                        source_path=str(filepath)
                    )
                    all_chunks.extend(metrics_chunks)
                    
                logger.info(f"Processed {ticker}: {len(all_chunks)} total chunks")
                
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
        
        # Save to JSONL file
        output_file = output_path / "yahoo_finance_chunks.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Created {len(all_chunks)} Yahoo Finance chunks")
        return all_chunks
    
    def _create_financial_statement_chunks(self,
                                           ticker: str,
                                           company_name: str,
                                           country: str,
                                           currency: str,
                                           statement_type: str,
                                           statement_data: Dict,
                                           source_path: str) -> List[Dict]:
        """Create chunks from a financial statement."""
        
        chunks = []
        
        # Map statement types to readable names
        statement_names = {
            'income_statement': 'Income Statement (P&L)',
            'balance_sheet': 'Balance Sheet',
            'cash_flow': 'Cash Flow Statement',
            'quarterly_income': 'Quarterly Income Statement',
            'quarterly_balance': 'Quarterly Balance Sheet',
            'quarterly_cashflow': 'Quarterly Cash Flow'
        }
        
        statement_name = statement_names.get(statement_type, statement_type)
        is_quarterly = 'quarterly' in statement_type
        
        for period_date, line_items in statement_data.items():
            if not line_items:
                continue
                
            # Build readable text for this period
            text_lines = [
                f"Company: {company_name} ({ticker})",
                f"Country: {country}",
                f"Statement: {statement_name}",
                f"Period: {period_date}",
                f"Currency: {currency}",
                "",
                "Financial Data:"
            ]
            
            for item_name, value in line_items.items():
                if value is not None:
                    # Format large numbers
                    if isinstance(value, (int, float)) and abs(value) >= 1e6:
                        formatted = f"${value/1e6:,.1f}M" if abs(value) < 1e9 else f"${value/1e9:,.2f}B"
                    elif isinstance(value, (int, float)):
                        formatted = f"${value:,.2f}" if abs(value) < 100000 else f"${value:,.0f}"
                    else:
                        formatted = str(value)
                    text_lines.append(f"  {item_name}: {formatted}")
            
            text = "\n".join(text_lines)
            
            # Create chunk ID
            chunk_id = hashlib.sha256(
                f"{ticker}_{statement_type}_{period_date}".encode()
            ).hexdigest()[:16]
            
            # Extract fiscal info from period date
            fiscal_year = None
            fiscal_period = None
            try:
                dt = datetime.strptime(period_date, "%Y-%m-%d")
                fiscal_year = dt.year
                # Estimate quarter
                month = dt.month
                quarter_map = {1: 'Q4', 2: 'Q4', 3: 'Q4',  # Jan-Mar = prev fiscal Q4
                              4: 'Q1', 5: 'Q1', 6: 'Q1',
                              7: 'Q2', 8: 'Q2', 9: 'Q2',
                              10: 'Q3', 11: 'Q3', 12: 'Q3'}
                fiscal_period = quarter_map.get(month, 'FY') if is_quarterly else 'FY'
            except:
                pass
            
            chunk = {
                'chunk_id': chunk_id,
                'chunk_index': 0,
                'total_chunks': 1,
                'text': text,
                'text_length': len(text),
                
                # Temporal metadata
                'ticker': ticker,
                'filing_type': statement_type,
                'filing_date': period_date.replace('-', ''),
                'fiscal_year': fiscal_year,
                'fiscal_period': fiscal_period,
                
                # Source metadata
                'source': 'yahoo_finance',
                'source_path': source_path,
                'country': country,
                'currency': currency,
                'company_name': company_name,
                'statement_type': statement_type,
                
                # Processing info
                'processing_date': datetime.now().isoformat(),
                'chunk_hash': hashlib.sha256(text.encode()).hexdigest(),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _create_metrics_chunks(self,
                               ticker: str,
                               company_name: str,
                               country: str,
                               currency: str,
                               metrics: Dict,
                               source_path: str) -> List[Dict]:
        """Create a chunk from company metrics."""
        
        # Group metrics by category
        categories = {
            'Valuation': ['pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book', 
                         'price_to_sales', 'ev_to_ebitda', 'ev_to_revenue'],
            'Profitability': ['profit_margin', 'operating_margin', 'gross_margin', 'roe', 'roa'],
            'Growth': ['revenue_growth', 'earnings_growth', 'earnings_quarterly_growth'],
            'Dividends': ['dividend_yield', 'dividend_rate', 'payout_ratio'],
            'Financial Health': ['current_ratio', 'quick_ratio', 'debt_to_equity', 
                                'total_debt', 'total_cash', 'free_cash_flow', 'operating_cash_flow'],
            'Per Share': ['eps_trailing', 'eps_forward', 'book_value', 'revenue_per_share'],
        }
        
        text_lines = [
            f"Company: {company_name} ({ticker})",
            f"Country: {country}",
            f"Type: Key Financial Metrics & Ratios",
            f"Currency: {currency}",
            ""
        ]
        
        for category, metric_keys in categories.items():
            values_found = []
            for key in metric_keys:
                val = metrics.get(key)
                if val is not None:
                    # Format the value
                    display_name = key.replace('_', ' ').title()
                    if isinstance(val, float) and val < 10:
                        formatted = f"{val:.2f}"
                    elif isinstance(val, (int, float)) and abs(val) >= 1e6:
                        formatted = f"${val/1e6:.1f}M" if abs(val) < 1e9 else f"${val/1e9:.2f}B"
                    else:
                        formatted = str(val)
                    values_found.append(f"  {display_name}: {formatted}")
            
            if values_found:
                text_lines.append(f"\n{category}:")
                text_lines.extend(values_found)
        
        text = "\n".join(text_lines)
        
        chunk_id = hashlib.sha256(f"{ticker}_metrics".encode()).hexdigest()[:16]
        
        return [{
            'chunk_id': chunk_id,
            'chunk_index': 0,
            'total_chunks': 1,
            'text': text,
            'text_length': len(text),
            
            'ticker': ticker,
            'filing_type': 'metrics',
            'filing_date': datetime.now().strftime('%Y%m%d'),
            'fiscal_year': datetime.now().year,
            'fiscal_period': 'current',
            
            'source': 'yahoo_finance',
            'source_path': source_path,
            'country': country,
            'currency': currency,
            'company_name': company_name,
            
            'processing_date': datetime.now().isoformat(),
            'chunk_hash': hashlib.sha256(text.encode()).hexdigest(),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }]


# Usage
if __name__ == "__main__":
    # Initialize chunker
    chunker = TemporalChunker(chunk_size=500, chunk_overlap=50)
    
    # Process all filings
    chunks = chunker.process_all_filings()
    
    # Print sample chunk
    if chunks:
        print("\nSample Chunk:")
        print(json.dumps(chunks[0], indent=2))
