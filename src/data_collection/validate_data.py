"""
Data Validation Script for TemporalGuard-RAG

Validates that all required data sources are present and properly formatted.
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_data_collection(data_root: str = "data") -> dict:
    """
    Validate all data collection components.
    
    Args:
        data_root: Root directory for data
        
    Returns:
        Dictionary with validation results
    """
    data_path = Path(data_root)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_root': str(data_path.absolute()),
        'checks': {}
    }
    
    # Check SEC Filings
    sec_path = data_path / "raw" / "sec_filings"
    sec_files = list(sec_path.rglob("*.html")) + list(sec_path.rglob("*.htm"))
    results['checks']['sec_filings'] = {
        'exists': sec_path.exists(),
        'file_count': len(sec_files),
        'status': 'pass' if len(sec_files) >= 100 else 'warning' if len(sec_files) > 0 else 'fail',
        'threshold': 100
    }
    
    # Check XBRL Data
    xbrl_path = data_path / "raw" / "xbrl_structured"
    xbrl_files = list(xbrl_path.glob("*_facts.json"))
    results['checks']['xbrl_data'] = {
        'exists': xbrl_path.exists(),
        'file_count': len(xbrl_files),
        'status': 'pass' if len(xbrl_files) >= 10 else 'warning' if len(xbrl_files) > 0 else 'fail',
        'threshold': 10
    }
    
    # Check Stock Prices
    stock_path = data_path / "raw" / "stock_prices"
    stock_files = list(stock_path.glob("*.csv"))
    results['checks']['stock_prices'] = {
        'exists': stock_path.exists(),
        'file_count': len(stock_files),
        'status': 'pass' if len(stock_files) >= 10 else 'warning' if len(stock_files) > 0 else 'fail',
        'threshold': 10
    }
    
    # Check Earnings Transcripts
    transcript_path = data_path / "raw" / "earnings_transcripts"
    transcript_files = list(transcript_path.glob("*.json"))
    results['checks']['transcripts'] = {
        'exists': transcript_path.exists(),
        'file_count': len(transcript_files),
        'status': 'pass' if len(transcript_files) >= 20 else 'warning' if len(transcript_files) > 0 else 'fail',
        'threshold': 20
    }
    
    # Check Processed Data
    chunks_path = data_path / "processed" / "chunks"
    chunk_file = chunks_path / "temporal_chunks.jsonl"
    if chunk_file.exists():
        with open(chunk_file, 'r') as f:
            chunk_count = sum(1 for _ in f)
    else:
        chunk_count = 0
        
    results['checks']['processed_chunks'] = {
        'exists': chunk_file.exists(),
        'chunk_count': chunk_count,
        'status': 'pass' if chunk_count >= 1000 else 'warning' if chunk_count > 0 else 'not_started',
        'threshold': 1000
    }
    
    # Check Embeddings
    embeddings_path = data_path / "processed" / "embeddings"
    embeddings_file = embeddings_path / "embeddings.npy"
    results['checks']['embeddings'] = {
        'exists': embeddings_file.exists(),
        'status': 'pass' if embeddings_file.exists() else 'not_started'
    }
    
    # Check Evaluation Data
    eval_path = data_path / "evaluation"
    test_questions_file = eval_path / "test_questions.json"
    results['checks']['evaluation_data'] = {
        'exists': test_questions_file.exists(),
        'status': 'pass' if test_questions_file.exists() else 'not_started'
    }
    
    # Overall summary
    all_checks = results['checks']
    pass_count = sum(1 for c in all_checks.values() if c.get('status') == 'pass')
    warning_count = sum(1 for c in all_checks.values() if c.get('status') == 'warning')
    fail_count = sum(1 for c in all_checks.values() if c.get('status') == 'fail')
    not_started_count = sum(1 for c in all_checks.values() if c.get('status') == 'not_started')
    
    results['summary'] = {
        'total_checks': len(all_checks),
        'passed': pass_count,
        'warnings': warning_count,
        'failed': fail_count,
        'not_started': not_started_count,
        'overall_status': 'ready' if fail_count == 0 and pass_count > 3 else 'needs_work'
    }
    
    return results


def print_validation_report(results: dict):
    """Print formatted validation report."""
    
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Data Root: {results['data_root']}")
    print("-" * 60)
    
    status_icons = {
        'pass': '✅',
        'warning': '⚠️',
        'fail': '❌',
        'not_started': '⏸️'
    }
    
    for check_name, check_data in results['checks'].items():
        status = check_data.get('status', 'unknown')
        icon = status_icons.get(status, '❓')
        
        count_str = ""
        if 'file_count' in check_data:
            count_str = f" ({check_data['file_count']} files)"
        elif 'chunk_count' in check_data:
            count_str = f" ({check_data['chunk_count']} chunks)"
            
        threshold_str = ""
        if 'threshold' in check_data:
            threshold_str = f" [threshold: {check_data['threshold']}]"
            
        print(f"{icon} {check_name}: {status.upper()}{count_str}{threshold_str}")
    
    print("-" * 60)
    print("SUMMARY:")
    summary = results['summary']
    print(f"  Passed: {summary['passed']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Not Started: {summary['not_started']}")
    print(f"\n  Overall Status: {summary['overall_status'].upper()}")
    print("=" * 60)
    
    # Recommendations
    if summary['overall_status'] == 'needs_work':
        print("\nRECOMMENDATIONS:")
        
        checks = results['checks']
        
        if checks['sec_filings']['status'] in ['fail', 'warning']:
            print("  1. Run SEC downloader: python src/data_collection/sec_downloader.py")
            
        if checks['xbrl_data']['status'] in ['fail', 'warning']:
            print("  2. Run XBRL parser: python src/data_collection/xbrl_parser.py")
            
        if checks['stock_prices']['status'] in ['fail', 'warning']:
            print("  3. Run stock collector: python src/data_collection/stock_data.py")
            
        if checks['processed_chunks']['status'] == 'not_started':
            print("  4. Run temporal chunker: python src/preprocessing/temporal_chunker.py")
            
        if checks['embeddings']['status'] == 'not_started':
            print("  5. Run embedder: python src/preprocessing/embedder.py")


def validate_temporal_consistency(chunks_path: str = "data/processed/chunks/temporal_chunks.jsonl") -> dict:
    """
    Validate temporal consistency of processed chunks.
    
    Args:
        chunks_path: Path to temporal chunks file
        
    Returns:
        Dictionary with validation results
    """
    if not Path(chunks_path).exists():
        return {'status': 'error', 'message': 'Chunks file not found'}
        
    issues = []
    total_chunks = 0
    missing_dates = 0
    invalid_dates = 0
    
    with open(chunks_path, 'r') as f:
        for line in f:
            total_chunks += 1
            chunk = json.loads(line)
            
            # Check for filing date
            if 'filing_date' not in chunk or not chunk['filing_date']:
                missing_dates += 1
                
            # Validate date format
            if chunk.get('filing_date'):
                try:
                    # Try to parse date
                    date_str = chunk['filing_date']
                    if len(date_str) == 8:  # YYYYMMDD format
                        datetime.strptime(date_str, '%Y%m%d')
                    elif len(date_str) == 10:  # YYYY-MM-DD format
                        datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    invalid_dates += 1
                    
    return {
        'status': 'pass' if missing_dates == 0 and invalid_dates == 0 else 'warning',
        'total_chunks': total_chunks,
        'missing_dates': missing_dates,
        'invalid_dates': invalid_dates,
        'issues': issues
    }


# Usage
if __name__ == "__main__":
    # Run validation
    results = validate_data_collection()
    
    # Print report
    print_validation_report(results)
    
    # Save results
    output_path = Path("data/validation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nReport saved to: {output_path}")
