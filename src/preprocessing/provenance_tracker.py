"""
Provenance Tracker for TemporalGuard-RAG

Tracks document provenance with cryptographic verification for audit trails.
Implements chain of custody tracking for security and compliance.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import cryptography for encrypted provenance
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography package not installed. Encrypted provenance disabled.")


class ProvenanceTracker:
    """
    Cryptographic provenance tracking for document integrity and audit trails.
    
    Features:
    - Document fingerprinting (SHA-256/SHA-512)
    - Chain of custody tracking
    - Optional encrypted storage
    - Audit trail generation
    """
    
    def __init__(self, 
                 key_path: str = "data/security/provenance.key",
                 enable_encryption: bool = True):
        """
        Initialize Provenance Tracker.
        
        Args:
            key_path: Path to encryption key file
            enable_encryption: Whether to enable encrypted provenance storage
        """
        self.key_path = Path(key_path)
        self.enable_encryption = enable_encryption and CRYPTO_AVAILABLE
        
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.cipher = None
        if self.enable_encryption:
            self._setup_encryption()
            
        self.provenance_log = []
        
        logger.info(f"Initialized Provenance Tracker (encryption: {self.enable_encryption})")
        
    def _setup_encryption(self):
        """Set up encryption key and cipher."""
        if not self.key_path.exists():
            # Generate new key
            key = Fernet.generate_key()
            self.key_path.write_bytes(key)
            logger.info(f"Generated new encryption key at {self.key_path}")
        else:
            key = self.key_path.read_bytes()
            
        self.cipher = Fernet(key)
        
    def create_document_fingerprint(self, filepath: str) -> Dict:
        """
        Create cryptographic fingerprint for source document.
        
        Args:
            filepath: Path to the document
            
        Returns:
            Dictionary containing fingerprint information
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None
            
        with open(filepath, 'rb') as f:
            content = f.read()
            
        # Create multiple hashes for security
        sha256_hash = hashlib.sha256(content).hexdigest()
        sha512_hash = hashlib.sha512(content).hexdigest()
        md5_hash = hashlib.md5(content).hexdigest()  # For compatibility
        
        fingerprint = {
            'filepath': str(filepath),
            'filename': filepath.name,
            'sha256': sha256_hash,
            'sha512': sha512_hash,
            'md5': md5_hash,
            'size_bytes': len(content),
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        return fingerprint
        
    def track_chunk_provenance(self, 
                              chunk_id: str,
                              source_filepath: str,
                              chunk_text: str,
                              metadata: Dict) -> Dict:
        """
        Track provenance for a specific chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            source_filepath: Path to source document
            chunk_text: The chunk text content
            metadata: Additional metadata
            
        Returns:
            Provenance record dictionary
        """
        # Create fingerprint of source document
        doc_fingerprint = self.create_document_fingerprint(source_filepath)
        
        # Create chunk hash
        chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
        
        provenance_record = {
            'chunk_id': chunk_id,
            'chunk_hash': chunk_hash,
            'source_fingerprint': doc_fingerprint,
            'metadata': metadata,
            'created_at': datetime.now().isoformat(),
            'chain_of_custody': [
                {
                    'action': 'created',
                    'timestamp': datetime.now().isoformat(),
                    'agent': 'TemporalChunker',
                    'details': 'Initial chunk creation'
                }
            ]
        }
        
        # Optionally encrypt the record
        if self.enable_encryption and self.cipher:
            encrypted_record = self.cipher.encrypt(
                json.dumps(provenance_record).encode()
            )
            
            self.provenance_log.append({
                'chunk_id': chunk_id,
                'encrypted_record': encrypted_record.decode(),
                'encrypted': True
            })
        else:
            self.provenance_log.append({
                'chunk_id': chunk_id,
                'record': provenance_record,
                'encrypted': False
            })
            
        return provenance_record
        
    def verify_document_integrity(self, 
                                  filepath: str,
                                  original_fingerprint: Dict) -> Dict:
        """
        Verify document hasn't been tampered with.
        
        Args:
            filepath: Current path to document
            original_fingerprint: Original fingerprint to compare against
            
        Returns:
            Dictionary with verification results
        """
        current_fingerprint = self.create_document_fingerprint(filepath)
        
        if current_fingerprint is None:
            return {
                'verified': False,
                'reason': 'File not found',
                'filepath': filepath
            }
            
        # Check all hashes
        sha256_match = current_fingerprint['sha256'] == original_fingerprint['sha256']
        sha512_match = current_fingerprint['sha512'] == original_fingerprint['sha512']
        size_match = current_fingerprint['size_bytes'] == original_fingerprint['size_bytes']
        
        verified = sha256_match and sha512_match and size_match
        
        return {
            'verified': verified,
            'sha256_match': sha256_match,
            'sha512_match': sha512_match,
            'size_match': size_match,
            'original_fingerprint': original_fingerprint,
            'current_fingerprint': current_fingerprint,
            'verification_time': datetime.now().isoformat()
        }
        
    def add_to_chain_of_custody(self,
                               chunk_id: str,
                               action: str,
                               agent: str,
                               details: str = None) -> bool:
        """
        Add action to chain of custody.
        
        Args:
            chunk_id: Chunk identifier
            action: Action performed (e.g., 'retrieved', 'verified', 'cited')
            agent: Agent or component performing the action
            details: Optional additional details
            
        Returns:
            True if successful, False otherwise
        """
        for record in self.provenance_log:
            if record['chunk_id'] == chunk_id:
                if record.get('encrypted') and self.cipher:
                    # Decrypt, update, re-encrypt
                    encrypted_data = record['encrypted_record'].encode()
                    decrypted_data = self.cipher.decrypt(encrypted_data)
                    prov_record = json.loads(decrypted_data)
                else:
                    prov_record = record.get('record', {})
                    
                # Add to chain
                custody_entry = {
                    'action': action,
                    'timestamp': datetime.now().isoformat(),
                    'agent': agent
                }
                if details:
                    custody_entry['details'] = details
                    
                prov_record['chain_of_custody'].append(custody_entry)
                
                # Re-encrypt if needed
                if record.get('encrypted') and self.cipher:
                    encrypted_record = self.cipher.encrypt(
                        json.dumps(prov_record).encode()
                    )
                    record['encrypted_record'] = encrypted_record.decode()
                else:
                    record['record'] = prov_record
                    
                return True
                
        return False
        
    def get_provenance(self, chunk_id: str) -> Optional[Dict]:
        """
        Retrieve provenance record for a chunk.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Provenance record or None if not found
        """
        for record in self.provenance_log:
            if record['chunk_id'] == chunk_id:
                if record.get('encrypted') and self.cipher:
                    encrypted_data = record['encrypted_record'].encode()
                    decrypted_data = self.cipher.decrypt(encrypted_data)
                    return json.loads(decrypted_data)
                else:
                    return record.get('record')
                    
        return None
        
    def save_provenance_log(self, output_path: str = "data/security/provenance_log.jsonl"):
        """
        Save provenance log to file.
        
        Args:
            output_path: Path to save the log
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for record in self.provenance_log:
                f.write(json.dumps(record) + '\n')
                
        logger.info(f"Saved provenance log to {output_path}")
        logger.info(f"Total records: {len(self.provenance_log)}")
        
    def load_provenance_log(self, input_path: str = "data/security/provenance_log.jsonl"):
        """
        Load provenance log from file.
        
        Args:
            input_path: Path to load from
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.warning(f"Provenance log not found: {input_path}")
            return
            
        self.provenance_log = []
        with open(input_path, 'r') as f:
            for line in f:
                self.provenance_log.append(json.loads(line))
                
        logger.info(f"Loaded {len(self.provenance_log)} provenance records")
        
    def generate_audit_report(self, output_path: str = "results/provenance_audit.json") -> Dict:
        """
        Generate comprehensive audit report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Audit report dictionary
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_records': len(self.provenance_log),
            'encryption_enabled': self.enable_encryption,
            'statistics': {
                'chunks_tracked': 0,
                'unique_sources': set(),
                'actions_logged': 0,
                'agents_involved': set()
            },
            'chain_of_custody_summary': []
        }
        
        for record in self.provenance_log:
            # Get the actual provenance data
            if record.get('encrypted') and self.cipher:
                encrypted_data = record['encrypted_record'].encode()
                decrypted_data = self.cipher.decrypt(encrypted_data)
                prov = json.loads(decrypted_data)
            else:
                prov = record.get('record', {})
                
            report['statistics']['chunks_tracked'] += 1
            
            if prov.get('source_fingerprint'):
                report['statistics']['unique_sources'].add(
                    prov['source_fingerprint'].get('filepath', 'unknown')
                )
                
            chain = prov.get('chain_of_custody', [])
            report['statistics']['actions_logged'] += len(chain)
            
            for entry in chain:
                report['statistics']['agents_involved'].add(entry.get('agent', 'unknown'))
                
        # Convert sets to lists for JSON serialization
        report['statistics']['unique_sources'] = list(report['statistics']['unique_sources'])
        report['statistics']['agents_involved'] = list(report['statistics']['agents_involved'])
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Audit report saved to {output_path}")
        
        return report


def track_all_chunks_provenance(chunks_path: str = "data/processed/chunks/temporal_chunks.jsonl"):
    """
    Track provenance for all processed chunks.
    
    Args:
        chunks_path: Path to temporal chunks file
    """
    tracker = ProvenanceTracker()
    
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        return
        
    logger.info(f"Loading chunks from {chunks_path}...")
    
    with open(chunks_path, 'r') as f:
        for line in f:
            chunk = json.loads(line)
            
            tracker.track_chunk_provenance(
                chunk_id=chunk['chunk_id'],
                source_filepath=chunk['source_path'],
                chunk_text=chunk['text'],
                metadata={
                    'ticker': chunk.get('ticker'),
                    'filing_date': chunk.get('filing_date'),
                    'filing_type': chunk.get('filing_type')
                }
            )
            
    tracker.save_provenance_log()
    tracker.generate_audit_report()
    
    logger.info(f"✅ Tracked provenance for {len(tracker.provenance_log)} chunks")


# Usage
if __name__ == "__main__":
    # Example: Track provenance for a single document
    tracker = ProvenanceTracker()
    
    # Create fingerprint
    sample_path = "data/raw/sec_filings/sec-edgar-filings/AAPL/10-K/sample.html"
    
    if Path(sample_path).exists():
        fingerprint = tracker.create_document_fingerprint(sample_path)
        print("Document Fingerprint:")
        print(json.dumps(fingerprint, indent=2))
    else:
        print(f"Sample file not found: {sample_path}")
        print("\nTo track all chunks, run:")
        print("  from src.preprocessing.provenance_tracker import track_all_chunks_provenance")
        print("  track_all_chunks_provenance()")
