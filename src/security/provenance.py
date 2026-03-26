"""
Data Provenance Tracking for TemporalGuard-RAG

Ensures data integrity and tracks the complete chain of custody
for all financial data used in the RAG system.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
from enum import Enum
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformationType(Enum):
    """Types of data transformations."""
    DOWNLOAD = "download"
    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"
    FILTER = "filter"
    RETRIEVE = "retrieve"
    AGGREGATE = "aggregate"
    ANONYMIZE = "anonymize"


@dataclass
class ProvenanceRecord:
    """Single provenance record in the chain."""
    record_id: str
    timestamp: str
    transformation: TransformationType
    input_hash: str
    output_hash: str
    parameters: Dict[str, Any]
    agent: str
    parent_id: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'record_id': self.record_id,
            'timestamp': self.timestamp,
            'transformation': self.transformation.value,
            'input_hash': self.input_hash,
            'output_hash': self.output_hash,
            'parameters': self.parameters,
            'agent': self.agent,
            'parent_id': self.parent_id,
            'notes': self.notes
        }


@dataclass
class DocumentProvenance:
    """Complete provenance information for a document."""
    document_id: str
    source_uri: str
    source_type: str  # 'sec_edgar', 'xbrl', 'earnings_call', etc.
    original_hash: str
    filing_date: str
    download_timestamp: str
    ticker: str
    filing_type: str
    chain: List[ProvenanceRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_record(self, record: ProvenanceRecord):
        """Add a provenance record to the chain."""
        # Link to previous record
        if self.chain:
            record.parent_id = self.chain[-1].record_id
        self.chain.append(record)
        
    def get_current_hash(self) -> str:
        """Get the hash of the current document state."""
        if self.chain:
            return self.chain[-1].output_hash
        return self.original_hash
        
    def verify_chain(self) -> bool:
        """Verify the integrity of the provenance chain."""
        if not self.chain:
            return True
            
        # First record should have original hash as input
        if self.chain[0].input_hash != self.original_hash:
            logger.error("First record input doesn't match original hash")
            return False
            
        # Each subsequent record should chain correctly
        for i in range(1, len(self.chain)):
            expected_input = self.chain[i-1].output_hash
            actual_input = self.chain[i].input_hash
            
            if expected_input != actual_input:
                logger.error(f"Chain break at record {i}: expected {expected_input}, got {actual_input}")
                return False
                
        return True
        
    def to_dict(self) -> Dict:
        return {
            'document_id': self.document_id,
            'source_uri': self.source_uri,
            'source_type': self.source_type,
            'original_hash': self.original_hash,
            'filing_date': self.filing_date,
            'download_timestamp': self.download_timestamp,
            'ticker': self.ticker,
            'filing_type': self.filing_type,
            'current_hash': self.get_current_hash(),
            'chain_length': len(self.chain),
            'chain_valid': self.verify_chain(),
            'chain': [r.to_dict() for r in self.chain],
            'metadata': self.metadata
        }


class ProvenanceChain:
    """
    Manages provenance tracking for the RAG system.
    
    Features:
    - Cryptographic hashing for data integrity
    - Complete chain of custody tracking
    - Verification and audit capabilities
    - Export for compliance
    """
    
    def __init__(self, storage_path: str = "data/provenance"):
        """
        Initialize provenance chain manager.
        
        Args:
            storage_path: Path to store provenance records
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.documents: Dict[str, DocumentProvenance] = {}
        self._load_existing()
        
    def _load_existing(self):
        """Load existing provenance records."""
        if self.storage_path.exists():
            for file_path in self.storage_path.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        doc_id = data.get('document_id')
                        if doc_id:
                            self.documents[doc_id] = self._dict_to_provenance(data)
                except Exception as e:
                    logger.warning(f"Could not load provenance file {file_path}: {e}")
                    
    def _dict_to_provenance(self, data: Dict) -> DocumentProvenance:
        """Convert dictionary to DocumentProvenance."""
        provenance = DocumentProvenance(
            document_id=data['document_id'],
            source_uri=data['source_uri'],
            source_type=data['source_type'],
            original_hash=data['original_hash'],
            filing_date=data['filing_date'],
            download_timestamp=data['download_timestamp'],
            ticker=data['ticker'],
            filing_type=data['filing_type'],
            metadata=data.get('metadata', {})
        )
        
        for record_data in data.get('chain', []):
            record = ProvenanceRecord(
                record_id=record_data['record_id'],
                timestamp=record_data['timestamp'],
                transformation=TransformationType(record_data['transformation']),
                input_hash=record_data['input_hash'],
                output_hash=record_data['output_hash'],
                parameters=record_data['parameters'],
                agent=record_data['agent'],
                parent_id=record_data.get('parent_id'),
                notes=record_data.get('notes')
            )
            provenance.chain.append(record)
            
        return provenance
        
    @staticmethod
    def compute_hash(content: Any) -> str:
        """
        Compute SHA-256 hash of content.
        
        Args:
            content: Content to hash (string, bytes, or serializable object)
            
        Returns:
            Hexadecimal hash string
        """
        if isinstance(content, str):
            data = content.encode('utf-8')
        elif isinstance(content, bytes):
            data = content
        else:
            data = json.dumps(content, sort_keys=True).encode('utf-8')
            
        return hashlib.sha256(data).hexdigest()
        
    def register_document(self,
                          content: Any,
                          source_uri: str,
                          source_type: str,
                          ticker: str,
                          filing_type: str,
                          filing_date: str,
                          metadata: Optional[Dict] = None) -> DocumentProvenance:
        """
        Register a new document and create initial provenance.
        
        Args:
            content: Document content
            source_uri: Original source URI
            source_type: Type of source (e.g., 'sec_edgar')
            ticker: Company ticker
            filing_type: Type of filing (e.g., '10-K')
            filing_date: Filing date
            metadata: Additional metadata
            
        Returns:
            DocumentProvenance object
        """
        doc_id = str(uuid.uuid4())
        original_hash = self.compute_hash(content)
        
        provenance = DocumentProvenance(
            document_id=doc_id,
            source_uri=source_uri,
            source_type=source_type,
            original_hash=original_hash,
            filing_date=filing_date,
            download_timestamp=datetime.now().isoformat(),
            ticker=ticker,
            filing_type=filing_type,
            metadata=metadata or {}
        )
        
        # Create initial download record
        download_record = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            transformation=TransformationType.DOWNLOAD,
            input_hash=original_hash,  # External source
            output_hash=original_hash,
            parameters={
                'source_uri': source_uri,
                'source_type': source_type
            },
            agent='data_collector'
        )
        
        provenance.add_record(download_record)
        
        self.documents[doc_id] = provenance
        self._save_provenance(provenance)
        
        logger.info(f"Registered document {doc_id} from {source_uri}")
        
        return provenance
        
    def record_transformation(self,
                              document_id: str,
                              transformation: TransformationType,
                              input_content: Any,
                              output_content: Any,
                              parameters: Dict[str, Any],
                              agent: str,
                              notes: Optional[str] = None) -> ProvenanceRecord:
        """
        Record a data transformation.
        
        Args:
            document_id: Document ID
            transformation: Type of transformation
            input_content: Input data
            output_content: Output data
            parameters: Transformation parameters
            agent: Agent/module performing transformation
            notes: Optional notes
            
        Returns:
            ProvenanceRecord
        """
        if document_id not in self.documents:
            raise ValueError(f"Unknown document: {document_id}")
            
        provenance = self.documents[document_id]
        
        input_hash = self.compute_hash(input_content)
        output_hash = self.compute_hash(output_content)
        
        # Verify input hash matches current state
        current_hash = provenance.get_current_hash()
        if input_hash != current_hash:
            logger.warning(f"Input hash mismatch: expected {current_hash}, got {input_hash}")
            
        record = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            transformation=transformation,
            input_hash=input_hash,
            output_hash=output_hash,
            parameters=parameters,
            agent=agent,
            notes=notes
        )
        
        provenance.add_record(record)
        self._save_provenance(provenance)
        
        logger.debug(f"Recorded {transformation.value} for document {document_id}")
        
        return record
        
    def _save_provenance(self, provenance: DocumentProvenance):
        """Save provenance to file."""
        file_path = self.storage_path / f"{provenance.document_id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(provenance.to_dict(), f, indent=2)
            
    def get_provenance(self, document_id: str) -> Optional[DocumentProvenance]:
        """Get provenance for a document."""
        return self.documents.get(document_id)
        
    def verify_document(self, document_id: str, current_content: Any) -> Dict:
        """
        Verify document integrity against provenance.
        
        Args:
            document_id: Document ID
            current_content: Current document content
            
        Returns:
            Verification result dict
        """
        if document_id not in self.documents:
            return {
                'valid': False,
                'error': 'Unknown document ID',
                'document_id': document_id
            }
            
        provenance = self.documents[document_id]
        
        # Verify chain integrity
        chain_valid = provenance.verify_chain()
        
        # Verify current content matches latest hash
        current_hash = self.compute_hash(current_content)
        expected_hash = provenance.get_current_hash()
        content_valid = current_hash == expected_hash
        
        return {
            'valid': chain_valid and content_valid,
            'document_id': document_id,
            'chain_valid': chain_valid,
            'content_valid': content_valid,
            'current_hash': current_hash,
            'expected_hash': expected_hash,
            'chain_length': len(provenance.chain),
            'source_uri': provenance.source_uri
        }
        
    def get_lineage(self, document_id: str) -> List[Dict]:
        """
        Get complete lineage (transformation history) for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of transformation records
        """
        if document_id not in self.documents:
            return []
            
        provenance = self.documents[document_id]
        
        lineage = []
        for record in provenance.chain:
            lineage.append({
                'step': len(lineage) + 1,
                'timestamp': record.timestamp,
                'transformation': record.transformation.value,
                'agent': record.agent,
                'parameters': record.parameters,
                'input_hash_prefix': record.input_hash[:8] + '...',
                'output_hash_prefix': record.output_hash[:8] + '...',
                'notes': record.notes
            })
            
        return lineage
        
    def export_audit_report(self, 
                            output_path: str,
                            document_ids: Optional[List[str]] = None) -> str:
        """
        Export provenance audit report.
        
        Args:
            output_path: Path to save report
            document_ids: Specific documents (None = all)
            
        Returns:
            Path to report file
        """
        if document_ids is None:
            document_ids = list(self.documents.keys())
            
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_documents': len(document_ids),
            'documents': []
        }
        
        for doc_id in document_ids:
            if doc_id in self.documents:
                provenance = self.documents[doc_id]
                report['documents'].append({
                    'document_id': doc_id,
                    'source': provenance.source_uri,
                    'ticker': provenance.ticker,
                    'filing_type': provenance.filing_type,
                    'filing_date': provenance.filing_date,
                    'chain_valid': provenance.verify_chain(),
                    'chain_length': len(provenance.chain),
                    'lineage': self.get_lineage(doc_id)
                })
                
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Exported audit report to {output_path}")
        
        return str(output)


# Usage
if __name__ == "__main__":
    # Initialize provenance chain
    chain = ProvenanceChain()
    
    # Simulate document registration
    sample_content = "Apple Inc. reported revenue of $94.8 billion..."
    
    provenance = chain.register_document(
        content=sample_content,
        source_uri="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=AAPL",
        source_type="sec_edgar",
        ticker="AAPL",
        filing_type="10-K",
        filing_date="20231103"
    )
    
    print(f"Document registered: {provenance.document_id}")
    
    # Simulate chunking transformation
    chunks = [sample_content[:20], sample_content[20:]]
    
    chain.record_transformation(
        document_id=provenance.document_id,
        transformation=TransformationType.CHUNK,
        input_content=sample_content,
        output_content=chunks,
        parameters={'chunk_size': 20, 'overlap': 0},
        agent='temporal_chunker'
    )
    
    # Get lineage
    lineage = chain.get_lineage(provenance.document_id)
    print("\nDocument Lineage:")
    for step in lineage:
        print(f"  {step['step']}. {step['transformation']} by {step['agent']}")
        
    # Verify document
    verification = chain.verify_document(provenance.document_id, chunks)
    print(f"\nVerification: {'✅ Valid' if verification['valid'] else '❌ Invalid'}")
