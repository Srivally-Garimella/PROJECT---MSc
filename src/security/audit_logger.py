"""
Audit Logger for TemporalGuard-RAG

Comprehensive audit logging for compliance, debugging, and security monitoring.
Records all system actions with timestamps and user context.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
from pathlib import Path
import threading
import uuid
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of audit events."""
    # Data Operations
    DATA_DOWNLOAD = "data_download"
    DATA_PROCESS = "data_process"
    DATA_INDEX = "data_index"
    DATA_DELETE = "data_delete"
    
    # Query Operations
    QUERY_RECEIVED = "query_received"
    QUERY_VALIDATED = "query_validated"
    QUERY_REJECTED = "query_rejected"
    
    # Retrieval Operations
    RETRIEVAL_START = "retrieval_start"
    RETRIEVAL_COMPLETE = "retrieval_complete"
    TEMPORAL_FILTER = "temporal_filter"
    
    # Agent Operations
    AGENT_INVOKED = "agent_invoked"
    AGENT_RESPONSE = "agent_response"
    AGENT_ERROR = "agent_error"
    
    # Security Events
    SECURITY_ALERT = "security_alert"
    ACCESS_DENIED = "access_denied"
    ANOMALY_DETECTED = "anomaly_detected"
    
    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    ERROR = "error"


class EventSeverity(Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Single audit event record."""
    event_id: str
    event_type: EventType
    severity: EventSeverity
    timestamp: str
    component: str  # Which system component
    action: str     # What action was taken
    details: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp,
            'component': self.component,
            'action': self.action,
            'details': self.details,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'ip_address': self.ip_address,
            'duration_ms': self.duration_ms,
            'success': self.success,
            'error_message': self.error_message
        }
        
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AuditLogger:
    """
    Comprehensive audit logging system.
    
    Features:
    - Thread-safe logging
    - Rotating log files
    - Searchable event storage
    - Compliance report generation
    - Real-time alerts for critical events
    """
    
    def __init__(self,
                 log_dir: str = "data/audit",
                 max_file_size_mb: int = 100,
                 retention_days: int = 90,
                 enable_console: bool = True):
        """
        Initialize audit logger.
        
        Args:
            log_dir: Directory for audit logs
            max_file_size_mb: Max size per log file before rotation
            retention_days: Days to retain logs
            enable_console: Also log to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.retention_days = retention_days
        self.enable_console = enable_console
        
        self._lock = threading.Lock()
        self._current_file = None
        self._current_file_size = 0
        
        # In-memory buffer for recent events
        self._event_buffer: List[AuditEvent] = []
        self._buffer_max_size = 1000
        
        # Alert callbacks for critical events
        self._alert_callbacks = []
        
        self._init_log_file()
        self._cleanup_old_logs()
        
        logger.info(f"Audit logger initialized: {self.log_dir}")
        
    def _init_log_file(self):
        """Initialize current log file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._current_file = self.log_dir / f"audit_{timestamp}.jsonl"
        self._current_file_size = 0
        
    def _cleanup_old_logs(self):
        """Remove logs older than retention period."""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        
        for log_file in self.log_dir.glob("audit_*.jsonl"):
            try:
                # Parse timestamp from filename
                name = log_file.stem
                date_str = name.replace('audit_', '')[:8]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                
                if file_date < cutoff:
                    log_file.unlink()
                    logger.info(f"Cleaned up old log: {log_file.name}")
            except (ValueError, OSError) as e:
                logger.warning(f"Could not process log file {log_file}: {e}")
                
    def _rotate_if_needed(self):
        """Rotate log file if size exceeded."""
        if self._current_file_size >= self.max_file_size:
            self._init_log_file()
            
    def log(self,
            event_type: EventType,
            component: str,
            action: str,
            details: Dict[str, Any],
            severity: EventSeverity = EventSeverity.INFO,
            user_id: Optional[str] = None,
            session_id: Optional[str] = None,
            request_id: Optional[str] = None,
            ip_address: Optional[str] = None,
            duration_ms: Optional[float] = None,
            success: bool = True,
            error_message: Optional[str] = None) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            component: System component
            action: Description of action
            details: Event details
            severity: Event severity
            user_id: User identifier
            session_id: Session identifier
            request_id: Request identifier
            ip_address: Client IP
            duration_ms: Operation duration
            success: Whether operation succeeded
            error_message: Error message if failed
            
        Returns:
            Created AuditEvent
        """
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            component=component,
            action=action,
            details=details,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            ip_address=ip_address,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message
        )
        
        with self._lock:
            self._write_event(event)
            self._buffer_event(event)
            
        # Console logging
        if self.enable_console:
            log_msg = f"[{event.severity.value.upper()}] {component}/{action}"
            if severity == EventSeverity.CRITICAL:
                logger.critical(log_msg)
            elif severity == EventSeverity.ERROR:
                logger.error(log_msg)
            elif severity == EventSeverity.WARNING:
                logger.warning(log_msg)
            else:
                logger.info(log_msg)
                
        # Trigger alerts for critical events
        if severity in [EventSeverity.CRITICAL, EventSeverity.ERROR]:
            self._trigger_alerts(event)
            
        return event
        
    def _write_event(self, event: AuditEvent):
        """Write event to log file."""
        self._rotate_if_needed()
        
        try:
            line = event.to_json() + '\n'
            with open(self._current_file, 'a') as f:
                f.write(line)
            self._current_file_size += len(line.encode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to write audit event: {e}")
            
    def _buffer_event(self, event: AuditEvent):
        """Add event to in-memory buffer."""
        self._event_buffer.append(event)
        
        # Trim buffer if too large
        if len(self._event_buffer) > self._buffer_max_size:
            self._event_buffer = self._event_buffer[-self._buffer_max_size:]
            
    def _trigger_alerts(self, event: AuditEvent):
        """Trigger alert callbacks for critical events."""
        for callback in self._alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
                
    def register_alert_callback(self, callback):
        """Register callback for critical event alerts."""
        self._alert_callbacks.append(callback)
        
    # ═══════════════════════════════════════════════════════════════
    # Convenience Methods
    # ═══════════════════════════════════════════════════════════════
    
    def log_query(self,
                  query: str,
                  analysis_date: str,
                  ticker: str,
                  user_id: Optional[str] = None,
                  request_id: Optional[str] = None) -> AuditEvent:
        """Log a query event."""
        return self.log(
            event_type=EventType.QUERY_RECEIVED,
            component='query_processor',
            action='query_received',
            details={
                'query': query,
                'analysis_date': analysis_date,
                'ticker': ticker,
                'query_hash': hashlib.md5(query.encode()).hexdigest()[:8]
            },
            user_id=user_id,
            request_id=request_id
        )
        
    def log_retrieval(self,
                      query: str,
                      num_docs: int,
                      duration_ms: float,
                      cutoff_date: str,
                      request_id: Optional[str] = None) -> AuditEvent:
        """Log a retrieval event."""
        return self.log(
            event_type=EventType.RETRIEVAL_COMPLETE,
            component='retriever',
            action='temporal_retrieval',
            details={
                'query_hash': hashlib.md5(query.encode()).hexdigest()[:8],
                'documents_retrieved': num_docs,
                'cutoff_date': cutoff_date
            },
            duration_ms=duration_ms,
            request_id=request_id
        )
        
    def log_agent_invocation(self,
                              agent_name: str,
                              input_summary: str,
                              request_id: Optional[str] = None) -> AuditEvent:
        """Log agent invocation."""
        return self.log(
            event_type=EventType.AGENT_INVOKED,
            component=f'agent_{agent_name}',
            action='agent_invoked',
            details={
                'agent': agent_name,
                'input_summary': input_summary[:200]
            },
            request_id=request_id
        )
        
    def log_security_alert(self,
                           alert_type: str,
                           description: str,
                           evidence: Dict[str, Any],
                           severity: EventSeverity = EventSeverity.WARNING) -> AuditEvent:
        """Log security alert."""
        return self.log(
            event_type=EventType.SECURITY_ALERT,
            component='security',
            action=alert_type,
            details={
                'description': description,
                'evidence': evidence
            },
            severity=severity,
            success=False
        )
        
    def log_error(self,
                  component: str,
                  error_message: str,
                  exception: Optional[Exception] = None,
                  request_id: Optional[str] = None) -> AuditEvent:
        """Log an error."""
        details = {
            'message': error_message
        }
        if exception:
            details['exception_type'] = type(exception).__name__
            details['exception_str'] = str(exception)
            
        return self.log(
            event_type=EventType.ERROR,
            component=component,
            action='error',
            details=details,
            severity=EventSeverity.ERROR,
            success=False,
            error_message=error_message,
            request_id=request_id
        )
        
    # ═══════════════════════════════════════════════════════════════
    # Query and Reporting
    # ═══════════════════════════════════════════════════════════════
    
    def get_recent_events(self,
                          limit: int = 100,
                          event_type: Optional[EventType] = None,
                          severity: Optional[EventSeverity] = None,
                          component: Optional[str] = None) -> List[AuditEvent]:
        """Get recent events from buffer with optional filtering."""
        events = self._event_buffer
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if severity:
            events = [e for e in events if e.severity == severity]
        if component:
            events = [e for e in events if e.component == component]
            
        return events[-limit:]
        
    def search_events(self,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      event_types: Optional[List[EventType]] = None,
                      user_id: Optional[str] = None,
                      component: Optional[str] = None) -> List[AuditEvent]:
        """Search events across log files."""
        events = []
        
        # Parse date range
        start_dt = datetime.fromisoformat(start_date) if start_date else datetime.min
        end_dt = datetime.fromisoformat(end_date) if end_date else datetime.max
        
        for log_file in sorted(self.log_dir.glob("audit_*.jsonl")):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            event_dt = datetime.fromisoformat(data['timestamp'])
                            
                            # Date filter
                            if not (start_dt <= event_dt <= end_dt):
                                continue
                                
                            # Type filter
                            if event_types and data['event_type'] not in [e.value for e in event_types]:
                                continue
                                
                            # User filter
                            if user_id and data.get('user_id') != user_id:
                                continue
                                
                            # Component filter
                            if component and data.get('component') != component:
                                continue
                                
                            events.append(self._dict_to_event(data))
                            
                        except (json.JSONDecodeError, KeyError):
                            continue
                            
            except Exception as e:
                logger.warning(f"Could not read log file {log_file}: {e}")
                
        return events
        
    def _dict_to_event(self, data: Dict) -> AuditEvent:
        """Convert dictionary to AuditEvent."""
        return AuditEvent(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            severity=EventSeverity(data['severity']),
            timestamp=data['timestamp'],
            component=data['component'],
            action=data['action'],
            details=data['details'],
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            request_id=data.get('request_id'),
            ip_address=data.get('ip_address'),
            duration_ms=data.get('duration_ms'),
            success=data.get('success', True),
            error_message=data.get('error_message')
        )
        
    def generate_compliance_report(self,
                                    start_date: str,
                                    end_date: str,
                                    output_path: str) -> str:
        """
        Generate compliance report for date range.
        
        Args:
            start_date: Report start date (ISO format)
            end_date: Report end date (ISO format)
            output_path: Path to save report
            
        Returns:
            Path to generated report
        """
        events = self.search_events(start_date=start_date, end_date=end_date)
        
        # Aggregate statistics
        stats = {
            'report_period': {'start': start_date, 'end': end_date},
            'total_events': len(events),
            'events_by_type': {},
            'events_by_severity': {},
            'events_by_component': {},
            'errors': [],
            'security_alerts': []
        }
        
        for event in events:
            # By type
            type_key = event.event_type.value
            stats['events_by_type'][type_key] = stats['events_by_type'].get(type_key, 0) + 1
            
            # By severity
            sev_key = event.severity.value
            stats['events_by_severity'][sev_key] = stats['events_by_severity'].get(sev_key, 0) + 1
            
            # By component
            stats['events_by_component'][event.component] = stats['events_by_component'].get(event.component, 0) + 1
            
            # Collect errors and security alerts
            if event.event_type == EventType.ERROR:
                stats['errors'].append({
                    'timestamp': event.timestamp,
                    'component': event.component,
                    'message': event.error_message
                })
            elif event.event_type == EventType.SECURITY_ALERT:
                stats['security_alerts'].append({
                    'timestamp': event.timestamp,
                    'description': event.details.get('description', '')
                })
                
        # Generate report
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': stats,
            'summary': {
                'total_queries': stats['events_by_type'].get('query_received', 0),
                'total_retrievals': stats['events_by_type'].get('retrieval_complete', 0),
                'total_errors': len(stats['errors']),
                'security_incidents': len(stats['security_alerts'])
            }
        }
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Generated compliance report: {output_path}")
        
        return str(output)


# Usage
if __name__ == "__main__":
    # Initialize audit logger
    audit = AuditLogger()
    
    # Log some events
    audit.log(
        event_type=EventType.SYSTEM_START,
        component='main',
        action='system_initialized',
        details={'version': '1.0.0'}
    )
    
    request_id = str(uuid.uuid4())
    
    audit.log_query(
        query="What was Apple's revenue in Q2 2023?",
        analysis_date="20230701",
        ticker="AAPL",
        user_id="user_001",
        request_id=request_id
    )
    
    audit.log_retrieval(
        query="What was Apple's revenue in Q2 2023?",
        num_docs=5,
        duration_ms=234.5,
        cutoff_date="20230601",
        request_id=request_id
    )
    
    audit.log_agent_invocation(
        agent_name="temporal",
        input_summary="Validating query temporal consistency",
        request_id=request_id
    )
    
    # Get recent events
    recent = audit.get_recent_events(limit=10)
    print(f"\nRecent events: {len(recent)}")
    for event in recent:
        print(f"  [{event.severity.value}] {event.component}/{event.action}")
