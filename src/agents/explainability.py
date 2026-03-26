"""
Explainable Agent Traces for TemporalGuard-RAG

This module provides transparency into agent decision-making for:
1. Compliance/audit requirements in financial applications
2. Debugging and system improvement
3. User trust through explainability

Key Features:
- Decision traces for each agent step
- Reasoning chain documentation
- Evidence linking (what data influenced each decision)
- Confidence justification

Research Contribution:
- Explainable multi-agent systems for financial compliance
- Decision provenance tracking across agent handoffs
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of agent decisions."""
    ROUTING = "routing"           # Query routed to specific agent
    FILTERING = "filtering"       # Data filtered (e.g., temporal)
    RETRIEVAL = "retrieval"       # Documents retrieved
    CALCULATION = "calculation"   # Financial calculation performed
    VERIFICATION = "verification" # Fact-checking decision
    PROJECTION = "projection"     # Future projection made
    SYNTHESIS = "synthesis"       # Final answer synthesized
    REJECTION = "rejection"       # Query/data rejected
    WARNING = "warning"           # Warning issued


class ConfidenceReason(Enum):
    """Reasons affecting confidence."""
    HIGH_DATA_QUALITY = "high_data_quality"
    LOW_DATA_QUALITY = "low_data_quality"
    MULTIPLE_SOURCES = "multiple_sources"
    SINGLE_SOURCE = "single_source"
    RECENT_DATA = "recent_data"
    STALE_DATA = "stale_data"
    EXACT_MATCH = "exact_match"
    APPROXIMATE_MATCH = "approximate_match"
    MODEL_AGREEMENT = "model_agreement"
    MODEL_DISAGREEMENT = "model_disagreement"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    TEMPORAL_VIOLATION = "temporal_violation"


@dataclass
class Evidence:
    """Evidence supporting a decision."""
    source_type: str  # "document", "xbrl", "calculation", "market_data"
    source_id: str    # Document ID, ticker, etc.
    content: str      # Relevant excerpt or value
    relevance_score: float = 1.0
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "content": self.content[:200] if len(self.content) > 200 else self.content,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp
        }


@dataclass
class DecisionStep:
    """A single decision step in the agent trace."""
    step_id: int
    agent_name: str
    decision_type: DecisionType
    
    # What was decided
    decision: str
    reasoning: str
    
    # Supporting information
    input_data: Dict = field(default_factory=dict)
    output_data: Dict = field(default_factory=dict)
    evidence: List[Evidence] = field(default_factory=list)
    
    # Confidence
    confidence: float = 0.0
    confidence_reasons: List[ConfidenceReason] = field(default_factory=list)
    
    # Timing
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_ms: float = 0.0
    
    # Alternative paths considered
    alternatives_considered: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "step_id": self.step_id,
            "agent": self.agent_name,
            "type": self.decision_type.value,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "confidence_reasons": [r.value for r in self.confidence_reasons],
            "evidence": [e.to_dict() for e in self.evidence],
            "duration_ms": self.duration_ms,
            "alternatives": self.alternatives_considered
        }


@dataclass
class AgentTrace:
    """Complete trace of agent decision-making for a query."""
    trace_id: str
    query: str
    ticker: Optional[str] = None
    analysis_date: Optional[str] = None
    
    steps: List[DecisionStep] = field(default_factory=list)
    
    # Aggregate stats
    total_duration_ms: float = 0.0
    overall_confidence: float = 0.0
    
    # Final outcome
    final_answer: str = ""
    success: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_used: str = ""
    
    def add_step(self, step: DecisionStep):
        """Add a decision step to the trace."""
        self.steps.append(step)
        self.total_duration_ms += step.duration_ms
        
        # Update overall confidence (weighted average)
        if self.steps:
            self.overall_confidence = sum(s.confidence for s in self.steps) / len(self.steps)
    
    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "ticker": self.ticker,
            "analysis_date": self.analysis_date,
            "steps": [s.to_dict() for s in self.steps],
            "total_duration_ms": self.total_duration_ms,
            "overall_confidence": self.overall_confidence,
            "final_answer": self.final_answer[:500] if len(self.final_answer) > 500 else self.final_answer,
            "success": self.success,
            "warnings": self.warnings,
            "errors": self.errors,
            "timestamp": self.timestamp,
            "model_used": self.model_used
        }
    
    def format_readable(self) -> str:
        """Generate human-readable trace."""
        lines = [
            "═" * 60,
            "📋 AGENT DECISION TRACE",
            "═" * 60,
            f"Query: {self.query}",
            f"Ticker: {self.ticker or 'N/A'}",
            f"Analysis Date: {self.analysis_date or 'N/A'}",
            f"Trace ID: {self.trace_id}",
            "",
            "─" * 60,
            "DECISION CHAIN:",
            "─" * 60
        ]
        
        for i, step in enumerate(self.steps, 1):
            emoji = {
                DecisionType.ROUTING: "🔀",
                DecisionType.FILTERING: "🔍",
                DecisionType.RETRIEVAL: "📄",
                DecisionType.CALCULATION: "🔢",
                DecisionType.VERIFICATION: "✓",
                DecisionType.PROJECTION: "📈",
                DecisionType.SYNTHESIS: "🎯",
                DecisionType.REJECTION: "❌",
                DecisionType.WARNING: "⚠️"
            }.get(step.decision_type, "•")
            
            lines.extend([
                f"\n{i}. {emoji} [{step.agent_name}] {step.decision_type.value.upper()}",
                f"   Decision: {step.decision}",
                f"   Reasoning: {step.reasoning}",
                f"   Confidence: {step.confidence:.0%}"
            ])
            
            if step.confidence_reasons:
                reasons = ", ".join(r.value.replace("_", " ") for r in step.confidence_reasons)
                lines.append(f"   Confidence factors: {reasons}")
            
            if step.evidence:
                lines.append(f"   Evidence ({len(step.evidence)} sources):")
                for ev in step.evidence[:3]:  # Show top 3
                    lines.append(f"     • [{ev.source_type}] {ev.content[:80]}...")
            
            if step.alternatives_considered:
                lines.append(f"   Alternatives considered: {', '.join(step.alternatives_considered)}")
            
            lines.append(f"   Time: {step.duration_ms:.0f}ms")
        
        lines.extend([
            "",
            "─" * 60,
            "SUMMARY:",
            "─" * 60,
            f"Total Steps: {len(self.steps)}",
            f"Total Time: {self.total_duration_ms:.0f}ms",
            f"Overall Confidence: {self.overall_confidence:.0%}",
            f"Status: {'✅ Success' if self.success else '❌ Failed'}"
        ])
        
        if self.warnings:
            lines.append(f"\n⚠️ Warnings:")
            for w in self.warnings:
                lines.append(f"   • {w}")
        
        if self.errors:
            lines.append(f"\n❌ Errors:")
            for e in self.errors:
                lines.append(f"   • {e}")
        
        lines.extend([
            "",
            "═" * 60
        ])
        
        return "\n".join(lines)


class TraceCollector:
    """
    Collects decision traces from agents.
    
    Usage:
        collector = TraceCollector()
        trace = collector.start_trace("What is Apple's revenue?", "AAPL")
        
        # As each agent makes decisions:
        collector.add_decision(
            agent_name="TemporalAgent",
            decision_type=DecisionType.FILTERING,
            decision="Query is temporally consistent",
            reasoning="Analysis date 2023-10 allows access to FY2022 data",
            confidence=0.95,
            evidence=[...]
        )
        
        trace = collector.end_trace(final_answer="...", success=True)
    """
    
    def __init__(self, save_dir: str = "data/traces"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_trace: Optional[AgentTrace] = None
        self.step_counter = 0
        self._step_start_time = None
        
        logger.info("Initialized Trace Collector")
    
    def start_trace(
        self,
        query: str,
        ticker: str = None,
        analysis_date: str = None,
        model_used: str = None
    ) -> AgentTrace:
        """Start a new trace for a query."""
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.current_trace = AgentTrace(
            trace_id=trace_id,
            query=query,
            ticker=ticker,
            analysis_date=analysis_date,
            model_used=model_used or "unknown"
        )
        
        self.step_counter = 0
        
        logger.debug(f"Started trace {trace_id}")
        return self.current_trace
    
    def start_step(self):
        """Mark the start of a decision step (for timing)."""
        self._step_start_time = datetime.now()
    
    def add_decision(
        self,
        agent_name: str,
        decision_type: DecisionType,
        decision: str,
        reasoning: str,
        confidence: float = 0.5,
        confidence_reasons: List[ConfidenceReason] = None,
        evidence: List[Evidence] = None,
        input_data: Dict = None,
        output_data: Dict = None,
        alternatives: List[str] = None
    ) -> DecisionStep:
        """Add a decision step to the current trace."""
        if not self.current_trace:
            raise ValueError("No active trace. Call start_trace() first.")
        
        self.step_counter += 1
        
        # Calculate duration
        duration_ms = 0
        if self._step_start_time:
            duration_ms = (datetime.now() - self._step_start_time).total_seconds() * 1000
            self._step_start_time = None
        
        step = DecisionStep(
            step_id=self.step_counter,
            agent_name=agent_name,
            decision_type=decision_type,
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
            confidence_reasons=confidence_reasons or [],
            evidence=evidence or [],
            input_data=input_data or {},
            output_data=output_data or {},
            alternatives_considered=alternatives or [],
            duration_ms=duration_ms
        )
        
        self.current_trace.add_step(step)
        
        logger.debug(f"Added step {self.step_counter}: {agent_name} - {decision_type.value}")
        return step
    
    def add_warning(self, warning: str):
        """Add a warning to the current trace."""
        if self.current_trace:
            self.current_trace.warnings.append(warning)
    
    def add_error(self, error: str):
        """Add an error to the current trace."""
        if self.current_trace:
            self.current_trace.errors.append(error)
    
    def end_trace(
        self,
        final_answer: str = "",
        success: bool = True,
        save: bool = True
    ) -> AgentTrace:
        """End the current trace and optionally save it."""
        if not self.current_trace:
            raise ValueError("No active trace to end.")
        
        self.current_trace.final_answer = final_answer
        self.current_trace.success = success
        
        trace = self.current_trace
        
        if save:
            self._save_trace(trace)
        
        self.current_trace = None
        self.step_counter = 0
        
        logger.debug(f"Ended trace {trace.trace_id}")
        return trace
    
    def _save_trace(self, trace: AgentTrace):
        """Save trace to file."""
        filename = f"{trace.trace_id}.json"
        filepath = self.save_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(trace.to_dict(), f, indent=2)
        
        logger.debug(f"Saved trace to {filepath}")
    
    def get_current_trace(self) -> Optional[AgentTrace]:
        """Get the current active trace."""
        return self.current_trace


class TraceAnalyzer:
    """
    Analyzes collected traces for patterns and insights.
    """
    
    def __init__(self, traces_dir: str = "data/traces"):
        self.traces_dir = Path(traces_dir)
    
    def load_traces(self, limit: int = 100) -> List[Dict]:
        """Load recent traces from disk."""
        traces = []
        
        for filepath in sorted(self.traces_dir.glob("trace_*.json"), reverse=True)[:limit]:
            with open(filepath) as f:
                traces.append(json.load(f))
        
        return traces
    
    def analyze_confidence_distribution(self, traces: List[Dict] = None) -> Dict:
        """Analyze confidence score distribution across traces."""
        if traces is None:
            traces = self.load_traces()
        
        confidences = [t.get("overall_confidence", 0) for t in traces]
        
        if not confidences:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        import numpy as np
        return {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "n_traces": len(traces)
        }
    
    def analyze_agent_usage(self, traces: List[Dict] = None) -> Dict[str, int]:
        """Count how often each agent is used."""
        if traces is None:
            traces = self.load_traces()
        
        agent_counts = {}
        
        for trace in traces:
            for step in trace.get("steps", []):
                agent = step.get("agent", "unknown")
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        return agent_counts
    
    def find_low_confidence_patterns(self, traces: List[Dict] = None, threshold: float = 0.5) -> List[Dict]:
        """Find common patterns in low-confidence decisions."""
        if traces is None:
            traces = self.load_traces()
        
        low_confidence_decisions = []
        
        for trace in traces:
            for step in trace.get("steps", []):
                if step.get("confidence", 1.0) < threshold:
                    low_confidence_decisions.append({
                        "trace_id": trace.get("trace_id"),
                        "agent": step.get("agent"),
                        "type": step.get("type"),
                        "decision": step.get("decision"),
                        "reasoning": step.get("reasoning"),
                        "confidence": step.get("confidence"),
                        "reasons": step.get("confidence_reasons", [])
                    })
        
        return low_confidence_decisions
