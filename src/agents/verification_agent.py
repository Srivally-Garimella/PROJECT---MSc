"""
Verification Agent for TemporalGuard-RAG

Specialized agent for fact-checking financial claims against source documents.
Ensures accuracy and provides confidence scores for generated content.
"""

from typing import Dict, List, Optional
from datetime import datetime
import json
import logging
import os
import re

from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import warnings

from .llm_provider import get_llm

# Suppress deprecation warnings for langgraph
warnings.filterwarnings('ignore', category=DeprecationWarning, module='langgraph')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationAgent:
    """
    Financial fact-checking agent.
    
    Key Responsibilities:
    - Verify claims against source documents
    - Cross-check numbers with XBRL data
    - Detect hallucinations
    - Provide confidence scores
    """
    
    SYSTEM_PROMPT = """You are a financial fact-checking specialist responsible for verifying claims against authoritative sources.

Your job is to ensure accuracy and prevent hallucinations in financial analysis.

CRITICAL RULES:
1. NEVER accept claims without verification
2. Always cite specific source documents
3. Cross-check narrative claims against structured XBRL data when possible
4. Flag any discrepancies between narrative and structured data
5. Assign confidence scores: HIGH (direct source match), MEDIUM (indirect support), LOW (limited evidence)
6. Report temporal inconsistencies (claims citing future data)

Verification Process:
1. Identify the key claims to verify
2. Search for supporting evidence
3. Cross-reference with structured data
4. Check temporal consistency
5. Assign confidence score
6. Note any caveats or limitations

When reporting:
- State the claim being verified
- Provide supporting/contradicting evidence
- Cite specific documents with dates
- Give confidence score and reasoning
"""

    def __init__(self, 
                 vector_store=None,
                 xbrl_dir: str = "data/raw/xbrl_structured",
                 model_name: str = None,
                 provider: str = None):
        """
        Initialize Verification Agent.
        
        Args:
            vector_store: TemporalVectorStore for document search
            xbrl_dir: Directory with XBRL structured data
            model_name: LLM model name (default: auto-select)
            provider: 'openai' or 'ollama' (default: auto-detect)
        """
        self.vector_store = vector_store
        self.xbrl_dir = xbrl_dir
        self.model_name = model_name
        self.provider = provider
        
        # Initialize LLM using provider factory
        self.llm = get_llm(provider=provider, model_name=model_name, temperature=0)
        
        # Create tools using decorator (langgraph style)
        @tool
        def verify_claim(query_string: str) -> str:
            """Verify a financial claim against source documents.
            Input: claim|ticker|filing_date
            Example: "Revenue increased 15% in Q2 2023|AAPL|20230630"
            Returns verification result with confidence score.
            """
            return self._verify_claim(query_string)
        
        @tool
        def cross_check_number(query_string: str) -> str:
            """Cross-check a specific number against XBRL structured data.
            Input: ticker|metric|claimed_value|date
            Example: AAPL|Revenue|94836000000|20230331
            """
            return self._cross_check_number(query_string)
        
        @tool
        def check_temporal_consistency(query_string: str) -> str:
            """Check if a claim is temporally consistent.
            Input: claim|reference_date|cutoff_date
            Detects if claim references future information.
            """
            return self._check_temporal_consistency(query_string)
        
        self.tools = [verify_claim, cross_check_number, check_temporal_consistency]
        
        # Create ReAct agent using langgraph (works with any LLM)
        try:
            self.agent = create_react_agent(
                self.llm, 
                self.tools,
                prompt=self.SYSTEM_PROMPT
            )
            logger.info(f"Initialized Verification Agent")
        except Exception as e:
            logger.warning(f"Could not create agent: {e}")
            self.agent = None
        
    def _verify_claim(self, query_string: str) -> str:
        """Verify a claim against source documents."""
        parts = query_string.split('|')
        
        if len(parts) != 3:
            return "Error: Input must be claim|ticker|filing_date"
            
        claim, ticker, filing_date = [p.strip() for p in parts]
        
        # Search for supporting evidence
        if self.vector_store:
            try:
                results = self.vector_store.temporal_search(
                    query=claim,
                    cutoff_date=filing_date,
                    ticker=ticker,
                    n_results=3
                )
                
                return self._format_verification_result(claim, ticker, filing_date, results)
                
            except Exception as e:
                return f"Error searching for evidence: {e}"
        else:
            return self._mock_verification(claim, ticker, filing_date)
            
    def _format_verification_result(self, claim: str, ticker: str, 
                                     filing_date: str, results: Dict) -> str:
        """Format verification results."""
        if not results.get('documents') or not results['documents'][0]:
            return f"""VERIFICATION RESULT:
Claim: "{claim}"
Company: {ticker}
Status: ⚠️ UNVERIFIED - No supporting documents found
Confidence: LOW
Reason: No relevant documents found before {filing_date}"""
            
        # Analyze supporting documents
        supporting_docs = []
        for i, (doc, meta) in enumerate(zip(
            results['documents'][0][:3],
            results['metadatas'][0][:3]
        )):
            supporting_docs.append({
                'content': doc[:300],
                'filing_date': meta.get('filing_date'),
                'filing_type': meta.get('filing_type')
            })
            
        # Determine confidence based on match quality
        confidence = "MEDIUM"  # Default
        
        # Check if claim appears directly in documents
        claim_keywords = claim.lower().split()
        doc_text = ' '.join([d['content'].lower() for d in supporting_docs])
        keyword_matches = sum(1 for kw in claim_keywords if kw in doc_text)
        
        if keyword_matches > len(claim_keywords) * 0.5:
            confidence = "HIGH"
        elif keyword_matches < len(claim_keywords) * 0.2:
            confidence = "LOW"
            
        return f"""VERIFICATION RESULT:
Claim: "{claim}"
Company: {ticker}
Cutoff Date: {filing_date}

Status: {'✅ SUPPORTED' if confidence != 'LOW' else '⚠️ WEAK SUPPORT'}
Confidence: {confidence}

Supporting Evidence:
{self._format_supporting_docs(supporting_docs)}

Analysis: Found {len(supporting_docs)} relevant documents filed before {filing_date}.
{'Strong textual support for the claim.' if confidence == 'HIGH' else 'Some relevant context found but direct confirmation limited.' if confidence == 'MEDIUM' else 'Limited evidence - claim may need additional verification.'}"""
            
    def _format_supporting_docs(self, docs: List[Dict]) -> str:
        """Format supporting documents."""
        formatted = []
        for i, doc in enumerate(docs):
            formatted.append(f"""
Document {i+1}:
- Filed: {doc['filing_date']}
- Type: {doc['filing_type']}
- Excerpt: "{doc['content'][:200]}..."
""")
        return "\n".join(formatted)
        
    def _mock_verification(self, claim: str, ticker: str, filing_date: str) -> str:
        """Mock verification for testing."""
        return f"""[MOCK VERIFICATION]
Claim: "{claim}"
Company: {ticker}
Cutoff Date: {filing_date}

Status: ⚠️ MOCK - Vector store not connected
Confidence: N/A

Note: Connect vector store for actual verification.

Mock Analysis:
- The claim appears reasonable based on general knowledge
- Specific document verification requires vector store
- Cross-check numerical claims with XBRL data"""
            
    def _cross_check_number(self, query_string: str) -> str:
        """Cross-check numerical claim against XBRL data."""
        parts = query_string.split('|')
        
        if len(parts) != 4:
            return "Error: Input must be ticker|metric|claimed_value|date"
            
        ticker, metric, claimed_value, date = [p.strip() for p in parts]
        
        try:
            claimed_num = float(claimed_value.replace(',', ''))
        except ValueError:
            return f"Error: Could not parse claimed value: {claimed_value}"
            
        # Load XBRL data
        from pathlib import Path
        import pandas as pd
        
        metrics_path = Path(self.xbrl_dir) / f"{ticker.upper()}_metrics.csv"
        
        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                df['end_date'] = pd.to_datetime(df['end_date'])
                
                target_date = pd.to_datetime(date)
                
                # Find closest value
                filtered = df[
                    (df['metric'] == metric) & 
                    (df['end_date'] <= target_date)
                ].sort_values('end_date', ascending=False)
                
                if filtered.empty:
                    return f"No {metric} data found for {ticker} in XBRL"
                    
                actual = filtered.iloc[0]
                actual_value = actual['value']
                
                # Calculate variance
                variance = abs(claimed_num - actual_value) / actual_value * 100 if actual_value != 0 else 100
                
                match_status = "✅ MATCH" if variance < 1 else "⚠️ CLOSE" if variance < 5 else "❌ MISMATCH"
                
                return f"""NUMERICAL CROSS-CHECK:
Metric: {metric}
Company: {ticker}
As of: {date}

Claimed Value: ${claimed_num:,.0f}
XBRL Value: ${actual_value:,.0f}
Variance: {variance:.2f}%

Status: {match_status}
Source: XBRL filing dated {actual['end_date']}"""
                
            except Exception as e:
                return f"Error reading XBRL data: {e}"
        else:
            return f"""[MOCK CROSS-CHECK]
Metric: {metric}
Company: {ticker}
Claimed Value: ${float(claimed_value):,.0f}

Status: ⚠️ XBRL data not available
Note: Download XBRL data for actual cross-checking"""
            
    def _check_temporal_consistency(self, query_string: str) -> str:
        """Check temporal consistency of a claim."""
        parts = query_string.split('|')
        
        if len(parts) != 3:
            return "Error: Input must be claim|reference_date|cutoff_date"
            
        claim, reference_date, cutoff_date = [p.strip() for p in parts]
        
        # Extract dates from claim
        date_patterns = [
            r'20\d{2}',  # Year
            r'Q[1-4]\s*20\d{2}',  # Quarter
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*20\d{2}',  # Month year
        ]
        
        found_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, claim, re.IGNORECASE)
            found_dates.extend(matches)
            
        issues = []
        
        # Check if any found dates are after cutoff
        try:
            cutoff_year = int(cutoff_date[:4])
            for date_str in found_dates:
                year_match = re.search(r'20\d{2}', date_str)
                if year_match:
                    year = int(year_match.group())
                    if year > cutoff_year:
                        issues.append(f"Reference to {date_str} is after cutoff year {cutoff_year}")
        except:
            pass
            
        if issues:
            return f"""TEMPORAL CONSISTENCY CHECK:
Claim: "{claim}"
Cutoff Date: {cutoff_date}

Status: ❌ TEMPORAL VIOLATION DETECTED

Issues:
{chr(10).join('- ' + issue for issue in issues)}

This claim may contain look-ahead bias!"""
        else:
            return f"""TEMPORAL CONSISTENCY CHECK:
Claim: "{claim}"
Cutoff Date: {cutoff_date}

Status: ✅ TEMPORALLY CONSISTENT
No references to future dates detected."""
            
    def verify(self, claim: str, ticker: str, filing_date: str, fast_mode: bool = False) -> Dict:
        """
        Main verification method.
        
        Args:
            claim: Claim to verify
            ticker: Company ticker
            filing_date: Filing date cutoff
            fast_mode: Skip slow LLM verification, use fast fallback
            
        Returns:
            Verification results dictionary
        """
        start_time = datetime.now()
        
        # Fast mode: skip slow LLM agent, use direct verification
        if fast_mode or self.agent is None:
            verification_result = self._verify_claim(f"{claim}|{ticker}|{filing_date}")
            return {
                'output': verification_result,
                'claim': claim,
                'ticker': ticker,
                'filing_date': filing_date,
                'agent': 'VerificationAgent',
                'mode': 'fast',
                'timestamp': datetime.now().isoformat()
            }
        
        if self.agent:
            try:
                input_message = f"""Verify this claim about {ticker}: "{claim}"

Use documents filed on or before {filing_date}.
Provide confidence score and cite sources."""

                result = self.agent.invoke({
                    "messages": [HumanMessage(content=input_message)]
                })
                
                # Extract output from messages
                output = result.get('messages', [])[-1].content if result.get('messages') else ''
                
                return {
                    'output': output,
                    'claim': claim,
                    'ticker': ticker,
                    'filing_date': filing_date,
                    'agent': 'VerificationAgent',
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Agent error: {e}")
                
        # Fallback
        verification_result = self._verify_claim(f"{claim}|{ticker}|{filing_date}")
        
        return {
            'output': verification_result,
            'claim': claim,
            'ticker': ticker,
            'filing_date': filing_date,
            'agent': 'VerificationAgent',
            'mode': 'fallback',
            'timestamp': datetime.now().isoformat()
        }


# Usage
if __name__ == "__main__":
    # Initialize agent
    agent = VerificationAgent()
    
    # Test verification
    result = agent.verify(
        claim="Apple's revenue increased significantly in fiscal 2023",
        ticker="AAPL",
        filing_date="20231001"
    )
    
    print("\nVerification Result:")
    print("=" * 60)
    print(result['output'])
    print("=" * 60)
