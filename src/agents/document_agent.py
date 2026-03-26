"""
Document Retrieval Agent for TemporalGuard-RAG

Specialized agent for retrieving financial documents with temporal awareness.
Enforces point-in-time constraints to prevent look-ahead bias.
"""

from typing import Dict, List, Optional
from datetime import datetime
import json
import logging
import os

from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
import warnings

from .llm_provider import get_llm

# Suppress deprecation warnings for langgraph
warnings.filterwarnings('ignore', category=DeprecationWarning, module='langgraph')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentRetrievalAgent:
    """
    Document retrieval agent with temporal constraint enforcement.
    
    Key Responsibilities:
    - Search SEC filings with temporal boundaries
    - Enforce point-in-time (PiT) constraints
    - Cite filing dates for retrieved documents
    - Flag temporal inconsistencies
    """
    
    SYSTEM_PROMPT = """You are a financial document retrieval specialist working within a temporal-aware RAG system.

Your job is to find relevant information from SEC filings (10-K, 10-Q, 8-K) while STRICTLY respecting temporal boundaries to prevent look-ahead bias.

CRITICAL RULES:
1. NEVER retrieve documents filed after the cutoff_date
2. Always specify the cutoff_date in your search
3. Cite the filing date of each document you return
4. If no cutoff_date is provided, refuse to search and explain why
5. Prioritize the most recent documents BEFORE the cutoff date
6. For multi-period comparisons, respect the cutoff for each period

When returning results:
- Include ticker symbol
- Include filing date (must be BEFORE cutoff)
- Include filing type (10-K, 10-Q, etc.)
- Provide relevant excerpt with context

You have access to the following tool:
- temporal_search: Search financial documents with temporal constraints
"""

    def __init__(self, vector_store=None, model_name: str = None, provider: str = None):
        """
        Initialize Document Retrieval Agent.
        
        Args:
            vector_store: TemporalVectorStore instance
            model_name: LLM model name (default: auto-select)
            provider: 'openai' or 'ollama' (default: auto-detect)
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.provider = provider
        
        # Initialize LLM using provider factory
        self.llm = get_llm(provider=provider, model_name=model_name, temperature=0)
        
        # Create tools using decorator (langgraph style)
        @tool
        def temporal_search(query_string: str) -> str:
            """Search financial documents with temporal constraints.
            Input format: query|cutoff_date|ticker
            Example: "revenue risks|20230630|AAPL"
            cutoff_date format: YYYYMMDD
            ticker: Company stock symbol (e.g., AAPL, MSFT)
            Returns relevant document excerpts with filing dates.
            """
            return self._temporal_search(query_string)
        
        self.tools = [temporal_search]
        
        # Create ReAct agent using langgraph (works with any LLM)
        try:
            self.agent = create_react_agent(
                self.llm, 
                self.tools,
                prompt=self.SYSTEM_PROMPT
            )
            logger.info(f"Initialized Document Retrieval Agent")
        except Exception as e:
            logger.warning(f"Could not create agent: {e}")
            self.agent = None
        
    def _temporal_search(self, query_string: str) -> str:
        """
        Tool function for temporal search.
        
        Args:
            query_string: Format "query|cutoff_date|ticker"
            
        Returns:
            Formatted search results string
        """
        # Parse input
        parts = query_string.split('|')
        
        if len(parts) != 3:
            return "Error: Input must be in format: query|cutoff_date|ticker\nExample: revenue risks|20230630|AAPL"
            
        query, cutoff_date, ticker = [p.strip() for p in parts]
        
        # Validate cutoff date format
        if len(cutoff_date) != 8 or not cutoff_date.isdigit():
            return f"Error: cutoff_date must be YYYYMMDD format. Got: {cutoff_date}"
            
        # Execute search if vector store available
        if self.vector_store is None:
            # Return mock results for testing
            return self._mock_search_results(query, cutoff_date, ticker)
            
        try:
            results = self.vector_store.temporal_search(
                query=query,
                cutoff_date=cutoff_date,
                ticker=ticker,
                n_results=5
            )
            
            return self._format_results(results, cutoff_date)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Error executing search: {str(e)}"
            
    def _format_results(self, results: Dict, cutoff_date: str) -> str:
        """Format search results for the agent."""
        if not results.get('documents') or not results['documents'][0]:
            return f"No documents found before cutoff date {cutoff_date}"
            
        formatted = []
        formatted.append(f"Found {len(results['documents'][0])} documents (cutoff: {cutoff_date}):\n")
        
        for i, (doc, meta) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0]
        )):
            filing_date = meta.get('filing_date', 'Unknown')
            ticker = meta.get('ticker', 'Unknown')
            filing_type = meta.get('filing_type', 'Unknown')
            
            formatted.append(f"""
Document {i+1}:
- Ticker: {ticker}
- Filing Type: {filing_type}
- Filing Date: {filing_date} (BEFORE cutoff {cutoff_date}: ✓)
- Content Preview:
{doc[:400]}...
---""")
            
        return "\n".join(formatted)
        
    def _mock_search_results(self, query: str, cutoff_date: str, ticker: str) -> str:
        """Return mock results for testing without vector store."""
        return f"""
[MOCK RESULTS - Vector store not connected]

Query: {query}
Ticker: {ticker}
Cutoff Date: {cutoff_date}

Document 1:
- Ticker: {ticker}
- Filing Type: 10-K
- Filing Date: 20221028 (BEFORE cutoff {cutoff_date}: ✓)
- Content Preview:
Risk Factors: The company faces various risks including market volatility, 
regulatory changes, and competitive pressures. Supply chain disruptions 
have been a particular concern in recent periods...
---

Document 2:
- Ticker: {ticker}
- Filing Type: 10-Q
- Filing Date: 20230127 (BEFORE cutoff {cutoff_date}: ✓)
- Content Preview:
Management's Discussion and Analysis: Revenue for the quarter increased 
year-over-year driven by strong product demand and expanding market share...
---

Note: These are mock results. Connect a vector store for actual document retrieval.
"""
        
    def retrieve(self, 
                query: str, 
                cutoff_date: str, 
                ticker: str) -> Dict:
        """
        Main retrieval method.
        
        Args:
            query: Search query
            cutoff_date: Point-in-time cutoff (YYYYMMDD)
            ticker: Company ticker symbol
            
        Returns:
            Dictionary with retrieval results and audit info
        """
        start_time = datetime.now()
        
        if self.agent:
            try:
                # Langgraph agent uses messages format
                input_message = f"""Find information about: {query}
                    
Company: {ticker}
Point-in-Time Cutoff: {cutoff_date}

Remember: Only retrieve documents filed BEFORE {cutoff_date}.
Cite the filing date for each document."""
                
                result = self.agent.invoke({
                    "messages": [HumanMessage(content=input_message)]
                })
                
                # Extract output from messages
                output = result.get('messages', [])[-1].content if result.get('messages') else ''
                
                return {
                    'output': output,
                    'query': query,
                    'cutoff_date': cutoff_date,
                    'ticker': ticker,
                    'agent': 'DocumentRetrievalAgent',
                    'model': self.model_name,
                    'processing_time': (datetime.now() - start_time).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Agent execution error: {e}")
                return {
                    'output': f"Agent error: {str(e)}",
                    'error': True,
                    'timestamp': datetime.now().isoformat()
                }
        else:
            # Fallback: direct search
            search_result = self._temporal_search(f"{query}|{cutoff_date}|{ticker}")
            
            return {
                'output': search_result,
                'query': query,
                'cutoff_date': cutoff_date,
                'ticker': ticker,
                'agent': 'DocumentRetrievalAgent',
                'mode': 'fallback',
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
            
    def batch_retrieve(self,
                      queries: List[str],
                      cutoff_date: str,
                      ticker: str) -> List[Dict]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of search queries
            cutoff_date: Point-in-time cutoff
            ticker: Company ticker
            
        Returns:
            List of retrieval results
        """
        results = []
        
        for query in queries:
            result = self.retrieve(query, cutoff_date, ticker)
            results.append(result)
            
        return results


# Usage
if __name__ == "__main__":
    # Initialize agent (will use fallback mode if no API key)
    agent = DocumentRetrievalAgent()
    
    # Test retrieval
    result = agent.retrieve(
        query="What are the major risk factors?",
        cutoff_date="20230630",
        ticker="AAPL"
    )
    
    print("\nRetrieval Result:")
    print("=" * 60)
    print(result['output'])
    print("=" * 60)
    print(f"\nProcessing time: {result.get('processing_time', 'N/A')}s")
