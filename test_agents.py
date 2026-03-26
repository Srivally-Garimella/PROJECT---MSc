"""
Test script for multi-agent system with local LLM (Ollama).

Prerequisites:
1. Install Ollama: https://ollama.com/download
2. Run Ollama: `ollama serve`
3. Pull a model: `ollama pull llama3.2`

Usage:
    python test_agents.py
"""

import sys
from src.agents import check_ollama_available, list_ollama_models

print("=" * 60)
print("TemporalGuard-RAG Multi-Agent System Test")
print("=" * 60)

# Check Ollama availability
print("\n1. Checking Ollama...")
if not check_ollama_available():
    print("   ❌ Ollama is not running!")
    print("\n   To use local LLM, please:")
    print("   1. Download Ollama from: https://ollama.com/download")
    print("   2. Install and run: ollama serve")
    print("   3. Pull a model: ollama pull llama3.2")
    print("\n   Alternatively, set OPENAI_API_KEY environment variable to use OpenAI.")
    sys.exit(1)

print("   ✅ Ollama is running")

# List available models
models = list_ollama_models()
print(f"\n2. Available models: {models}")

if not models:
    print("   ❌ No models found. Please run: ollama pull llama3.2")
    sys.exit(1)

# Import agents
print("\n3. Importing agents...")
from src.agents import (
    DocumentRetrievalAgent,
    CalculationAgent,
    VerificationAgent,
    TemporalAgent,
    MultiAgentOrchestrator
)
from src.rag_system.vector_store import TemporalVectorStore

print("   ✅ All agents imported successfully")

# Initialize vector store
print("\n4. Loading vector store...")
vector_store = TemporalVectorStore()
print(f"   ✅ Vector store loaded with {vector_store.collection.count()} documents")

# Initialize orchestrator
print("\n5. Initializing Multi-Agent Orchestrator...")
orchestrator = MultiAgentOrchestrator(
    vector_store=vector_store,
    provider="ollama",  # Use local Ollama
    model_name="llama3.2"  # Or use first available model: models[0]
)
print("   ✅ Orchestrator initialized")

# Test query
print("\n6. Running test query...")
print("-" * 60)

result = orchestrator.process_query(
    query="What were Apple's total revenues?",
    ticker="AAPL",
    analysis_date="20251001",  # Point-in-time: October 2025
    verbose=True
)

print("\n" + "=" * 60)
print("FINAL RESULT:")
print("=" * 60)
print(f"\nQuery: {result.get('query', 'N/A')}")
print(f"Ticker: {result.get('ticker', 'N/A')}")
print(f"Analysis Date: {result.get('analysis_date', 'N/A')}")
print(f"\nAnswer:\n{result.get('final_answer', result.get('output', 'No answer generated'))}")

print("\n✅ Test completed successfully!")
