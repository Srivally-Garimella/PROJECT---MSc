"""Test vector store search functionality."""
from src.rag_system.vector_store import TemporalVectorStore

vs = TemporalVectorStore()
print(f"Vector store count: {vs.collection.count()}")

# Test basic search
results = vs.temporal_search(query='revenue growth', n_results=3)
print(f"\n=== Search for 'revenue growth' ===")
print(f"Found {len(results['documents'][0])} results")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"\n{i+1}. {meta['ticker']} ({meta.get('filing_date_str', meta['filing_date'])}):")
    print(f"   {doc[:150]}...")

# Test temporal filtering (documents before March 2025)
results2 = vs.temporal_search(
    query='net income profit',
    cutoff_date='20250301',
    n_results=3
)
print(f"\n=== Search for 'net income profit' BEFORE March 2025 ===")
print(f"Found {len(results2['documents'][0])} results")
for i, (doc, meta) in enumerate(zip(results2['documents'][0], results2['metadatas'][0])):
    print(f"\n{i+1}. {meta['ticker']} ({meta.get('filing_date_str', meta['filing_date'])}):")
    print(f"   {doc[:150]}...")

# Test ticker filter
results3 = vs.temporal_search(
    query='operating expenses',
    ticker='AAPL',
    n_results=3
)
print(f"\n=== Search for 'operating expenses' - AAPL only ===")
print(f"Found {len(results3['documents'][0])} results")
for i, (doc, meta) in enumerate(zip(results3['documents'][0], results3['metadatas'][0])):
    print(f"\n{i+1}. {meta['ticker']} ({meta.get('filing_date_str', meta['filing_date'])}):")
    print(f"   {doc[:150]}...")

# Test combined filters: JPM before 2026
results4 = vs.temporal_search(
    query='loan portfolio risk',
    ticker='JPM',
    cutoff_date='20260101',
    n_results=3
)
print(f"\n=== Search for 'loan portfolio risk' - JPM only, before 2026 ===")
print(f"Found {len(results4['documents'][0])} results")
for i, (doc, meta) in enumerate(zip(results4['documents'][0], results4['metadatas'][0])):
    print(f"\n{i+1}. {meta['ticker']} ({meta.get('filing_date_str', meta['filing_date'])}):")
    print(f"   {doc[:150]}...")

print("\n✅ Vector store search tests completed!")
