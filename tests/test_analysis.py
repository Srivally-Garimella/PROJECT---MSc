"""
Test the new Financial Analysis capabilities.
"""

import sys
sys.path.insert(0, '.')

from src.agents import MultiAgentOrchestrator

def test_projection_query():
    """Test projection query routing."""
    print("=" * 60)
    print("Testing Projection Query")
    print("=" * 60)
    
    orchestrator = MultiAgentOrchestrator(
        xbrl_dir="data/raw/xbrl_structured"
    )
    
    # Test projection query
    result = orchestrator.process_query(
        query="What is the projected cash flow for Apple in 2027?",
        ticker="AAPL",
        analysis_date="20231231",
        verbose=True,
        fast_mode=True
    )
    
    print("\n" + "=" * 60)
    print("Query Type:", result['metadata'].get('query_type'))
    print("Pipeline:", result['metadata'].get('pipeline'))
    print("=" * 60)
    
    return result

def test_historical_extreme():
    """Test historical extreme query."""
    print("\n" + "=" * 60)
    print("Testing Historical Extreme Query")
    print("=" * 60)
    
    orchestrator = MultiAgentOrchestrator(
        xbrl_dir="data/raw/xbrl_structured"
    )
    
    result = orchestrator.process_query(
        query="When was Microsoft's highest EPS recorded?",
        ticker="MSFT",
        analysis_date="20231231",
        verbose=True,
        fast_mode=True
    )
    
    print("\n" + "=" * 60)
    print("Query Type:", result['metadata'].get('query_type'))
    print("=" * 60)
    
    return result

def test_ratio_query():
    """Test ratio calculation query."""
    print("\n" + "=" * 60)
    print("Testing Ratio Query")
    print("=" * 60)
    
    orchestrator = MultiAgentOrchestrator(
        xbrl_dir="data/raw/xbrl_structured"
    )
    
    result = orchestrator.process_query(
        query="What is Apple's ROE?",
        ticker="AAPL",
        analysis_date="20231231",
        verbose=True,
        fast_mode=True
    )
    
    print("\n" + "=" * 60)
    print("Query Type:", result['metadata'].get('query_type'))
    print("=" * 60)
    
    return result

if __name__ == "__main__":
    print("\n🧪 TESTING FINANCIAL ANALYSIS CAPABILITIES\n")
    
    # Run tests
    try:
        result1 = test_projection_query()
        print("\n✅ Projection query test completed")
    except Exception as e:
        print(f"\n❌ Projection test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-" * 60)
    
    try:
        result2 = test_historical_extreme()
        print("\n✅ Historical extreme test completed")
    except Exception as e:
        print(f"\n❌ Historical extreme test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-" * 60)
    
    try:
        result3 = test_ratio_query()
        print("\n✅ Ratio query test completed")
    except Exception as e:
        print(f"\n❌ Ratio test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 All tests completed!")
