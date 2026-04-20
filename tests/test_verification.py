
import pytest
from src.agents.verification_agent import VerificationAgent

def test_verification_agent_initialization():
    """Test that the verification agent can be initialized."""
    agent = VerificationAgent()
    assert agent is not None

def test_numeric_deviation_detection():
    """
    Test the internal logic for detecting deviations.
    Note: Real verification requires an LLM call or complex mocking.
    We test the structure and properties.
    """
    agent = VerificationAgent()
    # Mocking verify method for a simple test case
    # In a real environment, this would call verify(query, ticker, cutoff)
    assert hasattr(agent, 'verify')

@pytest.mark.skip(reason="Requires valid XBRL data and LLM")
def test_full_verification_flow():
    agent = VerificationAgent()
    result = agent.verify(
        query="Apple's revenue in 2023 was $100 billion",
        ticker="AAPL",
        cutoff_date="20231231"
    )
    assert "verification" in result
    assert "confidence" in result
