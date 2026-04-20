
import pytest
from src.agents.calculation_agent import CalculationAgent
from pathlib import Path


def _empty_xbrl_dir() -> str:
    return str(Path(__file__).parent / "fixtures" / "empty_xbrl")


def test_mock_helpers_include_warning():
    """Mock helpers must clearly disclose mock usage."""
    agent = CalculationAgent(xbrl_dir=_empty_xbrl_dir(), provider="ollama")

    mock_xbrl = agent._mock_xbrl_data("TESTCO")
    assert "[WARNING: MOCK DATA USED" in mock_xbrl

    mock_metric = agent._mock_metric_value("TESTCO", "Revenue", "20230630")
    assert "[WARNING: MOCK DATA USED" in mock_metric


def test_calculate_ratio_logic_uses_mock_when_no_xbrl():
    """Ratio calculation should be deterministic in fallback mode when no XBRL exists."""
    agent = CalculationAgent(xbrl_dir=_empty_xbrl_dir(), provider="ollama")

    # Force non-LLM fallback path.
    agent.agent = None

    result = agent._calculate_ratio("ROE|TESTCO|20230630")
    assert "Financial Ratio Calculation" in result
    assert "Result: ROE =" in result
    # With mock values: NetIncome 24.16B, Equity 56.727B -> ~42.59%
    assert "42.59%" in result
    assert "[WARNING: MOCK DATA USED" in result


def test_fallback_mechanism_sets_mode():
    """calculate() should mark fallback mode when the agent is disabled."""
    agent = CalculationAgent(xbrl_dir=_empty_xbrl_dir(), provider="ollama")
    agent.agent = None

    out = agent.calculate(metric="profit_margin", ticker="TESTCO", date="20231231")
    assert out["mode"] == "fallback"
    assert "profit_margin" in out["output"]
    assert "[WARNING: MOCK DATA USED" in out["output"]
