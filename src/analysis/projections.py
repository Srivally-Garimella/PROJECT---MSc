"""
Financial Projection Engine for TemporalGuard-RAG

Provides forecasting capabilities including:
- Linear trend projections
- Growth rate-based forecasting
- DCF valuation
- Revenue/earnings projections
- Cash flow forecasting
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import pandas as pd
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Projection:
    """Container for a financial projection."""
    metric_name: str
    base_year: int
    target_year: int
    base_value: float
    projected_value: float
    method: str
    assumptions: Dict[str, float]
    confidence_interval: Tuple[float, float] = None
    confidence_level: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    explanation: str = ""


@dataclass
class ProjectionScenario:
    """Multiple projections under different scenarios."""
    metric_name: str
    base_case: Projection
    bull_case: Projection = None
    bear_case: Projection = None


class ProjectionEngine:
    """
    Financial projection and forecasting engine.
    
    Methods:
    - Linear regression trend
    - CAGR-based projection
    - Analyst consensus extrapolation
    - DCF valuation
    - Monte Carlo simulation
    """
    
    def __init__(self):
        """Initialize projection engine."""
        logger.info("Initialized Financial Projection Engine")
    
    # ═══════════════════════════════════════════════════════════════════
    # TREND-BASED PROJECTIONS
    # ═══════════════════════════════════════════════════════════════════
    
    def linear_trend_projection(self, 
                                historical_data: List[Tuple[int, float]],
                                target_year: int,
                                metric_name: str = "Metric") -> Projection:
        """
        Project future value using linear regression on historical data.
        
        Args:
            historical_data: List of (year, value) tuples
            target_year: Year to project to
            metric_name: Name of the metric being projected
            
        Returns:
            Projection object with forecasted value
        """
        if len(historical_data) < 2:
            return Projection(
                metric_name=metric_name,
                base_year=historical_data[-1][0] if historical_data else 0,
                target_year=target_year,
                base_value=historical_data[-1][1] if historical_data else 0,
                projected_value=0,
                method="Linear Regression",
                assumptions={"error": "Insufficient data (need at least 2 points)"},
                confidence_level="LOW"
            )
        
        years = np.array([d[0] for d in historical_data])
        values = np.array([d[1] for d in historical_data])
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
        
        projected_value = slope * target_year + intercept
        
        # Calculate confidence interval (95%)
        n = len(years)
        mean_x = np.mean(years)
        se = std_err * np.sqrt(1 + 1/n + (target_year - mean_x)**2 / np.sum((years - mean_x)**2))
        ci_lower = projected_value - 1.96 * se
        ci_upper = projected_value + 1.96 * se
        
        # Determine confidence based on R-squared
        r_squared = r_value ** 2
        confidence = "HIGH" if r_squared > 0.8 else "MEDIUM" if r_squared > 0.5 else "LOW"
        
        return Projection(
            metric_name=metric_name,
            base_year=int(years[-1]),
            target_year=target_year,
            base_value=float(values[-1]),
            projected_value=round(projected_value, 2),
            method="Linear Regression",
            assumptions={
                "slope": round(slope, 2),
                "intercept": round(intercept, 2),
                "r_squared": round(r_squared, 4),
                "data_points": n
            },
            confidence_interval=(round(ci_lower, 2), round(ci_upper, 2)),
            confidence_level=confidence,
            explanation=f"Based on linear trend of {n} years of data (R² = {r_squared:.2%})"
        )
    
    def cagr_projection(self,
                        historical_data: List[Tuple[int, float]],
                        target_year: int,
                        metric_name: str = "Metric") -> Projection:
        """
        Project future value using Compound Annual Growth Rate.
        
        Formula: Future Value = Present Value × (1 + CAGR)^years
        """
        if len(historical_data) < 2:
            return Projection(
                metric_name=metric_name,
                base_year=0,
                target_year=target_year,
                base_value=0,
                projected_value=0,
                method="CAGR",
                assumptions={"error": "Insufficient data"},
                confidence_level="LOW"
            )
        
        # Sort by year
        sorted_data = sorted(historical_data, key=lambda x: x[0])
        first_year, first_value = sorted_data[0]
        last_year, last_value = sorted_data[-1]
        
        years_of_data = last_year - first_year
        
        if years_of_data == 0 or first_value <= 0:
            return Projection(
                metric_name=metric_name,
                base_year=last_year,
                target_year=target_year,
                base_value=last_value,
                projected_value=last_value,
                method="CAGR",
                assumptions={"error": "Cannot calculate CAGR"},
                confidence_level="LOW"
            )
        
        # Calculate CAGR
        cagr = (last_value / first_value) ** (1 / years_of_data) - 1
        
        # Project forward
        years_to_project = target_year - last_year
        projected_value = last_value * ((1 + cagr) ** years_to_project)
        
        # Confidence based on data consistency
        intermediate_cagrs = []
        for i in range(1, len(sorted_data)):
            y1, v1 = sorted_data[i-1]
            y2, v2 = sorted_data[i]
            if y2 > y1 and v1 > 0:
                intermediate_cagrs.append((v2/v1) ** (1/(y2-y1)) - 1)
        
        cagr_volatility = np.std(intermediate_cagrs) if intermediate_cagrs else 1
        confidence = "HIGH" if cagr_volatility < 0.1 else "MEDIUM" if cagr_volatility < 0.3 else "LOW"
        
        # Calculate range (±1 std dev of historical CAGR)
        if intermediate_cagrs:
            high_cagr = cagr + np.std(intermediate_cagrs)
            low_cagr = cagr - np.std(intermediate_cagrs)
            ci_lower = last_value * ((1 + low_cagr) ** years_to_project)
            ci_upper = last_value * ((1 + high_cagr) ** years_to_project)
        else:
            ci_lower = projected_value * 0.8
            ci_upper = projected_value * 1.2
        
        return Projection(
            metric_name=metric_name,
            base_year=last_year,
            target_year=target_year,
            base_value=round(last_value, 2),
            projected_value=round(projected_value, 2),
            method="CAGR Projection",
            assumptions={
                "cagr": round(cagr * 100, 2),
                "years_of_data": years_of_data,
                "years_projected": years_to_project
            },
            confidence_interval=(round(ci_lower, 2), round(ci_upper, 2)),
            confidence_level=confidence,
            explanation=f"Projected at {cagr*100:.1f}% CAGR based on {years_of_data} years of historical data"
        )
    
    def growth_rate_projection(self,
                               base_value: float,
                               base_year: int,
                               target_year: int,
                               growth_rate: float,
                               metric_name: str = "Metric",
                               growth_decay: float = 0) -> Projection:
        """
        Project using a specified growth rate with optional decay.
        
        Args:
            base_value: Starting value
            base_year: Starting year
            target_year: Year to project to
            growth_rate: Annual growth rate (e.g., 0.10 for 10%)
            metric_name: Name of metric
            growth_decay: Annual reduction in growth rate (e.g., 0.01 = 1% decay per year)
        """
        years = target_year - base_year
        projected_value = base_value
        
        # Apply growth with optional decay
        for year in range(years):
            effective_rate = max(0, growth_rate - (growth_decay * year))
            projected_value *= (1 + effective_rate)
        
        return Projection(
            metric_name=metric_name,
            base_year=base_year,
            target_year=target_year,
            base_value=round(base_value, 2),
            projected_value=round(projected_value, 2),
            method="Growth Rate Projection",
            assumptions={
                "initial_growth_rate": round(growth_rate * 100, 2),
                "growth_decay": round(growth_decay * 100, 2),
                "years": years
            },
            confidence_level="MEDIUM",
            explanation=f"Projected at {growth_rate*100:.1f}% growth rate" + 
                       (f" with {growth_decay*100:.1f}% annual decay" if growth_decay else "")
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # CASH FLOW PROJECTIONS
    # ═══════════════════════════════════════════════════════════════════
    
    def project_free_cash_flow(self,
                               revenue_history: List[Tuple[int, float]],
                               fcf_margin_history: List[Tuple[int, float]],
                               target_year: int,
                               revenue_growth_assumption: float = None,
                               margin_assumption: float = None) -> Projection:
        """
        Project Free Cash Flow based on revenue and FCF margin trends.
        
        FCF = Revenue × FCF Margin
        """
        # Get revenue projection
        if revenue_growth_assumption:
            last_revenue = revenue_history[-1][1] if revenue_history else 0
            last_year = revenue_history[-1][0] if revenue_history else target_year - 1
            years = target_year - last_year
            projected_revenue = last_revenue * ((1 + revenue_growth_assumption) ** years)
        else:
            revenue_proj = self.cagr_projection(revenue_history, target_year, "Revenue")
            projected_revenue = revenue_proj.projected_value
        
        # Get FCF margin
        if margin_assumption:
            fcf_margin = margin_assumption
        elif fcf_margin_history:
            # Use average of recent margins
            recent_margins = [m[1] for m in fcf_margin_history[-3:]]
            fcf_margin = np.mean(recent_margins)
        else:
            fcf_margin = 0.15  # Default 15% FCF margin
        
        projected_fcf = projected_revenue * fcf_margin
        
        return Projection(
            metric_name="Free Cash Flow",
            base_year=revenue_history[-1][0] if revenue_history else target_year - 1,
            target_year=target_year,
            base_value=round(revenue_history[-1][1] * (fcf_margin_history[-1][1] if fcf_margin_history else fcf_margin), 2),
            projected_value=round(projected_fcf, 2),
            method="Revenue × FCF Margin",
            assumptions={
                "projected_revenue": round(projected_revenue, 2),
                "fcf_margin": round(fcf_margin * 100, 2),
                "revenue_growth": round((revenue_growth_assumption or 0) * 100, 2) if revenue_growth_assumption else "CAGR-based"
            },
            confidence_level="MEDIUM",
            explanation=f"FCF = ${projected_revenue/1e9:.2f}B revenue × {fcf_margin*100:.1f}% margin"
        )
    
    def project_operating_cash_flow(self,
                                    net_income: float,
                                    target_year: int,
                                    base_year: int,
                                    depreciation_rate: float = 0.05,
                                    working_capital_change: float = 0,
                                    ni_growth_rate: float = 0.05) -> Projection:
        """
        Project Operating Cash Flow.
        
        OCF = Net Income + Depreciation - Changes in Working Capital
        
        Simplified: OCF ≈ Net Income × 1.2 (typical for mature companies)
        """
        years = target_year - base_year
        projected_ni = net_income * ((1 + ni_growth_rate) ** years)
        
        # Estimate depreciation as % of revenue (we'll use NI as proxy)
        depreciation = projected_ni * depreciation_rate * 3  # D&A typically 15-20% of NI
        
        # Working capital typically grows with revenue
        wc_change = projected_ni * working_capital_change
        
        projected_ocf = projected_ni + depreciation - wc_change
        
        return Projection(
            metric_name="Operating Cash Flow",
            base_year=base_year,
            target_year=target_year,
            base_value=round(net_income * 1.2, 2),  # Estimated current OCF
            projected_value=round(projected_ocf, 2),
            method="Net Income + D&A - ΔWC",
            assumptions={
                "net_income_growth": round(ni_growth_rate * 100, 2),
                "projected_net_income": round(projected_ni, 2),
                "depreciation_rate": round(depreciation_rate * 100, 2)
            },
            confidence_level="MEDIUM",
            explanation=f"Based on {ni_growth_rate*100:.0f}% NI growth and typical D&A levels"
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # SCENARIO-BASED PROJECTIONS
    # ═══════════════════════════════════════════════════════════════════
    
    def scenario_projection(self,
                           base_value: float,
                           base_year: int,
                           target_year: int,
                           metric_name: str,
                           base_growth: float = 0.08,
                           bull_growth: float = 0.15,
                           bear_growth: float = 0.02) -> ProjectionScenario:
        """
        Generate bull, base, and bear case projections.
        """
        years = target_year - base_year
        
        base_proj = Projection(
            metric_name=metric_name,
            base_year=base_year,
            target_year=target_year,
            base_value=base_value,
            projected_value=round(base_value * ((1 + base_growth) ** years), 2),
            method="Scenario Analysis - Base Case",
            assumptions={"growth_rate": round(base_growth * 100, 2)},
            confidence_level="MEDIUM",
            explanation=f"Base case: {base_growth*100:.0f}% annual growth"
        )
        
        bull_proj = Projection(
            metric_name=metric_name,
            base_year=base_year,
            target_year=target_year,
            base_value=base_value,
            projected_value=round(base_value * ((1 + bull_growth) ** years), 2),
            method="Scenario Analysis - Bull Case",
            assumptions={"growth_rate": round(bull_growth * 100, 2)},
            confidence_level="LOW",
            explanation=f"Bull case: {bull_growth*100:.0f}% annual growth (optimistic)"
        )
        
        bear_proj = Projection(
            metric_name=metric_name,
            base_year=base_year,
            target_year=target_year,
            base_value=base_value,
            projected_value=round(base_value * ((1 + bear_growth) ** years), 2),
            method="Scenario Analysis - Bear Case",
            assumptions={"growth_rate": round(bear_growth * 100, 2)},
            confidence_level="LOW",
            explanation=f"Bear case: {bear_growth*100:.0f}% annual growth (conservative)"
        )
        
        return ProjectionScenario(
            metric_name=metric_name,
            base_case=base_proj,
            bull_case=bull_proj,
            bear_case=bear_proj
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # DCF VALUATION
    # ═══════════════════════════════════════════════════════════════════
    
    def dcf_valuation(self,
                      current_fcf: float,
                      growth_rate_5yr: float,
                      terminal_growth: float,
                      discount_rate: float,
                      shares_outstanding: float,
                      projection_years: int = 5) -> Dict:
        """
        Discounted Cash Flow valuation model.
        
        Args:
            current_fcf: Current Free Cash Flow
            growth_rate_5yr: Expected FCF growth rate for projection period
            terminal_growth: Perpetual growth rate after projection period
            discount_rate: WACC or required rate of return
            shares_outstanding: Number of shares for per-share value
            projection_years: Number of years to project (default 5)
            
        Returns:
            DCF valuation results including fair value per share
        """
        # Project FCF for each year
        projected_fcfs = []
        fcf = current_fcf
        for year in range(1, projection_years + 1):
            fcf = fcf * (1 + growth_rate_5yr)
            projected_fcfs.append(fcf)
        
        # Calculate present value of projected FCFs
        pv_fcfs = []
        for i, fcf in enumerate(projected_fcfs):
            pv = fcf / ((1 + discount_rate) ** (i + 1))
            pv_fcfs.append(pv)
        
        sum_pv_fcfs = sum(pv_fcfs)
        
        # Terminal value using Gordon Growth Model
        terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1 + discount_rate) ** projection_years)
        
        # Enterprise value
        enterprise_value = sum_pv_fcfs + pv_terminal
        
        # Equity value (simplified - should subtract debt and add cash)
        equity_value = enterprise_value
        
        # Per share value
        fair_value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
        
        return {
            "current_fcf": current_fcf,
            "projected_fcfs": projected_fcfs,
            "pv_of_fcfs": round(sum_pv_fcfs, 2),
            "terminal_value": round(terminal_value, 2),
            "pv_terminal_value": round(pv_terminal, 2),
            "enterprise_value": round(enterprise_value, 2),
            "fair_value_per_share": round(fair_value_per_share, 2),
            "assumptions": {
                "growth_rate_projection": f"{growth_rate_5yr*100:.1f}%",
                "terminal_growth": f"{terminal_growth*100:.1f}%",
                "discount_rate": f"{discount_rate*100:.1f}%",
                "projection_years": projection_years
            },
            "interpretation": f"DCF fair value: ${fair_value_per_share:.2f} per share"
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # EARNINGS PROJECTIONS
    # ═══════════════════════════════════════════════════════════════════
    
    def project_eps(self,
                    eps_history: List[Tuple[int, float]],
                    target_year: int,
                    analyst_estimate: float = None,
                    use_method: str = "cagr") -> Projection:
        """
        Project Earnings Per Share.
        
        Args:
            eps_history: Historical EPS data as (year, eps) tuples
            target_year: Year to project
            analyst_estimate: Override with analyst consensus if available
            use_method: "cagr", "linear", or "analyst"
        """
        if analyst_estimate:
            base_year = eps_history[-1][0] if eps_history else target_year - 1
            return Projection(
                metric_name="Earnings Per Share",
                base_year=base_year,
                target_year=target_year,
                base_value=eps_history[-1][1] if eps_history else 0,
                projected_value=analyst_estimate,
                method="Analyst Consensus",
                assumptions={"source": "Analyst estimates"},
                confidence_level="MEDIUM",
                explanation="Based on analyst consensus estimates"
            )
        
        if use_method == "linear":
            return self.linear_trend_projection(eps_history, target_year, "Earnings Per Share")
        else:
            return self.cagr_projection(eps_history, target_year, "Earnings Per Share")
    
    def project_revenue(self,
                        revenue_history: List[Tuple[int, float]],
                        target_year: int,
                        industry_growth: float = None) -> Projection:
        """
        Project Revenue using historical trend adjusted for industry expectations.
        """
        cagr_proj = self.cagr_projection(revenue_history, target_year, "Revenue")
        
        if industry_growth and abs(cagr_proj.assumptions.get("cagr", 0)/100 - industry_growth) > 0.1:
            # Blend historical CAGR with industry growth
            historical_cagr = cagr_proj.assumptions.get("cagr", 0) / 100
            blended_rate = (historical_cagr * 0.6) + (industry_growth * 0.4)
            
            return self.growth_rate_projection(
                base_value=revenue_history[-1][1],
                base_year=revenue_history[-1][0],
                target_year=target_year,
                growth_rate=blended_rate,
                metric_name="Revenue"
            )
        
        return cagr_proj
    
    # ═══════════════════════════════════════════════════════════════════
    # HELPER: Format projection for display
    # ═══════════════════════════════════════════════════════════════════
    
    def format_projection(self, projection: Projection, currency_format: bool = True) -> str:
        """Format a projection for human-readable display."""
        if currency_format and "currency" in str(type(projection.projected_value)).lower():
            value_str = f"${projection.projected_value/1e9:.2f}B" if projection.projected_value >= 1e9 else \
                       f"${projection.projected_value/1e6:.2f}M" if projection.projected_value >= 1e6 else \
                       f"${projection.projected_value:,.2f}"
        else:
            value_str = f"{projection.projected_value:,.2f}"
        
        lines = [
            f"📊 {projection.metric_name} Projection",
            f"━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Base Year ({projection.base_year}): {projection.base_value:,.0f}",
            f"Target Year ({projection.target_year}): {value_str}",
            f"Method: {projection.method}",
            f"Confidence: {projection.confidence_level}",
        ]
        
        if projection.confidence_interval:
            lines.append(f"Range: {projection.confidence_interval[0]:,.0f} - {projection.confidence_interval[1]:,.0f}")
        
        lines.append(f"\n{projection.explanation}")
        
        if projection.assumptions:
            lines.append("\nAssumptions:")
            for key, val in projection.assumptions.items():
                lines.append(f"  • {key}: {val}")
        
        return "\n".join(lines)
    
    def format_scenario(self, scenario: ProjectionScenario) -> str:
        """Format a scenario projection for display."""
        lines = [
            f"📈 {scenario.metric_name} Scenario Analysis",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            f"🐻 Bear Case: {scenario.bear_case.projected_value:,.0f}",
            f"   {scenario.bear_case.explanation}",
            "",
            f"📊 Base Case: {scenario.base_case.projected_value:,.0f}",
            f"   {scenario.base_case.explanation}",
            "",
            f"🐂 Bull Case: {scenario.bull_case.projected_value:,.0f}",
            f"   {scenario.bull_case.explanation}",
        ]
        return "\n".join(lines)
