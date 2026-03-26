"""
Financial Formulas Engine

Comprehensive library of financial calculations including:
- Profitability ratios
- Liquidity ratios
- Solvency ratios
- Efficiency ratios
- Valuation metrics
- Cash flow analysis
- Growth calculations
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FinancialMetric:
    """Container for a financial metric with metadata."""
    name: str
    value: float
    unit: str  # 'ratio', 'percent', 'currency', 'days'
    formula: str
    inputs: Dict[str, float]
    interpretation: str
    confidence: str = "HIGH"


class FinancialFormulas:
    """
    Comprehensive financial formula library.
    
    Categories:
    - Profitability: ROE, ROA, Profit Margins, ROIC
    - Liquidity: Current Ratio, Quick Ratio, Cash Ratio
    - Solvency: Debt/Equity, Interest Coverage, Debt Ratio
    - Efficiency: Asset Turnover, Inventory Turnover, DSO
    - Valuation: P/E, P/B, EV/EBITDA, PEG Ratio
    - Cash Flow: FCF, OCF, Cash Conversion
    - Growth: Revenue Growth, EPS Growth, CAGR
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # PROFITABILITY RATIOS
    # ═══════════════════════════════════════════════════════════════════
    
    @staticmethod
    def return_on_equity(net_income: float, shareholders_equity: float) -> FinancialMetric:
        """
        ROE = Net Income / Shareholders' Equity
        
        Measures how effectively a company uses equity to generate profits.
        """
        if shareholders_equity == 0:
            return FinancialMetric("ROE", 0, "percent", "Net Income / Shareholders' Equity", 
                                   {}, "Cannot calculate - zero equity", "LOW")
        
        roe = (net_income / shareholders_equity) * 100
        
        interpretation = "Excellent" if roe > 20 else "Good" if roe > 15 else "Average" if roe > 10 else "Poor"
        
        return FinancialMetric(
            name="Return on Equity (ROE)",
            value=round(roe, 2),
            unit="percent",
            formula="Net Income / Shareholders' Equity × 100",
            inputs={"net_income": net_income, "shareholders_equity": shareholders_equity},
            interpretation=f"{interpretation} - {roe:.2f}% return on shareholder investment"
        )
    
    @staticmethod
    def return_on_assets(net_income: float, total_assets: float) -> FinancialMetric:
        """
        ROA = Net Income / Total Assets
        
        Measures how efficiently a company uses its assets to generate profit.
        """
        if total_assets == 0:
            return FinancialMetric("ROA", 0, "percent", "Net Income / Total Assets",
                                   {}, "Cannot calculate - zero assets", "LOW")
        
        roa = (net_income / total_assets) * 100
        
        interpretation = "Excellent" if roa > 10 else "Good" if roa > 5 else "Average" if roa > 2 else "Poor"
        
        return FinancialMetric(
            name="Return on Assets (ROA)",
            value=round(roa, 2),
            unit="percent",
            formula="Net Income / Total Assets × 100",
            inputs={"net_income": net_income, "total_assets": total_assets},
            interpretation=f"{interpretation} - {roa:.2f}% return on assets"
        )
    
    @staticmethod
    def return_on_invested_capital(nopat: float, invested_capital: float) -> FinancialMetric:
        """
        ROIC = NOPAT / Invested Capital
        
        NOPAT = Operating Income × (1 - Tax Rate)
        Invested Capital = Total Debt + Equity - Cash
        """
        if invested_capital == 0:
            return FinancialMetric("ROIC", 0, "percent", "NOPAT / Invested Capital",
                                   {}, "Cannot calculate - zero invested capital", "LOW")
        
        roic = (nopat / invested_capital) * 100
        
        return FinancialMetric(
            name="Return on Invested Capital (ROIC)",
            value=round(roic, 2),
            unit="percent",
            formula="NOPAT / Invested Capital × 100",
            inputs={"nopat": nopat, "invested_capital": invested_capital},
            interpretation=f"ROIC of {roic:.2f}% - compare to WACC for value creation"
        )
    
    @staticmethod
    def gross_profit_margin(revenue: float, cost_of_goods_sold: float) -> FinancialMetric:
        """Gross Profit Margin = (Revenue - COGS) / Revenue"""
        if revenue == 0:
            return FinancialMetric("Gross Margin", 0, "percent", "(Revenue - COGS) / Revenue",
                                   {}, "Cannot calculate - zero revenue", "LOW")
        
        gross_profit = revenue - cost_of_goods_sold
        margin = (gross_profit / revenue) * 100
        
        return FinancialMetric(
            name="Gross Profit Margin",
            value=round(margin, 2),
            unit="percent",
            formula="(Revenue - COGS) / Revenue × 100",
            inputs={"revenue": revenue, "cogs": cost_of_goods_sold},
            interpretation=f"Gross margin of {margin:.2f}%"
        )
    
    @staticmethod
    def operating_profit_margin(operating_income: float, revenue: float) -> FinancialMetric:
        """Operating Margin = Operating Income / Revenue"""
        if revenue == 0:
            return FinancialMetric("Operating Margin", 0, "percent", "Operating Income / Revenue",
                                   {}, "Cannot calculate - zero revenue", "LOW")
        
        margin = (operating_income / revenue) * 100
        
        return FinancialMetric(
            name="Operating Profit Margin",
            value=round(margin, 2),
            unit="percent",
            formula="Operating Income / Revenue × 100",
            inputs={"operating_income": operating_income, "revenue": revenue},
            interpretation=f"Operating margin of {margin:.2f}%"
        )
    
    @staticmethod
    def net_profit_margin(net_income: float, revenue: float) -> FinancialMetric:
        """Net Profit Margin = Net Income / Revenue"""
        if revenue == 0:
            return FinancialMetric("Net Margin", 0, "percent", "Net Income / Revenue",
                                   {}, "Cannot calculate - zero revenue", "LOW")
        
        margin = (net_income / revenue) * 100
        
        return FinancialMetric(
            name="Net Profit Margin",
            value=round(margin, 2),
            unit="percent",
            formula="Net Income / Revenue × 100",
            inputs={"net_income": net_income, "revenue": revenue},
            interpretation=f"Net margin of {margin:.2f}%"
        )
    
    @staticmethod
    def ebitda_margin(ebitda: float, revenue: float) -> FinancialMetric:
        """EBITDA Margin = EBITDA / Revenue"""
        if revenue == 0:
            return FinancialMetric("EBITDA Margin", 0, "percent", "EBITDA / Revenue",
                                   {}, "Cannot calculate - zero revenue", "LOW")
        
        margin = (ebitda / revenue) * 100
        
        return FinancialMetric(
            name="EBITDA Margin",
            value=round(margin, 2),
            unit="percent",
            formula="EBITDA / Revenue × 100",
            inputs={"ebitda": ebitda, "revenue": revenue},
            interpretation=f"EBITDA margin of {margin:.2f}%"
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # LIQUIDITY RATIOS
    # ═══════════════════════════════════════════════════════════════════
    
    @staticmethod
    def current_ratio(current_assets: float, current_liabilities: float) -> FinancialMetric:
        """Current Ratio = Current Assets / Current Liabilities"""
        if current_liabilities == 0:
            return FinancialMetric("Current Ratio", float('inf'), "ratio", 
                                   "Current Assets / Current Liabilities",
                                   {}, "No current liabilities", "MEDIUM")
        
        ratio = current_assets / current_liabilities
        
        interpretation = "Strong" if ratio > 2 else "Adequate" if ratio > 1 else "Weak - liquidity concern"
        
        return FinancialMetric(
            name="Current Ratio",
            value=round(ratio, 2),
            unit="ratio",
            formula="Current Assets / Current Liabilities",
            inputs={"current_assets": current_assets, "current_liabilities": current_liabilities},
            interpretation=f"{interpretation} ({ratio:.2f}x coverage)"
        )
    
    @staticmethod
    def quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> FinancialMetric:
        """Quick Ratio = (Current Assets - Inventory) / Current Liabilities"""
        if current_liabilities == 0:
            return FinancialMetric("Quick Ratio", float('inf'), "ratio",
                                   "(Current Assets - Inventory) / Current Liabilities",
                                   {}, "No current liabilities", "MEDIUM")
        
        ratio = (current_assets - inventory) / current_liabilities
        
        return FinancialMetric(
            name="Quick Ratio (Acid Test)",
            value=round(ratio, 2),
            unit="ratio",
            formula="(Current Assets - Inventory) / Current Liabilities",
            inputs={"current_assets": current_assets, "inventory": inventory, 
                   "current_liabilities": current_liabilities},
            interpretation=f"Quick ratio of {ratio:.2f}x"
        )
    
    @staticmethod
    def cash_ratio(cash: float, current_liabilities: float) -> FinancialMetric:
        """Cash Ratio = Cash & Equivalents / Current Liabilities"""
        if current_liabilities == 0:
            return FinancialMetric("Cash Ratio", float('inf'), "ratio",
                                   "Cash / Current Liabilities", {}, "No current liabilities", "MEDIUM")
        
        ratio = cash / current_liabilities
        
        return FinancialMetric(
            name="Cash Ratio",
            value=round(ratio, 2),
            unit="ratio",
            formula="Cash & Equivalents / Current Liabilities",
            inputs={"cash": cash, "current_liabilities": current_liabilities},
            interpretation=f"Cash ratio of {ratio:.2f}x"
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # SOLVENCY RATIOS
    # ═══════════════════════════════════════════════════════════════════
    
    @staticmethod
    def debt_to_equity(total_debt: float, shareholders_equity: float) -> FinancialMetric:
        """Debt to Equity = Total Debt / Shareholders' Equity"""
        if shareholders_equity == 0:
            return FinancialMetric("D/E Ratio", float('inf'), "ratio",
                                   "Total Debt / Equity", {}, "Zero equity - extreme leverage", "LOW")
        
        ratio = total_debt / shareholders_equity
        
        interpretation = "Conservative" if ratio < 0.5 else "Moderate" if ratio < 1 else \
                        "Leveraged" if ratio < 2 else "Highly leveraged"
        
        return FinancialMetric(
            name="Debt to Equity Ratio",
            value=round(ratio, 2),
            unit="ratio",
            formula="Total Debt / Shareholders' Equity",
            inputs={"total_debt": total_debt, "shareholders_equity": shareholders_equity},
            interpretation=f"{interpretation} ({ratio:.2f}x)"
        )
    
    @staticmethod
    def debt_ratio(total_liabilities: float, total_assets: float) -> FinancialMetric:
        """Debt Ratio = Total Liabilities / Total Assets"""
        if total_assets == 0:
            return FinancialMetric("Debt Ratio", 0, "ratio",
                                   "Total Liabilities / Total Assets", {}, "Zero assets", "LOW")
        
        ratio = total_liabilities / total_assets
        
        return FinancialMetric(
            name="Debt Ratio",
            value=round(ratio, 2),
            unit="ratio",
            formula="Total Liabilities / Total Assets",
            inputs={"total_liabilities": total_liabilities, "total_assets": total_assets},
            interpretation=f"{ratio*100:.1f}% of assets financed by debt"
        )
    
    @staticmethod
    def interest_coverage(ebit: float, interest_expense: float) -> FinancialMetric:
        """Interest Coverage = EBIT / Interest Expense"""
        if interest_expense == 0:
            return FinancialMetric("Interest Coverage", float('inf'), "ratio",
                                   "EBIT / Interest Expense", {}, "No interest expense", "HIGH")
        
        ratio = ebit / interest_expense
        
        interpretation = "Strong" if ratio > 5 else "Adequate" if ratio > 2 else "Weak"
        
        return FinancialMetric(
            name="Interest Coverage Ratio",
            value=round(ratio, 2),
            unit="ratio",
            formula="EBIT / Interest Expense",
            inputs={"ebit": ebit, "interest_expense": interest_expense},
            interpretation=f"{interpretation} - can cover interest {ratio:.1f}x"
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # EFFICIENCY RATIOS
    # ═══════════════════════════════════════════════════════════════════
    
    @staticmethod
    def asset_turnover(revenue: float, average_assets: float) -> FinancialMetric:
        """Asset Turnover = Revenue / Average Total Assets"""
        if average_assets == 0:
            return FinancialMetric("Asset Turnover", 0, "ratio",
                                   "Revenue / Avg Assets", {}, "Zero assets", "LOW")
        
        ratio = revenue / average_assets
        
        return FinancialMetric(
            name="Asset Turnover",
            value=round(ratio, 2),
            unit="ratio",
            formula="Revenue / Average Total Assets",
            inputs={"revenue": revenue, "average_assets": average_assets},
            interpretation=f"Generates ${ratio:.2f} revenue per $1 of assets"
        )
    
    @staticmethod
    def inventory_turnover(cogs: float, average_inventory: float) -> FinancialMetric:
        """Inventory Turnover = COGS / Average Inventory"""
        if average_inventory == 0:
            return FinancialMetric("Inventory Turnover", float('inf'), "ratio",
                                   "COGS / Avg Inventory", {}, "Zero inventory", "HIGH")
        
        ratio = cogs / average_inventory
        
        return FinancialMetric(
            name="Inventory Turnover",
            value=round(ratio, 2),
            unit="ratio",
            formula="COGS / Average Inventory",
            inputs={"cogs": cogs, "average_inventory": average_inventory},
            interpretation=f"Inventory turns over {ratio:.1f}x per year"
        )
    
    @staticmethod
    def days_sales_outstanding(accounts_receivable: float, revenue: float, days: int = 365) -> FinancialMetric:
        """DSO = (Accounts Receivable / Revenue) × Days"""
        if revenue == 0:
            return FinancialMetric("DSO", 0, "days",
                                   "(AR / Revenue) × Days", {}, "Zero revenue", "LOW")
        
        dso = (accounts_receivable / revenue) * days
        
        return FinancialMetric(
            name="Days Sales Outstanding",
            value=round(dso, 1),
            unit="days",
            formula="(Accounts Receivable / Revenue) × 365",
            inputs={"accounts_receivable": accounts_receivable, "revenue": revenue},
            interpretation=f"Collects receivables in {dso:.0f} days"
        )
    
    @staticmethod
    def days_inventory_outstanding(inventory: float, cogs: float, days: int = 365) -> FinancialMetric:
        """DIO = (Inventory / COGS) × Days"""
        if cogs == 0:
            return FinancialMetric("DIO", 0, "days",
                                   "(Inventory / COGS) × Days", {}, "Zero COGS", "LOW")
        
        dio = (inventory / cogs) * days
        
        return FinancialMetric(
            name="Days Inventory Outstanding",
            value=round(dio, 1),
            unit="days",
            formula="(Inventory / COGS) × 365",
            inputs={"inventory": inventory, "cogs": cogs},
            interpretation=f"Holds inventory for {dio:.0f} days"
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # VALUATION METRICS
    # ═══════════════════════════════════════════════════════════════════
    
    @staticmethod
    def price_to_earnings(market_price: float, eps: float) -> FinancialMetric:
        """P/E Ratio = Market Price / EPS"""
        if eps == 0:
            return FinancialMetric("P/E Ratio", float('inf'), "ratio",
                                   "Price / EPS", {}, "Zero or negative earnings", "LOW")
        
        pe = market_price / eps
        
        interpretation = "Undervalued" if pe < 15 else "Fair" if pe < 25 else "Growth premium" if pe < 40 else "Expensive"
        
        return FinancialMetric(
            name="Price to Earnings (P/E)",
            value=round(pe, 2),
            unit="ratio",
            formula="Market Price / Earnings Per Share",
            inputs={"market_price": market_price, "eps": eps},
            interpretation=f"{interpretation} at {pe:.1f}x earnings"
        )
    
    @staticmethod
    def price_to_book(market_price: float, book_value_per_share: float) -> FinancialMetric:
        """P/B Ratio = Market Price / Book Value per Share"""
        if book_value_per_share == 0:
            return FinancialMetric("P/B Ratio", float('inf'), "ratio",
                                   "Price / Book Value", {}, "Zero book value", "LOW")
        
        pb = market_price / book_value_per_share
        
        return FinancialMetric(
            name="Price to Book (P/B)",
            value=round(pb, 2),
            unit="ratio",
            formula="Market Price / Book Value per Share",
            inputs={"market_price": market_price, "book_value_per_share": book_value_per_share},
            interpretation=f"Trading at {pb:.1f}x book value"
        )
    
    @staticmethod
    def ev_to_ebitda(enterprise_value: float, ebitda: float) -> FinancialMetric:
        """EV/EBITDA = Enterprise Value / EBITDA"""
        if ebitda == 0:
            return FinancialMetric("EV/EBITDA", float('inf'), "ratio",
                                   "EV / EBITDA", {}, "Zero EBITDA", "LOW")
        
        ratio = enterprise_value / ebitda
        
        return FinancialMetric(
            name="EV/EBITDA",
            value=round(ratio, 2),
            unit="ratio",
            formula="Enterprise Value / EBITDA",
            inputs={"enterprise_value": enterprise_value, "ebitda": ebitda},
            interpretation=f"Valued at {ratio:.1f}x EBITDA"
        )
    
    @staticmethod
    def peg_ratio(pe_ratio: float, earnings_growth_rate: float) -> FinancialMetric:
        """PEG = P/E Ratio / Earnings Growth Rate"""
        if earnings_growth_rate == 0:
            return FinancialMetric("PEG Ratio", float('inf'), "ratio",
                                   "P/E / Growth Rate", {}, "Zero growth", "LOW")
        
        peg = pe_ratio / earnings_growth_rate
        
        interpretation = "Undervalued" if peg < 1 else "Fair" if peg < 2 else "Overvalued"
        
        return FinancialMetric(
            name="PEG Ratio",
            value=round(peg, 2),
            unit="ratio",
            formula="P/E Ratio / Earnings Growth Rate (%)",
            inputs={"pe_ratio": pe_ratio, "earnings_growth_rate": earnings_growth_rate},
            interpretation=f"{interpretation} (PEG = {peg:.2f})"
        )
    
    @staticmethod
    def enterprise_value(market_cap: float, total_debt: float, cash: float) -> FinancialMetric:
        """EV = Market Cap + Total Debt - Cash"""
        ev = market_cap + total_debt - cash
        
        return FinancialMetric(
            name="Enterprise Value",
            value=round(ev, 0),
            unit="currency",
            formula="Market Cap + Total Debt - Cash",
            inputs={"market_cap": market_cap, "total_debt": total_debt, "cash": cash},
            interpretation=f"Enterprise value of ${ev/1e9:.2f}B"
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # CASH FLOW METRICS
    # ═══════════════════════════════════════════════════════════════════
    
    @staticmethod
    def free_cash_flow(operating_cash_flow: float, capex: float) -> FinancialMetric:
        """FCF = Operating Cash Flow - Capital Expenditures"""
        fcf = operating_cash_flow - capex
        
        return FinancialMetric(
            name="Free Cash Flow",
            value=round(fcf, 0),
            unit="currency",
            formula="Operating Cash Flow - Capital Expenditures",
            inputs={"operating_cash_flow": operating_cash_flow, "capex": capex},
            interpretation=f"FCF of ${fcf/1e9:.2f}B available for dividends/buybacks"
        )
    
    @staticmethod
    def fcf_margin(free_cash_flow: float, revenue: float) -> FinancialMetric:
        """FCF Margin = Free Cash Flow / Revenue"""
        if revenue == 0:
            return FinancialMetric("FCF Margin", 0, "percent",
                                   "FCF / Revenue", {}, "Zero revenue", "LOW")
        
        margin = (free_cash_flow / revenue) * 100
        
        return FinancialMetric(
            name="FCF Margin",
            value=round(margin, 2),
            unit="percent",
            formula="Free Cash Flow / Revenue × 100",
            inputs={"free_cash_flow": free_cash_flow, "revenue": revenue},
            interpretation=f"FCF margin of {margin:.1f}%"
        )
    
    @staticmethod
    def cash_conversion_cycle(dso: float, dio: float, dpo: float) -> FinancialMetric:
        """CCC = DSO + DIO - DPO"""
        ccc = dso + dio - dpo
        
        return FinancialMetric(
            name="Cash Conversion Cycle",
            value=round(ccc, 1),
            unit="days",
            formula="DSO + DIO - DPO",
            inputs={"dso": dso, "dio": dio, "dpo": dpo},
            interpretation=f"Cash tied up in operations for {ccc:.0f} days"
        )
    
    @staticmethod
    def operating_cash_flow_ratio(ocf: float, current_liabilities: float) -> FinancialMetric:
        """OCF Ratio = Operating Cash Flow / Current Liabilities"""
        if current_liabilities == 0:
            return FinancialMetric("OCF Ratio", float('inf'), "ratio",
                                   "OCF / Current Liabilities", {}, "No current liabilities", "MEDIUM")
        
        ratio = ocf / current_liabilities
        
        return FinancialMetric(
            name="Operating Cash Flow Ratio",
            value=round(ratio, 2),
            unit="ratio",
            formula="Operating Cash Flow / Current Liabilities",
            inputs={"ocf": ocf, "current_liabilities": current_liabilities},
            interpretation=f"Cash from operations covers current liabilities {ratio:.1f}x"
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # GROWTH METRICS
    # ═══════════════════════════════════════════════════════════════════
    
    @staticmethod
    def growth_rate(current_value: float, previous_value: float) -> FinancialMetric:
        """Growth Rate = (Current - Previous) / Previous × 100"""
        if previous_value == 0:
            return FinancialMetric("Growth Rate", float('inf') if current_value > 0 else 0,
                                   "percent", "(Current - Previous) / Previous",
                                   {}, "Previous value was zero", "LOW")
        
        growth = ((current_value - previous_value) / abs(previous_value)) * 100
        
        direction = "growth" if growth > 0 else "decline"
        
        return FinancialMetric(
            name="Growth Rate",
            value=round(growth, 2),
            unit="percent",
            formula="(Current - Previous) / Previous × 100",
            inputs={"current_value": current_value, "previous_value": previous_value},
            interpretation=f"{abs(growth):.1f}% {direction} year-over-year"
        )
    
    @staticmethod
    def cagr(beginning_value: float, ending_value: float, years: int) -> FinancialMetric:
        """CAGR = (Ending/Beginning)^(1/Years) - 1"""
        if beginning_value <= 0 or years == 0:
            return FinancialMetric("CAGR", 0, "percent",
                                   "(Ending/Beginning)^(1/Years) - 1",
                                   {}, "Invalid inputs", "LOW")
        
        cagr = ((ending_value / beginning_value) ** (1 / years) - 1) * 100
        
        return FinancialMetric(
            name="Compound Annual Growth Rate",
            value=round(cagr, 2),
            unit="percent",
            formula="(Ending Value / Beginning Value)^(1/Years) - 1",
            inputs={"beginning_value": beginning_value, "ending_value": ending_value, "years": years},
            interpretation=f"{cagr:.1f}% compound annual growth over {years} years"
        )
    
    @staticmethod
    def rule_of_40(revenue_growth: float, profit_margin: float) -> FinancialMetric:
        """Rule of 40 = Revenue Growth % + Profit Margin %"""
        score = revenue_growth + profit_margin
        
        interpretation = "Excellent (SaaS benchmark passed)" if score >= 40 else \
                        "Good" if score >= 30 else "Below benchmark"
        
        return FinancialMetric(
            name="Rule of 40",
            value=round(score, 2),
            unit="percent",
            formula="Revenue Growth % + Profit Margin %",
            inputs={"revenue_growth": revenue_growth, "profit_margin": profit_margin},
            interpretation=f"{interpretation} (Score: {score:.1f})"
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # PER-SHARE METRICS
    # ═══════════════════════════════════════════════════════════════════
    
    @staticmethod
    def earnings_per_share(net_income: float, shares_outstanding: float, 
                          preferred_dividends: float = 0) -> FinancialMetric:
        """EPS = (Net Income - Preferred Dividends) / Shares Outstanding"""
        if shares_outstanding == 0:
            return FinancialMetric("EPS", 0, "currency",
                                   "(Net Income - Pref Div) / Shares", {}, "Zero shares", "LOW")
        
        eps = (net_income - preferred_dividends) / shares_outstanding
        
        return FinancialMetric(
            name="Earnings Per Share",
            value=round(eps, 2),
            unit="currency",
            formula="(Net Income - Preferred Dividends) / Shares Outstanding",
            inputs={"net_income": net_income, "shares_outstanding": shares_outstanding,
                   "preferred_dividends": preferred_dividends},
            interpretation=f"EPS of ${eps:.2f}"
        )
    
    @staticmethod
    def book_value_per_share(shareholders_equity: float, shares_outstanding: float) -> FinancialMetric:
        """BVPS = Shareholders' Equity / Shares Outstanding"""
        if shares_outstanding == 0:
            return FinancialMetric("BVPS", 0, "currency",
                                   "Equity / Shares", {}, "Zero shares", "LOW")
        
        bvps = shareholders_equity / shares_outstanding
        
        return FinancialMetric(
            name="Book Value Per Share",
            value=round(bvps, 2),
            unit="currency",
            formula="Shareholders' Equity / Shares Outstanding",
            inputs={"shareholders_equity": shareholders_equity, "shares_outstanding": shares_outstanding},
            interpretation=f"Book value of ${bvps:.2f} per share"
        )
    
    @staticmethod
    def dividend_yield(annual_dividend: float, market_price: float) -> FinancialMetric:
        """Dividend Yield = Annual Dividend / Market Price"""
        if market_price == 0:
            return FinancialMetric("Dividend Yield", 0, "percent",
                                   "Dividend / Price", {}, "Zero price", "LOW")
        
        yield_pct = (annual_dividend / market_price) * 100
        
        return FinancialMetric(
            name="Dividend Yield",
            value=round(yield_pct, 2),
            unit="percent",
            formula="Annual Dividend Per Share / Market Price × 100",
            inputs={"annual_dividend": annual_dividend, "market_price": market_price},
            interpretation=f"Dividend yield of {yield_pct:.2f}%"
        )
    
    @staticmethod
    def payout_ratio(dividends_paid: float, net_income: float) -> FinancialMetric:
        """Payout Ratio = Dividends / Net Income"""
        if net_income == 0:
            return FinancialMetric("Payout Ratio", 0, "percent",
                                   "Dividends / Net Income", {}, "Zero net income", "LOW")
        
        ratio = (dividends_paid / net_income) * 100
        
        interpretation = "Sustainable" if ratio < 60 else "High" if ratio < 80 else "Unsustainable"
        
        return FinancialMetric(
            name="Dividend Payout Ratio",
            value=round(ratio, 2),
            unit="percent",
            formula="Dividends Paid / Net Income × 100",
            inputs={"dividends_paid": dividends_paid, "net_income": net_income},
            interpretation=f"{interpretation} payout of {ratio:.1f}%"
        )


# Convenience function to get all available formulas
def list_formulas() -> Dict[str, str]:
    """List all available financial formulas."""
    return {
        # Profitability
        "ROE": "Return on Equity = Net Income / Shareholders' Equity",
        "ROA": "Return on Assets = Net Income / Total Assets",
        "ROIC": "Return on Invested Capital = NOPAT / Invested Capital",
        "Gross Margin": "(Revenue - COGS) / Revenue",
        "Operating Margin": "Operating Income / Revenue",
        "Net Margin": "Net Income / Revenue",
        "EBITDA Margin": "EBITDA / Revenue",
        
        # Liquidity
        "Current Ratio": "Current Assets / Current Liabilities",
        "Quick Ratio": "(Current Assets - Inventory) / Current Liabilities",
        "Cash Ratio": "Cash / Current Liabilities",
        
        # Solvency
        "Debt/Equity": "Total Debt / Shareholders' Equity",
        "Debt Ratio": "Total Liabilities / Total Assets",
        "Interest Coverage": "EBIT / Interest Expense",
        
        # Efficiency
        "Asset Turnover": "Revenue / Average Total Assets",
        "Inventory Turnover": "COGS / Average Inventory",
        "DSO": "(Accounts Receivable / Revenue) × 365",
        "DIO": "(Inventory / COGS) × 365",
        
        # Valuation
        "P/E": "Price / Earnings Per Share",
        "P/B": "Price / Book Value Per Share",
        "EV/EBITDA": "Enterprise Value / EBITDA",
        "PEG": "P/E / Earnings Growth Rate",
        "EV": "Market Cap + Total Debt - Cash",
        
        # Cash Flow
        "FCF": "Operating Cash Flow - CapEx",
        "FCF Margin": "Free Cash Flow / Revenue",
        "Cash Conversion Cycle": "DSO + DIO - DPO",
        "OCF Ratio": "Operating Cash Flow / Current Liabilities",
        
        # Growth
        "YoY Growth": "(Current - Previous) / Previous",
        "CAGR": "(Ending/Beginning)^(1/Years) - 1",
        "Rule of 40": "Revenue Growth % + Profit Margin %",
        
        # Per Share
        "EPS": "(Net Income - Pref Dividends) / Shares Outstanding",
        "BVPS": "Shareholders' Equity / Shares Outstanding",
        "Dividend Yield": "Annual Dividend / Market Price",
        "Payout Ratio": "Dividends / Net Income",
    }
