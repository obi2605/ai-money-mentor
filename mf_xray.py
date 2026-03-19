# ==============================================================================
# mf_xray.py
# AI Money Mentor — Mutual Fund Portfolio X-Ray Engine
# ------------------------------------------------------------------------------
# ARCHITECTURE CONTRACT:
#   • All inputs come from privacy_parser.py (already sanitised).
#   • No LLM involvement. All calculations are deterministic.
#   • Must complete in < 10 seconds for a typical 20-fund, 200-transaction portfolio.
#   • yfinance is used ONLY to fetch current NAVs if not present in CAMS data.
#
# MODULES:
#   1. Portfolio XIRR       — true return on each holding and overall
#   2. Fund Overlap Analysis — stock-level duplication across schemes
#   3. Expense Ratio Drag   — TER impact on 20-year final corpus
#   4. Category Allocation  — equity/debt/hybrid/sectoral breakdown
#   5. X-Ray Report Builder — combines all above into a single report object
# ==============================================================================

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

# Internal imports
from quant_engine import calculate_xirr, XIRRResult

logger = logging.getLogger(__name__)

# ============================================================================ #
#  SECTION 1 — DATA CLASSES                                                    #
# ============================================================================ #

@dataclass
class FundXIRRResult:
    """XIRR result for a single fund holding."""
    scheme_name: str
    isin: str
    total_invested: float
    current_value: float
    absolute_return_pct: float          # (current_value - invested) / invested * 100
    xirr_pct: float                     # Annualised XIRR
    holding_period_years: float
    num_transactions: int
    category: Optional[str] = None      # "Large Cap", "Mid Cap", etc.
    expense_ratio_pct: Optional[float] = None


@dataclass
class OverlapPair:
    """Stock overlap between two funds."""
    fund_a: str
    fund_b: str
    overlap_pct: float          # % of stocks in fund_a that also appear in fund_b
    common_stocks: list[str]    # Names of overlapping stocks
    overlap_weight_pct: float   # Weighted overlap by portfolio weight


@dataclass
class ExpenseDragResult:
    """Impact of TER on long-term corpus growth."""
    scheme_name: str
    ter_pct: float
    monthly_sip: float
    years: int
    corpus_with_ter: float
    corpus_without_ter: float
    drag_amount: float          # corpus_without - corpus_with
    drag_pct: float             # drag as % of corpus_without_ter


@dataclass
class PortfolioXRayReport:
    """Complete X-Ray report for a mutual fund portfolio."""
    # Summary
    investor_name: str
    pan_masked: str
    statement_date: Optional[str]
    num_funds: int
    num_transactions: int
    total_invested: float
    total_current_value: float
    absolute_return_pct: float
    portfolio_xirr_pct: float       # Weighted XIRR of all holdings
    analysis_duration_sec: float

    # Detailed results
    fund_xirr_results: list[FundXIRRResult]
    overlap_pairs: list[OverlapPair]
    expense_drag_results: list[ExpenseDragResult]
    category_allocation: dict[str, float]   # category → % of total value

    # Flags and recommendations (rule-based, not LLM)
    high_overlap_pairs: list[str]           # Fund pairs with >40% overlap
    high_ter_funds: list[str]               # Funds with TER > 1.5%
    concentrated_categories: list[str]      # Categories with >60% allocation
    recommendations: list[str]             # Plain-text rule-based suggestions


# ============================================================================ #
#  SECTION 2 — PORTFOLIO XIRR CALCULATOR                                      #
# ============================================================================ #

def calculate_portfolio_xirr(
    cams_data: dict,
) -> tuple[list[FundXIRRResult], float]:
    """
    Calculate per-fund and portfolio-level XIRR from parsed CAMS data.

    Strategy:
        For each holding:
            - Cash flows = all BUY transactions (negative) + current value (positive)
            - SELL transactions are incorporated: they reduce the final redemption value
        Portfolio XIRR:
            - Treat ALL transactions across ALL funds as a single cash flow stream
            - Final value = sum of all current values

    Parameters
    ----------
    cams_data : dict
        The dict returned by privacy_parser.parse_cams_pdf().

    Returns
    -------
    (list[FundXIRRResult], portfolio_xirr_pct)
    """
    fund_results: list[FundXIRRResult] = []
    all_dates: list[date] = []
    all_cashflows: list[float] = []
    total_current_value = 0.0

    mf_universe = _load_mf_universe()

    for holding in cams_data.get("holdings", []):
        transactions = holding.get("transactions", [])
        if not transactions:
            continue

        scheme_name = holding.get("scheme_name", "Unknown Fund")
        isin = holding.get("isin", "")
        current_value = holding.get("current_value") or 0.0
        expense_ratio = holding.get("expense_ratio_pct")

        # Look up fund metadata
        meta = _lookup_fund_meta(scheme_name, isin, mf_universe)
        if meta:
            if expense_ratio is None:
                expense_ratio = meta.get("expense_ratio_pct")
            category = meta.get("category")
        else:
            category = None

        # Build cash flow lists for this fund
        fund_dates: list[date] = []
        fund_cfs: list[float] = []
        total_invested = 0.0

        for tx in transactions:
            tx_date = date.fromisoformat(tx["date"]) if isinstance(tx["date"], str) else tx["date"]
            amount = tx["amount"]
            tx_type = tx.get("tx_type", "OTHER")

            if tx_type in ("BUY", "SWITCH_IN"):
                fund_dates.append(tx_date)
                fund_cfs.append(amount)       # Already negative (outflow)
                total_invested += abs(amount)
                all_dates.append(tx_date)
                all_cashflows.append(amount)

            elif tx_type in ("SELL", "SWITCH_OUT"):
                # Partial redemptions are positive cashflows within the fund stream
                fund_dates.append(tx_date)
                fund_cfs.append(amount)       # Positive (inflow)
                all_dates.append(tx_date)
                all_cashflows.append(amount)

        if not fund_dates or current_value <= 0:
            logger.debug("Skipping %s — no cash flows or zero current value", scheme_name)
            continue

        # Append current value as the final redemption cash flow
        today = date.today()
        fund_dates.append(today)
        fund_cfs.append(current_value)
        all_dates.append(today)
        all_cashflows.append(current_value)
        total_current_value += current_value

        # Calculate XIRR for this fund
        try:
            xirr_res = calculate_xirr(fund_dates, fund_cfs)
            xirr_pct = xirr_res.xirr_pct
            period_years = xirr_res.period_years
        except (ValueError, RuntimeError) as e:
            logger.warning("XIRR failed for %s: %s — using absolute return", scheme_name, e)
            # Fallback: simple absolute return (not annualised)
            xirr_pct = 0.0
            period_years = 0.0

        abs_return = (
            (current_value - total_invested) / total_invested * 100
            if total_invested > 0 else 0.0
        )

        fund_results.append(FundXIRRResult(
            scheme_name=scheme_name,
            isin=isin,
            total_invested=round(total_invested, 2),
            current_value=round(current_value, 2),
            absolute_return_pct=round(abs_return, 2),
            xirr_pct=xirr_pct,
            holding_period_years=period_years,
            num_transactions=len(transactions),
            category=category,
            expense_ratio_pct=expense_ratio,
        ))

    # Portfolio-level XIRR (all funds combined)
    portfolio_xirr_pct = 0.0
    if len(set(all_dates)) >= 2 and total_current_value > 0:
        try:
            # Replace the last entry (multiple current values → one combined value)
            # De-duplicate today's entries and sum them
            non_today = [(d, cf) for d, cf in zip(all_dates, all_cashflows) if d != date.today()]
            combined_dates = [d for d, _ in non_today] + [date.today()]
            combined_cfs = [cf for _, cf in non_today] + [total_current_value]
            port_xirr = calculate_xirr(combined_dates, combined_cfs)
            portfolio_xirr_pct = port_xirr.xirr_pct
        except (ValueError, RuntimeError) as e:
            logger.warning("Portfolio XIRR calculation failed: %s", e)
            # Fallback: weighted average of fund XIRRs
            if fund_results:
                total_val = sum(f.current_value for f in fund_results)
                if total_val > 0:
                    portfolio_xirr_pct = sum(
                        f.xirr_pct * f.current_value / total_val
                        for f in fund_results
                    )

    return fund_results, round(portfolio_xirr_pct, 2)


# ============================================================================ #
#  SECTION 3 — FUND OVERLAP ANALYSIS                                           #
# ============================================================================ #

def compute_overlap_analysis(
    fund_results: list[FundXIRRResult],
    mf_universe: Optional[dict] = None,
) -> list[OverlapPair]:
    """
    Compute stock-level overlap between all pairs of equity funds in the portfolio.

    Uses mf_universe.json top_holdings data. For funds not in the universe,
    overlap is reported as 0 with a note.

    Only analyses equity funds (Large Cap, Mid Cap, Flexi Cap, etc.) —
    debt and liquid funds are excluded as they hold bonds, not stocks.

    Parameters
    ----------
    fund_results : list[FundXIRRResult]
        The per-fund results from calculate_portfolio_xirr().
    mf_universe : dict, optional
        Pre-loaded universe. Loaded from file if None.

    Returns
    -------
    list[OverlapPair] — sorted by overlap_pct descending.
    """
    if mf_universe is None:
        mf_universe = _load_mf_universe()

    # Filter to equity funds only
    equity_categories = {
        "Large Cap", "Mid Cap", "Small Cap", "Flexi Cap", "Multi Cap",
        "Large & Mid Cap", "ELSS", "Sectoral", "Thematic", "Focused",
        "Value/Contra", "Dividend Yield"
    }
    equity_funds = [
        f for f in fund_results
        if f.category in equity_categories or f.category is None
    ]

    if len(equity_funds) < 2:
        return []

    # Build holdings map: fund → set of top stock names
    fund_holdings: dict[str, set[str]] = {}
    for fund in equity_funds:
        meta = _lookup_fund_meta(fund.scheme_name, fund.isin, mf_universe)
        if meta and meta.get("top_holdings"):
            fund_holdings[fund.scheme_name] = set(
                s.upper().strip() for s in meta["top_holdings"]
            )
        else:
            fund_holdings[fund.scheme_name] = set()

    # Compute total portfolio value for weight calculation
    total_value = sum(f.current_value for f in equity_funds) or 1.0

    overlap_pairs: list[OverlapPair] = []

    funds_with_data = [f for f in equity_funds if fund_holdings.get(f.scheme_name)]

    for i in range(len(funds_with_data)):
        for j in range(i + 1, len(funds_with_data)):
            fa = funds_with_data[i]
            fb = funds_with_data[j]
            holdings_a = fund_holdings[fa.scheme_name]
            holdings_b = fund_holdings[fb.scheme_name]

            if not holdings_a or not holdings_b:
                continue

            common = holdings_a & holdings_b
            overlap_pct = len(common) / len(holdings_a) * 100 if holdings_a else 0.0

            # Weighted overlap: penalise more if both funds are large parts of portfolio
            weight_a = fa.current_value / total_value
            weight_b = fb.current_value / total_value
            overlap_weight = overlap_pct * (weight_a + weight_b) / 2 * 100

            overlap_pairs.append(OverlapPair(
                fund_a=fa.scheme_name,
                fund_b=fb.scheme_name,
                overlap_pct=round(overlap_pct, 1),
                common_stocks=sorted(common)[:15],  # Top 15 for display
                overlap_weight_pct=round(overlap_weight, 1),
            ))

    return sorted(overlap_pairs, key=lambda x: x.overlap_pct, reverse=True)


# ============================================================================ #
#  SECTION 4 — EXPENSE RATIO DRAG CALCULATOR                                  #
# ============================================================================ #

def compute_expense_drag(
    fund_results: list[FundXIRRResult],
    projection_years: int = 20,
    base_return_pct: float = 12.0,
) -> list[ExpenseDragResult]:
    """
    Quantify the long-term corpus impact of each fund's TER.

    Formula:
        corpus_with_ter    = FV of current monthly SIP at (base_return - TER)
        corpus_without_ter = FV of current monthly SIP at base_return
        drag_amount        = corpus_without - corpus_with

    This makes the TER drag tangible in rupees — a far more powerful
    framing than just quoting a % number.

    Parameters
    ----------
    fund_results : list[FundXIRRResult]
    projection_years : int      Default 20 — a typical remaining investment horizon.
    base_return_pct : float     Gross return before expenses. Default 12%.

    Returns
    -------
    list[ExpenseDragResult] sorted by drag_amount descending.
    """
    import numpy_financial as npf

    results: list[ExpenseDragResult] = []

    for fund in fund_results:
        if fund.expense_ratio_pct is None or fund.expense_ratio_pct <= 0:
            continue

        # Estimate average monthly SIP from transaction history
        monthly_sip = _estimate_avg_monthly_sip(fund)
        if monthly_sip <= 0:
            monthly_sip = 5000.0  # Fallback: ₹5K/month

        ter = fund.expense_ratio_pct
        net_return = base_return_pct - ter
        gross_return = base_return_pct

        if net_return <= 0:
            continue

        months = projection_years * 12
        monthly_net = net_return / 100 / 12
        monthly_gross = gross_return / 100 / 12

        # Future value of SIP: npf.fv(rate, nper, pmt) — pmt is negative (outflow)
        corpus_with = float(npf.fv(monthly_net, months, -monthly_sip, 0))
        corpus_without = float(npf.fv(monthly_gross, months, -monthly_sip, 0))

        drag = corpus_without - corpus_with
        drag_pct = drag / corpus_without * 100 if corpus_without > 0 else 0.0

        results.append(ExpenseDragResult(
            scheme_name=fund.scheme_name,
            ter_pct=ter,
            monthly_sip=round(monthly_sip, 2),
            years=projection_years,
            corpus_with_ter=round(corpus_with, 2),
            corpus_without_ter=round(corpus_without, 2),
            drag_amount=round(drag, 2),
            drag_pct=round(drag_pct, 2),
        ))

    return sorted(results, key=lambda x: x.drag_amount, reverse=True)


# ============================================================================ #
#  SECTION 5 — CATEGORY ALLOCATION                                             #
# ============================================================================ #

def compute_category_allocation(
    fund_results: list[FundXIRRResult],
) -> dict[str, float]:
    """
    Compute the portfolio's allocation by SEBI fund category.

    Returns
    -------
    dict[category_name, percentage_of_total_value]
    """
    total_value = sum(f.current_value for f in fund_results)
    if total_value == 0:
        return {}

    allocation: dict[str, float] = {}
    for fund in fund_results:
        cat = fund.category or "Unknown"
        allocation[cat] = allocation.get(cat, 0.0) + fund.current_value

    return {
        cat: round(val / total_value * 100, 1)
        for cat, val in sorted(allocation.items(), key=lambda x: -x[1])
    }


# ============================================================================ #
#  SECTION 6 — X-RAY REPORT BUILDER (main entry point)                        #
# ============================================================================ #

def build_xray_report(cams_data: dict) -> PortfolioXRayReport:
    """
    Orchestrate all X-Ray analyses and return a comprehensive report.

    This is the function called by app.py when the user uploads a CAMS PDF.
    It must complete in < 10 seconds for typical portfolios.

    Parameters
    ----------
    cams_data : dict
        The sanitised dict from privacy_parser.parse_cams_pdf().

    Returns
    -------
    PortfolioXRayReport
    """
    import time
    start = time.time()

    mf_universe = _load_mf_universe()

    # ── 1. Per-fund and portfolio XIRR ───────────────────────────────────── #
    logger.info("Running XIRR analysis...")
    fund_results, portfolio_xirr_pct = calculate_portfolio_xirr(cams_data)

    if not fund_results:
        raise ValueError(
            "No valid holdings with transactions found in the CAMS data. "
            "Please ensure the statement contains active fund holdings."
        )

    # ── 2. Overlap analysis ───────────────────────────────────────────────── #
    logger.info("Running overlap analysis...")
    overlap_pairs = compute_overlap_analysis(fund_results, mf_universe)

    # ── 3. Expense drag ───────────────────────────────────────────────────── #
    logger.info("Computing expense drag...")
    expense_drag = compute_expense_drag(fund_results)

    # ── 4. Category allocation ────────────────────────────────────────────── #
    category_allocation = compute_category_allocation(fund_results)

    # ── 5. Aggregate portfolio metrics ────────────────────────────────────── #
    total_invested = sum(f.total_invested for f in fund_results)
    total_value = sum(f.current_value for f in fund_results)
    abs_return = (
        (total_value - total_invested) / total_invested * 100
        if total_invested > 0 else 0.0
    )

    # ── 6. Rule-based flags ───────────────────────────────────────────────── #
    high_overlap = [
        f"{p.fund_a} ↔ {p.fund_b} ({p.overlap_pct:.0f}%)"
        for p in overlap_pairs if p.overlap_pct > 40
    ]
    high_ter = [
        f"{f.scheme_name} (TER: {f.expense_ratio_pct:.2f}%)"
        for f in fund_results
        if f.expense_ratio_pct and f.expense_ratio_pct > 1.5
    ]
    concentrated = [
        f"{cat} ({pct:.0f}%)"
        for cat, pct in category_allocation.items()
        if pct > 60
    ]

    recommendations = _generate_recommendations(
        fund_results, overlap_pairs, expense_drag,
        category_allocation, portfolio_xirr_pct
    )

    duration = round(time.time() - start, 2)
    logger.info("X-Ray complete in %.2fs | %d funds | XIRR: %.2f%%",
                duration, len(fund_results), portfolio_xirr_pct)

    return PortfolioXRayReport(
        investor_name=cams_data.get("investor_name", "Investor"),
        pan_masked=cams_data.get("pan_masked", "**MASKED**"),
        statement_date=cams_data.get("statement_date"),
        num_funds=len(fund_results),
        num_transactions=cams_data.get("num_transactions", 0),
        total_invested=round(total_invested, 2),
        total_current_value=round(total_value, 2),
        absolute_return_pct=round(abs_return, 2),
        portfolio_xirr_pct=portfolio_xirr_pct,
        analysis_duration_sec=duration,
        fund_xirr_results=fund_results,
        overlap_pairs=overlap_pairs,
        expense_drag_results=expense_drag,
        category_allocation=category_allocation,
        high_overlap_pairs=high_overlap,
        high_ter_funds=high_ter,
        concentrated_categories=concentrated,
        recommendations=recommendations,
    )


def format_xray_for_llm(report: PortfolioXRayReport) -> str:
    """
    Serialise the key X-Ray metrics for the LLM response generator.
    Strips full transaction lists to stay within token limits.
    """
    def _fmt(v: float) -> str:
        if abs(v) >= 1e7: return f"₹{v/1e7:.2f}Cr"
        if abs(v) >= 1e5: return f"₹{v/1e5:.1f}L"
        return f"₹{v:,.0f}"

    top_funds = sorted(report.fund_xirr_results, key=lambda x: -x.xirr_pct)[:5]
    worst_funds = sorted(report.fund_xirr_results, key=lambda x: x.xirr_pct)[:3]

    payload = {
        "portfolio_summary": {
            "num_funds": report.num_funds,
            "total_invested": _fmt(report.total_invested),
            "total_current_value": _fmt(report.total_current_value),
            "absolute_return_pct": f"{report.absolute_return_pct:.1f}%",
            "portfolio_xirr_pct": f"{report.portfolio_xirr_pct:.2f}%",
        },
        "top_performing_funds": [
            {"name": f.scheme_name[:50], "xirr_pct": f.xirr_pct,
             "absolute_return_pct": f.absolute_return_pct}
            for f in top_funds
        ],
        "underperforming_funds": [
            {"name": f.scheme_name[:50], "xirr_pct": f.xirr_pct}
            for f in worst_funds
        ],
        "category_allocation": report.category_allocation,
        "high_overlap_alerts": report.high_overlap_pairs[:3],
        "high_ter_alerts": report.high_ter_funds[:3],
        "total_expense_drag_20yr": _fmt(
            sum(d.drag_amount for d in report.expense_drag_results)
        ),
        "recommendations": report.recommendations,
    }

    return json.dumps(payload, indent=2, ensure_ascii=False)


# ============================================================================ #
#  SECTION 7 — INTERNAL HELPERS                                                #
# ============================================================================ #

def _load_mf_universe() -> dict:
    """Load and index mf_universe.json by normalised scheme name."""
    try:
        universe_path = os.path.join(os.path.dirname(__file__), "data", "mf_universe.json")
        with open(universe_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Index by ISIN (primary) and normalised name (secondary)
        universe: dict = {}
        for fund in data:
            if fund.get("isin"):
                universe[fund["isin"]] = fund
            # Also index by normalised name for fuzzy matching
            norm_name = _normalise_scheme_name(fund.get("scheme_name", ""))
            if norm_name:
                universe[norm_name] = fund
        return universe
    except FileNotFoundError:
        logger.warning("mf_universe.json not found — overlap/TER analysis limited.")
        return {}
    except Exception as e:
        logger.error("Error loading mf_universe.json: %s", e)
        return {}


def _normalise_scheme_name(name: str) -> str:
    """Normalise for fuzzy matching."""
    import re
    name = name.lower()
    for suffix in [" - growth", " - direct", " - regular", " plan", " option",
                   " (g)", "(d)", " growth", " direct", " regular"]:
        name = name.replace(suffix, "")
    return re.sub(r'\s+', ' ', name).strip()


def _lookup_fund_meta(scheme_name: str, isin: str, universe: dict) -> Optional[dict]:
    """Look up fund metadata by ISIN first, then normalised name."""
    if isin and isin in universe:
        return universe[isin]
    norm = _normalise_scheme_name(scheme_name)
    if norm in universe:
        return universe[norm]
    # Partial match on first 3 words
    words = norm.split()[:3]
    fingerprint = " ".join(words)
    for key, fund in universe.items():
        if isinstance(key, str) and fingerprint in key:
            return fund
    return None


def _estimate_avg_monthly_sip(fund: FundXIRRResult) -> float:
    """
    Heuristic: divide total invested by number of months held.
    Used for expense drag projection when a precise SIP amount isn't known.
    """
    if fund.holding_period_years <= 0:
        return fund.total_invested
    months = fund.holding_period_years * 12
    return fund.total_invested / months if months > 0 else fund.total_invested


def _generate_recommendations(
    fund_results: list[FundXIRRResult],
    overlap_pairs: list[OverlapPair],
    expense_drag: list[ExpenseDragResult],
    category_allocation: dict[str, float],
    portfolio_xirr_pct: float,
) -> list[str]:
    """
    Generate rule-based, actionable portfolio recommendations.
    These are NOT LLM-generated — they are deterministic rules.
    The LLM will wrap these in natural language during response generation.
    """
    recs: list[str] = []

    # 1. Overlap
    high_overlap = [p for p in overlap_pairs if p.overlap_pct > 40]
    if high_overlap:
        worst = high_overlap[0]
        recs.append(
            f"HIGH OVERLAP: {worst.fund_a} and {worst.fund_b} share "
            f"{worst.overlap_pct:.0f}% of top holdings ({', '.join(worst.common_stocks[:3])}...). "
            f"Consider consolidating into one fund."
        )

    # 2. Expense ratio
    regular_plan_funds = [
        f for f in fund_results
        if f.expense_ratio_pct and f.expense_ratio_pct > 1.5
           and "direct" not in f.scheme_name.lower()
    ]
    if regular_plan_funds:
        worst_ter = max(regular_plan_funds, key=lambda x: x.expense_ratio_pct)
        drag = next((d for d in expense_drag if d.scheme_name == worst_ter.scheme_name), None)
        drag_str = f" — ₹{drag.drag_amount/1e5:.1f}L in 20-year drag" if drag else ""
        recs.append(
            f"SWITCH TO DIRECT PLAN: {worst_ter.scheme_name[:45]} has TER "
            f"{worst_ter.expense_ratio_pct:.2f}%{drag_str}. "
            f"The Direct plan equivalent saves ~0.5-1.0% per year."
        )

    # 3. Category concentration
    for cat, pct in category_allocation.items():
        if pct > 65:
            recs.append(
                f"CONCENTRATION RISK: {pct:.0f}% of your portfolio is in {cat} funds. "
                f"Consider adding Mid/Small Cap or Debt allocation for balance."
            )

    # 4. Underperformers
    laggards = [
        f for f in fund_results
        if f.xirr_pct > 0 and f.xirr_pct < 8.0 and f.holding_period_years >= 3
    ]
    if laggards:
        recs.append(
            f"UNDERPERFORMER: {laggards[0].scheme_name[:45]} has delivered only "
            f"{laggards[0].xirr_pct:.1f}% XIRR over {laggards[0].holding_period_years:.1f} years. "
            f"Compare against its benchmark before switching."
        )

    # 5. Portfolio XIRR vs benchmark
    if portfolio_xirr_pct < 10.0 and portfolio_xirr_pct > 0:
        recs.append(
            f"BELOW BENCHMARK: Your portfolio XIRR of {portfolio_xirr_pct:.1f}% "
            f"is below the Nifty 50's historical ~12% CAGR. "
            f"Review your fund selection and expense ratios."
        )

    # 6. Too many funds (over-diversification)
    equity_fund_count = sum(
        1 for f in fund_results
        if f.category not in ("Debt", "Liquid", "Overnight", "Ultra Short Duration", None)
    )
    if equity_fund_count > 6:
        recs.append(
            f"OVER-DIVERSIFICATION: You hold {equity_fund_count} equity funds. "
            f"More than 5-6 equity funds rarely adds diversification — "
            f"it only increases complexity and overlap."
        )

    if not recs:
        recs.append(
            "Your portfolio looks reasonably well-structured. "
            "Continue your SIPs and review annually."
        )

    return recs


# ============================================================================ #
#  SECTION 8 — SELF-TEST                                                       #
# ============================================================================ #

if __name__ == "__main__":
    print("=" * 65)
    print("  mf_xray.py — Self-Test Suite")
    print("=" * 65)

    # Synthetic CAMS data for testing without a real PDF
    synthetic_cams = {
        "investor_name": "Test Investor",
        "pan_masked": "ABCDE****F",
        "statement_date": "2024-12-31",
        "num_transactions": 60,
        "holdings": [
            {
                "amc_name": "HDFC Mutual Fund",
                "scheme_name": "HDFC Top 100 Fund - Growth Option",
                "isin": "INF179K01CX3",
                "folio_hash": "ABCD1234",
                "closing_units": 185.432,
                "current_nav": 1053.45,
                "current_value": 195350.0,
                "expense_ratio_pct": 1.64,
                "transactions": [
                    {"date": "2022-01-05", "description": "SIP", "amount": -5000,
                     "units": 6.234, "nav": 802.33, "balance_units": 6.234, "tx_type": "BUY"},
                    {"date": "2022-02-05", "description": "SIP", "amount": -5000,
                     "units": 6.012, "nav": 831.72, "balance_units": 12.246, "tx_type": "BUY"},
                    {"date": "2023-01-05", "description": "SIP", "amount": -5000,
                     "units": 5.456, "nav": 916.44, "balance_units": 80.123, "tx_type": "BUY"},
                    {"date": "2024-01-05", "description": "SIP", "amount": -5000,
                     "units": 4.987, "nav": 1002.65, "balance_units": 150.432, "tx_type": "BUY"},
                    {"date": "2024-12-05", "description": "SIP", "amount": -5000,
                     "units": 4.745, "nav": 1054.21, "balance_units": 185.432, "tx_type": "BUY"},
                ],
            },
            {
                "amc_name": "Mirae Asset Mutual Fund",
                "scheme_name": "Mirae Asset Large Cap Fund - Direct Plan - Growth",
                "isin": "INF769K01EW0",
                "folio_hash": "EFGH5678",
                "closing_units": 312.765,
                "current_nav": 450.12,
                "current_value": 140812.0,
                "expense_ratio_pct": 0.54,
                "transactions": [
                    {"date": "2021-06-10", "description": "SIP", "amount": -5000,
                     "units": 14.321, "nav": 349.15, "balance_units": 14.321, "tx_type": "BUY"},
                    {"date": "2022-06-10", "description": "SIP", "amount": -5000,
                     "units": 13.456, "nav": 371.60, "balance_units": 150.123, "tx_type": "BUY"},
                    {"date": "2024-06-10", "description": "SIP", "amount": -5000,
                     "units": 11.234, "nav": 445.05, "balance_units": 312.765, "tx_type": "BUY"},
                ],
            },
        ],
    }

    print("\n[TEST 1] Per-fund and portfolio XIRR")
    fund_results, portfolio_xirr = calculate_portfolio_xirr(synthetic_cams)
    for fr in fund_results:
        print(f"  {fr.scheme_name[:45]:<45} XIRR: {fr.xirr_pct:>6.2f}%  "
              f"Abs: {fr.absolute_return_pct:>6.1f}%")
    print(f"\n  Portfolio XIRR: {portfolio_xirr:.2f}%")

    print("\n[TEST 2] Category Allocation")
    cat_alloc = compute_category_allocation(fund_results)
    for cat, pct in cat_alloc.items():
        print(f"  {cat}: {pct:.1f}%")

    print("\n[TEST 3] Expense Drag (20 years, 12% base return)")
    drag_results = compute_expense_drag(fund_results, projection_years=20)
    for dr in drag_results:
        print(f"  {dr.scheme_name[:45]:<45} TER: {dr.ter_pct:.2f}%  "
              f"20Y Drag: ₹{dr.drag_amount/1e5:.1f}L")

    print("\n[TEST 4] Full X-Ray Report")
    report = build_xray_report(synthetic_cams)
    print(f"  Total Invested   : ₹{report.total_invested/1e5:.1f}L")
    print(f"  Current Value    : ₹{report.total_current_value/1e5:.1f}L")
    print(f"  Portfolio XIRR   : {report.portfolio_xirr_pct:.2f}%")
    print(f"  Analysis Time    : {report.analysis_duration_sec:.2f}s")
    print(f"\n  Recommendations:")
    for rec in report.recommendations:
        print(f"    → {rec[:80]}")

    print("\n✅ All self-tests completed.")
