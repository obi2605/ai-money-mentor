# ==============================================================================
# fire_planner.py
# AI Money Mentor — FIRE Path Planner & Backtesting Engine
# ==============================================================================

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import numpy_financial as npf
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

# ============================================================================ #
#  SECTION 1 — DATA CLASSES                                                    #
# ============================================================================ #

@dataclass
class AssetAllocation:
    equity_pct: float
    debt_pct: float
    gold_pct: float

    def __post_init__(self):
        total = self.equity_pct + self.debt_pct + self.gold_pct
        if not (99.0 <= total <= 101.0):
            raise ValueError(f"Asset allocation must sum to 100%. Got {total:.1f}%")

    @property
    def blended_return_pct(self) -> float:
        return (
            (self.equity_pct / 100) * 12.0
            + (self.debt_pct / 100) * 7.0
            + (self.gold_pct / 100) * 8.0
        )


@dataclass
class RoadmapMilestone:
    year: int
    age: int
    annual_sip: float
    monthly_sip: float
    allocation: AssetAllocation
    projected_corpus: float
    target_corpus: float
    progress_pct: float
    annual_return_pct: float


@dataclass
class FIRERoadmap:
    current_age: int
    retirement_age: int
    monthly_income: float
    monthly_expenses: float
    current_savings: float
    assumed_inflation_pct: float
    required_corpus: float
    # Two SIP figures — both shown to user
    user_monthly_sip: float         # What user is currently investing
    required_monthly_sip: float     # What user NEEDS to invest to hit target
    step_up_pct: float
    projected_corpus: float         # Based on USER'S actual SIP
    shortfall_surplus: float        # Positive = on track, negative = gap
    years_to_fire: int
    milestones: list[RoadmapMilestone]
    monthly_schedule: pd.DataFrame
    backtest_result: Optional["BacktestResult"] = None
    stress_results: Optional[dict] = None
    monte_carlo_result: Optional["MonteCarloResult"] = None


@dataclass
class BacktestResult:
    strategy_label: str
    start_date: str
    end_date: str
    total_invested: float
    final_corpus: float
    actual_xirr_pct: float
    benchmark_cagr_pct: float
    sip_vs_lumpsum_advantage_pct: float
    worst_drawdown_pct: float
    recovery_months: int
    annual_returns: list[dict]


@dataclass
class StressTestResult:
    scenario_name: str
    crash_start: str
    crash_end: str
    recovery_date: Optional[str]
    max_drawdown_pct: float
    recovery_months: int
    sip_rupee_cost_advantage: float


@dataclass
class MonteCarloResult:
    num_simulations: int
    success_rate_pct: float
    median_corpus: float
    p10_corpus: float
    p90_corpus: float
    probability_shortfall_pct: float
    confidence_label: str


# ============================================================================ #
#  SECTION 2 — ASSET ALLOCATION GLIDE PATH                                     #
# ============================================================================ #

def get_glide_path_allocation(current_age: int, retirement_age: int) -> AssetAllocation:
    """Age-based glide path: aggressive early, conservative near retirement."""
    years_to_retirement = retirement_age - current_age
    if years_to_retirement >= 15:
        return AssetAllocation(equity_pct=80.0, debt_pct=10.0, gold_pct=10.0)
    elif years_to_retirement >= 5:
        equity = 80.0 - (15 - years_to_retirement) * 2.0
        equity = max(equity, 60.0)
        gold = 10.0
        debt = 100.0 - equity - gold
        return AssetAllocation(equity_pct=round(equity, 1), debt_pct=round(debt, 1), gold_pct=gold)
    elif years_to_retirement > 0:
        equity = 60.0 - (5 - years_to_retirement) * 4.0
        equity = max(equity, 40.0)
        gold = 15.0
        debt = 100.0 - equity - gold
        return AssetAllocation(equity_pct=round(equity, 1), debt_pct=round(debt, 1), gold_pct=gold)
    else:
        return AssetAllocation(equity_pct=25.0, debt_pct=60.0, gold_pct=15.0)


# ============================================================================ #
#  SECTION 3 — FIRE CORPUS CALCULATOR                                          #
# ============================================================================ #

def calculate_fire_corpus(
    current_monthly_expenses: float,
    current_age: int,
    retirement_age: int,
    life_expectancy: int = 85,
    inflation_pct: float = 6.0,
    safe_withdrawal_rate_pct: float = 4.0,
) -> dict:
    """Calculate required FIRE corpus using the inflation-adjusted 4% rule."""
    years_to_retirement = retirement_age - current_age
    post_retirement_years = life_expectancy - retirement_age
    if years_to_retirement <= 0:
        raise ValueError("Retirement age must be greater than current age.")
    if post_retirement_years <= 0:
        raise ValueError("Life expectancy must exceed retirement age.")

    monthly_expenses_at_retirement = current_monthly_expenses * (
        (1 + inflation_pct / 100) ** years_to_retirement
    )
    annual_expenses_at_retirement = monthly_expenses_at_retirement * 12
    required_corpus = annual_expenses_at_retirement / (safe_withdrawal_rate_pct / 100)

    return {
        "required_corpus": round(required_corpus, 2),
        "monthly_expenses_at_retirement": round(monthly_expenses_at_retirement, 2),
        "annual_expenses_at_retirement": round(annual_expenses_at_retirement, 2),
        "post_retirement_years": post_retirement_years,
        "years_to_retirement": years_to_retirement,
        "inflation_pct": inflation_pct,
        "safe_withdrawal_rate_pct": safe_withdrawal_rate_pct,
    }


# ============================================================================ #
#  SECTION 4 — REQUIRED SIP REVERSE-SOLVER                                     #
# ============================================================================ #

def solve_required_sip(
    target_corpus: float,
    years: int,
    annual_return_pct: float,
    current_savings: float = 0.0,
    step_up_pct: float = 10.0,
) -> float:
    """Reverse-solve: what monthly SIP is needed to reach target_corpus?"""
    if target_corpus <= 0:
        raise ValueError("Target corpus must be positive.")
    if years <= 0:
        raise ValueError("Years must be positive.")

    monthly_rate = annual_return_pct / 100 / 12
    fv_savings = current_savings * (1 + monthly_rate) ** (years * 12)
    remaining_target = max(target_corpus - fv_savings, 0)

    if remaining_target == 0:
        return 0.0

    def _corpus_for_sip(monthly_sip: float) -> float:
        corpus = 0.0
        current_sip = monthly_sip
        for month in range(1, years * 12 + 1):
            if step_up_pct > 0 and month > 1 and (month - 1) % 12 == 0:
                current_sip *= (1 + step_up_pct / 100)
            corpus = corpus * (1 + monthly_rate) + current_sip
        return corpus

    def _objective(sip: float) -> float:
        return _corpus_for_sip(sip) - remaining_target

    lower = 500.0
    upper = target_corpus / max(years * 12, 1) * 3

    try:
        required_sip = brentq(_objective, lower, upper, xtol=1.0, maxiter=500)
    except ValueError:
        logger.warning("brentq failed for SIP solver; falling back to npf.pmt")
        required_sip = abs(
            npf.pmt(rate=monthly_rate, nper=years * 12, pv=0, fv=-remaining_target)
        )

    return round(required_sip, 2)


# ============================================================================ #
#  SECTION 5 — MONTH-BY-MONTH ROADMAP BUILDER                                  #
# ============================================================================ #

def build_fire_roadmap(
    current_age: int,
    retirement_age: int,
    monthly_income: float,
    monthly_expenses: float,
    current_savings: float = 0.0,
    user_monthly_sip: Optional[float] = None,   # What user IS investing
    target_corpus: Optional[float] = None,
    assumed_inflation_pct: float = 6.0,
    assumed_return_pct: Optional[float] = None,
    step_up_pct: float = 10.0,
    life_expectancy: int = 85,
    run_backtest: bool = True,
    run_stress_test: bool = True,
    run_monte_carlo: bool = True,
) -> FIRERoadmap:
    """
    Build a complete, backtest-validated FIRE roadmap.

    KEY LOGIC:
    - required_monthly_sip = what the user NEEDS to invest to hit the target
    - user_monthly_sip     = what the user IS CURRENTLY investing
    - projected_corpus     = corpus if user continues with their CURRENT SIP
    - shortfall/surplus    = projected_corpus vs required_corpus

    This separation means a user investing ₹10,000 when they need ₹16,565
    will correctly see a shortfall, not a false "on track" result.
    """
    years = retirement_age - current_age
    if years <= 0:
        raise ValueError("Retirement age must be greater than current age.")

    # ── Step 1: Corpus Target ──────────────────────────────────────────────── #
    corpus_data = calculate_fire_corpus(
        current_monthly_expenses=monthly_expenses,
        current_age=current_age,
        retirement_age=retirement_age,
        life_expectancy=life_expectancy,
        inflation_pct=assumed_inflation_pct,
    )
    required_corpus = target_corpus or corpus_data["required_corpus"]

    # ── Step 2: Return Assumption ──────────────────────────────────────────── #
    if assumed_return_pct is None:
        allocation_now = get_glide_path_allocation(current_age, retirement_age)
        assumed_return_pct = allocation_now.blended_return_pct

    if run_backtest:
        try:
            from quant_engine import fetch_historical_rolling_return
            market = fetch_historical_rolling_return("nifty50", "10Y")
            assumed_return_pct = min(assumed_return_pct, market.cagr_pct)
            logger.info("Return assumption capped at Nifty 10Y CAGR: %.2f%%", assumed_return_pct)
        except Exception as e:
            logger.warning("Could not fetch Nifty data for return cap: %s", e)

    # ── Step 3: Required SIP (recommendation) ─────────────────────────────── #
    required_sip = solve_required_sip(
        target_corpus=required_corpus,
        years=years,
        annual_return_pct=assumed_return_pct,
        current_savings=current_savings,
        step_up_pct=step_up_pct,
    )

    # ── Step 4: Projection SIP (user's actual SIP or default to 30% income) ── #
    # CRITICAL: project with what user IS investing, not what they SHOULD invest.
    # This gives an honest shortfall/surplus figure.
    if user_monthly_sip and user_monthly_sip > 0:
        projection_sip = user_monthly_sip
    else:
        # No SIP provided — default to 30% of income as a starting assumption
        projection_sip = monthly_income * 0.30

    # ── Step 5: Build month-by-month schedule using PROJECTION SIP ─────────── #
    monthly_rate = assumed_return_pct / 100 / 12
    current_sip = projection_sip
    corpus = current_savings
    records = []
    milestones = []
    total_invested = 0.0

    for month in range(1, years * 12 + 1):
        age_now = current_age + (month - 1) // 12
        year_num = (month - 1) // 12 + 1

        if step_up_pct > 0 and month > 1 and (month - 1) % 12 == 0:
            current_sip *= (1 + step_up_pct / 100)

        corpus = corpus * (1 + monthly_rate) + current_sip
        total_invested += current_sip

        records.append({
            "month": month,
            "year": year_num,
            "age": age_now,
            "monthly_sip": round(current_sip, 2),
            "corpus_value": round(corpus, 2),
            "total_invested": round(total_invested, 2),
            "gains": round(corpus - total_invested, 2),
            "target_corpus": round(required_corpus, 2),
            "progress_pct": round(corpus / required_corpus * 100, 1),
        })

        if month % 12 == 0:
            alloc = get_glide_path_allocation(age_now, retirement_age)
            milestones.append(RoadmapMilestone(
                year=year_num,
                age=age_now,
                annual_sip=round(current_sip * 12, 2),
                monthly_sip=round(current_sip, 2),
                allocation=alloc,
                projected_corpus=round(corpus, 2),
                target_corpus=round(required_corpus, 2),
                progress_pct=round(corpus / required_corpus * 100, 1),
                annual_return_pct=round(alloc.blended_return_pct, 2),
            ))

    monthly_schedule = pd.DataFrame(records)
    projected_corpus = round(corpus, 2)
    shortfall_surplus = round(projected_corpus - required_corpus, 2)

    roadmap = FIRERoadmap(
        current_age=current_age,
        retirement_age=retirement_age,
        monthly_income=monthly_income,
        monthly_expenses=monthly_expenses,
        current_savings=current_savings,
        assumed_inflation_pct=assumed_inflation_pct,
        required_corpus=required_corpus,
        user_monthly_sip=projection_sip,
        required_monthly_sip=required_sip,
        step_up_pct=step_up_pct,
        projected_corpus=projected_corpus,
        shortfall_surplus=shortfall_surplus,
        years_to_fire=years,
        milestones=milestones,
        monthly_schedule=monthly_schedule,
    )

    # ── Optional enrichment ────────────────────────────────────────────────── #
    if run_backtest:
        try:
            roadmap.backtest_result = backtest_sip_against_nifty(
                monthly_sip=projection_sip,
                step_up_pct=step_up_pct,
                years=min(years, 15),
            )
        except Exception as e:
            logger.warning("Backtest failed (non-fatal): %s", e)

    if run_stress_test:
        try:
            roadmap.stress_results = run_stress_scenarios(
                monthly_sip=projection_sip,
                current_corpus=current_savings,
            )
        except Exception as e:
            logger.warning("Stress test failed (non-fatal): %s", e)

    if run_monte_carlo:
        try:
            roadmap.monte_carlo_result = run_monte_carlo_simulation(
                monthly_sip=projection_sip,
                years=years,
                target_corpus=required_corpus,
                current_savings=current_savings,
                step_up_pct=step_up_pct,
                base_return_pct=assumed_return_pct,
            )
        except Exception as e:
            logger.warning("Monte Carlo failed (non-fatal): %s", e)

    return roadmap


# ============================================================================ #
#  SECTION 6 — HISTORICAL BACKTESTER                                           #
# ============================================================================ #

def _fetch_nifty_monthly(years: int = 15) -> pd.Series:
    raw = yf.download(
        "^NSEI", period=f"{years}y", interval="1mo",
        auto_adjust=True, progress=False, timeout=15,
    )
    if raw is None or raw.empty:
        # Try ETF fallback
        raw = yf.download(
            "NIFTYBEES.NS", period=f"{years}y", interval="1mo",
            auto_adjust=True, progress=False, timeout=15,
        )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no Nifty 50 data.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw["Close"].dropna()


def backtest_sip_against_nifty(
    monthly_sip: float,
    step_up_pct: float = 10.0,
    years: int = 10,
) -> BacktestResult:
    """Replay a SIP strategy against actual Nifty 50 monthly closing prices."""
    logger.info("Running SIP backtest | SIP=₹%.0f | Step-up=%.0f%% | Years=%d",
                monthly_sip, step_up_pct, years)

    nifty = _fetch_nifty_monthly(years=max(years + 2, 15))
    end_date = nifty.index[-1]
    start_date = end_date - pd.DateOffset(years=years)
    nifty = nifty[nifty.index >= start_date]

    if len(nifty) < 12:
        raise RuntimeError(f"Insufficient Nifty data for {years}Y backtest.")

    total_units = 0.0
    total_invested = 0.0
    current_sip = monthly_sip
    cash_flows: list[float] = []
    cf_dates: list[date] = []
    corpus_series: list[float] = []
    annual_snapshots: list[dict] = []
    start_nav = float(nifty.iloc[0])

    for i, (dt, nav) in enumerate(nifty.items()):
        nav = float(nav)
        month_num = i + 1
        if step_up_pct > 0 and month_num > 1 and (month_num - 1) % 12 == 0:
            current_sip *= (1 + step_up_pct / 100)
        units_bought = current_sip / nav
        total_units += units_bought
        total_invested += current_sip
        cash_flows.append(-current_sip)
        cf_dates.append(dt.date())
        corpus_now = total_units * nav
        corpus_series.append(corpus_now)
        if month_num % 12 == 0:
            year_num = month_num // 12
            year_start_corpus = corpus_series[month_num - 12] if month_num >= 12 else 0
            annual_return = (corpus_now / year_start_corpus - 1) * 100 if year_start_corpus > 0 else 0
            annual_snapshots.append({
                "year": year_num,
                "date": dt.strftime("%b %Y"),
                "nifty_level": round(nav, 2),
                "corpus": round(corpus_now, 2),
                "annual_return_pct": round(annual_return, 2),
            })

    final_corpus = total_units * float(nifty.iloc[-1])
    cash_flows.append(final_corpus)
    cf_dates.append(nifty.index[-1].date())

    from quant_engine import calculate_xirr
    xirr_result = calculate_xirr(cf_dates, cash_flows)

    actual_years = (nifty.index[-1] - nifty.index[0]).days / 365.25
    benchmark_cagr = (float(nifty.iloc[-1]) / start_nav) ** (1 / actual_years) - 1

    corpus_arr = np.array(corpus_series)
    rolling_max = np.maximum.accumulate(corpus_arr)
    drawdowns = (corpus_arr - rolling_max) / rolling_max
    worst_drawdown = float(drawdowns.min())
    trough_idx = int(np.argmin(drawdowns))
    pre_crash_peak = rolling_max[trough_idx]
    recovery_months = 0
    for j in range(trough_idx, len(corpus_arr)):
        if corpus_arr[j] >= pre_crash_peak:
            recovery_months = j - trough_idx
            break

    return BacktestResult(
        strategy_label=f"₹{monthly_sip:,.0f}/mo SIP | {step_up_pct:.0f}% step-up | {years}Y",
        start_date=nifty.index[0].strftime("%Y-%m-%d"),
        end_date=nifty.index[-1].strftime("%Y-%m-%d"),
        total_invested=round(total_invested, 2),
        final_corpus=round(final_corpus, 2),
        actual_xirr_pct=xirr_result.xirr_pct,
        benchmark_cagr_pct=round(benchmark_cagr * 100, 2),
        sip_vs_lumpsum_advantage_pct=round(xirr_result.xirr_pct - benchmark_cagr * 100, 2),
        worst_drawdown_pct=round(worst_drawdown * 100, 2),
        recovery_months=recovery_months,
        annual_returns=annual_snapshots,
    )


# ============================================================================ #
#  SECTION 7 — STRESS TESTER                                                   #
# ============================================================================ #

HISTORICAL_CRASHES: dict[str, dict] = {
    "2008_gfc": {
        "name": "2008 Global Financial Crisis",
        "crash_start": "2008-01-01",
        "crash_bottom": "2009-03-09",
        "drawdown_pct": -60.1,
        "recovery_months": 24,
    },
    "2020_covid": {
        "name": "2020 COVID-19 Crash",
        "crash_start": "2020-01-17",
        "crash_bottom": "2020-03-23",
        "drawdown_pct": -39.5,
        "recovery_months": 6,
    },
}


def run_stress_scenarios(
    monthly_sip: float,
    current_corpus: float = 0.0,
) -> dict[str, StressTestResult]:
    results: dict[str, StressTestResult] = {}
    for key, crash in HISTORICAL_CRASHES.items():
        try:
            crash_dt = pd.Timestamp(crash["crash_start"])
            fetch_start = (crash_dt - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
            fetch_end = (crash_dt + pd.DateOffset(years=2)).strftime("%Y-%m-%d")
            raw = yf.download("^NSEI", start=fetch_start, end=fetch_end,
                              interval="1mo", auto_adjust=True, progress=False, timeout=15)
            if raw is None or raw.empty:
                raise RuntimeError("No data")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            prices = raw["Close"].dropna()
            pre_crash_price = float(prices.iloc[0])
            units_held = current_corpus / pre_crash_price if pre_crash_price > 0 else 0
            units_without_sip = units_held
            corpus_series = []
            corpus_no_sip = []
            invested_through_crash = 0.0
            for nav_val in prices:
                nav = float(nav_val)
                units_bought = monthly_sip / nav if nav > 0 else 0
                units_held += units_bought
                invested_through_crash += monthly_sip
                corpus_series.append(units_held * nav)
                corpus_no_sip.append(units_without_sip * nav)
            corpus_arr = np.array(corpus_series)
            rolling_peak = np.maximum.accumulate(corpus_arr)
            drawdowns = (corpus_arr - rolling_peak) / rolling_peak
            max_dd = float(drawdowns.min())
            trough_idx = int(np.argmin(drawdowns))
            peak_val = float(rolling_peak[trough_idx])
            recovery_months = 0
            recovery_date = None
            for j in range(trough_idx, len(corpus_arr)):
                if corpus_arr[j] >= peak_val:
                    recovery_months = j - trough_idx
                    recovery_date = prices.index[j].strftime("%Y-%m-%d")
                    break
            rca_advantage = corpus_arr[-1] - np.array(corpus_no_sip)[-1]
            results[key] = StressTestResult(
                scenario_name=crash["name"],
                crash_start=crash["crash_start"],
                crash_end=crash["crash_bottom"],
                recovery_date=recovery_date,
                max_drawdown_pct=round(max_dd * 100, 2),
                recovery_months=recovery_months,
                sip_rupee_cost_advantage=round(rca_advantage, 2),
            )
        except Exception as e:
            logger.warning("Stress test '%s' failed, using analytical fallback: %s", key, e)
            results[key] = StressTestResult(
                scenario_name=crash["name"],
                crash_start=crash["crash_start"],
                crash_end=crash["crash_bottom"],
                recovery_date=None,
                max_drawdown_pct=crash["drawdown_pct"],
                recovery_months=crash["recovery_months"],
                sip_rupee_cost_advantage=monthly_sip * crash["recovery_months"] * 0.3,
            )
    return results


# ============================================================================ #
#  SECTION 8 — MONTE CARLO SIMULATOR                                           #
# ============================================================================ #

def run_monte_carlo_simulation(
    monthly_sip: float,
    years: int,
    target_corpus: float,
    current_savings: float = 0.0,
    step_up_pct: float = 10.0,
    base_return_pct: float = 12.0,
    num_simulations: int = 1000,
    return_volatility_pct: float = 16.0,
) -> MonteCarloResult:
    """Monte Carlo simulation using log-normal return distribution."""
    logger.info("Running Monte Carlo | %d simulations | μ=%.1f%% | σ=%.1f%%",
                num_simulations, base_return_pct, return_volatility_pct)

    total_months = years * 12
    mu_annual = base_return_pct / 100
    sigma_annual = return_volatility_pct / 100
    mu_monthly = mu_annual / 12
    sigma_monthly = sigma_annual / np.sqrt(12)
    lognorm_mu = np.log(1 + mu_monthly) - 0.5 * sigma_monthly ** 2
    lognorm_sigma = sigma_monthly

    final_corpora: list[float] = []
    rng = np.random.default_rng(seed=42)

    for _ in range(num_simulations):
        corpus = current_savings
        current_sip = monthly_sip
        monthly_returns = rng.lognormal(lognorm_mu, lognorm_sigma, total_months)
        for i, ret in enumerate(monthly_returns):
            month = i + 1
            if step_up_pct > 0 and month > 1 and (month - 1) % 12 == 0:
                current_sip *= (1 + step_up_pct / 100)
            corpus = corpus * ret + current_sip
        final_corpora.append(corpus)

    corpora_arr = np.array(final_corpora)
    success_rate = float(np.mean(corpora_arr >= target_corpus)) * 100

    if success_rate >= 80:
        confidence = "High — Your plan is resilient."
    elif success_rate >= 60:
        confidence = "Moderate — Minor adjustments recommended."
    else:
        confidence = "Low — Significant changes needed to your SIP or timeline."

    return MonteCarloResult(
        num_simulations=num_simulations,
        success_rate_pct=round(success_rate, 1),
        median_corpus=round(float(np.median(corpora_arr)), 2),
        p10_corpus=round(float(np.percentile(corpora_arr, 10)), 2),
        p90_corpus=round(float(np.percentile(corpora_arr, 90)), 2),
        probability_shortfall_pct=round(100.0 - success_rate, 1),
        confidence_label=confidence,
    )


# ============================================================================ #
#  SECTION 9 — REPORT FORMATTER                                                #
# ============================================================================ #

def format_roadmap_for_llm(roadmap: FIRERoadmap) -> str:
    import json

    def _fmt(v: float) -> str:
        if abs(v) >= 1e7: return f"₹{v/1e7:.2f}Cr"
        elif abs(v) >= 1e5: return f"₹{v/1e5:.1f}L"
        return f"₹{v:,.0f}"

    on_track = roadmap.shortfall_surplus >= 0
    payload = {
        "summary": {
            "current_age": roadmap.current_age,
            "retirement_age": roadmap.retirement_age,
            "years_to_fire": roadmap.years_to_fire,
            "required_corpus": _fmt(roadmap.required_corpus),
            "projected_corpus": _fmt(roadmap.projected_corpus),
            "shortfall_surplus": _fmt(roadmap.shortfall_surplus),
            "on_track": on_track,
            "user_current_sip": _fmt(roadmap.user_monthly_sip),
            "required_monthly_sip": _fmt(roadmap.required_monthly_sip),
            "sip_gap": _fmt(roadmap.required_monthly_sip - roadmap.user_monthly_sip),
            "step_up_pct": roadmap.step_up_pct,
            "assumed_inflation_pct": roadmap.assumed_inflation_pct,
        },
        "allocation_today": {
            "equity_pct": roadmap.milestones[0].allocation.equity_pct if roadmap.milestones else "N/A",
            "debt_pct": roadmap.milestones[0].allocation.debt_pct if roadmap.milestones else "N/A",
            "gold_pct": roadmap.milestones[0].allocation.gold_pct if roadmap.milestones else "N/A",
        },
    }

    if roadmap.backtest_result:
        bt = roadmap.backtest_result
        payload["backtest"] = {
            "period": f"{bt.start_date} to {bt.end_date}",
            "actual_xirr_pct": bt.actual_xirr_pct,
            "benchmark_nifty_cagr_pct": bt.benchmark_cagr_pct,
            "worst_drawdown_pct": bt.worst_drawdown_pct,
            "recovery_months": bt.recovery_months,
        }

    if roadmap.stress_results:
        payload["stress_tests"] = {
            k: {
                "scenario": v.scenario_name,
                "max_drawdown_pct": v.max_drawdown_pct,
                "recovery_months": v.recovery_months,
                "rca_advantage": _fmt(v.sip_rupee_cost_advantage),
            }
            for k, v in roadmap.stress_results.items()
        }

    if roadmap.monte_carlo_result:
        mc = roadmap.monte_carlo_result
        payload["monte_carlo"] = {
            "success_rate_pct": mc.success_rate_pct,
            "confidence": mc.confidence_label,
            "median_corpus": _fmt(mc.median_corpus),
            "pessimistic_p10": _fmt(mc.p10_corpus),
            "optimistic_p90": _fmt(mc.p90_corpus),
        }

    return json.dumps(payload, indent=2, ensure_ascii=False)


# ============================================================================ #
#  SECTION 10 — SELF-TEST                                                      #
# ============================================================================ #

if __name__ == "__main__":
    print("=" * 65)
    print("  fire_planner.py — Self-Test Suite")
    print("=" * 65)

    print("\n[TEST 1] Glide Path Allocation")
    for age, retire in [(25, 55), (35, 60), (50, 60), (58, 60)]:
        alloc = get_glide_path_allocation(age, retire)
        print(f"  Age {age} -> Retire {retire}: "
              f"E={alloc.equity_pct}% D={alloc.debt_pct}% G={alloc.gold_pct}% "
              f"Blended={alloc.blended_return_pct:.1f}%")

    print("\n[TEST 2] FIRE Corpus")
    corpus = calculate_fire_corpus(60000, 30, 50)
    print(f"  Required: Rs.{corpus['required_corpus']/1e7:.2f}Cr")
    print(f"  Monthly expenses at retirement: Rs.{corpus['monthly_expenses_at_retirement']:,.0f}")

    print("\n[TEST 3] Full roadmap — user invests Rs.10,000 but needs Rs.16,565")
    roadmap = build_fire_roadmap(
        current_age=30, retirement_age=55,
        monthly_income=100000, monthly_expenses=55000,
        current_savings=200000,
        user_monthly_sip=10000,   # User's ACTUAL SIP
        run_backtest=False, run_stress_test=False, run_monte_carlo=True,
    )
    print(f"  Required SIP     : Rs.{roadmap.required_monthly_sip:,.0f}")
    print(f"  User SIP         : Rs.{roadmap.user_monthly_sip:,.0f}")
    print(f"  Required Corpus  : Rs.{roadmap.required_corpus/1e7:.2f}Cr")
    print(f"  Projected Corpus : Rs.{roadmap.projected_corpus/1e7:.2f}Cr")
    print(f"  On Track?        : {'YES' if roadmap.shortfall_surplus >= 0 else 'NO'}")
    print(f"  Shortfall        : Rs.{abs(roadmap.shortfall_surplus)/1e7:.2f}Cr")

    print("\nAll self-tests completed.")
