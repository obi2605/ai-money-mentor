# ==============================================================================
# quant_engine.py
# AI Money Mentor — Deterministic Quantitative Engine
# ==============================================================================

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import numpy_financial as npf
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ============================================================================ #
#  SECTION 1 — DATA CLASSES                                                    #
# ============================================================================ #

@dataclass
class XIRRResult:
    xirr_annual: float
    xirr_pct: float
    num_cashflows: int
    period_years: float


@dataclass
class RollingReturnResult:
    ticker: str
    period_label: str
    cagr_pct: float
    total_return_pct: float
    start_date: str
    end_date: str
    start_nav: float
    end_nav: float
    max_drawdown_pct: float
    volatility_annualised_pct: float


@dataclass
class SIPProjectionResult:
    monthly_sip: float
    target_corpus: float
    years: int
    assumed_cagr_pct: float
    projected_corpus: float
    shortfall_surplus: float
    monthly_schedule: pd.DataFrame


# ============================================================================ #
#  SECTION 2 — XIRR CALCULATOR                                                 #
# ============================================================================ #

def _xirr_npv(rate: float, dates: list[date], cashflows: list[float]) -> float:
    t0 = dates[0]
    return sum(
        cf / (1 + rate) ** ((d - t0).days / 365.0)
        for d, cf in zip(dates, cashflows)
    )


def calculate_xirr(
    dates: list[date],
    cashflows: list[float],
    bracket: tuple[float, float] = (-0.999, 100.0),
) -> XIRRResult:
    """
    Calculate XIRR using scipy.optimize.brentq for guaranteed convergence.
    Investments are NEGATIVE (outflows); redemption/current value is POSITIVE.
    """
    if len(dates) != len(cashflows):
        raise ValueError(f"dates and cashflows must have equal length.")
    if len(dates) < 2:
        raise ValueError("XIRR requires at least 2 cash flows.")
    if not any(cf > 0 for cf in cashflows):
        raise ValueError("XIRR requires at least one positive cash flow.")
    if not any(cf < 0 for cf in cashflows):
        raise ValueError("XIRR requires at least one negative cash flow.")

    paired = sorted(zip(dates, cashflows), key=lambda x: x[0])
    sorted_dates, sorted_cfs = zip(*paired)
    sorted_dates = list(sorted_dates)
    sorted_cfs = list(sorted_cfs)

    try:
        rate = brentq(
            _xirr_npv,
            bracket[0],
            bracket[1],
            args=(sorted_dates, sorted_cfs),
            xtol=1e-8,
            maxiter=1000,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not calculate XIRR — verify the redemption value and investment dates."
        ) from exc

    period_years = (sorted_dates[-1] - sorted_dates[0]).days / 365.0
    return XIRRResult(
        xirr_annual=rate,
        xirr_pct=round(rate * 100, 2),
        num_cashflows=len(sorted_cfs),
        period_years=round(period_years, 2),
    )


# ============================================================================ #
#  SECTION 3 — YFINANCE HISTORICAL RETURN FETCHER                              #
# ============================================================================ #

INDIAN_TICKER_MAP: dict[str, str] = {
    "nifty50":                   "^NSEI",
    "nifty 50":                  "^NSEI",
    "sensex":                    "^BSESN",
    "nifty bank":                "^NSEBANK",
    "niftybank":                 "^NSEBANK",
    "nifty it":                  "^CNXIT",
    "gold":                      "GC=F",
    "sgb":                       "GC=F",
    "nippon india etf nifty 50": "NIFTYBEES.NS",
    "sbi nifty 50 etf":          "SETFNIF50.NS",
    "hdfc nifty 50 etf":         "HDFCNIFETF.NS",
}

# Fallback chains: ^NSEI intermittently returns empty JSON from Yahoo Finance.
TICKER_FALLBACKS: dict[str, list[str]] = {
    "^NSEI":    ["^NSEI", "NIFTYBEES.NS", "SETFNIF50.NS"],
    "^BSESN":   ["^BSESN", "SENSEXBEES.NS"],
    "^NSEBANK": ["^NSEBANK", "BANKBEES.NS"],
}

VALID_PERIODS = {"1Y", "3Y", "5Y", "7Y", "10Y"}


def _load_cached_return(ticker: str, period: str) -> Optional[RollingReturnResult]:
    """
    Load a pre-cached benchmark return from data/benchmark_returns.json.
    Last-resort fallback when yfinance is completely unavailable.
    """
    cache_path = os.path.join(os.path.dirname(__file__), "data", "benchmark_returns.json")
    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
    except Exception as e:
        logger.warning("Could not load benchmark_returns.json: %s", e)
        return None

    all_candidates = [ticker] + TICKER_FALLBACKS.get(ticker, [])
    for candidate in all_candidates:
        entry = cache.get(candidate)
        if entry and period in entry.get("periods", {}):
            p = entry["periods"][period]
            logger.info("Using CACHED data for %s %s (yfinance unavailable)", candidate, period)
            return RollingReturnResult(
                ticker=candidate + " [cached]",
                period_label=period,
                cagr_pct=p["cagr_pct"],
                total_return_pct=p["total_return_pct"],
                start_date=p["start_date"],
                end_date=p["end_date"],
                start_nav=p["start_nav"],
                end_nav=p["end_nav"],
                max_drawdown_pct=p["max_drawdown_pct"],
                volatility_annualised_pct=p["volatility_annualised_pct"],
            )
    return None


def fetch_historical_rolling_return(
    ticker_or_alias: str,
    period: str = "5Y",
) -> RollingReturnResult:
    """
    Fetch historical price data for an Indian index/ETF.
    Priority order:
      1. Live yfinance data (primary ticker)
      2. Live yfinance data (fallback tickers)
      3. Cached data from benchmark_returns.json
    """
    period = period.upper().strip()
    if period not in VALID_PERIODS:
        raise ValueError(f"Period '{period}' is not supported. Choose from: {VALID_PERIODS}")

    resolved_ticker = INDIAN_TICKER_MAP.get(
        ticker_or_alias.lower().strip(), ticker_or_alias.strip()
    )
    fallback_chain = TICKER_FALLBACKS.get(resolved_ticker, [resolved_ticker])

    yf_period_map = {
        "1Y": "1y", "3Y": "3y", "5Y": "5y", "7Y": "max", "10Y": "max"
    }
    yf_period = yf_period_map[period]

    raw = None
    used_ticker = resolved_ticker

    for candidate in fallback_chain:
        logger.info("Trying ticker '%s' for period %s", candidate, period)
        try:
            data = yf.download(
                candidate,
                period=yf_period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=15,
            )
            if data is not None and not data.empty:
                raw = data
                used_ticker = candidate
                logger.info("Successfully fetched data for '%s'", candidate)
                break
            else:
                logger.warning("Empty response for '%s', trying next fallback", candidate)
        except Exception as exc:
            logger.warning("Ticker '%s' failed: %s — trying next fallback", candidate, exc)

    if raw is None or raw.empty:
        # Last resort: serve from local cache
        cached = _load_cached_return(resolved_ticker, period)
        if cached:
            return cached
        raise RuntimeError(
            f"Could not fetch market data for '{resolved_ticker}' or any fallback "
            f"in {fallback_chain}. Check your internet connection or try again later."
        )

    # For 7Y / 10Y, manually trim to requested lookback
    year_map = {"7Y": 7, "10Y": 10}
    if period in year_map:
        cutoff = pd.Timestamp.today() - pd.DateOffset(years=year_map[period])
        raw = raw[raw.index >= cutoff]
        if raw.empty:
            cached = _load_cached_return(resolved_ticker, period)
            if cached:
                return cached
            raise RuntimeError(f"Insufficient history for '{used_ticker}' over {period}.")

    # Flatten MultiIndex columns (yfinance >=0.2.x)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    close = raw["Close"].dropna()

    if len(close) < 2:
        raise RuntimeError(f"Insufficient data points for '{used_ticker}'.")

    start_nav = float(close.iloc[0])
    end_nav = float(close.iloc[-1])
    start_date = close.index[0].strftime("%Y-%m-%d")
    end_date = close.index[-1].strftime("%Y-%m-%d")

    actual_years = (close.index[-1] - close.index[0]).days / 365.25
    if actual_years <= 0:
        raise RuntimeError("Date range too short to compute CAGR.")

    cagr = (end_nav / start_nav) ** (1 / actual_years) - 1
    total_return = (end_nav / start_nav) - 1

    rolling_max = close.cummax()
    drawdown = (close - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    log_returns = np.log(close / close.shift(1)).dropna()
    annualised_vol = float(log_returns.std() * np.sqrt(252))

    result = RollingReturnResult(
        ticker=used_ticker,
        period_label=period,
        cagr_pct=round(cagr * 100, 2),
        total_return_pct=round(total_return * 100, 2),
        start_date=start_date,
        end_date=end_date,
        start_nav=round(start_nav, 2),
        end_nav=round(end_nav, 2),
        max_drawdown_pct=round(max_drawdown * 100, 2),
        volatility_annualised_pct=round(annualised_vol * 100, 2),
    )
    logger.info(
        "Fetched %s | CAGR: %.2f%% | MaxDD: %.2f%% | Vol: %.2f%%",
        used_ticker, result.cagr_pct, result.max_drawdown_pct,
        result.volatility_annualised_pct,
    )
    return result


# ============================================================================ #
#  SECTION 4 — SIP PROJECTION ENGINE                                           #
# ============================================================================ #

def project_sip_corpus(
    monthly_sip: float,
    years: int,
    assumed_cagr_pct: float,
    target_corpus: Optional[float] = None,
    step_up_pct: float = 0.0,
) -> SIPProjectionResult:
    """Project month-by-month SIP growth with optional annual step-up."""
    if monthly_sip <= 0:
        raise ValueError("Monthly SIP must be a positive value.")
    if years <= 0 or years > 50:
        raise ValueError("Investment horizon must be between 1 and 50 years.")
    if not (1.0 <= assumed_cagr_pct <= 50.0):
        raise ValueError(f"Assumed CAGR {assumed_cagr_pct}% is outside the valid range (1-50%).")

    monthly_rate = assumed_cagr_pct / 100 / 12
    total_months = years * 12
    records = []
    corpus = 0.0
    current_sip = monthly_sip
    total_invested = 0.0

    for month in range(1, total_months + 1):
        if step_up_pct > 0 and month > 1 and (month - 1) % 12 == 0:
            current_sip = current_sip * (1 + step_up_pct / 100)
        corpus = corpus * (1 + monthly_rate) + current_sip
        total_invested += current_sip
        records.append({
            "month": month,
            "year": (month - 1) // 12 + 1,
            "monthly_sip": round(current_sip, 2),
            "corpus_value": round(corpus, 2),
            "total_invested": round(total_invested, 2),
            "gains": round(corpus - total_invested, 2),
        })

    schedule_df = pd.DataFrame(records)
    projected_corpus = round(corpus, 2)
    shortfall_surplus = round(projected_corpus - target_corpus, 2) if target_corpus else 0.0

    return SIPProjectionResult(
        monthly_sip=monthly_sip,
        target_corpus=target_corpus or 0.0,
        years=years,
        assumed_cagr_pct=assumed_cagr_pct,
        projected_corpus=projected_corpus,
        shortfall_surplus=shortfall_surplus,
        monthly_schedule=schedule_df,
    )


# ============================================================================ #
#  SECTION 5 — MONEY HEALTH SCORE ENGINE                                       #
# ============================================================================ #

def calculate_money_health_score(
    monthly_income: float,
    monthly_expenses: float,
    emergency_fund: float,
    total_insurance_cover: float,
    total_debt_emi: float,
    equity_pct: float,
    debt_pct: float,
    gold_pct: float,
    other_pct: float,
    epf_ppf_nps_monthly: float,
    tax_saving_investments: float,
    gross_annual_income: float,
) -> dict:
    """
    Score across 6 SEBI-aligned dimensions. Fully deterministic — no LLM.
    Weights: Emergency 20%, Insurance 20%, Diversity 15%, Debt 20%, Tax 10%, Retirement 15%
    """
    scores: dict[str, float] = {}

    # 1. Emergency Preparedness (6-month benchmark)
    recommended_ef = monthly_expenses * 6
    ef_ratio = min(emergency_fund / recommended_ef, 1.0) if recommended_ef > 0 else 0.0
    scores["emergency_preparedness"] = round(ef_ratio * 100, 1)

    # 2. Insurance Coverage (10x annual income benchmark)
    recommended_cover = gross_annual_income * 10
    ins_ratio = min(total_insurance_cover / recommended_cover, 1.0) if recommended_cover > 0 else 0.0
    scores["insurance_coverage"] = round(ins_ratio * 100, 1)

    # 3. Investment Diversification (ideal: 60% equity, 30% debt, 10% gold)
    if (equity_pct + debt_pct + gold_pct + other_pct) > 0:
        deviations = (
            abs(equity_pct - 60) + abs(debt_pct - 30)
            + abs(gold_pct - 10) + abs(other_pct - 0)
        )
        div_score = max(0.0, 1.0 - (deviations / 200.0))
    else:
        div_score = 0.0
    scores["investment_diversity"] = round(div_score * 100, 1)

    # 4. Debt Health (EMI/income < 30%)
    if monthly_income > 0:
        emi_ratio = total_debt_emi / monthly_income
        if emi_ratio <= 0.30:
            debt_score = 1.0
        elif emi_ratio <= 0.50:
            debt_score = 1.0 - ((emi_ratio - 0.30) / 0.20)
        else:
            debt_score = 0.0
    else:
        debt_score = 0.0
    scores["debt_health"] = round(debt_score * 100, 1)

    # 5. Tax Efficiency (80C ₹1.5L + 80D ₹25K = ₹1.75L max)
    max_tax_saving = 175000
    tax_ratio = min(tax_saving_investments / max_tax_saving, 1.0) if max_tax_saving > 0 else 0.0
    scores["tax_efficiency"] = round(tax_ratio * 100, 1)

    # 6. Retirement Readiness (15% of gross income benchmark)
    annual_retirement_savings = epf_ppf_nps_monthly * 12
    recommended_retirement = gross_annual_income * 0.15
    ret_ratio = min(annual_retirement_savings / recommended_retirement, 1.0) if recommended_retirement > 0 else 0.0
    scores["retirement_readiness"] = round(ret_ratio * 100, 1)

    weights = {
        "emergency_preparedness": 0.20,
        "insurance_coverage":     0.20,
        "investment_diversity":   0.15,
        "debt_health":            0.20,
        "tax_efficiency":         0.10,
        "retirement_readiness":   0.15,
    }
    composite = sum(scores[k] * weights[k] for k in weights)

    return {
        "dimensions": scores,
        "composite_score": round(composite, 1),
        "grade": _score_to_grade(composite),
    }


def _score_to_grade(score: float) -> str:
    if score >= 85:   return "A+ (Excellent)"
    elif score >= 70: return "A (Good)"
    elif score >= 55: return "B (Fair)"
    elif score >= 40: return "C (Needs Attention)"
    else:             return "D (Critical)"


# ============================================================================ #
#  SECTION 6 — SELF-TEST                                                       #
# ============================================================================ #

if __name__ == "__main__":
    print("=" * 60)
    print("  quant_engine.py — Self-Test Suite")
    print("=" * 60)

    print("\n[TEST 1] XIRR Calculation")
    cf_dates = [
        date(2022, 1, 1), date(2022, 4, 1),
        date(2022, 7, 1), date(2022, 10, 1), date(2023, 1, 1),
    ]
    cashflows = [-10000, -10000, -10000, -10000, 48000]
    xirr_res = calculate_xirr(cf_dates, cashflows)
    print(f"  XIRR: {xirr_res.xirr_pct:.2f}%  |  Period: {xirr_res.period_years:.2f}Y")

    print("\n[TEST 2] Nifty 50 — 5Y Rolling Return (with cache fallback)")
    try:
        ret = fetch_historical_rolling_return("nifty50", period="5Y")
        print(f"  Ticker: {ret.ticker}")
        print(f"  CAGR  : {ret.cagr_pct:.2f}%")
        print(f"  MaxDD : {ret.max_drawdown_pct:.2f}%")
        print(f"  Vol   : {ret.volatility_annualised_pct:.2f}%")
    except RuntimeError as e:
        print(f"  [WARN] {e}")

    print("\n[TEST 3] SIP Projection — Rs.10,000/mo | 10Y | 12% | 10% step-up")
    sip_res = project_sip_corpus(
        monthly_sip=10000, years=10, assumed_cagr_pct=12.0,
        target_corpus=2500000, step_up_pct=10.0,
    )
    print(f"  Projected: Rs.{sip_res.projected_corpus:,.0f}")
    print(f"  Surplus  : Rs.{sip_res.shortfall_surplus:,.0f}")

    print("\n[TEST 4] Money Health Score")
    health = calculate_money_health_score(
        monthly_income=80000, monthly_expenses=50000,
        emergency_fund=200000, total_insurance_cover=5000000,
        total_debt_emi=20000, equity_pct=70, debt_pct=20,
        gold_pct=5, other_pct=5, epf_ppf_nps_monthly=8000,
        tax_saving_investments=100000, gross_annual_income=960000,
    )
    print(f"  Score: {health['composite_score']} / 100  -> {health['grade']}")

    print("\nAll self-tests completed.")
