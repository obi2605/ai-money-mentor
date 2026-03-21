# ==============================================================================
# tax_wizard.py
# AI Money Mentor — Indian Tax Wizard
# ------------------------------------------------------------------------------
# FY 2025-26 (AY 2026-27) tax rules.
# Handles: old regime vs new regime comparison, deduction finder,
#          ranked tax-saving suggestions by risk + liquidity.
#
# ARCHITECTURE: LLM extracts salary/investment figures. Python does ALL math.
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================ #
#  SECTION 1 — TAX SLABS (FY 2025-26)                                          #
# ============================================================================ #

# New Regime slabs (default from FY 2024-25 onwards)
_NEW_REGIME_SLABS = [
    (400_000,   0.00),   # 0 – 4L: 0%
    (800_000,   0.05),   # 4L – 8L: 5%
    (1_200_000, 0.10),   # 8L – 12L: 10%
    (1_600_000, 0.15),   # 12L – 16L: 15%
    (2_000_000, 0.20),   # 16L – 20L: 20%
    (2_400_000, 0.25),   # 20L – 24L: 25%
    (float("inf"), 0.30),# > 24L: 30%
]
_NEW_REGIME_STD_DEDUCTION = 75_000
_NEW_REGIME_87A_LIMIT = 1_200_000  # Income up to ₹12L → rebate up to ₹60k
_NEW_REGIME_87A_REBATE = 60_000

# Old Regime slabs
_OLD_REGIME_SLABS = [
    (250_000,   0.00),   # 0 – 2.5L: 0%
    (500_000,   0.05),   # 2.5L – 5L: 5%
    (1_000_000, 0.20),   # 5L – 10L: 20%
    (float("inf"), 0.30),# > 10L: 30%
]
_OLD_REGIME_STD_DEDUCTION = 50_000
_OLD_REGIME_87A_LIMIT = 500_000   # Income ≤ 5L → rebate up to ₹12,500
_OLD_REGIME_87A_REBATE = 12_500

_SURCHARGE_THRESHOLDS = [
    (5_000_000,  0.0),
    (10_000_000, 0.10),
    (20_000_000, 0.15),
    (50_000_000, 0.25),
    (float("inf"), 0.37),
]
_CESS = 0.04  # Health & Education cess


def _slab_tax(taxable_income: float, slabs: list) -> float:
    """Calculate tax from progressive slabs."""
    tax = 0.0
    prev = 0.0
    for limit, rate in slabs:
        if taxable_income <= prev:
            break
        taxable_in_band = min(taxable_income, limit) - prev
        tax += taxable_in_band * rate
        prev = limit
    return tax


def _surcharge(income: float, base_tax: float) -> float:
    for limit, rate in _SURCHARGE_THRESHOLDS:
        if income <= limit:
            return base_tax * rate
    return base_tax * 0.37


def _total_tax(base_tax: float, surcharge: float) -> float:
    return (base_tax + surcharge) * (1 + _CESS)


# ============================================================================ #
#  SECTION 2 — DEDUCTION CALCULATOR                                            #
# ============================================================================ #

@dataclass
class DeductionBreakdown:
    """All deductions applicable under old regime."""
    # 80C components
    epf_employee: float = 0.0
    ppf: float = 0.0
    elss: float = 0.0
    life_insurance_premium: float = 0.0
    home_loan_principal: float = 0.0
    tuition_fees: float = 0.0
    nsc: float = 0.0
    ulip: float = 0.0
    # 80C cap
    section_80c_total: float = 0.0
    section_80c_claimed: float = 0.0  # min(total, 150000)

    # 80CCD
    nps_employee: float = 0.0         # 80CCD(1) — within 80C limit
    nps_employer: float = 0.0         # 80CCD(2) — extra, 10% of basic
    nps_additional: float = 0.0       # 80CCD(1B) — extra ₹50k over 80C

    # 80D
    health_insurance_self: float = 0.0
    health_insurance_parents: float = 0.0
    section_80d_claimed: float = 0.0  # max ₹50k self + ₹50k parents

    # HRA
    hra_received: float = 0.0
    hra_received_annual: float = 0.0
    rent_paid_annual: float = 0.0
    hra_exempt: float = 0.0

    # Others
    home_loan_interest: float = 0.0    # 24B — up to ₹2L self-occupied
    lta: float = 0.0                   # Leave Travel Allowance
    section_80e: float = 0.0           # Education loan interest (no cap)
    section_80g: float = 0.0           # Donations

    # Standard deduction (always available)
    standard_deduction: float = 50_000

    @property
    def total_old_regime(self) -> float:
        return (
            self.standard_deduction
            + self.section_80c_claimed
            + self.nps_additional
            + self.nps_employer
            + self.section_80d_claimed
            + self.hra_exempt
            + min(self.home_loan_interest, 200_000)
            + self.section_80e
        )


def calculate_deductions(
    gross_salary: float,
    epf_employee_annual: float = 0,
    ppf_annual: float = 0,
    elss_annual: float = 0,
    life_insurance_premium: float = 0,
    home_loan_principal: float = 0,
    nps_employee_annual: float = 0,
    nps_employer_annual: float = 0,
    nps_additional_annual: float = 0,
    health_insurance_self: float = 0,
    health_insurance_parents: float = 0,
    parents_senior_citizen: bool = False,
    hra_received_annual: float = 0,
    rent_paid_annual: float = 0,
    is_metro: bool = False,
    basic_salary_annual: float = 0,
    home_loan_interest: float = 0,
    education_loan_interest: float = 0,
) -> DeductionBreakdown:
    d = DeductionBreakdown()

    # 80C
    d.epf_employee = epf_employee_annual
    d.ppf = ppf_annual
    d.elss = elss_annual
    d.life_insurance_premium = life_insurance_premium
    d.home_loan_principal = home_loan_principal
    d.nps_employee = nps_employee_annual
    d.section_80c_total = (
        epf_employee_annual + ppf_annual + elss_annual +
        life_insurance_premium + home_loan_principal + nps_employee_annual
    )
    d.section_80c_claimed = min(d.section_80c_total, 150_000)

    # 80CCD
    d.nps_employer = min(nps_employer_annual, basic_salary_annual * 0.10)
    d.nps_additional = min(nps_additional_annual, 50_000)

    # 80D
    self_limit = 25_000
    parents_limit = 50_000 if parents_senior_citizen else 25_000
    d.health_insurance_self = min(health_insurance_self, self_limit)
    d.health_insurance_parents = min(health_insurance_parents, parents_limit)
    d.section_80d_claimed = d.health_insurance_self + d.health_insurance_parents

    # HRA exemption (least of 3 rules)
    d.hra_received_annual = hra_received_annual
    if hra_received_annual > 0 and rent_paid_annual > 0:
        basic = basic_salary_annual or gross_salary * 0.40
        rule1 = hra_received_annual
        rule2 = rent_paid_annual - 0.10 * basic
        rule3 = basic * (0.50 if is_metro else 0.40)
        d.hra_exempt = max(0, min(rule1, rule2, rule3))
    d.rent_paid_annual = rent_paid_annual

    # Home loan interest (24B)
    d.home_loan_interest = home_loan_interest

    # 80E — education loan (no cap)
    d.section_80e = education_loan_interest

    return d


# ============================================================================ #
#  SECTION 3 — REGIME COMPARISON                                               #
# ============================================================================ #

@dataclass
class RegimeResult:
    regime: str
    gross_salary: float
    taxable_income: float
    total_deductions: float
    base_tax: float
    surcharge: float
    cess: float
    total_tax: float
    effective_rate_pct: float
    take_home_monthly: float
    rebate_87a: float = 0.0


def compute_old_regime(gross_salary: float, deductions: DeductionBreakdown) -> RegimeResult:
    taxable = max(0, gross_salary - deductions.total_old_regime)
    base = _slab_tax(taxable, _OLD_REGIME_SLABS)

    # 87A rebate
    rebate = 0.0
    if taxable <= _OLD_REGIME_87A_LIMIT:
        rebate = min(base, _OLD_REGIME_87A_REBATE)
    base_after_rebate = max(0, base - rebate)

    sur = _surcharge(taxable, base_after_rebate)
    total = _total_tax(base_after_rebate, sur)
    cess = total - base_after_rebate - sur

    return RegimeResult(
        regime="Old Regime",
        gross_salary=gross_salary,
        taxable_income=taxable,
        total_deductions=deductions.total_old_regime,
        base_tax=base_after_rebate,
        surcharge=sur,
        cess=cess,
        total_tax=total,
        effective_rate_pct=round(total / gross_salary * 100, 2) if gross_salary > 0 else 0,
        take_home_monthly=round((gross_salary - total) / 12, 0),
        rebate_87a=rebate,
    )


def compute_new_regime(gross_salary: float) -> RegimeResult:
    taxable = max(0, gross_salary - _NEW_REGIME_STD_DEDUCTION)
    base = _slab_tax(taxable, _NEW_REGIME_SLABS)

    # 87A rebate — income up to ₹12L effectively zero tax
    rebate = 0.0
    if taxable <= _NEW_REGIME_87A_LIMIT:
        rebate = min(base, _NEW_REGIME_87A_REBATE)
    base_after_rebate = max(0, base - rebate)

    sur = _surcharge(taxable, base_after_rebate)
    total = _total_tax(base_after_rebate, sur)
    cess = total - base_after_rebate - sur

    return RegimeResult(
        regime="New Regime",
        gross_salary=gross_salary,
        taxable_income=taxable,
        total_deductions=_NEW_REGIME_STD_DEDUCTION,
        base_tax=base_after_rebate,
        surcharge=sur,
        cess=cess,
        total_tax=total,
        effective_rate_pct=round(total / gross_salary * 100, 2) if gross_salary > 0 else 0,
        take_home_monthly=round((gross_salary - total) / 12, 0),
        rebate_87a=rebate,
    )


# ============================================================================ #
#  SECTION 4 — DEDUCTION GAP FINDER                                            #
# ============================================================================ #

@dataclass
class DeductionGap:
    """A missed or under-utilised deduction."""
    section: str
    description: str
    current_utilised: float
    max_allowed: float
    gap: float
    potential_tax_saving: float  # at 30% bracket
    action: str
    priority: int


def find_deduction_gaps(
    deductions: DeductionBreakdown,
    gross_salary: float,
    tax_bracket: float = 0.30,
) -> list[DeductionGap]:
    gaps = []

    # 80C gap
    c_gap = max(0, 150_000 - deductions.section_80c_total)
    if c_gap > 0:
        gaps.append(DeductionGap(
            section="80C",
            description="Tax-saving investments (EPF, PPF, ELSS, Life Insurance, NPS, NSC)",
            current_utilised=deductions.section_80c_total,
            max_allowed=150_000,
            gap=c_gap,
            potential_tax_saving=round(c_gap * tax_bracket * 1.04),
            action=f"Invest ₹{c_gap/1000:.0f}K more in ELSS (lock-in 3Y, equity returns) or PPF (safe, 7.1%)",
            priority=1,
        ))

    # 80CCD(1B) — extra NPS deduction
    if deductions.nps_additional < 50_000:
        nps_gap = 50_000 - deductions.nps_additional
        gaps.append(DeductionGap(
            section="80CCD(1B)",
            description="Additional NPS contribution (over and above 80C limit)",
            current_utilised=deductions.nps_additional,
            max_allowed=50_000,
            gap=nps_gap,
            potential_tax_saving=round(nps_gap * tax_bracket * 1.04),
            action=f"Contribute ₹{nps_gap/1000:.0f}K more to NPS Tier 1 to claim additional ₹{round(nps_gap*tax_bracket)/1000:.0f}K tax saving",
            priority=2,
        ))

    # 80D gap
    if deductions.health_insurance_self < 25_000:
        d_gap = 25_000 - deductions.health_insurance_self
        gaps.append(DeductionGap(
            section="80D (Self)",
            description="Health insurance premium for self, spouse, children",
            current_utilised=deductions.health_insurance_self,
            max_allowed=25_000,
            gap=d_gap,
            potential_tax_saving=round(d_gap * tax_bracket * 1.04),
            action=f"Buy/upgrade health insurance — ₹{d_gap/1000:.0f}K additional premium gives ₹{round(d_gap*tax_bracket)/1000:.0f}K tax saving",
            priority=3,
        ))

    if deductions.health_insurance_parents < 25_000:
        d_gap = 25_000 - deductions.health_insurance_parents
        gaps.append(DeductionGap(
            section="80D (Parents)",
            description="Health insurance premium for parents",
            current_utilised=deductions.health_insurance_parents,
            max_allowed=50_000,
            gap=d_gap,
            potential_tax_saving=round(d_gap * tax_bracket * 1.04),
            action=f"Buy health insurance for parents — ₹{d_gap/1000:.0f}K premium gives ₹{round(d_gap*tax_bracket)/1000:.0f}K tax saving",
            priority=4,
        ))

    # HRA — check if not claiming
    if deductions.hra_received_annual > 0 and deductions.hra_exempt == 0 and deductions.rent_paid_annual == 0:
        gaps.append(DeductionGap(
            section="HRA",
            description="House Rent Allowance exemption",
            current_utilised=0,
            max_allowed=deductions.hra_received_annual,
            gap=deductions.hra_received_annual,
            potential_tax_saving=round(deductions.hra_received_annual * 0.40 * tax_bracket * 1.04),
            action="Submit rent receipts to employer. HRA exemption can save significant tax",
            priority=2,
        ))

    return sorted(gaps, key=lambda x: x.priority)


# ============================================================================ #
#  SECTION 5 — RANKED INVESTMENT SUGGESTIONS                                   #
# ============================================================================ #

@dataclass
class InvestmentSuggestion:
    name: str
    section: str
    max_deduction: float
    expected_return_pct: float
    lock_in_years: int
    risk: str             # Low / Medium / High
    liquidity: str        # Immediate / 3Y / 5Y / 15Y / Till 60
    tax_on_returns: str
    why_now: str
    rank: int


def get_ranked_suggestions(
    tax_bracket: float,
    remaining_80c: float,
    risk_profile: str = "moderate",  # conservative / moderate / aggressive
) -> list[InvestmentSuggestion]:
    """Rank tax-saving instruments by net post-tax return for the given profile."""
    all_instruments = [
        InvestmentSuggestion(
            name="ELSS Mutual Fund",
            section="80C",
            max_deduction=150_000,
            expected_return_pct=12.0,
            lock_in_years=3,
            risk="High",
            liquidity="3Y",
            tax_on_returns="10% LTCG above ₹1L/year",
            why_now="Shortest lock-in among 80C instruments. Market-linked returns historically beat inflation.",
            rank=1 if risk_profile in ("moderate", "aggressive") else 3,
        ),
        InvestmentSuggestion(
            name="NPS (80CCD 1B)",
            section="80CCD(1B)",
            max_deduction=50_000,
            expected_return_pct=10.0,
            lock_in_years=0,  # till retirement
            risk="Medium",
            liquidity="Till 60",
            tax_on_returns="60% tax-free on maturity, 40% must buy annuity",
            why_now=f"Extra ₹50K deduction OVER 80C limit. Saves ₹{round(50000 * tax_bracket * 1.04)/1000:.0f}K in tax alone.",
            rank=1,  # Always highest priority — it's ADDITIONAL deduction
        ),
        InvestmentSuggestion(
            name="PPF (Public Provident Fund)",
            section="80C",
            max_deduction=150_000,
            expected_return_pct=7.1,
            lock_in_years=15,
            risk="Low",
            liquidity="15Y (partial from 7Y)",
            tax_on_returns="Completely tax-free (EEE)",
            why_now="Triple tax-free (EEE). Best for conservative investors. Government-backed.",
            rank=2 if risk_profile == "conservative" else 3,
        ),
        InvestmentSuggestion(
            name="Term Insurance Premium",
            section="80C",
            max_deduction=150_000,
            expected_return_pct=0.0,
            lock_in_years=1,
            risk="Low",
            liquidity="Annual",
            tax_on_returns="Death benefit tax-free under 10(10D)",
            why_now="Pure protection — you need this regardless of tax savings.",
            rank=1,
        ),
        InvestmentSuggestion(
            name="Health Insurance (80D)",
            section="80D",
            max_deduction=50_000,
            expected_return_pct=0.0,
            lock_in_years=1,
            risk="Low",
            liquidity="Annual",
            tax_on_returns="N/A",
            why_now=f"Saves up to ₹{round(50000 * tax_bracket * 1.04)/1000:.0f}K in tax AND provides health cover.",
            rank=2,
        ),
        InvestmentSuggestion(
            name="Sukanya Samriddhi Yojana",
            section="80C",
            max_deduction=150_000,
            expected_return_pct=8.2,
            lock_in_years=21,
            risk="Low",
            liquidity="21Y",
            tax_on_returns="Completely tax-free (EEE)",
            why_now="8.2% tax-free for girl child. Better than PPF returns.",
            rank=2,
        ),
    ]

    # Filter to relevant (only show instruments with headroom)
    relevant = []
    for s in all_instruments:
        if s.section == "80C" and remaining_80c <= 0 and s.name not in ("Term Insurance Premium",):
            continue
        relevant.append(s)

    return sorted(relevant, key=lambda x: x.rank)


# ============================================================================ #
#  SECTION 6 — FULL TAX REPORT                                                 #
# ============================================================================ #

@dataclass
class TaxReport:
    gross_annual_income: float
    old_regime: RegimeResult
    new_regime: RegimeResult
    recommended_regime: str
    tax_saving_by_switching: float      # how much more to save by choosing better regime
    deductions: DeductionBreakdown
    deduction_gaps: list[DeductionGap]
    suggestions: list[InvestmentSuggestion]
    total_potential_additional_saving: float  # if all gaps filled
    tax_bracket_pct: float


def build_tax_report(
    gross_annual_income: float,
    # Salary structure
    basic_salary_annual: float = 0,
    hra_received_annual: float = 0,
    # Existing investments
    epf_employee_annual: float = 0,
    ppf_annual: float = 0,
    elss_annual: float = 0,
    life_insurance_premium: float = 0,
    nps_employee_annual: float = 0,
    nps_employer_annual: float = 0,
    nps_additional_annual: float = 0,
    health_insurance_self: float = 0,
    health_insurance_parents: float = 0,
    parents_senior_citizen: bool = False,
    rent_paid_annual: float = 0,
    is_metro: bool = False,
    home_loan_principal: float = 0,
    home_loan_interest: float = 0,
    education_loan_interest: float = 0,
    risk_profile: str = "moderate",
) -> TaxReport:

    # Calculate deductions
    deductions = calculate_deductions(
        gross_salary=gross_annual_income,
        epf_employee_annual=epf_employee_annual,
        ppf_annual=ppf_annual,
        elss_annual=elss_annual,
        life_insurance_premium=life_insurance_premium,
        home_loan_principal=home_loan_principal,
        nps_employee_annual=nps_employee_annual,
        nps_employer_annual=nps_employer_annual,
        nps_additional_annual=nps_additional_annual,
        health_insurance_self=health_insurance_self,
        health_insurance_parents=health_insurance_parents,
        parents_senior_citizen=parents_senior_citizen,
        hra_received_annual=hra_received_annual,
        rent_paid_annual=rent_paid_annual,
        is_metro=is_metro,
        basic_salary_annual=basic_salary_annual,
        home_loan_interest=home_loan_interest,
        education_loan_interest=education_loan_interest,
    )

    old = compute_old_regime(gross_annual_income, deductions)
    new = compute_new_regime(gross_annual_income)

    recommended = "Old Regime" if old.total_tax < new.total_tax else "New Regime"
    saving = abs(old.total_tax - new.total_tax)

    # Tax bracket for gap analysis
    taxable_old = old.taxable_income
    if taxable_old > 1_000_000:
        bracket = 0.30
    elif taxable_old > 500_000:
        bracket = 0.20
    else:
        bracket = 0.05

    gaps = find_deduction_gaps(deductions, gross_annual_income, bracket)
    remaining_80c = max(0, 150_000 - deductions.section_80c_total)
    suggestions = get_ranked_suggestions(bracket, remaining_80c, risk_profile)
    total_potential = sum(g.potential_tax_saving for g in gaps)

    report = TaxReport(
        gross_annual_income=gross_annual_income,
        old_regime=old,
        new_regime=new,
        recommended_regime=recommended,
        tax_saving_by_switching=saving,
        deductions=deductions,
        deduction_gaps=gaps,
        suggestions=suggestions,
        total_potential_additional_saving=total_potential,
        tax_bracket_pct=bracket * 100,
    )
    logger.info(
        "Tax report: income=₹%.0f | old=₹%.0f | new=₹%.0f | recommend=%s | gaps=%d | potential_saving=₹%.0f",
        gross_annual_income, old.total_tax, new.total_tax,
        recommended, len(gaps), total_potential,
    )
    return report


def format_tax_report_for_llm(report: TaxReport) -> str:
    """Serialise for LLM narrative generation."""
    o, n = report.old_regime, report.new_regime
    lines = [
        f"Gross Annual Income: ₹{report.gross_annual_income:,.0f}",
        f"\nOld Regime:",
        f"  Total deductions: ₹{o.total_deductions:,.0f}",
        f"  Taxable income: ₹{o.taxable_income:,.0f}",
        f"  Total tax: ₹{o.total_tax:,.0f} (effective rate: {o.effective_rate_pct:.1f}%)",
        f"  Monthly take-home: ₹{o.take_home_monthly:,.0f}",
        f"\nNew Regime:",
        f"  Standard deduction: ₹75,000",
        f"  Taxable income: ₹{n.taxable_income:,.0f}",
        f"  Total tax: ₹{n.total_tax:,.0f} (effective rate: {n.effective_rate_pct:.1f}%)",
        f"  Monthly take-home: ₹{n.take_home_monthly:,.0f}",
        f"\nRECOMMENDATION: {report.recommended_regime} saves ₹{report.tax_saving_by_switching:,.0f}/year",
        f"\nDeduction Gaps (missed savings):",
    ]
    for g in report.deduction_gaps:
        lines.append(
            f"  {g.section}: Gap of ₹{g.gap:,.0f} — potential saving ₹{g.potential_tax_saving:,.0f} — {g.action}"
        )
    lines.append(f"\nTotal potential additional tax saving: ₹{report.total_potential_additional_saving:,.0f}")
    return "\n".join(lines)


# ============================================================================ #
#  SECTION 7 — SELF TEST                                                       #
# ============================================================================ #

if __name__ == "__main__":
    print("=" * 60)
    print("  tax_wizard.py — Self Test")
    print("=" * 60)

    # Test case: ₹18L salary, partial 80C, no NPS
    report = build_tax_report(
        gross_annual_income=1_800_000,
        basic_salary_annual=720_000,
        hra_received_annual=360_000,
        epf_employee_annual=86_400,   # 12% of basic
        life_insurance_premium=30_000,
        health_insurance_self=15_000,
        rent_paid_annual=240_000,
        is_metro=True,
        risk_profile="moderate",
    )

    print(f"\nGross: ₹{report.gross_annual_income/1e5:.1f}L")
    print(f"Old Regime Tax: ₹{report.old_regime.total_tax:,.0f} "
          f"(effective {report.old_regime.effective_rate_pct:.1f}%)")
    print(f"New Regime Tax: ₹{report.new_regime.total_tax:,.0f} "
          f"(effective {report.new_regime.effective_rate_pct:.1f}%)")
    print(f"Recommended: {report.recommended_regime} "
          f"(saves ₹{report.tax_saving_by_switching:,.0f})")
    print(f"\nDeduction gaps ({len(report.deduction_gaps)}):")
    for g in report.deduction_gaps:
        print(f"  {g.section}: ₹{g.gap:,.0f} gap → saves ₹{g.potential_tax_saving:,.0f}")
    print(f"\nTotal potential saving: ₹{report.total_potential_additional_saving:,.0f}")
    print("\n✅ Self-test passed.")
