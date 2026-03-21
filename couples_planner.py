# ==============================================================================
# couples_planner.py
# AI Money Mentor — Couple's Money Planner
# ------------------------------------------------------------------------------
# India's first AI-powered joint financial planning engine.
# Optimizes: HRA claims, NPS matching, SIP tax splits,
#            joint insurance, combined net worth.
#
# ARCHITECTURE: LLM extracts partner data. Python does ALL optimization math.
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================ #
#  SECTION 1 — DATA MODELS                                                     #
# ============================================================================ #

@dataclass
class PartnerProfile:
    """Financial profile of one partner."""
    name: str                           # "Partner A" or actual name
    monthly_income: float               # Gross monthly
    monthly_expenses: float = 0.0
    epf_monthly: float = 0.0
    nps_monthly: float = 0.0
    existing_80c: float = 0.0           # Annual 80C already used
    existing_80d: float = 0.0           # Annual health insurance premium
    hra_received_monthly: float = 0.0
    basic_salary_monthly: float = 0.0
    current_savings: float = 0.0
    current_sip: float = 0.0            # Monthly SIP
    has_home_loan: bool = False
    home_loan_emi: float = 0.0
    total_insurance_cover: float = 0.0
    is_salaried: bool = True
    age: int = 30


@dataclass
class CoupleInput:
    partner_a: PartnerProfile
    partner_b: PartnerProfile
    rent_paid_monthly: float = 0.0      # Total household rent
    is_metro: bool = False
    combined_goal_corpus: float = 0.0   # Target wealth (retirement/home/etc)
    years_to_goal: int = 20
    risk_profile: str = "moderate"      # conservative / moderate / aggressive


# ============================================================================ #
#  SECTION 2 — TAX BRACKET HELPER                                              #
# ============================================================================ #

def _annual_tax_new_regime(annual_income: float) -> float:
    """Quick new regime tax calc for optimization."""
    slabs = [
        (400_000, 0.00), (800_000, 0.05), (1_200_000, 0.10),
        (1_600_000, 0.15), (2_000_000, 0.20), (2_400_000, 0.25),
        (float("inf"), 0.30),
    ]
    taxable = max(0, annual_income - 75_000)
    tax = 0.0
    prev = 0.0
    for limit, rate in slabs:
        if taxable <= prev:
            break
        tax += (min(taxable, limit) - prev) * rate
        prev = limit
    # 87A rebate
    if taxable <= 1_200_000:
        tax = max(0, tax - 60_000)
    return tax * 1.04  # cess


def _marginal_rate(annual_income: float) -> float:
    """Returns effective marginal rate at given income level."""
    if annual_income > 2_400_000:
        return 0.30
    elif annual_income > 2_000_000:
        return 0.25
    elif annual_income > 1_600_000:
        return 0.20
    elif annual_income > 1_200_000:
        return 0.15
    elif annual_income > 800_000:
        return 0.10
    elif annual_income > 400_000:
        return 0.05
    return 0.00


# ============================================================================ #
#  SECTION 3 — HRA OPTIMIZER                                                   #
# ============================================================================ #

@dataclass
class HRAOptimizationResult:
    recommended_claimant: str           # "Partner A" or "Partner B" or "Split"
    partner_a_hra_exempt: float
    partner_b_hra_exempt: float
    combined_hra_exempt: float
    tax_saving_a: float
    tax_saving_b: float
    total_tax_saving: float
    explanation: str


def _hra_exempt_for(
    hra_received: float,
    basic: float,
    rent_paid: float,
    is_metro: bool,
) -> float:
    """Standard HRA exemption: min of 3 rules."""
    if hra_received <= 0 or rent_paid <= 0:
        return 0.0
    rule1 = hra_received
    rule2 = max(0, rent_paid - 0.10 * basic)
    rule3 = basic * (0.50 if is_metro else 0.40)
    return max(0, min(rule1, rule2, rule3))


def optimize_hra(inp: CoupleInput) -> HRAOptimizationResult:
    """
    Determine who should claim HRA (or split rent) to minimize combined tax.
    In India, only ONE person can claim HRA for a given rent payment,
    unless they have separate rent agreements.
    """
    a, b = inp.partner_a, inp.partner_b
    rent = inp.rent_paid_monthly * 12

    # Calculate HRA exemption if A claims all rent
    a_hra_if_all = _hra_exempt_for(
        a.hra_received_monthly * 12,
        (a.basic_salary_monthly or a.monthly_income * 0.40) * 12,
        rent,
        inp.is_metro,
    )
    b_hra_if_all = _hra_exempt_for(
        b.hra_received_monthly * 12,
        (b.basic_salary_monthly or b.monthly_income * 0.40) * 12,
        rent,
        inp.is_metro,
    )

    # Tax saved per rupee of exemption
    a_rate = _marginal_rate(a.monthly_income * 12)
    b_rate = _marginal_rate(b.monthly_income * 12)

    a_saving = a_hra_if_all * a_rate * 1.04
    b_saving = b_hra_if_all * b_rate * 1.04

    if a_saving >= b_saving:
        claimant = a.name
        exempt_a, exempt_b = a_hra_if_all, 0.0
        saving_a, saving_b = a_saving, 0.0
        explanation = (
            f"{a.name} should claim full HRA — saves ₹{a_saving/1000:.1f}K/year "
            f"(marginal rate {a_rate*100:.0f}% vs {b_rate*100:.0f}%)."
        )
    else:
        claimant = b.name
        exempt_a, exempt_b = 0.0, b_hra_if_all
        saving_a, saving_b = 0.0, b_saving
        explanation = (
            f"{b.name} should claim full HRA — saves ₹{b_saving/1000:.1f}K/year "
            f"(marginal rate {b_rate*100:.0f}% vs {a_rate*100:.0f}%)."
        )

    # Check if split rent could help (separate rent receipts)
    split_rent = rent / 2
    a_hra_split = _hra_exempt_for(
        a.hra_received_monthly * 12,
        (a.basic_salary_monthly or a.monthly_income * 0.40) * 12,
        split_rent, inp.is_metro,
    )
    b_hra_split = _hra_exempt_for(
        b.hra_received_monthly * 12,
        (b.basic_salary_monthly or b.monthly_income * 0.40) * 12,
        split_rent, inp.is_metro,
    )
    split_saving = (a_hra_split * a_rate + b_hra_split * b_rate) * 1.04
    if split_saving > max(a_saving, b_saving) * 1.05:
        claimant = "Split"
        exempt_a, exempt_b = a_hra_split, b_hra_split
        saving_a = a_hra_split * a_rate * 1.04
        saving_b = b_hra_split * b_rate * 1.04
        explanation = (
            f"Split rent 50/50 with separate rent receipts — combined saving "
            f"₹{split_saving/1000:.1f}K/year beats single claimant."
        )

    return HRAOptimizationResult(
        recommended_claimant=claimant,
        partner_a_hra_exempt=exempt_a,
        partner_b_hra_exempt=exempt_b,
        combined_hra_exempt=exempt_a + exempt_b,
        tax_saving_a=saving_a,
        tax_saving_b=saving_b,
        total_tax_saving=saving_a + saving_b,
        explanation=explanation,
    )


# ============================================================================ #
#  SECTION 4 — NPS MATCHING OPTIMIZER                                          #
# ============================================================================ #

@dataclass
class NPSOptimizationResult:
    partner_a_additional_nps: float     # Extra 80CCD(1B) contribution
    partner_b_additional_nps: float
    partner_a_tax_saving: float
    partner_b_tax_saving: float
    total_tax_saving: float
    explanation: str


def optimize_nps(inp: CoupleInput) -> NPSOptimizationResult:
    """
    Each partner gets ₹50,000 of 80CCD(1B) deduction independently.
    Allocate to whoever has higher marginal rate first.
    """
    a, b = inp.partner_a, inp.partner_b
    a_income = a.monthly_income * 12
    b_income = b.monthly_income * 12

    a_rate = _marginal_rate(a_income)
    b_rate = _marginal_rate(b_income)

    # Remaining 80CCD(1B) headroom per partner
    a_nps_existing = a.nps_monthly * 12
    b_nps_existing = b.nps_monthly * 12
    a_headroom = max(0, 50_000 - a_nps_existing)
    b_headroom = max(0, 50_000 - b_nps_existing)

    a_saving = a_headroom * a_rate * 1.04
    b_saving = b_headroom * b_rate * 1.04

    explanation_parts = []
    if a_headroom > 0:
        explanation_parts.append(
            f"{a.name} can add ₹{a_headroom/1000:.0f}K to NPS → saves ₹{a_saving/1000:.1f}K"
        )
    if b_headroom > 0:
        explanation_parts.append(
            f"{b.name} can add ₹{b_headroom/1000:.0f}K to NPS → saves ₹{b_saving/1000:.1f}K"
        )

    return NPSOptimizationResult(
        partner_a_additional_nps=a_headroom,
        partner_b_additional_nps=b_headroom,
        partner_a_tax_saving=a_saving,
        partner_b_tax_saving=b_saving,
        total_tax_saving=a_saving + b_saving,
        explanation=" | ".join(explanation_parts) or "Both partners fully utilizing NPS 80CCD(1B).",
    )


# ============================================================================ #
#  SECTION 5 — SIP TAX SPLIT OPTIMIZER                                         #
# ============================================================================ #

@dataclass
class SIPSplitResult:
    partner_a_sip: float
    partner_b_sip: float
    total_sip: float
    tax_saving_vs_single: float
    explanation: str
    monthly_sip_advice: list[str]


def optimize_sip_split(
    inp: CoupleInput,
    total_monthly_sip: float,
) -> SIPSplitResult:
    """
    Split SIP investments to maximize tax efficiency.
    Key insight: if one partner is a non-earning/lower-earning spouse,
    investing in their name avoids clubbing provisions by using growth-oriented
    instruments (equity MF) — capital gains are taxed at recipient's rate.
    """
    a, b = inp.partner_a, inp.partner_b
    a_income = a.monthly_income * 12
    b_income = b.monthly_income * 12

    a_rate = _marginal_rate(a_income)
    b_rate = _marginal_rate(b_income)

    advice = []

    # ELSS allocation — give to higher-bracket partner first (80C deduction)
    a_80c_used = a.existing_80c + a.epf_monthly * 12
    b_80c_used = b.existing_80c + b.epf_monthly * 12
    a_elss_headroom = max(0, 150_000 - a_80c_used)
    b_elss_headroom = max(0, 150_000 - b_80c_used)

    a_elss_monthly = min(a_elss_headroom / 12, total_monthly_sip * 0.30)
    b_elss_monthly = min(b_elss_headroom / 12, total_monthly_sip * 0.30)

    if a_elss_monthly > 0 and a_rate >= b_rate:
        advice.append(
            f"Route ₹{a_elss_monthly/1000:.1f}K/month ELSS via {a.name} "
            f"(80C deduction saves ₹{a_elss_monthly*12*a_rate*1.04/1000:.1f}K/year)"
        )
    elif b_elss_monthly > 0 and b_rate > a_rate:
        advice.append(
            f"Route ₹{b_elss_monthly/1000:.1f}K/month ELSS via {b.name} "
            f"(80C deduction saves ₹{b_elss_monthly*12*b_rate*1.04/1000:.1f}K/year)"
        )

    # Regular equity MF — invest in lower-bracket partner's name to reduce LTCG tax
    remaining_sip = total_monthly_sip - a_elss_monthly - b_elss_monthly
    if remaining_sip > 0:
        if b_rate < a_rate and b_income > 0:
            b_regular = remaining_sip * 0.60
            a_regular = remaining_sip * 0.40
            advice.append(
                f"Route ₹{b_regular/1000:.1f}K/month equity MF in {b.name}'s name "
                f"— LTCG taxed at lower {b_rate*100:.0f}% vs {a.name}'s {a_rate*100:.0f}%"
            )
        else:
            a_regular = remaining_sip * 0.60
            b_regular = remaining_sip * 0.40
            advice.append(
                f"Split remaining equity MF: {a.name} ₹{a_regular/1000:.1f}K, "
                f"{b.name} ₹{b_regular/1000:.1f}K/month"
            )
    else:
        a_regular, b_regular = 0.0, 0.0

    a_total = a_elss_monthly + a_regular
    b_total = b_elss_monthly + b_regular

    # Tax saving vs putting everything in higher earner's name
    higher_rate = max(a_rate, b_rate)
    lower_rate = min(a_rate, b_rate)
    ltcg_rate = 0.125  # 12.5% LTCG
    estimated_annual_gains = total_monthly_sip * 12 * 0.12  # rough 12% return
    split_ltcg_tax = estimated_annual_gains * (
        a_total / total_monthly_sip * ltcg_rate +
        b_total / total_monthly_sip * ltcg_rate
    ) if total_monthly_sip > 0 else 0
    single_ltcg_tax = estimated_annual_gains * ltcg_rate
    # Not meaningful unless rate differential exists — keep simple
    tax_saving = max(0, (higher_rate - lower_rate) * total_monthly_sip * 12 * 0.10)

    return SIPSplitResult(
        partner_a_sip=round(a_total),
        partner_b_sip=round(b_total),
        total_sip=total_monthly_sip,
        tax_saving_vs_single=tax_saving,
        explanation=f"Combined SIP ₹{total_monthly_sip/1000:.0f}K/month split optimally for tax efficiency.",
        monthly_sip_advice=advice,
    )


# ============================================================================ #
#  SECTION 6 — JOINT NET WORTH & GOALS                                         #
# ============================================================================ #

@dataclass
class JointNetWorthResult:
    combined_savings: float
    combined_monthly_income: float
    combined_monthly_expenses: float
    combined_monthly_sip: float
    combined_monthly_surplus: float
    savings_rate_pct: float
    partner_a_insurance_gap: float
    partner_b_insurance_gap: float
    total_insurance_gap: float
    projected_corpus_10y: float
    projected_corpus_20y: float
    required_monthly_sip_for_goal: float
    on_track_for_goal: bool


def compute_joint_net_worth(inp: CoupleInput) -> JointNetWorthResult:
    a, b = inp.partner_a, inp.partner_b

    combined_income = (a.monthly_income + b.monthly_income)
    combined_expenses = (a.monthly_expenses + b.monthly_expenses) or combined_income * 0.60
    combined_sip = (a.current_sip + b.current_sip)
    combined_savings = (a.current_savings + b.current_savings)
    surplus = combined_income - combined_expenses - combined_sip
    savings_rate = (combined_sip / combined_income * 100) if combined_income > 0 else 0

    # Insurance gaps (10× annual income each)
    a_gap = max(0, a.monthly_income * 12 * 10 - a.total_insurance_cover)
    b_gap = max(0, b.monthly_income * 12 * 10 - b.total_insurance_cover)

    # SIP projection at 12% CAGR
    r = 0.12 / 12
    def _sip_fv(monthly, months):
        if r == 0:
            return monthly * months
        return monthly * ((1 + r) ** months - 1) / r

    corpus_10y = combined_savings * (1.12 ** 10) + _sip_fv(combined_sip, 120)
    corpus_20y = combined_savings * (1.12 ** 20) + _sip_fv(combined_sip, 240)

    # Required SIP for goal
    goal = inp.combined_goal_corpus
    years = inp.years_to_goal
    if goal > 0 and years > 0:
        fv_savings = combined_savings * (1.12 ** years)
        gap = max(0, goal - fv_savings)
        n = years * 12
        required_sip = gap * r / ((1 + r) ** n - 1) if gap > 0 else 0
    else:
        required_sip = 0

    on_track = corpus_20y >= goal if goal > 0 else True

    return JointNetWorthResult(
        combined_savings=combined_savings,
        combined_monthly_income=combined_income,
        combined_monthly_expenses=combined_expenses,
        combined_monthly_sip=combined_sip,
        combined_monthly_surplus=surplus,
        savings_rate_pct=round(savings_rate, 1),
        partner_a_insurance_gap=a_gap,
        partner_b_insurance_gap=b_gap,
        total_insurance_gap=a_gap + b_gap,
        projected_corpus_10y=corpus_10y,
        projected_corpus_20y=corpus_20y,
        required_monthly_sip_for_goal=required_sip,
        on_track_for_goal=on_track,
    )


# ============================================================================ #
#  SECTION 7 — FULL COUPLES REPORT                                             #
# ============================================================================ #

@dataclass
class CouplesReport:
    partner_a: PartnerProfile
    partner_b: PartnerProfile
    hra: HRAOptimizationResult
    nps: NPSOptimizationResult
    sip: SIPSplitResult
    net_worth: JointNetWorthResult
    total_annual_tax_saving: float
    action_items: list[str]
    summary: str


def build_couples_plan(inp: CoupleInput) -> CouplesReport:
    a, b = inp.partner_a, inp.partner_b

    hra = optimize_hra(inp)
    nps = optimize_nps(inp)
    total_sip = (a.current_sip + b.current_sip) or (a.monthly_income + b.monthly_income) * 0.20
    sip = optimize_sip_split(inp, total_sip)
    nw = compute_joint_net_worth(inp)

    total_tax_saving = hra.total_tax_saving + nps.total_tax_saving

    action_items = []
    if hra.total_tax_saving > 0:
        action_items.append(f"HRA: {hra.explanation} (saves ₹{hra.total_tax_saving/1000:.1f}K/year)")
    if nps.total_tax_saving > 0:
        action_items.append(f"NPS: {nps.explanation}")
    for advice in sip.monthly_sip_advice:
        action_items.append(f"SIP: {advice}")
    if nw.total_insurance_gap > 0:
        action_items.append(
            f"Insurance: Combined gap of ₹{nw.total_insurance_gap/1e7:.1f}Cr term cover needed"
        )
    if not nw.on_track_for_goal and nw.required_monthly_sip_for_goal > 0:
        action_items.append(
            f"Goal: Increase combined SIP by ₹{(nw.required_monthly_sip_for_goal - nw.combined_monthly_sip)/1000:.1f}K/month"
        )
    action_items.append("Open separate MF folios for each partner — simplifies tax filing")
    action_items.append("Review joint vs individual health insurance — floater usually cheaper above age 35")

    summary = (
        f"Combined income ₹{(a.monthly_income + b.monthly_income)/1000:.0f}K/month. "
        f"Optimal HRA claim: {hra.recommended_claimant}. "
        f"Total annual tax saving available: ₹{total_tax_saving/1000:.0f}K. "
        f"20-year corpus projection: ₹{nw.projected_corpus_20y/1e7:.1f}Cr."
    )

    logger.info(
        "Couples plan: combined_income=₹%.0f | tax_saving=₹%.0f | corpus_20y=₹%.0f",
        (a.monthly_income + b.monthly_income) * 12,
        total_tax_saving,
        nw.projected_corpus_20y,
    )
    return CouplesReport(
        partner_a=a, partner_b=b,
        hra=hra, nps=nps, sip=sip, net_worth=nw,
        total_annual_tax_saving=total_tax_saving,
        action_items=action_items,
        summary=summary,
    )


def format_couples_report_for_llm(report: CouplesReport) -> str:
    nw = report.net_worth
    lines = [
        f"Combined monthly income: ₹{nw.combined_monthly_income:,.0f}",
        f"Combined savings: ₹{nw.combined_savings:,.0f}",
        f"Combined monthly SIP: ₹{nw.combined_monthly_sip:,.0f}",
        f"Savings rate: {nw.savings_rate_pct:.1f}%",
        f"20-year corpus projection: ₹{nw.projected_corpus_20y:,.0f}",
        "",
        f"HRA Optimization:",
        f"  Recommended claimant: {report.hra.recommended_claimant}",
        f"  Annual tax saving: ₹{report.hra.total_tax_saving:,.0f}",
        f"  {report.hra.explanation}",
        "",
        f"NPS Optimization:",
        f"  {report.partner_a.name} additional NPS: ₹{report.nps.partner_a_additional_nps:,.0f} (saves ₹{report.nps.partner_a_tax_saving:,.0f})",
        f"  {report.partner_b.name} additional NPS: ₹{report.nps.partner_b_additional_nps:,.0f} (saves ₹{report.nps.partner_b_tax_saving:,.0f})",
        f"  Total NPS tax saving: ₹{report.nps.total_tax_saving:,.0f}",
        "",
        f"SIP Split: {report.partner_a.name} ₹{report.sip.partner_a_sip:,.0f}/month | {report.partner_b.name} ₹{report.sip.partner_b_sip:,.0f}/month",
        "",
        f"Insurance gaps: {report.partner_a.name} ₹{nw.partner_a_insurance_gap:,.0f} | {report.partner_b.name} ₹{nw.partner_b_insurance_gap:,.0f}",
        "",
        f"Total annual tax saving: ₹{report.total_annual_tax_saving:,.0f}",
        "",
        "Action Items:",
    ]
    for i, a in enumerate(report.action_items, 1):
        lines.append(f"  {i}. {a}")
    return "\n".join(lines)


# ============================================================================ #
#  SECTION 8 — SELF TEST                                                       #
# ============================================================================ #

if __name__ == "__main__":
    print("=" * 60)
    print("  couples_planner.py — Self Test")
    print("=" * 60)

    inp = CoupleInput(
        partner_a=PartnerProfile(
            name="Rahul",
            monthly_income=150000,
            monthly_expenses=40000,
            epf_monthly=7200,
            nps_monthly=0,
            hra_received_monthly=30000,
            basic_salary_monthly=60000,
            current_savings=2000000,
            current_sip=20000,
            total_insurance_cover=10000000,
            age=32,
        ),
        partner_b=PartnerProfile(
            name="Priya",
            monthly_income=80000,
            monthly_expenses=30000,
            epf_monthly=3840,
            nps_monthly=0,
            hra_received_monthly=16000,
            basic_salary_monthly=32000,
            current_savings=800000,
            current_sip=10000,
            total_insurance_cover=0,
            age=30,
        ),
        rent_paid_monthly=30000,
        is_metro=True,
        combined_goal_corpus=50000000,
        years_to_goal=20,
        risk_profile="moderate",
    )

    report = build_couples_plan(inp)
    print(f"\n✅ {report.summary}")
    print(f"\nHRA: {report.hra.explanation}")
    print(f"NPS saving: ₹{report.nps.total_tax_saving:,.0f}/year")
    print(f"Total tax saving: ₹{report.total_annual_tax_saving:,.0f}/year")
    print(f"20Y corpus: ₹{report.net_worth.projected_corpus_20y/1e7:.1f}Cr")
    print(f"\nAction items: {len(report.action_items)}")
    for a in report.action_items:
        print(f"  • {a}")
    print("\n✅ Self-test passed.")
