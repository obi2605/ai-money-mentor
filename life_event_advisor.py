# ==============================================================================
# life_event_advisor.py
# AI Money Mentor — Life Event Financial Advisor
# ------------------------------------------------------------------------------
# Handles 6 life events with deterministic allocation logic:
#   BONUS | INHERITANCE | MARRIAGE | NEW_BABY | JOB_LOSS | HOME_PURCHASE
#
# ARCHITECTURE: LLM extracts event + amounts. Python does all allocation math.
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================ #
#  SECTION 1 — EVENT TYPES & PARAMS                                            #
# ============================================================================ #

class LifeEventType(str, Enum):
    BONUS           = "BONUS"
    INHERITANCE     = "INHERITANCE"
    MARRIAGE        = "MARRIAGE"
    NEW_BABY        = "NEW_BABY"
    JOB_LOSS        = "JOB_LOSS"
    HOME_PURCHASE   = "HOME_PURCHASE"


@dataclass
class LifeEventInput:
    """Extracted parameters for a life event."""
    event_type: LifeEventType

    # Financial context
    monthly_income: float = 0.0
    monthly_expenses: float = 0.0
    current_savings: float = 0.0
    current_emergency_fund: float = 0.0
    has_term_insurance: bool = False
    total_insurance_cover: float = 0.0
    existing_sip: float = 0.0

    # Event-specific
    event_amount: float = 0.0           # Bonus / inheritance / home price
    tax_bracket_pct: float = 30.0       # 10 / 20 / 30
    has_home_loan: bool = False
    home_loan_outstanding: float = 0.0
    num_dependents: int = 0             # Including spouse/children
    years_to_retirement: int = 20


@dataclass
class AllocationItem:
    """A single allocation recommendation."""
    label: str
    amount: float
    pct: float
    rationale: str
    priority: int                       # 1 = highest


@dataclass
class LifeEventResult:
    """Full output of the life event advisor."""
    event_type: LifeEventType
    event_amount: float

    # Core allocation plan
    allocations: list[AllocationItem] = field(default_factory=list)

    # Flags and warnings
    insurance_gap: float = 0.0          # Additional cover needed (INR)
    emergency_fund_gap: float = 0.0     # Additional liquid fund needed
    tax_liability: float = 0.0          # Estimated tax on event amount
    tax_tip: str = ""

    # Derived
    total_allocated: float = 0.0
    action_items: list[str] = field(default_factory=list)
    summary: str = ""


# ============================================================================ #
#  SECTION 2 — ALLOCATION LOGIC (one function per event)                       #
# ============================================================================ #

def _recommend_insurance(inp: LifeEventInput) -> float:
    """
    Returns recommended ADDITIONAL term cover needed.
    Benchmark: 10× annual income, less existing cover.
    """
    recommended = inp.monthly_income * 12 * 10
    gap = max(0, recommended - inp.total_insurance_cover)
    return gap


def _emergency_fund_target(inp: LifeEventInput) -> float:
    """6 months expenses = ideal emergency fund."""
    return inp.monthly_expenses * 6


def _tax_on_bonus(bonus: float, bracket: float) -> float:
    """Bonus is taxed as regular income at marginal rate."""
    return bonus * (bracket / 100)


def plan_bonus(inp: LifeEventInput) -> LifeEventResult:
    """
    Allocate a salary bonus optimally.
    Priority: Tax reserve → Emergency fund top-up → High-interest debt →
              Tax-saving investments (80C/NPS) → Equity SIP top-up → Goals
    """
    result = LifeEventResult(event_type=inp.event_type, event_amount=inp.event_amount)
    remaining = inp.event_amount

    if inp.event_amount <= 0:
        result.tax_tip = "Please share your bonus amount so I can create a specific allocation plan."
        result.summary = "Tell me your bonus amount and I'll show you exactly how to deploy it."
        return result

    # 1. Tax reserve (30% bracket = TDS already deducted usually, but top-up for advance tax)
    tax = _tax_on_bonus(inp.event_amount, inp.tax_bracket_pct)
    # Assume TDS deducted at source — only flag if bracket is 30%
    result.tax_liability = tax
    result.tax_tip = (
        f"Your bonus will be taxed at {inp.tax_bracket_pct:.0f}% (₹{tax/1e5:.1f}L). "
        f"TDS is usually deducted at source. File advance tax if bonus exceeds ₹10,000 "
        f"additional liability for the quarter."
    )

    # 2. Emergency fund top-up
    ef_target = _emergency_fund_target(inp)
    ef_gap = max(0, ef_target - inp.current_emergency_fund)
    ef_contribution = min(ef_gap, remaining * 0.30)
    if ef_contribution > 0:
        result.allocations.append(AllocationItem(
            label="Emergency Fund Top-up",
            amount=ef_contribution,
            pct=round(ef_contribution / inp.event_amount * 100, 1),
            rationale=f"Build to 6-month target of ₹{ef_target/1e5:.1f}L",
            priority=1,
        ))
        remaining -= ef_contribution
        result.emergency_fund_gap = ef_gap - ef_contribution

    # 3. High-interest debt (if home loan outstanding)
    if inp.home_loan_outstanding > 0:
        prepay = min(remaining * 0.25, inp.home_loan_outstanding * 0.10)
        result.allocations.append(AllocationItem(
            label="Home Loan Prepayment",
            amount=prepay,
            pct=round(prepay / inp.event_amount * 100, 1),
            rationale="10% prepayment reduces interest burden significantly",
            priority=2,
        ))
        remaining -= prepay

    # 4. 80C top-up (ELSS for best risk/return in tax-saving category)
    annual_80c_used = inp.existing_sip * 12 * 0.5  # Rough estimate
    elss_headroom = max(0, 150000 - annual_80c_used)
    elss_invest = min(elss_headroom, remaining * 0.20)
    if elss_invest > 0 and inp.tax_bracket_pct >= 20:
        result.allocations.append(AllocationItem(
            label="ELSS (80C Tax Saving)",
            amount=elss_invest,
            pct=round(elss_invest / inp.event_amount * 100, 1),
            rationale=f"Save ₹{elss_invest * inp.tax_bracket_pct / 100 / 1000:.0f}K in taxes via 80C",
            priority=3,
        ))
        remaining -= elss_invest

    # 5. NPS 80CCD(1B) — extra ₹50k deduction
    if inp.tax_bracket_pct >= 20:
        nps_top = min(50000, remaining * 0.10)
        result.allocations.append(AllocationItem(
            label="NPS (80CCD 1B top-up)",
            amount=nps_top,
            pct=round(nps_top / inp.event_amount * 100, 1),
            rationale=f"Additional ₹50k deduction saves ₹{50000 * inp.tax_bracket_pct / 100 / 1000:.0f}K in tax",
            priority=4,
        ))
        remaining -= nps_top

    # 6. Rest into equity mutual funds
    if remaining > 0:
        result.allocations.append(AllocationItem(
            label="Equity Mutual Funds (lump sum)",
            amount=remaining,
            pct=round(remaining / inp.event_amount * 100, 1),
            rationale="Long-term wealth creation via diversified equity",
            priority=5,
        ))

    result.insurance_gap = _recommend_insurance(inp)
    result.total_allocated = inp.event_amount
    result.action_items = [
        f"Check TDS certificate — bonus should have TDS deducted at {inp.tax_bracket_pct:.0f}%",
        "Invest ELSS before 31st March to claim 80C deduction this FY",
        "Use STP (Systematic Transfer Plan) for equity lump sum — invest in liquid fund, transfer ₹50k/month to equity",
    ]
    if result.insurance_gap > 0:
        result.action_items.append(
            f"Buy ₹{result.insurance_gap/1e7:.1f}Cr term plan — current cover is below 10× income benchmark"
        )
    result.summary = (
        f"Your ₹{inp.event_amount/1e5:.1f}L bonus is best split across "
        f"emergency fund top-up, tax-saving investments, and equity MFs. "
        f"Tax liability: ₹{tax/1e5:.1f}L (likely already deducted as TDS)."
    )
    return result


def plan_inheritance(inp: LifeEventInput) -> LifeEventResult:
    """
    Large windfall — conservative allocation given emotional context.
    Don't rush. Stagger equity entry. Prioritize stability first.
    """
    result = LifeEventResult(event_type=inp.event_type, event_amount=inp.event_amount)
    amt = inp.event_amount

    # 1. Liquid parking (12-month FD or liquid fund) — take time to decide
    liquid_park = amt * 0.30
    result.allocations.append(AllocationItem(
        label="Short-term FD / Liquid Fund (parking)",
        amount=liquid_park,
        pct=30.0,
        rationale="Park for 3-6 months while you decide. Never rush inheritance allocation.",
        priority=1,
    ))

    # 2. Emergency fund (if not adequate)
    ef_gap = max(0, _emergency_fund_target(inp) - inp.current_emergency_fund)
    ef_fill = min(ef_gap, amt * 0.10)
    if ef_fill > 0:
        result.allocations.append(AllocationItem(
            label="Emergency Fund",
            amount=ef_fill,
            pct=round(ef_fill / amt * 100, 1),
            rationale="Complete 6-month emergency fund",
            priority=2,
        ))

    # 3. Debt clearance
    if inp.home_loan_outstanding > 0:
        debt_clear = min(inp.home_loan_outstanding, amt * 0.20)
        result.allocations.append(AllocationItem(
            label="Loan Prepayment",
            amount=debt_clear,
            pct=round(debt_clear / amt * 100, 1),
            rationale="Becoming debt-free is the best guaranteed return",
            priority=3,
        ))

    # 4. Equity via STP (staggered over 12 months)
    remaining = amt - sum(a.amount for a in result.allocations)
    equity_via_stp = remaining * 0.50
    result.allocations.append(AllocationItem(
        label="Equity MF via STP (12-month stagger)",
        amount=equity_via_stp,
        pct=round(equity_via_stp / amt * 100, 1),
        rationale="Stagger entry over 12 months to average cost — never lump sum a windfall",
        priority=4,
    ))

    # 5. Debt/Gold for stability
    remaining -= equity_via_stp
    if remaining > 0:
        result.allocations.append(AllocationItem(
            label="Debt MF / SGBs",
            amount=remaining,
            pct=round(remaining / amt * 100, 1),
            rationale="Portfolio stability and inflation hedge",
            priority=5,
        ))

    result.tax_tip = (
        "Inheritance is NOT taxable in India (no inheritance tax). "
        "However, any income earned FROM the inherited amount is taxable. "
        "Consult a CA to document the source of funds properly."
    )
    result.action_items = [
        "Do NOT invest immediately — park in liquid fund for at least 30 days",
        "Consult a CA to document inheritance source (crucial for future ITR)",
        "Update your will to include this new wealth",
        "Review term insurance — your net worth has increased significantly",
    ]
    result.insurance_gap = _recommend_insurance(inp)
    result.total_allocated = amt
    result.summary = (
        f"A ₹{amt/1e7:.1f}Cr inheritance is life-changing. "
        f"Park it first, plan carefully, and stagger equity entry over 12 months."
    )
    return result


def plan_marriage(inp: LifeEventInput) -> LifeEventResult:
    """Marriage triggers insurance review, joint goal planning, expense re-assessment."""
    result = LifeEventResult(event_type=inp.event_type, event_amount=inp.event_amount)

    # Insurance is the #1 priority after marriage
    insurance_gap = _recommend_insurance(inp)
    result.insurance_gap = insurance_gap

    # Wedding cost allocation (if event_amount represents wedding budget)
    if inp.event_amount > 0:
        amt = inp.event_amount
        # Max 30% on wedding — rest goes to new life setup
        wedding_spend = min(amt * 0.30, amt)
        result.allocations.append(AllocationItem(
            label="Wedding Expenses",
            amount=wedding_spend,
            pct=round(wedding_spend / amt * 100, 1),
            rationale="Cap at 30% of available budget — don't start married life in debt",
            priority=1,
        ))
        emergency_fund = amt * 0.20
        result.allocations.append(AllocationItem(
            label="Joint Emergency Fund",
            amount=emergency_fund,
            pct=20.0,
            rationale="Build joint 6-month buffer — combined expenses will increase",
            priority=2,
        ))
        home_goal = amt * 0.30
        result.allocations.append(AllocationItem(
            label="Home Down Payment Fund",
            amount=home_goal,
            pct=30.0,
            rationale="Start STP into balanced hybrid fund for 3-5 year home goal",
            priority=3,
        ))
        remaining = amt - wedding_spend - emergency_fund - home_goal
        if remaining > 0:
            result.allocations.append(AllocationItem(
                label="Equity MF (long-term goals)",
                amount=remaining,
                pct=round(remaining / amt * 100, 1),
                rationale="Joint wealth creation — equity for 10+ year horizon",
                priority=4,
            ))

    result.tax_tip = (
        "After marriage: (1) Add spouse as nominee on all policies and MF folios. "
        "(2) If spouse has no income, invest in their name to split tax liability. "
        "(3) HRA — only one spouse can claim; choose the one with higher rent paid."
    )
    result.action_items = [
        f"Buy ₹{insurance_gap/1e7:.1f}Cr term plan immediately — you now have dependents",
        "Add spouse as nominee on all bank accounts, MFs, insurance policies",
        "Open a joint account for shared expenses — keep individual accounts too",
        "Start joint SIP for home down payment goal",
        "Review and update health insurance — add spouse to floater plan",
    ]
    result.total_allocated = inp.event_amount
    result.summary = (
        "Marriage is India's most expensive financial event. "
        "Prioritize insurance and a joint emergency fund before any other investment."
    )
    return result


def plan_new_baby(inp: LifeEventInput) -> LifeEventResult:
    """New baby triggers education corpus, term insurance, and emergency fund increase."""
    result = LifeEventResult(event_type=inp.event_type, event_amount=inp.event_amount)

    # Education corpus calculation
    # Assume ₹20L today for graduation, inflating at 10% for 18 years
    education_corpus_today = 2_000_000
    education_corpus_future = education_corpus_today * (1.10 ** 18)
    # Monthly SIP needed at 12% CAGR for 18 years
    r = 0.12 / 12
    n = 18 * 12
    education_sip = education_corpus_future * r / ((1 + r) ** n - 1)

    # Emergency fund should increase (new dependent)
    new_ef_target = inp.monthly_expenses * 9  # 9 months with baby
    ef_gap = max(0, new_ef_target - inp.current_emergency_fund)

    # Insurance gap (life cover must increase)
    new_insurance_needed = inp.monthly_income * 12 * 15  # 15× income with child
    insurance_gap = max(0, new_insurance_needed - inp.total_insurance_cover)
    result.insurance_gap = insurance_gap

    if inp.event_amount > 0:
        amt = inp.event_amount
        # Initial hospital/setup costs
        baby_setup = min(amt * 0.20, 200000)
        result.allocations.append(AllocationItem(
            label="Hospital & Baby Setup",
            amount=baby_setup,
            pct=round(baby_setup / amt * 100, 1),
            rationale="Medical costs + first-year essentials",
            priority=1,
        ))
        # Emergency fund top-up
        ef_top = min(ef_gap, amt * 0.30)
        result.allocations.append(AllocationItem(
            label="Emergency Fund (9-month target)",
            amount=ef_top,
            pct=round(ef_top / amt * 100, 1),
            rationale="Increase from 6 to 9 months with new dependent",
            priority=2,
        ))
        # Education corpus seed
        edu_seed = amt * 0.40
        result.allocations.append(AllocationItem(
            label="Child Education Corpus (seed)",
            amount=edu_seed,
            pct=40.0,
            rationale=f"Start SIP of ₹{education_sip/1000:.1f}K/mo. Need ₹{education_corpus_future/1e7:.1f}Cr in 18Y",
            priority=3,
        ))
        remaining = amt - baby_setup - ef_top - edu_seed
        if remaining > 0:
            result.allocations.append(AllocationItem(
                label="Equity MF (child's future)",
                amount=remaining,
                pct=round(remaining / amt * 100, 1),
                rationale="Long-term wealth for child's milestones",
                priority=4,
            ))

    result.tax_tip = (
        "New baby tax benefits: (1) Medical insurance premium for child — claim under 80D. "
        "(2) Sukanya Samriddhi (if girl child) — 8.2% tax-free, 80C eligible. "
        "(3) Tuition fees — claim under 80C for up to 2 children."
    )
    result.action_items = [
        f"Buy additional ₹{insurance_gap/1e7:.1f}Cr term plan immediately — child needs 20 years of income protection" if insurance_gap > 0 else "Get a term insurance quote — with a child, you need at least 15× annual income in cover",
        f"Start ₹{education_sip/1000:.1f}K/month SIP in Flexi Cap or Mid Cap fund for education corpus",
        "Add child as dependent on health insurance floater",
        "If girl child: open Sukanya Samriddhi account (8.2% tax-free, 80C eligible)",
        "Write or update your will immediately",
    ]
    result.total_allocated = inp.event_amount
    result.summary = (
        f"Congratulations! A child needs ₹{education_corpus_future/1e7:.1f}Cr for education in 18 years. "
        f"Start a ₹{education_sip/1000:.1f}K/month SIP today and buy term insurance immediately."
    )
    return result


def plan_job_loss(inp: LifeEventInput) -> LifeEventResult:
    """Job loss — cash flow crisis management, prioritize survival over growth."""
    result = LifeEventResult(event_type=inp.event_type, event_amount=inp.event_amount)

    # Runway calculation
    monthly_burn = inp.monthly_expenses
    emergency_fund_months = (
        inp.current_emergency_fund / monthly_burn
        if monthly_burn > 0 else 0
    )

    # Severance allocation (event_amount = severance pay)
    if inp.event_amount > 0:
        amt = inp.event_amount

        # 1. Extend runway — park in liquid fund
        result.allocations.append(AllocationItem(
            label="Liquid Fund (survival runway)",
            amount=amt * 0.70,
            pct=70.0,
            rationale=f"Park in liquid fund — current runway is {emergency_fund_months:.1f} months. Target: 12 months",
            priority=1,
        ))

        # 2. Keep SIPs running if possible
        sip_3months = inp.existing_sip * 3
        sip_reserve = min(sip_3months, amt * 0.15)
        result.allocations.append(AllocationItem(
            label="SIP Reserve (3 months)",
            amount=sip_reserve,
            pct=round(sip_reserve / amt * 100, 1),
            rationale="Keep SIPs running — stopping SIPs during market dips is the worst move",
            priority=2,
        ))

        # 3. Upskilling budget
        upskill = min(50000, amt * 0.10)
        result.allocations.append(AllocationItem(
            label="Upskilling / Job Search",
            amount=upskill,
            pct=round(upskill / amt * 100, 1),
            rationale="Invest in skills to accelerate re-employment",
            priority=3,
        ))

        remaining = amt - amt * 0.70 - sip_reserve - upskill
        if remaining > 0:
            result.allocations.append(AllocationItem(
                label="Buffer (misc expenses)",
                amount=remaining,
                pct=round(remaining / amt * 100, 1),
                rationale="Unexpected costs during job search",
                priority=4,
            ))

    result.tax_tip = (
        "Job loss tax tips: (1) Gratuity up to ₹20L is tax-free. "
        "(2) Leave encashment up to ₹25L is tax-free for private sector. "
        "(3) File ITR even for partial year income — may get refund if TDS was over-deducted."
    )
    result.action_items = [
        f"You have ~{emergency_fund_months:.1f} months of runway — target 12 months before panic" if emergency_fund_months > 0 else "⚠️ No emergency fund — this is critical. Park 70% of severance in liquid fund immediately",
        "Pause (don't stop) SIPs if needed — reduce amount, never stop entirely",
        "File for ESIC unemployment benefit if eligible",
        "Switch to individual health insurance immediately — don't lose cover",
        "Update LinkedIn and start networking TODAY — most jobs come through referrals",
    ]
    result.total_allocated = inp.event_amount
    result.summary = (
        f"With ₹{inp.current_emergency_fund/1e5:.1f}L in emergency funds, "
        f"you have {emergency_fund_months:.1f} months of runway. "
        f"Focus on survival first, job search second, wealth creation third."
    )
    return result


def plan_home_purchase(inp: LifeEventInput) -> LifeEventResult:
    """Home purchase — down payment strategy, EMI affordability, tax benefits."""
    result = LifeEventResult(event_type=inp.event_type, event_amount=inp.event_amount)

    home_price = inp.event_amount
    recommended_down_payment = home_price * 0.20  # 20% down payment
    loan_amount = home_price - recommended_down_payment
    # Affordable EMI = max 30% of take-home
    take_home = inp.monthly_income * 0.75
    max_emi = take_home * 0.30
    # EMI at 8.5% for 20 years
    r = 0.085 / 12
    n = 240
    emi_for_loan = loan_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)

    is_affordable = emi_for_loan <= max_emi

    result.allocations.append(AllocationItem(
        label="Down Payment (20%)",
        amount=recommended_down_payment,
        pct=20.0,
        rationale="20% down reduces loan amount and avoids PMI. Never go below 15%.",
        priority=1,
    ))
    result.allocations.append(AllocationItem(
        label="Registration & Stamp Duty (est. 7%)",
        amount=home_price * 0.07,
        pct=7.0,
        rationale="Stamp duty + registration: budget 6-8% of property value",
        priority=2,
    ))
    result.allocations.append(AllocationItem(
        label="Interior & Shifting",
        amount=home_price * 0.05,
        pct=5.0,
        rationale="Budget 4-6% for interior, furniture, and moving costs",
        priority=3,
    ))

    result.tax_tip = (
        f"Home loan tax benefits: "
        f"(1) Section 24B: ₹2L deduction on interest for self-occupied property. "
        f"(2) Section 80C: ₹1.5L deduction on principal repayment. "
        f"(3) First-time buyer: 80EEA — additional ₹1.5L on interest (if eligible). "
        f"Total potential saving: ₹{(200000 + 150000) * inp.tax_bracket_pct / 100 / 1000:.0f}K/year."
    )

    emi_warning = ""
    if not is_affordable:
        emi_warning = (
            f"⚠️ EMI of ₹{emi_for_loan/1000:.1f}K exceeds your 30% affordability limit "
            f"(₹{max_emi/1000:.1f}K). Consider a smaller property or larger down payment."
        )

    result.action_items = [
        f"Down payment needed: ₹{recommended_down_payment/1e5:.1f}L (20% of ₹{home_price/1e7:.1f}Cr)",
        f"Estimated EMI: ₹{emi_for_loan/1000:.1f}K/month at 8.5% for 20 years",
        emi_warning if emi_warning else f"EMI is within affordability limit ✅",
        "Get pre-approved for home loan before negotiating property price",
        "Keep 6-month emergency fund INTACT — don't use it for down payment",
    ]
    result.action_items = [a for a in result.action_items if a]

    result.total_allocated = recommended_down_payment + home_price * 0.12
    result.insurance_gap = _recommend_insurance(inp)
    result.summary = (
        f"For a ₹{home_price/1e7:.1f}Cr home: down payment ₹{recommended_down_payment/1e5:.1f}L, "
        f"loan ₹{loan_amount/1e7:.1f}Cr, EMI ~₹{emi_for_loan/1000:.1f}K/month. "
        f"{'Affordable ✅' if is_affordable else 'Consider a smaller property ⚠️'}"
    )
    return result


# ============================================================================ #
#  SECTION 3 — MAIN DISPATCHER                                                 #
# ============================================================================ #

def build_life_event_plan(inp: LifeEventInput) -> LifeEventResult:
    """Route to the appropriate event planner."""
    dispatch = {
        LifeEventType.BONUS:          plan_bonus,
        LifeEventType.INHERITANCE:    plan_inheritance,
        LifeEventType.MARRIAGE:       plan_marriage,
        LifeEventType.NEW_BABY:       plan_new_baby,
        LifeEventType.JOB_LOSS:       plan_job_loss,
        LifeEventType.HOME_PURCHASE:  plan_home_purchase,
    }
    fn = dispatch.get(inp.event_type)
    if not fn:
        raise ValueError(f"Unknown life event: {inp.event_type}")
    result = fn(inp)
    logger.info(
        "Life event plan: %s | amount=₹%.0f | allocations=%d | insurance_gap=₹%.0f",
        inp.event_type, inp.event_amount, len(result.allocations), result.insurance_gap,
    )
    return result


def format_life_event_for_llm(result: LifeEventResult) -> str:
    """Serialise result for LLM narrative generation."""
    lines = [
        f"Event: {result.event_type.value}",
        f"Amount: ₹{result.event_amount:,.0f}",
        f"Summary: {result.summary}",
        "",
        "Allocation Plan:",
    ]
    for a in sorted(result.allocations, key=lambda x: x.priority):
        lines.append(
            f"  {a.priority}. {a.label}: ₹{a.amount:,.0f} ({a.pct:.1f}%) — {a.rationale}"
        )
    if result.tax_liability > 0:
        lines.append(f"\nTax Liability: ₹{result.tax_liability:,.0f}")
    lines.append(f"Tax Tip: {result.tax_tip}")
    if result.insurance_gap > 0:
        lines.append(f"Insurance Gap: ₹{result.insurance_gap:,.0f} additional cover needed")
    if result.emergency_fund_gap > 0:
        lines.append(f"Emergency Fund Gap: ₹{result.emergency_fund_gap:,.0f}")
    lines.append("\nAction Items:")
    for i, action in enumerate(result.action_items, 1):
        lines.append(f"  {i}. {action}")
    return "\n".join(lines)


# ============================================================================ #
#  SECTION 4 — SELF TEST                                                       #
# ============================================================================ #

if __name__ == "__main__":
    print("=" * 60)
    print("  life_event_advisor.py — Self Test")
    print("=" * 60)

    # Test 1: Bonus
    bonus_inp = LifeEventInput(
        event_type=LifeEventType.BONUS,
        monthly_income=150000,
        monthly_expenses=80000,
        current_savings=3000000,
        current_emergency_fund=300000,
        event_amount=500000,
        tax_bracket_pct=30,
        existing_sip=25000,
    )
    r = build_life_event_plan(bonus_inp)
    print(f"\n✅ BONUS: {r.summary}")
    for a in r.allocations:
        print(f"   {a.label}: ₹{a.amount:,.0f} ({a.pct}%)")

    # Test 2: New Baby
    baby_inp = LifeEventInput(
        event_type=LifeEventType.NEW_BABY,
        monthly_income=100000,
        monthly_expenses=60000,
        current_emergency_fund=400000,
        total_insurance_cover=5000000,
        event_amount=300000,
    )
    r2 = build_life_event_plan(baby_inp)
    print(f"\n✅ NEW BABY: {r2.summary}")

    # Test 3: Job Loss
    jl_inp = LifeEventInput(
        event_type=LifeEventType.JOB_LOSS,
        monthly_expenses=70000,
        current_emergency_fund=500000,
        existing_sip=20000,
        event_amount=400000,  # severance
    )
    r3 = build_life_event_plan(jl_inp)
    print(f"\n✅ JOB LOSS: {r3.summary}")

    print("\n✅ All self-tests passed.")
