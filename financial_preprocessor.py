# ==============================================================================
# financial_preprocessor.py
# AI Money Mentor — Deterministic Financial Figure Pre-Processor
# ------------------------------------------------------------------------------
# PURPOSE:
#   Before calling the LLM extractor, scan ALL conversation messages with
#   deterministic regex to identify financial figures and their labels.
#   The output is a structured dict of "verified facts" that is prepended
#   to the extraction prompt — so the LLM is told the pre-computed totals
#   rather than having to calculate them from ambiguous natural language.
#
# This prevents two classes of errors:
#   1. Confusing income/salary figures with expense figures
#   2. Picking a single asset instead of summing all assets
#
# ARCHITECTURE: Pure Python, no LLM, no network. Always fast.
# ==============================================================================

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================ #
#  SECTION 1 — NUMBER PARSER                                                   #
# ============================================================================ #

def parse_inr(text: str) -> Optional[float]:
    """
    Parse an Indian currency expression into a float (INR).
    Handles: 1L, 10L, 1.5Cr, 80k, 2,50,000, ₹5 lakh, 70 lakhs, etc.
    Returns None if no valid number found.
    """
    text = text.strip().lower()
    text = text.replace(",", "").replace("₹", "").replace("rs.", "").replace("rs ", "")
    text = text.replace("rupees", "").replace("inr", "").strip()

    # Match patterns like: 1.5cr, 70l, 80k, 2 lakh, 1 crore
    patterns = [
        (r"([\d.]+)\s*(?:cr|crore|crores)", 1e7),
        (r"([\d.]+)\s*(?:l\b|lakh|lakhs|lac|lacs)", 1e5),
        (r"([\d.]+)\s*(?:k\b|thousand)", 1e3),
        (r"^([\d.]+)$", 1),
    ]
    for pattern, multiplier in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return float(m.group(1)) * multiplier
            except ValueError:
                continue
    return None


def parse_inr_from_context(amount_str: str, unit_hint: str = "") -> Optional[float]:
    """
    Parse a number that may lack an explicit unit, using context clues.
    e.g. "55" in "55 lakhs are in NPS" → 5500000
    """
    combined = (amount_str + " " + unit_hint).strip()
    result = parse_inr(combined)
    if result is not None:
        return result
    # Try the amount alone
    return parse_inr(amount_str)


# ============================================================================ #
#  SECTION 2 — ASSET PATTERN DEFINITIONS                                       #
# ============================================================================ #

# Each entry: (regex_pattern, asset_category)
# Categories: equity | debt | gold | retirement | other
ASSET_PATTERNS = [
    # Mutual Funds (equity by default unless specified as debt)
    (r"([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))\s*(?:in\s+)?(?:mutual\s+funds?|mf\b|elss)", "equity"),
    (r"(?:mutual\s+funds?|mf\b|elss)\s+(?:of\s+|worth\s+|totaling\s+)?([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))", "equity"),

    # Stocks
    (r"([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))\s*(?:in\s+)?(?:stocks?|equit(?:y|ies)|shares?)", "equity"),

    # NPS corpus (not monthly contribution)
    (r"([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))\s*(?:in\s+|are\s+in\s+)?nps\b", "retirement"),
    (r"nps\s+(?:corpus|balance|value|amount)?\s*(?:of\s+|is\s+|:)?\s*([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))", "retirement"),

    # EPF / PF corpus (not monthly)
    (r"([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))\s*(?:in\s+)?(?:epf|pf\b|provident\s+fund)", "retirement"),
    (r"(?:epf|pf\b|provident\s+fund)\s+(?:corpus|balance|value|amount|of)\s+([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))", "retirement"),
    # "PF of 80L"
    (r"(?:pf|epf)\s+of\s+([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))", "retirement"),

    # PPF
    (r"([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))\s*(?:in\s+)?ppf\b", "debt"),
    (r"ppf\s+(?:of\s+|balance\s+|corpus\s+)?([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))", "debt"),

    # FD / Fixed Deposit
    (r"(?:fd|fixed\s+deposit)\s+(?:of\s+|worth\s+)?([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))", "debt"),
    (r"([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))\s+(?:fd|fixed\s+deposit)", "debt"),
    # "FD of 1.5 crore for 2 years"
    (r"(?:fd|fixed\s+deposit)\s+of\s+([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))\s+for", "debt"),

    # SGBs / Sovereign Gold Bonds
    (r"([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))\s*(?:in\s+)?(?:sgb|sovereign\s+gold\s+bonds?)", "gold"),
    (r"(?:sgb|sovereign\s+gold\s+bonds?)\s+(?:of\s+|worth\s+)?([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))", "gold"),

    # Gold ETF / Physical Gold
    (r"([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))\s*(?:in\s+)?(?:gold\s+etf|gold\s+fund|physical\s+gold)", "gold"),

    # Real Estate
    (r"([\d.,]+\s*(?:cr|crore|crores|l\b|lakh|lakhs|k\b))\s*(?:in\s+)?(?:real\s+estate|property|house|flat)", "other"),
]

# Monthly contribution patterns — these are NOT assets, they are flows
# We use these to populate epf_ppf_nps_monthly, NOT current_savings
MONTHLY_CONTRIBUTION_PATTERNS = [
    # "60k per month goes into PF" / "60k goes to PF"
    (r"([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)\s*(?:per\s+month|pm|monthly|/month|a\s+month)?\s*(?:goes?\s+(?:into|to|towards|toward)|contribut\w+\s+to)?\s*(?:my\s+)?(?:pf|epf|provident\s+fund)", "epf"),
    (r"(?:pf|epf)\s+(?:contribution|contrib|amount)\s+(?:is|of|=)?\s*([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)", "epf"),

    # "24k pm goes towards NPS"
    (r"([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)\s*(?:per\s+month|pm|monthly|/month|a\s+month)?\s*(?:goes?\s+(?:into|to|towards|toward)|contribut\w+\s+to)?\s*(?:my\s+)?nps", "nps"),
    (r"nps\s+(?:contribution|contrib|amount)\s+(?:is|of|=)?\s*([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)", "nps"),

    # PPF monthly
    (r"([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)\s*(?:per\s+month|pm|monthly)\s*(?:in|into|to)?\s*ppf", "ppf"),
]

# Expense patterns — income figures are explicitly excluded
EXPENSE_PATTERNS = [
    # "expenses are 80k" / "spend 80k/month" / "monthly expenses of 80k"
    (r"(?:monthly\s+)?(?:expenses?|expenditure|outgo|spending|spend)\s+(?:is|are|of|=|around|about|approx)?\s*([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)", "expense"),
    (r"([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)\s*(?:per\s+month|pm|monthly|/month)?\s*(?:in\s+)?(?:expenses?|spending|outgo)", "expense"),
    (r"spend\s+([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)\s*(?:per\s+month|pm|monthly|/month|a\s+month)?", "expense"),
    (r"(?:my\s+)?(?:living\s+)?(?:cost\s+of\s+living|monthly\s+cost|cost\s+per\s+month)\s+(?:is\s+)?([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)", "expense"),
    # "around 80k is my monthly expense" / "80k is my expense"
    (r"(?:around|about|approx\.?)?\s*([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b))\s+is\s+(?:my\s+)?(?:monthly\s+)?(?:expenses?|expenditure|spending)", "expense"),
    # "around 80k" / "about 1L" as standalone expense answer (no keyword needed)
    (r"^(?:around|about|approx\.?|roughly)?\s*([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b))\s*\.?\s*(?:i\s+have|i\s+also|my|$)", "expense"),
]

# Income patterns — explicitly separate from expense
INCOME_PATTERNS = [
    (r"(?:earn|salary|income|take\s*home|ctc|gross)\s*[₹rs.]?\s*(?:is\s+|of\s+|=\s*)?([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)", "income"),
    (r"([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)\s*(?:per\s+month|pm|monthly|/month|a\s+month)\s+(?:salary|income|earning|take\s*home)", "income"),
    (r"(?:i\s+earn|earning)\s*[₹rs.]?\s*([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)", "income"),
    # "₹25L salary" / "salary of ₹18L"
    (r"[₹rs.]\s*([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b))\s*(?:salary|income|per\s+month|pm|monthly|/month|per\s+year|annually|ctc|gross)", "income"),
]

# Insurance annual premium — not sum assured
INSURANCE_PREMIUM_PATTERNS = [
    (r"([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)\s*(?:per\s+year|py|annually|yearly|p\.a\.)\s*(?:in\s+)?(?:ins[ue]a?r[ae]nce|premium)", "insurance_premium"),
    (r"(?:ins[ue]a?r[ae]nce)\s+premium\s+(?:of\s+|is\s+|=\s*)?([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)\s*(?:per\s+year|annually|yearly)?", "insurance_premium"),
    (r"([\d.,]+\s*(?:cr|crore|l\b|lakh|lakhs|k\b)?)\s*(?:per\s+year|py|annually|yearly)\s*(?:in\s+|towards?\s+|for\s+)?(?:ins[ue]a?r[ae]nce|premium)", "insurance_premium"),
]


# ============================================================================ #
#  SECTION 3 — CONVERSATION SCANNER                                            #
# ============================================================================ #

@dataclass
class ExtractedFacts:
    """Pre-processed deterministic facts from conversation."""
    # Assets by category (INR)
    equity_assets: dict[str, float] = field(default_factory=dict)   # label → amount
    debt_assets: dict[str, float] = field(default_factory=dict)
    gold_assets: dict[str, float] = field(default_factory=dict)
    retirement_assets: dict[str, float] = field(default_factory=dict)
    other_assets: dict[str, float] = field(default_factory=dict)

    # Monthly flows
    monthly_income: Optional[float] = None
    monthly_expenses: Optional[float] = None
    monthly_epf: float = 0.0
    monthly_nps: float = 0.0
    monthly_ppf: float = 0.0
    monthly_sip: Optional[float] = None

    # Annual
    annual_insurance_premium: float = 0.0

    @property
    def total_equity(self) -> float:
        return sum(self.equity_assets.values())

    @property
    def total_debt(self) -> float:
        return sum(self.debt_assets.values())

    @property
    def total_gold(self) -> float:
        return sum(self.gold_assets.values())

    @property
    def total_retirement(self) -> float:
        return sum(self.retirement_assets.values())

    @property
    def total_other(self) -> float:
        return sum(self.other_assets.values())

    @property
    def total_savings(self) -> float:
        return (self.total_equity + self.total_debt + self.total_gold
                + self.total_retirement + self.total_other)

    @property
    def equity_pct(self) -> float:
        t = self.total_savings
        return round(self.total_equity / t * 100, 1) if t > 0 else 0.0

    @property
    def debt_pct(self) -> float:
        t = self.total_savings
        return round((self.total_debt + self.total_retirement) / t * 100, 1) if t > 0 else 0.0

    @property
    def gold_pct(self) -> float:
        t = self.total_savings
        return round(self.total_gold / t * 100, 1) if t > 0 else 0.0

    @property
    def monthly_retirement_contributions(self) -> float:
        return self.monthly_epf + self.monthly_nps + self.monthly_ppf

    @property
    def annual_tax_saving(self) -> float:
        """80C/80D eligible: EPF×12 + NPS×12 + PPF×12 + insurance premium."""
        return (self.monthly_epf * 12 + self.monthly_nps * 12
                + self.monthly_ppf * 12 + self.annual_insurance_premium)

    def to_context_string(self) -> str:
        """
        Serialise as a verified facts block to prepend to extraction prompts.
        This tells the LLM: "use THESE numbers, don't recompute from scratch."
        """
        lines = ["[VERIFIED FACTS — extracted by deterministic parser — USE THESE EXACTLY]"]

        if self.monthly_income:
            lines.append(f"Monthly income: ₹{self.monthly_income:,.0f}")
        if self.monthly_expenses:
            lines.append(f"Monthly expenses: ₹{self.monthly_expenses:,.0f}")
        if self.monthly_sip:
            lines.append(f"Monthly SIP: ₹{self.monthly_sip:,.0f}")

        if self.total_savings > 0:
            lines.append(f"\nAssets breakdown:")
            if self.equity_assets:
                for k, v in self.equity_assets.items():
                    lines.append(f"  Equity — {k}: ₹{v:,.0f}")
            if self.debt_assets:
                for k, v in self.debt_assets.items():
                    lines.append(f"  Debt — {k}: ₹{v:,.0f}")
            if self.retirement_assets:
                for k, v in self.retirement_assets.items():
                    lines.append(f"  Retirement corpus — {k}: ₹{v:,.0f}")
            if self.gold_assets:
                for k, v in self.gold_assets.items():
                    lines.append(f"  Gold — {k}: ₹{v:,.0f}")
            if self.other_assets:
                for k, v in self.other_assets.items():
                    lines.append(f"  Other — {k}: ₹{v:,.0f}")
            lines.append(f"  TOTAL current_savings = ₹{self.total_savings:,.0f} "
                         f"({self.total_savings/1e7:.2f} Cr)")
            lines.append(f"  equity_pct={self.equity_pct}% | "
                         f"debt_pct={self.debt_pct}% | gold_pct={self.gold_pct}%")

        if self.monthly_retirement_contributions > 0:
            lines.append(f"\nMonthly retirement contributions:")
            if self.monthly_epf: lines.append(f"  EPF/PF: ₹{self.monthly_epf:,.0f}/mo")
            if self.monthly_nps: lines.append(f"  NPS: ₹{self.monthly_nps:,.0f}/mo")
            if self.monthly_ppf: lines.append(f"  PPF: ₹{self.monthly_ppf:,.0f}/mo")
            lines.append(f"  TOTAL epf_ppf_nps_monthly = ₹{self.monthly_retirement_contributions:,.0f}")

        if self.annual_tax_saving > 0:
            lines.append(f"\nAnnual tax-saving investments (80C/80D):")
            lines.append(f"  EPF×12: ₹{self.monthly_epf*12:,.0f}")
            lines.append(f"  NPS×12: ₹{self.monthly_nps*12:,.0f}")
            if self.annual_insurance_premium:
                lines.append(f"  Insurance premium: ₹{self.annual_insurance_premium:,.0f}")
            lines.append(f"  TOTAL tax_saving_investments = ₹{self.annual_tax_saving:,.0f}")

        lines.append("[END VERIFIED FACTS — do NOT recalculate these from the conversation]")
        return "\n".join(lines)


def scan_conversation(messages: list[dict]) -> ExtractedFacts:
    """
    Scan all user messages in the conversation and extract financial figures
    deterministically using regex patterns.

    Only processes role='user' messages — assistant messages contain
    our own computed outputs which would corrupt the extraction.

    Returns ExtractedFacts with all identified figures.
    """
    facts = ExtractedFacts()

    # Combine all user messages into one searchable text
    user_text = " ".join(
        msg["content"].lower()
        for msg in messages
        if msg.get("role") == "user"
    )

    logger.info("Pre-processing conversation (%d chars)", len(user_text))

    # ── Extract income ─────────────────────────────────────────────────────── #
    income_candidates = []
    for pattern, _ in INCOME_PATTERNS:
        for m in re.finditer(pattern, user_text, re.IGNORECASE):
            val = parse_inr(m.group(1))
            if val and val > 10000:  # Filter out noise
                income_candidates.append(val)

    if income_candidates:
        # Take the largest income figure mentioned (gross > take-home)
        facts.monthly_income = max(income_candidates)

    # ── Extract expenses ───────────────────────────────────────────────────── #
    expense_candidates = []
    for pattern, _ in EXPENSE_PATTERNS:
        for m in re.finditer(pattern, user_text, re.IGNORECASE):
            val = parse_inr(m.group(1))
            if val and val > 1000:
                expense_candidates.append(val)

    if expense_candidates:
        # Take the most recently mentioned expense figure
        # (user may correct it — we want the latest)
        facts.monthly_expenses = expense_candidates[-1]

    # ── Extract monthly retirement contributions ───────────────────────────── #
    for pattern, contrib_type in MONTHLY_CONTRIBUTION_PATTERNS:
        for m in re.finditer(pattern, user_text, re.IGNORECASE):
            raw = m.group(1).strip()
            # These are monthly amounts — usually K range
            val = parse_inr(raw)
            if val is None:
                # Try as bare number (e.g. "60" in "60k")
                try:
                    val = float(re.sub(r"[^\d.]", "", raw))
                    if val < 1000:  # Likely in thousands — e.g. "60" means 60k
                        val *= 1000
                except ValueError:
                    continue

            if val and 1000 <= val <= 500000:  # Sanity check: ₹1k–₹5L/month
                if contrib_type == "epf" and facts.monthly_epf == 0:
                    facts.monthly_epf = val
                    logger.info("Found monthly EPF: ₹%.0f", val)
                elif contrib_type == "nps" and facts.monthly_nps == 0:
                    facts.monthly_nps = val
                    logger.info("Found monthly NPS: ₹%.0f", val)
                elif contrib_type == "ppf" and facts.monthly_ppf == 0:
                    facts.monthly_ppf = val
                    logger.info("Found monthly PPF: ₹%.0f", val)

    # ── Extract insurance premium ──────────────────────────────────────────── #
    for pattern, _ in INSURANCE_PREMIUM_PATTERNS:
        for m in re.finditer(pattern, user_text, re.IGNORECASE):
            val = parse_inr(m.group(1))
            if val and val > 0 and facts.annual_insurance_premium == 0:
                facts.annual_insurance_premium = val
                logger.info("Found annual insurance premium: ₹%.0f", val)

    # ── Extract SIP ────────────────────────────────────────────────────────── #
    sip_patterns = [
        r"([\d.,]+\s*(?:cr|l\b|lakh|lakhs|k\b)?)\s*(?:per\s+month|pm|monthly|/month)?\s*(?:in\s+)?sips?\b",
        r"invest\s+([\d.,]+\s*(?:cr|l\b|lakh|lakhs|k\b)?)\s*(?:per\s+month|pm|monthly)?\s*(?:in\s+)?sips?",
        r"sip\s+(?:of\s+|amount\s+|is\s+)?([\d.,]+\s*(?:cr|l\b|lakh|lakhs|k\b)?)",
    ]
    for pattern in sip_patterns:
        for m in re.finditer(pattern, user_text, re.IGNORECASE):
            val = parse_inr(m.group(1))
            if val and val > 500:
                facts.monthly_sip = val
                logger.info("Found monthly SIP: ₹%.0f", val)
                break
        if facts.monthly_sip:
            break

    # ── Extract assets ─────────────────────────────────────────────────────── #
    for pattern, category in ASSET_PATTERNS:
        for m in re.finditer(pattern, user_text, re.IGNORECASE):
            # The captured group should be the amount
            raw_amount = m.group(1).strip()
            val = parse_inr(raw_amount)
            if val is None or val <= 0:
                continue

            # Use the full match as a label (truncated)
            label = m.group(0)[:40].strip()

            # Skip if this looks like a monthly contribution (too small for a corpus)
            # Monthly contributions are handled separately above
            if val < 50000:  # Less than ₹50k → probably a monthly figure, not a corpus
                continue

            # Deduplicate: don't add the same amount twice
            existing_vals = {
                "equity": list(facts.equity_assets.values()),
                "debt": list(facts.debt_assets.values()),
                "gold": list(facts.gold_assets.values()),
                "retirement": list(facts.retirement_assets.values()),
                "other": list(facts.other_assets.values()),
            }
            all_existing = [v for vs in existing_vals.values() for v in vs]
            if any(abs(v - val) < 1 for v in all_existing):
                continue  # Already captured this amount

            if category == "equity":
                facts.equity_assets[label] = val
            elif category == "debt":
                facts.debt_assets[label] = val
            elif category == "gold":
                facts.gold_assets[label] = val
            elif category == "retirement":
                facts.retirement_assets[label] = val
            elif category == "other":
                facts.other_assets[label] = val

            logger.info("Found %s asset: %s = ₹%.0f", category, label[:30], val)

    logger.info(
        "Pre-processing complete: income=₹%.0f expenses=₹%.0f "
        "total_savings=₹%.0f EPF=₹%.0f NPS=₹%.0f",
        facts.monthly_income or 0,
        facts.monthly_expenses or 0,
        facts.total_savings,
        facts.monthly_epf,
        facts.monthly_nps,
    )
    return facts


# ============================================================================ #
#  SECTION 4 — SELF TEST                                                       #
# ============================================================================ #

if __name__ == "__main__":
    print("=" * 65)
    print("  financial_preprocessor.py — Self-Test")
    print("=" * 65)

    test_messages = [
        {"role": "user", "content": "I'm 54. I earn 5L per month with a take home salary of 2L per month. I invest 1L per month in SIPs and 70L in mutual funds in total. 12L in SGBs. I plan to retire at 60."},
        {"role": "assistant", "content": "Great start. Can you tell me your monthly expenses?"},
        {"role": "user", "content": "around 80k. I have an education loan for my kid, 12L has to go there yearly. 55 lakhs are in NPS. 36k per year in insurance premiums."},
        {"role": "user", "content": "I also have a FD of 1.5 crore for 2 years at 9.3% pa."},
        {"role": "user", "content": "I also have a PF of 80L, education loan is 12L instead of 10L for the coming 5 years. 60k per month goes into my PF. and 24k pm goes towards NPS."},
    ]

    facts = scan_conversation(test_messages)

    print(f"\nMonthly income   : ₹{facts.monthly_income:,.0f}" if facts.monthly_income else "\nMonthly income   : NOT FOUND")
    print(f"Monthly expenses : ₹{facts.monthly_expenses:,.0f}" if facts.monthly_expenses else "Monthly expenses : NOT FOUND")
    print(f"Monthly SIP      : ₹{facts.monthly_sip:,.0f}" if facts.monthly_sip else "Monthly SIP      : NOT FOUND")
    print(f"Monthly EPF      : ₹{facts.monthly_epf:,.0f}")
    print(f"Monthly NPS      : ₹{facts.monthly_nps:,.0f}")
    print(f"Ins. premium/yr  : ₹{facts.annual_insurance_premium:,.0f}")

    print(f"\nEquity assets    : {facts.equity_assets}")
    print(f"Debt assets      : {facts.debt_assets}")
    print(f"Retirement assets: {facts.retirement_assets}")
    print(f"Gold assets      : {facts.gold_assets}")
    print(f"\nTotal savings    : ₹{facts.total_savings:,.0f} ({facts.total_savings/1e7:.2f} Cr)")
    print(f"Expected         : ₹3,67,00,000 (3.67 Cr = 70L+12L+55L+1.5Cr+80L)")

    print(f"\nEquity %         : {facts.equity_pct}%  (expected ~19%)")
    print(f"Debt %           : {facts.debt_pct}%   (expected ~78%)")
    print(f"Gold %           : {facts.gold_pct}%    (expected ~3%)")

    print(f"\nRetirement/mo    : ₹{facts.monthly_retirement_contributions:,.0f}  (expected ₹84,000)")
    print(f"Annual tax saving: ₹{facts.annual_tax_saving:,.0f}  (expected ₹1,044,000)")

    print("\n--- Context string for LLM ---")
    print(facts.to_context_string())
    print("\n✅ Self-test complete.")
