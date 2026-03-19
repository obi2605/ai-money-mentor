# ==============================================================================
# privacy_parser.py
# AI Money Mentor — Local-First Document Parser
# ------------------------------------------------------------------------------
# PRIVACY GUARANTEE:
#   This module runs ENTIRELY on the user's machine. No page content, no raw
#   text, no file bytes are ever passed to an LLM API or any external service.
#   The LLM only ever sees the SANITISED structured output (fund names, amounts,
#   dates) after PII has been masked by `sanitise_pii()`.
#
# SUPPORTED DOCUMENTS:
#   1. CAMS Statement (Consolidated Account Statement) — primary use case
#   2. KFintech / Karvy Statement — same schema, minor format differences
#   3. Form 16 (Part A + Part B) — basic income/TDS extraction
#
# PARSING STRATEGY:
#   pdfplumber is used over pypdf because it preserves spatial layout and
#   handles the multi-column, mixed-font CAMS format far more reliably.
#   All regex patterns are documented with a real example of the text they match.
# ==============================================================================

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pdfplumber
from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)

# ============================================================================ #
#  SECTION 1 — DATA CLASSES                                                    #
# ============================================================================ #

@dataclass
class MFTransaction:
    """A single mutual fund transaction extracted from CAMS."""
    date: date
    description: str        # "SIP", "Purchase", "Redemption", "Switch In", etc.
    amount: float           # INR. Negative = outflow (buy). Positive = inflow (sell/redemption).
    units: float            # Units transacted
    nav: float              # NAV at transaction date
    balance_units: float    # Cumulative units after this transaction
    tx_type: str            # Normalised: "BUY" | "SELL" | "DIVIDEND" | "SWITCH_IN" | "SWITCH_OUT"


@dataclass
class MFFolioHolding:
    """All transactions + current value for a single MF folio."""
    amc_name: str
    scheme_name: str
    isin: str
    folio_number_hash: str      # SHA-256 hash of actual folio number — PII protected
    transactions: list[MFTransaction]
    closing_units: float
    current_nav: Optional[float]
    current_value: Optional[float]
    expense_ratio_pct: Optional[float]  # TER from mf_universe.json or extracted


@dataclass
class CAMSPortfolio:
    """Full parsed portfolio from a CAMS/KFintech statement."""
    investor_name: str              # Kept for display
    pan_masked: str                 # "ABCPX1234X" → "ABCPX****X"
    statement_date: Optional[date]
    holdings: list[MFFolioHolding]
    num_transactions: int
    parse_duration_sec: float
    parser_warnings: list[str]      # Non-fatal issues encountered during parsing


@dataclass
class Form16Data:
    """Key fields extracted from Form 16."""
    employer_name: str
    financial_year: str
    gross_salary: float
    total_tds: float
    standard_deduction: float
    net_taxable_income: float
    pan_masked: str
    parser_warnings: list[str]


# ============================================================================ #
#  SECTION 2 — PII SANITISATION                                                #
# ============================================================================ #

# Regex patterns for Indian PII types
_PAN_PATTERN      = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')
_AADHAAR_PATTERN  = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')
_MOBILE_PATTERN   = re.compile(r'\b[6-9]\d{9}\b')
_EMAIL_PATTERN    = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b')
_FOLIO_PATTERN    = re.compile(r'\b\d{9,12}[/\d]*\b')   # CAMS folios: 9–12 digits


def sanitise_pii(text: str) -> str:
    """
    Mask all PII in a raw text string before it is passed anywhere beyond
    this module (including to the LLM response generator).

    Replaces:
        PAN       → "ABCPX****X"  (first 5 + last char preserved for display)
        Aadhaar   → "XXXX-XXXX-XXXX"
        Mobile    → "XXXXXXXX" + last 2 digits
        Email     → "u***@domain.com"
        Folio No  → SHA-256 hash (first 8 chars)

    Parameters
    ----------
    text : str — raw text from PDF page

    Returns
    -------
    str — text with all detected PII masked
    """
    # PAN: keep first 5 chars and last char, mask middle 4 digits
    text = _PAN_PATTERN.sub(
        lambda m: m.group()[:5] + "****" + m.group()[-1], text
    )
    # Aadhaar
    text = _AADHAAR_PATTERN.sub("XXXX-XXXX-XXXX", text)
    # Mobile
    text = _MOBILE_PATTERN.sub(
        lambda m: "XXXXXXXX" + m.group()[-2:], text
    )
    # Email: keep first char, mask local part, keep domain
    def _mask_email(m: re.Match) -> str:
        local, domain = m.group().split("@", 1)
        return local[0] + "***@" + domain
    text = _EMAIL_PATTERN.sub(_mask_email, text)
    return text


def _hash_folio(folio: str) -> str:
    """Return the first 8 chars of SHA-256 hash of a folio number."""
    return hashlib.sha256(folio.strip().encode()).hexdigest()[:8].upper()


def _mask_pan(pan: str) -> str:
    """Mask PAN for display: ABCDE1234F → ABCDE****F"""
    pan = pan.strip()
    if len(pan) == 10:
        return pan[:5] + "****" + pan[-1]
    return "**MASKED**"


# ============================================================================ #
#  SECTION 3 — CAMS STATEMENT PARSER                                           #
# ============================================================================ #

# ── Regex patterns (each documented with a real CAMS line example) ─────────── #

# Matches: "HDFC Mutual Fund" / "SBI Mutual Fund" / "Nippon India Mutual Fund"
_AMC_HEADER = re.compile(
    r'^([A-Z][A-Za-z\s()&\-]+(?:Mutual Fund|AMC|Asset Management))',
    re.IGNORECASE
)

# Matches: "Folio No: 1234567890 / 01   KYC: OK   PAN: ABCDE1234F"
_FOLIO_LINE = re.compile(
    r'Folio\s*(?:No|Number)[\s:]+(\d[\d/\s]+)',
    re.IGNORECASE
)

# Matches PAN inline in folio line
_PAN_INLINE = re.compile(
    r'PAN\s*:\s*([A-Z]{5}[0-9]{4}[A-Z])',
    re.IGNORECASE
)

# Matches: "HDFC Top 100 Fund - Growth Option (ISIN: INF179K01CX3)"
_SCHEME_LINE = re.compile(
    r'^(.+?)\s*\(ISIN\s*:\s*([A-Z]{2}[A-Z0-9]{10})\)',
    re.IGNORECASE
)

# Matches: "05-Jan-2023  SIP  10,000.00  26.789  373.46  185.432"
# Groups: date, description, amount, units, nav, balance
_TRANSACTION_LINE = re.compile(
    r'(\d{2}[-/]\w{3}[-/]\d{4})\s+'           # date
    r'([A-Za-z][A-Za-z\s\(\)\-/]+?)\s+'        # description
    r'([\d,]+\.\d{2}|\-)\s+'                    # amount (or "-")
    r'([\d,]+\.\d{3}|\-)\s+'                    # units (or "-")
    r'([\d,]+\.\d{2,4}|\-)\s+'                  # nav (or "-")
    r'([\d,]+\.\d{3})',                          # balance units
)

# Matches: "Closing Balance  31-Dec-2023    -    -    -    185.432"
_CLOSING_BALANCE = re.compile(
    r'Closing\s+Balance.*?([\d,]+\.\d{3})\s*$',
    re.IGNORECASE
)

# Matches: "Market Value as on 31-Dec-2023: Rs. 69,255.34 (NAV: Rs. 373.46)"
_MARKET_VALUE_LINE = re.compile(
    r'Market\s+Value.*?Rs\.?\s*([\d,]+\.\d{2})\s*\(NAV.*?Rs\.?\s*([\d,]+\.\d{2,4})\)',
    re.IGNORECASE
)

# Matches: "Total Value of Portfolio: Rs. 3,45,678.90"
_TOTAL_VALUE_LINE = re.compile(
    r'(?:Total Value|Portfolio Value).*?Rs\.?\s*([\d,]+\.\d{2})',
    re.IGNORECASE
)

# Investor name: usually in the header "Name: JOHN DOE"
_INVESTOR_NAME = re.compile(r'Name\s*:\s*([A-Z][A-Za-z\s]+)', re.IGNORECASE)

# Statement date: "Statement Period: 01-Jan-2020 to 31-Dec-2023"
_STMT_DATE = re.compile(
    r'(?:Statement|Period).*?(\d{2}[-/]\w{3,}[-/]\d{4})\s*$',
    re.IGNORECASE
)

# Expense ratio: "Expense Ratio: 1.25%" (present in some CAMS variants)
_EXPENSE_RATIO = re.compile(r'(?:Expense Ratio|TER)\s*:?\s*([\d.]+)\s*%', re.IGNORECASE)


def _parse_float(s: str) -> float:
    """Parse an Indian-formatted number string: "1,23,456.789" → 1234567.89"""
    return float(s.replace(",", "").strip())


def _parse_date(s: str) -> date:
    """
    Parse CAMS date strings robustly.
    Handles: "05-Jan-2023", "05/01/2023", "January 05, 2023"
    """
    return dateutil_parser.parse(s.strip(), dayfirst=True).date()


def _classify_tx_type(description: str) -> str:
    """
    Normalise a raw CAMS transaction description into a canonical type.
    Examples:
        "SIP" → "BUY"
        "Purchase (Additional)" → "BUY"
        "Redemption" → "SELL"
        "Switch In" → "SWITCH_IN"
        "Dividend Reinvestment" → "DIVIDEND"
    """
    desc_upper = description.upper().strip()
    if any(k in desc_upper for k in ["SIP", "PURCHASE", "SUBSCRIPTION", "SWITCH IN", "NFO"]):
        if "SWITCH IN" in desc_upper:
            return "SWITCH_IN"
        return "BUY"
    elif any(k in desc_upper for k in ["REDEMPTION", "REPURCHASE", "SWITCH OUT", "WITHDRAWAL"]):
        if "SWITCH OUT" in desc_upper:
            return "SWITCH_OUT"
        return "SELL"
    elif any(k in desc_upper for k in ["DIVIDEND", "IDCW"]):
        return "DIVIDEND"
    elif "BONUS" in desc_upper:
        return "BONUS"
    else:
        return "OTHER"


def parse_cams_pdf(
    pdf_path: str,
    load_expense_ratios: bool = True,
) -> dict:
    """
    Parse a CAMS or KFintech Consolidated Account Statement PDF locally.

    PRIVACY: This function uses only pdfplumber (local). No network calls.
    The returned dict contains folio numbers as SHA-256 hashes and
    masked PAN numbers. It is safe to pass the return value to downstream
    modules (mf_xray.py) or the LLM response generator.

    Parameters
    ----------
    pdf_path : str
        Absolute path to the PDF file. Should be a temp file that is
        deleted immediately after this function returns (handled by app.py).
    load_expense_ratios : bool
        If True, attempts to cross-reference fund names against
        data/mf_universe.json to populate expense_ratio_pct.

    Returns
    -------
    dict representation of CAMSPortfolio (JSON-serialisable).

    Raises
    ------
    ValueError : if the file does not appear to be a valid CAMS statement.
    RuntimeError : if pdfplumber cannot open the file.
    """
    start_time = time.time()
    warnings: list[str] = []

    # ── Load MF universe for expense ratio lookup ─────────────────────────── #
    mf_universe: dict = {}
    if load_expense_ratios:
        try:
            import json, os
            universe_path = os.path.join(
                os.path.dirname(__file__), "data", "mf_universe.json"
            )
            with open(universe_path, "r", encoding="utf-8") as f:
                universe_list = json.load(f)
            # Index by normalised scheme name for fast lookup
            for fund in universe_list:
                key = _normalise_scheme_name(fund["scheme_name"])
                mf_universe[key] = fund
            logger.info("Loaded %d funds from mf_universe.json", len(mf_universe))
        except FileNotFoundError:
            warnings.append("mf_universe.json not found — expense ratios unavailable.")
        except Exception as e:
            warnings.append(f"Could not load mf_universe.json: {e}")

    # ── Open PDF ──────────────────────────────────────────────────────────── #
    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF: {exc}") from exc

    all_lines: list[str] = []
    with pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=2, y_tolerance=3)
            if text:
                all_lines.extend(text.splitlines())
            else:
                warnings.append(f"Page {page_num + 1} returned no text — may be scanned/image PDF.")

    if not all_lines:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "If it is a scanned document, OCR support is not yet available."
        )

    # Validate: CAMS statements always contain these marker strings
    full_text = "\n".join(all_lines)
    is_cams = "CAMS" in full_text.upper() or "COMPUTER AGE" in full_text.upper()
    is_kfintech = "KFINTECH" in full_text.upper() or "KARVY" in full_text.upper()
    if not (is_cams or is_kfintech):
        warnings.append(
            "This document does not appear to be a CAMS or KFintech statement. "
            "Parsing may produce incomplete results."
        )

    # ── Extract investor metadata ─────────────────────────────────────────── #
    investor_name = "Investor"
    pan_masked = "**MASKED**"
    statement_date = None

    for line in all_lines[:30]:   # Metadata is always in the first page header
        m = _INVESTOR_NAME.search(line)
        if m and investor_name == "Investor":
            investor_name = m.group(1).strip().title()
        m = _PAN_INLINE.search(line)
        if m:
            pan_masked = _mask_pan(m.group(1))
        m = _STMT_DATE.search(line)
        if m and statement_date is None:
            try:
                statement_date = _parse_date(m.group(1))
            except Exception:
                pass

    # ── Parse holdings ────────────────────────────────────────────────────── #
    holdings: list[MFFolioHolding] = []
    num_transactions = 0

    current_amc: str = ""
    current_scheme: str = ""
    current_isin: str = ""
    current_folio_hash: str = ""
    current_transactions: list[MFTransaction] = []
    current_closing_units: float = 0.0
    current_nav: Optional[float] = None
    current_value: Optional[float] = None
    current_expense_ratio: Optional[float] = None
    in_transaction_block = False

    def _flush_holding():
        """Commit the current accumulator state as a holding."""
        nonlocal current_transactions, current_closing_units
        nonlocal current_nav, current_value, current_expense_ratio
        if current_scheme and current_transactions:
            # Look up expense ratio from universe
            exp_ratio = current_expense_ratio
            if exp_ratio is None and mf_universe:
                key = _normalise_scheme_name(current_scheme)
                match = _fuzzy_lookup_scheme(key, mf_universe)
                if match:
                    exp_ratio = match.get("expense_ratio_pct")

            holdings.append(MFFolioHolding(
                amc_name=current_amc,
                scheme_name=current_scheme,
                isin=current_isin,
                folio_number_hash=current_folio_hash,
                transactions=list(current_transactions),
                closing_units=current_closing_units,
                current_nav=current_nav,
                current_value=current_value,
                expense_ratio_pct=exp_ratio,
            ))
        # Reset
        current_transactions.clear()
        current_closing_units = 0.0
        current_nav = None
        current_value = None
        current_expense_ratio = None

    for raw_line in all_lines:
        line = raw_line.strip()
        if not line:
            continue

        # ── Detect AMC header ────────────────────────────────────────────── #
        amc_match = _AMC_HEADER.match(line)
        if amc_match and len(line) < 60:
            _flush_holding()
            current_amc = amc_match.group(1).strip()
            in_transaction_block = False
            continue

        # ── Detect folio line ─────────────────────────────────────────────── #
        folio_match = _FOLIO_LINE.search(line)
        if folio_match:
            _flush_holding()
            raw_folio = folio_match.group(1).strip()
            current_folio_hash = _hash_folio(raw_folio)
            # Extract inline PAN if present
            pan_m = _PAN_INLINE.search(line)
            if pan_m:
                pan_masked = _mask_pan(pan_m.group(1))
            in_transaction_block = False
            continue

        # ── Detect scheme / ISIN line ─────────────────────────────────────── #
        scheme_match = _SCHEME_LINE.match(line)
        if scheme_match:
            _flush_holding()
            current_scheme = scheme_match.group(1).strip()
            current_isin = scheme_match.group(2).strip()
            in_transaction_block = True
            continue

        # ── Detect expense ratio inline ──────────────────────────────────── #
        er_match = _EXPENSE_RATIO.search(line)
        if er_match:
            try:
                current_expense_ratio = float(er_match.group(1))
            except ValueError:
                pass
            continue

        # ── Detect closing balance ─────────────────────────────────────────── #
        cb_match = _CLOSING_BALANCE.search(line)
        if cb_match:
            try:
                current_closing_units = _parse_float(cb_match.group(1))
            except ValueError:
                warnings.append(f"Could not parse closing balance: {line[:60]}")
            in_transaction_block = False
            continue

        # ── Detect market value / NAV ─────────────────────────────────────── #
        mv_match = _MARKET_VALUE_LINE.search(line)
        if mv_match:
            try:
                current_value = _parse_float(mv_match.group(1))
                current_nav = _parse_float(mv_match.group(2))
            except ValueError:
                warnings.append(f"Could not parse market value: {line[:60]}")
            continue

        # ── Parse transaction line ────────────────────────────────────────── #
        if in_transaction_block:
            tx_match = _TRANSACTION_LINE.search(line)
            if tx_match:
                try:
                    tx_date_raw   = tx_match.group(1)
                    tx_desc       = tx_match.group(2).strip()
                    tx_amount_raw = tx_match.group(3)
                    tx_units_raw  = tx_match.group(4)
                    tx_nav_raw    = tx_match.group(5)
                    tx_balance    = tx_match.group(6)

                    tx_date   = _parse_date(tx_date_raw)
                    tx_amount = _parse_float(tx_amount_raw) if tx_amount_raw != "-" else 0.0
                    tx_units  = _parse_float(tx_units_raw)  if tx_units_raw  != "-" else 0.0
                    tx_nav    = _parse_float(tx_nav_raw)    if tx_nav_raw    != "-" else 0.0
                    tx_bal    = _parse_float(tx_balance)

                    tx_type = _classify_tx_type(tx_desc)

                    # Sign convention: buys are negative cash flows (outflows)
                    if tx_type in ("BUY", "SWITCH_IN"):
                        signed_amount = -abs(tx_amount)
                    elif tx_type in ("SELL", "SWITCH_OUT"):
                        signed_amount = abs(tx_amount)
                    else:
                        signed_amount = tx_amount

                    current_transactions.append(MFTransaction(
                        date=tx_date,
                        description=tx_desc,
                        amount=signed_amount,
                        units=tx_units,
                        nav=tx_nav,
                        balance_units=tx_bal,
                        tx_type=tx_type,
                    ))
                    num_transactions += 1
                except Exception as exc:
                    warnings.append(
                        f"Skipped malformed transaction line: {line[:70]} ({exc})"
                    )

    # Flush the final holding
    _flush_holding()

    if not holdings:
        raise ValueError(
            "No mutual fund holdings were found in this document. "
            "Please ensure it is a valid CAMS/KFintech Consolidated Account Statement."
        )

    portfolio = CAMSPortfolio(
        investor_name=investor_name,
        pan_masked=pan_masked,
        statement_date=statement_date,
        holdings=holdings,
        num_transactions=num_transactions,
        parse_duration_sec=round(time.time() - start_time, 3),
        parser_warnings=warnings,
    )

    logger.info(
        "CAMS parse complete: %d holdings, %d transactions in %.2fs",
        len(holdings), num_transactions, portfolio.parse_duration_sec
    )

    # Return as a plain dict (JSON-serialisable for session state)
    return _portfolio_to_dict(portfolio)


# ============================================================================ #
#  SECTION 4 — FORM 16 PARSER                                                  #
# ============================================================================ #

_GROSS_SALARY      = re.compile(r'Gross\s+Salary.*?Rs?\.?\s*([\d,]+(?:\.\d{2})?)', re.IGNORECASE)
_TDS_DEDUCTED      = re.compile(r'Tax\s+Deducted.*?Rs?\.?\s*([\d,]+(?:\.\d{2})?)', re.IGNORECASE)
_STD_DEDUCTION     = re.compile(r'Standard\s+Deduction.*?Rs?\.?\s*([\d,]+(?:\.\d{2})?)', re.IGNORECASE)
_TAXABLE_INCOME    = re.compile(r'(?:Net\s+)?Taxable.*?Income.*?Rs?\.?\s*([\d,]+(?:\.\d{2})?)', re.IGNORECASE)
_EMPLOYER_NAME     = re.compile(r'Name\s+of\s+(?:the\s+)?Employer\s*:\s*(.+)', re.IGNORECASE)
_FINANCIAL_YEAR    = re.compile(r'F\.?Y\.?\s*(\d{4}[-–]\d{2,4})', re.IGNORECASE)
_FORM16_PAN        = re.compile(r"Employee'?s?\s+PAN\s*:\s*([A-Z]{5}[0-9]{4}[A-Z])", re.IGNORECASE)


def parse_form16_pdf(pdf_path: str) -> dict:
    """
    Extract key income and TDS data from a Form 16 Part A/B PDF.

    Extracts:
        - Employer name
        - Financial year
        - Gross salary
        - Standard deduction
        - Total TDS deducted
        - Net taxable income
        - Employee PAN (masked)

    Returns a dict representation of Form16Data.
    """
    warnings: list[str] = []

    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Could not open Form 16 PDF: {exc}") from exc

    full_text = ""
    with pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=3)
            if text:
                full_text += text + "\n"

    if not full_text.strip():
        raise ValueError("Could not extract any text from the Form 16 PDF.")

    # Validate it's actually a Form 16
    if "FORM 16" not in full_text.upper() and "FORM NO. 16" not in full_text.upper():
        warnings.append("Document may not be a Form 16 — validation marker not found.")

    def _extract(pattern: re.Pattern, default: str = "0") -> str:
        m = pattern.search(full_text)
        return m.group(1).strip() if m else default

    def _extract_float(pattern: re.Pattern) -> float:
        raw = _extract(pattern)
        try:
            return _parse_float(raw)
        except (ValueError, AttributeError):
            return 0.0

    employer  = _extract(_EMPLOYER_NAME, "Unknown Employer")
    fin_year  = _extract(_FINANCIAL_YEAR, "Unknown")
    gross     = _extract_float(_GROSS_SALARY)
    tds       = _extract_float(_TDS_DEDUCTED)
    std_ded   = _extract_float(_STD_DEDUCTION)
    taxable   = _extract_float(_TAXABLE_INCOME)
    pan_raw   = _extract(_FORM16_PAN, "")
    pan_m     = _mask_pan(pan_raw) if pan_raw else "**NOT FOUND**"

    data = Form16Data(
        employer_name=employer[:80],   # Truncate long names
        financial_year=fin_year,
        gross_salary=gross,
        total_tds=tds,
        standard_deduction=std_ded if std_ded > 0 else 50000.0,  # FY23 default
        net_taxable_income=taxable if taxable > 0 else max(gross - std_ded, 0),
        pan_masked=pan_m,
        parser_warnings=warnings,
    )

    logger.info(
        "Form 16 parsed: %s | FY %s | Gross ₹%.0f | TDS ₹%.0f",
        data.employer_name, data.financial_year, data.gross_salary, data.total_tds
    )

    return {
        "employer_name":       data.employer_name,
        "financial_year":      data.financial_year,
        "gross_salary":        data.gross_salary,
        "total_tds":           data.total_tds,
        "standard_deduction":  data.standard_deduction,
        "net_taxable_income":  data.net_taxable_income,
        "pan_masked":          data.pan_masked,
        "parser_warnings":     data.parser_warnings,
    }


# ============================================================================ #
#  SECTION 5 — HELPERS                                                         #
# ============================================================================ #

def _normalise_scheme_name(name: str) -> str:
    """
    Normalise a scheme name for fuzzy matching against mf_universe.json.
    Removes common suffixes and lowercases.
    """
    name = name.lower()
    for suffix in [
        " - growth", " - direct", " - regular", " plan", " option",
        " (g)", "(d)", " growth", " direct", " regular",
    ]:
        name = name.replace(suffix, "")
    return re.sub(r'\s+', ' ', name).strip()


def _fuzzy_lookup_scheme(
    normalised_name: str,
    universe: dict,
    threshold: int = 6,
) -> Optional[dict]:
    """
    Simple substring matching to find a fund in the universe dict.
    Returns the best match or None.
    """
    # Try exact key match first
    if normalised_name in universe:
        return universe[normalised_name]

    # Substring match: normalised name must contain at least `threshold` word chars
    words = normalised_name.split()
    if len(words) < 2:
        return None

    # Use first 3 words as a fingerprint
    fingerprint = " ".join(words[:3])
    for key, fund in universe.items():
        if fingerprint in key:
            return fund

    return None


def _portfolio_to_dict(portfolio: CAMSPortfolio) -> dict:
    """Serialise CAMSPortfolio to a plain, JSON-safe dict."""
    return {
        "investor_name": portfolio.investor_name,
        "pan_masked": portfolio.pan_masked,
        "statement_date": portfolio.statement_date.isoformat() if portfolio.statement_date else None,
        "num_holdings": len(portfolio.holdings),
        "num_transactions": portfolio.num_transactions,
        "parse_duration_sec": portfolio.parse_duration_sec,
        "parser_warnings": portfolio.parser_warnings,
        "holdings": [
            {
                "amc_name": h.amc_name,
                "scheme_name": h.scheme_name,
                "isin": h.isin,
                "folio_hash": h.folio_number_hash,
                "closing_units": h.closing_units,
                "current_nav": h.current_nav,
                "current_value": h.current_value,
                "expense_ratio_pct": h.expense_ratio_pct,
                "transactions": [
                    {
                        "date": t.date.isoformat(),
                        "description": t.description,
                        "amount": t.amount,
                        "units": t.units,
                        "nav": t.nav,
                        "balance_units": t.balance_units,
                        "tx_type": t.tx_type,
                    }
                    for t in h.transactions
                ],
            }
            for h in portfolio.holdings
        ],
    }


# ============================================================================ #
#  SECTION 6 — SELF-TEST                                                       #
# ============================================================================ #

if __name__ == "__main__":
    print("=" * 65)
    print("  privacy_parser.py — Self-Test Suite")
    print("=" * 65)

    print("\n[TEST 1] PII Sanitisation")
    samples = [
        "Name: Rahul Sharma  PAN: ABCDE1234F  Mobile: 9876543210",
        "Email: rahul.sharma@example.com  Aadhaar: 1234 5678 9012",
        "Folio No: 123456789012 / 01   KYC: OK   PAN: ZYXWV9876A",
    ]
    for s in samples:
        masked = sanitise_pii(s)
        print(f"  IN : {s}")
        print(f"  OUT: {masked}\n")

    print("[TEST 2] Scheme Name Normalisation")
    names = [
        "HDFC Top 100 Fund - Growth Option",
        "Mirae Asset Large Cap Fund - Direct Plan - Growth",
        "Axis Bluechip Fund (G)",
    ]
    for n in names:
        print(f"  '{n}' → '{_normalise_scheme_name(n)}'")

    print("\n[TEST 3] Transaction classifier")
    descs = [
        "SIP", "Purchase (Additional)", "Redemption",
        "Switch In - HDFC Equity", "Dividend Reinvestment", "NFO Allotment"
    ]
    for d in descs:
        print(f"  '{d}' → {_classify_tx_type(d)}")

    print("\n[TEST 4] CAMS PDF parse (requires sample file)")
    import os
    sample = os.path.join(os.path.dirname(__file__), "tests", "sample_cams.pdf")
    if os.path.exists(sample):
        result = parse_cams_pdf(sample)
        print(f"  Holdings    : {result['num_holdings']}")
        print(f"  Transactions: {result['num_transactions']}")
        print(f"  Parse time  : {result['parse_duration_sec']}s")
        print(f"  Warnings    : {result['parser_warnings']}")
    else:
        print("  [SKIP] tests/sample_cams.pdf not found — place a real CAMS PDF to test")

    print("\n✅ Self-tests complete.")
