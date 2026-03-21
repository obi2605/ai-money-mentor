# ==============================================================================
# llm_orchestrator.py
# AI Money Mentor — LangChain Orchestration Layer
# ------------------------------------------------------------------------------
# ARCHITECTURE CONTRACT:
#   • This module's ONLY jobs are:
#       1. Classify user intent into a typed enum
#       2. Extract structured financial variables from natural language
#       3. Wrap deterministic quant results in natural language for the user
#   • This module does NOT do any math. Ever.
#   • All variable extraction uses Pydantic + `with_structured_output` so
#     LangChain returns typed objects, not raw strings we have to parse.
#   • Every chain has a hardcoded system prompt that forbids the LLM from
#     inventing financial figures.
# ==============================================================================

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, field_validator

# Import our typed result dataclasses so the response generator is type-safe
from quant_engine import (
    RollingReturnResult,
    SIPProjectionResult,
    XIRRResult,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================================ #
#  SECTION 1 — LLM CLIENT                                                      #
# ============================================================================ #

def _build_llm(temperature: float = 0.0) -> ChatGroq:
    """
    Build and return a Groq client.
    Free tier at console.groq.com — no billing required.
    temperature=0 for extraction (deterministic), higher for generation.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Create a free key at console.groq.com "
            "and add it to your .env file as: GROQ_API_KEY=gsk_..."
        )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=api_key,
        max_retries=2,
        timeout=30,
    )


# ============================================================================ #
#  SECTION 2 — INTENT CLASSIFICATION                                           #
# ============================================================================ #

class Intent(str, Enum):
    """All supported user intents. Maps 1:1 to a module in the app."""
    HEALTH_SCORE    = "HEALTH_SCORE"
    FIRE_PLANNER    = "FIRE_PLANNER"
    MF_XRAY         = "MF_XRAY"
    MARKET_DATA     = "MARKET_DATA"
    SIP_PROJECTION  = "SIP_PROJECTION"
    LIFE_EVENT      = "LIFE_EVENT"
    GENERAL_QUERY   = "GENERAL_QUERY"
    CLARIFY         = "CLARIFY"


class IntentResult(BaseModel):
    """Structured output for intent classification."""
    intent: Intent = Field(description="The primary financial intent of the user's message.")
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0.",
        ge=0.0, le=1.0
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description=(
            "List of critical pieces of information missing to fulfil this intent. "
            "E.g. ['monthly_income', 'target_corpus']. Empty if enough info is available."
        )
    )
    reasoning: str = Field(description="One-sentence explanation of why this intent was chosen.")


_INTENT_SYSTEM = """You are an intent classifier for an Indian financial advisory chatbot.
Classify the user's message into exactly ONE intent from this list:

- HEALTH_SCORE    : User wants to know their financial health, get a score, or assess their finances overall.
- FIRE_PLANNER    : User mentions retirement, financial independence, FIRE, saving for a goal, or building a corpus.
- MF_XRAY         : User wants to analyse mutual fund portfolio, mentions CAMS, NAV, returns, overlap, or expense ratio.
- MARKET_DATA     : User asks about Nifty 50, Sensex, index returns, market performance, or specific fund NAVs.
- SIP_PROJECTION  : User asks "how much SIP do I need", "will my SIP be enough", or wants to project SIP growth.
- LIFE_EVENT      : User mentions a specific life event: bonus, salary hike, inheritance, windfall, marriage, wedding, baby, child, job loss, layoff, fired, home purchase, buying a house/flat.
- GENERAL_QUERY   : General financial question that doesn't require personal data or computation.
- CLARIFY         : User's message is ambiguous or missing critical data; you need to ask a follow-up.

PRIORITY RULES (apply in order, highest priority first):
1. If the message contains income/expenses/salary AND any of (emergency fund, insurance, debt, savings) → HEALTH_SCORE. Always. Even if CAMS was mentioned before.
2. If the message explicitly says "analyse", "CAMS", "statement", "portfolio", "my funds" → MF_XRAY.
3. If the message mentions bonus, inheritance, windfall, marriage, baby, job loss, home purchase → LIFE_EVENT.
4. If the message mentions retirement age, FIRE, "retire at", "corpus" → FIRE_PLANNER.
5. If the message asks about Nifty, Sensex, index, market returns → MARKET_DATA.
6. If the message asks about SIP amounts or projections → SIP_PROJECTION.
7. Never let conversation history override these rules. Each message is classified on its OWN content.

Rules:
- Never invent financial data. Only classify and identify missing information.
- If the user provides SOME data but not all, still classify to the best intent and list what's missing.
- Respond ONLY with the structured JSON output. No prose."""

_INTENT_HUMAN = "User message: {user_message}\n\nConversation so far:\n{history}"


def detect_intent(user_message: str, history: str = "") -> IntentResult:
    """Classify the user's message into a typed Intent."""
    llm = _build_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(IntentResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _INTENT_SYSTEM),
        ("human", _INTENT_HUMAN),
    ])
    chain = prompt | structured_llm
    try:
        result: IntentResult = chain.invoke({
            "user_message": user_message,
            "history": history or "No prior conversation.",
        })
        logger.info("Intent detected: %s (confidence=%.2f)", result.intent, result.confidence)
        return result
    except Exception as exc:
        logger.error("Intent detection failed: %s", exc)
        raise RuntimeError(f"Intent classification failed: {exc}") from exc


# ============================================================================ #
#  SECTION 3 — VARIABLE EXTRACTION (one Pydantic model per intent)            #
# ============================================================================ #

# ---------------------------------------------------------------------------- #
#  3a. FIRE Planner Parameters                                                  #
# ---------------------------------------------------------------------------- #

class FIREParams(BaseModel):
    """Parameters extracted from conversation for the FIRE Path Planner."""
    current_age: int = Field(description="User's current age in years.", ge=18, le=80)
    retirement_age: int = Field(description="Target retirement age.", ge=30, le=90)
    monthly_income: float = Field(description="Current gross monthly income in INR.", gt=0)
    monthly_expenses: float = Field(description="Current monthly expenses in INR.", gt=0)
    current_savings: float = Field(
        default=0.0,
        description=(
            "TOTAL existing savings/investments corpus in INR. "
            "This MUST be the SUM of ALL assets mentioned: mutual funds + FDs + SGBs + "
            "PPF + NPS corpus + stocks + real estate + any other investments. "
            "Example: '70L in MF, 12L in SGBs, 55L in NPS, 1.5Cr FD' → 70L+12L+55L+1.5Cr = 2,87,00,000. "
            "Do NOT pick just one asset. Sum everything explicitly mentioned."
        )
    )
    target_corpus: Optional[float] = Field(
        default=None,
        description=(
            "Target retirement corpus in INR. If not explicitly stated, set null — "
            "the quant engine will calculate it from expenses and life expectancy."
        )
    )
    monthly_sip: Optional[float] = Field(
        default=None,
        description="Current monthly SIP amount in INR. Null if not mentioned."
    )
    assumed_inflation_pct: float = Field(
        default=6.0,
        description="Assumed annual inflation rate in %. Default 6.0 for India.",
        ge=2.0, le=15.0
    )
    assumed_return_pct: float = Field(
        default=12.0,
        description=(
            "Assumed annual portfolio return in %. Default 12.0 for a balanced Indian "
            "equity-heavy portfolio. DO NOT change this unless the user specifies."
        ),
        ge=1.0, le=30.0
    )
    step_up_pct: float = Field(
        default=10.0,
        description="Annual SIP step-up percentage. Default 10% (common in India).",
        ge=0.0, le=50.0
    )

    @field_validator("retirement_age")
    @classmethod
    def retirement_after_current(cls, v: int, info) -> int:
        if "current_age" in info.data and v <= info.data["current_age"]:
            raise ValueError("retirement_age must be greater than current_age.")
        return v


class HealthScoreParams(BaseModel):
    """Parameters for the Money Health Score (6-dimension assessment)."""
    monthly_income: float = Field(description="Gross monthly income in INR.", gt=0)
    monthly_expenses: float = Field(description="Monthly expenses in INR.", gt=0)
    emergency_fund: float = Field(
        default=0.0,
        description=(
            "Total LIQUID emergency fund in INR — only highly liquid assets: "
            "savings account, liquid MF, FD maturing within 1 year. "
            "Do NOT include equity MF, NPS, EPF here."
        )
    )
    total_insurance_cover: float = Field(
        default=0.0,
        description=(
            "Total life insurance SUM ASSURED in INR (not premium, not annual payment). "
            "Term plan cover + LIC sum assured. If only premium is mentioned, set 0."
        )
    )
    total_debt_emi: float = Field(
        default=0.0,
        description=(
            "Total MONTHLY EMI payments in INR. "
            "Include: home loan EMI, car loan EMI, personal loan EMI, education loan EMI. "
            "If user says 'education loan 12L over 5 years', EMI ≈ 12L/60 = 20,000/mo."
        )
    )
    equity_pct: float = Field(
        default=0.0,
        description=(
            "% of total investments in equity (equity MF, stocks, ELSS). "
            "Apply Rule B: if amounts given, calculate equity/total × 100."
        ),
        ge=0.0, le=100.0
    )
    debt_pct: float = Field(
        default=0.0,
        description=(
            "% of investments in debt (FD, debt MF, EPF/PF corpus, NPS corpus, PPF, bonds). "
            "Apply Rule B: if amounts given, calculate (FD+EPF+NPS+PPF+bonds)/total × 100."
        ),
        ge=0.0, le=100.0
    )
    gold_pct: float = Field(
        default=0.0,
        description=(
            "% of investments in gold (SGB, gold ETF, physical gold). "
            "Apply Rule B: if amounts given, calculate gold/total × 100."
        ),
        ge=0.0, le=100.0
    )
    other_pct: float = Field(
        default=0.0,
        description="% of investments in other assets (real estate, ULIPs, etc.).",
        ge=0.0, le=100.0
    )
    epf_ppf_nps_monthly: float = Field(
        default=0.0,
        description=(
            "Total MONTHLY contributions to EPF + PPF + NPS in INR. "
            "Apply Rule D: sum all mentioned. '60k PF + 24k NPS' = 84000."
        )
    )
    tax_saving_investments: float = Field(
        default=0.0,
        description=(
            "ANNUAL amount across all 80C/80D eligible instruments in INR. "
            "Apply Rule C: EPF_monthly×12 + NPS_monthly×12 + insurance_annual + ELSS_annual + PPF_annual. "
            "Example: 60k/mo EPF + 24k/mo NPS + 36k/yr insurance = 1,044,000."
        )
    )
    gross_annual_income: float = Field(
        description="Gross annual income in INR.", gt=0
    )


class SIPQueryParams(BaseModel):
    """Parameters for a standalone SIP projection query."""
    monthly_sip: float = Field(description="Monthly SIP amount in INR.", gt=0)
    years: int = Field(description="Investment horizon in years.", ge=1, le=50)
    assumed_cagr_pct: float = Field(
        default=12.0,
        description="Assumed annual return. Default 12.0%.",
        ge=1.0, le=30.0
    )
    target_corpus: Optional[float] = Field(
        default=None,
        description="Target corpus in INR, if user specifies a goal. Null otherwise."
    )
    step_up_pct: float = Field(
        default=0.0,
        description="Annual SIP increase %. Default 0 unless user mentions top-up.",
        ge=0.0, le=50.0
    )


class MarketDataParams(BaseModel):
    """Parameters for a market data / historical return query."""
    ticker_or_alias: str = Field(
        description=(
            "The asset to look up. Use these exact aliases where applicable: "
            "'nifty50', 'sensex', 'nifty bank', 'nifty it', 'gold'. "
            "For ETFs, use the yfinance ticker (e.g. 'NIFTYBEES.NS')."
        )
    )
    period: str = Field(
        default="5Y",
        description="Historical period. Must be one of: '1Y', '3Y', '5Y', '7Y', '10Y'."
    )

    @field_validator("period")
    @classmethod
    def valid_period(cls, v: str) -> str:
        valid = {"1Y", "3Y", "5Y", "7Y", "10Y"}
        v = v.upper().strip()
        if v not in valid:
            raise ValueError(f"Period must be one of {valid}")
        return v


class LifeEventParams(BaseModel):
    """Parameters for the Life Event Financial Advisor."""
    event_type: str = Field(
        description=(
            "Type of life event. Must be exactly one of: "
            "BONUS, INHERITANCE, MARRIAGE, NEW_BABY, JOB_LOSS, HOME_PURCHASE"
        )
    )
    event_amount: float = Field(
        default=0.0,
        description=(
            "The financial amount associated with the event in INR. "
            "For BONUS: bonus amount. For INHERITANCE: windfall amount. "
            "For MARRIAGE: available budget. For HOME_PURCHASE: property price. "
            "For JOB_LOSS: severance pay. For NEW_BABY: available savings for baby."
        )
    )
    monthly_income: float = Field(default=0.0, description="Monthly income in INR.")
    monthly_expenses: float = Field(default=0.0, description="Monthly expenses in INR.")
    current_savings: float = Field(default=0.0, description="Total existing savings in INR.")
    current_emergency_fund: float = Field(default=0.0, description="Liquid emergency fund in INR.")
    total_insurance_cover: float = Field(default=0.0, description="Total life insurance sum assured in INR.")
    existing_sip: float = Field(default=0.0, description="Current monthly SIP in INR.")
    tax_bracket_pct: float = Field(default=30.0, description="Income tax bracket: 10, 20, or 30.")
    home_loan_outstanding: float = Field(default=0.0, description="Outstanding home loan in INR.")
    num_dependents: int = Field(default=0, description="Number of dependents.")
    years_to_retirement: int = Field(default=20, description="Years to retirement.")


# ============================================================================ #
#  SECTION 4 — EXTRACTION CHAINS (LangChain → Pydantic)                       #
# ============================================================================ #

_EXTRACTION_SYSTEM = """You are a financial data extractor for an Indian financial advisory chatbot.

CRITICAL RULES:
1. If the conversation starts with a [VERIFIED FACTS] block, USE THOSE VALUES EXACTLY.
   Do NOT recalculate from the raw conversation — the verified block has already done the math.
2. Extract ONLY values the user has explicitly stated. NEVER invent financial figures.
3. For unmentioned optional fields, use schema defaults.
4. All monetary values in INR. Convert: "1L"=100000, "10L"=1000000, "1Cr"=10000000, "50K"=50000.
5. Do NOT provide advice or commentary. Only extract and structure data.

━━━ RULE A: ASSET SUMMATION (current_savings) ━━━
IF a [VERIFIED FACTS] block is present → use the TOTAL current_savings from it directly.
IF no block → Step 1: list every asset mentioned anywhere. Step 2: sum them all.
  Equity: MF, stocks, ELSS | Debt: FD, PPF, bonds | Retirement: EPF/PF corpus, NPS corpus | Gold: SGB, gold ETF
  NEVER pick just one asset. Example: 70L MF + 12L SGB + 55L NPS + 1.5Cr FD + 80L PF = 3.67Cr

━━━ RULE B: ALLOCATION % INFERENCE ━━━
IF [VERIFIED FACTS] block present → use equity_pct / debt_pct / gold_pct from it.
IF no block → calculate: equity_pct = equity_INR/total_INR × 100, etc.

━━━ RULE C: TAX SAVING INVESTMENTS (annual 80C/80D) ━━━
IF [VERIFIED FACTS] block present → use TOTAL tax_saving_investments from it.
IF no block → sum: EPF_monthly×12 + NPS_monthly×12 + insurance_annual + ELSS + PPF

━━━ RULE D: MONTHLY RETIREMENT CONTRIBUTIONS ━━━
IF [VERIFIED FACTS] block present → use TOTAL epf_ppf_nps_monthly from it.
IF no block → sum all monthly: EPF + NPS + PPF contributions."""


def extract_fire_params(user_message: str, history: str = "") -> FIREParams:
    """Extract FIRE Planner parameters from user conversation."""
    llm = _build_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(FIREParams)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _EXTRACTION_SYSTEM + (
            "\n\nExtract FIREParams from the full conversation. "
            "CRITICAL: Apply Rule A to sum ALL assets into current_savings. "
            "Read every message in the conversation to find all asset mentions."
        )),
        ("human", "Full conversation history:\n{history}\n\nLatest message: {user_message}"),
    ])
    try:
        return (prompt | structured_llm).invoke({
            "user_message": user_message, "history": history
        })
    except Exception as exc:
        logger.error("FIREParams extraction failed: %s", exc)
        raise RuntimeError(f"Could not extract FIRE parameters: {exc}") from exc


def extract_health_params(user_message: str, history: str = "") -> HealthScoreParams:
    """Extract Money Health Score parameters from user conversation."""
    llm = _build_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(HealthScoreParams)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _EXTRACTION_SYSTEM + (
            "\n\nExtract HealthScoreParams from the full conversation. "
            "CRITICAL: Apply Rule B to calculate equity/debt/gold % from asset amounts. "
            "Apply Rule C to compute annual tax_saving_investments from EPF/NPS/insurance. "
            "Apply Rule D to sum monthly retirement contributions. "
            "Read every message for asset and contribution mentions."
        )),
        ("human", "Full conversation history:\n{history}\n\nLatest message: {user_message}"),
    ])
    try:
        return (prompt | structured_llm).invoke({
            "user_message": user_message, "history": history
        })
    except Exception as exc:
        logger.error("HealthScoreParams extraction failed: %s", exc)
        raise RuntimeError(f"Could not extract Health Score parameters: {exc}") from exc


def extract_sip_params(user_message: str, history: str = "") -> SIPQueryParams:
    """Extract SIP Projection parameters from user conversation."""
    llm = _build_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(SIPQueryParams)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _EXTRACTION_SYSTEM + "\n\nExtract SIPQueryParams from the conversation. "
         "monthly_sip is REQUIRED — if the user hasn't stated a specific SIP amount, "
         "use their monthly income × 0.30 as a reasonable default. "
         "years is REQUIRED — if not stated, default to 10."),
        ("human", "Conversation:\n{history}\n\nLatest message: {user_message}"),
    ])
    try:
        return (prompt | structured_llm).invoke({
            "user_message": user_message, "history": history
        })
    except Exception as exc:
        logger.error("SIPQueryParams extraction failed: %s", exc)
        err_str = str(exc)
        if "monthly_sip" in err_str or "tool_use_failed" in err_str or "invalid_request" in err_str:
            raise RuntimeError(
                "I need a specific monthly SIP amount to run this projection. "
                "Please tell me how much you want to invest per month — "
                "for example: 'Project ₹15,000/month SIP for 10 years.'"
            ) from exc
        raise RuntimeError(f"Could not extract SIP parameters: {exc}") from exc


def extract_market_params(user_message: str, history: str = "") -> MarketDataParams:
    """Extract Market Data query parameters from user conversation."""
    llm = _build_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(MarketDataParams)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _EXTRACTION_SYSTEM + "\n\nExtract MarketDataParams from the conversation."),
        ("human", "Conversation:\n{history}\n\nLatest message: {user_message}"),
    ])
    try:
        return (prompt | structured_llm).invoke({
            "user_message": user_message, "history": history
        })
    except Exception as exc:
        logger.error("MarketDataParams extraction failed: %s", exc)
        raise RuntimeError(f"Could not extract market query parameters: {exc}") from exc


def extract_life_event_params(user_message: str, history: str = "") -> LifeEventParams:
    """Extract Life Event parameters from user conversation."""
    llm = _build_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(LifeEventParams)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _EXTRACTION_SYSTEM + (
            "\n\nExtract LifeEventParams from the conversation. "
            "event_type MUST be one of: BONUS, INHERITANCE, MARRIAGE, NEW_BABY, JOB_LOSS, HOME_PURCHASE. "
            "Map user language: 'got fired/laid off/lost job' → JOB_LOSS, "
            "'expecting/having a baby/child' → NEW_BABY, "
            "'getting married/wedding' → MARRIAGE, "
            "'received bonus/increment/hike' → BONUS, "
            "'buying house/flat/property' → HOME_PURCHASE, "
            "'received inheritance/windfall/gift' → INHERITANCE."
        )),
        ("human", "Full conversation history:\n{history}\n\nLatest message: {user_message}"),
    ])
    try:
        return (prompt | structured_llm).invoke({
            "user_message": user_message, "history": history
        })
    except Exception as exc:
        logger.error("LifeEventParams extraction failed: %s", exc)
        raise RuntimeError(f"Could not extract life event parameters: {exc}") from exc


# ============================================================================ #
#  SECTION 5 — RESPONSE GENERATION (quant result → natural language)           #
# ============================================================================ #

_RESPONSE_SYSTEM = """You are an AI Money Mentor — a warm, precise, and empathetic Indian financial advisor.
You have just been given structured results computed by a deterministic financial engine.
Your job is to explain these results to the user in clear, natural language.

STRICT RULES:
1. NEVER change, round, or reinterpret the numbers provided to you. Use them exactly.
2. Use Indian number formatting: ₹10,00,000 not ₹1,000,000. Lakh/Crore notation is encouraged.
3. Be encouraging but honest. If the outlook is poor, say so clearly with actionable next steps.
4. Keep responses under 300 words unless the result is a detailed roadmap.
5. Always end with ONE specific, actionable recommendation.
6. Do NOT mention "the quant engine", "the model", or any internal system names."""

_RESPONSE_HUMAN = """User's original question: {user_message}

Computed result (use these numbers exactly):
{result_json}

Additional context: {context}

Write a helpful, natural response explaining these results to the user."""


def generate_response(
    user_message: str,
    result_json: str,
    context: str = "",
) -> str:
    """
    Wrap a JSON-serialised quant result in natural language.

    Parameters
    ----------
    user_message : str
        The original user question (for contextual tone matching).
    result_json : str
        JSON string of the quant result. Use dataclass.__dict__ or model_dump().
    context : str
        Any additional context (e.g., "User is 28 years old and risk-tolerant.").

    Returns
    -------
    str — formatted natural language response ready for st.chat_message().
    """
    llm = _build_llm(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _RESPONSE_SYSTEM),
        ("human", _RESPONSE_HUMAN),
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({
            "user_message": user_message,
            "result_json": result_json,
            "context": context or "No additional context.",
        })
        return response.content
    except Exception as exc:
        logger.error("Response generation failed: %s", exc)
        return (
            f"I've calculated your results. Here's the raw output:\n\n"
            f"```\n{result_json}\n```\n\n"
            f"_(Response formatting failed: {exc})_"
        )


def generate_clarification_request(
    user_message: str,
    missing_fields: list[str],
    intent: Intent,
) -> str:
    """Ask the user for specific missing information needed to proceed."""
    llm = _build_llm(temperature=0.4)
    field_descriptions = {
        "current_age": "your current age",
        "retirement_age": "your target retirement age",
        "monthly_income": "your monthly income",
        "monthly_expenses": "your monthly expenses",
        "current_savings": "your existing savings or investments",
        "target_corpus": "your target retirement corpus / goal amount",
        "monthly_sip": "your current monthly SIP amount",
        "emergency_fund": "your emergency fund balance",
        "total_insurance_cover": "your total life insurance cover",
        "total_debt_emi": "your total monthly EMI payments",
        "gross_annual_income": "your annual gross income",
    }
    missing_readable = [field_descriptions.get(f, f.replace("_", " ")) for f in missing_fields]
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a friendly Indian financial advisor collecting information from a user. "
            "Ask for the missing information in a warm, conversational way. "
            "Do NOT ask for all fields in one overwhelming list — pick the 1-2 most critical ones first. "
            "Keep it under 60 words."
        )),
        ("human", (
            f"User wants: {{intent}}\n"
            f"User said: {{user_message}}\n"
            f"Missing info: {{missing}}\n"
            "Ask a natural follow-up question to get this info."
        )),
    ])
    try:
        response = (prompt | llm).invoke({
            "intent": intent.value,
            "user_message": user_message,
            "missing": ", ".join(missing_readable),
        })
        return response.content
    except Exception as exc:
        logger.warning("Clarification generation failed, using fallback: %s", exc)
        return f"To help you with this, I need a bit more information. Could you please share {missing_readable[0]}?"


def generate_general_response(user_message: str, history: str = "") -> str:
    """
    Handle GENERAL_QUERY intents and follow-up conversational questions.
    Has access to full conversation history so it can answer "what data did you use?"
    type questions without re-asking for information.
    """
    llm = _build_llm(temperature=0.3)
    messages = [
        SystemMessage(content=(
            "You are an AI Money Mentor — a knowledgeable, warm Indian financial advisor. "
            "You have memory of the full conversation and should use it to answer follow-up "
            "questions directly without asking for information already provided. "
            "If the user asks 'what data did you use?' or 'how did you calculate this?', "
            "explain the methodology using the numbers from the conversation history. "
            "Answer general financial questions about Indian products: MFs, EPF, PPF, NPS, ELSS, SGBs. "
            "If unsure, say so. Never invent statistics. Keep responses under 250 words."
        )),
        HumanMessage(content=f"Conversation so far:\n{history}\n\nQuestion: {user_message}"),
    ]
    try:
        return llm.invoke(messages).content
    except Exception as exc:
        logger.error("General response failed: %s", exc)
        return "I'm having trouble connecting right now. Please try again in a moment."


# ============================================================================ #
#  SECTION 6 — MAIN ORCHESTRATION FUNCTION (called by app.py)                 #
# ============================================================================ #

def format_history(messages: list[dict], max_messages: int = 10) -> str:
    """
    Format Streamlit session_state.messages into a plain-text string
    for LangChain prompt context.
    """
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:500] if len(msg["content"]) > 500 else msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
