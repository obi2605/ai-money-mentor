# 🏦 AI Money Mentor
### ET AI Hackathon 2026 — Problem Statement 9

> A chat-driven financial advisor that translates casual conversation into strict mathematical parameters — built for the Economic Times ecosystem.

**The LLM never does math.** Variable extraction and natural language generation are its only jobs. Every calculation is deterministic Python — backtest-validated and regulation-accurate for Indian markets.

---

## Features

### 🏥 Money Health Score
Scores the user across 6 SEBI-aligned dimensions with a weighted composite formula:

| Dimension | Benchmark | Weight |
|-----------|-----------|--------|
| Emergency Preparedness | 6 months of expenses in liquid assets | 20% |
| Insurance Coverage | 10× annual income in term cover | 20% |
| Investment Diversification | 60% equity / 30% debt / 10% gold | 15% |
| Debt Health | EMI-to-income ratio ≤ 30% | 20% |
| Tax Efficiency | 80C/80D utilisation | 10% |
| Retirement Readiness | 15% of gross income in EPF/PPF/NPS | 15% |

### 🔥 FIRE Path Planner
Month-by-month retirement roadmap with three layers of analysis:
- Inflation-adjusted corpus target (4% SWR, 6% Indian inflation)
- Required SIP back-solved via `scipy.optimize.brentq` with optional 10% step-up
- Historical SIP backtest against real Nifty 50 monthly closing prices via `yfinance`
- 2008 GFC and 2020 COVID crash stress tests
- 1,000-run Monte Carlo simulation (log-normal returns, μ=11.7%, σ=16%) with P10/P50/P90 range

### 🔬 MF Portfolio X-Ray
Parses a CAMS or KFintech Consolidated Account Statement PDF **100% locally**:
- True XIRR per fund and portfolio-level XIRR from actual transaction cash flows
- Stock-level overlap analysis between equity funds
- 20-year expense ratio drag in rupees (not percentages)
- Direct plan switch recommendations

### 📅 Life Event Advisor
Six deterministic allocation engines — each with priority-ordered allocation logic, tax tips, and insurance gap analysis:
- **BONUS** — tax reserve → emergency top-up → debt clearance → 80C/NPS → equity SIP
- **INHERITANCE** — same as bonus with estate planning notes
- **MARRIAGE** — joint account setup, insurance review, HRA optimisation
- **NEW_BABY** — education corpus target (~₹1.1Cr at 10% inflation for 18 years), required SIP back-solved
- **JOB_LOSS** — runway in months, SIP pause strategy, severance deployment
- **HOME_PURCHASE** — EMI at 8.5%/20yr, 30% affordability check, Section 24B/80C/80EEA tax benefits

### 🧾 Tax Wizard
Full FY 2025-26 (AY 2026-27) Indian income tax calculator:
- New regime: 7-slab structure, ₹75K standard deduction, 87A rebate up to ₹12L taxable
- Old regime: 3-slab structure, ₹50K standard deduction, HRA exemption (min of 3 rules)
- Deduction gap analysis: identifies unused 80C, 80CCD(1B), 80D, HRA headroom with rupee savings
- Ranked investment suggestions by risk profile

### 👫 Couples Planner
Joint financial optimiser for two-income households:
- HRA optimiser: tests three split scenarios and picks the combination that maximises combined tax saving
- NPS matching: allocates ₹50K 80CCD(1B) headroom to the higher-bracket partner first
- SIP split: routes ELSS to higher-bracket partner; blocks homemaker equity routing due to **Section 64(1)(iv) clubbing**
- 20-year joint net worth projection with insurance gap check

---

## Architecture

```
User message / PDF upload
        │
        ▼
financial_preprocessor.py   ← Deterministic regex scanner — runs BEFORE the LLM
        │                      Extracts income, expenses, assets from raw text
        │                      Values hard-override LLM extraction for numeric fields
        ▼
llm_orchestrator.py
  ├── detect_intent()         ← Classifies into 9 intents (deterministic overrides enforce priority)
  └── extract_*_params()      ← Pydantic structured output; preprocessor values prepended
        │
        ▼
Quant engine (pure Python, no LLM, no network)
  ├── quant_engine.py         ← XIRR, SIP projection, health score
  ├── fire_planner.py         ← FIRE roadmap, backtest, Monte Carlo
  ├── life_event_advisor.py   ← 6 life event allocation engines
  ├── tax_wizard.py           ← FY 2025-26 tax math
  ├── couples_planner.py      ← Joint financial optimiser
  └── mf_xray.py              ← Portfolio XIRR, overlap, expense drag
        │
        ▼
llm_orchestrator.py
  └── generate_response()     ← LLM narrates quant results (never alters numbers)
        │
        ▼
app.py (Streamlit UI + Plotly charts)
```

**Routing rule:** The HEALTH_SCORE override backs off if the message contains FIRE/retirement, couple, or tax signals. The life event override uses word-boundary regex (`\bhome purchase\b` not `home`) to prevent false matches like "homemaker".

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit 1.43.0 — custom ET red/black/white CSS |
| LLM | LangChain 0.3+ — Groq API (llama-3.3-70b-versatile) |
| Quant | `numpy-financial 1.0`, `scipy 1.15.2`, `pandas 2.2.3` |
| Market data | `yfinance 0.2.51` with cached fallback (`benchmark_returns.json`) |
| PDF parsing | `pdfplumber 0.11.4` — local only, never hits network |
| Visualisation | `plotly 5.24.1` |

---

## Setup

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com) — no credit card required

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/obi2605/ai-money-mentor.git
cd ai-money-mentor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Open .env and set: GROQ_API_KEY=gsk_...

# 5. Run module self-tests (optional but recommended)
python quant_engine.py
python fire_planner.py
python tax_wizard.py

# 6. Launch
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Project Structure

```
ai_money_mentor/
├── app.py                      # Streamlit UI, routing engine, all render functions
├── quant_engine.py             # XIRR, SIP projection, money health score math
├── llm_orchestrator.py         # LangChain intent detection + extraction (Groq)
├── financial_preprocessor.py   # Deterministic regex pre-processor (runs before LLM)
├── fire_planner.py             # FIRE roadmap, backtest, Monte Carlo
├── privacy_parser.py           # Local CAMS / Form 16 PDF parser
├── mf_xray.py                  # MF portfolio XIRR, overlap, expense drag
├── life_event_advisor.py       # 6 life event allocation engines
├── tax_wizard.py               # Indian tax calculator (FY 2025-26)
├── couples_planner.py          # Joint financial optimiser
├── requirements.txt
├── .env.example
├── data/
│   ├── mf_universe.json        # 24 Indian MF schemes — TER, top holdings
│   └── benchmark_returns.json  # Cached Nifty 50 / Sensex stats (offline fallback)
├── uploads/                    # Temp dir for uploaded PDFs (gitignored)
└── tests/
    └── sample_cams.pdf         # Synthetic CAMS statement for testing
```

---

## Privacy

CAMS statements and Form 16s are parsed entirely locally using `pdfplumber`. The raw PDF bytes and extracted text never leave the user's machine. Only sanitised structured data is passed downstream:

- PAN numbers masked to `ABCDE****F`
- Folio numbers replaced with SHA-256 hash (first 8 chars)
- Aadhaar, mobile, and email addresses stripped by `privacy_parser.sanitise_pii()`

No user financial data is ever transmitted to the Groq API. The LLM receives only structured parameters — never raw conversation text containing personal identifiers.

---

## Sample Prompts

```
# Money Health Score
I earn ₹80,000/month, expenses ₹50,000, emergency fund ₹1.8L,
term insurance ₹50L, EMI ₹15,000, portfolio 60% equity 30% debt 10% gold

# FIRE Planner
I am 30 years old, earn ₹1L/month, expenses ₹55,000, savings ₹2L,
SIP ₹10,000/month, want to retire at 55

# Tax Wizard
I earn ₹7.5L/year, no investments, should I file ITR?

# Life Event — Home Purchase
I want to buy a house worth ₹1.2 crore, I earn ₹1.8L/month take-home

# Couples Planner
I earn ₹5L/month, wife is a homemaker, we have ₹50L in savings

# MF X-Ray
Upload your CAMS PDF in the sidebar, then type: Analyse my mutual fund portfolio
```

---

## Self-Tests

Each module has a built-in self-test. Run before launching:

```bash
python quant_engine.py      # XIRR, SIP projection, health score
python fire_planner.py      # FIRE math, Monte Carlo (fetches live Nifty data)
python tax_wizard.py        # FY 2025-26 slab math, 87A rebate edge cases
python privacy_parser.py    # PII masking, transaction classifier
python mf_xray.py           # XIRR + overlap + drag on synthetic data
```

---

## Disclaimer

For educational purposes only. Not SEBI-registered financial advice. Mutual fund investments are subject to market risks. Please read all scheme-related documents carefully before investing.

---

*Built for ET AI Hackathon 2026 — Problem Statement 9*
