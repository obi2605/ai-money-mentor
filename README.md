# 🏦 AI Money Mentor
### ET AI Hackathon 2026 — Problem Statement 9

> A mobile-first, chat-driven financial advisor that translates casual conversation into strict mathematical parameters — built for the Economic Times ecosystem.

---

## What It Does

AI Money Mentor is an embeddable chat widget that acts as an intelligent financial co-pilot. It extracts financial variables from natural language, routes them to a deterministic quantitative engine, and returns backtested, data-grounded advice.

**The LLM never does math.** Variable extraction and natural language generation are the only jobs the LLM performs. All calculations are handled by pure Python.

---

## Core Modules

### 🏥 Money Health Score
Conversational onboarding that scores the user across 6 SEBI-aligned dimensions:
- Emergency Preparedness (6-month expenses benchmark)
- Insurance Coverage (10x income benchmark)
- Investment Diversification (ideal 60/30/10 equity/debt/gold)
- Debt Health (EMI-to-income ratio)
- Tax Efficiency (80C/80D utilisation)
- Retirement Readiness (15% savings rate benchmark)

### 🔥 FIRE Path Planner
Takes age, income, and goals and outputs a month-by-month financial roadmap including:
- Inflation-adjusted corpus target (4% rule, 6% Indian inflation)
- Required vs actual SIP gap analysis
- Asset allocation glide path (age-based equity de-risking)
- **Backtested against real Nifty 50 historical data** (not simulated)
- 2008 GFC and 2020 COVID crash stress tests with RCA advantage
- 1,000-run Monte Carlo simulation with success probability

### 🔬 MF Portfolio X-Ray
Uploads a CAMS/KFintech Consolidated Account Statement PDF and produces:
- True XIRR (Extended Internal Rate of Return) per fund and portfolio
- Stock-level overlap analysis between equity funds
- 20-year expense ratio drag in rupees (not percentages)
- Direct plan switch recommendations
- **Processed 100% locally — raw PDF never sent to any API**

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit (custom ET red/black/white CSS) |
| LLM Orchestration | LangChain + Groq (Llama 3.3 70B) |
| Quant Engine | `numpy-financial`, `scipy.optimize` |
| Market Data | `yfinance` (with cached fallback for offline demos) |
| PDF Parsing | `pdfplumber` (local-only, privacy-first) |
| Visualisation | Plotly |

---

## Setup

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com) (no credit card required)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ai-money-mentor.git
cd ai-money-mentor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Open .env and add: GROQ_API_KEY=gsk_...

# 5. Launch
python -m streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
ai_money_mentor/
├── app.py                  # Streamlit UI + routing engine
├── quant_engine.py         # XIRR, SIP projection, health score math
├── llm_orchestrator.py     # LangChain intent detection + extraction
├── fire_planner.py         # FIRE roadmap + backtest + Monte Carlo
├── privacy_parser.py       # Local CAMS/Form-16 PDF parser
├── mf_xray.py              # MF portfolio XIRR + overlap + expense drag
├── requirements.txt
├── .env.example
├── data/
│   ├── mf_universe.json        # 24 popular Indian MF schemes (TER, holdings)
│   └── benchmark_returns.json  # Cached Nifty 50 / Sensex stats (offline fallback)
├── uploads/                # Temp dir for uploaded PDFs (gitignored)
└── tests/
    └── sample_cams.pdf     # Synthetic CAMS statement for testing
```

---

## Architecture

```
User (Chat / PDF Upload)
        │
        ▼
   app.py (Streamlit)
        │
        ├──► llm_orchestrator.py  →  Intent detection + variable extraction
        │         │
        │         ▼
        ├──► quant_engine.py      →  XIRR, SIP math, health score
        ├──► fire_planner.py      →  FIRE roadmap + backtest
        ├──► mf_xray.py           →  Portfolio X-Ray
        │
        └──► privacy_parser.py   →  Local PDF parsing (no network)
                  │
                  └── Raw PDF never leaves the device
```

**Data flow rule:** LLM extracts → Python calculates → LLM narrates. The LLM never sees raw financial data; it only sees structured results.

---

## Privacy

CAMS statements and Form 16s are parsed entirely locally using `pdfplumber`. The raw PDF bytes and extracted text never leave the user's machine. Only sanitised, structured data (fund names, transaction amounts, dates) is passed downstream — with PAN numbers masked to `ABCDE****F` and folio numbers hashed with SHA-256.

---

## Self-Tests

Each module has a built-in self-test. Run before launching the app:

```bash
python quant_engine.py    # XIRR, SIP projection, health score
python fire_planner.py    # FIRE math, Monte Carlo (no network needed)
python privacy_parser.py  # PII masking, transaction classifier
python mf_xray.py         # XIRR + overlap + drag on synthetic data
```

---

## Disclaimer

For educational purposes only. Not SEBI-registered financial advice. Mutual fund investments are subject to market risks. Please read all scheme-related documents carefully before investing.

---

*Built for ET AI Hackathon 2026 — Problem Statement 9: AI Money Mentor*
