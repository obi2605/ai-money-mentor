# AI Money Mentor — Project Structure
## ET AI Hackathon 2026 | Problem Statement 9

```
ai_money_mentor/
│
├── 📄 requirements.txt          # All dependencies (pinned versions)
├── 📄 .env.example              # Template: OPENAI_API_KEY=sk-...
├── 📄 .gitignore                # Excludes .env, __pycache__, uploads/
│
├── 📄 app.py                    # ★ ENTRY POINT — `streamlit run app.py`
│                                #   Handles routing, session state, and
│                                #   renders the chat widget with ET branding.
│
├── 📄 quant_engine.py           # ★ CORE MATH ENGINE (no LLM, pure Python)
│                                #   - calculate_xirr()
│                                #   - fetch_historical_rolling_return()
│                                #   - project_sip_corpus()
│                                #   - calculate_money_health_score()
│
├── 📄 llm_orchestrator.py       # ★ LangChain layer (intent routing + variable extraction)
│                                #   - detect_intent()       → routes to correct module
│                                #   - extract_fire_params() → age, income, goal, horizon
│                                #   - extract_health_params()
│                                #   - generate_response()   → wraps quant output in
│                                #                             natural language
│
├── 📄 privacy_parser.py         # ★ LOCAL-FIRST document processor (never hits an API)
│                                #   - parse_cams_pdf()      → extracts transactions
│                                #   - parse_form16_pdf()    → extracts income/TDS data
│                                #   - sanitise_pii()        → masks Folio/PAN before
│                                #                             passing to quant_engine
│
├── 📄 fire_planner.py           # FIRE Path Planner module
│                                #   - build_fire_roadmap()  → month-by-month schedule
│                                #   - backtest_against_nifty()  → 2020-crash stress test
│                                #   - suggest_asset_allocation()  → age-based glide path
│
├── 📄 mf_xray.py                # Mutual Fund Portfolio X-Ray module
│                                #   - reconstruct_portfolio()  → from parsed CAMS data
│                                #   - calculate_portfolio_xirr()
│                                #   - compute_overlap()        → fund-to-fund stock overlap
│                                #   - compute_expense_drag()   → TER impact over 20Y
│
├── 📁 static/
│   └── 📄 style.css             # ET branding: red (#E2001A), black, white
│                                #   Streamlit custom CSS injected via st.markdown()
│
├── 📁 data/
│   ├── 📄 mf_universe.json      # Static list of popular MF tickers + TER data
│   │                            # (sourced from AMFI; updated periodically)
│   └── 📄 benchmark_returns.json # Cached Nifty 50 / Sensex historical stats
│                                 # Fallback if yfinance is rate-limited at demo time
│
├── 📁 uploads/                  # Temp dir for user-uploaded CAMS/Form-16 PDFs
│   └── 📄 .gitkeep             # Committed empty; actual files in .gitignore
│
└── 📁 tests/
    ├── 📄 test_quant_engine.py  # Unit tests for all quant functions
    ├── 📄 test_privacy_parser.py
    └── 📄 sample_cams.pdf       # Anonymised CAMS statement for demo / testing
```

---

## Data Flow Diagram

```
User (Chat Input / PDF Upload)
        │
        ▼
  ┌─────────────┐
  │   app.py    │  ← Streamlit UI, session state, ET styling
  └──────┬──────┘
         │ raw text / file bytes
         ▼
  ┌──────────────────┐     PDF only    ┌──────────────────┐
  │ llm_orchestrator │ ───────────────►│ privacy_parser   │
  │  (LangChain)     │                 │  LOCAL ONLY      │
  │                  │◄── clean data ──│  No API calls    │
  └──────┬───────────┘                 └──────────────────┘
         │ structured params (dicts, floats)
         ▼
  ┌──────────────────┐
  │  quant_engine /  │  ← Pure deterministic Python math
  │  fire_planner /  │  ← numpy-financial, scipy, yfinance
  │  mf_xray         │
  └──────┬───────────┘
         │ typed result dataclasses
         ▼
  ┌──────────────────┐
  │ llm_orchestrator │  ← LLM wraps results in natural language
  │  generate_       │    ("Your XIRR is 14.2%, which means...")
  │  response()      │
  └──────┬───────────┘
         │ formatted string
         ▼
  ┌──────────────┐
  │   app.py     │  → st.chat_message() renders the reply
  └──────────────┘
```

---

## Build Order (Recommended for Solo Dev)

| Sprint | Files | Deliverable |
|--------|-------|-------------|
| Day 1 (Today) | `requirements.txt`, `quant_engine.py` | Core math engine + self-tests ✅ |
| Day 2 | `llm_orchestrator.py`, `app.py` (skeleton) | Chat UI + intent routing |
| Day 3 | `fire_planner.py` | FIRE roadmap + Nifty backtest |
| Day 4 | `privacy_parser.py`, `mf_xray.py` | CAMS local parser + XIRR X-Ray |
| Day 5 | `static/style.css`, `app.py` (polish) | ET branding, widget embed mode |
| Day 6 | `tests/`, `data/`, README | Testing, demo data, submission polish |
