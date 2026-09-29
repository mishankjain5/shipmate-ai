<div align="center">

# 📦 ShipMate AI

### Conversational, Agentic & Explainable AI for Logistics Support

**Enterprise-ready AI support platform for European cross-border supply chain operations.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Gemini Flash](https://img.shields.io/badge/Gemini_Flash-Google_GenAI-8E75B2.svg)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)

**[🚀 Try the Live Demo](https://shipmate-ai.streamlit.app)**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [The Hybrid Intelligence Advantage](#-the-hybrid-intelligence-advantage)
- [Features](#-features)
- [Conversational, Agentic & Explainable AI](#-conversational-agentic--explainable-ai)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Running Tests](#-running-tests)
- [Evaluation](#-evaluation)
- [Tech Stack](#-tech-stack)

---

## 🎯 Overview

ShipMate AI decouples **subjective customer sentiment** from **quantitative carrier telemetry**.

It pairs **Gemini Flash structured NLP** (which reads what the customer actually wants) with a **Scikit-Learn Random Forest classifier** (which predicts the real-time probability that a parcel breaches its SLA transit window). Support agents get both signals side by side, so urgency and risk are never confused with each other.

On top of that sit **tool-using AI agents**: a customer chat assistant, a ticket investigation agent and an ops copilot. They decide which tools to call (carrier lookup, delay model, policies, CRM), queue any real-world action for **human approval**, and **explain every decision** with Shapley values, verbatim evidence and a decision trace.

---

## 🏗 System Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │         Customer Support Interfaces         │
                    │      Streamlit App  ·  React 19 + Vite      │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │            ShipMate Core Service            │
                    │           FastAPI  ·  Python 3.11           │
                    └────────┬─────────────┬─────────────┬────────┘
                             │             │             │
             ┌───────────────┘             │             └───────────────┐
             ▼                             ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│      Gemini Flash       │   │  Scikit-Learn Pipeline  │   │ Carrier Simulator (API) │
│     Structured NLP      │   │      RandomForest       │   │     DHL · DPD · GLS     │
│    Entity Extraction    │   │ SLA Breach Probability  │   │   PostNL · Colissimo    │
└─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
```

---

## 💡 The Hybrid Intelligence Advantage

Traditional support bots only measure **how upset the customer sounds**. Standard logistics dashboards only show **static tracking codes**. Neither answers the question an agent actually has: *what do I do with this ticket right now?*

ShipMate AI scores both axes independently and routes on the combination.

| Operational Scenario | LLM Urgency (Gemini) | ML SLA Breach Risk (Random Forest) | Operational Result |
| :--- | :--- | :--- | :--- |
| **Damaged cargo on arrival** | `CRITICAL` — refund / claim needed | **N/A** — parcel already delivered, so the model is not run | Flags immediate claim action without raising a false delay alert |
| **Customs invoice discrepancy** | `HIGH` — customer inquiry | **98.2%** `CRITICAL` — held 44h at Roissy customs | Auto-escalates the ticket to the logistics lead via webhook alert |
| **Routine status check** | `LOW` — polite check-in | **12.0%** `LOW` — normal hub throughput | AI-generated draft reply ready for one-click dispatch |

> Figures above are illustrative walkthroughs of the demo scenarios shipped with the app.

---

## 🚀 Features

**Dual-portal workflow**
- **Customer portal** — clean intake form with one-click European scenario presets (delayed orders, customs holds, damaged cargo).
- **Agent workspace** — queue monitoring, telemetry inspection, ML breach gauge, and an auto-draft reply editor.

**Deterministic NLP entity extraction**
- Gemini Flash constrained by strict Pydantic schemas.
- Extracts tracking numbers, categorises the root issue, and classifies urgency as `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`.

**Predictive ML delay model**
- Trained on simulated cross-border transit data across European hubs: Potsdam, Leipzig, Frankfurt, Roissy CDG Customs, and Amsterdam.
- Features: `carrier`, `last_hub`, `dwell_time_hours`, `is_cross_border`.
- Only scores shipments that are actually in transit. Missing, unknown or already-delivered parcels are shown as "not applicable" instead of getting a made-up risk.

**Carrier integration layer**
- The app depends on a `CarrierClient` interface. The demo uses a **deterministic shipment simulator**: any tracking number maps (by hash) to one consistent shipment across 5 European lanes, with a realistic scan-event history.
- Dwell time is computed from the last carrier scan, not hard-coded, so the ML inputs vary per shipment.
- Malformed or unregistered tracking numbers return `NOT_FOUND`, so typos are caught.
- Demo presets: `SHIP-1001` (stalled at Potsdam), `SHIP-2002` (customs hold at Roissy CDG), `SHIP-3003` (delivered to Amsterdam). A real carrier or tracking-aggregator API can replace the simulator without changing the AI layer.

**Automated escalation router**
- Webhook alert dispatcher for high-risk customs exceptions and critical delays.

**Full automated test suite**
- `pytest` coverage across schema integrity, the carrier simulator, ML pipeline inference, Shapley explanations, and the agent loop (with a scripted fake LLM, so no API key is needed).

---

## 🧠 Conversational, Agentic & Explainable AI

### 💬 Conversational: AI Chat Assistant
- Multi-turn chat with memory. The assistant asks for missing details (e.g. a tracking number) instead of guessing, and replies in the customer's language.
- When the assistant hands off to a human or requests an action, the conversation automatically becomes a triaged ticket with its full transcript.
- **Ops Copilot**: support staff ask questions about the queue in plain language ("Which tickets have a delay risk above 70%?"), answered only from live ticket data through tools.

### 🤖 Agentic: tool-calling agents with human-in-the-loop
The LLM decides which tools to call through a Gemini function-calling loop (`app/agent.py`):

| Tool | Effect |
| :--- | :--- |
| `lookup_shipment`, `assess_delay_risk`, `check_policy`, `get_customer_history` | Read-only: run automatically |
| `escalate_to_operations`, `open_carrier_investigation`, `handoff_to_human` | Internal: run automatically |
| `issue_voucher`, `request_customs_documents` | **Queued for human approval** |

Guardrails: a step limit, business-rule validators (voucher cap of EUR 200, manager approval above EUR 50), customer data bound server-side (the model cannot look up another customer), prompt-injection instructions, retry with backoff on rate limits, and a full reasoning trace for every step.

### 🔍 Explainable: every decision comes with a reason
- **Exact Shapley values** for the Random Forest. There are only 4 features, so all 16 coalitions are enumerated with no approximation. Base value plus contributions equals the prediction exactly.
- **Counterfactuals**: "If dwell time were 34h instead of 44h, risk would drop to 16%."
- **Data-quality warnings**: flags carrier or hub values the model never saw in training.
- **LLM reasoning and verbatim evidence**: the triage LLM must quote the ticket, and any quote that doesn't appear in the text is discarded automatically (hallucination guard). The quotes are highlighted in the UI.
- **Escalation decision trace**: lists exactly which rules fired (ML risk, customs hold, LLM urgency).
- **Plain-English narration**: the LLM turns the Shapley values into an explanation a non-technical agent can read.

---

## 📁 Repository Structure

```text
shipmate-ai/
├── app/
│   ├── schemas.py            # Pydantic schemas & response contracts
│   ├── llm_agent.py          # Gemini Flash structured extraction & drafting
│   ├── mock_carrier_api.py   # CarrierClient interface + deterministic shipment simulator
│   ├── ml_delay_model.py     # Scikit-learn Random Forest delay classifier
│   ├── notifier.py           # Escalation alert webhook dispatcher
│   ├── triage.py             # Shared triage pipeline + escalation decision trace
│   ├── explainability.py     # Exact Shapley values, counterfactuals, data warnings
│   ├── agent.py              # Tool-calling agent loop, guardrails, approval queue
│   ├── conversation.py       # Multi-turn customer chat sessions
│   ├── knowledge_base.py     # Mock support policies & CRM history
│   ├── actions.py            # Side-effecting actions (run only after approval)
│   └── main.py               # FastAPI REST backend service
├── frontend/                 # Optional React 19 + TypeScript + Tailwind UI
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── tests/
│   ├── test_triage.py        # Carrier, ML and schema tests
│   ├── test_explainability.py# Shapley, counterfactual, evidence and decision-trace tests
│   ├── test_carrier_simulator.py # Determinism, scan history and not-found handling
│   └── test_agent.py         # Agent loop tests with a scripted fake LLM
├── evals/
│   ├── triage_cases.json     # 30 hand-labelled tickets
│   ├── run_evals.py          # Triage, agent-behaviour and ML evaluation harness
│   └── results/latest.md     # Latest evaluation report
├── streamlit_app.py          # Unified interactive Streamlit application
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Prerequisites

- Python 3.11 or newer
- A [Google AI Studio API key](https://aistudio.google.com/)
- Node.js 18+ *(only if you want to run the optional React frontend)*

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/mishankjain5/shipmate-ai.git
cd shipmate-ai

# Create and activate a virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
# Optional: post escalation alerts to Slack (otherwise they are printed to the console)
SLACK_WEBHOOK_URL=
```

### 4. Run the Streamlit App (unified demo)

```bash
streamlit run streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

### 5. Run the FastAPI Backend (optional)

```bash
uvicorn app.main:app --reload
```

Interactive API docs are served at **http://localhost:8000/docs**.

### 6. Run the React Frontend (optional)

```bash
cd frontend
npm install
npm run dev
```

---

## 🧪 Running Tests

```bash
python -m pytest
```

The suite covers Pydantic schema integrity, the carrier simulator, ML inference and explanations, and the agent loop.

---

## 📏 Evaluation

Unit tests check the code. The evaluation harness checks the **AI behaviour** against the live model:

```bash
python -m evals.run_evals            # all suites (~5 min on the free tier; throttled to 14 requests/min)
python -m evals.run_evals --suite ml # ML only, no API calls
```

Latest run (`gemini-3.5-flash-lite`, full report in [evals/results/latest.md](evals/results/latest.md)):

| Suite | What it measures | Result |
| :--- | :--- | :--- |
| **Triage extraction** (30 hand-labelled tickets in 5 languages, incl. sarcasm, missing tracking numbers and prompt injection) | Category accuracy | 100% |
| | Urgency exact / within one level | 70% / 100% |
| | Tracking-number exact match (incl. correctly returning none) | 100% |
| | Language detection | 100% |
| | Invented evidence quotes caught by the verbatim filter | 1 of 60 |
| **Agent behaviour** (10 tool-use scenarios) | Right tools, actions queued not executed, guardrails, injection, data isolation, handoff | 10 / 10 |
| **Delay model** (held-out 25%) | ROC-AUC / Brier score | 0.987 / 0.034 |

Where the LLM disagrees with the labels, it is on urgency by one level: it rates damage claims harsher (critical instead of high) and delays or lost parcels softer. The datasets are small and self-labelled, and the ML labels are synthetic, so these numbers show the method and catch regressions. They are not production accuracy claims. Slack alerts are disabled during evaluation runs.

---

## 🛠 Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend & API** | Python 3.11+, FastAPI, Uvicorn, Pydantic v2 |
| **Machine Learning** | Scikit-Learn (`RandomForestClassifier`, `ColumnTransformer`), exact Shapley explanations, Pandas, NumPy, Joblib |
| **LLM & Agents** | Google GenAI SDK, Gemini 3.5 Flash-Lite (structured output, function calling), custom agent loop with human-in-the-loop approvals |
| **Frontend** | Streamlit, React 19, TypeScript, Vite, Tailwind CSS, Lucide Icons |
| **Testing & Evaluation** | Pytest (fake-LLM agent tests), live LLM / agent / ML evaluation harness |

---

<div align="center">

Built for European cross-border logistics operations.

</div>