<div align="center">

# 📦 ShipMate AI

### Hybrid LLM & ML Logistics Triage Engine

**Enterprise-ready AI support platform for European cross-border supply chain operations.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![Gemini Flash](https://img.shields.io/badge/Gemini_Flash-Google_GenAI-8E75B2.svg)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

**[🚀 Try the Live Demo](https://shipmate-ai.streamlit.app)**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [The Hybrid Intelligence Advantage](#-the-hybrid-intelligence-advantage)
- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Running Tests](#-running-tests)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🎯 Overview

ShipMate AI decouples **subjective customer sentiment** from **quantitative carrier telemetry**.

It pairs **Gemini Flash structured NLP** (which reads what the customer actually wants) with a **Scikit-Learn Random Forest classifier** (which predicts the real-time probability that a parcel breaches its SLA transit window). Support agents get both signals side by side, so urgency and risk are never confused with each other.

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
│      Gemini Flash       │   │  Scikit-Learn Pipeline  │   │ Mock Carrier Telemetry  │
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
| **Damaged cargo on arrival** | `CRITICAL` — refund / claim needed | **3.7%** `LOW` — parcel already delivered | Flags immediate claim action without raising a false delay alert |
| **Customs invoice discrepancy** | `HIGH` — customer inquiry | **82.4%** `CRITICAL` — dwell time > 36h | Auto-escalates the ticket to the logistics lead via webhook alert |
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

**Automated escalation router**
- Webhook alert dispatcher for high-risk customs exceptions and critical delays.

**Full automated test suite**
- `pytest` coverage across schema integrity, carrier fallback logic, and ML pipeline inference.

---

## 📁 Repository Structure

```text
logistics-ai-triage/
├── app/
│   ├── __init__.py
│   ├── schemas.py            # Pydantic schemas & response contracts
│   ├── llm_agent.py          # Gemini Flash structured extraction & drafting
│   ├── mock_carrier_api.py   # European parcel routing & scan telemetry
│   ├── ml_delay_model.py     # Scikit-learn Random Forest delay classifier
│   ├── notifier.py           # Escalation alert webhook dispatcher
│   └── main.py               # FastAPI REST backend service
├── frontend/                 # Optional React 19 + TypeScript + Tailwind UI
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── tests/
│   └── test_triage.py        # Pytest automated test suite
├── streamlit_app.py          # Unified interactive Streamlit application
├── requirements.txt          # Python dependencies
├── pytest.ini                # Pytest configuration
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
git clone https://github.com/<YOUR_USERNAME>/logistics-ai-triage.git
cd logistics-ai-triage

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

The suite covers Pydantic schema integrity, carrier fallback behaviour, and end-to-end ML pipeline inference.

---

## 🛠 Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend & API** | Python 3.11+, FastAPI, Uvicorn, Pydantic v2 |
| **Machine Learning** | Scikit-Learn (`RandomForestClassifier`, `ColumnTransformer`), Pandas, NumPy, Joblib |
| **NLP** | Google GenAI SDK, Gemini Flash |
| **Frontend** | Streamlit, React 19, TypeScript, Vite, Tailwind CSS, Lucide Icons |
| **Testing** | Pytest |

---

## 📄 License

Released under the [MIT License](LICENSE).

---

<div align="center">

Built for European cross-border logistics operations.

</div>