# Smart Sales Forecasting System

> An end-to-end machine learning and full-stack platform for analyzing historical sales performance and forecasting future product demand across 7, 30, and 90-day horizons.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-Frontend-646CFF.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-v4-06B6D4.svg)](https://tailwindcss.com/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00.svg)](https://www.tensorflow.org/)
[![Tests](https://img.shields.io/badge/Backend%20Tests-302%2F302%20Passing-success.svg)](#testing)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

## Overview

The **Smart Sales Forecasting System** is an end-to-end data science and full-stack application designed to help businesses understand historical sales performance and forecast future demand.

The system combines:

- Historical sales analysis
- Feature engineering
- Statistical forecasting baselines
- Machine learning forecasting
- LSTM experimentation
- Production model selection
- FastAPI REST APIs
- JWT authentication
- Product and sales management
- Interactive React analytics dashboards
- Real-time forecast generation from trained ML artifacts

The platform supports demand forecasting for:

- **7 days**
- **30 days**
- **90 days**

The forecasting engine uses different production models for each horizon based on the evaluated model performance:

| Forecast Horizon | Production Model |
|---|---|
| 7 Days | Random Forest |
| 30 Days | Gradient Boosting |
| 90 Days | Linear Regression |

---

## Problem Statement

Businesses generate large volumes of sales data but often struggle to convert historical transactions into reliable demand forecasts.

Without forecasting, businesses may face:

- Overstocking
- Stockouts
- Poor inventory planning
- Inefficient purchasing
- Difficulty identifying demand patterns
- Weak sales planning
- Limited visibility into future demand

This project addresses the problem by building a complete forecasting pipeline that transforms historical sales data into actionable demand predictions through machine learning.

---

# Key Features

## Machine Learning

- Historical sales data processing
- Exploratory Data Analysis
- Time-series feature engineering
- Lag features
- Rolling statistics
- Calendar features
- Promotion and holiday indicators
- Baseline forecasting
- Classical ML forecasting
- LSTM experimentation
- Rolling-origin evaluation
- Horizon-specific production models
- Forecast generation through trained artifacts

## Analytics

- Total historical quantity
- Total historical sales
- Total historical profit
- Average daily demand
- Historical demand visualization
- Revenue visualization
- Profit visualization
- Forecast visualization
- Forecast statistics
- Peak and lowest predicted demand
- Forecast period demand

## Sales Anomaly Detection (Phase 6.1)

- Causal rolling historical baselines (strict chronological lookback, zero future data leakage)
- Robust statistical scoring using rolling Median and Median Absolute Deviation (MAD × 1.4826)
- Multi-metric outlier detection (Daily Quantity and Daily Sales Revenue)
- Flexible granularity: Aggregate System Sales, Category-level, and Product-level detection
- Statistically grounded severity classification (Low, Medium, High, Critical based on standard normal tail probabilities)
- Directional anomaly tracking (Spikes vs Drops with absolute & percentage deviations)
- Contextual, interpretable natural language explanations accounting for promotions and holidays
- Dedicated React Anomaly Insights UI with interactive KPI telemetry, multi-dimensional filters, and paginated records

## Sales Anomaly Investigation & Root-Cause Attribution (Phase 6.2)

- Multi-dimensional root-cause attribution: Promotion, Holiday, Category, Product, Price, Discount, and Trend Drift
- Empirical decomposition attributing aggregate deviations to specific categories and products
- Quantified business impact: Excess demand / deficit volume, potential revenue gap / excess revenue, with documented price basis assumptions
- Evidence confidence grading: Transparent confidence classification (High, Medium, Low) based on historical sample size and consistency
- Strict causal invariance: All reference baselines and event metrics use data strictly prior to the anomaly date (t < T)
- Interactive slide-over Investigation Drawer integrated directly into the `/anomalies` table
- Deterministic narrative synthesis without external black-box LLM dependencies
- Explicit observational disclaimer: Identifies statistically associated contributors and supporting evidence; does not establish causal relationships from observational sales data

## Sales Anomaly Explanation & Executive Narrative Reporting (Phase 6.3)

- **Purpose**: Converts structured multi-dimensional investigation evidence into deterministic, executive-ready narrative briefings for leadership and stakeholders.
- **Deterministic Explanation Engine**: 100% reproducible template-driven engine operating without external LLM dependencies, ensuring zero hallucination risk and consistent mathematical phrasing.
- **Structured Narrative Sections**:
  - *Executive Headline*: Concise 1-sentence summary of metric, deviation percentage, direction, and date.
  - *What Happened*: Detailed breakdown of actual vs expected 28-day baseline, net deviation, severity, and anomaly score.
  - *Why It Matters*: Executive interpretation of business impact, translating units to potential revenue with documented price basis assumptions.
  - *Key Contributors*: Top evidence-backed driver summaries with observed vs reference values, deviation percentage, contribution share, and confidence grade.
  - *Event & Calendar Context*: Factual alignment with active promotional campaigns or recognized holiday dates.
  - *Trend & Drift Context*: 7-day pre-anomaly drift assessment evaluating whether the anomaly followed a rising, falling, or stable demand trajectory.
  - *Evidence Quality & Confidence*: Synthesized overall confidence assessment (High, Medium, Low) based on empirical sample depth and driver corroboration.
  - *Methodological Limitations*: Explicit disclaimers regarding observational attribution, absence of counterfactual causality, and lookback window assumptions.
- **Evidence Prioritization Rules**: Selects up to top-$N$ (default 3) primary drivers prioritized by contribution magnitude, confidence weighting, and directional alignment with the anomaly.
- **Interactive UI with 1-Click Clipboard Export**: Features an "Executive Brief" view within the Investigation Drawer and a "Copy Brief" action providing clean Markdown formatting.
- **API Endpoint**: `GET /api/explanations/{anomaly_id}` supporting `top_n` and `format` query parameters.

## Sales Anomaly AI Reasoning Layer (Phase 6.4)

- **Purpose**: Augments deterministic statistical investigations and executive explanations with commercial business reasoning powered by Google Gemini (`gemini-3.8-flash`) via the official `google-genai` SDK.
- **Evidence-Grounded Intelligence**: Operates strictly on compact, structured JSON evidence packages (< 2 KB) generated from Phase 6.2 and 6.3. Never ingests raw transactional datasets, preventing context bloat and data leakage.
- **Provider Abstraction**: Extensible `LLMProvider` interface decoupled from concrete implementation, facilitating pluggable backends and reliable unit testing.
- **Strict Structured Outputs**: Enforces JSON schema compliance via Pydantic (`AIReasoningResponse`), guaranteeing structured headline, commercial interpretation, granular key insights, alternative explanations, data quality assessment, uncertainties, validation questions, risk flags, and standard causal disclaimers.
- **Causal Guardrails & Post-Validation**: Programmatically sanitizes unauthorized causal phrasing (e.g., converting "caused by" or "because of" to observational associations) and enforces mandatory causal disclaimers.
- **Cost & Latency Optimization**: In-memory caching per anomaly ID prevents redundant LLM inference costs and latency, with optional on-demand cache bypass (`?refresh=true`).
- **Graceful Degradation**: If `GEMINI_API_KEY` is omitted or upstream API errors occur, returns HTTP 503 with helpful setup instructions while keeping the rest of the application 100% operational.
- **Interactive UI Tab**: Dedicated "AI Reasoning" tab in the Investigation Drawer featuring confidence pills, driver tags, risk flags, validation checklists, and a 1-click refresh analysis trigger.
- **API Endpoint**: `GET /api/ai-reasoning/{anomaly_id}` supporting `refresh` query parameter.

## Prescriptive Action Recommendations (Phase 6.5)

- **Purpose**: Bridges the gap between diagnostic analytics and commercial decision-making by synthesizing structured, evidence-grounded advisory recommendations for operational stakeholders.
- **Strict Non-Autonomous Governance**: Recommendations are framed exclusively for human consideration using non-autonomous verbs (*Review*, *Consider*, *Validate*, *Monitor*, *Reassess*). Autonomous execution (e.g., automated purchase orders, algorithmic price cuts) is structurally prohibited, and `human_approval_required = True` is invariant across all schemas.
- **Deterministic Eligibility Engine**: Evaluates empirical anomaly evidence against deterministic policy rules across 8 supported categories:
  1. `inventory_review`: Triggered on positive volume/revenue spikes with category/product driver participation.
  2. `pricing_review`: Triggered when unit price realization shifted materially.
  3. `promotion_review`: Triggered when promotional events coincide with the deviation.
  4. `category_review`: Triggered when category contribution exceeds 5.0%.
  5. `product_review`: Triggered when product-level attribution is identified.
  6. `demand_monitoring`: Triggered on drops, unstable drift, or moderate/low evidence confidence.
  7. `forecast_review`: Triggered on high/critical severity or large percentage deviations (≥ 10.0%).
  8. `data_validation`: Triggered on low confidence signals or extreme anomaly scores (≥ 4.0).
- **Deterministic Prioritization & Top-3 Capping**: Candidate actions are prioritized using a deterministic scoring formula combining severity (weight 2.0), driver contribution (weight 0.1), evidence confidence (weight 1.5), and risk penalty (-0.5), strictly capping output at at most 3 primary recommendations.
- **Robust Guardrails & Post-Validation**:
  - Rejects ineligible recommendation types and external speculative actions.
  - Sanitizes hallucinated imperative targets (e.g. "increase inventory by 20%" → safe advisory phrasing).
  - Sanitizes price cut directives and ungrounded guaranteed revenue/profit claims.
  - Caps recommendation confidence against underlying empirical driver confidence.
  - Sanitizes causal language to maintain observational integrity.
- **Deterministic Fallback Engine**: If Gemini is unconfigured or encounters upstream API timeouts, gracefully synthesizes grounded recommendations from deterministic policy rules (`source = "deterministic_fallback"`), ensuring 100% platform availability.
- **Dedicated React UI Tab**: "Recommendations" tab (`Lightbulb` icon) in the Investigation Drawer featuring prominent Human Review Notice, fallback alert badges, executive synthesis card, action cards with risk/priority badges, trade-offs, and verification checklists.
- **API Endpoint**: `GET /api/recommendations/{anomaly_id}` supporting `refresh` and `fallback` query parameters.

## What-If Scenario Simulation (Phase 6.6)

- **Purpose**: Enables interactive, non-destructive counterfactual exploration of hypothetical commercial scenarios across 7, 30, and 90-day forecast horizons without altering underlying sales history.
- **Scenario Drivers**:
  - `demand_multiplier`: Uniform demand shifts (-50% to +50%).
  - `temporary_shock`: Bounded transient demand shocks reverting to baseline.
  - `persistent_shift`: Permanent structural changes in baseline volume.
  - `trend_continuation`: Forward projection of recent 7–90 day velocity.
  - `promotion_scenario`: Evaluates promotional lift via production ML model.
  - `holiday_scenario`: Evaluates calendar holiday lift via production ML model.
- **Strict Elasticity Boundaries**: Price and discount adjustments return explicit `status = "requires_model"` disclaimers, preventing ungrounded price-elasticity assertions.
- **Numerical Invariants**: Preserves 10 strict mathematical invariants across all runs (daily additivity, delta coherence, non-negative quantities, finite bounds, and date continuity).
- **API Endpoints**: `POST /api/simulations` and `GET /api/simulations/{id}`.

## Human Approval & Decision Governance (Phase 6.7)

- **Purpose**: Establishes an auditable, enterprise decision governance workflow connecting AI/prescriptive recommendations and simulation findings to accountable human stakeholders.
- **Strict Non-Autonomous Execution**: Guarantees `human_approval_required: true` and `automatic_execution: false` across all records. Approval explicitly represents human sign-off for manual business execution; no purchase orders, price updates, or inventory reallocations are triggered autonomously.
- **State Machine Governance**: Formal state machine enforcing transitions: `pending_review` → `approved` / `rejected` / `changes_requested`, `changes_requested` → `pending_review` (resubmit), and terminal immutability for `approved` and `rejected` states. Invalid transitions return HTTP 409 Conflict.
- **Original Recommendation Immutability**: The machine-generated recommendation is permanently frozen; human revisions are strictly isolated in `modified_action`.
- **Mandatory Decision Rationale**: Rejections and change requests strictly require human rationales (1–2,000 characters).
- **Multi-Layer Evidence Snapshotting**: Captures immutable point-in-time JSON snapshots of the recommendation, anomaly investigation evidence, and associated what-if simulation run upon creation.
- **Immutable Audit Trail**: Chronologically records all lifecycle events (creation, review, notes, modifications, and state transitions) with authenticated actor attribution.
- **Dedicated React Decision Center**: Interactive UI at `/decisions` and `/decisions/:id` with status filtering, approval/rejection dialogs, proposed vs modified action diffs, audit timelines, and direct review package submission from the Investigation Drawer.
- **API Endpoints**: `POST /api/decisions`, `GET /api/decisions`, `GET /api/decisions/{id}`, `POST /api/decisions/{id}/approve`, `POST /api/decisions/{id}/reject`, `POST /api/decisions/{id}/request-changes`, and `POST /api/decisions/{id}/resubmit`.

## Proactive Intelligence & Monitoring (Phase 6.8)

- **Purpose**: Proactively evaluates sales data to identify newly significant commercial shifts, spikes, drops, category movements, and repeated patterns, surfacing prioritized alerts for human attention.
- **Strict Non-Autonomous Operational Guarantee**: Monitoring produces ALERTS only. Structurally never executes automated price changes, inventory adjustments, purchase orders, promotional actions, or recommendation approvals. Every alert remains subject to the existing Human Approval governance layer.
- **Materiality & Priority Scoring**: Leverages empirical robust z-score statistical significance to assign operational priorities (`urgent`, `high`, `medium`, `low`). Low-severity anomalies are suppressed unless participating in repeated anomaly patterns.
- **Repeated Anomaly Pattern Detection**: Flags frequency clusters across identical metrics and scopes as `repeated_anomaly`, highlighting persistent commercial behavior without asserting ungrounded causal claims.
- **Deterministic Alert Fingerprinting & Duplicate Suppression**: Prevents alert fatigue by computing unique fingerprint hashes (`{date}_{metric}_{entity_type}_{entity_id}_{alert_type}`). Automatically suppresses duplicate unresolved alerts and reports suppression metrics.
- **Evidence Snapshotting**: Captures point-in-time anomaly investigation metrics, driver attributions, and revenue impact at scan time without mutating underlying data.
- **Alert Lifecycle Governance**: Enforces human-controlled states: `new` → `acknowledged` → `resolved` (or `dismissed`). Invalid transitions return HTTP 409 Conflict.
- **Deep Workflow Integration**: Every alert links seamlessly to Root-Cause Investigation (`InvestigationDrawer`), What-If Simulation (`/simulation`), and Prescriptive Decision Review (`/decisions`).
- **Notification Boundary**: Phase 6.8 establishes the in-app intelligence alert center; external delivery channels (Email/SMS/WhatsApp/Webhooks) are designed as future extension points with zero side effects.
- **API Endpoints**: `POST /api/monitoring/run`, `GET /api/monitoring/alerts`, `GET /api/monitoring/alerts/{id}`, `POST /api/monitoring/alerts/{id}/acknowledge`, `POST /api/monitoring/alerts/{id}/resolve`, `POST /api/monitoring/alerts/{id}/dismiss`, `GET /api/monitoring/summary`, and `GET /api/monitoring/runs`.

## Full-Stack Application

- User registration
- Secure login
- JWT authentication
- Password hashing using Argon2
- Protected application routes
- Product CRUD
- Sales CRUD
- Forecast generation
- Dashboard API
- Historical data API
- API validation
- Error handling
- Responsive dashboard

---

# System Architecture

```text
                         ┌─────────────────────┐
                         │     User / Browser  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   React Frontend    │
                         │   Vite + Tailwind   │
                         │                     │
                         │ Dashboard           │
                         │ Products            │
                         │ Sales               │
                         │ Forecast            │
                         │ Anomaly Insights    │
                         │ Investigation Drawer│
                         │ Executive Brief     │
                         │ AI Reasoning Tab    │
                         │ Recommendations Tab │
                         │ Simulation Panel    │
                         │ Decision Center     │
                         │ Intelligence Monitor│
                         └──────────┬──────────┘
                                    │
                               REST / JSON
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   FastAPI Backend   │
                         │                     │
                         │ Authentication      │
                         │ Products API        │
                         │ Sales API           │
                         │ Forecast API        │
                         │ Anomalies API       │
                         │ Investigations API  │
                         │ Explanations API    │
                         │ AI Reasoning API    │
                         │ Recommendations API │
                         │ Simulations API     │
                         │ Decisions API       │
                         │ Monitoring API      │
                         └───────┬─────┬───────┘
                                 │     │
                    ┌────────────┘     └────────────┐
                    ▼                               ▼
          ┌─────────────────┐             ┌──────────────────┐
          │    Database     │             │ ML & Intelligence│
          │                 │             │                  │
          │ Users           │             │ Forecast Models  │
          │ Products        │             │ Anomaly Engine   │
          │ Sales Records   │             │ Root-Cause Engine│
          │ Decision Records│             │ Narrative Engine │
          │ Audit Events    │             │ Gemini Reasoning │
          │ MonitoringAlerts│             │ Prescriptive Recs│
          │ Monitoring Runs │             │ What-If Simulator│
          │                 │             │ Governance Engine│
          │                 │             │ Proactive Monitor│
          └─────────────────┘             └──────────────────┘
```

---

# Machine Learning Pipeline

The ML workflow was developed in multiple phases.

```text
Raw Sales Data
      │
      ▼
Data Validation
      │
      ▼
Data Preprocessing
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ├── Calendar Features
      ├── Lag Features
      ├── Rolling Statistics
      ├── Promotion
      └── Holiday Indicators
      │
      ▼
Baseline Forecasting
      │
      ▼
Feature-Based ML Models
      │
      ├── Linear Regression
      ├── Random Forest
      └── HistGradientBoosting
      │
      ▼
LSTM Experiment
      │
      ▼
Rolling-Origin Evaluation
      │
      ▼
Model Comparison
      │
      ▼
Production Model Selection
      │
      ▼
Forecast API
```

---

# Dataset

The project uses a synthetic enhanced forecasting dataset built from the original sales/catalog data.

The forecasting dataset contains:

- **1,168,800 product-day records**
- **400 products**
- **8 complete years**
- Date range: **2018-01-01 to 2025-12-31**

Important forecasting fields include:

```text
Date
Product_ID
Category_ID
Quantity
Unit_Price
Discount_Percent
Promotion
Is_Holiday
Sales_Amount
Profit
Current_Inventory
Supplier_Lead_Time_Days
Reorder_Point
Day_of_Week
Month
```

The primary forecasting target is:

```text
Quantity
```

The secondary business metric is:

```text
Sales_Amount
```

---

# Feature Engineering

The production forecasting pipeline uses time-series features including:

### Calendar Features

- Day of week
- Day of month
- Week of year
- Month
- Quarter
- Weekend indicator
- Days since start

### Lag Features

```text
quantity_lag_1
quantity_lag_7
quantity_lag_14
quantity_lag_28
```

### Rolling Features

```text
quantity_rolling_mean_7
quantity_rolling_std_7

quantity_rolling_mean_14
quantity_rolling_std_14

quantity_rolling_mean_28
quantity_rolling_std_28
```

### Business/Event Features

```text
Promotions
Holiday_Flag
```

---

# Model Development

## Baseline Models

The project first established simple forecasting benchmarks using:

- Naive forecasting
- Seasonal Naive forecasting

The seasonal benchmark provided a strong baseline for comparison.

---

# Feature-Based Machine Learning

Three classical machine learning models were evaluated:

### Linear Regression

Used as a lightweight interpretable forecasting model.

### Random Forest

Used to capture nonlinear relationships between lag, rolling, calendar, and event features.

### HistGradientBoosting

Used for efficient gradient-boosted nonlinear forecasting.

The models were evaluated using chronological splits rather than random shuffling.

---

# Time-Based Validation Strategy

Because sales forecasting is a time-dependent problem, random train/test splitting was avoided.

The project uses:

```text
Training:
2018 – 2023

Validation:
2024

Testing:
2025
```

This prevents future information from leaking into the training process.

Production evaluation also uses a **rolling-origin forecasting methodology**.

---

# Production Models

After model comparison, the following models were selected for production inference:

| Horizon | Model |
|---|---|
| 7 days | Random Forest |
| 30 days | HistGradientBoosting |
| 90 days | Linear Regression |

Production evaluation results:

| Horizon | Model | WAPE |
|---|---|---:|
| 7 Days | Random Forest | ~2.83% |
| 30 Days | Gradient Boosting | ~2.93% |
| 90 Days | Linear Regression | ~4.41% |

> Metrics represent the production evaluation performed using the project's rolling-origin methodology.

---

# LSTM Experiment

An LSTM model was also developed using TensorFlow to evaluate whether a neural-network approach could outperform the classical models.

LSTM results:

| Horizon | LSTM WAPE |
|---|---:|
| 7 Days | ~1.35% |
| 30 Days | ~2.81% |
| 90 Days | ~3.69% |

The LSTM experiment was retained as an experimental model rather than replacing the production model selection layer.

This demonstrates an important ML engineering principle:

> The most complex model is not automatically the best production model.

---

# Backend

The backend is implemented using:

- FastAPI
- SQLAlchemy
- SQLite
- Pydantic
- JWT
- PyJWT
- Argon2
- Uvicorn

## Backend Structure

```text
backend/
└── app/
    ├── __init__.py
    ├── main.py
    ├── dependencies.py
    │
    ├── core/
    │   ├── config.py
    │   ├── security.py
    │   ├── logging_config.py
    │   └── exceptions.py
    │
    ├── api/
    │   ├── auth.py
    │   ├── forecast.py
    │   ├── products.py
    │   ├── sales.py
    │   ├── anomalies.py
    │   ├── investigations.py
    │   ├── explanations.py
    │   ├── ai_reasoning.py
    │   ├── recommendations.py
    │   ├── simulations.py
    │   ├── decisions.py
    │   └── monitoring.py
    │
    ├── schemas/
    │   ├── auth.py
    │   ├── forecast.py
    │   ├── products.py
    │   ├── sales.py
    │   ├── anomalies.py
    │   ├── investigations.py
    │   ├── explanations.py
    │   ├── ai_reasoning.py
    │   ├── recommendations.py
    │   ├── simulations.py
    │   ├── decisions.py
    │   └── monitoring.py
    │
    ├── services/
    │   ├── forecast_service.py
    │   ├── history_service.py
    │   ├── anomaly_service.py
    │   ├── investigation_service.py
    │   ├── explanation_service.py
    │   ├── ai_reasoning_service.py
    │   ├── recommendation_service.py
    │   ├── simulation_service.py
    │   ├── decision_service.py
    │   └── monitoring_service.py
    │
    ├── llm/
    │   ├── provider.py
    │   ├── gemini_provider.py
    │   └── prompts.py
    │
    └── database/
        ├── database.py
        ├── models.py
        ├── crud.py
        ├── user_crud.py
        └── sales_crud.py
```

---

# REST API

Base URL during local development:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Authentication

### Register

```http
POST /api/auth/register
```

### Login

```http
POST /api/auth/login
```

### Current User

```http
GET /api/auth/me
```

Authentication uses:

```text
JWT Bearer Token
```

Passwords are securely hashed using Argon2.

---

# Products API

```http
POST   /api/products
GET    /api/products
GET    /api/products/{product_id}
PUT    /api/products/{product_id}
DELETE /api/products/{product_id}
```

The Products API uses an external string identifier such as:

```text
PROD-001
```

---

# Sales API

```http
POST   /api/sales
GET    /api/sales
GET    /api/sales/{sales_id}
PUT    /api/sales/{sales_id}
DELETE /api/sales/{sales_id}
```

The Sales API uses the integer database ID of the product.

The frontend handles the distinction between:

```text
Product external ID → "PROD-001"
```

and:

```text
Product database ID → 1
```

---

# Forecast API

### Generate Forecast

```http
POST /api/forecast
```

Request:

```json
{
  "horizon": 7
}
```

Supported values:

```text
7
30
90
```

Response:

```json
{
  "horizon": 7,
  "model": "Random Forest",
  "forecast": [
    {
      "date": "2026-01-01",
      "predicted_quantity": 5000
    }
  ]
}
```

---

# Forecast History

```http
GET /api/forecast/history
```

Optional:

```text
/api/forecast/history?limit=365
```

Returns historical:

- Quantity
- Sales Amount
- Profit
- Date

---

# Dashboard API

```http
GET /api/forecast/dashboard
```

Optional parameters:

```text
horizon
history_limit
```

Example:

```text
/api/forecast/dashboard?horizon=30&history_limit=365
```

The dashboard endpoint returns:

- Historical KPIs
- Historical records
- Forecast
- Forecast horizon
- Forecast model

---

# Frontend

The frontend is built using:

- React 19
- Vite
- Tailwind CSS v4
- React Router
- Axios
- Recharts
- Lucide React

## Frontend Structure

```text
frontend/
├── index.html
├── vite.config.js
├── package.json
├── .env
│
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── index.css
    │
    ├── api/
    │   ├── axios.js
    │   ├── auth.js
    │   ├── products.js
    │   ├── sales.js
    │   ├── forecast.js
    │   ├── anomalies.js
    │   ├── investigations.js
    │   ├── explanations.js
    │   ├── aiReasoning.js
    │   ├── recommendations.js
    │   ├── simulations.js
    │   ├── decisions.js
    │   └── monitoring.js
    │
    ├── context/
    │   └── AuthContext.jsx
    │
    ├── layouts/
    │   └── DashboardLayout.jsx
    │
    ├── components/
    │   ├── Navbar.jsx
    │   ├── Sidebar.jsx
    │   ├── ProtectedRoute.jsx
    │   ├── KpiCard.jsx
    │   ├── LoadingSpinner.jsx
    │   ├── ErrorMessage.jsx
    │   ├── EmptyState.jsx
    │   ├── SalesChart.jsx
    │   ├── ForecastChart.jsx
    │   ├── Modal.jsx
    │   ├── ConfirmDialog.jsx
    │   ├── InvestigationDrawer.jsx
    │   ├── SimulationPanel.jsx
    │   ├── DecisionReviewPanel.jsx
    │   ├── AlertCard.jsx
    │   └── MonitoringSummary.jsx
    │
    ├── pages/
    │   ├── Login.jsx
    │   ├── Register.jsx
    │   ├── Dashboard.jsx
    │   ├── Products.jsx
    │   ├── Sales.jsx
    │   ├── Forecast.jsx
    │   ├── Anomalies.jsx
    │   ├── Simulation.jsx
    │   ├── DecisionCenter.jsx
    │   ├── IntelligenceMonitor.jsx
    │   └── NotFound.jsx
    │
    └── utils/
        ├── formatters.js
        └── constants.js
```

---

# Dashboard

The executive dashboard provides:

### KPI Cards

- Total Historical Quantity
- Total Sales Revenue
- Total Net Profit
- Average Daily Demand

### Visualizations

- Revenue chart
- Profit chart
- Historical demand chart
- Forecast chart

### Forecast Selection

```text
7 Days
30 Days
90 Days
```

The selected forecast is retrieved from the real backend ML API.

---

# Products Management

The Products page provides:

- Product search
- Product listing
- Product creation
- Product editing
- Product deletion
- Validation
- Confirmation dialogs

---

# Sales Management

The Sales page provides:

- Sales listing
- Product selection
- Sale creation
- Sale editing
- Sale deletion
- Promotion indicator
- Holiday indicator
- Discount handling
- Sales amount preview
- Profit preview

---

# Forecast Dashboard

The dedicated Forecast page provides:

- Forecast horizon selection
- Production model name
- Total predicted demand
- Average predicted demand
- Peak demand day
- Lowest demand day
- Forecast demand shift
- Forecast chart
- Daily prediction table

---

# Authentication Architecture

```text
User
 │
 ▼
Login
 │
 ▼
FastAPI Authentication
 │
 ▼
JWT Access Token
 │
 ▼
React AuthContext
 │
 ▼
localStorage
 │
 ▼
Axios Interceptor
 │
 ▼
Authorization: Bearer <token>
 │
 ▼
Protected API
```

Unauthenticated users are redirected to:

```text
/login
```

Protected pages include:

```text
/dashboard
/products
/sales
/forecast
```

---

# Project Structure

The complete repository is organized as:

```text
smart-sales-forecasting-system/
│
├── backend/
│
├── data/
│
├── ml/
│
├── models/
│
├── notebooks/
│
├── results/
│
├── tests/
│
├── frontend/
│
├── docs/
│
└── README.md
```

---

# ML Production Artifacts

Production model artifacts are maintained under:

```text
ml/artifacts/
```

The production forecasting layer includes horizon-specific artifacts for:

```text
7-day demand
30-day demand
90-day demand
```

Model metadata and production evaluation outputs are also maintained with the ML artifacts.

---

# Installation

## Prerequisites

Install:

- Python 3.12+
- Node.js 18+
- npm
- Git

Recommended:

- VS Code / Antigravity
- Virtual environment

---

# Backend Setup

From the project root:

```powershell
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
pip install -r ml/requirements.txt
```

Install backend dependencies if maintained separately:

```powershell
pip install fastapi uvicorn sqlalchemy pydantic-settings pwdlib[argon2] PyJWT python-multipart
```

---

# Environment Variables

Create:

```text
backend/.env
```

or use the project's configured environment location.

Example:

```env
SECRET_KEY=replace-with-a-secure-secret
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

Never commit real secrets.

---

# Start Backend

From the project root:

```powershell
uvicorn backend.app.main:app --reload --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

---

# Frontend Setup

Navigate to:

```powershell
cd frontend
```

Install dependencies:

```powershell
npm install
```

Create:

```text
frontend/.env
```

with:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Start development server:

```powershell
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

# Running the Complete Application

### Terminal 1 — Backend

```powershell
.\.venv\Scripts\Activate.ps1

uvicorn backend.app.main:app --reload --port 8000
```

### Terminal 2 — Frontend

```powershell
cd frontend

npm run dev
```

Open:

```text
http://localhost:5173
```

---

# Testing

The backend contains automated API and statistical tests covering:

- Authentication
- Registration
- Login
- Protected routes
- Products
- Sales
- User isolation
- Forecast generation
- Historical data
- Dashboard
- Sales Anomaly Detection (Median + MAD robust baseline)
- Sales Anomaly Root-Cause Investigation (`/api/investigations/{id}`)
- Sales Anomaly Executive Narrative Explanation (`/api/explanations/{id}`)
- Deterministic template-driven narrative generation (Headline, What Happened, Why It Matters, Key Contributors)
- Event, calendar, and pre-anomaly drift context reporting
- Multi-dimensional driver decomposition (Promotions, Holidays, Categories, Products, Prices, Discounts, Drift)
- Financial & demand impact quantification with price basis documentation
- Causal time-series invariance (zero future data leakage)
- Deterministic reproducibility (byte-for-byte consistency across multiple invocations)
- Synthetic spike and drop detection
- Severity, metric, date, and direction filtering
- Category and product-level anomaly endpoints
- Sales Anomaly AI Reasoning (`/api/ai-reasoning/{id}`)
- Google Gemini structured response validation & Pydantic schema conformance
- Deterministic evidence entailment validation (dimension, driver matching, confidence capping)
- Deterministic numerical grounding validation (actual values, baselines, deviations, driver figures)
- Unsupported product-tier & luxury claim rejection (premium lines, luxury goods, higher tiers)
- Validation questions grammatical question enforcement & product-mix hypothesis conversion
- Unsupported holiday and promotion claim rejection
- External speculation sanitization (competitors, marketing, inventory, operations)
- Causal language sanitization & mandatory disclaimer enforcement
- Adversarial prompt injection defense
- Upstream provider error & unconfigured key graceful degradation (HTTP 503)
- In-memory reasoning cache & cache invalidation (`?refresh=true`)
- Prescriptive Action Recommendations (`/api/recommendations/{id}`)
- Deterministic eligibility gating across 8 recommendation categories
- Non-autonomous governance & human approval required invariant
- Recommendation-specific supporting evidence grounding
- Promotion review rejection when promotion is inactive
- Deterministic fallback engine for high availability
- What-If Scenario Simulation (`/api/simulations`)
- 6 counterfactual scenarios (multiplier, shock, persistent shift, trend, promotion, holiday)
- Elasticity boundaries returning `status = "requires_model"`
- 10 simulation numerical invariants & zero-data-mutation validation
- Human Approval & Decision Governance (`/api/decisions`)
- Decision lifecycle state machine (`pending_review`, `approved`, `rejected`, `changes_requested`, `resubmit`)
- Immutable original recommendations with distinct `modified_action` preservation
- Mandatory reviewer rationales for rejections and revisions
- Immutable append-only audit trail with actor and event metadata tracking
- Cross-layer point-in-time snapshots (recommendation, anomaly evidence, simulation)
- Conflict prevention (HTTP 409) on terminal decisions and tenant isolation (HTTP 403)
- Non-autonomous invariants (`human_approval_required: true`, `automatic_execution: false`)
- Proactive Intelligence & Monitoring (`/api/monitoring`)
- Automated detection of sales spikes, drops, category deviations, and demand shifts
- Materiality threshold gating based on robust z-scores and deviation percentage
- Frequency analysis & repeated anomaly pattern detection
- Deterministic fingerprint computation (`{date}_{metric}_{entity_type}_{entity_id}_{alert_type}`)
- Automatic duplicate alert suppression for unresolved events
- Alert lifecycle state transitions (`new` → `acknowledged` → `resolved`, `dismissed`)
- State machine conflict rejection (HTTP 409) for finalized alerts
- Investigation, simulation, and decision review cross-layer linkage
- Zero mutation of historical sales records, products, or decision records
- Non-autonomous operational invariants (`human_review_required: true`, `automatic_execution: false`)
- Validation
- Error handling

Run:

```powershell
pytest -q
```

Current result:

```text
258 passed
```

The remaining warnings are dependency-level FastAPI/Starlette/AnyIO warnings and do not represent application test failures.

---

# Frontend Build Verification

From:

```text
frontend/
```

run:

```powershell
npm run build
```

The production build currently completes successfully.

---

# Verification Checklist

## Backend

- [x] FastAPI application starts
- [x] Authentication works
- [x] JWT authentication works
- [x] Products CRUD works
- [x] Sales CRUD works
- [x] User isolation works
- [x] Forecast API works
- [x] 7-day forecast works
- [x] 30-day forecast works
- [x] 90-day forecast works
- [x] Dashboard API works
- [x] Sales Anomaly Detection API (`/api/anomalies`)
- [x] Sales Anomaly Investigation API (`/api/investigations/{id}`)
- [x] Executive Narrative Explanation API (`/api/explanations/{id}`)
- [x] Sales Anomaly AI Reasoning API (`/api/ai-reasoning/{id}`)
- [x] Gemini SDK (`google-genai`) structured outputs integration
- [x] Grounded reasoning strictly from Phase 6.2/6.3 evidence packages (< 2 KB)
- [x] Deterministic evidence entailment & grounding validation layer
- [x] Deterministic numerical grounding consistency post-validation
- [x] Product-tier hypothesis elimination & grammatical validation question enforcement
- [x] False holiday & promotion claim rejection
- [x] External factor assertion filtering (competitors, inventory, marketing)
- [x] Causal guardrails & deterministic post-validation
- [x] In-memory reasoning cache with on-demand refresh bypass
- [x] Graceful degradation on missing API key / upstream failure (HTTP 503)
- [x] Multi-driver root-cause decomposition without lookahead leakage
- [x] Category and Product anomaly queries & contributions
- [x] Quantified business impact estimation
- [x] Deterministic narrative generation without LLM
- [x] Key driver prioritization & top-N capping
- [x] Prescriptive Action Recommendations API (`/api/recommendations/{id}`)
- [x] Deterministic recommendation eligibility gating & causal guardrails
- [x] Non-autonomous governance & human approval required invariant
- [x] What-If Scenario Simulation API (`/api/simulations`)
- [x] 6 counterfactual simulation scenarios & elasticity boundary handling
- [x] 10 simulation numerical invariants & zero-data-mutation validation
- [x] Human Approval & Decision Governance API (`/api/decisions`)
- [x] Decision lifecycle state machine & transition conflict prevention (HTTP 409)
- [x] Original recommendation immutability & separate modified action storage
- [x] Mandatory reviewer rationales for rejections and change requests
- [x] Immutable chronological audit trail events
- [x] Point-in-time recommendation, anomaly, and simulation snapshot integrity
- [x] Tenant isolation (HTTP 403) and authentication verification
- [x] Proactive Intelligence & Monitoring API (`/api/monitoring`)
- [x] Deterministic materiality scoring & priority classification
- [x] Frequency-based repeated anomaly pattern detection
- [x] Deterministic alert fingerprinting & duplicate suppression
- [x] Alert lifecycle state machine (`new`, `acknowledged`, `resolved`, `dismissed`)
- [x] State transition conflict rejection (HTTP 409)
- [x] Monitoring run observability logging (`MonitoringRun`)
- [x] Cross-layer linking (Investigation, Simulation, Decision Governance)
- [x] Zero mutation of sales records, products, or decisions
- [x] Validation works
- [x] Error handling works
- [x] 258/258 tests passing

## Frontend

- [x] React application
- [x] Vite build
- [x] Tailwind CSS
- [x] Authentication UI
- [x] Protected routes
- [x] Dashboard
- [x] KPI cards
- [x] Recharts visualizations
- [x] Products CRUD
- [x] Sales CRUD
- [x] Forecast page
- [x] Anomaly Insights page (`/anomalies`)
- [x] Investigation Drawer with driver ranking & evidence
- [x] Executive Brief drawer tab & 1-click Markdown clipboard export
- [x] AI Reasoning drawer tab with insight cards, confidence pills, & risk flags
- [x] Recommendations drawer tab with action cards, priority badges, & human review notices
- [x] Interactive Simulation Panel & What-If scenario explorer (`/simulations`)
- [x] Decision Center page (`/decisions`, `/decisions/:id`) with status filters
- [x] Decision Review Panel with action dialogs (Approve / Reject / Request Changes)
- [x] Audit timeline and proposed vs modified action diff view
- [x] Intelligence Monitor page (`/intelligence`) with status, severity, and type filters
- [x] Proactive monitoring summary KPI telemetry (`MonitoringSummary`)
- [x] Alert cards with priority/severity badges and state transitions (`AlertCard`)
- [x] Interactive "Run Monitoring Scan" with real-time feedback banner
- [x] Alert detail drawer with connected workflows & governance notice
- [x] On-demand "Refresh Analysis" trigger
- [x] Graceful error state banner for HTTP 503 / missing API key
- [x] Responsive layout
- [x] Loading states
- [x] Error states
- [x] Production build

---

# Security Considerations

The project implements several security practices:

- Password hashing using Argon2
- JWT-based authentication
- Protected API routes
- Protected frontend routes
- Authorization headers
- Input validation using Pydantic
- User-level sales record isolation
- Environment-based secret configuration
- No production secrets committed to source control

For production deployment, additional hardening should include:

- HTTPS
- Secure cookie/token strategy where appropriate
- PostgreSQL instead of local SQLite
- Production CORS configuration
- Secret management through deployment environment variables
- Rate limiting
- Database backups
- Monitoring and centralized logging

---

# Deployment

The application is designed to support deployment as separate frontend and backend services.

Recommended architecture:

```text
React + Vite
      │
      ▼
   Vercel
      │
      │ HTTPS
      ▼
 FastAPI
      │
      ▼
  Render
      │
      ├───────────────┐
      ▼               ▼
PostgreSQL       ML Artifacts
```

For local development:

```text
Frontend → localhost:5173
Backend  → 127.0.0.1:8000
```

For production, configure:

```env
VITE_API_BASE_URL=<production-api-url>
```

and configure the backend CORS policy to allow only the deployed frontend domain.

---

# Future Improvements

Potential future improvements include:

### Forecasting

- Product-level forecasting at scale
- Category-level forecasting
- Hierarchical forecasting
- Probabilistic prediction intervals
- Automated model retraining
- Model drift monitoring
- Feature importance explanations
- Advanced ensemble models

### Inventory Intelligence

- Stockout prediction
- Reorder recommendations
- Safety stock optimization
- Supplier lead-time optimization
- Inventory risk alerts

### Business Intelligence

- Sales anomaly detection
- Customer segmentation
- Product profitability analysis
- Promotion effectiveness analysis
- Demand-driver analysis

### Infrastructure

- PostgreSQL production database
- Docker containerization
- CI/CD pipeline
- Automated model retraining
- Cloud object storage for model artifacts
- Monitoring and observability

---

# Lessons Learned

This project demonstrates several practical machine learning engineering principles:

### 1. Start with a baseline

A forecasting model should be compared against simple baselines before introducing complex algorithms.

### 2. Respect temporal ordering

Time-series problems require chronological validation to prevent future information leakage.

### 3. Model complexity is not the objective

An LSTM can be useful experimentally without necessarily being the best production choice.

### 4. Evaluate multiple horizons

Short-term and long-term forecasting can have different optimal models.

### 5. Separate ML from application logic

The forecasting engine is separated from the FastAPI API layer, allowing the application to consume trained models through a clean service interface.

### 6. Test the complete system

The backend contains automated tests covering authentication, CRUD operations, forecasting, validation, and error handling.

---

# API Documentation

When the backend is running, interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Alternative OpenAPI documentation:

```text
http://127.0.0.1:8000/redoc
```

---

# Financial Intelligence Pipeline (Phase 6)

The Smart Sales Forecasting System features a comprehensive, non-autonomous executive intelligence pipeline:

```text
DETECT → INVESTIGATE → EXPLAIN → AI REASONING → RECOMMEND → SIMULATE → HUMAN APPROVAL → AUDIT TRAIL → MONITOR → ALERT
```

### Phase 6.1 — Sales Anomaly Detection
- Detection across Aggregate, Category, and Product grains.
- Dual-metric tracking: Quantity and Sales Amount.
- Bi-directional detection: Demand Spikes and Drops.

### Phase 6.2 — Root-Cause Attribution
- Multi-dimensional decomposition: Promotions, Holidays, Categories, Products, Pricing, Discounts, and Drift.
- Zero-lookahead causal historical baselines.
- Quantified business impact analysis.

### Phase 6.3 — Executive Narrative Reporting
- Deterministic, audit-ready natural language executive briefings.
- Indian Rupees (₹) monetary formatting.

### Phase 6.4 — Grounded Gemini AI Reasoning Layer
- Gemini 2.5 Flash / Flash Lite executive reasoning over empirical evidence packages.
- Strict causal boundaries: zero hallucinated causal factors or unsupported product tiers.
- Guaranteed numerical consistency with underlying ML evidence.

### Phase 6.5 — Prescriptive Action Recommendations
- Non-autonomous prescriptive recommendations for human evaluation.
- Deterministic eligibility gating across 8 recommendation categories.
- Gemini strategic rationale, trade-off evaluation, and risk considerations.

### Phase 6.6 — What-If Scenario Simulation
- Non-destructive hypothetical scenario exploration over 7, 30, and 90-day horizons without modifying historical sales records.
- **Supported Scenarios**:
  - `demand_multiplier`: Uniform percentage shift (-50% to +50%).
  - `temporary_shock`: Short-term demand disturbance (1 to horizon days) with automatic reversion to baseline.
  - `persistent_shift`: Sustained structural demand level shift.
  - `trend_continuation`: Extrapolates recent historical linear velocity (7 to 90-day window).
  - `promotion_scenario`: Evaluates promotional lift via production ML model.
  - `holiday_scenario`: Evaluates calendar holiday lift via production ML model.
- **Model Boundaries**:
  - `price_change` & `discount_change` return `status = "requires_model"` with explicit explanations; elasticity numbers are never fabricated.
- **Baseline Realized Price**: Revenue calculated under constant baseline realized unit price assumption (₹3,400.87/unit).
- **10 Numerical Invariants**: Daily sums match summary totals, delta math is consistent, non-negative quantities, finite numbers, and sequential continuous dates.
- **API Endpoints**:
  - `POST /api/simulations`: Authenticated scenario execution with JWT Bearer token.
  - `GET /api/simulations/{id}`: Cached simulation retrieval.

### Phase 6.7 — Human Approval & Decision Governance
- Enterprise human decision governance over prescriptive recommendations and simulation outcomes.
- **Strict Non-Autonomous Execution**: `human_approval_required: true` and `automatic_execution: false` are structurally guaranteed across all records. Human approval authorizes human execution; the system never triggers purchase orders, price updates, or inventory reallocations autonomously.
- **State Machine Lifecycle**: Formal transition paths (`pending_review` → `approved` / `rejected` / `changes_requested`, `changes_requested` → `pending_review`), with terminal immutability for approved/rejected decisions and HTTP 409 conflict protection.
- **Original Recommendation Immutability**: Frozen original recommendation payload with distinct reviewer adjustments isolated in `modified_action`.
- **Mandatory Decision Notes**: Explicit rationale enforcement (1–2,000 characters) on rejections and changes requests.
- **Point-in-Time Evidence Snapshotting**: Deep JSON preservation of recommendation payload, anomaly root-cause attribution, and what-if simulation run upon creation.
- **Chronological Audit Trail**: Append-only event history capturing actors, timestamps, state transitions, and reviewer rationales.
- **Dedicated React Decision Center**: Interactive UI at `/decisions` and `/decisions/:id` with status filters, action dialogs (Approve / Reject / Request Changes), audit history timeline, diff view, and 1-click submission from the Investigation Drawer.
- **API Endpoints**:
  - `POST /api/decisions`: Package recommendation into an auditable decision record.
  - `GET /api/decisions`: Filterable and paginated decision listing.
  - `GET /api/decisions/{id}`: Full decision record with audit events and snapshots.
  - `POST /api/decisions/{id}/approve`: Approve decision with optional note.
  - `POST /api/decisions/{id}/reject`: Reject decision with mandatory rationale.
  - `POST /api/decisions/{id}/request-changes`: Request changes with mandatory rationale.
  - `POST /api/decisions/{id}/resubmit`: Resubmit modified decision package.

### Phase 6.8 — Proactive Intelligence & Monitoring
- In-app proactive intelligence engine periodically evaluating sales data to identify newly significant commercial shifts, demand changes, and repeated patterns.
- **Strict Non-Autonomous Operational Guarantee**: Monitoring produces ALERTS only. The system structurally never executes automated price changes, inventory adjustments, purchase orders, promotional actions, or recommendation approvals. Every alert remains subject to the existing Human Approval governance layer.
- **Materiality & Priority Scoring**: Classifies operational priority (`urgent`, `high`, `medium`, `low`) deterministically based on robust z-score statistical significance and relative percentage deviation ($\ge 15\%$). Low-severity deviations are filtered out unless repeated.
- **Repeated Anomaly Pattern Detection**: Analyzes frequency clusters across identical metrics and scopes to flag chronic or recurring operational issues (`alert_type = "repeated_anomaly"`).
- **Deterministic Alert Fingerprinting & Duplicate Suppression**: Computes unique fingerprint hashes (`{date}_{metric}_{entity_type}_{entity_id}_{alert_type}`) to suppress duplicate unresolved alerts and prevent alert fatigue.
- **Evidence Snapshotting**: Captures point-in-time anomaly investigation metrics, driver attributions, and revenue impact at scan time without mutating underlying data.
- **Alert Lifecycle Governance**: State machine tracking `new` → `acknowledged` → `resolved` (or `dismissed`), with HTTP 409 conflict protection against invalid transitions.
- **Deep Workflow Integration**: Every alert links seamlessly to Root-Cause Investigation (`InvestigationDrawer`), What-If Simulation (`/simulation`), and Prescriptive Decision Review (`/decisions`).
- **Notification Boundary**: Phase 6.8 establishes the in-app intelligence alert center; external delivery channels (Email/SMS/WhatsApp/Webhooks) are designed as future extension points with zero side effects.
- **API Endpoints**:
  - `POST /api/monitoring/run`: Trigger proactive monitoring scan.
  - `GET /api/monitoring/alerts`: Filterable, paginated alert list.
  - `GET /api/monitoring/alerts/{id}`: Detailed alert context with workflow linkages.
  - `POST /api/monitoring/alerts/{id}/acknowledge`: Mark alert acknowledged.
  - `POST /api/monitoring/alerts/{id}/resolve`: Mark alert resolved.
  - `POST /api/monitoring/alerts/{id}/dismiss`: Mark alert dismissed.
  - `GET /api/monitoring/summary`: Executive monitoring KPI telemetry.
  - `GET /api/monitoring/runs`: Observability history of monitoring runs.

### Phase 6.9 — Intelligence Platform Integration & Stabilization
- **Unified Executive Intelligence Workflow**: Complete seamless integration from anomaly detection to governance and proactive monitoring:
  $$\text{DETECT} \rightarrow \text{INVESTIGATE} \rightarrow \text{EXPLAIN} \rightarrow \text{AI REASONING} \rightarrow \text{RECOMMEND} \rightarrow \text{SIMULATE} \rightarrow \text{HUMAN APPROVAL} \rightarrow \text{AUDIT TRAIL} \rightarrow \text{MONITOR} \rightarrow \text{ALERT}$$
- **Executive Dashboard Modernization**:
  - Live enterprise telemetry widgets integrating real-time intelligence feeds (`GET /api/monitoring/summary`, `GET /api/decisions?status=pending_review`).
  - Active commercial alerts count, critical attention indicators, pending governance reviews badge, anomaly tally, and what-if simulation launcher.
  - Complete standardization to Indian Rupee (`₹`) monetary formatting and icons across all summary cards.
  - 1-click quick-action navigation to Intelligence Monitor, Decision Center, What-If Simulator, and Anomaly Investigations.
- **Route-Level Code Splitting & Performance Optimization**:
  - Implemented dynamic route splitting using `React.lazy()` and `<Suspense>` across all major analytical pages (`Dashboard`, `IntelligenceMonitor`, `DecisionCenter`, `Simulation`, `Anomalies`, `Forecast`, `Products`, `Sales`).
  - Core vendor chunk bundle optimized under 350 kB (318.8 kB), completely eliminating monolithic Vite bundle warnings.
- **Cross-Feature Navigational Continuity (Flows A–H)**:
  - Strict preservation of entity IDs (`anomaly_id`, `recommendation_id`, `simulation_id`, `decision_id`, `alert_id`) across state parameters and URL query parameters.
  - Context-preserving 1-click cross-feature transitions:
    - *Flow A*: Dashboard $\rightarrow$ Intelligence Monitor (`/intelligence`)
    - *Flow B*: Alert Card $\rightarrow$ Anomaly Investigation Drawer (`/anomalies`)
    - *Flow C*: Anomaly Investigation $\rightarrow$ Prescriptive Recommendations $\rightarrow$ Decision Center (`/decisions`)
    - *Flow D*: Anomaly Investigation $\rightarrow$ What-If Scenario Simulator (`/simulation?anomaly_id=...`)
    - *Flow E*: Simulation Results $\rightarrow$ Decision Center Submission (`/decisions?id=...`)
    - *Flow F*: Decision Review Panel $\rightarrow$ Root-Cause Investigation & Simulation Verification
    - *Flow G*: Decision Lifecycle (Approve / Reject / Request Changes / Resubmit)
    - *Flow H*: Alert Lifecycle (Acknowledge / Mark Resolved / Dismiss)
- **Executive Terminology & Non-Autonomous Invariant Enforcement**:
  - Empirical observational associations explicitly framed as "associated contributor" or "associated factor" rather than counterfactual causality.
  - What-If simulations framed as "scenarios" and "hypothetical projections" under explicit constant baseline price assumptions, distinct from ML forecasts.
  - Clear human agency affirmed with "human approved" decisions and "marked resolved" alerts.
  - `human_approval_required: true` and `automatic_execution: false` strictly enforced across all database models, API schemas, and UI action panels.
- **Gemini Operational Boundaries**:
  - Gemini LLM strictly bounded to strategic reasoning synthesis; zero numerical calculation, zero severity/priority scoring, and zero automated transaction execution.
- **Enterprise Error Resiliency**:
  - Standardized, sanitized error response contracts across 401, 403, 404, 409, 422, and 500 status codes with zero leakage of internal stack traces or server file paths.
- **Comprehensive End-to-End Integration Suite**:
  - 22 new end-to-end integration tests in `tests/test_intelligence_end_to_end.py` covering the complete 10-step lifecycle, state machines, immutability, data safety, and error contracts.

---

# Development Roadmap

```text
[x] Dataset preparation
[x] Exploratory Data Analysis
[x] Baseline forecasting
[x] Feature-based ML
[x] LSTM experimentation
[x] Production model selection
[x] ML inference service
[x] FastAPI backend
[x] Authentication
[x] Products CRUD
[x] Sales CRUD
[x] Forecast API
[x] Backend testing
[x] React frontend
[x] Dashboard
[x] Charts
[x] Products UI
[x] Sales UI
[x] Forecast UI
[x] Frontend production build
[x] Sales Anomaly Detection (Phase 6.1)
[x] Anomaly Investigation & Root-Cause Attribution (Phase 6.2)
[x] Executive Narrative Reporting (Phase 6.3)
[x] Grounded AI Reasoning Layer (Phase 6.4)
[x] Prescriptive Action Recommendations (Phase 6.5)
[x] What-If Scenario Simulation (Phase 6.6)
[x] Human Approval & Decision Governance (Phase 6.7)
[x] Proactive Intelligence & Monitoring (Phase 6.8)
[x] Intelligence Platform Integration & Stabilization (Phase 6.9)
[x] Production Readiness, Deployment & Portfolio Hardening (Phase 7)
[ ] Dockerization
[ ] Enterprise CI/CD Pipelines
```

---

# Production Deployment & Architecture

```mermaid
graph TD
    User["Client Browser (SPA)"] -->|HTTPS| CDN["Frontend Static Site (Render CDN / Vercel)"]
    CDN -->|Vite SPA Bundle| User
    User -->|REST APIs + JWT Auth| Backend["FastAPI Backend (Uvicorn ASGI)"]
    Backend -->|SQLAlchemy Connection Pool| DB[("Managed PostgreSQL / SQLite")]
    Backend -->|In-Process Inference| ML["Trained ML Models (Joblib Artifacts)"]
    Backend -.->|Grounded AI Insights| Gemini["Google Gemini Pro 2.5 API"]
```

### Environment Variables Reference

| Variable | Description | Default | Required in Prod? |
|---|---|---|---|
| `ENVIRONMENT` | Deployment environment mode (`development`, `test`, `production`) | `development` | Yes |
| `DATABASE_URL` | SQLAlchemy database connection string (PostgreSQL or SQLite) | `sqlite:///./sales_forecasting.db` | Yes (`postgresql://...`) |
| `SECRET_KEY` | Secret key for JWT cryptographic signing ($\ge 16$ characters) | Dev fallback key | Yes (must be set securely) |
| `ALGORITHM` | JWT signing algorithm | `HS256` | No |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT token validity lifespan | `1440` (24h) | No |
| `GEMINI_API_KEY` | Google Gemini API key for grounded narrative & reasoning | Empty | Optional (graceful fallback) |
| `GEMINI_MODEL` | Gemini LLM model identifier | `gemini-2.5-pro` | No |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (e.g., `https://my-app.onrender.com`) | `http://localhost:5173,...` | Yes in production |
| `LOG_LEVEL` | Application logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` | No |

### Production Secret Management (Platform Provisioning vs. Application Runtime)

The platform enforces a strict security policy regarding secret management:

- **Platform-Managed Secret Provisioning**: When deploying to cloud environments (e.g. Render, AWS, GCP), the platform provisions a cryptographically secure `SECRET_KEY` once at service provisioning time (configured in `render.yaml` via `generateValue: true`). This generated key is permanently stored in the hosting platform's encrypted vault and injected as a persistent `$SECRET_KEY` environment variable into the backend container across all restarts and redeployments.
- **Prohibition of Application-Generated Runtime Secrets**: The FastAPI application **never** generates a secret key at runtime in production. If `ENVIRONMENT=production` and `SECRET_KEY` is missing, empty, trivial, or less than 16 characters, `backend/app/core/config.py` raises a fatal `ValidationError` and terminates process initialization. This ensures that JWT signing keys are strictly persistent across container restarts and horizontally scaled replicas, preventing silent session invalidation and silent fallback vulnerabilities.

### Backend Deployment Steps

1. **Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. **Configure Production Environment**:
   ```bash
   export ENVIRONMENT=production
   export DATABASE_URL=postgresql://user:password@host:5432/dbname
   export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
   export ALLOWED_ORIGINS=https://your-frontend-domain.com
   export GEMINI_API_KEY=your-gemini-key
   ```
3. **Launch ASGI Server**:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 4
   ```

### Frontend Deployment Steps

1. **Configure API Base URL**:
   Set `VITE_API_BASE_URL=https://your-backend-domain.com` in environment or `.env.production`.
2. **Build Production Assets**:
   ```bash
   cd frontend
   npm install
   npm run build
   ```
3. **Deploy Static Files**:
   Publish `frontend/dist` directory to your static host (Render, Vercel, Netlify, Cloudflare Pages, S3/CloudFront) with SPA rewrite rule `/* -> /index.html`.

### Database Setup & Connection Pooling

- The backend automatically detects PostgreSQL URLs and applies production connection pooling:
  - `pool_size = 10`
  - `max_overflow = 20`
  - `pool_pre_ping = True` (detects and recycles stale connections)
  - `pool_recycle = 300` (recycles connections every 5 minutes)
- Auto-normalizes legacy `postgres://` URLs to SQLAlchemy-compliant `postgresql://`.
- Auto-initializes schema tables on startup via `Base.metadata.create_all(bind=engine)` with safety rollback session cleanup.

### CORS Configuration Guide

- In development mode, `localhost` and `127.0.0.1` ports `5173`, `3000`, `8000` are permitted.
- In production mode:
  - Exact origins must be explicitly specified in `ALLOWED_ORIGINS` (comma-delimited).
  - Wildcard `*` origins are strictly rejected when credentials are enabled.
  - Rejecting non-whitelisted origins prevents cross-site request forgery and data exfiltration.

### Health Check Verification Commands

```bash
# Liveness probe (returns HTTP 200 {"status": "ok"})
curl -i https://your-backend-domain.com/health

# Readiness probe (verifies database connectivity with SELECT 1; returns HTTP 200 {"status": "ready", "database": "ok"})
curl -i https://your-backend-domain.com/ready
```

### Local Production Simulation

```bash
# 1. Start backend in production mode locally
$env:ENVIRONMENT="production"
$env:SECRET_KEY="production-simulation-secret-key-32chars"
$env:DATABASE_URL="sqlite:///./sales_forecasting.db"
$env:ALLOWED_ORIGINS="http://localhost:4173"
uvicorn backend.app.main:app --port 8000

# 2. Build and preview frontend locally
cd frontend
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run build
npm run preview -- --port 4173
```

### Model Artifact Deployment

- Trained production models (`Random Forest`, `Gradient Boosting`, `Linear Regression`) are persisted via `joblib` inside `models/saved_models/`.
- The ML inference service (`MLForecastService`) loads models once into memory on application startup.
- If model artifacts are not yet generated, the system falls back gracefully with informative HTTP 404/503 guidance without crashing.

---

# Enterprise Architecture & Portfolio Highlights

1. **8-Year Commercial Dataset**: 8 years of realistic retail transactions with weekly seasonality, annual cycles, holidays, promotion events, and secular demand trends.
2. **Multi-Horizon ML Forecasting**: Horizon-optimized models (Random Forest for 7-day, Gradient Boosting for 30-day, Linear Regression for 90-day) evaluated via rolling-origin cross-validation.
3. **Sales Anomaly Detection (Phase 6.1)**: Strictly causal rolling-window statistical scoring (Median + MAD $\times 1.4826$) with zero future data leakage.
4. **Root-Cause Attribution Engine (Phase 6.2)**: Multi-dimensional decomposition isolating promotion impact, price elasticity, category volume shifts, and product-level drivers.
5. **Grounded AI Reasoning (Phase 6.4)**: Gemini-powered executive narratives bounded by deterministic statistical evidence and strict anti-hallucination post-validation.
6. **Prescriptive Action Recommendations (Phase 6.5)**: Domain-specific interventions (pricing, inventory, promotions, category strategy) with quantified impact and feasibility scoring.
7. **What-If Scenario Simulation (Phase 6.6)**: In-memory counterfactual simulator evaluating demand shifts, price adjustments, and promotion scenarios without altering historical transactions.
8. **Human Decision Governance & Audit Trail (Phase 6.7)**: Maker-checker approval gate (`human_approval_required: true`, `automatic_execution: false`) with immutable append-only audit event logs.
9. **Proactive Intelligence & Monitoring (Phase 6.8)**: Automated commercial health scanner surfacing newly significant demand shifts and threshold breaches.
10. **Enterprise Hardening & Resilience (Phase 7)**: Connection pooling, sliding-window rate limiting, correlation ID tracing (`X-Request-ID`), sanitized 500 error envelopes, and automated regression suite.

---

# Known Limitations & Engineering Horizon

- **Price & Discount Elasticity Modeling**: Current simulation multipliers use calibrated domain heuristics; future iterations can integrate econometric double machine learning (DML) or causal forest estimators.
- **Simulation Modeling**: Current simulation models are piece-wise linear approximations; non-linear macroeconomic or cross-product cannibalization dynamics can be added.
- **Rate Limiting**: Current sliding-window rate limiter is in-memory per worker instance; distributed enterprise deployments should bind to Redis or Valkey clusters.
- **Background Orchestration**: Proactive monitoring scans are invoked via API / in-process timers; production scale can incorporate temporal workers, Celery, or Cloud Tasks.
- **Database Migrations**: Initial deployments leverage SQLAlchemy declarative schema syncing; future enterprise CI/CD can integrate Alembic for zero-downtime migration tracks.

---

# License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

# Author

**Durga Prasad Shetty**

GitHub:

https://github.com/shettyprasad-git

---

# Project Summary

**Smart Sales Forecasting System** combines data science, machine learning, backend engineering, and frontend development into a single end-to-end application.

```text
Data
 ↓
EDA
 ↓
Feature Engineering
 ↓
Forecasting Models
 ↓
Model Evaluation
 ↓
Production ML Artifacts
 ↓
FastAPI
 ↓
REST APIs
 ↓
React Dashboard
 ↓
Business Insights
```

The result is a full-stack forecasting platform capable of transforming historical sales data into future demand predictions through an interactive business intelligence interface.