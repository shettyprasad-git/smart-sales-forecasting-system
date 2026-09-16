# Smart Sales Forecasting System

> An end-to-end machine learning and full-stack platform for analyzing historical sales performance and forecasting future product demand across 7, 30, and 90-day horizons.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-Frontend-646CFF.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-v4-06B6D4.svg)](https://tailwindcss.com/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00.svg)](https://www.tensorflow.org/)
[![Tests](https://img.shields.io/badge/Backend%20Tests-74%2F74%20Passing-success.svg)](#testing)
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
    │   └── sales.py
    │
    ├── schemas/
    │   ├── auth.py
    │   ├── forecast.py
    │   ├── products.py
    │   └── sales.py
    │
    ├── services/
    │   ├── forecast_service.py
    │   └── history_service.py
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
    │   └── forecast.js
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
    │   └── ConfirmDialog.jsx
    │
    ├── pages/
    │   ├── Login.jsx
    │   ├── Register.jsx
    │   ├── Dashboard.jsx
    │   ├── Products.jsx
    │   ├── Sales.jsx
    │   ├── Forecast.jsx
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
- Multi-dimensional driver decomposition (Promotions, Holidays, Categories, Products, Prices, Discounts, Drift)
- Financial & demand impact quantification with price basis documentation
- Causal time-series invariance (zero future data leakage)
- Synthetic spike and drop detection
- Severity, metric, date, and direction filtering
- Category and product-level anomaly endpoints
- Validation
- Error handling

Run:

```powershell
pytest -q
```

Current result:

```text
74 passed
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
- [x] Multi-driver root-cause decomposition without lookahead leakage
- [x] Category and Product anomaly queries & contributions
- [x] Quantified business impact estimation
- [x] Validation works
- [x] Error handling works
- [x] 74/74 tests passing

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
[ ] Production database
[ ] Dockerization
[ ] CI/CD
[ ] Cloud deployment
[ ] Monitoring
```

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