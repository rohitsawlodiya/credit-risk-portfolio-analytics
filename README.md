# Credit Risk Portfolio Analytics

An AI-powered credit risk monitoring platform that combines portfolio analytics, machine learning, anomaly detection, and grounded LLM-based interaction in an interactive Streamlit dashboard.

The project demonstrates an end-to-end approach to building a data-driven risk monitoring solution, from multi-table data ingestion and feature engineering to model evaluation, portfolio analysis, explainable outputs, and human-in-the-loop anomaly review.

## Project Overview

Financial institutions need to monitor portfolio quality, identify emerging risk concentrations, understand account-level risk, and investigate unusual borrower behaviour.

This project addresses these needs through five integrated components:

1. **Data Engineering:** A reproducible pipeline that processes five interconnected banking datasets into a unified, analysis-ready loan-level dataset.
2. **Portfolio Analytics:** Interactive monitoring of default rates, outstanding exposure, customer segments, loan characteristics, and portfolio trends.
3. **Machine Learning:** Logistic regression and histogram gradient boosting classifiers for account-level default classification.
4. **LLM-Powered Portfolio Assistant:** A natural-language interface that translates business questions into validated analytical queries and returns evidence-backed answers.
5. **Anomaly Detection and Review:** Isolation Forest and rule-based detection combined with an auditable human review workflow.

## Architecture

The application follows a modular architecture that separates data processing, analytical computation, model execution, AI-assisted interpretation, and dashboard presentation.

```text
             Five Banking Data Tables
                       |
                       v
               Data Ingestion
                       |
                       v
          Cleaning, Validation & Joins
                       |
                       v
             Feature Engineering
                       |
             +---------+---------+
             |                   |
             v                   v
       Portfolio Analytics   ML Pipeline
                                 |
                         +-------+-------+
                         |               |
                         v               v
                   Default Model   Anomaly Detection
                         |               |
                         v               v
                    Risk Scores     Review Workflow
                         |               |
                         +-------+-------+
                                 |
                                 v
                         Streamlit Dashboard
                                 |
                    +------------+------------+
                    |                         |
                    v                         v
             Portfolio Assistant       Risk Monitoring
                 (OpenAI LLM)           & Exploration
                    |
                    v
             Validated Query Plan
                    |
                    v
          Deterministic Python Analytics
                    |
                    v
           Evidence-backed Answer
```

## Key Features

### 1. Data Engineering and Processing

The solution integrates five source tables:

- Customers
- Loan accounts
- Bureau credit information
- Repayment history
- Collections events

The processing pipeline includes:

- Data type normalization and validation.
- Customer-level and account-level aggregations.
- Validated joins with explicit cardinality checks.
- Missing-value and data-quality handling.
- Reconciliation of exposure, defaults, and recovery totals.
- Feature engineering for portfolio analysis and modelling.

The final analytical dataset contains one record per loan and includes engineered features, model scores, data partitions, and anomaly indicators.

**Implementation:** `src/ingestion.py`, `src/pipeline.py`

### 2. Portfolio Analytics Dashboard

The Streamlit dashboard provides an interactive view of portfolio performance and risk distribution.

Key capabilities include:

- Total loan exposure and outstanding balances.
- Observed default rates.
- Portfolio segmentation by income, employment, region, age, and loan type.
- Interactive filtering and cross-segment analysis.
- Relative-month repayment trends.
- Portfolio-level findings and risk concentrations.

**Implementation:** `src/analytics.py`, `app/app.py`

### 3. Machine Learning and Default Risk Classification

Two classification approaches are implemented and evaluated:

- Logistic Regression — baseline model.
- Histogram Gradient Boosting — non-linear machine learning model.

The modelling pipeline includes:

- Customer-level train, validation, and test separation.
- Training-only preprocessing and imputation.
- Missing-value indicators.
- Model comparison using validation performance.
- Threshold selection using validation KS.
- Test-set evaluation using AUC, KS, Gini, and Brier score.
- Feature importance and account-level score explanations.

Model selection is based on validation AUC, with the test set retained for independent evaluation.

**Implementation:** `src/model.py`

### Model Performance

Results on the supplied synthetic dataset:

| Metric | Logistic Regression | Histogram Gradient Boosting |
|---|---:|---:|
| Validation AUC | 0.6792 | 0.6829 |
| Test AUC | 0.6857 | 0.6603 |
| Test KS | 0.2765 | 0.2292 |
| Test Gini | 0.3714 | 0.3206 |
| Test Brier Score | 0.1246 | 0.1275 |

Although gradient boosting achieved a slightly higher validation AUC, logistic regression performed better on the untouched test set. The original validation-based model selection rule was retained rather than changed after observing test performance.

These results do not establish that either model is production-ready. Further validation using dated observations and independent out-of-time testing would be required.

### 4. LLM-Powered Ask the Portfolio Assistant

The dashboard includes a natural-language assistant that allows users to ask questions about portfolio performance without manually constructing analytical queries.

Example questions:

- Which loan type has the highest default rate?
- Which region contributes the most defaults?
- Compare default rates across employment categories.
- What is the outstanding exposure for a selected segment?

**How it works:**

1. The user submits a natural-language question.
2. The LLM interprets the question and produces a structured query plan.
3. Pydantic validates the plan against an allowlisted schema.
4. Python applies the requested dimensions, metrics, and active dashboard filters.
5. The analytical result is calculated deterministically.
6. The assistant presents the result with its calculation and data provenance.

The LLM does not generate executable SQL or Python, and its generated numerical answers are not trusted.

The implementation also includes safeguards for unsupported categories, malformed plans, provider failures, and questions involving unsupported future predictions or causal claims.

**Implementation:** `src/assistant.py`

**Technology:** OpenAI API, Pydantic, Python

### 5. Anomaly Detection and Human Review

The anomaly detection module identifies accounts that exhibit unusual repayment or bureau characteristics.

It combines:

- Isolation Forest for multivariate outlier detection.
- Rule-based detection for repayment deterioration.
- Bureau balance-to-sanctioned-amount checks.
- Peer-group comparisons using loan type and income band.
- Explanations containing observed and comparison values.

Detected accounts are presented in an anomaly review interface.

Analysts can record review decisions, notes, and account-level outcomes. Review events are stored in SQLite with timestamps and dataset identifiers, providing a local audit trail.

**Implementation:** `src/anomalies.py`, `src/reviews.py`

## Portfolio Results

The supplied synthetic dataset contains:

| Portfolio Measure | Result |
|---|---:|
| Customers | 10,000 |
| Loans | 14,685 |
| Defaults | 2,364 |
| Observed default rate | 16.10% |
| Original loan exposure | $601.12 million |
| Latest observed loan balances | $512.64 million |
| Anomaly review flags | 588 |

The anomaly flags comprise 441 Isolation Forest outliers and 221 deterioration-rule flags, with overlap between the two groups.

These figures describe the supplied dataset and should not be interpreted as real-world banking portfolio statistics.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python 3.12 |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| LLM integration | OpenAI API |
| Structured validation | Pydantic |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Review persistence | SQLite |
| Testing | Pytest |
| Model serialization | Joblib |
| Documentation | ReportLab |

## Project Structure

```text
credit-risk-portfolio-analytics/
│
├── app/
│   └── app.py
│
├── artifacts/
│   ├── default_model.joblib
│   ├── anomaly_model.joblib
│   ├── data_quality.json
│   ├── report.json
│   └── feature and segment outputs
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── architecture.pdf
│   └── solution_overview.pdf
│
├── src/
│   ├── ingestion.py
│   ├── analytics.py
│   ├── model.py
│   ├── anomalies.py
│   ├── assistant.py
│   ├── reviews.py
│   ├── documents.py
│   └── pipeline.py
│
├── tests/
│
├── scripts/
├── requirements.txt
├── Dockerfile
└── README.md
```

## Running the Project Locally

### Prerequisites

- Python 3.12
- Git
- An OpenAI API key for live assistant functionality

### Installation

Clone the repository:

```bash
git clone https://github.com/rohitsawlodiya/credit-risk-portfolio-analytics.git

cd credit-risk-portfolio-analytics
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the data pipeline:

```bash
python -m src.pipeline
```

Launch the dashboard:

```bash
python -m streamlit run app/app.py
```

Open the application at:

`http://localhost:8501`

### LLM Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key
```

The application uses `gpt-4.1-mini` by default. The API key must be configured privately and must never be committed to the repository.

Without an API key, the non-LLM dashboard functionality remains available, while the assistant displays a configuration message.

## Testing

The project includes automated tests covering:

- Data ingestion and data-quality checks.
- Join integrity and analytical dataset construction.
- Model evaluation and leakage controls.
- Anomaly detection reproducibility.
- LLM query-plan validation and grounding.
- Dashboard behaviour.
- Review storage and audit records.
- Generated documentation constraints.

Run the test suite:

```bash
pytest -v
```

The existing test suite has reported **32 passed and 1 skipped**. The skipped test is the optional live LLM provider test; mocked SDK tests verify the request and response contract without requiring a live API call.

## Data and Modelling Limitations

The dataset does not contain a default event date, observation timestamp, or defined performance horizon. Bureau information is a snapshot, and the supplied repayment history is aggregated across 12 months.

Consequently, this implementation is a **retrospective portfolio monitoring and classification exercise**, not a validated underwriting model or a calibrated 12-month probability-of-default model.

Additional limitations include:

- Synthetic data may not represent real banking populations.
- Customer-level separation prevents shared-borrower leakage but does not resolve temporal ambiguity.
- Model scores and illustrative risk bands should not be used directly for lending decisions.
- Local account explanations represent feature sensitivity, not causal effects or additive SHAP values.
- Anomaly flags indicate cases for investigation, not confirmed fraud or misconduct.
- The SQLite review store is suitable for this demonstration, not a production multi-user deployment without additional persistence and access controls.

Production use would require dated historical data, independent model validation, monitoring, governance, access control, and appropriate operational infrastructure.

## Project Documentation

Additional technical documentation is available in the repository:

- `docs/architecture.pdf` — system architecture.
- `docs/solution_overview.pdf` — solution overview and design decisions.

---

**Project focus:** Credit Risk Analytics | Machine Learning Engineering | LLM Integration | Anomaly Detection | Data Engineering | Explainable AI | Human-in-the-Loop Monitoring