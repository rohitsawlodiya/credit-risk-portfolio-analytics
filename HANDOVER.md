# Technical Operations and Deployment Guide

## 1. Purpose

This document describes the operational, technical, and deployment considerations for the Credit Risk Portfolio Analytics application.

It is intended for developers, technical reviewers, and maintainers who need to understand how to configure, execute, verify, and maintain the application.

For a business-oriented overview of the solution, architecture, features, and model results, refer to `README.md`.

## 2. Application Components

The application consists of the following modules:

| Module | Responsibility |
|---|---|
| `src/ingestion.py` | Data ingestion, validation, cleaning, aggregation, and joins |
| `src/analytics.py` | Portfolio metrics, segmentation, and analytical findings |
| `src/model.py` | Default classification, model evaluation, and account-level explanations |
| `src/anomalies.py` | Isolation Forest and rule-based anomaly detection |
| `src/assistant.py` | LLM-based question interpretation and validated query execution |
| `src/reviews.py` | SQLite-based anomaly review and audit history |
| `src/pipeline.py` | End-to-end data and model pipeline orchestration |
| `src/documents.py` | Generation of analytical PDF documents |
| `app/app.py` | Streamlit dashboard and user interaction |

The application separates LLM-based interpretation from numerical computation. The LLM produces a structured query plan, while Python validates and executes the analytical calculations.

## 3. Environment Setup

### Requirements

- Python 3.12
- Git
- An OpenAI API key for live assistant functionality

### Installation

Clone the repository:

```bash
git clone https://github.com/rohitsawlodiya/credit-risk-portfolio-analytics.git
cd credit-risk-portfolio-analytics
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Environment configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4.1-mini
REVIEW_DB_PATH=runtime/reviews.sqlite3
```

The API key must remain private.

Never commit `.env`, `.streamlit/secrets.toml`, or other files containing credentials.

The application can run its non-LLM functionality without an API key. The portfolio assistant requires a valid provider configuration.

## 4. Running the Application

### Generate the analytical pipeline

From the repository root:

```bash
python -m src.pipeline
```

This executes the data processing, feature engineering, model training, anomaly detection, and artifact generation workflow.

### Generate documentation

```bash
python -m src.documents
```

### Launch the dashboard

```bash
python -m streamlit run app/app.py
```

The application is available at:

`http://localhost:8501`

### Verify application health

With Streamlit running, execute:

```bash
python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8501/_stcore/health').read().decode())"
```

Expected response:

```text
ok
```

Stop the application using `Ctrl+C`.

## 5. Testing and Verification

Run the automated test suite:

```bash
python -m pytest -q
```

The recorded project test result is:

**32 passed, 1 skipped.**

The skipped test is the optional live LLM provider test. The standard assistant tests use mocked HTTP transport to validate the SDK integration and query-plan behaviour without making paid provider calls.

The test suite covers:

- Data ingestion and validation.
- Join integrity and data quality.
- Model evaluation and leakage controls.
- Anomaly detection reproducibility.
- LLM schema validation and analytical grounding.
- Dashboard functionality.
- SQLite review persistence.
- PDF generation constraints.

After modifying data-processing or modelling logic, run:

```bash
python -m src.pipeline
python -m src.documents
python -m pytest -q
```

Restart Streamlit after rebuilding artifacts so that cached data and models are refreshed.

## 6. LLM Configuration and Validation

The portfolio assistant uses the OpenAI API, with `gpt-4.1-mini` as the default model.

Configuration variables:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | API authentication |
| `OPENAI_MODEL` | Optional model override |
| `REVIEW_DB_PATH` | SQLite review database location |

### Live assistant verification

After configuring the API key, restart the dashboard and test the following questions:

1. Which loan type has the highest default rate?
2. Compare original exposure by region.
3. What is the default rate for unemployed customers below 30k income?
4. What will the exact default rate be in 2030?

The first three questions should produce analytical answers grounded in the supplied dataset. The fourth should be declined because the dataset cannot establish future default rates.

Also test the assistant after changing dashboard filters. The resulting analysis should respect the active filter selections.

### Optional live API test

The live provider test makes an actual API request and may incur a small usage charge.

On Windows PowerShell:

```powershell
$env:RUN_LIVE_LLM_TESTS = "1"
python -m pytest tests/test_assistant.py::test_live_llm_end_to_end -q
```

This test requires `OPENAI_API_KEY` in the process environment. A key configured only through Streamlit secrets or `.env` may need to be exported for this CLI test, depending on the project's environment-loading implementation.

Do not expose API keys in terminal screenshots, shared logs, source files, or Git commits.

## 7. Data Processing and Join Design

The loan accounts table is the primary fact table, with one row per loan.

The joins follow these rules:

- Customer data is joined at customer level.
- Bureau records are aggregated to one row per customer.
- Repayment records are aggregated to one row per account.
- Collections events are aggregated to one row per account.
- Aggregated tables are joined to the loan table using validated relationships.

This approach prevents row multiplication when customers have multiple bureau records or accounts have multiple repayment and collections events.

Monthly outstanding balances represent stocks. Therefore, the latest observed balance is used rather than summing balances across months.

Raw source files remain unchanged. Data-quality decisions and reconciliation outputs are available in `artifacts/data_quality.json`.

## 8. Model Design and Interpretation

The project implements:

- Logistic Regression.
- Histogram Gradient Boosting.

The customer-level split uses training, validation, and test partitions. All loans belonging to a customer remain in the same partition.

Preprocessing transformations are fitted using training data only.

Model evaluation includes:

- ROC-AUC.
- KS statistic.
- Gini coefficient.
- Average precision.
- Brier score.
- Threshold-based performance measures.

The selected model is determined using validation AUC. Test results are retained as independent evaluation evidence.

### Important modelling limitation

The dataset does not provide a default event date, observation timestamp, or defined performance horizon.

Consequently, the model represents retrospective portfolio classification and monitoring, not a validated forward-looking probability-of-default model.

A production underwriting model would require dated historical observations, a defined prediction horizon, appropriate out-of-time validation, and independent model governance.

Account-level explanations are feature-reference substitutions. They should be interpreted as local score sensitivities, not causal effects or additive SHAP explanations.

## 9. Anomaly Detection and Review Operations

The anomaly detection module combines:

1. Isolation Forest.
2. Rule-based repayment deterioration detection.
3. Bureau balance-to-sanctioned-amount checks.

The detection process uses repayment and bureau characteristics and does not use the default target to fit the unsupervised detector.

Anomaly flags are indicators for investigation. They are not proof of fraud, default, or misconduct.

### Review persistence

Analyst decisions are stored in SQLite.

The default database path is:

```text
runtime/reviews.sqlite3
```

The optional `REVIEW_DB_PATH` variable can point to another writable location.

Each review event records:

- Analyst identifier.
- Account identifier.
- Review decision.
- Analyst note.
- UTC timestamp.
- Dataset identifier.

Review history is append-only within the application workflow.

For a production environment, additional authentication, authorization, shared durable storage, and audit controls would be required.

## 10. Deployment Considerations

### Streamlit Community Cloud

Recommended configuration:

| Setting | Value |
|---|---|
| Repository | `rohitsawlodiya/credit-risk-portfolio-analytics` |
| Branch | `main` |
| Main file | `app/app.py` |
| Python | 3.12 |
| Dependencies | `requirements.txt` |

Configure the API key through the application's server-side Secrets settings:

```toml
OPENAI_API_KEY = "your_api_key"
```

Do not commit this configuration to GitHub.

The repository includes processed data and model artifacts so the dashboard can load without retraining the models at application startup.

Official documentation:

- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [Streamlit secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

### Storage considerations

The local SQLite review database may not survive application recycling or redeployment on hosts with ephemeral storage.

A production deployment should use persistent, access-controlled storage.

The application should not claim durable review history unless the deployed environment has been configured and tested accordingly.

### Security considerations

Before exposing the application publicly:

- Store API credentials only in server-side secrets.
- Never expose credentials to browser-side code.
- Apply provider spending limits and usage monitoring.
- Introduce appropriate request quotas for public access.
- Protect analyst review functionality with authentication and authorization.
- Avoid logging sensitive customer-level information.
- Use a trusted source for model artifacts and dependencies.

## 11. Troubleshooting

| Problem | Recommended action |
|---|---|
| Missing processed data | Run `python -m src.pipeline` |
| Missing Python package | Activate the virtual environment and reinstall `requirements.txt` |
| Assistant unavailable | Verify the server-side API key and model access |
| Provider request failure | Check API quota, billing, model availability, and network access |
| Empty dashboard | Review active filters and ensure valid selections |
| Stale dashboard results | Restart Streamlit after rebuilding artifacts |
| SQLite write failure | Set `REVIEW_DB_PATH` to a writable directory |
| Review history lost after redeployment | Configure persistent storage |
| Model compatibility warning | Rebuild artifacts using the pinned dependency environment |

## 12. Production Readiness Roadmap

The current solution is an analytical demonstration using synthetic data. Before production use, the following improvements would be necessary:

1. Introduce dated observations and a defined default prediction horizon.
2. Perform independent out-of-time model validation.
3. Establish model monitoring, drift detection, and recalibration procedures.
4. Introduce authenticated analyst identities and role-based access.
5. Move review persistence to a durable shared database.
6. Implement provider request quotas, cost monitoring, and stronger operational controls.
7. Establish model versioning, approval, rollback, and audit procedures.

The application is designed to demonstrate an end-to-end credit risk analytics workflow while making its modelling, data, and operational limitations explicit.