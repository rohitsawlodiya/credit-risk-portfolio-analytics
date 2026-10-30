# Credit Risk Portfolio Analytics

A complete local solution to the take-home assignment: a reproducible five-table data pipeline, portfolio analytics, two default classifiers, a Streamlit dashboard, grounded LLM Q&A, and anomaly review with a persistent local audit log.

**Start here:** run the three commands below. **Next owner:** read [HANDOVER.md](HANDOVER.md) for verification, deployment, live-key checks and the demo script.

## Run locally

Tested with **Python 3.12.14** on macOS Apple Silicon. Use a clean Python 3.12 virtual environment, open a terminal in this repository, and run:

```bash
python -m pip install -r requirements.txt
python -m src.pipeline
python -m streamlit run app/app.py
```

The dashboard opens at `http://localhost:8501`. The ZIP already includes the processed data, fitted models and two PDFs. The build regenerates data/model artifacts from the original CSVs. All installed packages, including transitive dependencies, are pinned in `requirements.txt`. Linux/container deployment has not been executed in this handover.

### LLM configuration

Set `OPENAI_API_KEY` privately in the process environment, in an ignored `.env` file at the repository root, or in Streamlit's server-side secrets. `.env.example` contains variable names with no secret values. Never commit `.env` or `.streamlit/secrets.toml`.

`OPENAI_MODEL` is optional and defaults to `gpt-4.1-mini`. `REVIEW_DB_PATH` is optional and defaults to `runtime/reviews.sqlite3`. Use a durable writable volume for that path on a host. Environment variables take precedence over Streamlit secrets.

**Without a key:** all non-LLM views work. The assistant displays a configuration message and disables submission. It never substitutes canned or simulated LLM output. The supplied tests exercise the real SDK against a mocked HTTP transport; the live provider test is separately opt-in and was not run.

## What is delivered

| Brief item | Implementation and evidence |
| --- | --- |
| All five source tables | `data/raw/` contains the byte-for-byte original CSVs and the supplied dictionary |
| Ingestion, joins, quality decisions | `src/ingestion.py`; `artifacts/data_quality.json` |
| One analysis-ready dataset | `data/processed/portfolio.csv`: one row per loan, including features, scores, partitions and anomalies |
| Income, employment, region, age and loan-type profiling | `src/analytics.py`; `artifacts/segment_*.csv`; Portfolio view |
| Five findings with implications, including a cross-cut | `artifacts/report.json`; Portfolio view |
| Logistic baseline and ML model | `src/model.py`; `artifacts/default_model.joblib` |
| AUC, KS, Gini, importance and monitoring | Model & data view; `artifacts/report.json`; `artifacts/feature_importance.csv` |
| Account scores and local drivers | Account explorer; `src/model.py:account_drivers` |
| Interactive filters, exposure, segments and trends | `app/app.py`; shared filters, dimension selector, relative-month chart |
| LLM Q&A and provenance | `src/assistant.py`; Ask the Portfolio view |
| Anomaly flags, reasons and decisions | `src/anomalies.py`; `src/reviews.py`; Anomaly review view |
| Trigger, false positive/negative and human step for both AI features | Embedded explanations in their dashboard views; solution overview PDF |
| Exactly two one-page assignment documents | `docs/architecture.pdf`, `docs/solution_overview.pdf` |
| Document regeneration | `src/documents.py` reads actual pipeline results |
| Tests | `tests/`: ingestion, model leakage/metrics, anomaly reproducibility, LLM grounding/SDK contract, UI, review storage, PDF constraints |
| Deployment handover and interview/demo guide | `HANDOVER.md`; optional `Dockerfile` |
| Hosted link, repository link, video | Intentionally left to the next owner, as requested; none is claimed to exist |

## Results on the supplied snapshot

- 10,000 customers, 14,685 loans, 2,364 defaults: **16.10%** observed account default rate.
- **$601,119,531.64** original exposure. Latest observed loan balances total **$512,637,943.19**, at different loan ages/dates.
- **588** anomaly review flags: 441 isolation outliers and 221 deterioration-rule flags, with overlap. No over-limit flag in this snapshot.

| Model | Validation AUC | Test AUC | Test KS | Test Gini | Test Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.6792 | 0.6857 | 0.2765 | 0.3714 | 0.1246 |
| Histogram gradient boosting | 0.6829 | 0.6603 | 0.2292 | 0.3206 | 0.1275 |

The split contains 6,000 training, 2,000 validation and 2,000 test customers; their loans never cross partitions. The test set has 2,948 loans and 472 defaults. Boosting was selected using validation AUC; logistic regression performs better on the untouched test set. The selection rule remains frozen rather than using the test result to choose a new winner. The small validation advantage and overlapping bootstrap intervals do not establish a reliable superiority claim. A production decision needs new dated holdouts and independent validation.

`KS = max(TPR - FPR)` from the ROC curve, and `Gini = 2 × AUC - 1`. Both are stored on a 0–1 scale. AUC intervals resample test customers 250 times with the fitted model fixed. They do not account for model-training uncertainty.

## Data and modelling decisions

Loans are the fact-table grain. Bureau lines aggregate to customer; repayments and collections aggregate to account. These aggregates join to loans with validated cardinality. Original exposure, default count and raw recovery totals reconcile to independent source totals. Monthly balances are stocks, so only each loan's final observed balance is used.

Missing numeric values remain explicit in the analysis dataset. Model imputation learns training medians and missing indicators only. No-bureau customers have zero observed lines and a no-history flag; unknown DPD/utilisation stay missing. Gender labels are normalized. Employment/residence tenure exceeding age plus a one-year rounding margin is set to missing, with original values and invalidity flags preserved. Raw files are never modified.

There are **3,875 collections events before loan origination**, excluded from valid collection/recovery totals but preserved in raw totals and exception counts. There are **10,755 accounts with later-opened bureau lines** and **7,293 accounts whose customer-since date follows loan origination**. These inconsistencies are visible in the audit. Collections, identifiers, calendar dates, age, gender, marital status, dependents and region are excluded from the default feature allowlist.

**Important interpretation:** the target has no default event date, observation timestamp or performance horizon. Bureau attributes are a current snapshot, and all 12 supplied repayment months are aggregated. This is a **retrospective monitoring/classification exercise**, not a leakage-free underwriting model or validated 12-month PD. Customer separation prevents shared-borrower leakage; it cannot resolve temporal ambiguity. A no-bureau/no-repayment logistic ablation has test AUC 0.6550, but even profile timestamps are unknown.

Score bands (<10%, 10–<25%, ≥25%) are illustrative. A separate review threshold maximizes validation KS and is frozen before test evaluation; it is not a cost-optimal lending decision. Account explanations replace one raw feature with its training median/mode and show the score change. They are local sensitivities, not additive SHAP values or causal effects.

## Grounded LLM design

The server sends OpenAI the question, allowed metric/dimension catalogue, category values and active filter selections. It does not send raw account records, histories, scores or computed portfolio figures. A strict JSON schema returns an allowlisted query plan; Pydantic validates it again. Python intersects the question filters with the active dashboard filters, calculates the requested measure and renders every number deterministically. No generated SQL, code, or model-written numeric answer is executed or trusted.

The response includes the interpreted plan, source file, calculation, numerator/denominator where relevant, matched account count and data version. Unknown categories, malformed plans, empty selections, incomplete responses and provider failures fail visibly. Unsupported/future/causal questions are declined. Interpretation can still be wrong: the analyst checks the plan and provenance. An exact calculation is not proof that the LLM understood the question.

## Anomaly review design

Isolation Forest uses six repayment/bureau features expressed relative to loan-type/income-band peer medians and interquartile ranges. Sparse or effectively constant peer groups fall back to portfolio statistics; missing values map to peer medians and can conceal anomalies. Quantization below measurement precision avoids unstable tree partitions after CSV round trips. A 3% contamination setting represents review capacity, not known anomaly prevalence.

Two explicit rules supplement the detector: last-three-month average DPD rises by at least 20 days over earlier months and reaches at least 30 days; or bureau balance/sanctioned amount exceeds 1. Reasons include observed values and comparison values. Review events append to SQLite with analyst, account, decision, note, UTC timestamp and dataset ID. Analysts are self-declared in this demo; authentication, durable storage and access controls are production requirements.

## References

Definitions come from the supplied `data/raw/DATA_DICTIONARY.md`; all results come from the supplied CSVs. Implementation guidance: [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app), [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management), and [Hugging Face Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker). Deployment instructions were checked on 30 October 2026; account-specific availability still needs verification.
