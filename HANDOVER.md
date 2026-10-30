# Handover: Credit Risk Portfolio Analytics

Prepared 30 October 2026. This is a local implementation and source-code handover. The requester explicitly asked for a ZIP, with publishing and submission to be completed by someone else.

## 1. Current state and ownership boundary

The package contains the complete Python backend, Streamlit frontend, all five original tables, processed account-level data, fitted models, analytical outputs, tests and exactly two one-page PDFs. The application runs locally. No GitHub repository, hosted app, video, email or cloud storage has been created.

The OpenAI adapter is implemented and tested through the real Python SDK using a mocked HTTP transport. There was no API key in this environment, so **no real provider call was made**. Do not describe the LLM as live-verified until the next owner completes section 3. Docker configuration is supplied but has not been built or run here. No claim is made about 14-day hosting availability.

The final automated result is recorded in `artifacts/verification.json`. The expected local suite is **32 passed, 1 skipped**, where the skip is the opt-in live API test. The ZIP is also extracted into a clean directory and the backend/UI suite is run there to verify that no original workspace paths are required.

## 2. Local startup, verification and rebuild

Use Python 3.12; the tested interpreter is 3.12.14. Create/activate a clean virtual environment with your usual tooling. The three startup commands are in README: install pinned requirements, build the pipeline, start Streamlit.

Run the automated suite from the repository root:

```bash
python -m pytest -q
```

Check the running server without browser automation:

```bash
python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8501/_stcore/health').read().decode())"
```

Expected response: `ok`. Stop the server with Ctrl-C. The pipeline normally takes seconds to tens of seconds after installation; downloading dependencies is separate.

If data or feature logic changes, rebuild and regenerate both PDFs:

```bash
python -m src.pipeline
python -m src.documents
python -m pytest -q
```

Restart Streamlit after rebuilding. Its data/model cache assumes artifacts do not change within a running process. `src/documents.py` checks that both PDFs contain exactly one page; visually inspect regenerated PDFs as well. Update any human-written README/handover figures if the source data or model specification changes.

Never load a joblib artifact from an untrusted source. The included artifacts were created by this package's pipeline. Rebuild with the pinned environment if an estimator-version warning appears.

## 3. Activate and verify the LLM

1. Obtain an API key with access to the configured model and appropriate quota. Set `OPENAI_API_KEY` through the host's secret store, local environment, ignored `.env`, or ignored `.streamlit/secrets.toml`. Do not place it in source, a screenshot or a commit.
2. Optionally set `OPENAI_MODEL`; the default is `gpt-4.1-mini`. Restart the app and open Ask the Portfolio. A detected key is a configuration check, not evidence of working provider access.
3. Ask “Which loan type has the highest default rate?” on the full portfolio. It should identify Mortgage and reproduce **24.65%** when rounded to two decimal places. Check the displayed exact numerator, denominator and source against `artifacts/segment_loan_type.csv`.
4. Filter to North in the sidebar and ask the same question. Check that both the result and provenance refer only to North. Then ask “Compare original exposure by region” and “What is the default rate for unemployed customers below 30k income?”
5. Ask “What will the exact default rate be in 2030?” and “What is the portfolio's net profit?” Both should decline; those fields/outcomes do not exist.
6. Test an invalid key and a removed key. No answer should be invented; the app must display a safe error or disable submission. Restore the valid key before the demo.
7. Run the opt-in live test (this makes paid provider calls):

```bash
RUN_LIVE_LLM_TESTS=1 python -m pytest tests/test_assistant.py::test_live_llm_end_to_end -q
```

This CLI test reads `OPENAI_API_KEY` from the process environment. It does not automatically read Streamlit secrets or `.env`; export the variable securely in your environment first. No key should appear in a command saved to a shared transcript.

The adapter timeout is 25 seconds with at most one retry. The UI has a per-session three-second cooldown; this is **not** a global quota or abuse control. A public deployment needs provider spending limits, request quotas and an appropriate gateway if exposed beyond the assignment demo. Errors are sanitized so provider request details never appear in the public UI.

## 4. Backend walkthrough and key contracts

| Module | Contract |
| --- | --- |
| `src/ingestion.py` | `prepare_data()` returns one row per loan, a quality audit and cleaned working tables. Original CSVs are untouched. Invalid structural inputs raise errors. |
| `src/analytics.py` | `summary()` and `segment_table()` use explicit account-grain denominators; `build_insights()` generates five numerical findings. |
| `src/model.py` | `train_models()` builds train-only transformations and two estimators; `account_drivers()` reports one-feature reference substitutions. |
| `src/anomalies.py` | `detect_anomalies()` uses peer-relative residuals and history rules, never the target. All flags have numeric reasons. |
| `src/assistant.py` | `make_plan()` calls OpenAI; `execute_plan()` validates and computes locally; `ask_portfolio()` composes the two. |
| `src/reviews.py` | Parameterized SQLite inserts retain an append-only history; latest decision is selected by account and dataset ID. |
| `src/pipeline.py` | Orchestrates ingestion, training, anomaly detection and output serialization. |
| `src/documents.py` | Generates both one-page PDFs from real pipeline results. |
| `app/app.py` | Five dashboard views, shared filters, cached trusted artifacts, review form and server-side secret lookup. |

### Join logic

Loans are the base table. Customers join many-to-one on `customer_id`. Bureau records are reduced to one row per customer before that join. Repayments and collections reduce to one row per `account_id` before one-to-one left joins. Foreign keys and primary keys are checked first. This prevents a bureau-line × repayment-month × collections-event multiplication.

The only temporary many-to-many join is a date-quality diagnostic counting bureau lines opened after each loan. Its output is immediately reduced to account ID and is never used to sum exposure. Customer-level bureau balances repeat across a borrower's loans; do not sum that repeated feature as a portfolio customer-debt total.

### Key features

| Feature family | Definition |
| --- | --- |
| Loan-to-income | Original loan amount / annual income; not EMI affordability or total debt-service ratio |
| Bureau counts | Observed lines, active lines, and written-off lines per customer |
| Bureau utilisation | Sum of all supplied outstanding balances / sum of sanctioned amounts per customer; a broad balance/limit ratio, not revolving-only utilisation |
| Bureau DPD | Maximum/mean over nonmissing DPD plus missing-share flag; no line remains unknown, not zero DPD |
| Bureau debt-to-income | Observed external outstanding balance / annual income |
| Repayment DPD | Mean and maximum across all 12 supplied relative months |
| Missed/late share | Counts of corresponding payment-status values / observed repayment-month count |
| Paid ratio | Sum of EMI paid / sum of EMI due, not an average of per-month ratios |
| DPD change | Latest three observed months' mean minus preceding months' mean; requires at least six months |
| Latest balance | Last observed outstanding balance by payment_month; never sum balances across months |
| Collections | Raw event/recovery totals and separately date-valid totals; never default-model predictors |

Missing employment years, residence years and interest rates are imputed only inside the fitted training pipeline, with missing indicators. All 600 M/F labels are normalized. For 45 customers, employment tenure exceeds age plus a one-year rounding margin; four residence tenures do so. Those values become missing, with raw values and invalid flags preserved. A residence of 26.8 years at integer age 26 is retained because age may be rounded.

There are 1,008 customers with no bureau record. Absence means “no bureau history observed,” not proven absence of debt. Null `date_closed` for active lines is expected. No rows are silently deleted. See `artifacts/data_quality.json` for every count, source SHA-256, quality decision and reconciliation.

### Model design and limitations

The fixed seed is 42. GroupShuffleSplit allocates 60/20/20% of customers; all their loans remain together. Train/validation/test contain 8,818/2,919/2,948 loans. No transformation is fit on holdout data. Numeric features are median-imputed, standardized and augmented with missing indicators; categoricals are imputed and one-hot encoded with unknown-category handling.

Logistic regression is an unweighted L2 baseline (C=1). Histogram gradient boosting uses 180 iterations, learning rate 0.05, 15 maximum leaf nodes, minimum 40 observations per leaf and L2=5. Hyperparameters are fixed, not tuned on test. Class weighting is not used because the score's probability scale and observed prevalence matter; evaluation uses AUC, KS, Gini, AP, Brier and threshold metrics instead of raw accuracy.

Boosting wins validation AUC 0.6829 vs 0.6792. Logistic wins test AUC 0.6857 vs 0.6603. The dashboard retains the validation-selected model, and the report shows both results openly. The boosting threshold of 0.137237 maximizes validation KS; risk bands use fixed illustrative cutoffs 0.10 and 0.25. Neither is a calibrated, approved lending rule. Five-shuffle permutation importance is measured on validation data. Local drivers replace one raw feature with its training reference and recompute a score; correlated/derived features can make these substitutions unrealistic, so they are sensitivities, not causal explanations.

The dataset does not support a valid forward-looking experiment: no target date/horizon, no bureau reporting dates, and profile dates are inconsistent with some originations. All supplied repayment months are used for **retrospective monitoring**. Test customer separation does not fix this. Before underwriting use, obtain an observation timestamp, features frozen at that time, a defined future horizon with complete follow-up and a purged customer/time holdout. Do not infer that a high snapshot score predicts a future default.

The no-behavior logistic ablation has test AUC 0.6550. AUC 95% customer-bootstrap intervals are approximately 0.6568–0.7113 for logistic and 0.6293–0.6835 for boosting. They overlap. The ablation is a sensitivity check, not proof of temporally valid prediction.

### Anomaly behavior and review persistence

The six anomaly inputs are DPD change, missed share, paid/due ratio, bureau balance/limit, maximum bureau DPD and bureau debt/income. Peers share loan type and income band. Fewer than 30 nonmissing observations or near-zero IQR triggers a portfolio fallback. Quantized inputs and residuals make repeated CSV round trips stable. The forest has 200 trees, contamination 0.03 and seed 42. It fits the full unsupervised snapshot without default labels; anomaly percentiles are relative to that snapshot, not probabilities.

There are 441 forest flags and 221 deterioration flags, overlapping by 74, giving 588 unique flagged accounts. Rule-based flags may rank below forest flags because the queue sorts by forest severity; reasons still identify the rule. An account may be high default-risk but not anomalous, or anomalous but performing. Neither implies fraud.

Decisions persist locally at `runtime/reviews.sqlite3`. Each update appends an event; previous evidence is retained. `REVIEW_DB_PATH` can point to a mounted persistent directory. Hosted ephemeral disks may lose history on redeployment or recycling. Analyst names are self-declared and unauthenticated. For a real multi-user workflow, add authenticated identities, authorization, a durable shared database and conflict/audit controls. Never claim this demo has those controls already.

## 5. Deployment and submission by the next owner

**GitHub:** initialize a repository from the extracted package and publish it under the appropriate account. Include raw data, processed data and trusted artifacts so the app starts without training at cloud boot. Exclude `.env`, Streamlit secrets, `.venv`, runtime review logs and temporary files. Use the included `.gitignore`. Public or reviewer-accessible private visibility must match the assignment instructions. No repository publication has been done in this handover.

**Streamlit Community Cloud:** choose the repository/branch, set the entry point to `app/app.py`, choose Python 3.12 and configure `OPENAI_API_KEY` in server-side secrets. The bundled `requirements.txt` is the dependency file. The committed processed data/models let the app boot directly. See [deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) and [secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management). Add persistent review storage or disclose/reconcile its lifecycle before relying on it.

**Alternative Hugging Face Docker Space:** the included Dockerfile exposes port 7860 and runs as non-root UID 1000. The Space's README YAML needs `sdk: docker` and `app_port: 7860`; configure the key as a runtime secret. If persistent storage is provisioned and mounted, set `REVIEW_DB_PATH` to that writable location. See [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker). The Dockerfile uses the Python 3.12 slim base; verify/build it on the destination before claiming support. The local environment has no Docker runtime, so this path is untested here.

After deployment, verify the public link in a private/incognito browser without login. Exercise all five views, a changed filter, a real LLM answer and decline, and a saved anomaly decision. Restart the app and verify the decision persists if durable storage is claimed. Set spend/quota controls and arrange to check app availability for the required 14 days. Free hosting or idle-time behavior is not a guarantee of uninterrupted service.

The actual assignment submission is **one email with three links**: live dashboard, GitHub repository, and demo video. Do **not** submit this handover ZIP as the final assignment; the user requested it only for transferring the work. The required PDFs live in the repository. The next owner must send the email; no email has been sent.

## 6. Suggested 3–5 minute demo

| Time | What to demonstrate |
| --- | --- |
| 0:00–0:40 | Portfolio: 14,685 loans, 16.10% observed defaults, $601.12m original exposure; explain snapshot limitations briefly. |
| 0:40–1:20 | Filter a region, compare loan types, show income × employment and the distinction between relative loan age and calendar trends. |
| 1:20–2:00 | Open an account; show the score, evaluation partition, local drivers and repayment history. Explain drivers are reference substitutions. |
| 2:00–2:50 | Ask “Which loan type has the highest default rate?” with a live key. Show exact figures, plan, filters and source. Ask a future question and show the decline. |
| 2:50–3:40 | Open a flagged account, read its numeric reason, inspect history, and record a confirm/dismiss decision with an evidence note. Show the audit log. |
| 3:40–4:30 | Model & data: show both models' AUC/KS/Gini, the customer split, collections-date issue and production monitoring plan. |

Record the actual hosted app with a working LLM, not mocked test responses. Upload the video to the chosen sharing service and verify reviewer access. This package includes the script, not a recorded video.

## 7. Interview explanation in plain language

- **Why aggregate before joins?** One customer has many bureau lines and a loan has many payments/events. Joining those raw tables multiplies rows and inflates amounts. Aggregate each child to its intended key first.
- **Why keep missing values?** Missing is not the same as zero risk. Preserve it and learn replacements from training data only, with indicators.
- **Why split by customer?** The same borrower's profile must not appear in training and test through different loans.
- **Why not call this a production PD?** We cannot prove the predictors existed before the default because dates and horizon are missing. An honest retrospective model is the defensible result.
- **Why does logistic beat boosting on test?** More flexible models do not always generalize better. A tiny validation advantage can reverse. Report the result and validate on new dated data rather than selecting on test.
- **How are LLM numbers grounded?** The LLM chooses a permitted query, Python validates and calculates it, and deterministic text displays the answer and source. The model never writes the numeric answer.
- **Can grounded Q&A still be wrong?** Yes: it can choose the wrong allowed interpretation. The analyst must inspect the plan. Provider live behavior remains an explicit acceptance check.
- **Why anomaly detection plus rules?** The forest catches combinations of unusual peer patterns; rules provide clear history deterioration triggers. Unusual behavior is not automatically default or fraud.
- **How would you monitor it?** Daily data/operational checks; weekly drift, queue and Q&A quality; AUC/KS/Gini/calibration on mature future cohorts; human review and versioned rollback.

## 8. Troubleshooting

| Symptom | Action |
| --- | --- |
| “Build the data and models first” | Run the pipeline from the repository root; confirm `data/processed/portfolio.csv` and `artifacts/report.json` exist. |
| Module/package error | Confirm Python 3.12 and install the complete pinned requirements into the active environment. |
| Model version warning | Rebuild with the pinned environment; do not mix scikit-learn versions and pickled artifacts. |
| Q&A disabled | Set the server-side key and restart. Keys are not entered into the browser. |
| Q&A request failed | Check model access, quota, billing and network privately. Public UI errors intentionally omit provider details. |
| Empty portfolio | A cleared multiselect means no records. Restore at least one allowed value in each filter. |
| Wrong Q&A interpretation | Reject/rephrase the question and verify the interpreted metric, dimensions and filters. |
| Review log disappeared after redeploy | The host used ephemeral storage; provision durable storage and restore an audit export if available. |
| SQLite not writable | Set `REVIEW_DB_PATH` to a writable directory; ensure the service user owns it. |
| Stale charts/scores after rebuild | Restart Streamlit to clear artifact caches. |

The user-visible dashboard explains operational limitations. No automatic credit decisions, claims of causality, real anomaly precision or verified future-risk accuracy are made.
