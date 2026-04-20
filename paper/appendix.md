# Appendix: Implementation Details, Reproducibility, and Submission Checklist

Date: 2026-05-XX

---

## A. Architecture (Detailed)

### A.1 High-Level Pipeline

The system is organized as a pipeline coordinated by a central orchestrator. The orchestrator either:

* Runs a "historical RAG" pipeline for fact questions (temporal validation -> retrieval -> synthesis, optionally verification), or
* Routes to the "analysis pipeline" for ratios, extremes, and projections (structured loader + projection / historical analyzers).

Mermaid diagram (copy into a renderer or export via Markdown-to-PDF toolchains that support Mermaid):

```mermaid
flowchart LR
  U[User Query + Ticker + optional analysis_date] --> O[MultiAgentOrchestrator]
  O --> T[TemporalAgent: validate query + date semantics]
  T -->|cutoff_date| D[DocumentRetrievalAgent]
  D --> VS[TemporalVectorStore: ChromaDB with filed_date metadata filter]
  VS --> D
  O --> C[CalculationAgent: XBRL ratios (CSV/JSON)]
  O --> A[FinancialAnalysisAgent: projections, trends, DCF]
  O --> V[VerificationAgent: cross-check claims]
  D --> S[Synthesis LLM]
  C --> S
  A --> S
  V --> S
  S --> R[Final Answer + metadata]
```

### A.2 Key Modules (Code Map)

Core orchestration and agents:

* `src/agents/orchestrator.py`: routes query types and coordinates stages.
* `src/agents/temporal_agent.py`: temporal reasoning and warnings.
* `src/agents/document_agent.py`: temporally constrained retrieval.
* `src/agents/calculation_agent.py`: ratio/growth computations from structured files.
* `src/agents/analysis_agent.py`: projections and analytics.
* `src/agents/verification_agent.py`: lightweight verification functions.

Retrieval and storage:

* `src/rag_system/vector_store.py`: ChromaDB vector store with metadata filters.
* `src/rag_system/temporal_retriever.py`: point-in-time retrieval helpers and lag logic.

Ingestion:

* `scripts/ingest_all.py`: unified ingestion CLI.
* `src/data_collection/sec_downloader.py`: EDGAR filing downloads.
* `src/data_collection/xbrl_parser.py`: XBRL company facts download and metric extraction.
* `src/data_collection/ir_collector.py`: IR document discovery and optional download.

---

## B. Data Layout and Schemas

### B.1 Raw Data Directories (Expected)

The code expects conventional directories under `data/` such as:

* `data/raw/sec_filings/`: HTML filings downloaded from EDGAR.
* `data/raw/xbrl_structured/`: structured facts JSON and/or `*_metrics.csv`.
* `data/raw/yahoo_finance/`: cached Yahoo Finance JSON profiles/fundamentals (optional).
* `data/raw/investor_relations/`: discovered IR docs/manifests (optional).

### B.2 Structured Numeric Files

Two structured formats are used:

1. `TICKER_facts.json`: raw SEC company facts JSON.
2. `TICKER_metrics.csv`: extracted records of key metrics (metric, value, end_date, filed_date, form, etc.).

For production robustness, the following fields should be treated as first-class:

* `concept` (XBRL tag)
* `unit` (USD, shares, USD/shares)
* `period_end` and `period_start` (duration vs instant)
* `filed_date` (for PiT)
* `form` (10-K vs 10-Q vs amendments)

---

## C. Temporal Semantics (Critical)

### C.1 Recommended Single Contract

To avoid subtle temporal leakage bugs, define:

* `analysis_date` = the as-of date that the user pretends to be at.
* Retrieval rule = only documents where `filed_date <= analysis_date` are eligible.
* Period availability rule (warning only) = requested period is "available" only if `period_end + filing_lag <= analysis_date`.

In other words:

* The retrieval filter should not "subtract lag" from analysis_date.
* Lag is used to explain whether a requested period could be known, not to exclude already-filed documents.

### C.2 Common Pitfalls

* Using `period_end` as though it were `filed_date`.
* Filtering on `period_end` instead of `filed_date`.
* Treating a year mention in a query as evidence that the year’s filing is available.
* Mixing annual and quarterly facts without a consistent aggregation rule.

---

## D. Projections: Making Them Scientific and Defensible

### D.1 What a Publishable Projection Should Look Like

A publishable output should include:

* The metric definition (e.g., OCF, FCF).
* The historical data points used (years and values).
* The projection method (CAGR, linear, etc.) and parameters.
* An interval or scenario range.
* A clear label: "estimate" not "fact."

### D.2 Recommended Constraints

To prevent "LLM storytelling" from producing arbitrary numbers:

* Compute projections deterministically in code.
* Use the LLM only to explain computed results and summarize assumptions.
* Enforce output schema (JSON) for the analysis stage, then render to text.

---

## E. Evaluation Plan and Benchmark Design

### E.1 Suggested Benchmark Categories

Build a benchmark with at least:

* Temporal leakage queries:
  - queries that try to bait the system into using future information.
* Numeric grounding queries:
  - questions with known numeric answers from XBRL.
* Qualitative filing questions:
  - risk factors, MD&A topics, accounting policies.
* Projection backtests:
  - project from year t to t+1 or t+2 and compare to realized outcomes.

### E.2 Suggested Metrics

Temporal integrity:

* Leakage rate: fraction of cited evidence with `filed_date > analysis_date`.
* Warning accuracy: fraction of period requests correctly flagged as unavailable.

Numeric:

* Absolute/relative error vs XBRL truth.
* "Abstain vs hallucinate" rate.

Retrieval:

* Precision@k and recall@k with a small labeled set.

Projections:

* MAPE / sMAPE on backtested forecasts.
* Coverage of intervals (if intervals are used).

---

## F. Reproducibility Checklist (What to Include in Submission)

1. Frozen environment:
   * Pin Python version and dependency versions.
2. Deterministic run mode:
   * Fixed random seeds where applicable.
   * Fixed retrieval parameters (k, filters).
3. Data snapshot:
   * A small curated dataset included (or instructions to recreate it).
4. Scripted end-to-end run:
   * A single command that downloads a small set of tickers and runs evaluation.
5. Clear limitations:
   * Which parts require network access and which are offline.

---

## G. Presentation Outline (15 minutes)

1. Problem: Why look-ahead bias breaks financial QA (2 min)
2. Key idea: Point-in-time retrieval + structured numeric grounding (3 min)
3. System demo: One historical query and one calculation query (3 min)
4. Projection demo: Project OCF/FCF with explicit assumptions (3 min)
5. Evaluation: what tests show today and what dataset study would show next (2 min)
6. Limitations + future work (2 min)

---

## H. Submission-Ready Improvements (Prioritized)

If time is limited before 2026-05-06, prioritize:

1. Fix the tests that `return` instead of `assert` (pytest future-proofing).
2. Unify as-of semantics (filed_date <= analysis_date).
3. Disable mock numeric data by default or make it explicit opt-in.
4. Add a small curated benchmark (10-20 queries) and show results vs a baseline.
5. Make the final output schema explicit:
   * answer_type: reported_fact / calculated_metric / model_estimate
   * evidence: doc ids, filed dates
   * warnings: temporal, missing evidence, etc.

