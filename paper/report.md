# TemporalGuard-RAG: Point-in-Time Financial Question Answering With Temporal Integrity

Author: (Your Name Here)

Course / Program: (Course Name Here)

Instructor: (Professor Name Here)

Institution: (Institution Name Here)

Date: 2026-05-XX

Repository: TemporalGuard-RAG (local workspace)

---

## Abstract

Large language models (LLMs) can answer financial questions in natural language, but in financial research and decision-making settings they often suffer from a critical failure mode: **temporal leakage** (also called look-ahead bias), where an answer implicitly or explicitly relies on information that would not have been available at the time of analysis. This problem undermines the validity of backtests, historical analyses, and compliance-oriented workflows. This project implements a multi-component financial question answering system, **TemporalGuard-RAG**, designed to enforce a point-in-time (PiT) constraint while retrieving and summarizing evidence from financial filings and structured statements. The system couples a temporally filtered vector store for filing excerpts with structured numeric extraction and calculation from SEC XBRL company facts. It exposes the pipeline through a programmatic orchestrator and optional API/UI entry points.

This report documents (i) the motivating problem, (ii) data sources and ingestion automation, (iii) system architecture and temporal enforcement mechanisms, (iv) numeric calculation and projection features, (v) an evaluation plan and test suite, and (vi) limitations and future work. A key finding is that PiT correctness requires a single consistent "as-of" definition across the entire stack (retrieval, structured data, verification, and projection). The repository includes a functioning prototype with a growing test suite; however, additional engineering and empirical evaluation are required to reach production readiness and publishable research standards.

---

## 1. Introduction

### 1.1 Motivation

Financial question answering is a practical, high-impact application of LLMs: analysts and students routinely ask questions such as:

* "What were the key risk factors for Company X in fiscal year 2022?"
* "Compute ROE and debt ratios as of a given year."
* "Project operating cash flow for the next two fiscal years."

In many cases, the question implicitly contains an **as-of date** constraint: the user wants an answer that is correct for the information that was available at that time, not with hindsight. This requirement is essential for:

* Backtesting investment strategies.
* Studying historical corporate communications.
* Regulatory and audit contexts where decision documentation matters.
* Preventing a system from presenting "future knowledge" as historical fact.

Standard LLM chat systems are not designed to enforce point-in-time constraints, and even standard retrieval-augmented generation (RAG) systems can leak future knowledge if the retrieval layer is not temporally filtered (e.g., retrieving a later 10-K when answering a question set in an earlier year).

### 1.2 Project Goal

TemporalGuard-RAG aims to produce answers that are:

* **Temporally consistent**: evidence comes only from documents filed on or before an as-of analysis date.
* **Evidence-aware**: the system can provide excerpts/metadata of the filings used for an answer.
* **Numerically grounded**: when computing ratios or metrics, results use structured XBRL data rather than narrative text.
* **Explicit about uncertainty**: forecasts are labeled as estimates with assumptions and (when possible) ranges.

### 1.3 Key Contribution (Project Perspective)

From a project perspective, the core novelty is not "multi-agent orchestration" itself, but rather the combination of:

* A temporally filtered retrieval layer, and
* A structured numeric path (XBRL) for computations,

packaged into a workflow that attempts to prevent look-ahead bias and to label answer types (reported fact vs calculated metric vs model estimate).

---

## 2. Background and Problem Definition

### 2.1 What Is Look-Ahead Bias?

Look-ahead bias occurs when an analysis uses information that would not have been available at the time the analysis purports to be conducted. In financial modeling and research, it can arise due to:

* Using filings released after an assumed "analysis date."
* Using revised data that was not available historically.
* Using annual results to answer a question set mid-year without acknowledging reporting lags.
* Using "current" market data when simulating decisions in the past.

In LLM systems, look-ahead bias often appears as:

* An answer referencing future earnings, future guidance, or later filings.
* A summary that blends multiple years without respecting availability.
* Numeric hallucinations presented as historical facts.

### 2.2 Point-in-Time (PiT) Retrieval

PiT retrieval is the principle that retrieval should only surface content that existed at the as-of time. A simple implementation is:

* Filter documents where `filed_date <= analysis_date`.

However, financial reporting adds complexity: users often ask about a fiscal period (e.g., "Q3 2023"), and the filing for that period is only available after a delay. The system may need to:

* Warn users if a requested period's filing would not have been available.
* Allow "period availability" reasoning that is separate from the PiT retrieval filter.

TemporalGuard-RAG includes both "as-of cutoff filtering" and "filing lag reasoning," but a major engineering requirement is to keep these concepts consistent and correctly applied (see Section 8).

---

## 3. System Overview (Simple-English Explanation)

This section describes what happens in simple language for a non-technical reader.

When you type a question like:

> "Project cashflow for Apple for FY 2026-27"

the system tries to do the following:

1. **Understand the date context**. If you give an analysis date ("as of 2023-12-31"), the system uses it as a rule: it must not use documents filed after that date.
2. **Decide what kind of question it is**:
   * If it is a historical question (facts from filings), it tries to retrieve filing excerpts.
   * If it is a calculation question (like ROE), it tries to compute it from structured financial data.
   * If it is a projection question, it tries to project a metric using historical trends and a forecasting method.
3. **Retrieve evidence** (for historical questions): it searches a database of filing chunks and only returns chunks that are "old enough" (filed before the cutoff date).
4. **Compute numbers** (for calculation questions): it loads structured values such as revenue and net income from saved data files, then applies formulas.
5. **Optionally verify**: it attempts to check that claims match retrieved evidence and that numbers match structured data.
6. **Write the final answer**: a final response is generated based only on the data collected by the earlier steps.

The main value-add relative to a generic chat model is: **it tries to show its sources and obey an as-of date, so it does not accidentally use future information**.

---

## 4. Repository and Implementation Summary

### 4.1 Main Components

The repository implements:

* A multi-agent orchestrator that routes queries and coordinates stages.
* A temporal vector store (ChromaDB) with metadata filters for filed dates.
* A temporal agent and retriever logic for temporal reasoning.
* Data collectors for SEC filings, SEC XBRL company facts, Yahoo Finance, prices, and optional investor relations (IR) discovery.
* An analysis agent with basic forecasting methods.
* A test suite that validates temporal reasoning, basic calculations, and routing.

### 4.2 Where the "Truth" Lives

This project relies on two broad classes of data:

1. **Unstructured text**: filing excerpts chunked and embedded in a vector store for semantic retrieval.
2. **Structured data**: numeric facts in JSON/CSV derived from the SEC XBRL company facts API.

In a robust financial QA system:

* **Numbers should come from structured sources** whenever possible.
* **Text should be used to provide context and citations** for qualitative questions.

TemporalGuard-RAG contains both paths but requires additional integration work to make answer grounding systematic (Section 8).

---

## 5. Data Sources and Ingestion Automation

### 5.1 Primary Data Sources (Implemented)

The repo includes collectors for:

* **SEC EDGAR filings** (e.g., 10-K, 10-Q): downloaded to local storage.
* **SEC XBRL company facts**: downloaded as JSON and used to build metric histories.
* **Stock prices and certain fundamentals** via Yahoo Finance: optional and environment-dependent.
* **Investor relations document discovery**: attempts to discover official IR pages and optionally download a curated set of files.

This report focuses on SEC filings + SEC XBRL facts as the most standardized and auditable sources for broad coverage across U.S. public companies.

### 5.2 Unified Ingestion CLI

The repository provides a unified ingestion script:

* `scripts/ingest_all.py`

It allows running various collectors with a single command (examples shown as documentation; actual run depends on network access and credentials):

* SEC filing download (requires an email for SEC user-agent)
* XBRL company facts download
* Yahoo Finance fetch
* Price history download
* IR discovery and optional download

### 5.3 Scaling to Thousands of Companies (Practical Constraints)

For "thousands of companies," the key challenges are:

* Rate limits and politeness requirements (SEC and company websites).
* Storage size (filings are large, especially with exhibits).
* Processing time for chunking/embedding.
* Managing updates and re-ingestion incrementally.

A production system should:

* Maintain a queue and resumable state for ingestion.
* Store raw artifacts in a content-addressed store (hash-based) to deduplicate.
* Separate ingestion from indexing (vector store build) and from query-time retrieval.

TemporalGuard-RAG includes early structure for these goals but does not yet include a full ingestion pipeline scheduler or incremental indexing (Section 8).

---

## 6. Retrieval and Temporal Enforcement

### 6.1 TemporalVectorStore

The vector store is built on ChromaDB and stores document chunks with metadata including:

* `ticker`
* `filing_type`
* `filing_date` (as an integer, used for comparisons)
* `source_path` and other fields

PiT filtering is implemented by a metadata filter:

* `filing_date < cutoff_date`

This design is conceptually sound for preventing retrieval of future filings, provided that:

* The metadata is correct and consistent.
* The cutoff date semantics are clearly defined (see Section 8).

### 6.2 TemporalRetriever

The repository also includes a temporal retriever module that:

* Computes available period estimates based on filing lags.
* Logs temporal context for a given query date.

This is useful for warnings and for interpreting whether a requested period could have been known as-of a date.

### 6.3 Key Design Requirement: One Definition of "As-Of"

A central requirement for correctness is:

* PiT retrieval must mean "filed on/before analysis_date."
* Filing lag reasoning must mean "period-end filings become available after a lag."

If these are conflated, the system can (a) exclude documents that were actually available or (b) allow documents that were not. Section 8 describes this as a top improvement item.

---

## 7. Calculation and Projection Capabilities

### 7.1 CalculationAgent

The calculation agent computes basic ratios from structured XBRL CSV files when available. When data is missing, the current code may fall back to mock values; this is flagged with a warning tag.

For research and production use, mock values should be:

* Disabled by default, or
* Strictly gated behind explicit user acknowledgement.

### 7.2 FinancialAnalysisAgent

The analysis agent provides:

* Metric projections (e.g., revenue, operating cash flow).
* Scenario projections (bull/base/bear).
* Historical extremes and trend analysis.
* A simplified DCF valuation method.

The underlying projection engine includes:

* Linear regression projection.
* CAGR-based projection.
* Growth-rate projection and scenario projection.

### 7.3 Why Projections Are Hard

Forecasting corporate financial metrics is inherently uncertain. A publishable and responsible system must:

* Separate "reported facts" from "forecast estimates."
* Constrain the forecasting methods (avoid arbitrary text-based values).
* Provide assumptions and ranges, and label confidence appropriately.
* Include a validation approach (backtesting or comparing to later realized values) on a dataset.

The current system has the start of the math tools, but still relies on an LLM-driven narrative that can drift away from computed projections if not constrained. This is an important gap for production readiness.

---

## 8. Evaluation and Current Test Evidence

### 8.1 Included Unit Tests

The repository includes a test suite under `tests/` validating:

* Temporal agent parsing and bias detection behavior.
* Basic calculation behaviors.
* Basic verification behaviors.
* Analysis routing and projections (currently closer to demos than strict assertions in places).

At the time this report was written, running `pytest` in the workspace produced:

* 24 passed, 1 skipped (with warnings)

Notably, some tests return values rather than asserting, which is expected to become an error in future pytest versions. This should be fixed before submission as "production-ready."

### 8.2 Evaluation Modules

The repository includes evaluation modules (e.g., `comprehensive_eval.py`) describing metrics like F1 for temporal bias detection, calibration error for uncertainty, and numeric hallucination detection. These are valuable scaffolding, but to be publishable they require:

* A real dataset with labeled ground truth.
* Baselines for comparison.
* A reproducible experiment script and frozen environment.

### 8.3 Recommended Evaluation Plan (Academic)

To make the project publishable, one feasible plan is:

1. Define a dataset of queries with:
   * ticker, analysis_date, query text, expected evidence period, expected numeric outputs when applicable.
2. Run the system in a deterministic configuration (frozen prompt templates, fixed retrieval parameters, fixed models).
3. Evaluate:
   * Temporal leakage rate (percentage of answers citing evidence after the as-of date).
   * Numeric grounding accuracy (for questions with numeric truth).
   * Coverage vs abstention rate (how often system refuses rather than hallucinating).
4. Compare against baselines:
   * Plain LLM (no retrieval).
   * Non-temporal RAG (no PiT filter).
   * Structured-only calculator (no text retrieval).

---

## 9. Production Readiness Assessment

### 9.1 What Is Already "Good"

* Clear separation between retrieval (vector store) and structured numeric paths.
* Core PiT filtering capability exists in the vector store.
* Ingestion automation exists and supports multiple sources.
* Tests exist and provide a foundation for regression protection.

### 9.2 Major Gaps to Production

1. **Single, consistent temporal contract** (as-of semantics) across all components.
2. **Strict data provenance for numbers** (no silent mocks; trace to concept/date/filed).
3. **Robust filing parsing and normalization**, including quarterly/annual separation and units.
4. **Hardening projection logic** so the final answer is computed, constrained, and labeled.
5. **API consolidation** (two backends exist) and consistent response envelope schema.
6. **Operational concerns**: caching, rate limits, retries, incremental indexing, logging, and monitoring.
7. **Security and compliance**: secrets management, audit log integrity, and safe prompt handling.

---

## 10. Conclusion

TemporalGuard-RAG demonstrates a credible approach to financial question answering with a focus on temporal integrity. The system’s strongest differentiator is its attempt to enforce point-in-time constraints at retrieval time and to compute certain metrics from structured XBRL facts rather than from narrative text alone. This addresses a real and important failure mode of generic LLM systems in finance: look-ahead bias and unsupported numeric claims.

To make the project publishable and production-ready, the next iteration should prioritize consistent as-of semantics, strict provenance for numeric answers, stronger normalization of financial statement data, and a real evaluation dataset with baselines. With these improvements, the system can become a meaningful contribution as an applied research prototype for temporally consistent financial RAG.

---

## References (Placeholders)

This repository report is written primarily from the local implementation. For an academic submission, replace this placeholder section with a proper bibliography (BibTeX recommended) and cite:

* SEC EDGAR and XBRL documentation
* ChromaDB documentation
* Foundational RAG literature
* Any 2025-2026 financial QA / temporal RAG papers used in the literature review (see Appendix plan)

---

## Appendix Pointer

See `paper/appendix.md` for:

* Detailed architecture diagrams (Mermaid)
* Data schema details and file layout
* Reproducibility checklist and recommended environment pinning
* A suggested query benchmark set and labeling rubric
* A submission-ready checklist (report + presentation)

