% TemporalGuard-RAG: Point-in-Time Financial Question Answering With Temporal Integrity
% <<Student Name>> (Reg. No: <<Registration Number>>)
% 2026-05-XX

\pagebreak

# Title of the Project

**TemporalGuard-RAG: Point-in-Time Financial Question Answering With Temporal Integrity**

Submitted in partial fulfillment of the requirements for the degree of

**Master of Science**

In

**Data Science**

by

**<<Student Name>>**  
**<<Registration Number>>**

Under the guidance of  
**Dr. <<Internal Guide Name>>** (Internal), VIT  
**<<External Guide Name>>** (External), <<Organization>>

School of Advanced Sciences  
VIT, Vellore.

May, 2026

\pagebreak

# DECLARATION

I hereby declare that the thesis entitled **"TemporalGuard-RAG: Point-in-Time Financial Question Answering With Temporal Integrity"** submitted by me, for the award of the degree of **Master of Science in Data Science** to VIT, is a record of bonafide work carried out by me under the supervision of **Dr. <<Internal Guide Name>> (Internal), VIT** and **<<External Guide Name>> (External), <<Organization>>**.

I further declare that the work reported in this thesis has not been submitted and will not be submitted, either in part or in full, for the award of any other degree or diploma in this institute or any other institute or university.

Place: Vellore  
Date: 2026-05-XX

Signature of the Candidate  
<<Student Name>>

\pagebreak

# CERTIFICATE

This is to certify that the thesis entitled **"TemporalGuard-RAG: Point-in-Time Financial Question Answering With Temporal Integrity"** submitted by **<<Student Name>> (Reg. No.: <<Registration Number>>)**, School of Advanced Sciences, VIT, for the award of the degree of **Master of Science in Data Science**, is a record of bonafide work carried out by him/her under my supervision during the period **<<Internship Start Date>> to <<Internship End Date>>**, as per the VIT code of academic and research ethics.

The contents of this report have not been submitted and will not be submitted either in part or in full, for the award of any other degree or diploma in this institute or any other institute or university. The thesis fulfils the University's requirements and regulations and, in my opinion, meets the necessary standards for submission.

Place: Vellore  
Date: 2026-05-XX

Signature of the Guide (Internal)  
Dr. <<Internal Guide Name>>  
Designation: <<Internal Guide Designation>>  
School of Advanced Sciences, VIT, Vellore

Signature of the Guide (External)  
<<External Guide Name>>  
Designation: <<External Guide Designation>>  
<<Organization>>

\pagebreak

# INTERNSHIP COMPLETION CERTIFICATE

(Attach/insert internship completion certificate issued by the organization. If the organization provides a PDF, it may be inserted here in the final submission version.)

\pagebreak

# ACKNOWLEDGEMENT

With immense pleasure and a deep sense of gratitude, I wish to express my sincere thanks to my guide **Dr. <<Internal Guide Name>> (Internal)**, School of Advanced Sciences, VIT, Vellore and **<<External Guide Name>> (External)**, <<Organization>>, for their continuous guidance, encouragement, and constructive feedback throughout the development of this thesis.

I would also like to thank the faculty members of the School of Advanced Sciences for their support and for providing the academic foundation required to conduct this work. I extend my gratitude to my peers and friends who helped by reviewing the report, discussing ideas, and assisting with testing and evaluation. Finally, I thank my family for their patience and support during the project period.

\pagebreak

# ABSTRACT

Large language models (LLMs) can answer financial questions in natural language, but they frequently fail in settings that require **point-in-time correctness**. In financial research and decision-making settings, a common failure mode is **temporal leakage (look-ahead bias)**, where an answer implicitly uses information that was not yet available at the analysis time. This invalidates backtests and can mislead users by presenting future knowledge as historical fact.

This thesis presents **TemporalGuard-RAG**, a retrieval-augmented question answering system designed for financial queries with an explicit or implicit **as-of analysis date**. The system combines (i) a temporally filtered vector store for retrieving filing excerpts and (ii) a structured numeric computation path using SEC XBRL company facts. A multi-stage orchestrator coordinates temporal validation, evidence retrieval, ratio computation, projection utilities, and optional verification. The system supports question types including historical filing-based questions, financial ratio calculations, and simplified projections.

The primary contribution of this work is an end-to-end prototype demonstrating how point-in-time constraints can be operationalized in a financial QA pipeline. The repository includes ingestion automation for SEC filings and SEC XBRL data, an implementation of PiT retrieval filtering, and an initial evaluation suite based on unit tests and synthetic benchmark scaffolding. Results indicate that temporal constraints reduce the risk of future-data leakage, but the system's correctness depends critically on consistent as-of semantics across retrieval and structured numeric datasets. The report also identifies key limitations and outlines a path to production readiness, including strict provenance tracking, improved statement normalization (annual vs quarterly, units), and a real benchmark dataset with baselines.

Keywords: financial question answering, retrieval augmented generation, point-in-time retrieval, look-ahead bias, SEC filings, XBRL, numeric grounding.

\pagebreak

# TABLE OF CONTENTS

(Table of Contents will be generated automatically when exporting to DOCX/PDF with appropriate tooling. In Word, right-click and update the table.)

\pagebreak

# LIST OF FIGURES

Figure 1: Conceptual pipeline for TemporalGuard-RAG (placeholder)  
Figure 2: Data ingestion and indexing flow (placeholder)  
Figure 3: As-of semantics and temporal filters (placeholder)  
Figure 4: Structured numeric grounding with XBRL (placeholder)  
Figure 5: Projection methods overview (placeholder)  
Figure 6: Evaluation design and benchmark categories (placeholder)

\pagebreak

# LIST OF TABLES

Table 1: Data sources considered and their roles  
Table 2: Metadata fields for PiT retrieval  
Table 3: Supported query types and routing decisions  
Table 4: Core financial ratios and required inputs  
Table 5: Projection methods and assumptions  
Table 6: Proposed evaluation metrics (temporal, numeric, retrieval, projection)  
Table 7: Production readiness checklist

\pagebreak

# LIST OF ABBREVIATIONS

API: Application Programming Interface  
BM25: Best Matching 25 (lexical retrieval scoring)  
CI: Confidence Interval  
DCF: Discounted Cash Flow  
EDGAR: Electronic Data Gathering, Analysis, and Retrieval system (SEC)  
FCF: Free Cash Flow  
IR: Investor Relations  
LLM: Large Language Model  
MD\&A: Management Discussion and Analysis  
PiT / PiT Retrieval: Point-in-Time Retrieval  
RAG: Retrieval-Augmented Generation  
ROA: Return on Assets  
ROE: Return on Equity  
SEC: Securities and Exchange Commission  
TOC: Table of Contents  
XBRL: eXtensible Business Reporting Language

\pagebreak

# CHAPTER 1: INTRODUCTION

## 1.1 Motivation

Financial information is abundant but difficult to query efficiently. Traditional workflows involve reading annual reports, quarterly filings, earnings releases, and other documents. Many questions are easy for humans but time-consuming:

* What risks did management highlight as of a specific year?
* How did revenue, margins, and cash flow evolve over time?
* What was the company's leverage ratio in a specific fiscal year?
* Given historical patterns, what is a reasonable range for next year's operating cash flow?

LLM-based chat systems can answer these questions quickly, but they frequently produce responses with two critical issues:

1. **Temporal inconsistency (look-ahead bias)**: mixing future information into a historical analysis.
2. **Numeric unreliability**: inventing numbers or performing incorrect calculations without traceable evidence.

In financial research, even small leakage errors can cause large downstream distortions. Backtesting research and historical decision simulations rely on strict as-of constraints: the system must act as if it is "living in the past" and must not "know" future filings or outcomes.

## 1.2 Problem Statement

Given a financial question *Q* about a company with ticker *T* and an analysis date *D* (as-of date), build a system that:

* Retrieves relevant evidence **only from documents filed on or before D**,
* Computes numeric metrics from structured sources when possible,
* Produces a natural-language answer that clearly distinguishes between:
  - reported facts,
  - calculated metrics,
  - and forward-looking estimates (projections),
* Provides transparency about data sources, dates, and assumptions,
* Avoids hallucinations and explicitly abstains when evidence is missing.

## 1.3 Objectives

The specific objectives of this thesis are:

1. Build an ingestion pipeline for standardized financial sources (SEC filings and SEC XBRL facts).
2. Implement a retrieval mechanism that enforces point-in-time constraints using filing dates.
3. Implement structured numeric extraction and ratio calculations using XBRL facts.
4. Provide projection utilities (CAGR, linear trend, scenario) for forward-looking queries, clearly labeled as estimates.
5. Implement an evaluation framework and tests that check:
   * temporal leakage prevention,
   * numeric correctness for calculations,
   * reliability and abstention behavior.

## 1.4 Scope and Assumptions

Scope:

* Primary coverage focuses on U.S. public companies that file standardized forms with the SEC.
* The system is designed as a research prototype intended to be extended into a production system.

Assumptions:

* Filing dates and XBRL facts downloaded from the SEC are treated as authoritative.
* For many question types, structured XBRL is sufficient for annual numeric analysis (10-K).
* Quarter-level analysis is possible but requires careful handling of 10-Q and duration/instant concepts.

Out of scope (current version):

* Real-time news sentiment integration (explicitly deferred).
* Full-scale coverage of non-SEC companies and international reporting standards.
* Full compliance with all SEC filing lags by filer category in a complete production ruleset (initial support exists but requires further hardening).

## 1.5 Contribution Summary

The contributions of this thesis are:

* A working codebase implementing PiT retrieval filtering for filing excerpts.
* A structured numeric pipeline using SEC XBRL company facts for metrics and ratios.
* A unified ingestion CLI for collecting multiple data sources for a chosen ticker set.
* A multi-stage orchestration design that routes queries by type.
* A test suite validating key behaviors and acting as a foundation for future evaluation.

## 1.6 Organization of the Thesis

Chapter 2 surveys related work and conceptual background. Chapter 3 describes system architecture. Chapter 4 details data sources and ingestion. Chapter 5 explains temporal enforcement and retrieval. Chapter 6 covers numeric grounding and projections. Chapter 7 describes evaluation and experiments. Chapter 8 discusses findings, limitations, and production-readiness. Chapter 9 concludes and outlines future work.

## 1.7 Use Cases and Stakeholders

TemporalGuard-RAG is designed for scenarios where "as-of" correctness matters. Representative stakeholders include:

Students and educators:

* Students can ask structured questions about filings without accidentally mixing future data into historical analysis.
* Instructors can grade work that depends on point-in-time reasoning.

Researchers:

* Researchers can use the system to generate hypotheses or summaries for a specific time window.
* The system can support reproducible study designs when paired with a benchmark dataset.

Analysts and auditors (prototype context):

* Analysts can quickly locate relevant filing passages while maintaining an audit trail.
* Auditors can review claims and evidence metadata.

For all stakeholders, the key differentiator is traceability: the system should be able to say not only "here is the answer" but also "here is what I used, and here is why it was allowed as of your date."

## 1.8 Ethical Considerations

Financial assistants can influence decisions. Therefore, the system should:

* avoid presenting estimates as facts,
* abstain when evidence is missing,
* include disclaimers for investment-related questions,
* and record provenance metadata so that users can verify claims.

This thesis treats ethical communication as part of technical correctness: a temporally safe system that still overstates confidence can mislead users, even if it avoids future-data leakage.

\pagebreak

# CHAPTER 2: LITERATURE REVIEW

## 2.1 Retrieval-Augmented Generation (RAG)

RAG systems combine two elements: a retriever (search) and a generator (LLM). The retriever returns relevant documents or passages, and the generator uses them to create an answer. RAG is attractive for finance because filings are long and not feasible to read exhaustively for every question.

However, standard RAG does not inherently address:

* **Temporal correctness**: retrieval may return a later filing when answering a question as of an earlier date.
* **Numeric grounding**: even with retrieved evidence, LLMs may produce incorrect calculations.

This thesis focuses on adapting the RAG paradigm to point-in-time finance constraints.

### 2.1.1 Dense vs Sparse Retrieval (Why Both Matter)

Retrieval in RAG is commonly implemented as either:

* **Sparse / lexical retrieval** (e.g., BM25): matches words and phrases; strong when exact terms appear.
* **Dense retrieval** (embeddings): matches semantic meaning; strong when wording differs.

Financial filings contain specialized terms and also many repeated boilerplate phrases. A purely dense approach can retrieve semantically related but non-specific passages; a purely lexical approach can miss relevant phrasing. Many production RAG systems use hybrid strategies such as reciprocal rank fusion (RRF), where both sparse and dense scores contribute.

TemporalGuard-RAG currently uses dense retrieval in the vector store as the default mechanism. A natural extension for production is hybrid dense+sparse retrieval with temporal filtering applied consistently to both retrieval channels.

### 2.1.2 Retrieval Grounding vs "Open-Book" Generation

Even with retrieval, an LLM may:

* ignore retrieved evidence,
* summarize incorrectly,
* or introduce extra facts not present in evidence.

To address this, modern systems use:

* constrained prompting ("use numbers only if present in evidence"),
* strict evidence quoting, or
* structured extraction first, narrative later.

In finance, this issue is especially prominent for numeric statements and for time-anchored claims.

## 2.2 Temporal Reasoning and Point-in-Time Retrieval

Point-in-time retrieval aims to replicate what information would have been available at a past date. In financial databases, point-in-time datasets are well-known to avoid backtest leakage; analogous constraints are required for text retrieval over filings.

Key issues:

* filing dates vs fiscal period end dates,
* differences between annual and quarterly reporting,
* amended filings and restatements,
* and late filings.

TemporalGuard-RAG operationalizes PiT with metadata filters over filing date and adds "period availability" reasoning as a user-facing warning system.

### 2.2.1 Filed Date vs Period End Date (The Core Distinction)

Two dates appear in essentially all finance QA tasks:

* **Period end**: e.g., fiscal year ended 2023-09-30.
* **Filed date**: date the filing becomes publicly available in EDGAR.

If a user asks a question "as of" 2023-11-01, it is incorrect to use a filing filed on 2023-11-15 even if it covers a period ending earlier. PiT retrieval must filter on filed date, while "period availability" warnings must reason from period end + lag to filed date. The separation of these concepts is a central design requirement for leakage-free QA.

### 2.2.2 Filer Status and Reporting Deadlines

In SEC reporting, filing deadlines differ by filer category (e.g., large accelerated vs non-accelerated). A robust temporal system should:

* determine filer status per company,
* apply appropriate filing windows per form type (10-K, 10-Q),
* and treat the filing date from EDGAR as the final authority when available.

TemporalGuard-RAG includes initial logic for deadline maps; production deployment would require consistent use of these deadlines in warnings, and reliance on filed dates for retrieval filtering.

## 2.3 Financial Question Answering and Numeric Faithfulness

Financial QA differs from general QA because:

* Questions often involve time periods and comparisons.
* Answers require numeric precision and traceability.
* Users may rely on answers for high-stakes decisions.

Numeric faithfulness is a known issue for LLMs. Common mitigation approaches include:

* restricting the model to retrieved evidence,
* using structured databases for computation,
* post-hoc verification, and
* explicit abstention policies.

TemporalGuard-RAG combines structured computation (XBRL) with evidence retrieval (filing chunks) to reduce numeric hallucination risk.

### 2.3.1 Why XBRL Matters for Numeric QA

Narrative filings contain numbers, but narrative text is not the safest place to compute from because:

* numbers can appear in multiple contexts (GAAP vs non-GAAP, segment vs total),
* units can differ (millions vs billions),
* and values can be re-stated or revised in later filings.

XBRL provides a more structured basis:

* standardized concept tags (e.g., NetIncomeLoss),
* explicit units,
* explicit periods,
* and direct linkage to form type and filed date.

However, XBRL is still not trivial:

* concept mappings differ by company and by taxonomy versions,
* the same "idea" can appear under multiple tags,
* "instant" vs "duration" facts must not be mixed incorrectly.

Therefore, an XBRL-based system requires careful normalization and metadata tracking to remain correct.

### 2.3.2 Numeric Faithfulness in LLM Outputs

Numeric faithfulness failure modes include:

* invented numbers ("hallucination"),
* incorrect arithmetic,
* using the wrong year or wrong denominator,
* mixing annual and quarterly series.

Common mitigations:

* compute numbers deterministically in code,
* use the LLM only to explain computed results,
* validate output numbers against structured truth when possible,
* and abstain when the system cannot verify.

## 2.4 Forecasting and Uncertainty Communication

Projection tasks are intrinsically uncertain. A key ethical and scientific requirement is to label projections as estimates and to state assumptions. For research credibility, projections should be evaluated by backtesting and error metrics.

This thesis implements basic projection methods and suggests a benchmark design for validation.

## 2.5 Summary of Gaps Addressed

This thesis addresses practical gaps in many LLM-based financial assistants:

* lack of systematic point-in-time enforcement,
* lack of structured numeric grounding for computed metrics,
* lack of explicit answer-type labeling and provenance,
* and limited evaluation practices.

Note: A submission-ready version of this chapter should include properly formatted citations for sources from 2025 to 2026 (journal articles, arXiv papers, and industry reports) and identify explicit statements of open problems from those papers. This can be added once the final bibliography list is selected.

## 2.6 Additional Background: SEC Filings as a Data Modality

SEC filings are long, heterogeneous documents containing:

* policy descriptions (revenue recognition, segment reporting),
* risk factor narratives,
* quantitative tables,
* and legal boilerplate.

From an NLP standpoint, they have specific challenges:

* heavy repetition across years,
* long-range references (a definition early in the filing influences later text),
* and inconsistent formatting in HTML and PDF exhibits.

This motivates chunking and indexing decisions:

* chunk sizes must preserve enough context to avoid misleading excerpts,
* indexing should attach strong metadata (filed date, form type, company),
* retrieval should consider both semantic similarity and lexical anchors (e.g., "Item 1A. Risk Factors").

## 2.7 Related Concepts: Auditability and Reproducibility in Financial AI

Financial AI systems are expected to be auditable:

* users and reviewers should be able to trace any claim back to evidence,
* and any numeric result back to the exact inputs.

Reproducibility requirements include:

* frozen datasets or snapshots,
* fixed retrieval parameters,
* fixed prompting templates,
* and deterministic computation steps.

TemporalGuard-RAG implements early pieces of this by persisting raw and processed artifacts locally and by providing an evaluation harness scaffold. A publishable version requires formalizing these aspects (see Chapter 7 and Chapter 8).

\pagebreak

# CHAPTER 3: SYSTEM DESIGN AND ARCHITECTURE

## 3.1 Design Principles

The design is guided by the following principles:

1. **Temporal integrity first**: no evidence should come from after the analysis date.
2. **Structured numbers for numeric answers**: calculations should use XBRL-derived data when possible.
3. **Separation of concerns**: ingestion, indexing, retrieval, computation, and synthesis are separate stages.
4. **Transparency**: the system should provide date context, warnings, and provenance cues.
5. **Fail safely**: in the absence of evidence, the system should abstain or downgrade confidence.

## 3.2 High-Level Architecture

TemporalGuard-RAG is implemented as a pipeline coordinated by an orchestrator. Conceptually, the pipeline has these stages:

* Stage 1: Temporal validation (understand date constraints and detect temporal pitfalls)
* Stage 2: Retrieval (PiT retrieval of filing passages)
* Stage 3: Calculation (compute ratios and metrics from structured data)
* Stage 4: Optional verification (cross-check claims and numbers)
* Stage 5: Synthesis (produce the final response)

In practice, some query types route to an analysis agent that uses structured data and forecasting utilities directly.

## 3.3 Module-Level View

Key modules:

* Orchestration: coordinates and routes query types.
* Temporal agent: checks date references, computes cutoffs, and emits warnings.
* Vector store: stores embedded text chunks with metadata for filtering.
* Data loader: loads structured financial statement data from XBRL facts and derived metrics.
* Projection engine: implements deterministic forecasting methods.

## 3.4 Query Types and Routing

TemporalGuard-RAG routes queries based on patterns:

Table 3: Supported query types and routing decisions

| Query Type | Examples | Primary Mechanism |
|---|---|---|
| Historical (filing-based) | "Summarize risk factors in FY2022" | PiT retrieval + synthesis |
| Ratio / calculation | "Compute ROE as of FY2023" | Structured XBRL computation |
| Historical extreme / trend | "When was EPS highest?" | Historical analyzer on metric history |
| Projection / forecast | "Project OCF for 2027" | Projection engine + assumptions |

In the current prototype, routing is keyword-based; a production version should use a more robust intent classifier and a constrained output schema.

## 3.4.1 Embeddings and Offline Mode

Financial assistants are often evaluated in restricted environments (classroom labs, internal networks, or offline grading setups). Dense retrieval typically relies on an embedding model. A practical engineering concern is that embedding models can require:

* downloading model weights the first time they are used,
* GPU support for speed,
* and large memory footprints.

TemporalGuard-RAG supports an offline-safe fallback embedding strategy for query embeddings if the default sentence-transformers model cannot be loaded. The offline fallback is intentionally low quality compared to true embeddings, but it has two benefits:

* the system can still run end-to-end in restricted environments, and
* unit tests can run without requiring model downloads.

For research claims about retrieval quality, the system must be evaluated using a real embedding model; the fallback should be treated as a development tool only.

## 3.4.2 LLM Provider Choices and Reproducibility

The system supports:

* a local LLM provider (Ollama), and
* a hosted provider (OpenAI) when an API key is configured.

For reproducibility, it is crucial to record:

* provider name and model identifier,
* temperature and decoding settings,
* and prompt templates.

In academic settings, it is acceptable to use a local model for demonstration if:

* the model is fixed and documented, and
* the evaluation does not rely on vendor-specific hidden model updates.

## 3.4.3 Persistence and Deterministic Re-runs

To support repeated experiments, the pipeline should persist:

* raw filings (immutable),
* extracted XBRL facts (immutable per download date),
* processed chunks and their hashes,
* vector store state and collection metadata.

Deterministic re-runs require:

* fixed chunking rules,
* fixed embedding model and version,
* and fixed retrieval parameters (k, filters).

This thesis treats persistence as a first-class requirement because it is necessary for temporal evaluation: you cannot evaluate leakage if you cannot reconstruct exactly what evidence was used.

## 3.5 Interfaces

The repository includes:

* A Python entry point via `MultiAgentOrchestrator`.
* A FastAPI application for programmatic access.
* A Streamlit UI for interactive use.

Because production systems require a single canonical API, future work should unify interfaces and stabilize response schemas.

\pagebreak

## 3.6 Data Model and Output Envelope

A financial QA system should not return only plain text. For auditability and evaluation, the output should include a structured envelope such as:

* `answer_type`: reported_fact / calculated_metric / model_estimate
* `analysis_date`: the as-of date
* `evidence`: list of document chunks used (with filed dates and identifiers)
* `numbers`: extracted numeric claims (optional)
* `warnings`: temporal warnings, missing evidence, missing structured data
* `assumptions`: projection assumptions (for estimates)

This structure enables:

* rendering consistent UI outputs,
* computing leakage metrics,
* and debugging errors.

The current prototype contains partial versions of these ideas. A production-ready version should treat this envelope as a stable, versioned API contract.

## 3.7 Orchestration and Control Flow (Conceptual Pseudocode)

The core orchestrator behavior can be described as:

1. Determine or infer `analysis_date`.
2. Validate temporal constraints and detect risky future references.
3. Classify the query type:
   * historical, ratio, extreme/trend, projection.
4. Route to the appropriate pipeline:
   * retrieval-based pipeline for qualitative/historical questions,
   * structured-analysis pipeline for ratios/projections.
5. Synthesize a final response with provenance.

Pseudocode:

```
process_query(query, ticker, analysis_date):
  D = analysis_date or infer_date(query)
  temporal = temporal_validate(query, D)
  qtype = classify(query)

  if qtype in {ratio, trend, projection}:
    analysis = run_structured_analysis(query, ticker, D)
    return envelope(answer=analysis.text, answer_type=qtype, warnings=temporal.warnings)

  cutoff = D  # recommended semantics: filed_date <= D
  docs = retrieve_documents(query, ticker, cutoff)
  if docs.empty:
    return envelope(answer="insufficient evidence", answer_type="reported_fact", warnings=["no docs"])

  answer = synthesize(query, docs, temporal)
  return envelope(answer=answer, evidence=docs.metadata, warnings=temporal.warnings)
```

This formulation highlights the key design rule: the retrieval cutoff should be tied to the as-of date (filed-date filter), while filing lags should be used for warnings and interpretation rather than excluding already-filed documents.

## 3.8 Security, Safety, and Abuse Considerations

Even in academic prototypes, financial QA systems should consider:

* Prompt injection in retrieved text (filings can contain adversarial strings, especially if the dataset includes external sources).
* PII and secrets handling (API keys in environment variables; avoid logging secrets).
* Output disclaimers for forecasts and investment advice.

TemporalGuard-RAG contains early audit logging scaffolding and can be extended to include:

* safe prompt templates,
* input validation and rate limiting,
* and redaction for logged data.

\pagebreak

# CHAPTER 4: DATA COLLECTION AND PREPROCESSING

## 4.1 Data Sources

Table 1: Data sources considered and their roles

| Source | Type | Role in System | Strengths | Limitations |
|---|---|---|---|---|
| SEC EDGAR filings | Unstructured text | qualitative answers, citations | standardized, authoritative | large, noisy HTML |
| SEC XBRL company facts | Structured | numeric grounding, ratios | standardized numeric facts | requires concept mapping |
| Yahoo Finance | Semi-structured | market context, convenience | easy access, broad coverage | reliability and licensing constraints |
| Investor relations pages | Unstructured | extra official docs | official sources | non-standard per company |

This thesis emphasizes SEC and XBRL as the most auditable and scalable foundation.

## 4.2 SEC Filing Downloads

The system uses an EDGAR downloader to fetch filings (10-K, 10-Q, optionally 8-K). Filings are stored locally with associated metadata so that they can be processed offline and re-indexed as needed.

Operational constraints:

* Rate limits and politeness requirements.
* Occasional download failures or missing filings for specific tickers.

## 4.3 SEC XBRL Company Facts

The SEC provides a company facts API containing structured financial statement values. The system downloads this JSON and extracts key metrics into a local CSV representation. This enables:

* fast metric history retrieval,
* deterministic ratio calculations,
* and numeric cross-checking.

### 4.3.1 From Raw Facts to Usable Time Series

Raw XBRL company facts are organized by concept tags. For each concept, there may be many entries:

* different fiscal periods (annual, quarterly),
* amendments,
* multiple units,
* and multiple filings across time.

To build a usable time series, the pipeline must:

1. select the desired unit (usually USD for monetary concepts),
2. select the desired form types (10-K for annual, optionally 10-Q for quarterly),
3. select a representative value for each fiscal period,
4. record the filed date and form used for the selected value.

In this thesis prototype, a simplified extraction strategy is used for initial functionality. For production readiness, this extraction policy should be formalized and tested against known cases, including restatements and amended filings.

### 4.3.2 Data Versioning

XBRL facts can change over time due to:

* restatements,
* amended filings,
* and taxonomy updates.

For point-in-time research, it is not sufficient to store only the latest version of facts. A robust approach is:

* store the raw facts JSON as downloaded with a timestamp,
* store derived metrics with a "built_at" timestamp,
* and treat those artifacts as immutable snapshots.

This thesis treats versioning as a recommended enhancement; the prototype is designed to be extended in this direction.

## 4.4 Chunking and Embedding

Unstructured filings are chunked into text passages and embedded into vectors for semantic search. Each chunk includes metadata such as:

* ticker
* filing type
* filed date
* source path
* fiscal year / fiscal period (when extractable)

Table 2: Metadata fields for PiT retrieval

| Field | Description | Used For |
|---|---|---|
| ticker | company ticker | company filtering |
| filing_type | 10-K / 10-Q / etc | query targeting |
| filing_date | filed date as YYYYMMDD int | PiT cutoff filter |
| source_path | local file path | provenance / debugging |
| chunk_hash | hash of chunk text | integrity checks |

## 4.5 Scaling Considerations

For a full-universe system:

* ingestion must be incremental and resumable,
* embeddings must be computed efficiently (batching and caching),
* and storage must be managed to avoid redundant copies.

TemporalGuard-RAG includes a unified ingestion script and persistent vector storage, but additional work is required for industrial-scale runs (Section 8).

## 4.6 Practical Handling of PDFs at Scale

In real corporate IR repositories, many documents are PDFs. For thousands of companies, manual download and manual extraction are infeasible. A scalable approach typically includes:

1. **Discovery**: find candidate URLs using a constrained crawler that targets official IR pages.
2. **Download**: store files with stable naming and metadata (URL, timestamp, hash).
3. **Text extraction**:
   * first attempt: extract embedded text layer,
   * fallback: OCR only when necessary (costly).
4. **Normalization**:
   * remove headers/footers and repeated page artifacts,
   * preserve page numbers for citations if possible.
5. **Chunking and indexing** with metadata:
   * source type = "IR PDF",
   * company,
   * publication date (if available),
   * download date,
   * and document hash.

TemporalGuard-RAG's IR collector is intentionally conservative because websites are not standardized. The thesis focuses on SEC sources for scalability; however, the same ingestion principles apply to PDFs when an IR extension is required.

## 4.7 Data Quality Controls

To be production-grade, ingestion must include quality checks such as:

* validate that ticker-to-CIK mapping is correct,
* detect empty/failed downloads,
* verify that filed dates are present and parseable,
* deduplicate documents via content hash,
* and record provenance metadata for auditability.

Without these controls, downstream retrieval can appear to work but silently mix incorrect companies or incorrect dates, which is catastrophic for point-in-time tasks.

## 4.8 Storage and Indexing Strategy

A scalable design separates:

* **raw storage** (immutable artifacts),
* **processed storage** (clean text, chunks),
* **index storage** (vector database, lexical index),
* and **cache storage** (query embeddings, frequently used computations).

This separation reduces recomputation and makes re-indexing feasible when chunking or embedding configurations change.

## 4.9 Legal and Ethical Considerations for Data Collection

Even though SEC filings are public, a research system should:

* identify itself via user-agent where required,
* respect rate limits,
* store only the minimum necessary data for research,
* and document the sources and intended usage.

For IR pages, ethical crawling requires stricter caution:

* obey robots policies where applicable,
* throttle requests,
* and avoid aggressive scraping at large scale.


\pagebreak

# CHAPTER 5: TEMPORAL ENFORCEMENT AND RETRIEVAL

## 5.1 As-Of Semantics

The most important concept in this thesis is the analysis date **D** ("as of" date). The system should behave as if it is only allowed to use information that existed by D.

Recommended contract:

* Retrieval eligibility: `filed_date <= analysis_date`
* Period availability warning: `period_end + filing_lag <= analysis_date`

It is essential not to conflate these, because filing lags are about when a period is reported, while PiT filtering is about when the report is filed and available.

## 5.2 Temporal Filtering in the Vector Store

The vector store uses metadata filters to exclude filings after the cutoff. This prevents retrieving future filings (a common leakage vector).

## 5.3 Temporal Validation Agent

The temporal validation stage checks:

* whether the query references future years beyond the analysis date,
* whether it requests a period not yet available,
* and whether it uses forward-looking language that should be labeled as an estimate rather than a fact.

The temporal stage is designed to output warnings rather than block all forward-looking queries, because forecasts are allowed as long as they are explicitly labeled and use only historical inputs.

## 5.4 Common Failure Modes

Failure modes in temporal QA systems include:

* Using a later 10-K to answer an earlier-year question.
* Using annual results that were not yet filed as of an analysis date.
* Using revised/restated values without tracking versioning.

Mitigations include:

* strict filed-date filtering,
* emitting warnings for unavailable periods,
* and explicit provenance reporting.

## 5.5 Worked Examples of Temporal Leakage

### Example 1: Annual Filing Not Yet Available

Suppose the user asks:

* Query: "What was Apple's fiscal 2023 revenue?"
* As-of analysis date: 2023-10-15

Even though the fiscal year may have ended near late September (company-specific), the 10-K for that fiscal year would not typically be filed by mid-October. A safe system should:

* warn that the FY2023 10-K is likely not yet available,
* retrieve only documents filed on/before 2023-10-15,
* and either abstain or answer using the latest available period (clearly labeled).

### Example 2: Retrieval Leaks Due to Missing Metadata

If a filing chunk lacks a valid `filing_date` metadata field, a vector store might treat it as eligible for all queries. This can leak future filings. Therefore:

* all chunks must include filed date, and
* the retrieval filter must exclude chunks with missing/invalid dates by default.

### Example 3: Forward-Looking Language Presented as Fact

Queries that include words like "will" and "expected" may be legitimate (projections) but must be labeled. A safe answer should:

* separate "reported facts" from "model estimates,"
* list the historical inputs used,
* and state the uncertainty and assumptions.

## 5.6 Filing Lags: Warnings vs Enforcement

This thesis recommends that filing lag logic should primarily drive:

* warnings ("this quarter's filing may not yet be available as of your date"),
* and suggestion ("use earlier quarter or move analysis date forward").

But the enforcement mechanism for retrieval should be:

* strict filed-date filter (filed_date <= analysis_date).

This keeps the system consistent and avoids excluding documents that were in fact filed before the as-of date.

## 5.7 Temporal Integrity Metrics (Operationalization)

To measure temporal integrity, the system should record:

* the analysis date,
* the list of evidence chunks,
* and each chunk's filed date.

Then leakage can be computed deterministically:

* leakage occurs if any evidence filed_date > analysis_date.

This metric can be reported per query and aggregated across benchmarks. It is a key differentiator relative to generic LLM systems where leakage is not measurable without provenance.

## 5.8 Period Availability Function (Recommended)

To communicate what information is likely available as-of a date, a recommended helper function is:

* Input: analysis_date D, filing_type F (10-K/10-Q), filer_status S (optional)
* Output: latest period_end P such that P + lag(F,S) <= D

This allows the UI to display:

* "As of 2023-08-15, Q2 2023 results are likely available for accelerated filers."

But importantly, it does not override filed-date filtering. It is an interpretability aid and can reduce user confusion when they ask for periods not yet filed.

## 5.9 Handling Amended Filings

Amended filings (e.g., 10-K/A) introduce a subtlety:

* As-of a given date, the amendment might not exist, but the original 10-K might.

Therefore, for PiT, the correct behavior is:

* include any filing (including amendments) only if filed on/before analysis_date,
* but prefer the most recent filing on/before analysis_date for a given period when summarizing.

This suggests a retrieval preference policy:

* prefer later filed dates within the eligible set,
* but never cross the as-of boundary.



\pagebreak

# CHAPTER 6: NUMERIC GROUNDING, CALCULATIONS, AND PROJECTIONS

## 6.1 Ratio Computation

The system computes standard ratios from structured data. Examples include:

Table 4: Core financial ratios and required inputs

| Ratio | Formula | Required Inputs |
|---|---|---|
| ROE | Net Income / Stockholders' Equity | NetIncome, StockholdersEquity |
| ROA | Net Income / Total Assets | NetIncome, Assets |
| Profit Margin | Net Income / Revenue | NetIncome, Revenue |
| Debt Ratio | Total Liabilities / Total Assets | Liabilities, Assets |
| Current Ratio | Current Assets / Current Liabilities | CurrentAssets, CurrentLiabilities |

The key benefit of using XBRL-derived values is that the system can show inputs and avoid narrative-number guessing.

### 6.1.1 Statement Normalization Challenges

To compute ratios correctly, the system must normalize:

* **annual vs quarterly** values,
* **instant vs duration** facts (e.g., assets at a point in time vs revenue over a period),
* **units** (USD, shares, USD/shares),
* and **taxonomy variations** (different tags for similar concepts).

For example:

* Revenue is usually a duration concept (start and end dates).
* Assets and Equity are instant concepts (end date only).

If these are mixed incorrectly (e.g., quarterly net income with annual equity), ratios can become nonsensical. A production-grade implementation should:

* enforce compatible period types,
* select consistent time windows,
* and record which facts were used.

### 6.1.2 Provenance for Computed Numbers

For each computed ratio, provenance should include:

* the concept names used (e.g., NetIncomeLoss, StockholdersEquity),
* the period end dates of those facts,
* the filed dates,
* the form (10-K or 10-Q),
* and the exact numeric inputs.

This enables auditability and also allows reviewers to detect mistakes in concept selection.

### 6.1.3 Worked Example: ROE Calculation With Provenance (Illustrative)

This example illustrates the information that should be shown for a ratio, even if the exact values differ by company.

Goal: compute ROE for ticker `AAPL` for fiscal year 2022.

1. Select annual net income for FY2022:
   * Concept: NetIncomeLoss (or ProfitLoss depending on taxonomy)
   * Period end: 2022-09-24 (example fiscal year end for illustration)
   * Form: 10-K
   * Filed date: 2022-10-XX
   * Value: $N (USD)
2. Select stockholders' equity at the same period end:
   * Concept: StockholdersEquity
   * Period end: 2022-09-24
   * Form: 10-K
   * Filed date: 2022-10-XX
   * Value: $E (USD)
3. Compute:
   * ROE = N / E
4. Report:
   * ROE (as percent) = (N / E) * 100
   * Provenance: both concepts, period end, form, and filed date

The key point is that the system should not merely output "ROE = 39%." It should show the inputs and the source metadata so that the computation can be verified.

## 6.2 Projection Methods

The system implements deterministic projection methods:

Table 5: Projection methods and assumptions

| Method | Description | Strengths | Weaknesses |
|---|---|---|---|
| CAGR | projects based on compound annual growth | simple, interpretable | unstable if values volatile |
| Linear trend | regression on time vs value | captures trend direction | sensitive to outliers |
| Scenario projection | bull/base/bear growth rates | communicates uncertainty | subjective scenario rates |

In a publishable system, the LLM should not invent projection numbers. Instead, the projection engine should compute results and the LLM should explain them.

### 6.2.1 CAGR Projection (Formula)

If the most recent historical value is \(V_T\) at year \(T\), and the earliest value is \(V_0\) at year \(0\), the CAGR is:

\[
\mathrm{CAGR} = \left(\frac{V_T}{V_0}\right)^{\frac{1}{T-0}} - 1
\]

Then a projection to year \(T+k\) is:

\[
\hat{V}_{T+k} = V_T \cdot (1+\mathrm{CAGR})^k
\]

In practice, financial series can include negative or near-zero values. Therefore the projection engine should:

* avoid CAGR when \(V_0 \le 0\),
* fall back to linear trends or scenario methods,
* and emit a LOW confidence label when data is insufficient.

### 6.2.2 Linear Regression Projection (Formula)

For yearly data points \((x_i, y_i)\) where \(x_i\) is year and \(y_i\) is value:

* Fit \(y = ax + b\) via least squares.
* Project \(\hat{y}(x^*) = a x^* + b\).

Confidence can be approximated using \(R^2\) and residual variance. This is not a full probabilistic forecast but provides a defensible trend estimate.

### 6.2.3 Scenario Projections (Interpretability)

Scenario projections are useful for communicating uncertainty:

* bull case: optimistic growth assumptions,
* base case: central estimate,
* bear case: pessimistic assumptions.

For an academic report, scenario rates should be:

* tied to historical volatility where possible, and
* explicitly documented.

## 6.3 DCF Valuation (Simplified)

DCF is implemented as a simplified illustrative tool. A production version would require:

* more robust WACC estimation,
* explicit modeling of reinvestment needs,
* careful handling of capital structure,
* and full disclosure of assumptions.

The system includes DCF primarily to demonstrate an extensible "analysis" layer.

### 6.3.1 DCF Core Equation

Simplified enterprise value:

\[
EV = \sum_{t=1}^{N} \frac{FCF_t}{(1+r)^t} + \frac{TV}{(1+r)^N}
\]

Terminal value (Gordon growth):

\[
TV = \frac{FCF_{N+1}}{r-g}
\]

Where:

* \(r\) is a discount rate (approximate WACC),
* \(g\) is terminal growth,
* and \(FCF_t\) is free cash flow for year \(t\).

In a production-grade valuation tool, \(r\) and \(g\) must be chosen carefully, and the system should not present a single number without sensitivity analysis.

### 6.3.2 Ethical Warning for Valuation Outputs

Valuation outputs can be misused as investment advice. A responsible system should:

* include disclaimers,
* present ranges and sensitivities,
* and encourage users to verify assumptions.


\pagebreak

# CHAPTER 7: EXPERIMENTS AND EVALUATION

## 7.1 Evaluation Goals

The goal of evaluation is to quantify whether the system:

* prevents temporal leakage,
* produces numerically grounded outputs for calculations,
* retrieves relevant evidence for qualitative questions,
* and communicates uncertainty for projections.

## 7.2 Test Suite Evidence

The repository includes unit tests that validate:

* temporal date parsing and lag logic,
* bias detection warnings,
* basic computation behaviors,
* and query routing.

These tests provide engineering confidence but are not a substitute for an academic benchmark dataset.

### 7.2.1 Test Run Snapshot (Engineering Evidence)

As an engineering checkpoint, the test suite was executed in the development workspace on:

* Date: 2026-04-17
* Outcome: 24 tests passed, 1 skipped

This indicates basic internal consistency for the behaviors covered by unit tests (temporal parsing, routing, and selected calculations). It does not validate real-world accuracy for arbitrary companies and questions; it primarily reduces regressions during development.

### 7.2.2 Known Test Gaps

Current tests are limited because:

* some tests are written like demos (printing outputs) rather than strict assertions,
* most tests do not validate numeric outputs against a ground-truth dataset,
* retrieval tests do not validate that the retrieved chunks are correct filings,
* and projection tests do not evaluate forecast error against realized future data.

These limitations motivate the benchmark design in Section 7.3.

## 7.3 Proposed Benchmark Dataset

To make the project publishable, a benchmark dataset should include:

* a fixed set of companies,
* a set of as-of dates,
* queries labeled by type,
* expected evidence windows,
* and numeric ground truth for select questions.

### 7.3.1 Query Taxonomy for Benchmarking

To cover the system's supported tasks, benchmark queries can be grouped as:

* **Qualitative filing questions**:
  - risk factors for a given filing.
  - major accounting policies.
  - management discussion highlights.
* **Numeric fact questions**:
  - revenue, net income, assets for a specific fiscal year.
* **Computed metric questions**:
  - ROE, ROA, profit margins as-of a year.
* **Temporal trap questions**:
  - query explicitly references a future year relative to analysis date.
* **Projection backtests**:
  - forecast year \(t+1\) using information as-of year \(t\), then compare to realized values.

### 7.3.2 Ground Truth Construction

Ground truth for numeric facts and computed metrics can be derived from:

* SEC XBRL company facts (annual 10-K for consistent comparisons),
* with a stable selection policy for which concept tag to use when multiple exist.

Qualitative ground truth is more subjective and often requires:

* labeled relevant passages,
* or human evaluation rubrics.

## 7.4 Metrics

Table 6: Proposed evaluation metrics

| Category | Metric | Description |
|---|---|---|
| Temporal | Leakage rate | % of evidence after analysis date |
| Temporal | Warning accuracy | correctness of "period not available" warnings |
| Numeric | Relative error | error vs XBRL truth for calculations |
| Retrieval | Precision@k | relevance of retrieved chunks |
| Projection | MAPE / sMAPE | forecast accuracy via backtesting |
| System | Abstention rate | % of cases the system refuses due to missing evidence |

## 7.5 Error Analysis (Qualitative)

Observed risks include:

* wrong concept mappings in XBRL,
* mixing annual and quarterly facts,
* unit inconsistencies,
* and LLM-driven narrative producing implausible values unless constrained.

This thesis recommends constraining final answers to computed values and tracked evidence fields.

## 7.6 Ablation Study Plan

To demonstrate the value of each component, an ablation study can compare:

1. LLM only (no retrieval, no XBRL).
2. Non-temporal RAG (retrieval without filed-date filtering).
3. Temporal RAG (PiT filtering).
4. Temporal RAG + structured calculations (XBRL).
5. Temporal RAG + calculations + verification.

For each ablation, evaluate:

* leakage rate,
* numeric error,
* abstention vs hallucination,
* and qualitative groundedness.

This makes the thesis argument concrete: each added module should measurably reduce a failure mode.

## 7.7 Performance and Latency Considerations

Latency is important in interactive systems. The end-to-end latency depends on:

* vector search speed (ChromaDB),
* LLM response time (OpenAI or local Ollama),
* and any network calls (SEC, Yahoo Finance).

A production system should:

* cache embeddings and frequent computations,
* precompute chunk embeddings offline,
* and implement timeouts and fallbacks for external calls.

In an academic evaluation, report:

* average latency per query type,
* and worst-case latencies under network failures.

## 7.8 Threats to Validity

Several threats can limit the validity of evaluation results:

Internal validity:

* Ingestion errors (wrong ticker/CIK mapping) can contaminate both retrieval and ground truth if not detected.
* If the benchmark uses the same source artifacts for both retrieval and evaluation, it may overestimate real-world performance.

External validity:

* Results measured on a small ticker set may not generalize to small-cap firms, foreign issuers, or firms with atypical fiscal calendars.
* Quarterly reporting behavior differs across companies and can break naive assumptions.

Construct validity:

* "Groundedness" for qualitative answers can be subjective; human rubrics must be clearly defined.
* Leakage detection requires evidence metadata; if evidence is not recorded, leakage cannot be measured reliably.

This thesis addresses these threats by recommending provenance-first output envelopes and a benchmark schema that records enough metadata to evaluate temporal integrity deterministically.

\pagebreak

# CHAPTER 8: DISCUSSION AND PRODUCTION READINESS

## 8.1 What Is Novel Here?

While multi-agent architectures are common, the novelty of this work is primarily:

* applying PiT constraints to filing retrieval,
* integrating structured XBRL numeric grounding,
* and building an end-to-end workflow with a clear as-of semantics goal.

## 8.2 How Is This Different From ChatGPT?

ChatGPT (or a general LLM) typically:

* does not guarantee it is using only "as-of" information,
* may answer without citing filings,
* may invent numbers or compute incorrectly.

TemporalGuard-RAG aims to be distinct by:

* enforcing retrieval cutoffs by filed date,
* using structured numeric inputs for computations,
* and explicitly labeling answers and warnings.

## 8.3 Does This Solve a Real Problem?

Yes. Temporal leakage is a well-known problem in financial research and backtesting. A system that answers questions "as of" a date and can show evidence addresses a real workflow gap for:

* academic case studies,
* internal audit trails,
* educational use,
* and prototype decision-support systems.

## 8.4 Production-Readiness Gaps

Table 7: Production readiness checklist (summary)

| Area | Current Status | Needed for Production |
|---|---|---|
| As-of semantics | partially implemented | unify contract across all modules |
| Provenance | partial | strict trace to filed facts/chunks |
| Mock data | exists | disable by default or explicit opt-in |
| Evaluation | unit tests | real dataset + baselines + reproducibility |
| API | multiple entrypoints | unify and version response schema |
| Ingestion | CLI exists | incremental indexing + job control |
| Security | basic | secrets management + audit integrity |

## 8.5 Limitations (Honest Assessment)

This thesis prototype has several important limitations:

1. **Evaluation is incomplete**: without a real benchmark dataset and baselines, claims about accuracy and leakage reduction remain partially qualitative.
2. **XBRL concept mapping is fragile**: companies can report similar metrics under different tags; a robust mapping strategy is required.
3. **Annual-only bias**: if the loader filters to 10-K forms only, quarter-level questions may be answered incorrectly or with insufficient evidence.
4. **LLM narrative drift**: if the LLM is allowed to produce numbers not computed in code, projections and even factual answers can become unreliable.
5. **Multiple entrypoints**: having multiple API/backends increases maintenance and can cause inconsistent behavior across deployments.

These limitations do not invalidate the system's central idea, but they define the boundary between a prototype and a production-grade system.

## 8.6 Risk Analysis

Key risks if deployed without additional guardrails:

* Users may interpret forecasts as facts.
* Numbers may be miscomputed due to period mismatches (quarter vs year).
* Evidence may be missing, but the system might still answer without abstaining.
* Ingestion errors (wrong ticker/CIK) can silently contaminate results.

Mitigations:

* strict abstention policy when evidence is missing,
* mandatory provenance for numeric outputs,
* and systematic end-to-end leakage checks.

## 8.7 Recommended Minimal Product (MVP) Definition

Given limited time, a defensible MVP is:

* reliable PiT retrieval for filing text (qualitative questions),
* reliable annual numeric facts (revenue, net income, assets, equity) from XBRL,
* correct computed annual ratios (ROE, ROA, margins) with provenance,
* and forecasts limited to deterministic projection outputs plus disclaimers.

This MVP is distinct from generic LLMs because it is explicitly evidence-based and time-aware.


\pagebreak

# CHAPTER 9: CONCLUSION AND FUTURE WORK

## 9.1 Conclusion

This thesis implemented TemporalGuard-RAG, a prototype system for financial question answering with point-in-time integrity. The system combines temporally filtered retrieval over filing text with structured numeric grounding via SEC XBRL. The prototype demonstrates that PiT constraints can be implemented in a practical pipeline and provides a codebase foundation for further research and productization.

## 9.2 Future Work

High-impact future work includes:

* unify as-of semantics (filed-date filtering and lag warnings),
* improve XBRL normalization (annual vs quarterly; units; concept selection),
* add strict provenance reporting for all numeric outputs,
* build a benchmark dataset and evaluate against baselines,
* add deterministic "compute then explain" constraints for projections,
* and unify API/UI entrypoints for stable usage.

\pagebreak

# REFERENCES

Note: Update this section to match the required citation style (IEEE/APA) and include 2025-2026 papers used for the final literature review.

1. U.S. Securities and Exchange Commission (SEC), EDGAR and XBRL Company Facts documentation.  
2. ChromaDB documentation (vector store with metadata filters).  
3. Foundational RAG surveys and papers (to be added).  
4. Financial QA and numeric faithfulness literature (to be added).  
5. Point-in-time data and backtesting leakage references (to be added).

\pagebreak

# APPENDIX A: REPRODUCIBILITY AND RUN INSTRUCTIONS

## A.1 Repository Structure (High-Level)

* `src/` contains pipeline modules and agents.
* `scripts/ingest_all.py` provides unified ingestion.
* `tests/` provides unit tests.
* `api.py` and `backend/main.py` provide API entrypoints.

## A.2 Example Commands

Ingestion (small run):

* Download SEC filings and XBRL facts for a small ticker set.
* Build chunks and index into vector store (if chunking scripts are configured).
* Run queries through the orchestrator.

Testing:

* Run `pytest` to validate regression checks.

### A.2.1 Suggested "Small, Reproducible" Demo Setup

For a reproducible academic demonstration, select a small fixed ticker set (e.g., 3 to 5 companies) and do:

1. Download SEC XBRL company facts for the chosen tickers.
2. Extract a fixed set of annual metrics and persist them to `*_metrics.csv`.
3. Download a small number of 10-K filings (e.g., last 3 years).
4. Chunk and embed the filings and build a local vector store.
5. Run a fixed set of queries with a fixed analysis date.

This yields an offline-capable snapshot that can be repeated during evaluation and grading.

### A.2.2 Output Recording for the Report

For each benchmark query, store:

* query text and ticker,
* analysis date,
* answer type and warnings,
* evidence identifiers and their filed dates,
* computed numeric inputs and outputs.

This is required to compute leakage metrics and to support manual audit in the thesis appendices.

## A.3 Ethical and Legal Considerations

This project uses public filings and official company sites. In a production deployment, ensure:

* compliance with SEC rate limits and fair use,
* adherence to website robots policies for IR pages,
* correct user-agent identification,
* and protection of any private API keys.

\pagebreak

# APPENDIX B: PROJECT TIMELINE (APRIL 17, 2026 TO MAY 6, 2026)

Given a submission/presentation date of **May 6, 2026**, a realistic plan is:

Week 1 (Apr 17-Apr 23):

* Unify temporal semantics across modules.
* Disable mock data by default or make it explicit.
* Fix tests that return values instead of asserting.

Week 2 (Apr 24-Apr 30):

* Create a small benchmark dataset (10-30 queries).
* Run baseline comparisons (LLM only vs non-temporal RAG vs TemporalGuard-RAG).
* Create tables/plots for results.

Week 3 (May 1-May 6):

* Finalize report formatting to template.
* Prepare presentation slides and a short demo.
* Re-run experiments for final numbers.

\pagebreak

# APPENDIX C: EXAMPLE QUERIES AND EXPECTED SAFE BEHAVIOR

This appendix provides example queries and describes the expected safe behavior of TemporalGuard-RAG. These examples can be used directly in the final presentation demo.

## C.1 Historical Filing Query (Qualitative)

Query: "Summarize the main risk factors for AAPL in the annual report as of fiscal 2022."  
Expected safe behavior:

* Retrieve risk factor passages from 10-K filings filed on or before the analysis date.
* Provide evidence metadata (filing type, filed date).
* Do not introduce new claims not present in evidence.
* If no relevant evidence exists, abstain and explain what is missing.

## C.2 Numeric Fact Query (Structured)

Query: "What was MSFT net income in fiscal 2021?"  
Expected safe behavior:

* Extract from XBRL facts for the correct fiscal year.
* Return a single value with units.
* Provide provenance: concept tag, period end, filed date, and form.

## C.3 Ratio Computation Query

Query: "Compute ROE for JPM as of fiscal 2022."  
Expected safe behavior:

* Compute from annual NetIncome and StockholdersEquity for the same year.
* Present formula, inputs, and result.
* If equity is missing or zero, abstain with a clear explanation.

## C.4 Temporal Trap Query

Query: "Compare NVDA 2025 revenue to 2024, as of 2024-01-15."  
Expected safe behavior:

* Warn that 2025 revenue is a future reference relative to the analysis date.
* Retrieve only filings filed on or before 2024-01-15.
* Suggest alternatives: change analysis date or restrict to available periods.

## C.5 Projection Backtest Query

Query: "As of 2022-12-31, project AAPL OperatingCashFlow for 2023."  
Expected safe behavior:

* Use only data available as of 2022-12-31 (by filed dates).
* Output a deterministic projection (CAGR or linear) with a confidence label.
* Provide assumptions and avoid presenting the estimate as a reported fact.

\pagebreak

# APPENDIX D: RECOMMENDED API RESPONSE SCHEMA

For stable integrations (UI, evaluation harnesses, and audit), a recommended response schema is:

```
{
  "query": "...",
  "ticker": "AAPL",
  "analysis_date": "YYYYMMDD",
  "answer_type": "reported_fact|calculated_metric|model_estimate",
  "answer": "...",
  "warnings": ["..."],
  "evidence": [
    {
      "chunk_id": "...",
      "ticker": "AAPL",
      "filing_type": "10-K",
      "filed_date": "YYYYMMDD",
      "source_path": "..."
    }
  ],
  "numbers": [
    {
      "name": "NetIncome",
      "value": 0,
      "unit": "USD",
      "period_end": "YYYY-MM-DD",
      "filed_date": "YYYY-MM-DD",
      "concept": "NetIncomeLoss",
      "form": "10-K"
    }
  ],
  "assumptions": {
    "method": "CAGR|Linear|Scenario",
    "notes": "..."
  }
}
```

This schema is intentionally redundant so that evaluation (leakage checks, numeric checks) and manual auditing are straightforward.

\pagebreak

# APPENDIX E: DETAILED IMPLEMENTATION WALKTHROUGH (TEMPORALGUARD-RAG)

This appendix provides a detailed walkthrough of the repository implementation so that the thesis can be evaluated and reproduced without requiring the reader to inspect the full codebase.

## E.1 Core Question: What Happens End-to-End?

When a user asks a question (for example, "Project cashflow for Apple for FY 2026-27"), the system performs an end-to-end workflow that can be summarized as:

1. Interpret the user request:
   * Identify the company (ticker).
   * Identify whether the user is asking for history, a calculation, or a forecast.
   * Determine the analysis date if provided (as-of date).
2. Collect data for answering:
   * For qualitative/history: retrieve filing excerpts from the vector store.
   * For numeric/calculation: load XBRL-derived metrics and compute formulas.
   * For projections: load historical metric series and apply deterministic projection methods.
3. Generate a final response:
   * Present a concise answer.
   * Provide key supporting points.
   * Emit warnings if evidence is missing or if the query is temporally risky.

The central promise is: for any answer that claims to be "as of" a past date, the system attempts to avoid using filings or facts that were not yet filed by that date.

## E.2 Orchestrator: Stage Coordination and Routing

The orchestrator is responsible for choosing a pipeline. It performs:

* analysis date inference when none is provided,
* query type classification,
* execution of the retrieval/calculation/analysis stages,
* and packaging of results.

In a production system, this module defines the "contract" for what a query means and what an answer must include.

Important design note:

* For point-in-time correctness, the orchestrator should treat the analysis date as an "evidence eligibility cutoff" by filed date.
* Filing lag logic should be used to warn about period availability, not to exclude eligible evidence.

## E.3 Temporal Enforcement: Validation and Warnings

The temporal component performs several distinct tasks:

1. Date parsing:
   * accept formats like YYYYMMDD or YYYY-MM-DD.
2. Cutoff analysis:
   * compute which documents could be filed by the as-of date.
3. Bias detection:
   * detect references to future years or quarters relative to the analysis date.
4. Warnings:
   * label forward-looking language and unavailable periods.

A key reason to implement a separate temporal stage is interpretability:

* the system can explain what it can and cannot know as of a date,
* rather than silently failing or leaking future information.

## E.4 Vector Store and Retrieval Layer

### E.4.1 What Is Stored?

The retrieval layer stores "chunks" of filings. Each chunk typically includes:

* chunk text (unstructured),
* an embedding vector,
* and metadata.

The most important metadata for temporal integrity is the filed date:

* filed_date in YYYYMMDD form (stored as an integer for comparisons).

### E.4.2 How Retrieval Works

Given:

* query text Q,
* ticker T,
* cutoff date D (as-of),

the store returns the top-k chunks that are:

* semantically similar to Q, and
* satisfy the temporal predicate filed_date <= D, and
* satisfy ticker and (optionally) filing type filters.

If the metadata is correct, this is a powerful anti-leakage control because it blocks later filings regardless of their semantic relevance.

### E.4.3 Offline Embeddings

The repository includes a deterministic, offline-safe embedding fallback for query embeddings. This is useful for:

* restricted environments with no model downloads,
* unit testing,
* and consistent execution when the embedding model is unavailable.

However, retrieval quality in offline mode will be significantly worse than real embeddings. For any research claim about retrieval quality, offline embeddings should not be used.

## E.5 Structured Data Layer: XBRL Facts and Derived Metrics

### E.5.1 Why Structured Data Exists

Numeric questions cannot safely depend on text extraction alone. Structured data exists to provide:

* authoritative values,
* consistent units,
* and machine-readable periods.

### E.5.2 Metric Histories

The data loader exposes metric histories such as:

* Revenue by year,
* Net income by year,
* Assets and equity by year,
* Operating cash flow by year.

For each year, a robust representation should include:

* value,
* period end,
* filed date,
* form type,
* and concept tag.

This enables point-in-time enforcement for numeric answers in the same way retrieval enforces PiT for text.

### E.5.3 Common Pitfall: Period Mismatch

Many ratios combine:

* duration values (revenue, net income over a year), and
* instant values (assets/equity at year-end).

To avoid errors, the system should ensure:

* the instant value corresponds to the end of the same period as the duration value,
* or explicitly uses an averaging strategy (e.g., average equity) when appropriate.

## E.6 Calculation Layer: Deterministic Financial Ratios

The ratio calculator should:

1. select appropriate input metrics,
2. validate that the inputs exist and are compatible,
3. compute ratios deterministically,
4. format outputs with formula and provenance,
5. refuse or downgrade confidence if data is missing.

A strict policy is important:

* If the system cannot fetch a reliable denominator, it should not invent one.

For example, if equity is missing, ROE should be abstained rather than guessed.

## E.7 Projection Layer: Deterministic Forecasts With Labels

Projections are allowed in this thesis, but only as **model estimates**. A robust projection response must include:

* method chosen (CAGR or linear),
* historical inputs used (years and values),
* forecast output and optional range,
* explicit assumptions,
* and warnings about uncertainty.

The LLM should not create new numeric forecasts; it should be constrained to explain deterministic computations.

## E.8 Verification Layer (Optional)

A verification stage can:

* check whether key claims appear in evidence chunks,
* cross-check numeric claims against structured XBRL values,
* and produce a confidence label.

In this prototype, verification is intentionally lightweight. A publishable system would require:

* structured claim extraction (numbers + units + periods),
* deterministic matching of claims to XBRL facts,
* and a quantified verification metric on a benchmark dataset.

## E.9 Case Study: Why the System Previously Failed on Projections

The reported issue ("project cashflow for Apple for FY 2026-27" produced an error) is representative of a common gap in RAG systems:

* RAG is optimized for retrieving and summarizing existing text.
* A projection question requires computation and modeling, not retrieval alone.

Adding a projection-capable analysis component is a valid extension, but it introduces new requirements:

* explicit labeling of estimates,
* deterministic computation,
* and evaluation via backtesting.

This thesis treats projections as a bounded feature rather than a solved forecasting problem.

## E.10 What Makes the Work "Publishable" (Concrete Criteria)

For an academic audience, "publishable" does not require a commercial product, but it does require:

1. A clearly defined problem and threat model:
   * what constitutes leakage and how it is detected.
2. A method:
   * PiT retrieval filter and structured numeric grounding.
3. A reproducible evaluation:
   * a fixed dataset of queries and evidence snapshots,
   * baselines,
   * and quantitative metrics.
4. Transparent limitations:
   * conditions under which the method fails.

TemporalGuard-RAG already contains the architectural foundation. The remaining work is primarily in:

* dataset construction,
* measurement and baselines,
* and hardening the as-of contract across all numeric and textual paths.

## E.11 Repository Walkthrough (Folders and Their Purpose)

This section summarizes how the repository is organized and what artifacts are expected in each directory. This is useful both for graders and for future maintainers.

Top-level:

* `README.md`: project overview and quick start instructions.
* `requirements.txt`: Python dependencies (prototype-level; production should pin versions).
* `api.py`: FastAPI app entrypoint (one of the API options).
* `streamlit_app.py`: Streamlit UI entrypoint.
* `backend/`: an additional FastAPI backend (should be unified in production).
* `src/`: core implementation (agents, retrieval, analysis, ingestion, evaluation).
* `scripts/`: command-line utilities; includes unified ingestion.
* `tests/`: unit tests.
* `paper/`: thesis/report artifacts (this document and exports).

Data directories:

* `data/raw/sec_filings/`: downloaded SEC filings.
* `data/raw/xbrl_structured/`: XBRL company facts JSON and derived metrics CSV.
* `data/processed/`: processed chunks, embeddings, and vector store persistence.

Important artifact expectations:

* If `*_facts.json` exists for a ticker, structured analysis can run for that ticker.
* If filing chunks have been embedded and loaded into the vector store, retrieval-based questions can cite filing passages.

Common reasons a query fails in prototypes:

* missing ingestion for the ticker,
* missing filed dates in metadata (breaks PiT filtering),
* missing embedding model (offline environment),
* or incomplete concept mappings for a requested metric.

By making this structure explicit in the thesis, the reader can distinguish "system logic failure" from "missing data artifact."

\pagebreak

# APPENDIX F: EXTENDED RELATED WORK AND GAP ANALYSIS (NARRATIVE)

This appendix expands the literature review in Chapter 2 into a longer narrative that can be adapted into a formal, citation-complete related work section. The content below is written in a thesis tone but intentionally uses placeholder citation markers (e.g., [R1], [R2]) to be replaced with the final selected bibliography.

## F.1 Financial Text as a Retrieval Corpus

SEC filings are a high-value corpus for financial NLP because they are:

* standardized at the form level (10-K, 10-Q),
* legally significant and carefully reviewed,
* and persistently accessible through EDGAR.

However, they are not "clean text":

* documents can be extremely long,
* they include tables, exhibit references, and boilerplate sections,
* and the structure differs across issuers and across time.

Prior work in financial NLP often emphasizes preprocessing choices:

* HTML parsing vs raw extraction,
* section segmentation (Item 1A Risk Factors, Item 7 MD&A),
* and the handling of tables and numeric columns.

For RAG systems, the retrieval corpus quality directly impacts answer quality. If the corpus is noisy or incorrectly segmented, retrieval produces misleading evidence and the generator may hallucinate or misinterpret context.

Gap addressed by this thesis:

* The thesis focuses on metadata-rich chunking and explicit temporal filtering, which is often missing in generic RAG pipelines built for general web text.

## F.2 Point-in-Time (PiT) Data and Backtest Integrity

In quantitative finance, point-in-time datasets are considered essential for valid backtests. Many commercial data vendors provide PiT datasets precisely because:

* it is easy to accidentally use revised or future-reported data,
* and such leakage can materially overstate performance.

In text-based systems, a similar problem arises:

* the assistant may retrieve a later filing, or later restatement, when answering a question about an earlier time.

Research on temporal information retrieval and temporal QA highlights that time-aware systems must:

* represent document time (published/available time),
* represent event time (what the document is about),
* and enforce constraints depending on the task.

Gap addressed by this thesis:

* The project operationalizes "available time" as filed date and uses it as a retrieval constraint, turning a qualitative idea ("don't use future filings") into a measurable mechanism.

## F.3 Temporal QA vs Temporal RAG

Temporal QA is often studied on datasets where:

* questions explicitly mention dates,
* and answers depend on time-scoped facts.

Financial temporal RAG is more specific:

* the key time variable is often the as-of analysis date,
* and the evidence must be constrained to what would have been filed by that date.

Moreover, the finance domain introduces:

* filing lags,
* amended filings,
* and the distinction between period end and filed date.

Gap addressed by this thesis:

* The thesis frames temporal integrity as a first-class constraint for the retrieval layer, rather than as an after-the-fact reasoning prompt to the LLM.

## F.4 Numeric Faithfulness and "Math" in LLM Systems

Multiple strands of literature discuss numeric errors in LLM outputs:

* hallucinated numbers,
* incorrect arithmetic,
* and incorrect aggregation of values across time.

Mitigation approaches include:

* tool use (calculator calls, database queries),
* constrained generation (only output numbers from a trusted table),
* and verification models that detect mismatches.

In finance, numeric correctness is non-negotiable for many tasks. A plausible-sounding wrong number can be more dangerous than an abstention. Therefore, a finance assistant should implement a conservative policy:

* compute deterministically when possible,
* provide provenance for numeric inputs,
* and abstain when evidence is missing.

Gap addressed by this thesis:

* The project integrates an XBRL structured data path, which is more appropriate for numeric questions than relying on narrative text.

## F.5 Verification and Auditing in Retrieval Systems

Verification can be performed at multiple levels:

* evidence existence: did retrieval produce any relevant sources?
* textual support: do retrieved passages contain the claim?
* numeric support: do structured facts match the numeric claim?
* temporal support: are sources available as-of the analysis date?

Auditability requires that the system not only compute a confidence label, but also expose:

* which sources were checked,
* what was matched,
* and what could not be verified.

Gap addressed by this thesis:

* The project emphasizes logging and evidence metadata, enabling deterministic leakage checks and review workflows.

## F.6 Evaluation Methodology Gaps in Many Prototypes

Many application prototypes report qualitative demos but lack:

* fixed benchmark datasets,
* baselines,
* reproducible run scripts,
* and robust metrics beyond general user satisfaction.

For publishability in an applied research context, it is critical to show:

* leakage metrics (before vs after temporal constraints),
* numeric accuracy for computable queries,
* and a clear abstention policy (avoiding hallucination).

Gap addressed by this thesis:

* The repository includes a unit test foundation and evaluation scaffolding; the thesis proposes a complete benchmark design as the next step.

## F.7 Summary: The Niche for TemporalGuard-RAG

TemporalGuard-RAG targets a specific niche:

* financial QA where "as-of" correctness matters.

The key gap it fills relative to generic LLMs and generic RAG systems is:

* an enforceable, measurable point-in-time constraint applied at retrieval time,
* combined with a structured numeric grounding path for calculations.

The remaining publishability work is primarily empirical:

* build the benchmark,
* run ablations and baselines,
* and report results with reproducibility artifacts.

\pagebreak

# APPENDIX G: BENCHMARK SPECIFICATION AND ANNOTATION GUIDE

This appendix describes a concrete benchmark specification that can be implemented within the project timeline and used to produce publishable evaluation results. The benchmark is designed to measure the specific claims of this thesis:

* point-in-time integrity (no leakage),
* numeric grounding correctness,
* and safe abstention behavior.

## G.1 Benchmark Data Model

Each benchmark item should be stored as a single JSON object with fields:

* `id`: unique identifier
* `ticker`: company identifier
* `analysis_date`: as-of date (YYYYMMDD)
* `query`: natural language question
* `task_type`: historical_qualitative | numeric_fact | ratio | projection_backtest | temporal_trap
* `expected_behavior`: one of:
  - answer_with_evidence,
  - abstain_due_to_missing_evidence,
  - warn_future_reference,
  - warn_period_unavailable
* `ground_truth` (optional):
  - numeric value(s) for numeric_fact and ratio tasks
  - acceptable tolerance band
* `notes`:
  - human annotation notes
  - relevant filing identifiers if known

This schema allows both automated and human evaluation.

## G.2 Query Templates (Coverage)

To systematically generate queries, define templates such as:

Historical qualitative:

* "Summarize the main risk factors for {TICKER} in fiscal {YEAR} as of {ASOF_DATE}."
* "What did management highlight in the MD&A for {TICKER} in fiscal {YEAR}?"

Numeric fact:

* "What was {TICKER} revenue in fiscal {YEAR}?"
* "What was {TICKER} net income in fiscal {YEAR}?"

Ratio:

* "Compute ROE for {TICKER} for fiscal {YEAR}."
* "Compute debt ratio for {TICKER} for fiscal {YEAR}."

Temporal trap:

* "Compare {TICKER} {FUTURE_YEAR} revenue to {PAST_YEAR}, as of {ASOF_DATE}."
* "What will {TICKER} revenue be in {FUTURE_YEAR}, as of {ASOF_DATE}?"

Projection backtest:

* "As of {ASOF_DATE}, project {TICKER} OperatingCashFlow for {NEXT_YEAR}."

This templating approach ensures benchmark breadth without requiring fully manual query writing.

## G.3 Ground Truth Strategy

Ground truth should be constructed with a clear policy:

1. Use SEC XBRL annual 10-K facts as the default ground truth for annual metrics.
2. Select a stable concept mapping policy (e.g., prefer Revenues, else fallback to RevenueFromContract...).
3. For ratios, compute ground truth deterministically from those facts.
4. Store not only the value but also:
   * concept used,
   * period end,
   * filed date,
   * and form.

This ensures the benchmark itself is point-in-time compatible: you can check that a ground truth value corresponds to a filing that would exist by a specific as-of date.

## G.4 Temporal Leakage Evaluation (Automatic)

Leakage evaluation can be automated if the system records evidence metadata.

Given:

* analysis_date D,
* evidence list E where each evidence item has filed_date f,

Leakage predicate:

* leak = any(f > D)

Report:

* leakage_per_query: True/False
* leakage_rate: mean across benchmark

This is a key advantage of retrieval-based temporal enforcement: it makes leakage measurable.

## G.5 Numeric Accuracy Evaluation (Automatic)

For numeric tasks:

1. Extract numeric outputs from the system response envelope (preferred) or by regex (fallback).
2. Compare to ground truth:
   * absolute error,
   * relative error,
   * pass/fail under a tolerance band.

Recommendation:

* Store numeric outputs in structured form to avoid brittle regex extraction.

## G.6 Abstention and Safety Evaluation

The benchmark should include cases where evidence is intentionally unavailable, for example:

* request a fiscal year that has not yet been filed as of the analysis date,
* request a company not included in the ingested dataset.

Expected safe behavior:

* abstain (do not hallucinate),
* and explain missing evidence.

Metrics:

* abstention_correct_rate: fraction of "should abstain" cases where the system abstains,
* hallucination_rate: fraction of "should abstain" cases where the system answers anyway.

## G.7 Human Evaluation Rubric (Qualitative Questions)

For qualitative questions (risk factors, MD&A summaries), automatic ground truth is harder. A human rubric can rate:

1. Evidence use:
   * 0 = no evidence cited
   * 1 = evidence cited but not relevant
   * 2 = evidence cited and relevant
2. Temporal integrity:
   * 0 = evidence after as-of date or future references presented as facts
   * 1 = warnings present but still ambiguous
   * 2 = clear as-of framing and safe evidence
3. Faithfulness:
   * 0 = introduces claims not supported by evidence
   * 1 = minor ungrounded details
   * 2 = faithful summary of evidence

A benchmark can include 20 to 50 qualitative items and be rated by 2 reviewers for inter-rater agreement.

## G.8 Baselines (Minimum Required)

To support a publishable claim, compare against:

* Baseline A: LLM-only (no retrieval, no structured data).
* Baseline B: RAG without PiT filtering (retrieval ignores filed date).
* TemporalGuard-RAG: PiT filtering + structured numeric path.

If time permits, add:

* Structured-only baseline: XBRL computation without text retrieval for numeric questions.

The most important comparison is Baseline B vs TemporalGuard-RAG for leakage reduction.

## G.9 Reporting Results in the Thesis

A submission-ready report should include:

* leakage rate comparison plot/table,
* numeric accuracy table (mean error, pass rates),
* abstention/hallucination table,
* and 3 to 5 case studies showing:
  - a failure case of baseline,
  - the corrected behavior under TemporalGuard-RAG,
  - and evidence metadata supporting the difference.

This appendix can be converted into a "Methodology and Evaluation" chapter in a publishable paper.

\pagebreak

# APPENDIX H: PRODUCTIONIZATION PLAN AND REQUIREMENTS (DETAILED)

This appendix provides a detailed plan for turning the thesis prototype into a production-ready system. While productionization is out of scope for an academic submission, presenting a concrete plan strengthens the thesis by demonstrating feasibility and by clarifying what remains unfinished.

## H.1 Functional Requirements

Minimum functional requirements for a production point-in-time financial assistant:

1. Accept queries with:
   * ticker,
   * analysis date (optional, but recommended),
   * and a question text.
2. Return answers with:
   * clear answer type labeling,
   * evidence citations for qualitative claims,
   * and deterministic numeric computations for ratios and facts.
3. Enforce point-in-time integrity:
   * never cite evidence filed after the analysis date.
4. Provide safe failure behavior:
   * abstain when evidence is missing,
   * avoid hallucinated numeric values,
   * and present projections strictly as estimates.

## H.2 Non-Functional Requirements

Non-functional requirements are often the difference between a demo and a product:

* Reliability: predictable outputs, resilient to missing data.
* Observability: logs, audit trails, and debugging metadata.
* Performance: interactive latency for typical queries.
* Security: safe handling of API keys, protection against prompt injection.
* Reproducibility: versioned data snapshots and deterministic computations.

## H.3 Data Engineering Requirements

To support thousands of companies:

* ingestion must be incremental and resumable,
* re-indexing must be possible without re-downloading raw artifacts,
* and storage must be deduplicated.

Recommended approach:

1. Store raw artifacts as immutable objects with content hashes.
2. Maintain a manifest database linking tickers to artifacts and metadata.
3. Run scheduled jobs:
   * fetch new filings,
   * update XBRL facts,
   * chunk and embed new text,
   * and update indexes.

## H.4 Retrieval Quality Requirements

Temporal filtering alone does not guarantee that the retrieved evidence is relevant. A production system should:

* incorporate hybrid retrieval (dense + sparse),
* implement section-aware indexing (risk factors vs MD&A),
* and use re-ranking (cross-encoder or LLM-based) within the eligible evidence set.

Critically, all retrieval enhancements must still respect the same temporal filter.

## H.5 Numeric Grounding Requirements

For numeric questions, production quality requires:

* a canonical concept mapping library,
* unit conversions,
* handling of amended filings and restatements,
* and explicit rules for annual vs quarterly selection.

A practical strategy is to:

* support annual-only first (10-K), where the dataset is simpler,
* then add quarterly support with rigorous tests and careful duration/instant handling.

## H.6 Projection Requirements

Forecasts introduce reputational and ethical risk. A production system should:

* limit forecasts to deterministic methods with known behavior,
* include uncertainty bands and disclaimers,
* and support backtesting evaluation as a built-in feature.

It should also prevent the LLM from inventing forecast numbers by:

* restricting numeric outputs to computed values,
* and treating the LLM only as an explanation generator.

## H.7 Prompt Injection Threat Model

RAG systems are vulnerable to prompt injection because retrieved text is passed into the model context. In finance, filings are typically trustworthy, but a system may ingest:

* IR web pages,
* transcripts,
* or third-party datasets,

which can contain adversarial text. Threats include:

* instructions embedded in retrieved text that attempt to override system prompts,
* "data poisoning" by inserting misleading passages into the corpus.

Mitigations include:

* explicit prompt injection filtering ("ignore instructions from documents"),
* provenance restrictions (only official sources),
* anomaly detection in embeddings (poisoning detection),
* and sandboxing of any tool calls.

## H.8 Engineering Plan (Milestones)

Milestone 1: Temporal contract unification (1 week)

* Standardize as-of semantics across retrieval and structured data.
* Implement leakage checks in the response envelope.

Milestone 2: Numeric provenance and strict abstention (1 to 2 weeks)

* Disable mock data by default.
* Require provenance metadata for numeric answers.

Milestone 3: Benchmark and baselines (2 weeks)

* Build dataset, run ablations, generate plots/tables.

Milestone 4: Retrieval improvements (2 weeks)

* Hybrid search, re-ranking, section-aware chunking.

Milestone 5: Production API hardening (2 weeks)

* Unify API entrypoints, version schema, add observability.

This plan is realistic for an extended project beyond the thesis submission date.

## H.9 Deployment Options (Practical)

Even for prototypes, it is useful to clarify how the system can be deployed:

Local single-user mode:

* Run ingestion for a small ticker set.
* Run Streamlit UI locally for interactive questions.
* Best for demonstrations and grading.

Local service mode:

* Run a single FastAPI backend process.
* Use a persisted vector store on local disk.
* Allows a lightweight web UI or programmatic clients.

Team / lab mode:

* Centralize the vector store and raw artifacts on shared storage.
* Run ingestion as a scheduled job.
* Provide a stable API endpoint with authentication.

Cloud mode (future):

* Store raw filings in object storage.
* Store embeddings in a managed vector database.
* Use an LLM provider with strict logging and key management.
* Implement cost controls for LLM calls.

For the thesis submission timeline, local single-user mode is sufficient. The additional modes are included to show a clear path from prototype to product.

\pagebreak

# APPENDIX I: EXTENDED CASE STUDIES (QUALITATIVE)

This appendix provides longer case studies that can be used in the final report and presentation. The intent is to show how the system behaves, what errors are prevented, and what limitations remain.

## I.1 Case Study 1: Historical Risk Factors With Point-in-Time Constraint

Scenario:

* Company: AAPL
* Query: "As of 2020-06-01, what were Apple's main risk factors?"

Why this is a good test:

* It is time-scoped.
* Risk factors evolve year-to-year (e.g., supply chain disruptions, regulatory changes).
* A naive system might cite a later filing (e.g., from 2021 or 2022) and incorrectly attribute it to 2020.

Expected safe workflow:

1. Temporal validation:
   * interpret as-of date 2020-06-01.
   * detect whether the query requests a period not available as-of that date.
2. Retrieval:
   * retrieve chunks from filings filed on/before 2020-06-01.
   * prefer 10-K risk factor sections, but do not allow later filings.
3. Answer synthesis:
   * summarize only retrieved evidence.
   * include filing dates in citations or metadata.
4. Warnings:
   * if evidence set is small or missing, warn and abstain.

What this demonstrates:

* The system can prevent "future risk factor" leakage.
* The system's quality depends on whether filings were ingested and chunked correctly.

Limitations:

* If the vector store is missing filings for that period, the system must abstain.
* If chunking breaks section boundaries, retrieval may return incomplete context.

## I.2 Case Study 2: Ratio Calculation With Structured Data

Scenario:

* Company: MSFT
* Query: "Compute ROE for fiscal 2022. Show the inputs."

Why this is a good test:

* It requires numeric correctness.
* ROE is sensitive to the denominator choice (equity can be defined in multiple ways).

Expected safe workflow:

1. Structured data lookup:
   * fetch NetIncome for FY2022 from XBRL facts.
   * fetch StockholdersEquity at FY2022 period end.
2. Validate:
   * confirm both values exist and are compatible periods.
   * confirm denominator not zero.
3. Compute:
   * ROE = NetIncome / StockholdersEquity.
4. Report:
   * formula,
   * inputs with provenance metadata,
   * computed ROE as percent,
   * and any caveats (e.g., equity definition used).

What this demonstrates:

* Structured computation reduces hallucination risk.
* Provenance allows a reviewer to verify the number.

Limitations:

* If the system's concept mapping selects an unexpected tag, the ratio may differ from a textbook computation.
* A production system should define a canonical ROE computation policy (e.g., average equity vs end equity).

## I.3 Case Study 3: Projection Query and "Estimate vs Fact"

Scenario:

* Company: JPM
* Query: "As of 2022-12-31, project OperatingCashFlow for 2024."

Why this is a good test:

* It is forward-looking and therefore inherently uncertain.
* It forces the system to distinguish estimates from reported numbers.

Expected safe workflow:

1. Temporal validation:
   * treat 2022-12-31 as the information boundary.
2. Data selection:
   * use only historical operating cash flow values from filings filed on/before 2022-12-31.
3. Forecast computation:
   * compute a deterministic projection (CAGR and/or linear).
   * produce a range if available (confidence interval or scenario band).
4. Reporting:
   * label answer_type = model_estimate.
   * include the method, inputs, and assumptions.
   * avoid language that implies certainty.

What this demonstrates:

* A finance assistant can support planning-style questions while staying honest.
* The system remains distinct from generic LLM chat because it ties the estimate to explicit historical inputs and a time boundary.

Limitations:

* Without a backtesting dataset, the forecast cannot be validated quantitatively.
* Economic regimes change; a simple trend model can fail dramatically.

## I.4 How to Use These Case Studies in the Presentation

For a short demo (5 to 7 minutes), select:

* one qualitative PiT question (risk factors),
* one numeric ratio question (ROE),
* one projection question (OCF forecast).

For each, show:

* the query and analysis date,
* the system answer,
* the evidence filed dates,
* and a 1-sentence interpretation of why PiT matters.

\pagebreak

# APPENDIX J: FORMULA COMPENDIUM AND METRIC DEFINITIONS

This appendix lists core financial formulas and provides metric definitions as used in the thesis. The goal is to clarify exactly what the system means when it computes or projects a metric.

## J.1 Definitions (Annual, Unless Specified)

Revenue:

* Total revenue recognized over the fiscal year.
* XBRL concepts may include `Revenues` or `RevenueFromContractWithCustomerExcludingAssessedTax`.

Net Income:

* Profit after taxes attributable to shareholders over the fiscal year.
* XBRL concepts may include `NetIncomeLoss` or `ProfitLoss`.

Total Assets:

* Balance sheet total assets at period end.
* XBRL concept: `Assets`.

Total Liabilities:

* Balance sheet total liabilities at period end.
* XBRL concept: `Liabilities`.

Stockholders' Equity:

* Equity at period end.
* XBRL concept: `StockholdersEquity` (may have variants including noncontrolling interest).

Operating Cash Flow (OCF):

* Net cash provided by operating activities over the fiscal year.
* XBRL concept: `NetCashProvidedByUsedInOperatingActivities`.

Capital Expenditures (CapEx):

* Cash spent to acquire property, plant, and equipment.
* XBRL concept: `PaymentsToAcquirePropertyPlantAndEquipment`.

Free Cash Flow (FCF):

* In this thesis prototype, computed as:
  - FCF = OCF - |CapEx|
* Note: Many practitioners compute FCF with additional adjustments; this thesis uses a simplified definition for consistency and transparency.

## J.2 Ratio Formulas

ROE (Return on Equity):

* ROE = Net Income / Stockholders' Equity
* Output unit: percent

ROA (Return on Assets):

* ROA = Net Income / Total Assets
* Output unit: percent

Net Profit Margin:

* Net Margin = Net Income / Revenue
* Output unit: percent

Debt Ratio:

* Debt Ratio = Total Liabilities / Total Assets
* Output unit: ratio (or percent when multiplied by 100)

Current Ratio:

* Current Ratio = Current Assets / Current Liabilities
* Output unit: ratio

## J.3 Projection Methods (Recap)

CAGR:

* CAGR = (V_last / V_first)^(1/years) - 1
* Projection: V_hat = V_last * (1 + CAGR)^k

Linear trend:

* Fit y = ax + b
* Project: y_hat(x*) = a x* + b

Scenario:

* Choose base, bull, bear growth rates (ideally tied to historical volatility)
* Project separately under each scenario

## J.4 XBRL Concept Mapping (Illustrative)

The table below provides an illustrative mapping used by the structured loader. In practice, a production system should maintain a versioned mapping table and test it across a representative set of companies.

| Standard Metric | Example XBRL Concepts (Candidates) |
|---|---|
| Revenue | Revenues; RevenueFromContractWithCustomerExcludingAssessedTax; SalesRevenueNet |
| NetIncome | NetIncomeLoss; ProfitLoss |
| Assets | Assets |
| Liabilities | Liabilities |
| Equity | StockholdersEquity; StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest |
| OCF | NetCashProvidedByUsedInOperatingActivities |
| CapEx | PaymentsToAcquirePropertyPlantAndEquipment |

This appendix can be referenced whenever the thesis discusses "using structured data" so that readers have an explicit definition of what is being computed.

\pagebreak

# APPENDIX K: GLOSSARY (TERMS USED IN THIS THESIS)

As-of date (analysis date):

* The date at which the analysis is assumed to occur. The system should behave as if no information after this date exists.

Available time:

* The time when a document becomes accessible to the analyst, typically the filing date in EDGAR for SEC forms.

Event time:

* The time that the content describes (e.g., fiscal quarter end). Event time can be earlier than availability.

Chunk:

* A short passage of a long document used for retrieval. Chunking is a practical compromise between context size and retrieval granularity.

Embedding:

* A numeric vector representation of text used for semantic similarity search.

Vector store:

* A database that stores embeddings and associated documents/metadata and supports similarity search.

Point-in-time (PiT) retrieval:

* Retrieval constrained to documents that would have been available at a specific as-of time.

Look-ahead bias / temporal leakage:

* Using information that occurs after the as-of date in an analysis that claims to be anchored at that date.

Ground truth:

* The authoritative reference values used for evaluation. In this thesis, annual numeric ground truth is derived from SEC XBRL company facts.

Abstention:

* A safe behavior where the system refuses to answer because evidence or required inputs are missing.

Numeric grounding:

* The practice of producing numbers from a trusted structured source rather than generating them from an LLM's internal knowledge.

Verification:

* Checking whether a claim is supported by retrieved text or structured data, and producing a confidence label.

Backtesting:

* Evaluating forecasts or decision policies using historical data while preventing leakage.

This glossary is included to help non-technical reviewers interpret terms consistently across chapters.
