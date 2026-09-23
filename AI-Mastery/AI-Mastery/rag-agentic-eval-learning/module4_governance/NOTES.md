# Module 4: Enterprise Safety, Privacy & Governance

## 1. The Enterprise Problem
Even a technically-accurate AI system can violate compliance, privacy, or
access-control rules: leaking a customer's SSN into a log, letting a junior
employee's chatbot query fetch payroll data because a retriever didn't
respect permissions, or letting a malicious/off-topic prompt through to
production. These are legal and trust failures, not just quality failures.

## 2. The Engineering Solution

### 2.1 PII Masking / Redaction
Before any text (user input or retrieved document) reaches an external LLM
API, run it through a redaction layer that detects and masks patterns like
credit card numbers, SSNs, phone numbers, and names — via regex for
structured patterns (fast, deterministic) and/or an NER model for names
(context-dependent). This is directly analogous to `redact_pii` operators
you'd add to an ETL pipeline before data lands in a shared bucket
(referenced in Module 1's `runbook_4`).

### 2.2 Access Controls (ACL / RBAC Integration)
The retrieval step (Module 1) must filter by the *querying user's*
permissions, not just semantic similarity. Concretely: tag every chunk in
the vector DB with metadata (e.g. `required_role: "finance"`), and apply a
metadata filter *before or during* the vector search so a junior employee's
query can never even retrieve a payroll chunk — regardless of how similar
it is semantically. This mirrors FGAC on a Datalake bucket: the access
check happens at the data layer, not as an afterthought on the output.

### 2.3 Guardrails: Input/Output Moderation
A lightweight classifier (rule-based, or a small model) sits in front of the
LLM call and blocks toxic, off-topic, or policy-violating prompts before
they consume LLM cycles, and similarly scans the output before it's shown
to the user. Cheap and fast because it runs before/after the expensive LLM
call, not instead of it.

## 3. The Evaluation Metric
- **PII leak rate**: % of test documents containing planted synthetic PII
  that make it through ungasked (target: 0%).
- **RBAC violation rate**: % of test queries, run as a low-privilege user,
  that successfully retrieve a restricted-role chunk (target: 0%).
- **Moderation precision/recall**: on a labeled set of benign vs.
  policy-violating prompts, how many violating ones are correctly blocked
  (recall) without blocking too many benign ones (precision/false-positive
  rate).

## Run it
```bash
python module4_governance/governance_guardrails.py
```
Demonstrates: (1) regex-based PII redaction on a sample support ticket,
(2) RBAC-filtered retrieval where a "junior" role query never sees a
"finance"-tagged chunk even though it's the best semantic match, and
(3) a simple keyword-based input moderation gate.
