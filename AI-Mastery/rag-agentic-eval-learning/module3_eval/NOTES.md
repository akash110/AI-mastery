# Module 3: LLM Evaluation & Validation

## 1. The Enterprise Problem
Before you let an AI answer customer or oncall questions, you need
*mathematical* evidence it's not making things up, actually answers what
was asked, and pulled from the right source docs — not vibes-based "looks
good to me" testing. This is the equivalent of data quality checks
(row counts, null checks, schema validation) but for natural-language
outputs.

## 2. The Engineering Solution

### 2.1 The RAG Triad
Three independent metrics, each catching a different failure mode:

- **Context Relevance**: did retrieval fetch documents that actually relate
  to the query? (Failure here = a retrieval/chunking problem, Module 1.)
- **Groundedness / Faithfulness**: is every claim in the answer traceable
  back to the retrieved context, with nothing invented? (Failure here =
  hallucination even when retrieval was fine — the model "went rogue".)
- **Answer Relevance**: does the answer actually address the user's
  specific question, or is it a correct-but-irrelevant tangent?

A system can fail any one of these independently — e.g. perfect retrieval
+ perfectly grounded answer that never actually answers the question asked.

### 2.2 LLM-as-a-Judge
Since grading "is this grounded?" for thousands of Q&A pairs by hand doesn't
scale, use a second (often larger/more capable) LLM with a strict rubric
prompt to grade each dimension automatically, e.g.:

> "Given this CONTEXT and this ANSWER, output a score 1-5 for whether every
> factual claim in the ANSWER is supported by the CONTEXT. Output JSON only."

Frameworks that formalize this: **Ragas**, **TruLens**, and (in a
ServiceNow context) the **Now Assist Data Kit**. This demo implements the
same idea from scratch with a local model as the judge, so you see exactly
what those frameworks are doing under the hood.

Caveat worth knowing: an LLM judge is itself an LLM — it can be biased,
inconsistent, or gamed by verbose answers. Spot-check judge outputs against
human judgment periodically; don't treat judge scores as ground truth.

## 3. The Evaluation Metric
This module *is* the metric layer for the other three modules. The
deliverable is a repeatable score you can track over time/releases:
`context_relevance`, `groundedness`, `answer_relevance` — each 1-5, plus a
pass/fail threshold (e.g. groundedness < 3 blocks deployment).

## Run it
```bash
python module3_eval/evaluate_rag.py
```
Runs 3 example (query, context, answer) triples — one clean, one subtly
hallucinated, one off-topic — through an LLM-as-judge rubric for all three
RAG-triad metrics, using the same local Ollama model as judge.
