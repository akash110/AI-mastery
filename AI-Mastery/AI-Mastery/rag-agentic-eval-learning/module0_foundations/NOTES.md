# Module 0: The AI Landscape — Evolution, Drawbacks, and Why RAG/Agents/Eval Exist

Read this before Module 1. It answers "why does any of this exist?" — every
later module is a patch for a specific, historically real failure covered
here.

---

## 0.1 A Brief Evolution of AI (so the jargon has a timeline)

| Era | ~Years | What it was | Core limitation |
|---|---|---|---|
| **Symbolic AI / Expert Systems** | 1950s–1980s | Hand-coded rules ("if fever AND rash then possible measles"). Think: a giant nested if/else tree written by humans. | Didn't scale — every new scenario needed a human to write a new rule. Couldn't handle ambiguity or novel inputs. |
| **Statistical Machine Learning** | 1990s–2010s | Models learn patterns from labeled data (spam filters, recommendation engines, fraud detection) — logistic regression, decision trees, SVMs, gradient boosting (XGBoost). | Needed hand-engineered features (a human still decided what "signals" mattered). Each model solved one narrow task. |
| **Deep Learning** | 2012–2017 | Neural networks learn features themselves from raw data (images, audio, text) — ImageNet (2012) was the "AlexNet moment" for vision. RNNs/LSTMs for sequential text. | Sequential text models (RNNs) processed one word at a time — slow, and struggled with long-range context ("what does 'it' refer to 40 words ago?"). |
| **The Transformer Era** | 2017–2020 | "Attention Is All You Need" (2017) — the Transformer architecture let models look at *all* words in a sequence at once (self-attention), instead of one at a time. This is the architecture behind every modern LLM (GPT, Claude, Llama, Gemini). | Needed massive compute + massive data to be good. Still just predicting "the next most likely word" — no built-in notion of *truth*. |
| **Large Language Models (LLMs)** | 2020–present | Scaling Transformers to billions/trillions of parameters, trained on huge internet-scale text corpora (GPT-3 2020, ChatGPT 2022 made this mainstream, GPT-4/Claude/Gemini/Llama since). | **This is where Modules 1–4 of this course pick up** — see drawbacks below. |
| **Agentic AI** | 2023–present | LLMs given tools, memory, and multi-step reasoning loops so they *act* on systems instead of just describing what to do. | New failure modes: acting on bad information, looping forever, taking unauthorized actions — Modules 2–4 exist because of this. |

**The one-sentence version**: we went from *humans writing rules* → *humans
labeling data for narrow models* → *models learning their own features* →
*one giant model that can do almost any language task* → *that model now
acting in the world via tools*. Every step removed a human bottleneck and
introduced a new *trust* problem.

---

## 0.2 How an LLM Actually Works (just enough to reason about its limits)

1. Text is broken into **tokens** (roughly word-pieces, e.g. "unhappiness"
   → "un" + "happi" + "ness").
2. Each token becomes a vector (Module 1 covers this same idea for
   embeddings).
3. The Transformer's **self-attention** mechanism lets every token "look at"
   every other token in the input to build context-aware representations.
4. The model is trained on one deceptively simple task: **predict the next
   token**, over trillions of tokens of internet text, books, code, etc.
5. At inference time, it repeats: predict next token → append it → predict
   the next one → ... until done.

**The critical insight**: the model has no database of "facts" it looks up.
It has learned *statistical patterns of language* so well that next-token
prediction *looks like* reasoning and knowledge. That distinction is the
root of almost every drawback below.

---

## 0.3 The Drawbacks That Created This Entire Curriculum

This is the direct link from "generic LLM" to "Module 1–4 exists":

| Drawback | What it looks like in practice | Which module fixes/mitigates it |
|---|---|---|
| **Hallucination** | Model states a wrong fact fluently and confidently, because it's predicting *plausible* text, not *true* text. | Module 1 (RAG grounds answers in real retrieved documents) + Module 3 (groundedness scoring catches it when it still happens) |
| **Stale / frozen knowledge** | Model's knowledge ends at its training cutoff — it doesn't know your company's data at all, or yesterday's incident. | Module 1 (RAG injects live, private, current data at query time) |
| **No access to private/enterprise data** | A generic LLM was never trained on your internal wikis, runbooks, ticket history, HCON configs. | Module 1 (ingestion pipeline) |
| **Can't take real-world action** | A chatbot can *describe* how to restart a service but can't actually do it. | Module 2 (tool calling / function calling) |
| **Unpredictable / unbounded behavior** | Given autonomy, an LLM agent can loop forever, call the wrong tool, or take an action nobody approved. | Module 2 (ReAct loop with iteration limits + deterministic guardrails) |
| **No way to prove it's safe before shipping** | "It seemed to work in my 5 manual tests" doesn't scale to production and doesn't satisfy an audit. | Module 3 (RAG triad metrics, LLM-as-judge, repeatable scoring) |
| **Privacy / compliance exposure** | Sending raw text containing SSNs/credit cards to a third-party LLM API, or letting retrieval ignore who's asking. | Module 4 (PII redaction, RBAC-filtered retrieval) |
| **Prompt injection / malicious input** | A user (or a *document* the system retrieves!) contains hidden instructions trying to hijack the model's behavior. | Module 4 (input/output moderation) — and note: this is exactly the instruction-source-boundary problem your own Claude Code session enforces on tool outputs |
| **Cost & latency at scale** | Sending huge context windows to a giant model for every query is slow and expensive. | Module 1 (retrieval narrows context to only relevant chunks instead of "paste the whole wiki") |

**Read that table again slowly** — it *is* the syllabus. Modules 1–4 are
not four unrelated topics; they are four sequential patches applied to one
underlying tool (the LLM) to make it enterprise-usable.

---

## 0.4 Why "Bigger Model" Alone Doesn't Fix This

A natural question: why not just train a bigger/smarter model instead of all
this pipeline engineering? Two structural reasons:

1. **The knowledge-cutoff problem is architectural, not a capability
   problem.** No matter how smart the model, it was trained on a fixed
   snapshot of data. Your company's internal runbook written *last week*
   cannot be in any model's training data yet. RAG is the only way to give a
   model information that didn't exist when it was trained — this holds
   regardless of model size.
2. **Hallucination is a property of the objective function, not the
   model's IQ.** The model is trained to produce *likely* text, not
   *verified* text. A bigger model hallucinates more *convincingly*, not
   necessarily less often, especially on specific/private facts it was
   never trained on. Grounding + evaluation are the only way to convert
   "plausible" into "verified."

This is why every serious enterprise AI system today — regardless of which
underlying model it uses — has a RAG layer, a tool-calling layer, an
evaluation layer, and a governance layer. The model is one component, not
the whole system.

## 0.5 Where Data Engineers Fit In This Picture

Notice that of the four failure-mode categories above, the LLM vendor
(Anthropic, OpenAI, Meta) only really controls the *model*. Everything else
— ingestion, chunking, retrieval, tool wiring, evaluation pipelines,
governance filters — is **data/platform engineering work**, built on top of
a model treated as a commodity component. This is why your existing skill
set (ETL pipeline design, schema validation, access control, data quality
monitoring) transfers so directly — see the table in the main
[README.md](../README.md).

## Run it
This module is conceptual — no code to run. Read this file, then start
[module1_rag/NOTES.md](../module1_rag/NOTES.md).
