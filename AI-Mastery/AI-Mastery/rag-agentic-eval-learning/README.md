# Enterprise AI Learning Path — RAG, Agents, Evaluation, Governance

Hands-on companion to the curriculum: a foundations module on the AI
landscape's evolution and drawbacks (Module 0), followed by the 4-module
technical curriculum. Modules 1–4 each follow the **3-part structure** you
specified:

1. **The Enterprise Problem**
2. **The Engineering Solution**
3. **The Evaluation Metric**

All code runs **locally** against an **open-source model** via
[Ollama](https://ollama.com) — no API keys, no cloud spend, safe to run
against synthetic/dummy data on your laptop.

## Why this matters for YOUR work (data engineering → AI impact)

You build data pipelines (QuickETL/HCON configs, FGAC-compliant Datalake
ingestion, etc.). That is *exactly* the skillset enterprise RAG/agent systems
need on the ingestion side:

| Your current DE skill | Direct AI-system application |
|---|---|
| ETL pipeline design (extract → transform → load) | RAG ingestion pipeline (extract docs → chunk/embed → load into vector DB) |
| Schema validation (HCON config validation) | Tool-calling JSON schema validation (Module 2) |
| FGAC / access-control on Datalake | RBAC-aware retrieval filtering so the LLM can't retrieve data a user can't see (Module 4) |
| Data quality checks / pipeline monitoring | Groundedness/faithfulness evaluation of LLM outputs (Module 3) |
| Incident/oncall response for pipeline failures | **Oncall Assistant**: an agent that retrieves runbooks, calls diagnostic tools, and drafts incident summaries — see `oncall_assistant_demo/` |

The `oncall_assistant_demo/` folder is a mini end-to-end project tying all
four modules together: a RAG-backed, tool-calling oncall assistant that
retrieves runbooks, checks (fake) service health, drafts an incident
summary, and is evaluated for groundedness before "paging" anyone.

## Setup (one-time)

```bash
# 1. Install Ollama (local LLM runtime) - macOS
brew install ollama

# 2. Start the Ollama server (keep running in a terminal, or use `brew services`)
ollama serve

# 3. Pull a small, fast open-source model (~4.7GB) good enough for this demo
ollama pull llama3.2

# 4. Pull an embedding model for RAG
ollama pull nomic-embed-text

# 5. Python deps (use a venv)
cd rag-agentic-eval-learning
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Module map

| Folder | Covers | Run |
|---|---|---|
| `module0_foundations/` | **Start here.** AI landscape evolution (symbolic AI → ML → deep learning → Transformers → LLMs → agents), how LLMs actually work, and the specific drawbacks (hallucination, stale knowledge, no real-world action, no proof of safety, privacy exposure) that Modules 1–4 exist to fix | `python module0_foundations/hallucination_demo.py` |
| `module1_rag/` | Chunking, embeddings, vector search, hybrid search, re-ranking | `python module1_rag/rag_pipeline.py` |
| `module2_agents/` | Function calling, ReAct loop, deterministic guardrails | `python module2_agents/agent_tool_calling.py` |
| `module3_eval/` | RAG triad (context relevance, groundedness, answer relevance), LLM-as-judge | `python module3_eval/evaluate_rag.py` |
| `module4_governance/` | PII redaction, RBAC-filtered retrieval, input/output moderation | `python module4_governance/governance_guardrails.py` |
| `oncall_assistant_demo/` | End-to-end: RAG + tools + eval + governance for an oncall assistant | `python oncall_assistant_demo/oncall_assistant.py` |

**Suggested order**: read `module0_foundations/NOTES.md` first (no setup
needed), then run `hallucination_demo.py` once Ollama is set up — it's the
single clearest "aha" moment that makes every later module click. Then work
through Modules 1 → 4 in order, since each one patches a specific gap left
by the previous one, and finish with the `oncall_assistant_demo/` capstone.

Each module's `.md` file has the concept notes; each `.py` file is runnable
and heavily commented to connect code back to the concept.
