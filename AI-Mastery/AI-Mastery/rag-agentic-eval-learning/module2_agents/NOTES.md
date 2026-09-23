# Module 2: Tool Calling & Agentic Workflows

## 1. The Enterprise Problem
A chatbot that only talks can't *do* anything — it can't reset a password,
query a database, restart a service, or open a ticket. To be useful for
real operational work (like oncall), the AI needs to safely trigger real
actions, not just generate text.

## 2. The Engineering Solution

### 2.1 Function Calling & JSON Generation
- **Schema definition**: you describe each tool as a JSON Schema — name,
  description, and typed parameters. The LLM is given this schema and
  decides *if* and *when* to call it, and *what arguments* to pass.
- **Parameter extraction**: the model reads free text ("Reset password for
  user 1234") and emits structured JSON (`{"user_id": 1234}`) matching the
  schema — this is the same idea as parsing an unstructured log line into a
  typed record in an ETL job, except the "parser" is the LLM itself.

### 2.2 Agent Orchestration Frameworks
- **LangChain / LangGraph**: libraries for wiring together multi-step,
  stateful LLM applications (retry logic, memory, branching). We don't
  require them here — the demo implements the loop by hand so you see
  exactly what those frameworks automate.
- **ReAct (Reason + Action)**: the core agent loop —
  1. **Think**: model reasons about what to do next
  2. **Act**: model calls a tool
  3. **Observe**: the tool's real output is fed back to the model
  4. Repeat until the model decides it has enough information to answer,
     or a hard iteration limit is hit (a guardrail against infinite loops —
     see `runbook_3` in Module 1's demo corpus).

### 2.3 Deterministic Guardrails
- **Hybrid routing**: never let the LLM alone decide on a compliance-
  critical action (e.g. "delete this account", "issue a refund"). Wrap
  those tool calls in ordinary `if/else` code that enforces hard business
  rules — the LLM can *propose* the action, but code decides whether it's
  *allowed*. This is identical in spirit to a validation layer in front of
  a pipeline write step: the LLM is "untrusted input" just like user input
  is in any other system.

## 3. The Evaluation Metric
- **Tool-call accuracy**: did the model call the *correct* tool with the
  *correct* arguments for a given instruction? (Precision/recall over a
  labeled set of instruction → expected-tool-call pairs.)
- **Task completion rate**: across a suite of test scenarios, what fraction
  did the agent resolve correctly without human intervention or without
  tripping a guardrail incorrectly?

## Run it
```bash
python module2_agents/agent_tool_calling.py
```
Implements a tiny ReAct loop with two tools (`check_service_health`,
`restart_service`) and a **deterministic guardrail**: the agent may *check*
health freely, but `restart_service` is hard-blocked by code unless the
service is confirmed unhealthy — the LLM cannot talk its way around that
check.
