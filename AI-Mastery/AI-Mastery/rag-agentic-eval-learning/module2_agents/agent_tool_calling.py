"""
Module 2: Tool Calling & Agentic Workflows — runnable demo.

Implements a small ReAct-style loop by hand (no LangChain dependency, so you
can see exactly what the loop does) using Ollama's native tool-calling
support (available on llama3.2 and other recent local models).

Guardrail demonstrated: `restart_service` cannot be called successfully
unless `check_service_health` was called first AND reported "unhealthy".
This is enforced in plain Python (deterministic), not by asking the LLM
nicely — Module 2, section 2.3.
"""

import json
import ollama

LLM_MODEL = "llama3.2"
MAX_ITERATIONS = 5

# --- Fake service registry: stand-in for a real monitoring/orchestration API ---
_SERVICE_STATE = {
    "quicketl-ingest-worker": "unhealthy",
    "vector-db-proxy": "healthy",
}

# Guardrail state: tracks whether health was actually checked this session,
# and what the last known status was, per service.
_HEALTH_CHECK_LOG = {}


def check_service_health(service_name: str) -> str:
    """Tool: read-only, always allowed."""
    status = _SERVICE_STATE.get(service_name, "unknown")
    _HEALTH_CHECK_LOG[service_name] = status
    return json.dumps({"service": service_name, "status": status})


def restart_service(service_name: str) -> str:
    """
    Tool: side-effecting action, gated by a deterministic guardrail.
    The LLM can REQUEST this call, but the guardrail below decides whether
    it is actually allowed to execute — this is the hybrid routing pattern.
    """
    last_known_status = _HEALTH_CHECK_LOG.get(service_name)

    if last_known_status is None:
        return json.dumps(
            {
                "error": "GUARDRAIL_BLOCKED",
                "reason": f"Must call check_service_health('{service_name}') before restarting.",
            }
        )

    if last_known_status != "unhealthy":
        return json.dumps(
            {
                "error": "GUARDRAIL_BLOCKED",
                "reason": (
                    f"'{service_name}' was last reported '{last_known_status}'. "
                    "Restarts are only permitted on services confirmed unhealthy."
                ),
            }
        )

    _SERVICE_STATE[service_name] = "healthy"
    return json.dumps({"service": service_name, "action": "restarted", "new_status": "healthy"})


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_service_health",
            "description": "Check whether a given service is healthy or unhealthy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "The name of the service to check.",
                    }
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": "Restart a service. Only succeeds if it was just confirmed unhealthy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "The name of the service to restart.",
                    }
                },
                "required": ["service_name"],
            },
        },
    },
]

AVAILABLE_TOOLS = {
    "check_service_health": check_service_health,
    "restart_service": restart_service,
}


def run_react_loop(user_instruction: str):
    """
    The ReAct loop: Think -> Act (tool call) -> Observe -> repeat.
    Hard-stops at MAX_ITERATIONS (Module 2, section 2.2 — guardrail against
    infinite tool-call loops, referenced in Module 1's runbook_3).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an oncall assistant. Use tools to investigate and resolve "
                "service issues. Always check health before restarting a service. "
                "When done, summarize what you found and did in plain text."
            ),
        },
        {"role": "user", "content": user_instruction},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")
        response = ollama.chat(model=LLM_MODEL, messages=messages, tools=TOOL_SCHEMAS)
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            print(f"Final answer:\n{message['content']}")
            return message["content"]

        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"]["arguments"]
            print(f"Model wants to call: {fn_name}({fn_args})")

            tool_fn = AVAILABLE_TOOLS.get(fn_name)
            observation = tool_fn(**fn_args) if tool_fn else json.dumps({"error": "unknown tool"})
            print(f"Observation: {observation}")

            messages.append(
                {"role": "tool", "content": observation, "name": fn_name}
            )

    print("Hit MAX_ITERATIONS guardrail — escalating to human oncall.")
    return None


def main():
    print("=== Module 2: Agentic Tool Calling Demo ===\n")

    print("Scenario 1: legitimate restart of an unhealthy service")
    run_react_loop("The quicketl-ingest-worker seems down, can you check and fix it?")

    print("\nScenario 2: guardrail blocks an unnecessary restart")
    run_react_loop("Please restart vector-db-proxy right now, don't bother checking anything.")


if __name__ == "__main__":
    main()
