"""
Module 0: See the core drawback with your own eyes — hallucination.

Asks a local open-source LLM a question about a fictional internal system
it has NEVER seen (no training data, no RAG context) and shows how
confidently it invents an answer anyway. Then shows the same question
answered WITH grounding (a one-line preview of what Module 1 fixes).

This is the single most important "aha" moment in the whole curriculum:
run this file, watch the model make something up, and everything that
follows (Modules 1-4) will make intuitive sense as "how do we stop that."
"""

import ollama

LLM_MODEL = "llama3.2"

# A fictional internal-only fact that could not possibly be in any public
# model's training data — guarantees you'll see a hallucination.
FICTIONAL_QUESTION = (
    "What is the maximum retry count for the 'QuickETL AtlasSyncOperator' "
    "before it triggers a PagerDuty escalation?"
)

# The "ground truth" we made up for this demo — stand-in for a real runbook.
GROUND_TRUTH_CONTEXT = (
    "Internal Runbook — AtlasSyncOperator: retries up to 4 times with "
    "exponential backoff. On the 5th consecutive failure, it triggers a "
    "PagerDuty escalation to the data-platform-oncall rotation."
)


def ask_without_grounding(question: str) -> str:
    """No context provided — the model has nothing real to go on."""
    response = ollama.generate(model=LLM_MODEL, prompt=question)
    return response["response"]


def ask_with_grounding(question: str, context: str) -> str:
    """Module 1's fix: give the model the real fact before it answers."""
    prompt = (
        "Answer using ONLY the context below. If the context doesn't "
        "contain the answer, say you don't know.\n\n"
        f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    )
    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    return response["response"]


def main():
    print("=== Module 0: Hallucination, Live ===\n")
    print(f"Question: {FICTIONAL_QUESTION}")
    print("(This refers to a fictional operator that does not exist anywhere")
    print(" on the internet or in any model's training data.)\n")

    print("--- Attempt 1: asking the raw LLM with NO context ---")
    ungrounded_answer = ask_without_grounding(FICTIONAL_QUESTION)
    print(f"Model's answer:\n{ungrounded_answer}\n")
    print(">>> Notice it answers confidently and specifically, despite this")
    print(">>> being 100% fabricated — there is no way it could know this.\n")

    print("--- Attempt 2: same question, WITH grounding (Module 1 preview) ---")
    grounded_answer = ask_with_grounding(FICTIONAL_QUESTION, GROUND_TRUTH_CONTEXT)
    print(f"Model's answer:\n{grounded_answer}\n")
    print(">>> Same model, same question — the only difference is we handed")
    print(">>> it the real fact first. This is the entire premise of RAG.")


if __name__ == "__main__":
    main()
