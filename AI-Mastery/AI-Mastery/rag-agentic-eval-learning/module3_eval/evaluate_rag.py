"""
Module 3: LLM Evaluation & Validation — runnable demo.

Implements the RAG triad (context relevance, groundedness, answer
relevance) as LLM-as-a-judge checks, using a local Ollama model as the
judge. Mirrors what Ragas/TruLens do under the hood, minus the dependency.
"""

import json
import ollama

JUDGE_MODEL = "llama3.2"

# --- Test cases: (query, retrieved_context, generated_answer) ---
TEST_CASES = [
    {
        "name": "clean_case",
        "query": "Why does my QuickETL job time out on a Datalake bucket?",
        "context": (
            "If a QuickETL HCON job fails with ERR_504_TIMEOUT, first check "
            "whether the source Datalake bucket has FGAC enabled. FGAC-enabled "
            "buckets require the SparkLoaderStep operator to use the "
            "'assumeRole' parameter, otherwise reads silently time out."
        ),
        "answer": (
            "The timeout is likely because the Datalake bucket has FGAC enabled "
            "and the SparkLoaderStep operator is missing the assumeRole parameter. "
            "Add assumeRole=datalake-reader-role to fix it."
        ),
    },
    {
        "name": "hallucinated_case",
        "query": "Why does my QuickETL job time out on a Datalake bucket?",
        "context": (
            "If a QuickETL HCON job fails with ERR_504_TIMEOUT, first check "
            "whether the source Datalake bucket has FGAC enabled. FGAC-enabled "
            "buckets require the SparkLoaderStep operator to use the "
            "'assumeRole' parameter, otherwise reads silently time out."
        ),
        "answer": (
            "This is caused by your Spark executor memory being set too low. "
            "Increase spark.executor.memory to 8g and the timeout will resolve."
        ),
    },
    {
        "name": "off_topic_case",
        "query": "Why does my QuickETL job time out on a Datalake bucket?",
        "context": (
            "If a QuickETL HCON job fails with ERR_504_TIMEOUT, first check "
            "whether the source Datalake bucket has FGAC enabled."
        ),
        "answer": (
            "QuickETL is a configuration-driven ETL framework that uses HOCON "
            "files to define pipelines. It supports many operator types."
        ),
    },
]

JUDGE_PROMPT_TEMPLATE = """You are a strict evaluation judge for a RAG system. \
Score the ANSWER on three dimensions given the QUERY and CONTEXT.

QUERY: {query}

CONTEXT: {context}

ANSWER: {answer}

Score each dimension from 1 (worst) to 5 (best):
- context_relevance: does the CONTEXT actually relate to the QUERY?
- groundedness: is every claim in the ANSWER directly supported by the CONTEXT \
(5 = fully supported, 1 = invented/contradicted by CONTEXT)?
- answer_relevance: does the ANSWER actually address the QUERY asked?

Respond with ONLY valid JSON in this exact format, no other text:
{{"context_relevance": <int>, "groundedness": <int>, "answer_relevance": <int>, "reasoning": "<one sentence>"}}
"""


def judge(query: str, context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, context=context, answer=answer)
    response = ollama.generate(model=JUDGE_MODEL, prompt=prompt, format="json")
    try:
        return json.loads(response["response"])
    except json.JSONDecodeError:
        return {"error": "judge did not return valid JSON", "raw": response["response"]}


def main():
    print("=== Module 3: RAG Triad Evaluation Demo ===\n")

    GROUNDEDNESS_THRESHOLD = 3  # below this, block deployment (Module 3, section 3)

    for case in TEST_CASES:
        print(f"--- {case['name']} ---")
        print(f"Query:   {case['query']}")
        print(f"Answer:  {case['answer'][:100]}...")

        scores = judge(case["query"], case["context"], case["answer"])
        print(f"Scores:  {scores}\n")

        if "groundedness" in scores and scores["groundedness"] < GROUNDEDNESS_THRESHOLD:
            print(f"  -> BLOCKED: groundedness {scores['groundedness']} < threshold {GROUNDEDNESS_THRESHOLD}\n")
        else:
            print("  -> PASSED groundedness check\n")


if __name__ == "__main__":
    main()
