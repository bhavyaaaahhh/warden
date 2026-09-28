"""Tiny fake agent for exercising the SDK and evals until Argus is ready.

Run once:   uv run python -m examples.demo_agent "your question"
Evaluate:   uv run python -m warden_sdk.evals run examples/datasets/demo.jsonl --agent examples.demo_agent:run

DEMO_VERSION=demo-v0.2 switches on a deliberately worse version, to have
something for the regression diff to catch.
"""

import os
import re
import sys
import time

from warden_sdk import trace

VERSION = os.environ.get("DEMO_VERSION", "demo-v0.1")
# v0.2 "optimisation": look up only the first keyword, and send a bigger prompt.
SINGLE_TERM_SEARCH = VERSION == "demo-v0.2"
SYSTEM_PROMPT_TOKENS = 400 if VERSION == "demo-v0.2" else 50

DOCS = [
    {"id": "doc1", "text": "Warden records traces and spans for LLM agents."},
    {"id": "doc2", "text": "Regression diffs compare eval runs across version tags."},
    {"id": "doc3", "text": "Postgres stores traces, spans, and eval results."},
    {"id": "doc4", "text": "FastAPI ingests traces over HTTP."},
    {"id": "doc5", "text": "Strict mode raises an error when a trace upload fails."},
    {"id": "doc6", "text": "Datasets are JSONL files versioned in git."},
    {"id": "doc7", "text": "Scorers turn an agent output into a pass or fail."},
    {"id": "doc8", "text": "The viewer shows a timeline of spans for each trace."},
    {"id": "doc9", "text": "Cost is tracked per LLM call in US dollars."},
    {"id": "doc10", "text": "Alembic migrations manage the database schema."},
]

STOPWORDS = {
    "a", "an", "and", "are", "do", "does", "each", "for", "how", "in", "is",
    "of", "over", "the", "to", "what", "when", "which",
}


def keywords(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
    return words[:1] if SINGLE_TERM_SEARCH else words


def overlap(query: str, doc: dict) -> int:
    return len(set(keywords(query)) & set(re.findall(r"[a-z0-9]+", doc["text"].lower())))


def hybrid_search(query: str) -> list[dict]:
    time.sleep(0.05)
    hits = [d for d in DOCS if overlap(query, d)]
    return hits or DOCS[:1]


def rerank(query: str, docs: list[dict]) -> list[dict]:
    time.sleep(0.02)
    return sorted(docs, key=lambda d: overlap(query, d), reverse=True)


def generate(query: str, context: list[dict]) -> tuple[str, int, int]:
    # Fake LLM: answers from the top doc. Returns (answer, tokens_in, tokens_out).
    tokens_in = SYSTEM_PROMPT_TOKENS + 20 * len(context) + len(query.split())
    time.sleep(tokens_in / 2000)
    answer = f"Based on {len(context)} docs: {context[0]['text']}"
    return answer, tokens_in, len(answer.split())


def run(query: str) -> str:
    with trace("demo_agent", input=query, version_tag=VERSION) as t:
        with t.span("retrieval", name="hybrid_search", input=query) as s:
            docs = hybrid_search(query)
            # Nested span: parent_span_id should point at hybrid_search.
            with t.span("tool_call", name="rerank", input=[d["id"] for d in docs]) as r:
                docs = rerank(query, docs)
                r.output = [d["id"] for d in docs]
            s.output = docs

        with t.span("llm_call", name="generate_answer", input=docs) as s:
            answer, tok_in, tok_out = generate(query, docs)
            s.output = answer
            s.tokens_input = tok_in
            s.tokens_output = tok_out
            s.cost_usd = tok_in * 3e-6 + tok_out * 15e-6

    return answer


if __name__ == "__main__":
    print(run(" ".join(sys.argv[1:]) or "how do regression diffs work"))
