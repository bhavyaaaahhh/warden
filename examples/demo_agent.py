"""Tiny fake agent for exercising the SDK until Argus is ready.

Run: uv run python -m examples.demo_agent "your question"
"""

import sys
import time

from warden_sdk import trace

VERSION = "demo-v0.1"

DOCS = [
    {"id": "doc1", "text": "Warden records traces and spans for LLM agents."},
    {"id": "doc2", "text": "Regression diffs compare runs across version tags."},
    {"id": "doc3", "text": "Postgres stores traces; FastAPI ingests them."},
]


def hybrid_search(query: str) -> list[dict]:
    time.sleep(0.1)
    words = set(query.lower().split())
    hits = [d for d in DOCS if words & set(d["text"].lower().split())]
    return hits or DOCS[:1]


def rerank(docs: list[dict]) -> list[dict]:
    time.sleep(0.05)
    return sorted(docs, key=lambda d: len(d["text"]))


def generate(query: str, context: list[dict]) -> tuple[str, int, int]:
    # Fake LLM: returns (answer, tokens_in, tokens_out).
    time.sleep(0.2)
    answer = f"Based on {len(context)} docs: {context[0]['text']}"
    return answer, 50 + 20 * len(context), len(answer.split())


def run(query: str) -> str:
    with trace("demo_agent", input=query, version_tag=VERSION) as t:
        with t.span("retrieval", name="hybrid_search", input=query) as s:
            docs = hybrid_search(query)
            # Nested span: parent_span_id should point at hybrid_search.
            with t.span("tool_call", name="rerank", input=[d["id"] for d in docs]) as r:
                docs = rerank(docs)
                r.output = [d["id"] for d in docs]
            s.output = docs

        with t.span("llm_call", name="generate_answer", input=docs) as s:
            answer, tok_in, tok_out = generate(query, docs)
            s.output = answer
            s.tokens_input = tok_in
            s.tokens_output = tok_out
            s.cost_usd = tok_in * 3e-6 + tok_out * 15e-6

    print(f"trace_id={t.trace_id}")
    return answer


if __name__ == "__main__":
    print(run(" ".join(sys.argv[1:]) or "how do regression diffs work"))
