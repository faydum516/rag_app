"""
Offline evaluation for the RAG stack.

Checks whether retrieved chunks contain required phrases (case-insensitive) and,
optionally, whether the generated answer contains expected phrases.

Usage (from project root, with OPENAI_API_KEY set):

  python evaluation/evaluate.py --golden evaluation/golden_questions.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_classic.memory import ConversationBufferMemory

from rag_core import (
    get_retriever_k,
    load_vector_store,
    make_conversation_chain,
    openai_api_key_configured,
)


def _norm(s: str) -> str:
    return s.casefold()


def _text_contains_all(text: str, phrases: list[str]) -> bool:
    t = _norm(text)
    return all(_norm(p) in t for p in phrases if p)


def _evaluate_one(
    vectorstore,
    case: dict[str, Any],
) -> dict[str, Any]:
    question = (case.get("question") or "").strip()
    must = [p for p in case.get("must_contain_in_chunks", []) if p]
    ans_need = [p for p in case.get("answer_must_contain", []) if p]

    retriever = vectorstore.as_retriever(search_kwargs={"k": get_retriever_k()})
    try:
        docs = retriever.invoke(question)
    except AttributeError:
        docs = retriever.get_relevant_documents(question)
    blob = "\n".join(d.page_content for d in docs)
    retrieval_ok = _text_contains_all(blob, must) if must else True

    answer = ""
    answer_ok = True
    if ans_need:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        chain = make_conversation_chain(vectorstore, memory)
        out = chain.invoke({"question": question})
        answer = out.get("answer") or ""
        answer_ok = _text_contains_all(answer, ans_need)

    return {
        "id": case.get("id"),
        "question": question,
        "retrieval_ok": retrieval_ok,
        "answer_ok": answer_ok,
        "retrieved_chars": len(blob),
        "answer_preview": (answer[:400] + "…") if len(answer) > 400 else answer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answers.")
    parser.add_argument(
        "--golden",
        default=os.path.join(ROOT, "evaluation", "golden_questions.json"),
        help="Path to golden JSON file.",
    )
    args = parser.parse_args()

    if not openai_api_key_configured():
        print("OPENAI_API_KEY is not set (or is only whitespace).", file=sys.stderr)
        return 2

    if not os.path.isfile(args.golden):
        print(f"Golden file not found: {args.golden}", file=sys.stderr)
        return 2

    with open(args.golden, encoding="utf-8") as f:
        cases: list[dict[str, Any]] = json.load(f)

    if not cases:
        print(
            json.dumps(
                {
                    "cases": 0,
                    "message": "No golden cases. Add objects to the JSON array, each with "
                    "question and must_contain_in_chunks (optional answer_must_contain).",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    vectorstore = load_vector_store()
    if vectorstore is None:
        print(
            "No vector store on disk. Index PDFs in the app or build one before running eval.",
            file=sys.stderr,
        )
        return 2

    results = [_evaluate_one(vectorstore, c) for c in cases]
    retrieval_pass = sum(1 for r in results if r["retrieval_ok"])
    answer_pass = sum(1 for r in results if r["answer_ok"])
    n = len(results)

    report = {
        "cases": n,
        "retrieval_pass": retrieval_pass,
        "answer_checks_pass": answer_pass,
        "details": results,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if results and not all(r["retrieval_ok"] and r["answer_ok"] for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
