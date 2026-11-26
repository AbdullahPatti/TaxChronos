import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
import types
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

vertex_stub = types.ModuleType("langchain_community.chat_models.vertexai")
vertex_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault(vertex_stub.__name__, vertex_stub)

from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from pipeline import llm

EVAL_SET = ROOT / "eval" / "codeswitch_eval_set.jsonl"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
JUDGE = os.getenv("EVAL_JUDGE", llm.FAST)
SCORE_ATTEMPTS = 3


def load_questions():
    items = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    limit = int(os.getenv("EVAL_LIMIT", 0))
    if not limit:
        return items
    by_source = defaultdict(list)
    for item in items:
        by_source[item["sources"][0]].append(item)
    # round-robin across sources so a cut set still covers every document
    picked = [group[i] for i in range(max(map(len, by_source.values()))) for group in by_source.values() if i < len(group)]
    return picked[:limit]


def pipelines():
    from pipeline import graph, naive_baseline

    return {"naive": naive_baseline.answer, "crag": graph.answer}


def found_source(item, docs):
    return any(d["source"] in item["sources"] and re.search(item["section"], d["section"]) for d in docs)


def run(name):
    path = ROOT / "eval" / f"{name}_results.json"
    done = {r["question"]: r for r in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
    answer = pipelines()[name]
    questions = load_questions()
    for i, item in enumerate(questions, 1):
        if item["query"] in done:
            continue
        print(f"[{i}] {item['query']}")
        as_of = date.fromisoformat(item["as_of"]) if item.get("as_of") else None
        started = time.perf_counter()
        try:
            state = answer(item["query"], as_of)
        except Exception as error:
            print(f"  skipped, run again to retry: {type(error).__name__}")
            continue
        docs = state.get("docs", [])
        done[item["query"]] = {
            "question": item["query"],
            "ground_truth": item["ground_truth"],
            "language": item["language"],
            "sources": item["sources"],
            "answer": state["response"]["answer"],
            "contexts": [f"{d['title']}, {d['section']}\n{d['content']}" for d in docs],
            "retrieved": [f"{d['source']} | {d['section']}" for d in docs],
            "hit": found_source(item, docs),
            "seconds": round(time.perf_counter() - started, 2),
        }
        path.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    return [done[item["query"]] for item in questions if item["query"] in done]


def judge():
    from openai import AsyncOpenAI

    return llm_factory(JUDGE, provider="openai", client=AsyncOpenAI(base_url=GEMINI_URL, api_key=os.environ["GOOGLE_API_KEY"]))


def metric_inputs():
    model = judge()
    embedder = embedding_factory("google", model=llm.EMBEDDER, client=llm.client)
    context = lambda r: r["contexts"] or [""]
    return {
        "faithfulness": (Faithfulness(llm=model), lambda r: dict(user_input=r["question"], response=r["answer"], retrieved_contexts=context(r))),
        "answer_relevancy": (AnswerRelevancy(llm=model, embeddings=embedder), lambda r: dict(user_input=r["question"], response=r["answer"])),
        "context_precision": (ContextPrecision(llm=model), lambda r: dict(user_input=r["question"], reference=r["ground_truth"], retrieved_contexts=context(r))),
        "context_recall": (ContextRecall(llm=model), lambda r: dict(user_input=r["question"], reference=r["ground_truth"], retrieved_contexts=context(r))),
    }


async def attempt(metric, inputs):
    for tries in range(SCORE_ATTEMPTS):
        try:
            value = (await metric.ascore(**inputs)).value
            return None if value is None or math.isnan(value) else value
        except Exception as error:
            print(f"  scoring failed: {type(error).__name__}")
            await asyncio.sleep(5 * (tries + 1))
    return None


async def score(name, results):
    path = ROOT / "eval" / f"{name}_scores.json"
    scores = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    metrics = metric_inputs()
    for result in results:
        row = scores.setdefault(result["question"], {})
        for metric_name, (metric, inputs) in metrics.items():
            if row.get(metric_name) is None:
                row[metric_name] = await attempt(metric, inputs(result))
                path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    return scores


def mean(values):
    present = [v for v in values if v is not None]
    return round(statistics.fmean(present), 4) if present else None


def report(results, scores):
    scores = {r["question"]: scores.get(r["question"], {}) for r in results}
    summary = {metric: mean(s.get(metric) for s in scores.values()) for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
    summary["unscored"] = sum(v is None for s in scores.values() for v in s.values())
    summary["source_hit_rate"] = mean(float(r["hit"]) for r in results)
    summary["latency_p50_s"] = statistics.median(r["seconds"] for r in results)
    groups = defaultdict(list)
    for result in results:
        groups[f"language:{result['language']}"].append(result)
        groups[f"source:{result['sources'][0]}"].append(result)
    summary["groups"] = {
        group: {"n": len(rows), "hit": mean(float(r["hit"]) for r in rows), "recall": mean(scores.get(r["question"], {}).get("context_recall") for r in rows), "faithfulness": mean(scores.get(r["question"], {}).get("faithfulness") for r in rows)}
        for group, rows in sorted(groups.items())
    }
    summary["misses"] = [r["question"] for r in results if not r["hit"]]
    return summary


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "crag"
    results = run(name)
    scores = asyncio.run(score(name, results))
    print(json.dumps({name: report(results, scores)}, ensure_ascii=False, indent=2))
