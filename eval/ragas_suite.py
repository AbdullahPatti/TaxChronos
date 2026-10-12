import asyncio
import json
import statistics
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

vertex_stub = types.ModuleType("langchain_community.chat_models.vertexai")
vertex_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault(vertex_stub.__name__, vertex_stub)

from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from pipeline import graph, llm, naive_baseline

PIPELINES = {"naive": naive_baseline.answer, "crag": graph.answer}
EVAL_SET = ROOT / "eval" / "codeswitch_eval_set.jsonl"


def load_questions():
    return [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(name):
    path = ROOT / "eval" / f"{name}_results.json"
    done = {r["question"]: r for r in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else {}
    for i, item in enumerate(load_questions(), 1):
        if item["query"] in done:
            continue
        print(f"[{i}] {item['query']}")
        started = time.perf_counter()
        state = PIPELINES[name](item["query"])
        done[item["query"]] = {
            "question": item["query"],
            "ground_truth": item["ground_truth"],
            "answer": state["response"]["answer"],
            "contexts": [d["content"] for d in state.get("docs", [])],
            "seconds": round(time.perf_counter() - started, 2),
        }
        path.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    return list(done.values())


async def score(results):
    judge = llm_factory(llm.FAST, provider="google", client=llm.client)
    embedder = embedding_factory("google", model=llm.EMBEDDER, client=llm.client)
    metrics = {
        "faithfulness": (Faithfulness(llm=judge), lambda r: dict(user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"] or [""])),
        "answer_relevancy": (AnswerRelevancy(llm=judge, embeddings=embedder), lambda r: dict(user_input=r["question"], response=r["answer"])),
        "context_precision": (ContextPrecision(llm=judge), lambda r: dict(user_input=r["question"], reference=r["ground_truth"], retrieved_contexts=r["contexts"] or [""])),
        "context_recall": (ContextRecall(llm=judge), lambda r: dict(user_input=r["question"], reference=r["ground_truth"], retrieved_contexts=r["contexts"] or [""])),
    }
    report = {}
    for name, (metric, inputs) in metrics.items():
        values = [(await metric.ascore(**inputs(r))).value for r in results]
        report[name] = round(statistics.fmean(values), 4)
    report["latency_p50_s"] = statistics.median(r["seconds"] for r in results)
    return report


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "crag"
    report = asyncio.run(score(run(name)))
    print(json.dumps({name: report}, indent=2))
