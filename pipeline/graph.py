import re
from datetime import date
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from pipeline import llm, prompts, store

MAX_REWRITES = 1
KEEP = 6


class State(TypedDict, total=False):
    question: str
    as_of: date | None
    intent: str
    language: str
    query: str
    docs: list[dict]
    relevant: int
    rewrites: int
    response: dict


class Route(BaseModel):
    intent: Literal["chitchat", "out_of_scope", "factual", "procedural", "comparison"]
    language: Literal["english", "urdu", "roman_urdu"]
    search_query: str
    reply: str


class Grades(BaseModel):
    relevant: list[int]
    ambiguous: list[int]


class Rewrite(BaseModel):
    search_query: str


def passages(docs, limit=None):
    return "\n\n".join(
        f"[{i}] {d['title']}, {d['section']} (page {d['page']}, in force {d['effective_from']} to {d['effective_to'] or 'present'})\n"
        f"{d['content'][:limit]}"
        + (f"\nAmendments: {'; '.join(d['amendments'])}" if d["amendments"] else "")
        for i, d in enumerate(docs, 1)
    )


def citation(n, d):
    return {
        "n": n,
        "source": d["title"],
        "section": d["section"],
        "page": d["page"],
        "effective_from": str(d["effective_from"]),
        "effective_to": d["effective_to"] and str(d["effective_to"]),
        "amendments": d["amendments"],
        "excerpt": d["content"][:600],
    }


def respond(state, answer, citations=()):
    return {
        "answer": answer,
        "citations": list(citations),
        "as_of": str(state.get("as_of") or date.today()),
        "language": state.get("language"),
    }


def route(state):
    r = llm.ask(prompts.ROUTE.format(question=state["question"]), Route)
    update = {"intent": r.intent, "language": r.language, "query": r.search_query, "rewrites": 0}
    if r.intent in ("chitchat", "out_of_scope"):
        update["response"] = respond({**state, **update}, r.reply)
    return update


def retrieve(state):
    return {"docs": store.search(state["query"], llm.embed([state["query"]])[0], state.get("as_of"))}


def grade(state):
    docs = state["docs"]
    g = llm.ask(prompts.GRADE.format(question=state["question"], query=state["query"], passages=passages(docs, 700)), Grades)
    relevant = [d for i, d in enumerate(docs, 1) if i in g.relevant]
    ambiguous = [d for i, d in enumerate(docs, 1) if i in g.ambiguous and i not in g.relevant]
    return {"docs": (relevant + ambiguous)[:KEEP], "relevant": len(relevant)}


def rewrite(state):
    r = llm.ask(prompts.REWRITE.format(question=state["question"], query=state["query"]), Rewrite)
    return {"query": r.search_query, "rewrites": state["rewrites"] + 1}


def generate(state):
    docs = state["docs"]
    if not docs:
        return {"response": respond(state, prompts.NOT_FOUND)}
    as_of = state.get("as_of") or date.today()
    answer = llm.ask(
        prompts.ANSWER.format(question=state["question"], language=state["language"], as_of=as_of, passages=passages(docs)),
        model=llm.SMART,
    )
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(docs)})
    return {"response": respond(state, answer, [citation(n, docs[n - 1]) for n in used])}


def after_grade(state):
    return "generate" if state["relevant"] or state["rewrites"] >= MAX_REWRITES else "rewrite"


builder = StateGraph(State)
for node in (route, retrieve, grade, rewrite, generate):
    builder.add_node(node)
builder.add_edge(START, "route")
builder.add_conditional_edges("route", lambda state: END if "response" in state else "retrieve", ["retrieve", END])
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", after_grade, ["generate", "rewrite"])
builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", END)
graph = builder.compile()


def answer(question, as_of=None):
    return graph.invoke({"question": question, "as_of": as_of})
