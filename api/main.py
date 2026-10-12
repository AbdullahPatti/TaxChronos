import json
import time
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from pipeline.graph import graph

REQUESTS_PER_MINUTE = 10
UI = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

app = FastAPI(title="TaxChronos")
recent = defaultdict(deque)


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    as_of: date | None = None


def throttle(ip):
    now, hits = time.monotonic(), recent[ip]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= REQUESTS_PER_MINUTE:
        raise HTTPException(429, "Too many questions, try again in a minute.")
    hits.append(now)


def event(name, data):
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
def index():
    return FileResponse(UI)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(body: Question, request: Request):
    throttle(request.client.host)

    def stream():
        try:
            for update in graph.stream({"question": body.question, "as_of": body.as_of}, stream_mode="updates"):
                for node, state in update.items():
                    yield event("step", {"node": node, "query": (state or {}).get("query")})
                    if state and "response" in state:
                        yield event("answer", state["response"])
        except Exception:
            yield event("error", {"message": "Something went wrong while answering. Please try again."})
            raise

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
