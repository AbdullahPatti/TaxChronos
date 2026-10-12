import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ingestion.parse import load_sources
from pipeline import llm
from pipeline.store import pool, vector

BATCH = 50


def migrate():
    with pool.connection() as conn:
        conn.execute((ROOT / "db" / "schema.sql").read_text(encoding="utf-8"))


def embedding_text(chunk):
    return f"{chunk['title']}\n{chunk['section']}\n{chunk['content']}"


def index(source):
    chunks = json.loads((ROOT / "data" / "processed" / f"{source['id']}.json").read_text(encoding="utf-8"))
    with pool.connection() as conn:
        stored = {r["id"] for r in conn.execute("select id from chunks where source = %s", (source["id"],))}
        stale = stored - {c["id"] for c in chunks}
        conn.execute("delete from chunks where id = any(%s)", (list(stale),))
    fresh = [c for c in chunks if c["id"] not in stored]
    print(f"{source['id']}: {len(chunks)} chunks, {len(fresh)} to embed, {len(stale)} removed")
    for start in range(0, len(fresh), BATCH):
        batch = fresh[start:start + BATCH]
        embeddings = llm.embed([embedding_text(c) for c in batch], task="RETRIEVAL_DOCUMENT")
        with pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                """insert into chunks (id, source, title, section, page, content, amendments, effective_from, effective_to, embedding)
                   values (%(id)s, %(source)s, %(title)s, %(section)s, %(page)s, %(content)s, %(amendments)s,
                           %(effective_from)s, %(effective_to)s, %(embedding)s::vector)
                   on conflict (id) do nothing""",
                [{**c, "embedding": vector(e)} for c, e in zip(batch, embeddings)],
            )
        print(f"  {start + len(batch)}/{len(fresh)}")
    return bool(fresh or stale)


def main(ids):
    migrate()
    changed = [index(s) for s in load_sources() if not ids or s["id"] in ids]
    if any(changed):
        with pool.connection() as conn:
            conn.execute("truncate answers")
        print("Corpus changed, semantic cache cleared.")


if __name__ == "__main__":
    main(sys.argv[1:])
