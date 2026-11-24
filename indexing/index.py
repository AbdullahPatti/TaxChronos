import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ingestion.parse import load_sources
from pipeline import llm

BATCH = 50
EMBED_BATCH = 10  # ponytail: free tier caps embeds at 30k tokens/min; 50 long chunks blow it in one request
CACHE = ROOT / "cache" / "embeddings"
out_of_quota = False


def embedding_text(chunk):
    return f"{chunk['title']}\n{chunk['section']}\n{chunk['content']}"


def load_chunks(source):
    return json.loads((ROOT / "data" / "processed" / f"{source['id']}.json").read_text(encoding="utf-8"))


def load_embeddings(path):
    known = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                chunk, embedding = json.loads(line)
                known[chunk] = embedding
            except ValueError:
                pass
    return known


def embed(source, chunks):
    global out_of_quota
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{source['id']}.jsonl"
    known = load_embeddings(path)
    fresh = [c for c in chunks if c["id"] not in known]
    print(f"{source['id']}: {len(chunks)} chunks, {len(fresh)} to embed")
    with path.open("a", encoding="utf-8") as out:
        for start in range(0, 0 if out_of_quota else len(fresh), EMBED_BATCH):
            batch = fresh[start:start + EMBED_BATCH]
            try:
                embeddings = llm.embed([embedding_text(c) for c in batch], task="RETRIEVAL_DOCUMENT")
            except llm.errors.ClientError as error:
                if error.code != 429:
                    raise
                out_of_quota = True
                print(f"  quota reached with {len(fresh) - start} left; uploading what is done, run again after the reset")
                break
            out.write("".join(json.dumps([c["id"], e]) + "\n" for c, e in zip(batch, embeddings)))
            out.flush()
            known.update((c["id"], e) for c, e in zip(batch, embeddings))
            print(f"  {start + len(batch)}/{len(fresh)}")
    return known


def upload(source, chunks, embeddings, pool, vector):
    with pool.connection() as conn:
        stored = {r["id"] for r in conn.execute("select id from chunks where source = %s", (source["id"],))}
        stale = stored - {c["id"] for c in chunks}
        conn.execute("delete from chunks where id = any(%s)", (list(stale),))
    fresh = [c for c in chunks if c["id"] not in stored and c["id"] in embeddings]
    print(f"{source['id']}: {len(fresh)} to upload, {len(stale)} removed")
    for start in range(0, len(fresh), BATCH):
        batch = fresh[start:start + BATCH]
        with pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                """insert into chunks (id, source, title, section, page, content, amendments, effective_from, effective_to, embedding)
                   values (%(id)s, %(source)s, %(title)s, %(section)s, %(page)s, %(content)s, %(amendments)s,
                           %(effective_from)s, %(effective_to)s, %(embedding)s::vector)
                   on conflict (id) do nothing""",
                [{**c, "embedding": vector(embeddings[c["id"]])} for c in batch],
            )
        print(f"  {start + len(batch)}/{len(fresh)}")


def main(ids):
    sources = [s for s in load_sources() if not ids or s["id"] in ids]
    prepared = [(s, chunks, embed(s, chunks)) for s in sources for chunks in [load_chunks(s)]]
    if not os.getenv("DATABASE_URL"):
        print(f"DATABASE_URL is not set. Embeddings are saved in {CACHE}; set it and run this again to upload.")
        return
    from pipeline.store import pool, vector

    with pool.connection() as conn:
        conn.execute((ROOT / "db" / "schema.sql").read_text(encoding="utf-8"))
    for s, chunks, embeddings in prepared:
        upload(s, chunks, embeddings, pool, vector)


if __name__ == "__main__":
    main(sys.argv[1:])
