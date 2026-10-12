import os
from datetime import date

from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

load_dotenv()

CACHE_SIMILARITY = 0.92

pool = ConnectionPool(
    os.environ["DATABASE_URL"],
    min_size=1,
    max_size=4,
    kwargs={"row_factory": dict_row, "prepare_threshold": None},
    open=True,
)


def vector(values):
    return "[" + ",".join(map(str, values)) + "]"


def search(query, embedding, as_of=None, k=12):
    with pool.connection() as conn:
        return conn.execute(
            "select * from hybrid_search(%s, %s::vector, %s, %s)",
            (query, vector(embedding), as_of or date.today(), k),
        ).fetchall()


def recall(embedding, language, as_of=None):
    with pool.connection() as conn:
        row = conn.execute(
            """select response, 1 - (embedding <=> %s::vector) as similarity from answers
               where language = %s and as_of is not distinct from %s
               order by embedding <=> %s::vector limit 1""",
            (vector(embedding), language, as_of, vector(embedding)),
        ).fetchone()
    return row["response"] if row and row["similarity"] >= CACHE_SIMILARITY else None


def remember(question, language, as_of, embedding, response):
    with pool.connection() as conn:
        conn.execute(
            "insert into answers (question, language, as_of, embedding, response) values (%s, %s, %s, %s::vector, %s)",
            (question, language, as_of, vector(embedding), Jsonb(response)),
        )
