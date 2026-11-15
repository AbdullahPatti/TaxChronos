import os
from datetime import date

from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

load_dotenv()

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
