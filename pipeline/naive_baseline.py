import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set in the environment."
    )

client_genai = genai.Client(
    api_key=GOOGLE_API_KEY
)


# ============================================================
# QDRANT CONFIGURATION
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "taxchronos_ito_2001_bge"


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)


# ============================================================
# QDRANT CLIENT
# ============================================================

def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_top_k(
    query: str,
    k: int = 4
) -> list[dict]:

    qdrant = get_qdrant_client()

    query_vector = embedding_model.encode(
        query
    ).tolist()

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k
    )

    retrieved_chunks = []

    for res in response.points:

        retrieved_chunks.append(
            {
                "text": res.payload.get("text"),
                "section": res.payload.get("section"),
                "source": res.payload.get("source"),
                "score": res.score
            }
        )

    return retrieved_chunks


# ============================================================
# GEMINI GENERATION WITH RETRIES
# ============================================================

def generate_with_retry(
    prompt: str,
    max_retries: int = 5
) -> str:

    model_name = "gemini-3.5-flash"

    for attempt in range(max_retries):

        try:

            response = client_genai.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except ServerError as e:

            attempt_number = attempt + 1

            if attempt_number >= max_retries:

                print(
                    f"\nGemini failed after "
                    f"{max_retries} attempts."
                )

                raise e

            # Exponential backoff:
            # 2, 4, 8, 16 seconds...
            delay = 2 ** attempt

            print(
                f"\nGemini returned 503 "
                f"(attempt {attempt_number}/{max_retries})."
            )

            print(
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)

        except Exception as e:

            print(
                f"\nGemini generation failed: "
                f"{type(e).__name__}: {e}"
            )

            raise


# ============================================================
# NAIVE RAG ANSWER GENERATION
# ============================================================

def generate_naive_answer(
    query: str,
    chunks: list[dict]
) -> str:

    context_str = ""

    for i, chunk in enumerate(
        chunks,
        1
    ):

        context_str += (
            f"\n--- Chunk {i} "
            f"({chunk['section']}) ---\n"
            f"{chunk['text']}\n"
        )

    prompt = f"""
You are a helpful tax compliance assistant for Pakistan.

Answer the user's question using ONLY the provided legal
context chunks below.

If the context does not contain enough information, state that
you do not know based on the provided text.

Always cite the relevant Section numbers in your answer.

Do not invent tax rates, legal provisions, dates, exemptions,
thresholds, or interpretations that are not supported by the
provided context.

Do not infer that a person, entity, transaction, or category
falls under a particular legal provision unless the provided
context explicitly establishes that connection.

Do not invent tax rates, exemptions, thresholds, dates,
conditions, or legal interpretations.

If the provided context does not explicitly answer the question,
say that the provided context is insufficient.

Context:

{context_str}

User Question: {query}

Answer:
"""

    return generate_with_retry(prompt)


# ============================================================
# NAIVE RAG TEST
# ============================================================

def run_naive_rag(query: str):

    print(
        f"Query: {query}\n"
    )

    print(
        "Retrieving context..."
    )

    chunks = retrieve_top_k(
        query,
        k=4
    )

    print(
        f"Retrieved {len(chunks)} chunks. "
        "Generating answer...\n"
    )

    answer = generate_naive_answer(
        query,
        chunks
    )

    print(
        "----- NAIVE RAG ANSWER -----"
    )

    print(answer)

    print(
        "----------------------------"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_query = (
        "What is the tax rate on IT export "
        "services and freelancers?"
    )

    run_naive_rag(test_query)