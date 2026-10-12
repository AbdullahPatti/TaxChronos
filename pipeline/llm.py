import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

FAST = os.getenv("FAST_MODEL", "gemini-3.5-flash-lite")
SMART = os.getenv("SMART_MODEL", "gemini-3.8-flash")
FALLBACK = os.getenv("FALLBACK_MODEL", "gemini-3.5-flash")
EMBEDDER = "gemini-embedding-001"
DIMENSIONS = 768

QUICK = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=3, initial_delay=1, max_delay=8))
PATIENT = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=8, initial_delay=5, max_delay=90))

client = genai.Client(http_options=QUICK)


def embed(texts, task="RETRIEVAL_QUERY"):
    response = client.models.embed_content(
        model=EMBEDDER,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task,
            output_dimensionality=DIMENSIONS,
            http_options=PATIENT if task == "RETRIEVAL_DOCUMENT" else None,
        ),
    )
    return [e.values for e in response.embeddings]


def generate(model, prompt, config):
    try:
        return client.models.generate_content(model=model, contents=prompt, config=config)
    except errors.ServerError:
        if model == FALLBACK:
            raise
        return client.models.generate_content(model=FALLBACK, contents=prompt, config=config)


def ask(prompt, schema=None, model=FAST):
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json" if schema else None,
        response_schema=schema,
    )
    response = generate(model, prompt, config)
    return response.parsed if schema else response.text.strip()


def search_web(prompt):
    config = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
    response = generate(SMART, prompt, config)
    metadata = response.candidates[0].grounding_metadata
    sources = [{"title": c.web.title, "url": c.web.uri} for c in (metadata and metadata.grounding_chunks or []) if c.web]
    return response.text.strip(), sources
