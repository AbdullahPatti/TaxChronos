import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

FAST = os.getenv("FAST_MODEL", "gemini-3.5-flash-lite")
SMART = os.getenv("SMART_MODEL", "gemini-3.8-flash")
FALLBACKS = os.getenv("FALLBACK_MODEL", "gemini-3.5-flash,gemini-3.1-flash-lite").split(",")
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


def can_fall_back(error):
    return isinstance(error, errors.ServerError) or error.code in (404, 429)


def generate(model, prompt, config):
    chain = [model] + [m for m in FALLBACKS if m != model]
    for candidate in chain:
        try:
            return client.models.generate_content(model=candidate, contents=prompt, config=config)
        except errors.APIError as error:
            if candidate == chain[-1] or not can_fall_back(error):
                raise


def ask(prompt, schema=None, model=FAST):
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json" if schema else None,
        response_schema=schema,
    )
    response = generate(model, prompt, config)
    return response.parsed if schema else response.text.strip()
