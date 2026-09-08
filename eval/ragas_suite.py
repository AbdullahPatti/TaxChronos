import sys
import os
import types
import json
import time
from pathlib import Path

# ============================================================
# PATH SETUP
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


# ============================================================
# LANGCHAIN COMMUNITY COMPATIBILITY
# ============================================================

try:
    import langchain_community.chat_models.vertexai
except ModuleNotFoundError:
    dummy_vertex = types.ModuleType(
        "langchain_community.chat_models.vertexai"
    )

    dummy_vertex.ChatVertexAI = type(
        "ChatVertexAI",
        (object,),
        {}
    )

    sys.modules[
        "langchain_community.chat_models.vertexai"
    ] = dummy_vertex


# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate

from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from pipeline.naive_baseline import (
    retrieve_top_k,
    generate_naive_answer,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. "
        "Add it to your .env file."
    )


# ============================================================
# FILE PATHS
# ============================================================

EVAL_FILE = (
    ROOT_DIR
    / "eval"
    / "codeswitch_eval_set.jsonl"
)

CACHE_FILE = (
    ROOT_DIR
    / "eval"
    / "naive_results.json"
)


# ============================================================
# LOAD EVALUATION SET
# ============================================================

def load_evaluation_set():
    queries = []
    ground_truths = []

    with open(
        EVAL_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            data = json.loads(line)

            queries.append(
                data["query"]
            )

            ground_truths.append(
                data["ground_truth"]
            )

    return queries, ground_truths


# ============================================================
# LOAD CACHE
# ============================================================

def load_cache():

    if not CACHE_FILE.exists():
        return []

    try:

        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        if not isinstance(data, list):
            return []

        return data

    except (
        json.JSONDecodeError,
        OSError,
    ):

        print(
            "Warning: Could not read cache. "
            "Starting from scratch."
        )

        return []


# ============================================================
# SAVE CACHE
# ============================================================

def save_cache(results):

    with open(
        CACHE_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# RUN / RESUME NAIVE PIPELINE
# ============================================================

def run_naive_pipeline(
    queries,
    ground_truths,
):

    cached_results = load_cache()

    # --------------------------------------------------------
    # Build lookup of already completed questions
    # --------------------------------------------------------

    cached_by_question = {
        item["question"]: item
        for item in cached_results
        if (
            isinstance(item, dict)
            and "question" in item
            and "answer" in item
            and "contexts" in item
        )
    }

    results = []

    total = len(queries)

    print(
        f"\nRunning naive pipeline on "
        f"{total} evaluation questions..."
    )

    for index, (
        query,
        ground_truth,
    ) in enumerate(
        zip(
            queries,
            ground_truths,
        ),
        start=1,
    ):

        # ----------------------------------------------------
        # Check cache
        # ----------------------------------------------------

        if query in cached_by_question:

            cached = cached_by_question[
                query
            ]

            results.append(
                cached
            )

            print(
                f"\n[{index}/{total}] "
                f"Already cached. Skipping:"
            )

            print(query)

            continue

        # ----------------------------------------------------
        # Generate new answer
        # ----------------------------------------------------

        print(
            f"\n[{index}/{total}] "
            f"Processing question:"
        )

        print(query)

        print(
            "Retrieving context..."
        )

        chunks = retrieve_top_k(
            query,
            k=4,
        )

        print(
            f"Retrieved {len(chunks)} chunks."
        )

        print(
            "Generating answer..."
        )

        try:

            answer = generate_naive_answer(
                query,
                chunks,
            )

        except Exception as e:

            print(
                "\nGeneration failed."
            )

            print(
                f"Error: {type(e).__name__}: {e}"
            )

            # Save everything completed so far
            save_cache(results)

            print(
                f"\nSaved {len(results)} "
                f"completed results."
            )

            print(
                "Run the script again to resume."
            )

            raise

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "question": query,
            "answer": answer,
            "contexts": [
                chunk["text"]
                for chunk in chunks
            ],
            "ground_truth": ground_truth,
        }

        results.append(result)

        # ----------------------------------------------------
        # IMPORTANT:
        # Save after EVERY successful question
        # ----------------------------------------------------

        save_cache(results)

        print(
            f"Saved progress: "
            f"{len(results)}/{total}"
        )

    return results


# ============================================================
# BUILD RAGAS DATASET
# ============================================================

def build_dataset(
    results,
):

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in results:

        questions.append(
            item["question"]
        )

        answers.append(
            item["answer"]
        )

        contexts.append(
            item["contexts"]
        )

        ground_truths.append(
            item["ground_truth"]
        )

    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    return Dataset.from_dict(
        dataset_dict
    )


# ============================================================
# RAGAS EVALUATION
# ============================================================

def run_ragas_evaluation(
    eval_dataset,
):

    print(
        "\n=========================================="
    )

    print(
        "Initializing RAGAS evaluator..."
    )

    print(
        "=========================================="
    )

    evaluator_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        max_retries=5,
    )

    evaluator_embeddings = (
        GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )
    )

    print(
        "\nEvaluating with RAGAS..."
    )

    results = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print(
        "\n=========================================="
    )

    print(
        "       NAIVE BASELINE RAGAS SCORES"
    )

    print(
        "=========================================="
    )

    print(results)

    print(
        "\nIndividual metrics:"
    )

    metric_names = [
        (
            "Faithfulness",
            "faithfulness",
        ),
        (
            "Answer Relevancy",
            "answer_relevancy",
        ),
        (
            "Context Precision",
            "context_precision",
        ),
        (
            "Context Recall",
            "context_recall",
        ),
    ]

    for display_name, key in metric_names:

        try:

            value = results[key]

            print(
                f"{display_name:<20} "
                f"{float(value):.4f}"
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            print(
                f"{display_name:<20} "
                f"Not available"
            )


# ============================================================
# MAIN
# ============================================================

def run_evaluation():

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    queries, ground_truths = (
        load_evaluation_set()
    )

    print(
        f"Loaded {len(queries)} "
        f"evaluation questions."
    )

    # --------------------------------------------------------
    # Generate / resume naive results
    # --------------------------------------------------------

    results = run_naive_pipeline(
        queries,
        ground_truths,
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(results) != len(queries):

        raise RuntimeError(
            f"Only {len(results)}/{len(queries)} "
            "questions have completed. "
            "RAGAS evaluation will not run."
        )

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------

    print(
        "\nAll naive pipeline questions "
        "completed."
    )

    eval_dataset = build_dataset(
        results
    )

    print(
        "RAGAS dataset created."
    )

    # --------------------------------------------------------
    # Run RAGAS
    # --------------------------------------------------------

    ragas_results = (
        run_ragas_evaluation(
            eval_dataset
        )
    )

    # --------------------------------------------------------
    # Print final scores
    # --------------------------------------------------------

    print_results(
        ragas_results
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_evaluation()