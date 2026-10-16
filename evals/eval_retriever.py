# evals/eval_retriever.py
import json
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

# 1. Use the working retriever from src/retriever.py
from src.retriever import build_retriever

load_dotenv()

# 2. Point to your actual golden dataset
GOLDEN_PATH = "golden/retriever_golden.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7


def run(retriever):
    # 3. Load the golden dataset
    with open(GOLDEN_PATH) as f:
        goldens = json.load(f)

    print(f"📊 Loaded {len(goldens)} golden test cases from '{GOLDEN_PATH}'.")

    # 4. Run retriever on each query to collect retrieval_context
    test_cases = []
    print("🔍 Fetching contexts from retriever...")
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 5. Metrics: Contextual Recall & Precision
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 6. Evaluate and print DeepEval report
    print(f"⚡ Running DeepEval evaluation with judge model '{JUDGE_MODEL}'...")
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "chroma",
            "embedding_model": "text-embedding-3-small",
            "top_k": 5,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
    )


if __name__ == "__main__":
    retriever = build_retriever()
    run(retriever)