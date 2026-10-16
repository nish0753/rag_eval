import os, sys, json
from dotenv import load_dotenv

# Ensure root directory is on Python path regardless of how the script is called
sys.path.insert(0, os.path.abspath("."))

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.reranker import RerankingRetriever

load_dotenv()

GOLDEN_PATH = "golden/retriever_golden.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7


# 1. LOAD the golden set --- the fixed, human-authored truth
print(f"📖 Loading golden dataset from '{GOLDEN_PATH}'...")
with open(GOLDEN_PATH) as f:
    goldens = json.load(f)
print(f"✅ Loaded {len(goldens)} test cases.")


# 2. RUN THE RETRIEVER on each question to fill retrieval_context,
#    then build one test case per golden.
print("🤖 Initializing RerankingRetriever (Cross-Encoder)...")
retriever = RerankingRetriever()

test_cases = []
print(f"🔍 Retrieving & neural reranking candidates across {len(goldens)} questions...")

for i, g in enumerate(goldens, 1):
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


# 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
metrics = [
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]


# 4. EVALUATE --- every metric on every case, batched + parallel, with a printed report
print(f"⚖️ Running DeepEval judge ({JUDGE_MODEL}) on {len(test_cases)} test cases...")
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters={
        "retriever": "reranker",          # vs "reranked" when you swap it in
        "embedding_model": "text-embedding-3-small",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 5,
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH,
    },
)