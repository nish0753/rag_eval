"""
evals/harness.py
================
Shared evaluation utilities for loading datasets and summarizing DeepEval results.
"""

import json
import os
from typing import Any, Dict, List


def load_goldens(path: str) -> List[Dict[str, Any]]:
    """Load golden test cases from a JSON file, with directory name fallback."""
    if not os.path.exists(path):
        if path.startswith("goldens/"):
            alt_path = path.replace("goldens/", "golden/", 1)
            if os.path.exists(alt_path):
                path = alt_path
        elif path.startswith("golden/"):
            alt_path = path.replace("golden/", "goldens/", 1)
            if os.path.exists(alt_path):
                path = alt_path

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_by_metric(result: Any) -> Dict[str, Dict[str, Any]]:
    """Summarize DeepEval EvaluationResult by metric name (average score and pass rate)."""
    test_results = getattr(result, "test_results", [])
    metrics_stats: Dict[str, Dict[str, Any]] = {}

    for test_result in test_results:
        metrics_data = getattr(test_result, "metrics_data", []) or []
        for metric in metrics_data:
            name = getattr(metric, "name", "Unknown Metric")
            score = getattr(metric, "score", 0.0)
            success = getattr(metric, "success", False)

            if name not in metrics_stats:
                metrics_stats[name] = {"scores": [], "passed": 0, "total": 0}

            metrics_stats[name]["scores"].append(score)
            metrics_stats[name]["total"] += 1
            if success:
                metrics_stats[name]["passed"] += 1

    summary = {}
    for name, stats in metrics_stats.items():
        total = stats["total"]
        scores = stats["scores"]
        passed = stats["passed"]
        avg_score = (sum(scores) / len(scores)) if scores else 0.0
        pass_rate = (passed / total) if total > 0 else 0.0
        summary[name] = {
            "avg_score": avg_score,
            "pass_rate": pass_rate,
            "passed": passed,
            "total": total,
        }

    return summary


def print_summary(component_name: str, summary: Dict[str, Any]) -> None:
    """Print a clean formatted summary of evaluation results."""
    print("\n" + "=" * 55)
    print(f"       EVALUATION SUMMARY: {component_name.upper()}")
    print("=" * 55)

    if not summary:
        print("No metric summary available.")
        print("=" * 55 + "\n")
        return

    for metric_name, stats in summary.items():
        if isinstance(stats, dict):
            avg_score = stats.get("avg_score", 0.0)
            pass_rate = stats.get("pass_rate", 0.0) * 100
            passed = stats.get("passed", 0)
            total = stats.get("total", 0)
            print(f"📊 {metric_name}:")
            print(f"   • Average Score: {avg_score:.2f}")
            print(f"   • Pass Rate:     {pass_rate:.1f}% ({passed}/{total} passed)")
        else:
            print(f"📊 {metric_name}: {stats}")

    print("=" * 55 + "\n")
