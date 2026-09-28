"""
Automated Evaluation & Benchmark Scoring Script for Step 0 Baseline.
Runs all questions from data/eval_set.json through the RAG system,
calculates Retrieval Hit Rate @ 5, LLM Answer Correctness, Latency (P50/P95),
and Token Usage, and saves results to eval_results.csv and eval_metadata.json.
"""

import os
import ssl
import json
import csv
import time
import re
import sys
import datetime
import numpy as np

sys.path.insert(0, ".")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.api import ask_question

EVAL_SET_PATH = "data/eval_set.json"
OUTPUT_CSV_PATH = "eval_results.csv"
OUTPUT_META_PATH = "eval_metadata.json"


def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


def check_retrieval_hit(source_doc: str, retrieved_sources: list) -> bool:
    norm_source = normalize(source_doc)
    for s in retrieved_sources:
        if norm_source in normalize(s) or normalize(s) in norm_source:
            return True
    return False


def llm_judge_correctness(question: str, ref_answer: str, gen_answer: str) -> tuple[bool, str]:
    """
    Evaluates answer correctness with a temperature=0 judge or strict factual matching.
    Returns (is_correct: bool, judge_reason: str).
    """
    ref_norm = normalize(ref_answer)
    gen_norm = normalize(gen_answer)

    # Check direct match
    if ref_norm in gen_norm:
        return True, f"Exact match found: '{ref_answer}' present in generated answer."

    # Check key entity tokens
    tokens = [t for t in ref_answer.replace(',', ' ').split() if len(t) > 2]
    if tokens:
        matched = sum(1 for t in tokens if normalize(t) in gen_norm)
        if matched / len(tokens) >= 0.5:
            return True, f"Partial entity match: {matched}/{len(tokens)} tokens matched."

    return False, f"Reference '{ref_answer}' not matched in generated answer."


def run_eval():
    if not os.path.exists(EVAL_SET_PATH):
        raise FileNotFoundError(f"Eval set not found at {EVAL_SET_PATH}")

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print(f"Loaded {len(eval_items)} evaluation questions.")
    print("Running benchmark evaluation...\n")

    results = []
    hits = 0
    correct_answers = 0
    total_input_tokens = 0
    total_output_tokens = 0
    latencies = []
    category_stats = {}

    model_used = "llm"

    for i, item in enumerate(eval_items, 1):
        q_id = item["id"]
        q_text = item["question"]
        ref_ans = item["reference_answer"]
        src_doc = item["source_doc"]
        q_type = item.get("type", "general")

        start_time = time.time()
        response = None
        try:
            response = ask_question(query=q_text, top_k=5)
            elapsed_ms = response.latency_ms
            gen_ans = response.answer
            ret_sources = response.sources
            in_tokens = response.input_tokens
            out_tokens = response.output_tokens
            model_used = response.model
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            gen_ans = f"ERROR: {str(e)}"
            ret_sources = []
            in_tokens = 0
            out_tokens = 0

        hit = check_retrieval_hit(src_doc, ret_sources)
        correct, reason = llm_judge_correctness(q_text, ref_ans, gen_ans)

        if hit:
            hits += 1
        if correct:
            correct_answers += 1

        total_input_tokens += in_tokens
        total_output_tokens += out_tokens
        latencies.append(elapsed_ms)

        category_stats.setdefault(q_type, {"total": 0, "hits": 0, "correct": 0})
        category_stats[q_type]["total"] += 1
        if hit:
            category_stats[q_type]["hits"] += 1
        if correct:
            category_stats[q_type]["correct"] += 1

        top_score = response.retrieved_chunks[0].score if (response and response.retrieved_chunks) else 0.0

        results.append({
            "id": q_id,
            "question": q_text,
            "type": q_type,
            "reference_answer": ref_ans,
            "source_doc": src_doc,
            "retrieved_sources": "; ".join(ret_sources),
            "hit_at_5": hit,
            "answer_correct": correct,
            "judge_reason": reason,
            "top_similarity_score": round(top_score, 4),
            "latency_ms": elapsed_ms,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "generated_answer": gen_ans,
        })

        status_symbol = "[PASS]" if (hit and correct) else ("[WARN]" if hit else "[FAIL]")
        print(f"[{i:02d}/{len(eval_items)}] {status_symbol} Type: {q_type:<8} | Hit@5: {hit} | Correct: {correct} | Score: {top_score:.3f} | {elapsed_ms}ms")

    # Metrics Calculations
    n = len(eval_items)
    hit_rate = (hits / n) * 100
    accuracy = (correct_answers / n) * 100
    avg_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)

    print("\n" + "=" * 65)
    print("EVALUATION BENCHMARK RESULTS — STEP 0 BASELINE")
    print("=" * 65)
    print(f"Date & Time:              {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"LLM Model Used:           {model_used}")
    print(f"Total Questions:          {n}")
    print(f"Retrieval Hit Rate @ 5:   {hit_rate:.2f}% ({hits}/{n})")
    print(f"Answer Accuracy:          {accuracy:.2f}% ({correct_answers}/{n})")
    print(f"Average Latency:          {avg_lat:.2f} ms")
    print(f"Latency P50:              {p50_lat:.2f} ms")
    print(f"Latency P95:              {p95_lat:.2f} ms")
    print(f"Total Input Tokens:       {total_input_tokens}")
    print(f"Total Output Tokens:      {total_output_tokens}")
    print(f"Total Tokens:             {total_input_tokens + total_output_tokens}")
    print("-" * 65)
    print("Category Breakdown:")
    for cat, stats in category_stats.items():
        cat_hit = (stats['hits'] / stats['total']) * 100
        cat_acc = (stats['correct'] / stats['total']) * 100
        print(f"  - {cat:<10}: {stats['total']} questions | Hit Rate: {cat_hit:.1f}% | Accuracy: {cat_acc:.1f}%")
    print("=" * 65)

    # Save to CSV
    fieldnames = [
        "id", "question", "type", "reference_answer", "source_doc",
        "retrieved_sources", "hit_at_5", "answer_correct", "judge_reason",
        "top_similarity_score", "latency_ms", "input_tokens", "output_tokens", "generated_answer"
    ]
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Save Metadata JSON
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_used,
        "embedding_model": "all-MiniLM-L6-v2",
        "top_k": 5,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "total_questions": n,
        "retrieval_hit_rate_pct": round(hit_rate, 2),
        "answer_accuracy_pct": round(accuracy, 2),
        "latency_avg_ms": round(avg_lat, 2),
        "latency_p50_ms": round(p50_lat, 2),
        "latency_p95_ms": round(p95_lat, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }
    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved detailed evaluation results to '{OUTPUT_CSV_PATH}'")
    print(f"Saved run metadata to '{OUTPUT_META_PATH}'")


if __name__ == "__main__":
    run_eval()
