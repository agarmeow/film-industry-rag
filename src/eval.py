"""
Automated Evaluation & Benchmark Scoring Script for Step 0 Baseline.
Runs all questions from data/eval_set.json through the RAG system,
calculates Retrieval Hit Rate @ 5 and Answer Correctness, and outputs eval_results.csv.
"""

import os
import ssl
import json
import csv
import time
import re
import sys

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


def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())


def check_retrieval_hit(source_doc: str, retrieved_sources: list) -> bool:
    norm_source = normalize(source_doc)
    for s in retrieved_sources:
        if norm_source in normalize(s) or normalize(s) in norm_source:
            return True
    return False


def check_answer_correctness(ref_answer: str, generated_answer: str) -> bool:
    """
    Evaluates factual correctness by checking whether key entities
    from reference_answer are preserved in generated_answer.
    """
    ref_norm = normalize(ref_answer)
    gen_norm = normalize(generated_answer)
    
    # Direct substring match
    if ref_norm in gen_norm:
        return True
    
    # Key tokens match
    tokens = [t for t in ref_answer.replace(',', ' ').split() if len(t) > 2]
    if tokens:
        matched = sum(1 for t in tokens if normalize(t) in gen_norm)
        if matched / len(tokens) >= 0.5:
            return True
            
    return False


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
    total_latency_ms = 0

    for i, item in enumerate(eval_items, 1):
        q_id = item["id"]
        q_text = item["question"]
        ref_ans = item["reference_answer"]
        src_doc = item["source_doc"]

        start_time = time.time()
        response = ask_question(query=q_text, top_k=5)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        total_latency_ms += elapsed_ms

        gen_ans = response.answer
        ret_sources = response.sources

        hit = check_retrieval_hit(src_doc, ret_sources)
        correct = check_answer_correctness(ref_ans, gen_ans)

        if hit:
            hits += 1
        if correct:
            correct_answers += 1

        top_score = response.retrieved_chunks[0].score if response.retrieved_chunks else 0.0

        results.append({
            "id": q_id,
            "question": q_text,
            "reference_answer": ref_ans,
            "source_doc": src_doc,
            "retrieved_sources": "; ".join(ret_sources),
            "hit_at_5": hit,
            "answer_correct": correct,
            "top_similarity_score": round(top_score, 4),
            "latency_ms": elapsed_ms,
            "generated_answer": gen_ans,
        })

        status_symbol = "[PASS]" if (hit and correct) else ("[WARN]" if hit else "[FAIL]")
        print(f"[{i:02d}/{len(eval_items)}] {status_symbol} Hit@5: {hit} | Correct: {correct} | Score: {top_score:.3f} | {elapsed_ms}ms")

    # Metrics Summary
    hit_rate = (hits / len(eval_items)) * 100
    accuracy = (correct_answers / len(eval_items)) * 100
    avg_latency = total_latency_ms / len(eval_items)

    print("\n" + "=" * 60)
    print("EVALUATION BENCHMARK RESULTS — STEP 0 BASELINE")
    print("=" * 60)
    print(f"Total Questions Evaluated:  {len(eval_items)}")
    print(f"Retrieval Hit Rate @ 5:    {hit_rate:.2f}% ({hits}/{len(eval_items)})")
    print(f"Answer Accuracy:           {accuracy:.2f}% ({correct_answers}/{len(eval_items)})")
    print(f"Average Response Latency:  {avg_latency:.2f} ms")
    print("=" * 60)

    # Save to CSV
    fieldnames = [
        "id", "question", "reference_answer", "source_doc",
        "retrieved_sources", "hit_at_5", "answer_correct",
        "top_similarity_score", "latency_ms", "generated_answer"
    ]
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDetailed evaluation results saved to '{OUTPUT_CSV_PATH}'")


if __name__ == "__main__":
    run_eval()
