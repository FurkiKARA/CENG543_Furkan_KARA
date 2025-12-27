import ir_measures
from ir_measures import *
import os

# CONFIGURATION
QRELS_FILE = "data/processed/qrels.tsv"
RUN_FILES = {
    "BM25 (Baseline)": "outputs/run_bm25.txt",
    "S-BERT (Dense)": "outputs/run_sbert.txt",
    "Gemini (Zero-Shot)": "outputs/run_gemini_zeroshot.txt",
    "Gemini (Few-Shot)": "outputs/run_gemini_fewshot.txt"
}
METRICS = [MAP, nDCG @ 10, R @ 10]


def load_run_ids(filepath):
    """Reads Run file to get the set of Query IDs found."""
    ids = set()
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                ids.add(parts[0])
    return ids


print("=== DIAGNOSTIC EVALUATION (FIXED) ===")

# 1. Load Qrels into memory (List) to allow multiple passes
if not os.path.exists(QRELS_FILE):
    print(f"CRITICAL ERROR: {QRELS_FILE} not found.")
    exit()

# FIX: Convert generator to list immediately
qrels = list(ir_measures.read_trec_qrels(QRELS_FILE))
print(f"QRELS Loaded. Total Qrels objects: {len(qrels)}")
print("-" * 60)

# 2. Evaluate Each System
results_table = []

for system_name, file_path in RUN_FILES.items():
    print(f"\nProcessing: {system_name}")

    if not os.path.exists(file_path):
        results_table.append({'System': system_name, 'Status': 'Missing File'})
        continue

    # Get IDs from the run file
    run_ids = load_run_ids(file_path)

    # FILTERING: Create a subset of Qrels that only contains queries found in this run
    # This ensures "Partial Runs" (like Gemini's 50 queries) get a fair score (0-1.0)
    # instead of being divided by 15,000 (which would make the score ~0.0001)
    filtered_qrels = [q for q in qrels if q.query_id in run_ids]

    count_eval = len(set(q.query_id for q in filtered_qrels))
    print(f"  -> Queries to evaluate: {count_eval}")

    if count_eval == 0:
        results_table.append({'System': system_name, 'Status': 'ID Mismatch'})
        continue

    try:
        run = ir_measures.read_trec_run(file_path)

        # Calculate scores on the filtered subset
        scores = ir_measures.calc_aggregate(METRICS, filtered_qrels, run)

        print(f"  -> MAP: {scores[MAP]:.4f}")

        results_table.append({
            'System': system_name,
            'Status': 'OK',
            'Count': count_eval,
            'MAP': scores[MAP],
            'nDCG@10': scores[nDCG @ 10],
            'Recall@10': scores[R @ 10]
        })

    except Exception as e:
        print(f"  -> EXCEPTION: {e}")
        results_table.append({'System': system_name, 'Status': f"Error: {str(e)[:20]}"})

# 3. Final Report
print("\n" + "=" * 85)
print(f"{'System':<20} | {'Count':<5} | {'MAP':<8} | {'nDCG@10':<8} | {'Recall@10':<8}")
print("-" * 85)
for row in results_table:
    if row['Status'] == 'OK':
        print(
            f"{row['System']:<20} | {row['Count']:<5} | {row['MAP']:.4f}   | {row['nDCG@10']:.4f}   | {row['Recall@10']:.4f}")
    else:
        print(f"{row['System']:<20} | {row['Status']}")
print("=" * 85)