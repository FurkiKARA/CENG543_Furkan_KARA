import ir_measures
import os
import pandas as pd

# CONFIGURATION

# Points to the "Judgments" file (Query Relevance), which contains the ground truth
QRELS_FILE = "data/processed/qrels.tsv"

# A dictionary containing the file paths for the outputs of four different systems:
RUN_FILES = {
    "BM25 (Baseline)": "outputs/run_bm25.txt", # BM25 (a standard statistical baseline)
    "S-BERT (Dense)": "outputs/run_sbert.txt", # S-BERT (a dense retrieval model)
    "Gemini (Zero-Shot)": "outputs/run_gemini_zeroshot.txt", # Gemini Zero-Shot (an LLM approach)
    "Gemini (Few-Shot)": "outputs/run_gemini_fewshot.txt" # Gemini Few-Shot (an LLM approach with examples)
}

# METRICS: It defines the specific metrics to calculate:
# MAP: Mean Average Precision (Focuses on precision across all recall levels)
# nDCG@10: Normalized Discounted Cumulative Gain at rank 10 (Focuses on ranking quality at top 10)
# R@10: Recall at rank 10 (Percentage of relevant documents retrieved in the top 10)
METRICS = [ir_measures.MAP, ir_measures.nDCG @ 10, ir_measures.R @ 10]


# HELPER FUNCTIONS
def load_run_ids(filepath: str) -> set:
    """
    Reads a TREC run file and returns a set of unique Query IDs present in it.

    Why this is needed:
        We need to know exactly which queries a specific system (like Gemini) attempted
        so we can filter the ground truth (qrels) to match only those queries.

    :param filepath: Path to the TREC run file.
    :return: A set of unique Query IDs strings.
    """
    assert os.path.exists(filepath)
    ids = set()
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                ids.add(parts[0])
    return ids


print("DIAGNOSTIC EVALUATION")


# LOADING GROUND TRUTH (QRELS)
# 1. Loading Qrels into memory to allow multiple passes
if not os.path.exists(QRELS_FILE):
    print(f"CRITICAL ERROR: {QRELS_FILE} not found.")
    exit()

# Load qrels into memory immediately.
# ir_measures.read_trec_qrels returns a generator. Converting to a list allows us
# to iterate over it multiple times (once for each system) without reloading the file.
qrels = list(ir_measures.read_trec_qrels(QRELS_FILE))
# The function ir_measures.read_trec_qrels is used to read TREC-style qrels
# files (relevance judgments) into a Python-friendly format for IR evaluation.

print(f"QRELS Loaded. Total Qrels objects: {len(qrels)}")
print("-" * 60)

# 2. Evaluating Each System
results_table = []

for system_name, file_path in RUN_FILES.items():
    print(f"\nProcessing: {system_name}")

    # Safety check: Ensure the run file actually exists
    if not os.path.exists(file_path):
        results_table.append({'System': system_name, 'Status': 'Missing File'})
        continue

    # Identify which queries this specific system actually answered.
    run_ids = load_run_ids(file_path)

    # Most evaluation tools assume  ALL queries in the dataset are attempted.
    # If the dataset has 10,000 queries but Gemini only processed 50, standard eval
    # treats the missing 9,950 as "scores of 0.0", resulting in a massive penalty.
    #
    # By filtering `qrels` to only include queries found in `run_ids`, we tell
    # the evaluator: "Only calculate the average score based on these specific queries."
    filtered_qrels = [q for q in qrels if q.query_id in run_ids]

    count_eval = len(set(q.query_id for q in filtered_qrels))
    print(f"  -> Queries to evaluate: {count_eval}")

    # If no matching queries are found, skip calculation to avoid division by zero errors
    # This safety measure is almost never used.
    if count_eval == 0:
        results_table.append({'System': system_name, 'Status': 'ID Mismatch'})
        continue

    try:
        # Read the run file using ir_measures
        run = ir_measures.read_trec_run(file_path)

        # Calculating aggregate scores (averages) for the filtered subset
        scores = ir_measures.calc_aggregate(METRICS, filtered_qrels, run)

        print(f"  -> MAP: {scores[ir_measures.MAP]:.4f}")

        results_table.append({
            'System': system_name,
            'Status': 'OK',
            'Count': count_eval,
            'MAP': scores[ir_measures.MAP],
            'nDCG@10': scores[ir_measures.nDCG @ 10],
            'Recall@10': scores[ir_measures.R @ 10]
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
        # Print error status if the run failed
        print(f"{row['System']:<20} | {row['Status']}")
print("=" * 85)

# Convert the list of dicts to a DataFrame
df_results = pd.DataFrame(results_table)
# Save only rows with status OK
df_results[df_results['Status'] == 'OK'].to_csv("outputs/evaluation_results.csv", index=False)
print("Saved results to outputs/evaluation_results.csv")