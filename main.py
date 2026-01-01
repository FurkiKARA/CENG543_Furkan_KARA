import subprocess
import sys
import time
import os

def run_step(script_name, description):
    """Runs a python script in order so that user doesn't have to run every file manually."""
    script_path = os.path.join("src", script_name)

    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running: {script_path}...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Check if file exists before running
    if not os.path.exists(script_path):
        print(f"❌ ERROR: File '{script_path}' not found. Make sure it is in correct folder.")
        sys.exit(1)

    try:
        # sys.executable ensures we use the same Python environment (virtualenv)
        # check=True raises an error if the script fails
        subprocess.run([sys.executable, script_path], check=True)
        
        elapsed = time.time() - start_time
        print(f"\n✅ SUCCESS: {script_name} finished in {elapsed:.2f} seconds.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {script_name} crashed with error code {e.returncode}.")
        print("Pipeline stopped.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user.")
        sys.exit(0)

# PIPELINE CONFIGURATION
pipeline_steps = [
    ("prepare_data.py",         "Converting raw CSV/Excel to Corpus/Query JSONL"),
    ("baseline_bm25.py",        "Running Sparse Retrieval (BM25) Baseline"),
    ("baseline_sbert.py",       "Running Dense Retrieval (S-BERT) Baseline"),
    ("rerank_gemini.py",        "Running Zero-Shot Reranking"),
    ("rerank_gemini_fewshot.py","Running Few-Shot Reranking"),
]

# Scripts that are in the ROOT folder (not src/)
root_scripts = [
    ("evaluate.py",             "Calculating MAP, nDCG, and Recall Scores"),
    ("plot_results.py",         "Generating Performance Comparison Charts")
]

if __name__ == "__main__":
    total_start = time.time()
    print("🚀 STARTING ORGANIZED RETRIEVAL PIPELINE")

    # 1. Run Source Scripts
    for script, desc in pipeline_steps:
        run_step(script, desc)

    # 2. Run Evaluation/Plotting
    for script, desc in root_scripts:
        print(f"\nSTEP: {desc}")
        subprocess.run([sys.executable, script], check=True)

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}\n ALL STEPS COMPLETED in {total_time:.2f} seconds!\n{'=' * 60}")