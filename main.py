import subprocess
import sys
import time
import os

def run_step(script_name, description):
    """Runs a python script and handles errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running file: {script_name}...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Check if file exists before running
    if not os.path.exists(script_name):
        print(f"❌ ERROR: File '{script_name}' not found. Make sure it is in this folder.")
        sys.exit(1)

    try:
        # sys.executable ensures we use the same Python environment (virtualenv/conda)
        # check=True raises an error if the script fails
        subprocess.run([sys.executable, script_name], check=True)
        
        elapsed = time.time() - start_time
        print(f"\n✅ SUCCESS: {script_name} finished in {elapsed:.2f} seconds.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {script_name} crashed with error code {e.returncode}.")
        print("Pipeline stopped.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user.")
        sys.exit(0)

# --- DEFINING THE PIPELINE ORDER ---
pipeline_steps = [
    ("prepare_data.py",         "Converting raw CSV/Excel to Corpus/Query JSONL"),
    ("fix_qrels.py",            "Formatting Truth Data (Qrels) to TREC Format"),
    ("baseline_bm25.py",        "Running Sparse Retrieval (BM25) Baseline"),
    ("baseline_sbert.py",       "Running Dense Retrieval (S-BERT) Baseline"),
    ("rerank_gemini.py",        "Running Zero-Shot Reranking (Gemini 2.0 Flash)"),
    ("rerank_gemini_fewshot.py","Running Few-Shot Reranking (Gemini 2.0 Flash)"),
    ("evaluate.py",             "Calculating MAP, nDCG, and Recall Scores"),
    ("plot_results.py",         "Generating Performance Comparison Charts")
]

if __name__ == "__main__":
    total_start = time.time()
    print("🚀 STARTING RETRIEVAL SYSTEM PIPELINE")
    print(f"Found {len(pipeline_steps)} steps to execute.\n")

    for script, desc in pipeline_steps:
        run_step(script, desc)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f} seconds!")
    print(f"Check the folder for 'run_*.txt' files and the results plot.")
    print(f"{'='*60}")