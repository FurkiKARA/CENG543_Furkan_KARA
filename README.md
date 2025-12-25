# Few-Shot and Zero-Shot Retrieval using Prompt Engineering

**Student:** Furkan KARA (300201046)  
**Course:** CENG543 - Term Project

## Project Overview
This project investigates the effectiveness of Large Language Models (LLMs) for information retrieval in the legal domain. It implements a **"Retrieve-then-Rerank"** pipeline that compares:
1.  **BM25 (Sparse Baseline):** Traditional keyword-based retrieval.
2.  **Sentence-BERT (Dense Baseline):** Semantic similarity retrieval.
3.  **Gemini Zero-Shot Reranking:** Using LLM prompt engineering to rank documents without examples.
4.  **Gemini Few-Shot Reranking:** Using LLM prompt engineering with in-context examples.

The project uses the **Turkish Law Dataset** for evaluation.

## Installation

1. **Clone or download** this repository.
2. **Install dependencies** using the provided requirements file: <br>
   ` pip install -r requirements.txt `
3. Set up API Key: <br>
Create a file named .env in the root directory.
Add your Google Gemini API key:
GOOGLE_API_KEY=your_api_key_here

## Usage Pipeline
Run the scripts in the following order to reproduce the results.

**1. Data Preparation** <br>
Converts raw data into the standard corpus, queries, and qrels (TREC format). <br>
`python prepare_data.py` <br>
`python fix_qrels.py`

**2. Run Baselines** <br>
Generates the initial retrieval runs using BM25 and S-BERT. <br>
`python baseline_bm25.py` <br>
`python baseline_sbert.py`

**3. Run LLM Reranking** <br>
Uses the Gemini API to rerank the top results from BM25. Note: These scripts require a valid API key and internet connection. <br>
`python rerank_gemini.py`          # Zero-Shot <br>
`python rerank_gemini_fewshot.py`  # Few-Shot

**4. Evaluation** <br>
Calculates MAP, nDCG@10, and Recall@10 for all systems. <br>
`python final_eval.py`
 
**5. Visualization (Optional)** <br>
Generates a bar chart comparison of the results. <br>
`python plot_results.py`

## File Structure
**prepare_data.py:** Preprocesses CSV/Excel data into JSONL format.<br>
**baseline_bm25.py:** Implements sparse retrieval using BM25Okapi.<br>
**baseline_sbert.py:** Implements dense retrieval using paraphrase-multilingual-MiniLM-L12-v2.<br>
**rerank_gemini.py:** Zero-shot reranking script.<br>
**rerank_gemini_fewshot.py:** Few-shot reranking script with prompt engineering.<br>
**final_eval.py:** Evaluation script using ir_measures.<br>
**requirements.txt:** List of Python dependencies.


## Final Check
1.  **Code:** All scripts are fixed and working.
2.  **Config:** `requirements.txt` and `README.md` are ready.
3.  **Results:** You have the evaluation table and the plot.

## Results Summary
Gemini-based reranking significantly outperforms traditional retrieval methods on the Turkish Law Dataset. Zero-shot reranking achieves the best overall performance (MAP: 0.7633, nDCG@10: 0.8040), followed closely by few-shot prompting. BM25 remains a strong baseline, while Sentence-BERT underperforms without domain adaptation. These results demonstrate that prompt engineering alone can yield competitive retrieval performance without fine-tuning.

![Results Comparison](results_chart.png)