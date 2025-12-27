import json
import time
import os
import re
from collections import defaultdict
from google import genai
from google.api_core import exceptions # We need this for the specific error
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

client = genai.Client(api_key=API_KEY)
model = client.models


# --- SETUP CONSTANTS ---
TEST_LIMIT = 50
TOP_K = 10

def load_jsonl_dict(filename, key_field):
    data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data[item[key_field]] = item
    return data

def load_run_file(filename):
    candidates = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid = parts[0]
            docid = parts[2]
            rank = int(parts[3])
            if rank <= TOP_K:
                candidates[qid].append(docid)
    return candidates

print("1. Loading Data...")
corpus = load_jsonl_dict("corpus.jsonl", "_id")
queries = load_jsonl_dict("queries.jsonl", "_id")
bm25_candidates = load_run_file("run_bm25.txt")

print(f"2. Starting Reranking on first {TEST_LIMIT} queries...")
output_file = "run_gemini_zeroshot.txt"

# Open file in append mode ('a') so we don't lose progress if we restart manually
# BUT for a fresh run, change to 'w'
mode = 'w' 
with open(output_file, mode) as f_out:
    count = 0
    
    for qid, doc_list in bm25_candidates.items():
        if count >= TEST_LIMIT:
            break

        if qid not in queries: continue

        query_text = queries[qid]['text']

        # Prompt
        prompt = f"""You are an expert Turkish lawyer.
Rank these documents by relevance to the query.
Query: {query_text}

Documents:
"""
        doc_map = {}
        for idx, doc_id in enumerate(doc_list):
            doc_text = corpus.get(doc_id, {}).get('text', '')[:500] 
            prompt += f"[{idx + 1}] {doc_text}\n\n"
            doc_map[str(idx + 1)] = doc_id

        prompt += """
Output ONLY the ranking as a list of numbers: [1] > [2]
Ranking:"""

        # --- RETRY LOGIC ---
        while True:
            try:
                print(f"Ranking Query {count + 1}/{TEST_LIMIT}: {qid}...")

                MODEL_NAME = "models/gemini-2.0-flash"

                # LIBRARY CALL
                response = model.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )

                # Check if response was blocked
                if not response.parts:
                    print(f"!! Blocked response for {qid}. Skipping.")
                    break

                response_text = response.text.strip()
                ranked_indices = re.findall(r'\[(\d+)\]', response_text)

                # Fallback
                if not ranked_indices:
                    ranked_indices = [str(k) for k in range(1, len(doc_list) + 1)]

                rank = 1
                for idx in ranked_indices:
                    if idx in doc_map:
                        original_doc_id = doc_map[idx]
                        f_out.write(f"{qid} Q0 {original_doc_id} {rank} {1.0 / rank:.4f} GEMINI\n")
                        rank += 1
                
                count += 1
                # Wait 4 seconds to avoid hitting the limit again too soon
                time.sleep(10.0)
                break 

            except exceptions.ResourceExhausted:
                # This catches the 429 Error
                print(">> Rate limit hit (429). Sleeping for 60 seconds...")
                time.sleep(60)
                # Loop restarts and tries again automatically
                
            except Exception as e:
                print(f"!! Fatal Error on {qid}: {e}")
                break 

print(f"Done! Results saved to {output_file}")