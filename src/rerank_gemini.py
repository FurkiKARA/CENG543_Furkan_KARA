import json
import time
import os
import re
from collections import defaultdict
from google import genai
from google.api_core import exceptions
from dotenv import load_dotenv

# CONFIGURATION
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

client = genai.Client(api_key=API_KEY)

# CONSTANTS
TEST_LIMIT = 50
TOP_K = 10
OUTPUT_FILE = "outputs/run_gemini_zeroshot.txt"
MODEL_ID = "gemini-2.0-flash"  # chosen model


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
            qid, docid, rank = parts[0], parts[2], int(parts[3])
            if rank <= TOP_K:
                candidates[qid].append(docid)
    return candidates


# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

print("1. Loading Data...")
corpus = load_jsonl_dict("data/processed/corpus.jsonl", "_id")
queries = load_jsonl_dict("data/processed/queries.jsonl", "_id")
bm25_candidates = load_run_file("outputs/run_bm25.txt")

print(f"2. Starting Reranking on first {TEST_LIMIT} queries...")

with open(OUTPUT_FILE, 'w') as f_out:
    count = 0
    for qid, doc_list in bm25_candidates.items():
        if count >= TEST_LIMIT: break
        if qid not in queries: continue

        query_text = queries[qid]['text']
        prompt = f"You are an expert Turkish lawyer. Rank these documents by relevance to the query.\nQuery: {query_text}\n\nDocuments:\n"

        doc_map = {}
        for idx, doc_id in enumerate(doc_list):
            doc_text = corpus.get(doc_id, {}).get('text', '')[:500]
            prompt += f"[{idx + 1}] {doc_text}\n\n"
            doc_map[str(idx + 1)] = doc_id

        prompt += "\nOutput ONLY the ranking as a list of numbers: [1] > [2]\nRanking:"

        while True:
            try:
                print(f"Ranking Query {count + 1}/{TEST_LIMIT}: {qid}...")

                # API Call
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt
                )

                # Safety check: if no text is returned (blocked)
                if not response.text:
                    print(f"!! Blocked response for {qid}. Skipping.")
                    break

                response_text = response.text.strip()
                ranked_indices = re.findall(r'\[(\d+)\]', response_text)

                if not ranked_indices:
                    ranked_indices = [str(k) for k in range(1, len(doc_list) + 1)]

                for rank, idx in enumerate(ranked_indices, 1):
                    if idx in doc_map:
                        f_out.write(f"{qid} Q0 {doc_map[idx]} {rank} {1.0 / rank:.4f} GEMINI_ZERO\n")

                count += 1
                break

            except exceptions.ResourceExhausted:
                print(">> Rate limit hit (429). Sleeping for 30 seconds...")
                time.sleep(30)
            except Exception as e:
                print(f"!! Error on {qid}: {e}")
                break

print(f"Done! Results saved to {OUTPUT_FILE}")