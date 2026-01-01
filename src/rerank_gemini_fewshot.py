import json
import time
import os
import re
from collections import defaultdict
from google import genai
from google.genai import types
from dotenv import load_dotenv

# CONFIGURATION
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in .env file.")
    exit()

client = genai.Client(api_key=API_KEY)

# SETTINGS
TEST_LIMIT = 1500  # Limiting processing to the first 1500 queries to manage costs/time
TOP_K_RERANK = 10  # Rerank top 10 results from BM25


# HELPER FUNCTIONS
def load_jsonl_dict(filename: str, key_field: str) -> dict:
    """
        Reads a JSONL file (one JSON object per line) and converts it
        into a dictionary for fast O(1) lookups.

        Args:
            filename: Path to the .jsonl file.
            key_field: The field in the JSON to use as the dictionary key (e.g., '_id').
        """
    assert os.path.exists(filename)
    data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data[item[key_field]] = item
    return data


def load_run_file(filename: str) -> dict[str, list]:
    """
        Parses a standard TREC run file, and returns a dictionary
        where key=qid and value=[list of docids to rerank].
        Only includes documents ranked within TOP_K_RERANK.
    """
    assert os.path.exists(filename)
    candidates = defaultdict(list)
    # Read BM25 file to get candidates
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid = parts[0]
            docid = parts[2]
            rank = int(parts[3])

            # Filter: We only care about the top N documents from the first stage retrieval
            if rank <= TOP_K_RERANK:
                candidates[qid].append(docid)
    return candidates


# MAIN EXECUTION FLOW
print("1. Loading Data for Few-Shot Reranking...")
corpus = load_jsonl_dict("data/processed/corpus.jsonl", "_id")
queries = load_jsonl_dict("data/processed/queries.jsonl", "_id")
# Load the initial retrieval results (BM25) which we will re-order
bm25_candidates = load_run_file("outputs/run_bm25.txt")

print(f"2. Starting Few-Shot Reranking on {TEST_LIMIT} queries...")
output_file = "outputs/run_gemini_fewshot.txt"

# The Few Shot examples I am giving to prompt
few_shot_example = """
--- EXAMPLE 1 ---
Query: "Hırsızlık suçunun cezası nedir?"

Documents:
[1] "TCK Madde 141: Zilyedinin rızası olmadan başkasına ait taşınır bir malı..." 
[2] "Borçlar Kanunu Madde 1: Sözleşme, tarafların iradelerini..." 
[3] "TCK Madde 142: Nitelikli hırsızlık halleri şunlardır..." 

Ranking: [3] > [1] > [2]
-------------------------------------------------------------
--- EXAMPLE 2 ---
Query: "İşçi yıllık ücretli izne ne zaman hak kazanır?"

Documents:
[1] "İşveren, işyerinde iş sağlığı ve güvenliği önlemlerini almakla yükümlüdür." 
[2] "İşçilere verilecek yıllık ücretli izin süresi, hizmet süresi bir yıldan beş yıla kadar olanlara on dört günden az olamaz
[3] "İşyerinde işe başladığı günden itibaren, deneme süresi de içinde olmak üzere, en az bir yıl çalışmış olan işçilere yıllık ücretli izin verilir." 

Ranking: [3] > [2] > [1]
-----------------
"""

with open(output_file, 'w') as f_out:
    count = 0

    # Iterate through each query in the BM25 results
    for qid, doc_list in bm25_candidates.items():
        if count >= TEST_LIMIT:
            break

        # Safety check: ensure we actually have the query text
        if qid not in queries: continue

        query_text = queries[qid]['text']

        # Constructing the Prompt
        prompt = f"""You are an expert Turkish lawyer. 
Here are examples of how to rank documents based on relevance:

{few_shot_example}

Now, rank these documents for the following new query.
Query: {query_text}

Documents:
"""
        # Injecting the Candidate Documents (the top 10 from BM25)
        doc_map = {}
        for idx, doc_id in enumerate(doc_list):
            # Truncate text to 1000 chars to save tokens and fit context window
            doc_text = corpus[doc_id]['text'][:1000]
            # We use temporary indices [1], [2] for the LLM to reference
            prompt += f"[{idx + 1}] {doc_text}\n\n"
            doc_map[str(idx + 1)] = doc_id

        # Forcing the model to give the output in this specified format.
        prompt += """
Output ONLY the ranking as a list of numbers: [1] > [2] ...
Ranking:"""

        # API Call & Error Handling
        # 'while True' creates a retry loop if after encountering errors.
        while True:
            try:
                if count % 10 == 0:
                    print(f"Processing {count}/{TEST_LIMIT}: {qid}...")

                # Call Gemini API
                # temperature=0.0 makes the model deterministic (less creative), better for ranking
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.0)
                )

                # Check for empty response (blocked content)
                if not response.text:
                    print(f"!! Blocked/Empty response for {qid}. Skipping.")
                    break  # Break inner loop, move to next query

                # Response Parsing
                response_text = response.text.strip()
                # Regex to find numbers inside brackets, e.g., [3], [1], [2]
                ranked_indices = re.findall(r'\[(\d+)\]', response_text)

                # Fallback: If model fails to output brackets, preserve original order
                if not ranked_indices:
                    ranked_indices = [str(k) for k in range(1, len(doc_list) + 1)]


                # Write to File (TREC Format)
                rank = 1
                for idx in ranked_indices:
                    if idx in doc_map:
                        original_doc_id = doc_map[idx]
                        # Create a score: 1.0 for 1st, 0.5 for 2nd, 0.33 for 3rd...
                        # This makes the output compatible with evaluation metrics like MAP/NDCG
                        score = 1.0 / rank
                        f_out.write(f"{qid} Q0 {original_doc_id} {rank} {score:.4f} GEMINI_FEWSHOT\n")
                        rank += 1

                count += 1
                break  # Success! Break the while loop to move to next query

            except Exception as e:
                # Error handling block
                err_str = str(e)
                if "429" in err_str or "ResourceExhausted" in err_str:
                    print(f">> Rate limit on {qid}. Sleeping 30s...")
                    time.sleep(30)
                    # Loop continues automatically after sleep, retrying the call
                else:
                    print(f"!! Critical Error on {qid}: {e}")
                    # For non-recoverable errors, we break to avoid infinite loops
                    break

print(f"Done! Results saved to {output_file}")