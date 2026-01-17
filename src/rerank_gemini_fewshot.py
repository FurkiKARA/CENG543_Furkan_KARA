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
Query: "Anayasa, egemenliğin kime ait olduğunu nasıl belirtiyor?"
Documents:
[1] "Madde 6 - Egemenlik, kayıtsız şartsız Milletindir. Türk Milleti, egemenliğini, Anayasanın koyduğu esaslara göre, yetkili organları eliyle kullanır."
[2] "Madde 7 - Yasama yetkisi Türk Milleti adına Türkiye Büyük Millet Meclisinindir. Bu yetki devredilemez."
[3] "Madde 41 - Seferberlik sırasında... işyerlerinde fazla çalışmaya lüzum görülürse Cumhurbaşkanı günlük çalışma süresini... çıkarabilir."
Ranking: [1] > [2] > [3]
-------------------------------------------------------------
--- EXAMPLE 2 ---
Query: "Anayasaya göre Türk Vatanı ve Milletinin ebedi varlığı neye dayanır?"
Documents:
[1] "Başlangıç - Türk Vatanı ve Milletinin ebedi varlığını... belirleyen bu Anayasa, Atatürk’ün belirlediği milliyetçilik anlayışı ve onun inkılap ve ilkeleri doğrultusunda..."
[2] "Başlangıç - Dünya milletleri ailesinin eşit haklara sahip şerefli bir üyesi olarak, Türkiye Cumhuriyetinin ebedi varlığı, refahı... yönünde..."
[3] "Madde 44 - Ulusal bayram ve genel tatil günlerinde işyerlerinde çalışılıp çalışılmayacağı toplu iş sözleşmesi ile kararlaştırılır."
Ranking: [1] > [2] > [3]
-------------------------------------------------------------
--- EXAMPLE 3 ---
Query: "Sulh zamanında seferberlikle ilgili görevlerini ihmal eden kamu görevlisine ne ceza verilir?"
Documents:
[1] "Madde 324 - Sulh zamanında seferberlikle ilgili görevlerini ihmal eden veya geciktiren kamu görevlisine altı aydan üç yıla kadar hapis cezası verilir."
[2] "Madde 321 - Savaş zamanında Devletin yetkili makam ve mercilerinin emir veya kararlarına bilerek aykırı harekette bulunan kimseye bir yıldan altı yıla kadar hapis cezası verilir."
[3] "Madde 317 - Kanunen yetkili olmadıkları hâlde, bir asker kıtasının... komutasını alanlara müebbet hapis cezası verilir."
Ranking: [1] > [2] > [3]
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
        Rank the following documents based on their relevance to the query.
        Study the examples below to understand the ranking logic.

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
        Output the ranking in the format: Ranking: [1] > [2] ...
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

                if "Ranking:" in response_text:
                    final_part = response_text.split("Ranking:")[-1]
                else:
                    final_part = response_text.split("\n")[-1]

                # Regex to find numbers inside brackets, e.g., [3], [1], [2]
                ranked_indices = re.findall(r'\[(\d+)\]', final_part)

                # Fallback: If model fails to output brackets, preserve original order
                if not ranked_indices:
                    ranked_indices = [str(k) for k in range(1, len(doc_list) + 1)]

                # Write to File (TREC Format)
                rank = 1
                seen_indices = set()
                for idx in ranked_indices:
                    if idx in doc_map and idx not in seen_indices:
                        original_doc_id = doc_map[idx]
                        # Create a score: 1.0 for 1st, 0.5 for 2nd, 0.33 for 3rd...
                        # This makes the output compatible with evaluation metrics like MAP/NDCG
                        score = 1.0 / rank
                        f_out.write(f"{qid} Q0 {original_doc_id} {rank} {score:.4f} GEMINI_FEWSHOT\n")
                        seen_indices.add(idx)
                        rank += 1

                # Append missing docs if model forgot any
                for idx in range(1, len(doc_list) + 1):
                    s_idx = str(idx)
                    if s_idx not in seen_indices:
                        original_doc_id = doc_map[s_idx]
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