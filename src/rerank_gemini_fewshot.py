import json
import time
import os
import re
from collections import defaultdict
from google import genai
from dotenv import load_dotenv

# CONFIGURATION
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in .env file.")
    exit()

client = genai.Client(api_key=API_KEY)

# SETTINGS
TEST_LIMIT = 100  # Run 100 queries for better stats
TOP_K_RERANK = 10  # Rerank top 10 results from BM25


def load_jsonl_dict(filename, key_field):
    data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data[item[key_field]] = item
    return data


def load_run_file(filename):
    candidates = defaultdict(list)
    # Read BM25 file to get candidates
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid = parts[0]
            docid = parts[2]
            rank = int(parts[3])
            if rank <= TOP_K_RERANK:
                candidates[qid].append(docid)
    return candidates


print("1. Loading Data for Few-Shot Reranking...")
corpus = load_jsonl_dict("data/processed/corpus.jsonl", "_id")
queries = load_jsonl_dict("data/processed/queries.jsonl", "_id")
# We rerank the output of BM25
bm25_candidates = load_run_file("outputs/run_bm25.txt")

print(f"2. Starting Few-Shot Reranking on {TEST_LIMIT} queries...")
output_file = "outputs/run_gemini_fewshot.txt"

# The Few Shot examples I am giving to prompt
few_shot_example = """
--- EXAMPLE 1 ---
Query: "Hırsızlık suçunun cezası nedir?"

Documents:
[1] "TCK Madde 141: Zilyedinin rızası olmadan başkasına ait taşınır bir malı..." (Relevant Law)
[2] "Borçlar Kanunu Madde 1: Sözleşme, tarafların iradelerini..." (Irrelevant)
[3] "TCK Madde 142: Nitelikli hırsızlık halleri şunlardır..." (Highly Relevant)

Ranking: [1] > [3] > [2]
-------------------------------------------------------------
--- EXAMPLE 2 ---
Query: "İşçi yıllık ücretli izne ne zaman hak kazanır?"

Documents:
[1] "İşveren, işyerinde iş sağlığı ve güvenliği önlemlerini almakla yükümlüdür." (Irrelevant)
[2] "İşçilere verilecek yıllık ücretli izin süresi, hizmet süresi bir yıldan beş yıla kadar olanlara on dört günden az olamaz." (Relevant Law)
[3] "İşyerinde işe başladığı günden itibaren, deneme süresi de içinde olmak üzere, en az bir yıl çalışmış olan işçilere yıllık ücretli izin verilir." (Highly Relevant)

Ranking: [3] > [2] > [1]
-----------------
"""

with open(output_file, 'w') as f_out:
    count = 0
    for qid, doc_list in bm25_candidates.items():
        if count >= TEST_LIMIT:
            break

        if qid not in queries: continue

        query_text = queries[qid]['text']

        # Constructing the Prompt
        prompt = f"""You are an expert Turkish Lawyer and Judge.
Your task is to rank the provided documents based on their relevance to the user query.
Use the logical reasoning of a legal expert.

{few_shot_example}

NOW IT IS YOUR TURN:
Query: {query_text}

Documents:
"""
        doc_map = {}
        for idx, doc_id in enumerate(doc_list):
            # Truncated text to 500 chars to save tokens/speed, this could be increased.
            doc_text = corpus[doc_id]['text'][:500]
            prompt += f"[{idx + 1}] {doc_text}\n\n"
            doc_map[str(idx + 1)] = doc_id

        prompt += """
Output ONLY the ranking as a list of numbers: [1] > [2] ...
Ranking:"""

        try:
            if count % 10 == 0:
                print(f"Processing {count}/{TEST_LIMIT}: {qid}...")

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            response_text = response.text.strip()

            # Parsing numbers like [1], [2]
            ranked_indices = re.findall(r'\[(\d+)\]', response_text)

            # Fallback if model fails to output numbers
            if not ranked_indices:
                ranked_indices = [str(k) for k in range(1, len(doc_list) + 1)]

            rank = 1
            for idx in ranked_indices:
                if idx in doc_map:
                    original_doc_id = doc_map[idx]
                    # Score = 1/rank
                    score = 1.0 / rank
                    f_out.write(f"{qid} Q0 {original_doc_id} {rank} {score:.4f} GEMINI_FEWSHOT\n")
                    rank += 1

            count += 1

        except Exception as e:
            print(f"Error on {qid}: {e}")
            time.sleep(2)

print(f"Done! Results saved to {output_file}")