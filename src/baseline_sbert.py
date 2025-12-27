import json
import time
from sentence_transformers import SentenceTransformer, util
import torch


def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# 1. Load Model
# If you have a GPU, this puts it on the GPU. If not, CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name, device=device)

print("2. Loading Data...")
corpus = load_jsonl("corpus.jsonl")
queries = load_jsonl("queries.jsonl")

corpus_texts = [doc['text'] for doc in corpus]
doc_ids = [doc['_id'] for doc in corpus]
query_texts = [q['text'] for q in queries]
query_ids = [q['_id'] for q in queries]

# 3. Encode Everything (Batched)
print(f"Encoding {len(corpus)} documents...")
start_time = time.time()
corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)

print(f"Encoding {len(queries)} queries...")
# This is the speedup: Encoding all 15,000 queries in one go
query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)

# 4. Search (Matrix Multiplication)
print("Searching (Batch Mode)...")
# This finds top 100 matches for ALL 15,000 queries instantly using matrix math
hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=100)

print("5. Saving Results...")
output_file = "run_sbert.txt"

with open(output_file, 'w') as f_out:
    for i, query_hits in enumerate(hits):
        qid = query_ids[i]
        for rank, hit in enumerate(query_hits):
            doc_id = doc_ids[hit['corpus_id']]
            score = hit['score']

            # TREC Format
            line = f"{qid} Q0 {doc_id} {rank + 1} {score:.4f} SBERT\n"
            f_out.write(line)

total_time = (time.time() - start_time) / 60
print(f"Done! Processed {len(queries)} queries in {total_time:.2f} minutes.")