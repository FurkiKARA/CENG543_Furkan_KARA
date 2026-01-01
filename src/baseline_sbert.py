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
# If current device has a GPU, this puts it on the GPU. If not, CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Select a specific pre-trained AI model to turn text into vectors so the computer can understand similarity.
#
# paraphrase: It was trained to recognize when two different sentences mean the same thing.
# This makes it excellent for search/similarity tasks.
# multilingual: It understands over 50 languages (including English, Turkish, Spanish, etc.).
# MiniLM: It is a "distilled" version. It is designed to be fast and lightweight while maintaining high accuracy.
# L12: It has 12 layers (depth of the neural network).
# v2: It is the second, improved version of this model.
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
# SentenceTransformer function downloads the weights for the model name defined above.
model = SentenceTransformer(model_name, device=device)

print("2. Loading Data...")
corpus = load_jsonl("data/processed/corpus.jsonl")
queries = load_jsonl("data/processed/queries.jsonl")

# Extract texts for the model to encode and preserve IDs to map results back later
corpus_texts = [doc['text'] for doc in corpus]
doc_ids = [doc['_id'] for doc in corpus]
query_texts = [q['text'] for q in queries]
query_ids = [q['_id'] for q in queries]

# 3. Encode Everything (Batched)
print(f"Encoding {len(corpus)} documents...")
start_time = time.time()
# model.encode(corpus_texts): This feeds every document in corpus_texts list into the NN.
# The model reads the text and outputs a vector for each document. The vector represents the meaning of the text.
# convert_to_tensor=True: Instead of a standard Python list or NumPy array, this forces the output to be a PyTorch Tensor.
# corpus_embeddings is a giant matrix where every row represents one document's meaning.
corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)

print(f"Encoding {len(queries)} queries...")
query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)

# 4. Search (Matrix Multiplication)
print("Searching (Batch Mode)...")
# util.semantic_search is a helper function from the Sentence Transformers library that
# efficiently finds the most semantically similar items between two sets of embeddings.
# query_embeddings: Tensor or NumPy array of query vectors.
# corpus_embeddings: Tensor or NumPy array of corpus vectors.
# top_k (int, default=10): Number of top matches to return for each query.
# score_function (optional): Similarity metric (default is cosine similarity).
#
# It returns a list of lists:
# [
#   [ {'corpus_id': int, 'score': float}, ... ],  # Matches for query 1
#   [ {'corpus_id': int, 'score': float}, ... ],  # Matches for query 2
#   ...
# ]
hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=100)

print("5. Saving Results...")
output_file = "outputs/run_sbert.txt"

with open(output_file, 'w') as f_out:
    # Loop through every query and its hits
    for i, query_hits in enumerate(hits):
        # Retrieve the original Query ID string (e.g., "q_1") using the current index
        qid = query_ids[i]

        # Loop through the top 100 matches found for this specific query
        for rank, hit in enumerate(query_hits):
            # Map the internal integer index (e.g., 502) back to the original Document ID string
            doc_id = doc_ids[hit['corpus_id']]
            score = hit['score']

            # TREC Format
            line = f"{qid} Q0 {doc_id} {rank + 1} {score:.4f} SBERT\n"
            f_out.write(line)

total_time = (time.time() - start_time) / 60
print(f"Done! Processed {len(queries)} queries in {total_time:.2f} minutes.")
