import json
import rank_bm25
from rank_bm25 import BM25Okapi
import nltk

# --- THE FIX IS HERE ---
# We need both 'punkt' and 'punkt_tab' for the tokenizer to work
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
# -----------------------

from nltk.tokenize import word_tokenize


def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


print("1. Loading Data...")
corpus = load_jsonl("corpus.jsonl")
queries = load_jsonl("queries.jsonl")

# Prepare corpus for BM25 (Tokenization)
doc_ids = [doc['_id'] for doc in corpus]
corpus_texts = [doc['text'] for doc in corpus]

print("2. Indexing Corpus (This might take a minute)...")
# We use a simple whitespace tokenizer if word_tokenize fails, but word_tokenize is better
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

print("3. Searching...")
output_file = "run_bm25.txt"

with open(output_file, 'w') as f_out:
    # process only first 100 queries if you want a quick test,
    # but for the full project loop through all 'queries'
    for i, query in enumerate(queries):
        if i % 100 == 0:
            print(f"Processing query {i}/{len(queries)}...")

        qid = query['_id']
        q_text = query['text']

        tokenized_query = word_tokenize(q_text.lower())

        # Get scores
        scores = bm25.get_scores(tokenized_query)

        # Zip scores with Doc IDs and sort by score (highest first)
        # We take the top 100 candidates
        scores_with_ids = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:100]

        for rank, (doc_id, score) in enumerate(scores_with_ids):
            # TREC Format: qid Q0 docid rank score run_id
            line = f"{qid} Q0 {doc_id} {rank + 1} {score:.4f} BM25\n"
            f_out.write(line)

print(f"Done! Results saved to {output_file}")