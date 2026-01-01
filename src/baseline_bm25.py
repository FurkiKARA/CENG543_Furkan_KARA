import json
import rank_bm25
from rank_bm25 import BM25Okapi
import nltk # The Natural Language Toolkit (NLTK) is a Python library for working with human language data.

# NLTK has hundreds of data packages for different languages and tasks.
# If NLTK included all of them by default, the installation would be 50GB size.
# Instead, NLTK makes us download only the specific package we need.
# And for this project we need both 'punkt' and 'punkt_tab' for the tokenizer to work

# These try-except blocks looks if packages are already downloaded and downloads it if not.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# We need tokenizer to peel the punctuation off the words so that  search engine can recognize them properly.
# If we used .split() computer would think "law." and "law" are completely different however
# tokenizer doesn't fall to this mistake.
from nltk.tokenize import word_tokenize


def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # json.loads() takes the string '{"id": "..."}'
            # and turns it into a Python Dictionary.
            data.append(json.loads(line))
            # Why not just use json.load()?
            # If we tried to use standard json.load(f) on this file, it would crash.
            # Standard JSON parsers expect the whole file to be wrapped in [] and separated by commas.
            # This file doesn't have those, so we have to read it manually line-by-line.
    return data


print("1. Loading Data...")
corpus = load_jsonl("data/processed/corpus.jsonl")
queries = load_jsonl("data/processed/queries.jsonl")

# Preparing corpus for BM25 (Tokenization)
doc_ids = [doc['_id'] for doc in corpus] # doc_ids (The Labels): ["doc_0", "doc_1", "doc_2"]
corpus_texts = [doc['text'] for doc in corpus] # corpus_texts (The Content)


# 1. tokenized_corpus = [...] (The Prep Work)
# This single line does three heavy jobs at once using a "List Comprehension":
# .lower(): It turns "Law" and "LAW" into "law". Otherwise, the computer thinks they are two different things.
# word_tokenize (Chopping): As discussed, it uses the "smart scissors" to split words and punctuation.
# The Structure: It converts list of strings into a List of Lists.
# Visualizing the change:
# Input (corpus_texts): ["The Law is here.", "Taxes are high."]
# Output (tokenized_corpus): [['the', 'law', 'is', 'here', '.'], ['taxes', 'are', 'high', '.']]
print("2. Indexing Corpus (This might take a minute)...")
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

print("3. Searching...")
output_file = "outputs/run_bm25.txt"

# This is the "Main Search Loop."
# It takes every single question, finds the best answers for it, and writes them down.
with open(output_file, 'w') as f_out:
    for i, query in enumerate(queries):
        if i % 100 == 0:
            print(f"Processing query {i}/{len(queries)}...")

        qid = query['_id']
        q_text = query['text']

        tokenized_query = word_tokenize(q_text.lower()) # So that it is in the same format as document

        # This compares query against all documents in corpus at once.
        # It returns a huge list of just numbers, like: [0.12, 4.5, 0.0, 1.2, ...].
        #
        # Index 0 is the score for doc_0.
        # Index 1 is the score for doc_1.
        scores = bm25.get_scores(tokenized_query)

        # The scores list is just numbers; it forgot which document is which. zip glues the Name Tag back onto the Score.
        # Before: ['doc_A', 'doc_B'] and [1.5, 9.2]
        # After Zip: [('doc_A', 1.5), ('doc_B', 9.2)]

        # Zip scores with Doc IDs and sort by score (highest first)
        # We take the top 100 candidates
        scores_with_ids = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:100]

        for rank, (doc_id, score) in enumerate(scores_with_ids):
            # TREC Format: query_id Q0 doc_id rank score run_id
            # query_id: The Question ID.
            # Q0: The "Dummy" column (required standard).
            # doc_id: The Answer ID.
            # rank: The Rank (1st place).
            # score: The BM25 Score.
            # BM25: The "Run ID" (A name tag for this specific experiment).
            line = f"{qid} Q0 {doc_id} {rank + 1} {score:.4f} BM25\n"
            f_out.write(line)

print(f"Done! Results saved to {output_file}")