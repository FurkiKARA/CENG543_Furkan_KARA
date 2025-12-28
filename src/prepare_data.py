import pandas as pd
import json

# CONFIGURATION
INPUT_FILE = "data/raw/raw_data.csv"

# Kaggle datasets usually have these column names.
COL_QUERY = "soru"  # Column containing the user question
COL_DOC = "cevap"  # Column containing the law/answer

# ----------------------------------------------------------------
print(f"1. Reading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# Basic cleanup
if COL_QUERY not in df.columns or COL_DOC not in df.columns:
    print(f"ERROR: Columns '{COL_QUERY}' or '{COL_DOC}' not found.")
    print(f"Your columns are: {list(df.columns)}")
    print("Please update the 'COL_QUERY' and 'COL_DOC' variables in the script.")
    exit()

print("2. Processing...")

# We need to handle duplicates.
# In real life, many questions might point to the same law article.
# We create a dictionary of Unique Documents to build our 'Corpus'
unique_docs = {}  # Text -> ID
doc_counter = 0

queries = []
qrels = []
corpus = []

for index, row in df.iterrows():
    query_text = str(row[COL_QUERY]).strip()
    doc_text = str(row[COL_DOC]).strip()

    # 1. Handle Document (The "Answer")
    # If we haven't seen this text before, give it a new ID
    if doc_text not in unique_docs:
        doc_id = f"doc_{doc_counter}"
        unique_docs[doc_text] = doc_id
        corpus.append({"_id": doc_id, "text": doc_text})
        doc_counter += 1
    else:
        doc_id = unique_docs[doc_text]

    # 2. Handle Query (The "Question")
    query_id = f"q_{index}"
    queries.append({"_id": query_id, "text": query_text})

    # 3. Handle Relation (The "Answer Key")
    # This query_id matches this doc_id
    qrels.append(f"{query_id}\t{doc_id}\t1")

print(f"3. Saving files...")

# Save Corpus (jsonl)
with open("data/processed/corpus.jsonl", "w", encoding="utf-8") as f:
    for item in corpus:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Save Queries (jsonl)
with open("data/processed/queries.jsonl", "w", encoding="utf-8") as f:
    for item in queries:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Save Qrels (tsv)
with open("data/processed/qrels.tsv", "w", encoding="utf-8") as f:
    f.write("query-id\tcorpus-id\tscore\n")  # Header
    for line in qrels:
        f.write(line + "\n")

print(f"SUCCESS!")
print(f"- Created 'corpus.jsonl' with {len(corpus)} documents.")
print(f"- Created 'queries.jsonl' with {len(queries)} queries.")
print(f"- Created 'qrels.tsv' with {len(qrels)} connections.")