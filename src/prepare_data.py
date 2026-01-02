import pandas as pd
import json

# CONFIGURATION
INPUT_FILE = "data/raw/raw_data.csv"

# This dataset have these column names.
COL_QUERY = "soru"  # Column containing the user question
COL_DOC = "cevap"  # Column containing the law/answer

# Reading the file.
print(f"1. Reading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# Basic check
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

# Creating empty arrays for later use.
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
    query_id = f"query_{index}"
    queries.append({"_id": query_id, "text": query_text})

    # 3. Handle Relation (The "Answer Key")
    # This query_id matches this doc_id
    # A TSV(Tab Separated Values) file is atype of text file that stores
    # data in a tabular format, where each value is separated by a tab character.
    qrels.append(f"{query_id}\t{doc_id}\t1")
    # Why is score just "1"?
    # Since raw CSV file pairs a question directly with its correct answer,it is assumed every pair is a "match."
    # 1 = Relevant.
    # 0 = Irrelevant.

print(f"3. Saving files...")

# Save Corpus (jsonl)
with open("data/processed/corpus.jsonl", "w", encoding="utf-8") as f:
    for item in corpus:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        # In Python’s json module, the parameter ensure_ascii=False is used with json.dumps()
        # to prevent non-ASCII characters from being escaped into \uXXXX sequences.
        # Instead, characters like ç, ü, ğ, あ, etc., will be written directly in UTF-8.

# Save Queries (jsonl)
with open("data/processed/queries.jsonl", "w", encoding="utf-8") as f:
    for item in queries:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Save Qrels (tsv)
with open("data/processed/qrels.tsv", "w", encoding="utf-8") as f:
    for line in qrels:
        # The 'line' variable currently has tabs: "query_id\tdoc_id\t1"
        # We need to split it and reassemble it into TREC format:
        # Standard TREC Format (4 columns, Space separated, No Header)
        # TREC Format: query_id | 0 | doc_id | score

        parts = line.split('\t')  # Split by tab
        q_id = parts[0]
        d_id = parts[1]
        score = parts[2]

        # Write: q_id SPACE 0 SPACE doc_id SPACE score
        f.write(f"{q_id} 0 {d_id} {score}\n")

print(f"Preparing processed data step executed successfully.")
print(f"- Created 'corpus.jsonl' with {len(corpus)} documents.")
print(f"- Created 'queries.jsonl' with {len(queries)} queries.")
print(f"- Created 'qrels.tsv' with {len(qrels)} connections.")