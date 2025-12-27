import pandas as pd

# 1. Read your existing 3-column qrels
print("Reading current qrels...")
df = pd.read_csv("data/processed/qrels.tsv", sep="\t")

# 2. Convert to Standard TREC Format (4 columns, Space separated, No Header)
# TREC Format: query_id | 0 | doc_id | score
print("Converting to standard format...")
with open("data/processed/qrels.tsv", "w") as f:
    for index, row in df.iterrows():
        # strict spacing: q_id 0 doc_id score
        line = f"{row['query-id']} 0 {row['corpus-id']} {row['score']}\n"
        f.write(line)

print("Success! 'qrels.tsv' is now in standard TREC format.")