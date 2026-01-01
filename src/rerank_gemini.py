import json
import time
import os
import re
from collections import defaultdict
from google import genai
from google.api_core import exceptions
from dotenv import load_dotenv

# CONFIGURATION
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# The variable client is now a Python object that holds our credentials.
# Every time we wish to generate text later in the code (client.models.generate_content),
# we use this client object because it already "knows" we are allowed to ask.
client = genai.Client(api_key=API_KEY)

# CONSTANTS
TEST_LIMIT = 1500
TOP_K = 10 # For every single question, it takes only the top 10 best guesses from previous model (BM25) and shows only those to Gemini.
OUTPUT_FILE = "outputs/run_gemini_zeroshot.txt"
MODEL_ID = "gemini-2.0-flash"  # chosen model


def load_jsonl_dict(filename: str, key_field: str) -> dict:
    """
        Reads a JSONL file and converts it into a Dictionary for fast lookups.

        :param filename: Path to the .jsonl file.
        :param key_field: The field name in the JSON object to use as the dictionary Key (e.g., '_id').
        :return: A dict where { Key: Whole_Object }.
    """
    assert os.path.exists(filename)
    data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data[item[key_field]] = item
    return data


def load_run_file(filename: str) -> dict[str, list[str]]:
    """
        Parses a TREC-formatted run file to extract the top candidate documents for each query.

        The function expects lines in the standard format: "qid Q0 docid rank score run_id".
        It filters the results to keep only documents with rank <= TOP_K (Global Constant).

        :param filename: Path to the run file (str) containing ranked results.
        :return: A dictionary where keys are Query IDs and values are lists of the top Document IDs.
                 Example: {'q1': ['doc_A', 'doc_B'], 'q2': ['doc_X', ...]}
     """
    assert os.path.exists(filename)
    # defaultdict is used here because it automatically initializes an empty list ([]) if a key is missing.
    # This eliminates the need of writing an if statement to check if the Query ID exists before appending the Document ID.
    candidates = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            # Clean and tokenize the line
            # .strip() removes the trailing newline character (\n)
            # .split() breaks the string into a list at every space
            parts = line.strip().split()
            # Extract specific columns based on TREC format
            # parts[0] -> Query ID
            # parts[2] -> Document ID
            # parts[3] -> Rank (converted to int so we can compare it <= TOP_K)
            qid, docid, rank = parts[0], parts[2], int(parts[3])
            if rank <= TOP_K:
                # Add the Document ID to the list associated with this Query ID
                candidates[qid].append(docid)
    return candidates


# Create the directory "outputs" if it doesn't exist.
# exist_ok=True: Prevents the program from crashing (FileExistsError)
# if the folder was already created in a previous run it does not create it again.
os.makedirs("outputs", exist_ok=True)

print("1. Loading Data...")
corpus = load_jsonl_dict("data/processed/corpus.jsonl", "_id")
queries = load_jsonl_dict("data/processed/queries.jsonl", "_id")
# This line creates a dictionary (bm25_candidates) that tells the script:
# "For Query 301, only look at these specific 10 documents."
bm25_candidates = load_run_file("outputs/run_bm25.txt")

print(f"2. Starting Reranking on first {TEST_LIMIT} queries...")

with open(OUTPUT_FILE, 'w') as f_out:
    count = 0

    # Iterate through the dictionary of candidate documents (from the BM25 run).
    for qid, doc_list in bm25_candidates.items():
        if count >= TEST_LIMIT: break

        # DATA INTEGRITY CHECK: If the QID exists in the run file but not in our queries file, skip it.
        if qid not in queries: continue

        # PROMPT ENGINEERING START
        # 1. Retrieve the actual question text.
        query_text = queries[qid]['text']

        # 2. Set the Persona and Task.
        # This "System Instruction" style helps the LLM understand the domain (Turkish Law).
        prompt = f"You are an expert Turkish lawyer. Rank these documents by relevance to the query.\nQuery: {query_text}\n\nDocuments:\n"

        # 3. Dynamic Context Injection:
        # We need to map simple numbers ([1], [2]) to complex Document IDs.
        doc_map = {}
        for idx, doc_id in enumerate(doc_list):
            # Truncating text to 1000 characters to save tokens and fit within context window limits.
            doc_text = corpus.get(doc_id, {}).get('text', '')[:1000]

            # Format: "[1] The text of the document..."
            prompt += f"[{idx + 1}] {doc_text}\n\n"

            # Save the mapping: '1' -> 'doc_00452' so we can translate the LLM's answer back later.
            doc_map[str(idx + 1)] = doc_id

        # 4. Output Formatting Instruction:
        # Explicitly telling the model how to reply.
        prompt += "\nOutput ONLY the ranking as a list of numbers: [1] > [2]\nRanking:"
        # PROMPT ENGINEERING END

        # RETRY LOGIC LOOP
        # We enter a 'while True' loop to handle API crashes (specifically Rate Limits).
        while True:
            try:
                print(f"Ranking Query {count + 1}/{TEST_LIMIT}: {qid}...")

                # API Call
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt
                )

                # Safety check: if no text is returned (blocked)
                # If that's the case we skip to avoid crashing
                if not response.text:
                    print(f"!! Blocked response for {qid}. Skipping.")
                    break

                response_text = response.text.strip()

                # PARSING
                # Regex logic: Find all digits that are inside brackets, e.g., "[1]", "[10]".
                # This ignores the ">" symbols and just extracts the order.
                ranked_indices = re.findall(r'\[(\d+)\]', response_text)
                # The re.findall() function in Python is used to find all non-overlapping
                # matches of a pattern in a string. It returns these matches as a list.

                # FALLBACK MECHANISM:
                # If no numbers found in the output, default to the original BM25 order.
                # This ensures we always have a result written to the file.
                if not ranked_indices:
                    ranked_indices = [str(k) for k in range(1, len(doc_list) + 1)]

                # WRITING RESULTS
                # Write to file in standard TREC format: "qid Q0 docid rank score run_id"
                for rank, idx in enumerate(ranked_indices, 1):
                    if idx in doc_map:
                        f_out.write(f"{qid} Q0 {doc_map[idx]} {rank} {1.0 / rank:.4f} GEMINI_ZERO\n")

                count += 1
                break # Success! Break the 'while True' loop and move to the next Query.

            # ERROR HANDLING:
            # If Google returns a 429 (Resource Exhausted), we catch it here.
            except exceptions.ResourceExhausted:
                print(">> Rate limit hit (429). Sleeping for 30 seconds...")
                time.sleep(30)# Wait and then the 'while True' loop will try again.

            # We catch all other crashes.
            except Exception as e:
                print(f"!! Error on {qid}: {e}")
                break

print(f"Done! Results saved to {OUTPUT_FILE}")