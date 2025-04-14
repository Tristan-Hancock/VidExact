import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity
import os
import csv

# -------------------------
# PARAMETERS
NUM_QUERIES = 5  # Number of queries to evaluate
TOP_K = 5        # Number of top results per query
# Adjust the paths by setting the PROJECT_ROOT to one directory above this script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.pkl")
INDEX_PATH = os.path.join(PROJECT_ROOT, "models", "scene_index.pkl")
EVAL_RESULTS_CSV = os.path.join(PROJECT_ROOT, "output", "csv", "evaluation_results.csv")

# -------------------------
# Step 1: Load the TF-IDF Vectorizer and Scene Index from Pickle Files
vectorizer = joblib.load(VECTORIZER_PATH)
index_data = joblib.load(INDEX_PATH)
# The scene index pickle contains both scene vectors and metadata.
scene_vectors = index_data["scene_vectors"]
metadata = index_data["metadata"]

# Convert metadata into a Pandas DataFrame if it isn't already one.
if not isinstance(metadata, pd.DataFrame):
    metadata = pd.DataFrame(metadata)

# -------------------------
# Step 2: Define a simple query preprocessing function
def preprocess_query(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", '', text)
    # Handle CamelCase situations if needed
    text = re.sub('([a-z])([A-Z])', r'\1 \2', text)
    return text.strip()

# -------------------------
# Step 3: Define the search function using cosine similarity
def search(query, top_k=TOP_K):
    query_clean = preprocess_query(query)
    q_vec = vectorizer.transform([query_clean])
    scores = cosine_similarity(q_vec, scene_vectors).flatten()
    top_idxs = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_idxs:
        score = float(scores[idx])
        # Get metadata row for this index
        row = metadata.iloc[idx]
        results.append({
            'clip_name': row['clip_name'],
            'timestamp_start': row['timestamp_start'],
            'timestamp_end': row['timestamp_end'],
            'combined_text': row.get('combined_text', ''),
            'scene_summary': row.get('scene_summary', ''),
            'score': score
        })
    return results

# -------------------------
# Step 4: Interactive Evaluation Loop for Multiple Queries
def interactive_search():
    print("Welcome to the Interactive NLP Search Evaluation Tool!")
    evaluation_data_all = []  # to store all responses

    for q in range(1, NUM_QUERIES + 1):
        query = input(f"\nEnter search query {q}/{NUM_QUERIES}: ").strip()
        if not query:
            print("No query entered. Skipping this query.")
            continue

        results = search(query, TOP_K)
        print(f"\nTop {TOP_K} results for query: '{query}'\n")
        
        for i, result in enumerate(results, start=1):
            print(f"Result {i}:")
            print(f"  Clip: {result['clip_name']} [{result['timestamp_start']}s - {result['timestamp_end']}s]")
            print(f"  Scene Summary: {result['scene_summary']}")
            print(f"  Combined Text: {result['combined_text']}")
            print(f"  Similarity Score: {result['score']:.4f}")
            
            # Ask the evaluator to mark the result as relevant or not
            while True:
                judgment = input("   Is this result relevant? (Y/N): ").strip().lower()
                if judgment in ['y', 'n']:
                    break
                else:
                    print("   Please enter 'Y' for yes or 'N' for no.")
            
            evaluation_data_all.append({
                'query': query,
                'clip_name': result['clip_name'],
                'timestamp_start': result['timestamp_start'],
                'timestamp_end': result['timestamp_end'],
                'scene_summary': result['scene_summary'],
                'score': result['score'],
                'relevant': judgment == 'y'
            })
            print("-" * 50)
    
    save_evaluation_data(evaluation_data_all)
    print(f"All evaluation results saved to {EVAL_RESULTS_CSV}")

def save_evaluation_data(data):
    file_exists = os.path.isfile(EVAL_RESULTS_CSV)
    with open(EVAL_RESULTS_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query', 'clip_name', 'timestamp_start', 'timestamp_end', 'scene_summary', 'score', 'relevant'])
        if not file_exists:
            writer.writeheader()
        for row in data:
            writer.writerow(row)

# -------------------------
# Run the interactive evaluation
if __name__ == "__main__":
    interactive_search()
    print("Thank you for using the Interactive NLP Search Evaluation Tool!")
