# import joblib
# import numpy as np
# import re
# import argparse
# from sklearn.metrics.pairwise import cosine_similarity
# import os
# import sys
# sys.stdout.reconfigure(encoding='utf-8')
# # Paths to saved model artifacts
# # Paths to saved model artifacts (top-level 'models' folder)
# BASE_DIR = os.path.dirname(__file__)
# ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
# MODEL_DIR = os.path.join(ROOT_DIR, 'models')
# VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
# INDEX_PATH = os.path.join(MODEL_DIR, 'scene_index.pkl')
# CSV_PATH = os.path.join(ROOT_DIR, 'output', 'csv', 'search_index.csv')

# # Load vectorizer and scene index
# tfidf = joblib.load(VECTORIZER_PATH)
# index_data = joblib.load(INDEX_PATH)
# scene_vectors = index_data['scene_vectors']
# metadata = index_data['metadata']

# # Simple text normalization
# def preprocess_query(text):
#     text = text.lower()
#     text = re.sub(r"[^a-z0-9\s]", '', text)
#     # split CamelCase in query if present
#     tokens = re.sub('([a-z])([A-Z])', r'\1 \2', text)
#     return tokens.strip()

# # Search function
# def search(query, top_k=5):
#     query_clean = preprocess_query(query)
#     q_vec = tfidf.transform([query_clean])
#     scores = cosine_similarity(q_vec, scene_vectors).flatten()
#     top_idxs = np.argsort(scores)[::-1][:top_k]

#     results = []
#     for idx in top_idxs:
#         score = float(scores[idx])
#         row = metadata.iloc[idx]
#         results.append({
#             'clip_name': row['clip_name'],
#             'start': row['timestamp_start'],
#             'end': row['timestamp_end'],
#             'score': score,
#             'summary': row['scene_summary']
#         })
#     return results

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Search video scenes offline using NLP index')
#     parser.add_argument('query', type=str, help='Natural language search query')
#     parser.add_argument('--top_k', type=int, default=5, help='Number of top results to return')
#     args = parser.parse_args()

#     results = search(args.query, args.top_k)
#     print(f"Top {args.top_k} results for query: '{args.query}'\n")
#     for r in results:
#         print(f"{r['clip_name']} [{r['start']:.2f}s - {r['end']:.2f}s] (score: {r['score']:.3f})")
#         print("  ►", r['summary'])
import sys
# Ensure stdout uses UTF-8 encoding (requires Python 3.7+)
sys.stdout.reconfigure(encoding='utf-8')

import joblib
import numpy as np
import re
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import os

# Define paths relative to this script (assuming this script is in backend/scripts)
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
INDEX_PATH = os.path.join(MODEL_DIR, 'scene_index.pkl')
CSV_PATH = os.path.join(ROOT_DIR, 'output', 'csv', 'search_index.csv')

print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] ROOT_DIR: {ROOT_DIR}")
print(f"[DEBUG] Loading TF-IDF vectorizer from: {VECTORIZER_PATH}")
print(f"[DEBUG] Loading scene index from: {INDEX_PATH}")

# Load vectorizer and scene index
tfidf = joblib.load(VECTORIZER_PATH)
index_data = joblib.load(INDEX_PATH)
scene_vectors = index_data['scene_vectors']
metadata = index_data['metadata']

# Simple text normalization
def preprocess_query(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", '', text)
    # Split CamelCase in query if present
    tokens = re.sub('([a-z])([A-Z])', r'\1 \2', text)
    return tokens.strip()

# Search function
def search(query, top_k=5):
    print(f"[DEBUG] Preprocessing query: '{query}'")
    query_clean = preprocess_query(query)
    print(f"[DEBUG] Query after preprocessing: '{query_clean}'")
    q_vec = tfidf.transform([query_clean])
    scores = cosine_similarity(q_vec, scene_vectors).flatten()
    top_idxs = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idxs:
        score = float(scores[idx])
        row = metadata.iloc[idx]
        results.append({
            'clip_name': row['clip_name'],
            'start': row['timestamp_start'],
            'end': row['timestamp_end'],
            'score': score,
            'summary': row['scene_summary']
        })
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search video scenes offline using NLP index')
    parser.add_argument('query', type=str, help='Natural language search query')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to return')
    args = parser.parse_args()

    print(f"[DEBUG] Running search for query: '{args.query}' with top_k={args.top_k}")
    results = search(args.query, args.top_k)
    print(f"Top {args.top_k} results for query: '{args.query}'\n")
    for r in results:
        # The Unicode arrow U+25BA (►) will be printed correctly with UTF-8 configured.
        print(f"{r['clip_name']} [{r['start']:.2f}s - {r['end']:.2f}s] (score: {r['score']:.3f})")
        print("  ►", r['summary'])
