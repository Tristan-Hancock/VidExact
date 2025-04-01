import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import os

# Paths
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
CSV_PATH = os.path.join(BASE_DIR, 'search_index.csv')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
INDEX_PATH = os.path.join(MODEL_DIR, 'scene_index.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

# Load cleaned data
df = pd.read_csv(CSV_PATH)

# Use the existing 'combined_text' column from the CSV
# If for some reason it's missing, you can create a fallback:
if 'combined_text' not in df.columns:
    df['combined_text'] = df['scene_summary']

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(min_df=2, max_df=0.8, ngram_range=(1,2))
scene_vectors = vectorizer.fit_transform(df['combined_text'])
scene_vectors = normalize(scene_vectors)

# Save vectorizer and scene index
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump({'scene_vectors': scene_vectors, 'metadata': df[['clip_name', 'timestamp_start', 'timestamp_end', 'scene_summary']]}, INDEX_PATH)

print(f"TF-IDF vectorizer saved to {VECTORIZER_PATH}")
print(f"Scene index saved to {INDEX_PATH}")
