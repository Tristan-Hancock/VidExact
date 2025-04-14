import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import joblib

# -----------------------------
# PARAMETERS & PATHS
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.pkl")
INDEX_PATH = os.path.join(PROJECT_ROOT, "models", "scene_index.pkl")

EVAL_RESULTS_CSV = os.path.join(PROJECT_ROOT, "output", "csv", "evaluation_results.csv")


TOP_K = 5  # number of top results per query used for evaluation

# -----------------------------
# 1. Load Evaluation Data from CSV
if not os.path.isfile(EVAL_RESULTS_CSV):
    raise FileNotFoundError(f"Evaluation CSV '{EVAL_RESULTS_CSV}' not found.")

eval_df = pd.read_csv(EVAL_RESULTS_CSV)

# Ensure the "relevant" column is boolean and add an integer version for metrics.
# If not boolean, convert to string first then map.
if eval_df['relevant'].dtype != 'bool':
    eval_df['relevant'] = eval_df['relevant'].astype(str).str.lower().map({'true': True, '1': True, 'yes': True}).fillna(False)
eval_df['relevant_int'] = eval_df['relevant'].astype(int)

# -----------------------------
# 2. Precision per Query & Bar Chart
precision_per_query = eval_df.groupby('query')['relevant'].agg(['sum', 'count'])
precision_per_query.rename(columns={'sum': 'num_relevant', 'count': 'num_returned'}, inplace=True)
precision_per_query['precision'] = precision_per_query['num_relevant'] / precision_per_query['num_returned']

print("Precision per query:")
print(precision_per_query)

plt.figure(figsize=(10, 6))
plt.bar(precision_per_query.index, precision_per_query['precision'], color='skyblue')
plt.xlabel('Query')
plt.ylabel('Precision')
plt.title('Precision per Query')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Box Plot of Similarity Scores: Relevant vs Non-Relevant
# Convert the 'relevant' field to a string column for hue mapping
eval_df['relevant_str'] = eval_df['relevant'].astype(str)

plt.figure(figsize=(10, 6))
sns.boxplot(x='relevant_str', y='score', data=eval_df, hue='relevant_str',
            palette={'True': 'green', 'False': 'red'}, dodge=False)
plt.xlabel('Judged as Relevant ("True") vs Non-Relevant ("False")')
plt.ylabel('Similarity Score')
plt.title('Distribution of Similarity Scores by Relevance Judgment')
plt.legend(title='Relevant', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Precision-Recall Curve
precision_vals, recall_vals, thresholds = precision_recall_curve(eval_df['relevant_int'], eval_df['score'])
avg_precision = average_precision_score(eval_df['relevant_int'], eval_df['score'])

plt.figure(figsize=(10, 6))
plt.plot(recall_vals, precision_vals, marker='.', label=f'AP = {avg_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. ROC Curve
fpr, tpr, roc_thresholds = roc_curve(eval_df['relevant_int'], eval_df['score'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------
# 6. DCG / NDCG Plot for One Query
def dcg(relevances):
    """Compute Discounted Cumulative Gain for a list of relevance scores."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

# Pick one query to analyze
if eval_df['query'].nunique() > 0:
    query_to_plot = eval_df['query'].unique()[0]
    query_results = eval_df[eval_df['query'] == query_to_plot].sort_values(by='score', ascending=False)
    
    # For binary relevance, use 1 for relevant, 0 for non-relevant.
    relevances = query_results['relevant_int'].tolist()
    ranks = list(range(1, len(relevances) + 1))
    dcg_values = [dcg(relevances[:i]) for i in range(1, len(relevances)+1)]
    
    # Compute ideal DCG (IDCG)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg_values = [dcg(ideal_relevances[:i]) for i in range(1, len(ideal_relevances)+1)]
    
    # Compute NDCG values
    ndcg_values = [dcg_val / idcg if idcg > 0 else 0 for dcg_val, idcg in zip(dcg_values, idcg_values)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, dcg_values, marker='o', label='DCG')
    plt.plot(ranks, idcg_values, marker='o', label='Ideal DCG')
    plt.xlabel('Rank')
    plt.ylabel('DCG')
    plt.title(f'DCG and Ideal DCG for Query: {query_to_plot}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, ndcg_values, marker='o', color='purple')
    plt.xlabel('Rank')
    plt.ylabel('NDCG')
    plt.title(f'NDCG for Query: {query_to_plot}')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("No queries found in evaluation data.")

# -----------------------------
# 7. t-SNE Visualization of Document Embeddings
if os.path.isfile(VECTORIZER_PATH) and os.path.isfile(INDEX_PATH):
    vectorizer = joblib.load(VECTORIZER_PATH)
    index_data = joblib.load(INDEX_PATH)
    scene_vectors = index_data["scene_vectors"]
    metadata_doc = index_data["metadata"]
    if not isinstance(metadata_doc, pd.DataFrame):
        metadata_doc = pd.DataFrame(metadata_doc)
    
    sample_size = min(200, scene_vectors.shape[0])
    if hasattr(scene_vectors, "toarray"):
        sample_vectors = scene_vectors[:sample_size].toarray()
    else:
        sample_vectors = scene_vectors[:sample_size]
    
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(sample_vectors)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Document Embeddings')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("Model pickle files not found; skipping t-SNE visualization.")

# -----------------------------
# 8. Summary Metrics
overall_precision = precision_per_query['precision'].mean()
print(f"\nOverall Average Precision across queries: {overall_precision:.4f}")
