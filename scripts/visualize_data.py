import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import torch

def load_queries(file_path, limit=500):
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line)['query'])
            if len(queries) >= limit:
                break
    return queries

def visualize(original_file, augmented_file):
    print("Loading models and data for visualization...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    
    orig_queries = load_queries(original_file)
    aug_queries = load_queries(augmented_file)
    
    all_queries = orig_queries + aug_queries
    
    print(f"Encoding {len(all_queries)} queries...")
    embeddings = model.encode(all_queries, show_progress_bar=True)
    
    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    n_orig = len(orig_queries)
    plt.scatter(embeddings_2d[:n_orig, 0], embeddings_2d[:n_orig, 1], 
                alpha=0.6, label='Original Data (Hard)', c='blue', s=50)
    plt.scatter(embeddings_2d[n_orig:, 0], embeddings_2d[n_orig:, 1], 
                alpha=0.6, label='Augmented Data (LLM)', c='red', s=50)
    
    plt.title("Distribution of Original vs Augmented Queries (t-SNE)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    output_path = "data/distribution_comparison.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    # Example paths
    orig_path = "data/hard_train_triplets.jsonl"
    aug_path = "data/augmented_train_triplets.jsonl"
    
    import os
    if os.path.exists(orig_path) and os.path.exists(aug_path):
        visualize(orig_path, aug_path)
    else:
        print("Required data files not found for visualization.")
