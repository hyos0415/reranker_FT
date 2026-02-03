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

def visualize(file_list, labels, output_path="data/distribution_comparison.png"):
    print("Loading models and data for visualization...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    
    all_queries = []
    file_indices = []
    
    for file_path in file_list:
        queries = load_queries(file_path)
        file_indices.append(len(queries))
        all_queries.extend(queries)
    
    print(f"Total queries to encode: {len(all_queries)}...")
    embeddings = model.encode(all_queries, show_progress_bar=True)
    
    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    start_idx = 0
    for i, (count, label) in enumerate(zip(file_indices, labels)):
        end_idx = start_idx + count
        plt.scatter(embeddings_2d[start_idx:end_idx, 0], 
                    embeddings_2d[start_idx:end_idx, 1], 
                    alpha=0.6, label=label, c=colors[i % len(colors)], s=50)
        start_idx = end_idx
    
    plt.title("Distribution of Queries: Original vs GPT vs Claude (t-SNE)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', required=True, help="List of jsonl files to visualize")
    parser.add_argument("--labels", nargs='+', required=True, help="Labels for each file")
    args = parser.parse_args()
    
    if len(args.files) != len(args.labels):
        print("Error: Number of files must match number of labels.")
    else:
        visualize(args.files, args.labels)
