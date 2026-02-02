import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Evaluator:
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3'):
        print(f"Loading model for evaluation: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # BGE-reranker-v2-m3 can be large, use float16 for eval
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def compute_score(self, pairs):
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.model.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1,).float().cpu().numpy()
        return scores

    def evaluate_triplets(self, triplet_file, limit=500):
        hit_at_1 = 0
        mrr = 0
        samples = []
        
        if not os.path.exists(triplet_file):
            print(f"Error: File not found {triplet_file}")
            return {}

        with open(triplet_file, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line))
        
        if limit and len(samples) > limit:
            import random
            samples = random.sample(samples, limit)
            
        print(f"Evaluating on {len(samples)} samples...")
        for item in tqdm(samples):
            query = item['query']
            positive = item['pos'][0]
            negatives = item['neg']
            
            passages = [positive] + negatives
            pairs = [[query, p] for p in passages]
            
            # Compute scores for all pairs (query + pos, query + neg1, ...)
            scores = self.compute_score(pairs)
            
            # Ranking based on scores (higher is better)
            ranked_indices = np.argsort(scores)[::-1]
            # Since index 0 was our positive sample
            found_rank = np.where(ranked_indices == 0)[0][0] + 1
            
            if found_rank == 1: hit_at_1 += 1
            mrr += 1.0 / found_rank
        
        n = len(samples)
        results = {
            "Hit@1": hit_at_1 / n,
            "MRR": mrr / n
        }
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplet_file", type=str, default="data/hard_val_triplets.jsonl")
    parser.add_argument("--model_path", type=str, default="BAAI/bge-reranker-v2-m3")
    args = parser.parse_args()
    
    evaluator = Evaluator(model_name=args.model_path)
    res = evaluator.evaluate_triplets(args.triplet_file)
    
    print(f"\nEvaluation Results ({args.model_path}):")
    for metric, score in res.items():
        print(f"{metric}: {score:.4f}")
