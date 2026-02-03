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
        
        # Check if it's a PEFT adapter path (contains adapter_config.json)
        is_peft = os.path.exists(os.path.join(model_name, "adapter_config.json"))
        
        if is_peft:
            from peft import PeftModel, PeftConfig
            config = PeftConfig.from_pretrained(model_name)
            base_model_name = config.base_model_name_or_path
            print(f"Detected PEFT adapter. Loading base model: {base_model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base_model, model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
            
            if found_rank == 1: 
                hit_at_1 += 1
            else:
                item['hit_failed'] = True
            mrr += 1.0 / found_rank
        
        n = len(samples)
        results = {
            "Hit@1": hit_at_1 / n,
            "MRR": mrr / n
        }
        
        # Save failure cases for targeted augmentation
        failure_cases = [s for s in samples if s.get('hit_failed', False)]
        if failure_cases:
            os.makedirs('data', exist_ok=True)
            with open('data/failure_cases.jsonl', 'w', encoding='utf-8') as f:
                for case in failure_cases:
                    # Remove the temporary flag before saving
                    case.pop('hit_failed', None)
                    f.write(json.dumps(case, ensure_ascii=False) + '\n')
            print(f"Saved {len(failure_cases)} failure cases to data/failure_cases.jsonl")
            
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
