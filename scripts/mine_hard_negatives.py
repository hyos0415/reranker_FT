import os
import sys
import json
import random
from tqdm import tqdm

# Add the parent directory to sys.path to allow importing from 'scripts'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.rag_system import RAGSystem

def mine_hard_negatives(input_file, output_file, top_k=10):
    rag = RAGSystem(model_name='BAAI/bge-m3')
    
    # 1. 문서 수집 및 인덱스 구축
    print("Loading triplets and collecting all unique contexts...")
    all_samples = []
    all_contexts = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            all_samples.append(item)
            all_contexts.add(item['pos'][0])
            for neg in item['neg']:
                all_contexts.add(neg)
    
    all_contexts = list(all_contexts)
    print(f"Total unique contexts: {len(all_contexts)}")
    
    # 인덱스가 없으면 빌드 (확장자 변경 감지: .pt)
    index_file = 'models/faiss_index.bin.pt'
    if not os.path.exists(index_file):
        print("Building RAG index (PyTorch GPU) for mining...")
        rag.build_index(all_contexts)
    else:
        print("Loading existing RAG index (PyTorch GPU)...")
        rag.load_index()

    # 2. Hard Negative Mining
    print(f"Mining hard negatives for {len(all_samples)} samples...")
    hard_triplets = []
    
    batch_size = 128 # Process 128 queries at once for RTX 5090 efficiency
    for i in tqdm(range(0, len(all_samples), batch_size)):
        batch = all_samples[i:i + batch_size]
        queries = [item['query'] for item in batch]
        
        # Bulk search for the entire batch
        batch_retrieved = rag.bulk_retrieve(queries, top_k=top_k + 5)
        
        for j, item in enumerate(batch):
            query = item['query']
            positive = item['pos'][0]
            retrieved = batch_retrieved[j]
            
            # Extract hard negatives
            hard_negs = [r['text'] for r in retrieved if r['text'] != positive]
            
            if len(hard_negs) < top_k:
                hard_negs.extend(item['neg'])
            
            hard_triplets.append({
                "query": query,
                "pos": [positive],
                "neg": hard_negs[:top_k]
            })

    # 3. 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in hard_triplets:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Hard negative mining complete: {output_file}")

if __name__ == "__main__":
    # 훈련 데이터에 대해 마이닝 수행
    mine_hard_negatives("data/train_triplets.jsonl", "data/hard_train_triplets.jsonl")
    # 검증 데이터에 대해서도 수행 (평가를 정확하게 하기 위함)
    mine_hard_negatives("data/val_triplets.jsonl", "data/hard_val_triplets.jsonl")
