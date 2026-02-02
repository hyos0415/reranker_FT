import os
import json
import random
from typing import List, Dict, Tuple

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir

    def load_mrc_json(self, file_path) -> List[Dict]:
        """
        AI-Hub 금융/법률 기계독해 데이터(JSON)를 파싱하여 Triplet 리스트를 생성합니다.
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []
            
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data_json = json.load(f)
        
        all_paragraphs = []
        triplets = []
        
        # 1. 모든 컨텍스트 수집 (Negative Sampling용)
        for doc in data_json.get('data', []):
            for para in doc.get('paragraphs', []):
                all_paragraphs.append(para.get('context', ''))

        # 2. Triplet 생성 (Query, Positive, Negative)
        for doc in data_json.get('data', []):
            for para in doc.get('paragraphs', []):
                context = para.get('context', '')
                for qa in para.get('qas', []):
                    query = qa.get('question', '')
                    if not query or not context:
                        continue
                        
                    # Positive: 해당 질문의 컨텍스트
                    # Negative: 전체 데이터 중 랜덤하게 1개 선택 (Hard Negative 추후 도입 가능)
                    neg_candidates = [p for p in all_paragraphs if p != context]
                    negative = random.choice(neg_candidates) if neg_candidates else ""
                    
                    triplets.append({
                        "query": query,
                        "pos": [context],
                        "neg": [negative]
                    })
                    
        return triplets

    def save_triplets(self, triplets: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in triplets:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(triplets)} triplets to {output_path}")

if __name__ == "__main__":
    loader = DataLoader()
    # 예시 실행 로직
    train_file = "data/Training/TL_span_extraction.json"
    if os.path.exists(train_file):
        print("Processing training data...")
        train_triplets = loader.load_mrc_json(train_file)
        loader.save_triplets(train_triplets, "data/train_triplets.jsonl")
    
    val_file = "data/Validation/VL_span_extraction.json"
    if os.path.exists(val_file):
        print("Processing validation data...")
        val_triplets = loader.load_mrc_json(val_file)
        loader.save_triplets(val_triplets, "data/val_triplets.jsonl")
