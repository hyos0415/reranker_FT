import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class DataAugmentor:
    def __init__(self, provider='openai', model=None):
        self.provider = provider
        if provider == 'openai':
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model or "gpt-4o"
        elif provider == 'anthropic':
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = model or "claude-3-5-sonnet-20240620"

    def generate_queries(self, text, num_queries=3):
        prompt = f"""
다음은 금융/법률 관련 문서의 일부입니다. 이 내용을 바탕으로 사용자가 검색할 법한 질문을 {num_queries}개 생성해주세요.
질문은 한국어로 작성하며, 문서의 핵심 정보를 포함해야 합니다.

문서 내용:
{text}

형식:
1. [질문 1]
2. [질문 2]
...
"""
        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                output = response.choices[0].message.content
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                output = response.content[0].text
            
            # 파싱 로직 (단순화)
            queries = [q.strip().split('. ')[-1] for q in output.strip().split('\n') if q.strip()]
            return queries[:num_queries]
        except Exception as e:
            print(f"Error during generation: {e}")
            return []

    def augment_dataset(self, docs, output_path, num_samples=100):
        augmented_data = []
        print(f"Augmenting {num_samples} samples...")
        
        for i, doc in enumerate(tqdm(docs[:num_samples])):
            queries = self.generate_queries(doc)
            if queries:
                augmented_data.append({
                    "query": queries[0], # 주 질문 하나를 학습용으로 사용
                    "pos": [doc],
                    "neg": [] # Negative는 나중에 Hard Negative Mining 등으로 채움
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in augmented_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Augmentation complete. Saved to {output_path}")

if __name__ == "__main__":
    # 사용 예시
    # docs = ["문서 1...", "문서 2..."]
    # augmentor = DataAugmentor(provider='openai')
    # augmentor.augment_dataset(docs, "data/aug_train.jsonl")
    print("DataAugmentor ready. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file.")
