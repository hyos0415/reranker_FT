import os
import json
import asyncio
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv

# Optional: Support both OpenAI and Anthropic
try:
    from openai import OpenAI
    import anthropic
except ImportError:
    pass

load_dotenv()

class DataAugmentor:
    def __init__(self, provider="openai"):
        self.provider = provider
        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-4o-mini" # Optimized for cost and speed
        elif provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-4-0-sonnet" # Specific model string as per user request

    def generate_paraphrased_query(self, query: str) -> str:
        """전문적인 법률/금융 용어를 사용하여 질문을 재구성합니다."""
        prompt = (
            "당신은 금융/법률 전문가입니다. 다음 질문을 더 고난도의 전문적인 질문으로 바꿔주세요.\n"
            "핵심 용어는 더 구체적인 전문 용어로 대체하고, 문장 구조를 복잡하게 만드세요.\n"
            f"질문: {query}\n"
            "결과만 한국어로 출력하세요."
        )
        
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        return ""

    def generate_hard_negative(self, context: str, positive_answer: str) -> str:
        """정답과 매우 비슷하지만 치명적인 오류가 포함된 오답을 생성합니다."""
        prompt = (
            "당신은 매우 까다로운 시험 출제 위원입니다. 다음 지문을 바탕으로 정답과 아주 유사하지만 틀린 오답을 만드세요.\n"
            "1. 숫자나 날짜를 미묘하게 변경\n"
            "2. '해야 한다'를 '할 수 있다'로 변경\n"
            "3. 주어나 목적어를 금융/법률적으로 헷갈릴만한 다른 주체로 변경\n"
            f"지문: {context}\n"
            f"정답: {positive_answer}\n"
            "오답 결과만 한국어로 출력하세요."
        )
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        return ""

    async def augment_triplets(self, input_file: str, output_file: str, limit: int = 100):
        """기존 트립릿을 읽어 증강된 트립릿을 생성합니다."""
        with open(input_file, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f][:limit]
        
        augmented = []
        print(f"Augmenting {len(samples)} samples using {self.model}...")
        
        for item in tqdm(samples):
            # 1. 쿼리 증강
            new_query = self.generate_paraphrased_query(item['query'])
            
            # 2. 결과 저장
            augmented.append({
                "query": new_query,
                "pos": item['pos'],
                "neg": item['neg'] # 기존 하드 네거티브 유지 또는 추가 생성
            })
            
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in augmented:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Augmentation complete. Saved to {output_file}")

if __name__ == "__main__":
    augmentor = DataAugmentor(provider="openai")
    # 예시 실행 (실제 실행 시 비용 발생 주의)
    # asyncio.run(augmentor.augment_triplets("data/hard_train_triplets.jsonl", "data/augmented_train_triplets.jsonl", limit=10))
