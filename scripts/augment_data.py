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
            self.model = "claude-sonnet-4-0" # Specific model string as per user request

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
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.content[0].text.strip()
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
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            return response.content[0].text.strip()
        return ""

    async def augment_failure_cases(self, input_file: str, output_file: str):
        """실패 사례를 분석하여 집중 증강합니다."""
        if not os.path.exists(input_file):
            print(f"No failure cases found at {input_file}")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
        
        print(f"Processing {len(samples)} failure cases using {self.model}...")
        augmented = []
        
        for item in tqdm(samples):
            # 1. 고난도 쿼리 생성
            new_query = self.generate_paraphrased_query(item['query'])
            
            # 2. 극악의 하드 네거티브 생성 (기존 포지티브 기반)
            hard_neg = self.generate_hard_negative(item['pos'][0], item['pos'][0])
            
            # 3. 조합
            augmented.append({
                "query": new_query,
                "pos": item['pos'],
                "neg": item['neg'] + [hard_neg]
            })
            
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in augmented:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Targeted augmentation complete. Saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--input", type=str, default="data/failure_cases.jsonl")
    parser.add_argument("--output", type=str, default="data/augmented_failures.jsonl")
    args = parser.parse_args()

    augmentor = DataAugmentor(provider=args.provider)
    asyncio.run(augmentor.augment_failure_cases(args.input, args.output))
