# 🚀 Korean Financial/Legal Reranker PEFT Project

이 프로젝트는 AI-Hub의 금융 및 법률 문서 기계독해(MRC) 데이터를 활용하여, 한국어 도메인에 특화된 고성능 리랭커(Reranker) 모델을 파인튜닝(PEFT)하는 것을 목표로 합니다.

## 🌟 Key Features
- **Model**: `BAAI/bge-reranker-v2-m3` 기반
- **Technique**: PEFT (LoRA) + 4-bit Quantization (BitsAndBytes)
- **Hard Negative Mining**: RTX 5090의 GPU 성능을 활용한 PyTorch 기반 광속 하드 네거티브 추출 (FAISS 의존성 제거)
- **High Performance**: RTX 5090 환경 최적화 (Batch Size 32+, FP16/BF16 지원)

## 📁 Project Structure
- `scripts/data_loader.py`: AI-Hub JSON 데이터를 Triplet 형식으로 변환
- `scripts/mine_hard_negatives.py`: 벡터 검색을 통한 어려운 오답(Hard Negative) 채굴
- `scripts/train_peft.py`: LoRA 기반 리랭커 파인튜닝 스크립트
- `scripts/evaluate.py`: 최종 모델 성능(Hit@k, MRR) 검증
- `scripts/rag_system.py`: PyTorch GPU 가속 기반 벡터 검색 시스템

## 🛠 Workflow

### 1. 전처리 및 하드 네거티브 마이닝
리랭커가 미묘한 차이를 학습할 수 있도록, 단순히 랜덤한 오답이 아닌 벡터 검색 상위에 랭크된 '유사하지만 틀린' 문서를 오답으로 구성합니다.
```bash
python scripts/mine_hard_negatives.py
```

### 2. PEFT (LoRA) 학습
RTX 5090의 32GB VRAM을 활용하여 효율적이고 빠르게 학습을 진행합니다.
```bash
python scripts/train_peft.py
```

### 3. 성능 검증
학습 전(Base) 모델과 학습 후(LoRA) 모델의 점수를 비교하여 고도화된 성능을 확인합니다.
```bash
python scripts/evaluate.py --model_path models/reranker-peft-v1
```

## 🚀 Optimization for RTX 5090
- **PyTorch-based Vector Search**: 별도의 벡터 DB 설치 없이 PyTorch 행렬 연산을 통해 수만 건의 검색을 GPU에서 밀리초 단위로 처리합니다.
- **Large Batch Training**: 배치 사이즈를 확대하여 학습 시간 단축 및 안정성 확보.

---
**Created by Antigravity (Advanced Agentic Coding Pair)**
