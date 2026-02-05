# 🚀 Korean Financial & Legal Reranker PEFT Project (Hit@1: 0.96)

이 프로젝트는 AI-Hub의 금융 및 법률 문서 기계독해(MRC) 데이터를 활용하여, 한국어 도메인에 특화된 고성능 리랭커(Reranker) 모델을 파인튜닝(PEFT)하는 프로젝트입니다. **오답 사례 기반 파인튜닝(Targeted Fine-tuning)** 전략을 통해 최종 **Hit@1 0.96**을 달성하였습니다.

## 📊 최종 성능 결과 (Performance)

| 단계 | 모델 정보 | Hit@1 | MRR | 비고 |
| :--- | :--- | :---: | :---: | :--- |
| **Stage 1** | BAAI/bge-reranker-v2-m3 (Base) | ~0.90 | - | 초기 상태 |
| **Stage 4** | 1차 PEFT (Hard Negatives) | 0.9300 | 0.9598 | 9.3만 개 트리플렛 학습 |
| **Stage 6** | **2차 Targeted PEFT (최종)** | **0.9600** | **0.9755** | **오답 집중 증량 학습 완료** |

## 🌟 Key Features
- **Targeted Augmentation**: GPT-4o-mini 및 Claude 4.0 Sonnet을 활용하여 모델의 취약점(Failure Cases)을 집중 보강.
- **Incremental Tuning**: 기존 LoRA 가중치를 효율적으로 계승하여 오답 노트에 대해서만 집중 훈련 (8시간 → 10분 단축).
- **Fast Vector Search**: PyTorch GPU 가속 기반 벡터 검색 시스템으로 하드 네거티브 수집.
- **Aesthetic Visualization**: t-SNE 분석을 통해 증강 데이터와 원본 데이터의 분포 및 타격 지점 시각화.

## 📁 Project Structure
- `scripts/mine_hard_negatives.py`: 벡터 검색 기반 하드 네거티브 채굴
- `scripts/augment_data.py`: GPT/Claude 기반 지능형 데이터 증강
- `scripts/train_final.py`: 오답 집중 타겟팅 증분 학습 스크립트
- `scripts/evaluate.py`: 최종 모델 성능(Hit@1, MRR) 검증 및 오답 노트 추출
- `portfolio.html`: 프로젝트 성과를 시각화한 프리미엄 랜딩 페이지

## 🛠️ Hardware & Environment
- **GPU**: NVIDIA RTX 4090 / 5090 (RunPod)
- **Framework**: PyTorch 2.6, HF Transformers, PEFT (LoRA)

## 🚀 How to Run (Final Training)
```bash
python scripts/create_targeted_data.py


python scripts/train_final.py \
  --model_name_or_path models/reranker-peft-v1/checkpoint-4000 \
  --train_data_path data/targeted_train_triplets.jsonl
```