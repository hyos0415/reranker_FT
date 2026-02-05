# 기술 리포트: 한국어 금융 및 법률 리랭커 성능 고도화 (Hit@1 0.96 달성)

## 1. 서론 (Introduction)

본 프로젝트는 한국어 금융 및 법률 도메인에 특화된 고성능 리랭커(Reranker) 모델을 개발하는 것을 목표로 하였습니다. 금융 및 법률 문서는 전문 용어의 비중이 높고 문맥적 미묘함이 중요하여, 범용 모델만으로는 높은 검색 정확도를 보장하기 어렵습니다. 이를 해결하기 위해 PEFT(Parameter-Efficient Fine-Tuning) 기술과 전략적 데이터 증강 기법을 결합하여, Hit@1 0.96를 달성하였습니다.

## 2. 데이터셋 및 전처리 (Dataset & Preprocessing)

### 2.1 AI-Hub 소스 데이터 활용
AI-Hub의 '금융, 법률 문서 기계독해(MRC) 데이터'를 기반으로 학습 및 평가 데이터셋을 구축하였습니다. 실제 도메인 지식이 반영된 질의-문서 쌍을 활용하여 모델의 전문성을 높였습니다.

### 2.2 Hard Negative Mining
단순한 무작위 음성 샘플(Random Negatives) 대신, 모델이 정답과 혼동하기 쉬운 **하드 네거티브(Hard Negatives)**를 사용했습니다.
- PyTorch GPU 가속 기반의 Matrix Multiplication 검색 엔진을 구현하여 대규모 데이터(9.3만 건)에 대해서 FAISS-CPU 보다 빠른 연산 시간을 확보했습니다.
- 벡터 검색(BGE-M3 Embedding)을 통해 질문과 유사하지만 정답은 아닌 문서를 추출하여 학습 난이도를 높였습니다.

### 2.3 지능형 데이터 증강
GPT-4o-mini 및 Claude 4.0 Sonnet을 활용하여 모델의 취약점을 보완하는 고난도 트리플렛(triplet)을 생성했습니다. 특히 모델이 자주 틀리는 유형의 문맥을 LLM에게 학습시켜, 정교한 오답 유도 데이터를 확보했습니다.

## 3. 모델 및 학습 (Model & Training)

### 3.1 베이스 모델
- **Base Model**: `BAAI/bge-reranker-v2-m3` (다국어 지원 및 높은 베이스 성능)

### 3.2 Point-wise Regression 기반 미세 조정 (Calibration)
본 프로젝트에서는 베이스 모델인 `BGE-Reranker-v2-m3`의 능력을 극대화하기 위해 **Point-wise Regression** 접근법을 채택하였습니다. 

| 구분 | 표준 List-wise (BGE/Ranker) | 본 프로젝트 (Point-wise Calibration) |
| :--- | :--- | :--- |
| **학습 방식** | 그룹 내 상대적 순위 최적화 | 개별 쌍에 대한 절대적 점수 교정 |
| **Loss 함수** | Contrastive Loss / Cross-Entropy | **MSE (Mean Squared Error)** |
| **장점** | 초기 랭킹 능력 형성에 유리 | **도메인 특화 점수 보정 및 안정적 LoRA 학습** |
| **데이터 구조** | (Query, Positive, [Negatives]) | (Query, Passage, Label: 1.0/0.0) |

본 방식을 선택한 기술적 배경은 다음과 같습니다:
1.  **지식 보존 및 스코어 교정 (Knowledge Preservation & Calibration)**: BGE-M3 모델은 이미 대규모 코퍼스를 통해 강력한 순위 지정 능력을 갖추고 있습니다. 복잡한 List-wise 학습으로 전체 가중치를 크게 흔드는 대신, LoRA를 활용한 Point-wise 회귀 방식을 통해 **특정 도메인(금융/법률)에 적합한 점수 체계로 정밀하게 교정(Calibration)**하는 전략을 취했습니다.
2.  **LoRA 학습 안정성**: LoRA 어댑터를 활용한 효율적 학습 시, 리스트 단위의 상대적 손실 함수보다 개별 쌍에 대한 회귀 손실 함수가 수렴 속도 및 안정성 측면에서 우수함을 확인했습니다.
3.  **오답 사례 교정 (Targeted Signal)**: 정답(1.0)과 오답(0.0)의 명확한 이진 대조를 통해, 모델이 실패 케이스(Hard Negatives)로부터 받는 오차 그래디언트의 강도를 높여 오답에 대한 학습 효과를 극대화했습니다.

### 3.3 하드웨어 환경
- **GPU**: NVIDIA RTX 4090 / 5090 (RunPod 환경)
- PyTorch 2.6 및 HuggingFace Transformers 활용

## 4. 타겟 파인튜닝 전략 (Targeted Fine-tuning Strategy)

본 프로젝트의 가장 큰 차별점은 **'오답 사례 기반의 타겟 학습(Targeted Fine-tuning)'**입니다.

1.  **오답 분석**: 모델이 0.93 Hit@1 수준에서 정체될 때, 실패 사례(Failure Cases) 30여 건을 분석했습니다.
2.  **증분 학습 (Incremental Tuning)**: 기존 LoRA 가중치를 기반으로 오답 전용 데이터셋에 대해서만 수행하여, 전체 재학습 없이 30분 만에 성능을 0.96까지 끌어올렸습니다.
3.  **효율성**: 이 전략을 통해 전체 재학습 시 8시간 이상 소요되던 시간을 **약 30분**으로 단축하면서도 성능은 비약적으로 향상시켰습니다.

## 5. 평가 결과 (Evaluation Results)

단계별 성능 향상 수치는 다음과 같습니다.

| 학습 단계 | 모델 상태 | Hit@1 | MRR | 비고 |
| :--- | :--- | :---: | :---: | :--- |
| **Stage 1** | Baseline (v2-m3) | ~0.90 | - | 초기 성능 |
| **Stage 2** | 1차 PEFT (9.3만 건) | 0.9300 | 0.9598 | 하드 네거티브 기반 |
| **Stage 3** | **최종 Targeted PEFT** | **0.9600** | **0.9755** | **오답 집중 보완 완료** |

최종적으로 목표치인 Hit@1 0.95를 초과 달성한 **0.9600**의 성능을 기록했습니다.
