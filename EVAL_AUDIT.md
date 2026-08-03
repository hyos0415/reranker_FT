# 평가 감사 기록 (2026-08)

이 저장소가 보고했던 성능 수치를 **철회한다.** 아래는 그 사유와 근거다.

## 철회 대상

| 위치 | 철회된 주장 |
|---|---|
| `README.md` (구) | Hit@1 0.96 달성 |
| `technical_report_ko.md` (구) | Stage 2 Hit@1 0.9300 / MRR 0.9598, Stage 3 Hit@1 0.9600 / MRR 0.9755 |

Stage 3만이 아니라 **Stage 2 수치도 철회한다.** 아래 결함 1·3·4가 두 단계에 공통으로 적용된다.

## 결함

### 1. 평가 표본에 시드가 없다

`scripts/evaluate.py`

```python
if limit and len(samples) > limit:
    import random
    samples = random.sample(samples, limit)   # random.seed() 없음
```

`limit` 기본값 500. 같은 모델을 두 번 평가하면 다른 표본에서 다른 점수가 나온다. Stage 2와 Stage 3의 수치는 서로 다른 표본에서 측정된 값일 수 있으므로 차이를 개선폭으로 볼 수 없다.

n=500, p≈0.93~0.96 기준 단일 측정의 95% 구간은 ±1.7~2.2%p이고, 서로 다른 표본 두 개의 차이에 대한 구간은 약 ±2.8%p다. 보고된 개선폭 3.0%p는 이 경계에 걸친다.

**수정**: `random.seed(42)` 추가, 또는 `--limit 0`으로 전량 평가.

### 2. 검증셋 실패 사례가 학습 데이터로 유입된다 (오염)

```
evaluate.py --triplet_file data/hard_val_triplets.jsonl
    └─ 실패 케이스 → data/failure_cases.jsonl

augment_data.py --input data/failure_cases.jsonl
    └─ GPT-4o-mini / Claude 로 증강

create_targeted_data.py
    high_priority = failure_cases + augmented_gpt + augmented_claude
    └─ data/targeted_train_triplets.jsonl

train_final.py  ← 위 파일로 학습
evaluate.py --triplet_file data/hard_val_triplets.jsonl   ← 같은 검증셋 재평가
```

Stage 3은 자신이 학습한 케이스를 포함한 세트에서 측정됐다. 실패 30건이 전부 정답으로 바뀌기만 해도 500건 기준 Hit@1은 6%p 오른다. 보고된 상승폭 3%p는 그 범위 안이다.

**수정**: 검증셋을 `val-A`(실패 사례 추출·early stopping) / `val-B`(보고 전용, 학습 경로에 미노출)로 분할. 또는 실패 사례를 `hard_train_triplets.jsonl`에서 추출.

### 3. 모델 선택 기준과 보고 지표가 같은 파일이다

`scripts/train_final.py`, `scripts/train_peft.py`

```python
val_data_path = "data/hard_val_triplets.jsonl"
training_args.load_best_model_at_end = True
training_args.metric_for_best_model = "loss"
callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
```

같은 파일이 early stopping 시점 결정, 최종 체크포인트 선택, 최종 성능 보고에 모두 쓰인다. 보고 수치가 체크포인트 선택의 상한에 가까워진다.

**수정**: 결함 2의 분할과 함께 처리. `val-A`로 early stopping, `val-B`로 보고.

### 4. 검증셋 하드 네거티브가 실행 순서에 의존한다

`scripts/mine_hard_negatives.py`

```python
index_file = 'models/faiss_index.bin.pt'
if not os.path.exists(index_file):
    rag.build_index(all_contexts)
else:
    rag.load_index()
```

`__main__`이 train → val 순으로 호출하므로, val 마이닝 시점에는 인덱스가 이미 존재해 **train 코퍼스 인덱스를 재사용**한다. 깨끗한 환경에서 val만 먼저 실행하면 다른 평가셋이 만들어진다. 같은 스크립트가 디스크 상태에 따라 다른 eval을 생성한다.

**수정**:
```python
index_file = f'models/faiss_index_{os.path.basename(input_file)}.pt'
```

## 결함이 실제로 무엇을 통과시키는가

- **잘못된 것을 통과시킴**: 일반화를 개선하지 않고 검증 실패 30건만 암기한 어댑터는 Hit@1 0.99를 기록한다. 현재 평가는 이를 성능 개선으로 보고한다.
- **올바른 것을 실패 처리함**: 실제로 개선된 모델이 어려운 표본을 뽑으면 이전 측정보다 낮게 나와 성능 하락으로 기록된다.

## 재측정하지 않는 이유

학습된 어댑터와 `data/hard_val_triplets.jsonl`이 외부 임대 GPU 인스턴스에 있었고 회수하지 못했다. `data/`와 `models/`는 `.gitignore` 대상이라 저장소에도 없다.

AI-Hub 원본부터 다시 받아 마이닝·2단계 학습을 재현하면 산출 가능하지만, 학습 경험 목적의 소규모 프로젝트이므로 그 비용을 들이지 않는다. **수정 없이 수치를 다시 싣지 않는 것으로 대신한다.**

## 유지되는 것

- 학습/평가 분리는 AI-Hub 원본의 `Training/`·`Validation/` 구분을 따랐다 (`scripts/data_loader.py`). split 미분리 문제는 없다.
- 채점은 Hit@1·MRR 전부 결정론 지표이며 LLM judge를 쓰지 않는다.
- 파이프라인 구현(QLoRA 4bit, 하드 네거티브 마이닝, point-wise regression, 2단계 증분 학습)은 코드로 남아 있다.

## 감사 방법

회귀 평가 전제 확정 항목 8개(정답 확정성 / 우선권 문구 / 고정 대상 / 분산 층수 / 채점 경계 / 안전·품질 분류 / 관측 해상도 / 책임 경계)를 기준으로 코드를 감사했다. 결함 1·4는 **고정 대상 누락**, 결함 3은 **책임 경계 미설정**에 해당한다.

실행 이력(W&B 런, 로그)에는 접근하지 못했으므로, 이 감사는 **코드가 무엇을 허용하는지**만 판정한다. 위 변동 범위는 코드가 허용하는 폭이지 관측된 폭이 아니다.
