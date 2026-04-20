# base2small 실험 보고서

**실험 날짜**: 2026-04-13 ~ 2026-04-20  
**브랜치**: feat/wav2vec2.0  
**실험 스크립트**: `scripts/train/wav2vec/base2small/run_sequential.sh`

---

## 실험 목적

Teacher(wav2vec2-base-960h)에서 half-size student로 KD 방식별 효과를 비교한다.  
CTC only / Logit KD / GRP-KD / DAG-KD 성능을 측정하여 KD 방식별 효과를 분석한다.

---

## 실험 설정

| 항목 | 값 |
|------|-----|
| Teacher | `facebook/wav2vec2-base-960h` (~94M) |
| 학습 데이터 | LibriSpeech train-clean-100 |
| 평가 데이터 | dev_clean, dev_other, test_clean, test_other |
| Epochs | 100 |
| Batch size | 8 |
| Learning rate | 1e-4 |
| Warmup epochs | 5 (KD warmup 10) |

### Student 아키텍처 — half-base (~24M, random init)

| 설정 | Teacher | Student |
|------|---------|---------|
| hidden_size | 768 | 384 |
| num_attention_heads | 12 | 6 |
| intermediate_size | 3072 | 1536 |
| 초기화 | pretrained | **random** |

### 실험 조건

| ID | 실험명 | use_ctc | use_logit_kd | use_grp_kd | use_disent | 비고 |
|----|--------|---------|--------------|------------|------------|------|
| E-A | CTC only | ✓ | - | - | - | baseline |
| E-B | Logit KD | ✓ | ✓ (α=0.5, T=1.0) | - | - | |
| E-C | GRP-KD | ✓ | ✓ (α=0.1) | ✓ | - | latent_dim=96, fm_steps=8, diff_steps=9 |
| E-D | DAG-KD | ✓ | - | - | ✓ | layer_kd_alpha=0.5, tch/stu spk=[1,2] txt=[11,12] |

---

## 결과

| 실험 | dev_clean | dev_other | test_clean | test_other |
|------|-----------|-----------|------------|------------|
| E-A: CTC only | 25.00% | 52.47% | 25.32% | 55.96% |
| E-B: Logit KD | 24.51% | 51.89% | 24.77% | 55.31% |
| E-C: **GRP-KD** | **22.41%** | **49.97%** | **22.93%** | **53.56%** |
| E-D: DAG-KD | 27.30% | 54.08% | 27.33% | 58.53% |

---

## 분석

### 가설 검증

> KD + disentanglement(DAG-KD)가 단순 KD보다 더 효율적으로 student를 학습시킨다.

**기각**: DAG-KD가 CTC only보다도 낮음 (test_clean +2.0%p 악화)

### 주요 발견

- **GRP-KD**가 유일하게 CTC only 대비 유의미한 개선 달성 (test_clean -2.4%p, test_other -2.4%p)
- **DAG-KD 역효과**: disentanglement loss(MI 최소화 + probe)가 random init 상태 student의 표현 학습을 방해한 것으로 추정. `kd_warmup_epochs=10`으로는 안정화 불충분 가능성 있음
- **Logit KD** 소폭 개선에 그침 — random init에서 teacher logit을 따라가는 것만으로는 한계 존재

---

## 결론 및 다음 실험 방향

GRP-KD의 generative KD가 random init 조건에서 효과를 보임을 확인.  
DAG-KD는 random init에서 불안정 — warmup 부족이 원인일 가능성이 높음.

**후속 실험 계획**:
1. **pretrained half-size student 재실험**: 레이어 수 절반(6 layers, hidden=768)으로 pretrained 가중치를 직접 로드하는 방식으로 재설계
2. **DAG-KD 안정화**: `kd_warmup_epochs`를 10→30으로 늘려 random init 조건에서 재실험
3. **GRP-KD 하이퍼파라미터 탐색**: `grp_rec_weight`/`grp_gen_weight` 조정
