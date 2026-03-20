# S-DisKD 실험 분석 리포트

**실험 브랜치**: `exp/s-diskd`
**실험 일시**: 2026-03-10 ~ 2026-03-20
**데이터셋**: LibriSpeech train-clean-100 / dev-clean / dev-other / test-clean / test-other
**Student**: NeMo EncDecCTCModelBPE (5.2M trainable params)
**Teacher**: `stt_en_conformer_ctc_small`

---

## 1. 실험 개요

### 핵심 가설
> DAG-KD의 Teacher-side factor (txt_emb, spk_emb)를 Student 중간 레이어 표현과 직접 정렬하면,
> Student가 Teacher의 disentangled factor 구조를 학습하여 ASR 성능이 향상된다.

### 실험 설계

| Exp | 설명 | use_stu_txt_kd | use_stu_spk_kd | use_stu_club |
|-----|------|:-:|:-:|:-:|
| E1 | DAG-KD Baseline | False | False | False |
| E2 | Student Text Factor KD | True | False | False |
| E3 | Student Speaker Factor KD | False | True | False |
| E4 | Student Text + Speaker Factor KD | True | True | False |
| E5 | S-DisKD Full (Text + Spk KD + CLUB MI) | True | True | True |

**공통 설정**: epochs=100, batch_size=32, flow_steps=8, disent_spk_layers=4, disent_txt_layers=16,
disen_mi_pairs=ts,tp,ps, disen_lll_weight=1.0, disen_mi_weight=1.0, global_step=89,200

---

## 2. 실험 결과

### 2-1. WER 요약표 (%, 낮을수록 좋음)

| Exp | dev_clean | dev_other | test_clean | test_other |
|-----|----------:|----------:|-----------:|-----------:|
| E1 Baseline | 14.00 | 31.95 | 14.08 | 32.56 |
| E2 Stu Txt KD | 14.03 | 31.80 | 14.34 | 32.82 |
| E3 Stu Spk KD | **13.43** | **30.44** | **13.55** | **31.23** |
| E4 Stu Txt+Spk KD | 13.62 | 30.47 | 13.75 | 31.25 |
| E5 S-DisKD Full | 13.72 | 30.67 | 13.77 | 31.66 |

### 2-2. Baseline 대비 개선량 (test_clean / test_other)

| Exp | test_clean 개선 | test_other 개선 |
|-----|:-:|:-:|
| E2 | -0.26% (악화) | -0.26% (악화) |
| E3 | **+0.53%** | **+1.33%** |
| E4 | +0.33% | +1.31% |
| E5 | +0.31% | +0.90% |

### 2-3. 손실 요약

| Exp | test_clean loss | test_other loss |
|-----|:-:|:-:|
| E1 | 16.51 | 39.35 |
| E2 | 16.57 | 39.16 |
| E3 | **15.83** | **37.09** |
| E4 | 15.89 | 37.34 |
| E5 | 16.03 | 37.43 |

---

## 3. 분석

### 3-1. E3이 가장 좋은 이유: Speaker Factor KD의 효과

E3 (Speaker KD만 적용)이 모든 지표에서 최고 성능을 기록했다.

**해석**: DAG-KD의 Teacher는 하위 레이어(layer 4)에서 speaker factor를, 상위 레이어(layer 16)에서 text factor를 추출한다. Student의 하위 레이어 표현을 Teacher의 speaker factor와 정렬하는 것이 효과적이었다. 이는 다음과 같은 이유로 설명된다:

- Speaker 정보는 저수준(acoustic) 특성이므로 Student의 중간 레이어에서 비교적 직접적으로 학습 가능
- Speaker factor를 정렬함으로써 Student의 acoustic 표현이 Teacher 수준으로 향상됨
- 결과적으로 text/linguistic 표현도 간접적으로 정제되어 WER 개선

### 3-2. E2가 오히려 악화된 이유: Text Factor KD의 역효과

E2 (Text KD만 적용)는 Baseline보다 test_clean -0.26%, test_other -0.26% 악화됐다.

**해석**: Teacher의 text factor (layer 16에서 추출)는 Teacher 인코더 깊은 레이어의 고수준 언어 표현이다. Student의 상위 레이어 표현을 이와 직접 정렬하면:

- Student 인코더의 최종 출력이 Teacher의 중간 레이어 표현에 "끌려가는" 현상 발생
- CTC loss와 text factor KD loss 간의 목적 충돌 가능성
- Teacher text factor가 이미 speaker-disentangled 된 표현인데, Student는 아직 disentanglement가 불완전하므로 정렬 자체가 부적절

### 3-3. E4가 E3보다 낮은 이유: Text KD의 방해 효과

E4 = E3 + Text KD이지만 E3보다 성능이 떨어졌다 (test_clean: 13.75% vs 13.55%).

**해석**: E2에서 확인된 Text KD의 부정적 효과가 E4에서도 동일하게 작용했다. Speaker KD의 긍정적 효과를 Text KD가 일부 상쇄했다.

### 3-4. E5 (CLUB MI)의 추가 효과 미미

E5는 E4에 CLUB 기반 Student-side MI 최소화를 추가했지만 E4 대비 개선이 없었다 (test_clean: 13.77% vs 13.75%).

**해석**:
- Student-side txt/spk factor 간의 MI 최소화는 이미 Teacher-side disentanglement (E1의 disen_mi_pairs=ts,tp,ps)로 간접적으로 달성되고 있을 가능성
- Student txt/spk factor 품질이 낮은 상태에서 MI 최소화를 추가해도 효과가 제한적
- stu_club_weight(1e-3)가 너무 작아 실질적인 regularization 효과 부족할 수 있음

---

## 4. 결론

| 순위 | Exp | test_clean | 평가 |
|:----:|-----|:-----------:|------|
| 1 | **E3 Stu Spk KD** | **13.55%** | Speaker factor 정렬만으로 최고 성능 |
| 2 | E4 Stu Txt+Spk KD | 13.75% | E3 대비 소폭 악화 |
| 3 | E5 S-DisKD Full | 13.77% | CLUB 추가 효과 없음 |
| 4 | E1 Baseline | 14.08% | 기준선 |
| 5 | E2 Stu Txt KD | 14.34% | Baseline보다 악화 |

**핵심 결론**: S-DisKD에서 Speaker Factor KD만 적용한 E3이 최적이다. Text Factor KD는 성능 저하를 유발하며, CLUB MI 최소화는 추가 이득이 없다.

---

## 5. 다음 실험 제안

### 방향 1: Text Factor KD 개선
- **문제**: Student 상위 레이어 → Teacher text factor 직접 정렬이 CTC loss와 충돌
- **제안**: Text factor KD를 CTC loss 대신 layer-wise KD로 대체하거나, teacher text factor 대신 teacher 최종 인코더 출력을 target으로 사용

### 방향 2: Speaker Factor KD 심화
- **제안**: `disent_spk_layers`를 다양하게 변경 (e.g., layer 2, 6, 8)하여 최적 레이어 탐색
- **제안**: MSE 대신 cosine similarity loss 또는 contrastive loss 사용

### 방향 3: E3 가중치 튜닝
- **제안**: `stu_spk_kd_weight`를 {0.1, 0.5, 2.0, 5.0}으로 변경하여 최적 가중치 탐색
