# Multi-Layer Factor KD 실험 레포트

**실험 브랜치**: `exp/multi-layer-factor-kd`
**실험 일시**: 2026-04-04 ~ 2026-04-06
**데이터셋**: LibriSpeech train-clean-100 / dev-clean / dev-other / test-clean / test-other
**Student**: NeMo EncDecCTCModelBPE (5.4M trainable params)
**Teacher**: stt_en_conformer_ctc_small

---

## 1. 실험 개요

## 핵심 가설

> 기존 DAG-KD는 Teacher의 spk factor를 layer 4 하나, txt factor를 layer 16 하나에서만 추출한다. 단일 레이어 추출은 해당 레이어의 특성에 과도하게 의존하며, Teacher 인코더의 계층적 표현 구조를 충분히 활용하지 못한다. 여러 레이어에서 factor를 추출하고 Shared AutoEncoder를 통해 동일한 latent space로 투영한 뒤, 대응하는 Student 레이어와 1:1로 KD하면 Teacher의 계층적 표현 구조가 Student에 더 풍부하게 전달되어 ASR WER이 개선될 것이다.

## 실험 설계

| 실험 ID | 설명 | KD 방법 | spk layers | txt layers |
| --- | --- | --- | --- | --- |
| **E1 (Baseline)** | 기존 single-layer DAG-KD | Flow/DiffKD (last→txt) | [4] | [16] |
| **E2** | Multi-Layer Shared AE + MSE KD | MSE | [2, 4, 6] | [12, 14, 16] |
| **E3** | Multi-Layer Shared AE + Flow/DiffKD | Flow Matching + DiffKD | [2, 4, 6] | [12, 14, 16] |

- E1은 MI_ablation best 설정 재사용 (ts 단독, Phys=✓, Rec=✓, test-other WER 32.23%)
- E2, E3는 동일 베이스 위에서 Multi-Layer KD 모듈 추가/교체
- E3는 기존 single-path Flow/DiffKD(step4) 비활성, multi-layer 6쌍으로 대체

**공통 설정**:

```text
teacher: stt_en_conformer_ctc_small
data: LibriSpeech train-clean-100 / dev.clean / dev.other / test.clean / test.other
use_ctc: True, use_logit_kd: True
use_disent: True, disen_mi_pairs: ts
disen_lll_weight: 1.0, disen_mi_weight: 1.0
batch_size: 32, epochs: 100, flow_steps: 8
```

**Multi-Layer Shared AE 구조**:

```text
Teacher spk_layers = [2, 4, 6]    (하위 3개, 0-based: [1,3,5])
Teacher txt_layers = [12, 14, 16] (상위 3개, 0-based: [11,13,15])

tch_feats[1,3,5] (B,176,T) → Shared spk_enc(176→96) → spk_emb_{1,2,3}
tch_feats[11,13,15] (B,176,T) → Shared txt_enc(176→96) → txt_emb_{1,2,3}

[E2] stu_feats[1,3,5] → Shared stu_spk_enc(96→96) → MSE(·, spk_emb_{1,2,3})
     stu_feats[11,13,15] → Shared stu_txt_enc(96→96) → MSE(·, txt_emb_{1,2,3})

[E3] 위 경로에서 MSE 대신 FlowMatchingModule + DiffKDModule 적용
```

---

## 2. 실험 결과

### 2-1. WER 전체 요약표 (%, 낮을수록 좋음)

| ID | 방법 | dev-clean | dev-other | test-clean | test-other |
| --- | --- | --- | --- | --- | --- |
| **E1** | Single-layer DAG-KD (Baseline) | 13.69 | 31.24 | 14.13 | 32.23 |
| **E2** | Multi-Layer MSE KD | **12.91** | 29.90 | 13.28 | 30.55 |
| **E3** | Multi-Layer Flow/DiffKD | 12.94 | **29.61** | **13.10** | **30.41** |

### 2-2. E1 대비 개선량

| ID | dev-clean | dev-other | test-clean | test-other |
| --- | --- | --- | --- | --- |
| E2 vs E1 | **−0.78%p** | **−1.34%p** | **−0.85%p** | **−1.68%p** |
| E3 vs E1 | **−0.75%p** | **−1.63%p** | **−1.03%p** | **−1.82%p** |
| E3 vs E2 | +0.03%p (악화) | −0.29%p | −0.18%p | −0.14%p |

---

## 3. 분석

### 3-1. Multi-Layer KD의 효과 (E1 → E2, E3)

E1 대비 E2/E3 모두 전 데이터셋에서 **1~2%p 수준의 일관된 개선**을 보임. 단일 레이어(layer 4, 16)에서 3개 레이어([2,4,6], [12,14,16])로 확장하여 Teacher의 계층적 표현을 다층으로 전달한 것이 효과적임을 확인.

- test-clean: 14.13% → 13.28%(E2) → 13.10%(E3), 최대 −1.03%p
- test-other: 32.23% → 30.55%(E2) → 30.41%(E3), 최대 −1.82%p

### 3-2. MSE vs Generative KD (E2 vs E3)

E3(Flow/DiffKD)가 E2(MSE) 대비 dev-other, test-clean, test-other 기준 소폭 우위. 단, 개선폭이 0.1~0.3%p로 미미하여 두 방법의 차이는 통계적으로 유의하지 않을 수 있음.

### 3-3. 학습 곡선에서 발견된 구조적 문제

`train/multi_spk_kd` 곡선이 E2/E3 모두에서 동일한 발산 패턴을 보임:
- Step ~10k(warmup 종료)까지 하강 (44~45 수준) → 이후 단조 증가 (~100+)
- E2(MSE)와 E3(Flow/DiffKD) 곡선이 거의 동일 → 손실 함수 무관한 근본 원인

**원인 분석**:

1. **목적함수 충돌**: CTC loss가 Student 하위 레이어(2,4,6)를 텍스트 특성으로 최적화하는 반면, spk KD는 동일 레이어에서 Teacher speaker 특성을 요구. CTC가 dominant해지면서 spk_kd loss 단조 증가.

2. **Moving Target**: `spk_enc`가 rec_spk loss에 의해 매 스텝 업데이트되면서 KD 타겟이 non-stationary. Student는 계속 이동하는 타겟을 추적해야 함.

3. **Shared Projection의 Gradient Interference**: `stu_spk_enc`(96→96 단일 Conv1d)가 분포가 다른 3개 레이어를 동시에 처리. 각 레이어에서 오는 gradient가 서로 상쇄되어 유효 학습이 제한됨.

→ **결론**: 현재 성능 개선의 주 기여는 `multi_txt_kd`(상위 레이어 KD)이며, `multi_spk_kd`는 발산하여 효과 미미 또는 부재. 즉, E1 → E2/E3 개선은 **txt 레이어 3개로 확장한 효과**가 주도.

### 3-4. 최고 성능: E3

E3가 test-clean 13.10%, test-other 30.41%로 전 실험 최고 성능. E1 Baseline 대비 test-clean −1.03%p, test-other −1.82%p 개선.

---

## 4. 결론

### 4-1. 최종 성능 순위

| 순위 | ID | test-clean | test-other | 비고 |
| --- | --- | --- | --- | --- |
| 1 | **E3** (Multi-Layer Flow/DiffKD) | **13.10%** | **30.41%** | 전 지표 최고 |
| 2 | **E2** (Multi-Layer MSE KD) | 13.28% | 30.55% | E3와 차이 미미 |
| 3 | **E1** (Single-layer Baseline) | 14.13% | 32.23% | — |

### 4-2. 컴포넌트별 결론

| 컴포넌트 | 효과 | 권고 |
| --- | --- | --- |
| **Multi-Layer txt KD** ([12,14,16]) | test 전 도메인 ~1%p 개선 | **유지 및 강화** |
| **Multi-Layer spk KD** ([2,4,6]) | 발산, 실질 기여 불명확 | **레이어 변경 또는 구조 수정 필요** |
| **Flow/DiffKD vs MSE** | 차이 미미(0.1~0.2%p) | 현재는 구분 실익 없음 |

### 4-3. 식별된 코드 설계 문제

| 문제 | 원인 | 개선 방향 |
| --- | --- | --- |
| spk_kd 발산 | CTC ↔ spk_kd 목적함수 충돌 | spk layers를 상위([14,15,16])로 변경 |
| Moving target | spk_enc가 rec_spk loss에 의해 매 스텝 변화 | KD 타겟 생성용 projection 분리(freeze) 또는 raw feature 직접 사용 |
| Gradient interference | stu_spk_enc 하나로 3레이어 동시 처리 | 레이어별 독립 projection head 사용 |

