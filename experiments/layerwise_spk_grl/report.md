# Layerwise Speaker GRL — 실험 레포트

## 테이블 주제

**Layerwise Spk-Free Layer KD**
모든 Teacher 레이어에 개별 인코더(Conv1d)와 Shared GRL Spk Classifier를 달아,
Layer KD 타겟을 화자 정보가 제거된 표현으로 만든다.

---

## Hypothesis

기존 Layer KD(`_layer_metric_kd`)는 Teacher의 각 레이어 출력을 **그대로** KD 타겟으로 사용한다.
Teacher의 lower layer에는 화자(spk) 정보가 많이 섞여 있어,
Student가 layer KD를 통해 불필요한 화자 정보까지 학습하게 된다.

→ **각 Teacher 레이어 출력에 개별 인코더 + Shared GRL Spk Classifier를 적용해
화자 정보를 제거한 feat_i를 만들고 이를 Layer KD 타겟으로 쓰면,
Student는 더 순수한 linguistic 표현을 학습하여 ASR 성능이 개선된다.**

---

## 구조

```
Teacher Layer i (frozen)
        ↓  X_i^T : (B, dim_t=176, T)
   enc_i  (Conv1d 176→88, layer별 개별)
        ↓  feat_i : (B, dim_s=88, T)
        │
   ┌────┴─────────────────────────────┐
   ↓                                  ↓
 GRL(alpha)                     stu_feat_i (hook)
   ↓                                  ↓
AdaptiveAvgPool1d(1)            L_kd_i = MSE(feat_i, stu_feat_i)
   ↓  (B, 88)
Shared Spk Classifier
  Linear(88→256) → ReLU → Linear(256→251)   ← LibriSpeech-100: 251명
   ↓
L_adv_i = CE(pred_spk, true_spk)
```

**Shared GRL Spk Classifier 동작:**
- 16개 enc_i가 각각 독립적으로 GRL+Classifier를 통과
- Classifier 파라미터는 16개가 공유 (하나의 모듈)
- 각 L_adv_i를 16개 평균 → enc_i에 역방향 gradient 전달
- spk label은 기존 manifest의 speaker_ids 사용 (train.py에서 배치에 이미 포함)

**전체 Loss:**
```
L_total = L_ctc                          (ASR 메인)
        + L_logit_kd                     (teacher 최종 logit KD)
        + (1/16) Σ L_kd_i               (layerwise KD, 16개 평균)
        + λ_adv * (1/16) Σ L_adv_i      (adversarial spk 제거, E2만)
```

- `use_disent=False` → rec_txt, rec_spk, spk_ce 전부 비활성 (이번 실험과 무관)

**Gradient 흐름:**

| 모듈 | gradient 출처 | 방향 |
|---|---|---|
| enc_i | L_kd_i | 정방향 — student layer i와 가까워져라 |
| enc_i | L_adv_i via GRL | **역방향** — spk 정보 지워라 |
| Shared Classifier | L_adv_i | 정방향 — spk 잘 분류해라 |
| Teacher (frozen) | — | gradient 없음 |

---

## 실험 테이블

| Exp | 설명 | use_layer_kd | spk_grl | λ_adv | grl_alpha |
|-----|------|:---:|:---:|:---:|:---:|
| E1  | Layer KD only (baseline) | ✓ | ✗ | — | — |
| E2  | Layer KD + Layerwise Spk GRL | ✓ | ✓ | 1e-1 | 0.1 |

- E1: GRL 없이 순수 Layer KD → 비교군
- E2: λ_adv=1e-1 선택 근거 — E5 cyclic 실험(cyclic_weight=5e-2, grl_alpha=0.1)이 안정적으로 수렴했고, 동일 계열 adversarial loss이므로 같은 스케일 적용. 1e-2는 효과 미미, 1.0은 layer KD loss를 압도할 위험.

---

## 주요 설계 결정

### Shared Classifier 사용 이유

- 16개 레이어 각각 별도 classifier를 달면 파라미터 과다 + 화자 분류 능력 분산
- Shared classifier는 모든 레이어의 feat에서 공통적으로 "spk를 분류하는 능력"을 학습
- 각 enc_i에 독립적으로 역방향 gradient를 흘려 spk 제거 유도

### enc_i를 separate로 두는 이유

- 레이어마다 feature 분포가 근본적으로 다름 (lower: acoustic, upper: linguistic)
- Shared enc를 쓰면 서로 다른 분포를 동시에 처리해야 해 압축 품질 저하
- KD 타겟의 품질은 enc_i가 얼마나 잘 압축하느냐에 달려있으므로 separate가 유리

### Upper layer에서 GRL 효과

- Upper layer (13~16)는 원래 spk 정보가 적어 L_adv가 약하게 걸림
- 이는 정상 동작 — "제거할 spk가 없으면 pressure 없음"
- 목적은 "전 레이어 균등 제거"가 아닌 "KD 타겟이 spk-free이면 충분"

---

## 모니터링 지표

- `val_wer` / `test_clean/wer` / `test_other/wer` — 핵심 ASR 성능
- `train/layer_kd` — Layer KD loss 수렴 여부
- `train/spk_grl_adv` — adversarial loss (16개 평균)
- `train/spk_grl_acc` — spk classifier 정확도 (0.5 수렴 = spk 제거 성공)

---

## 코드 수정 계획

1. `models.py`
   - `LayerwiseSpkGRL` 모듈 추가 (enc_i × L개 + GRL + Shared Spk Classifier)
   - `__init__`: `use_layerwise_spk_grl`, `spk_grl_alpha`, `spk_grl_adv_weight` 파라미터 추가
   - `training_step`: `_layerwise_spk_grl_kd()` 호출 + L_adv, L_kd 로깅
   - 기존 `_layer_metric_kd`는 E1용으로 유지, E2는 새 메서드로 대체

2. `train.py`
   - `--use_layerwise_spk_grl`, `--spk_grl_alpha`, `--spk_grl_adv_weight` 인자 추가

3. `experiments/layerwise_spk_grl/`
   - `E1_baseline.sh`, `E2_layerwise_grl.sh` 스크립트 작성
