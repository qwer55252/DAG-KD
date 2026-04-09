# Layerwise Speaker GRL — 실험 레포트

## 테이블 주제

**Layerwise Spk-Free Layer KD**
모든 Teacher 레이어에 개별 AE(enc_i + dec_i)와 Shared GRL Spk Classifier를 달아,
spk 정보가 제거된 feat_i를 만들고 이를 Student Layer KD 타겟으로 사용한다.

---

## Hypothesis

기존 Layer KD(`_layer_metric_kd`)는 Teacher의 각 레이어 출력을 **그대로** KD 타겟으로 사용한다.
Teacher의 lower layer에는 화자(spk) 정보가 많이 섞여 있어,
Student가 layer KD를 통해 불필요한 화자 정보까지 학습하게 된다.

→ **각 Teacher 레이어 출력에 개별 AE(enc_i + dec_i)를 달아 의미있는 표현을 보존하면서,
Shared GRL Spk Classifier로 화자 정보를 제거한 feat_i를 만들고 이를 KD 타겟으로 쓰면,
Student는 더 순수한 linguistic 표현을 학습하여 ASR 성능이 개선된다.**

---

## 구조

```
Teacher Layer i (frozen)
        ↓  X_i^T : (B, 176, T)
   enc_i  (Conv1d 176→88, layer별 개별)
        ↓  feat_i : (B, 88, T)
        │
   ┌────┼─────────────────────────────────┐
   ↓    ↓                                 ↓
dec_i  GRL(alpha)                   stu_feat_i (hook)
   ↓      ↓                               ↓
rec_i  AvgPool → Shared Classifier   L_stu_i = MSE(stu_feat_i, feat_i.detach())
   ↓      ↓
L_rec_i  L_adv_i = CE(pred_spk, true_spk)
```

**각 loss의 역할:**

| Loss | 학습 대상 | 역할 |
| --- | --- | --- |
| `L_rec_i` = MSE(dec_i(feat_i), X_i^T) | enc_i, dec_i | teacher 정보 보존 — 의미있는 표현 보장 |
| `L_adv_i` = CE via GRL | enc_i (역방향), classifier (정방향) | feat_i에서 spk 정보 제거 |
| `L_stu_i` = MSE(stu_i, feat_i.detach()) | student | spk-free feat_i 일방적으로 학습 |

**전체 Loss:**
```
L_total = L_ctc                              (ASR 메인)
        + L_logit_kd                         (teacher 최종 logit KD)
        + (1/16) Σ L_rec_i                   (AE reconstruction, 16개 평균)
        + (1/16) Σ L_stu_i                   (student KD, 16개 평균)
        + λ_adv * (1/16) Σ L_adv_i          (adversarial spk 제거, E2만)
```

- `use_disent=False` → rec_txt, rec_spk, spk_ce 전부 비활성
- spk label은 기존 manifest의 speaker_ids 사용 (train.py 배치에 이미 포함)

**Gradient 흐름:**

| 모듈 | gradient 출처 | 방향 |
|---|---|---|
| enc_i | L_rec_i | 정방향 — teacher 정보 보존 |
| enc_i | L_adv_i via GRL | **역방향** — spk 정보 지워라 |
| dec_i | L_rec_i | 정방향 — teacher 재구성 |
| Shared Classifier | L_adv_i | 정방향 — spk 잘 분류해라 |
| student | L_stu_i | 정방향 — spk-free feat_i 따라가라 |
| Teacher (frozen) | — | gradient 없음 |

---

## 실험 테이블

| Exp | 설명 | use_layer_kd | spk_grl | λ_adv | λ_rec | normalize_stu | grl_alpha |
|-----|------|:---:|:---:|:---:|:---:|:---:|:---:|
| E1  | Layer KD only (baseline) | ✓ | ✗ | — | — | — | — |
| E2  | Layerwise Spk GRL KD | ✗ | ✓ | 1e-1 | 1.0 | ✗ | 0.1 |
| E3  | E2 + Normalized Stu KD + λ_rec 감소 | ✗ | ✓ | 1e-1 | 0.1 | ✓ | 0.1 |

- E1: 기존 `_layer_metric_kd` (stu_to_tea_proj MSE) → 비교군
- E2: λ_adv=1e-1 선택 근거 — E5 cyclic 실험(grl_alpha=0.1)이 안정적으로 수렴했고, 동일 계열 adversarial loss이므로 같은 스케일 적용
- E3: E2 결과 분석 — `spk_grl_stu=50` (teacher scale vs student ASR scale mismatch) → `F.normalize(feat_i, dim=1)` / `F.normalize(s, dim=1)` 후 MSE로 방향만 정렬. `λ_rec=0.1`로 rec_loss 비중을 줄여 enc_i가 adv signal에 더 집중하게 함

---

## 주요 설계 결정

### AE 구조 (enc_i + dec_i) 사용 이유

- GRL만 있으면 enc_i가 상수 벡터를 출력해도 classifier를 못 속이는 전략 가능
- rec_loss로 teacher를 재구성해야 하므로 enc_i는 반드시 의미있는 표현을 만들어야 함
- dec_i는 rec_loss만 받고 adv gradient는 받지 않음 (역할 분리)

### enc_i/dec_i를 separate로 두는 이유

- 레이어마다 feature 분포가 근본적으로 다름 (lower: acoustic, upper: linguistic)
- Shared enc/dec를 쓰면 서로 다른 분포를 동시에 처리해야 해 압축 품질 저하

### Student KD에서 feat_i.detach() 사용 이유

- student가 "spk가 제거된 완성된 표현"을 일방적으로 따라가도록 함
- student gradient가 enc_i에 흘러들어가면 spk 제거 방향이 오염될 수 있음

### Shared Classifier 사용 이유

- 16개 레이어 각각 별도 classifier를 달면 파라미터 과다
- Shared classifier는 모든 레이어 feat에서 "spk 분류 능력"을 공통으로 학습
- 각 enc_i에 독립적으로 역방향 gradient 전달

---

## 모니터링 지표

- `val_wer` / `test_clean/wer` / `test_other/wer` — 핵심 ASR 성능
- `train/spk_grl_rec` — AE reconstruction loss (수렴 확인)
- `train/spk_grl_adv` — adversarial loss (16개 평균)
- `train/spk_grl_stu` — student KD loss (16개 평균)
- `train/spk_grl_acc` — spk classifier 정확도 (0.5 수렴 = spk 제거 성공)
