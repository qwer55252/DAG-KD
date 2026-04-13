# Layerwise Speaker GRL — 실험 레포트

## 테이블 주제

**Layerwise Spk-Free Layer KD**
모든 Teacher 레이어에 개별 enc_i와 Shared GRL Spk Classifier를 달아,
spk 정보가 제거된 feat_i를 만들고 이를 Student Layer KD 타겟으로 사용한다.

---

## Hypothesis

기존 Layer KD(`_layer_metric_kd`)는 Teacher의 각 레이어 출력을 **그대로** KD 타겟으로 사용한다.
Teacher의 lower layer에는 화자(spk) 정보가 많이 섞여 있어,
Student가 layer KD를 통해 불필요한 화자 정보까지 학습하게 된다.

→ **각 Teacher 레이어 출력에서 spk 정보를 제거한 feat_i를 만들고 이를 KD 타겟으로 쓰면,
Student는 더 순수한 linguistic 표현을 학습하여 ASR 성능이 개선된다.**

---

## 실험 결과

| Exp | 설명 | dev_clean | test_clean | test_other | spk_acc | spk_grl_stu |
|-----|------|:---:|:---:|:---:|:---:|:---:|
| E1 | Layer KD baseline | **13.42%** | **13.94%** | **31.90%** | — | — |
| E2 | Layerwise Spk GRL (enc 176→88) | 13.78% | — | — | 3.1% ✓ | 50.31 ✗ |
| E3 | E2 + normalize_stu + λ_rec=0.1 | 13.71% | 13.81% | 32.73% | 1.76% ✓ | 0.022 ✓ |
| E4 | Teacher Space KD (enc 176→176) | 13.57% | 13.74% | 32.18% | 64% ✗ | 23.16 ✗ |

- 모든 실험: `use_ctc=True`, `use_logit_kd=True`, `grl_alpha=0.1`, `λ_adv=0.1`, `batch_size=32`, `epochs=100`
- E1: `use_layer_kd=True`, `layer_kd_alpha=1.0`
- E2: `use_layerwise_spk_grl=True`, enc_dim=88, `λ_rec=1.0`
- E3: `use_layerwise_spk_grl=True`, enc_dim=88, `λ_rec=0.1`, `normalize_stu=True`
- E4: `use_layerwise_spk_grl=True`, enc_dim=176, `λ_rec=1.0`

---

## 실험별 분석

### E1 — Layer KD Baseline
- `stu_to_tea_proj` (shared conv1d 88→176) 로 student를 teacher space로 확장 후 MSE
- Teacher feature 그대로 사용 → 정보 손실 없음
- 모든 실험 중 **최고 WER** 기록

### E2 — Layerwise Spk GRL (실패)
- enc_i (176→88) 압축 + AE decoder + GRL adversarial
- `spk_grl_acc=3.1%` → spk 제거 성공 ✓
- `spk_grl_stu=50.31` → student KD 동작 안 함 ✗
- **원인**: enc_i(176→88)이 teacher space(176)에서 student space(88)로 전환되어, student와 enc_i 출력 간 scale mismatch 발생. KD 신호가 사실상 죽음

### E3 — Normalized Student KD (부분 성공)
- `F.normalize(feat_i, dim=1)`, `F.normalize(s, dim=1)` 후 MSE → scale 제거
- `spk_grl_stu=0.022` → KD 작동 ✓
- `spk_grl_acc=1.76%` → spk 제거 성공 ✓
- **그러나 E1보다 WER 나쁨**: spk 제거 + 차원 압축(176→88)으로 KD 타겟의 정보량이 raw teacher보다 적음

### E4 — Teacher Space KD (실패)
- enc_i (176→176), decoder 제거, `MSE(enc_i(t), t)`로 content 보존
- student proj (88→176) → E1과 동일 방향
- `spk_grl_rec=0.006` → enc_i가 거의 identity에 수렴 (teacher feature 거의 변환 안 함)
- `spk_grl_acc=64%` → spk 제거 **실패** ✗
- **원인**: `λ_rec=1.0`이 너무 강해 rec_loss가 adv_loss를 압도. enc_i가 teacher에 고착되어 GRL gradient가 enc_i를 실질적으로 못 움직임

---

## 결론 및 교훈

### 핵심 발견
**Teacher feature를 enc_i로 가공해서 KD 타겟으로 쓰는 접근 자체가 한계를 가짐.**

Teacher는 spk 정보가 포함된 채로 CTC를 최적화하도록 학습됐으며, 중간 레이어 표현은 spk 포함 상태에서 최적 구조를 가진다. 여기서 spk를 억지로 제거하면:
1. **정보 손실**: enc_i 압축(176→88) 또는 GRL로 인한 표현 품질 저하
2. **인공 타겟**: teacher가 한 번도 만들어낸 적 없는 표현을 student가 배워야 함
3. **rec_loss vs adv_loss 충돌**: content 보존과 spk 제거가 동시에 최적화되기 어려움

### 비교군(multi_layer_factor_kd)이 성공한 이유
동 브랜치의 E2_multi_mse(val_wer=12.91%)는 teacher feature를 건드리지 않고, **student 자신의 disentanglement 모듈**(use_disent=True)로 factor를 분리했음:
- Teacher 하위 레이어 → student의 spk_emb에 KD
- Teacher 상위 레이어 → student의 txt_emb에 KD
- CLUB MI로 student의 txt/spk 독립화

→ Teacher의 자연스러운 계층 구조(하위=acoustic, 상위=linguistic)를 student의 disentangle 모듈에 직접 매핑. 정보 손실 없이 student가 factor를 분리.

### Negative Result 정리
"Teacher 중간 레이어 feature에서 spk를 제거한 후 Student에 distill"하는 접근은 단순 Layer KD(E1)를 뛰어넘지 못함. 올바른 방향은 student-side disentanglement + teacher 계층적 구조 활용.

---

## 모니터링 지표 해석 가이드

| 지표 | 정상 범위 | 의미 |
|------|----------|------|
| `spk_grl_rec` | → 0 수렴 | enc_i가 teacher content 보존 |
| `spk_grl_adv` | ≈ ln(251) ≈ 5.5 | classifier가 random 수준 → spk 제거 성공 |
| `spk_grl_acc` | → 1/251 ≈ 0.4% | spk 분류 불가 → spk 정보 제거 성공 |
| `spk_grl_stu` | → 0 | student가 feat_i 잘 따라감 |
