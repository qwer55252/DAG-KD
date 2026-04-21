# wav2vec2 GRP-KD + Orthogonal Disentanglement 실험

## 1. 실험 목적

Conformer 기반 GRP-KD 실험(`experiments/grp_kd_based`)에서 확인된 핵심 결과 —
> "Teacher latent를 text/speaker subspace로 분리하고, text subspace에만 FM+Diffusion KD를 적용하면 dev/test_clean WER이 ~1.0%p 개선된다" (E1 → E2)

를 **wav2vec 2.0 파이프라인** 두 가지 teacher-student 구성에서 재현 가능한지 검증한다.

- **Track A — Large → Base**: teacher 24 layer(d=1024), student 12 layer(d=768). teacher가 student 대비 2배 깊고 1.33배 넓은 전형적인 KD setup.
- **Track B — Base → Base-half**: teacher 12 layer(d=768), student 12 layer(d=384, heads=6, intermediate=1536). 같은 layer 수에서 차원·헤드를 절반으로 줄인 "얇은" student로, 깊이/정렬 변수를 제거한 상태에서 latent disentanglement의 효과만 순수하게 측정.

두 track을 동시에 보면 **layer-depth mismatch가 orth-disen의 유효성에 핵심 변수인지**까지 동시에 검증된다.

---

## 2. 가설

> **wav2vec2 teacher의 layer-wise latent에 text/speaker 정보가 혼재되어 있으며, 각 aligned layer pair에서 이를 병렬 인코더로 96-dim text/speaker subspace로 분리하고 FM+Diffusion을 text subspace에만 적용하면, student가 speaker-clean KD signal을 받아 WER이 E1(GRP-KD baseline) 대비 개선된다. 이 효과는 (A) teacher-student layer depth가 다른 구성과 (B) layer depth는 같고 width만 다른 구성 **모두에서** 관찰되어야 한다.**

세부 가설:

1. **Layer alignment 보존**: Track A에서 teacher 24 → student 12는 **1-indexed 짝수 layer(2,4,...,24)** 만 사용하여 student layer 1~12와 1:1 매칭한다 (기존 `GRPKDModule._align_layers`가 이미 이 동작을 구현). Track B는 1:1 naive 매칭.
2. **직교 제약 보편성**: Conformer E2에서 검증된 `(z_t_text * z_t_spk).sum(1).pow(2).mean()` 제약이 **추가 하이퍼파라미터 튜닝 없이** 두 track 모두에서 baseline 대비 개선을 가져온다.
3. **학생 text-only proj**: student spk 인코더 없이 `proj_text_s` 단일 경로만으로도 KD 타겟(z_t_text)이 speaker-clean이므로 student가 자연스럽게 text-only 표현을 학습한다.
4. **구조 불변성**: Track A(depth mismatch)에서 관찰되는 개선폭과 Track B(width only)에서 관찰되는 개선폭이 유사 방향이면, orth-disen의 효과가 layer 정렬 방식이 아닌 latent 공간의 text/spk 분해 자체에서 오는 것임을 시사.

---

## 3. 실험 설계

### 3.1 비교군

두 track × 두 조건 = 총 4개 실험.

| ID | Track | Teacher | Student | disen_mode | 대응 Conformer |
| --- | --- | --- | --- | :---: | --- |
| **A-E1** | Large→Base | wav2vec2-large-960h (24L, d=1024) | wav2vec2-base-960h (12L, d=768) | 0 | `grp_kd_based/E1_ver4_baseline` |
| **A-E2** | Large→Base | 동상 | 동상 | 1 | `grp_kd_based/E2_disen_orth` |
| **B-E1** | Base→Half | wav2vec2-base-960h (12L, d=768) | Base-half (12L, d=384, heads=6, ffn=1536) | 0 | — (신규) |
| **B-E2** | Base→Half | 동상 | 동상 | 1 | — (신규) |

**변인 통제**:
- Track 내부(A-E1 vs A-E2, B-E1 vs B-E2): `--grp_disen_mode` 값 외 모든 하이퍼파라미터 동일 → orth-disen 효과 측정
- Track 간(A vs B): teacher·student 구조만 다르고 KD 하이퍼파라미터 동일 → depth mismatch의 영향 측정

공통 하이퍼파라미터: latent_dim=96, fm_steps=8, diff_steps=9, rec_weight=1.0, gen_weight=1.0, kd_alpha=0.1, kd_temperature=1.0, epochs=100, lr=1e-4, warmup=5, kd_warmup=10, freeze_feature_extractor=True.

### 3.2 Layer Alignment

```text
Track A: teacher 24 → student 12  (짝수 layer 선택)
  현재 _align_layers(L_t=24, L_s=12) = [1,3,5,...,23] (0-indexed)
                                     = 1-indexed {2,4,6,...,24} (짝수 12개)
  → student layer 1,2,...,12 ↔ teacher layer 2,4,...,24
  ** 기존 구현이 이미 짝수 layer 방식 **

Track B: teacher 12 → student 12  (1:1 naive)
  _align_layers(12,12) = zip(tch, stu) → 그대로 1:1 매칭
```

### 3.3 구조 다이어그램

```text
[E1 — GRPKDModule(disen_mode=0), 두 Track 공통 구조]
teacher layers ─┐ align (A: even, B: 1:1)
student layers ─┘
  per pair:
    TAE(d_t→96) : z_t, t_rec ← 재구성 손실 L_rec
    StudentProj(d_s→96): z_s
    FM(z_s → z_t.detach(), K=8)  : L_FM
    NoiseAdapter + Denoiser(9step)(z_s) → z_deno : L_DF = MSE(z_deno, z_t.detach())

  Total = (1-kd_alpha)*CTC + kd_alpha*LogitKD
        + grp_rec_weight * mean(L_rec) + grp_gen_weight * mean(L_FM + L_DF)


[E2 — GRPKDModule(disen_mode=1), 두 Track 공통 구조]
teacher layers ─┐ align (A: even, B: 1:1)
student layers ─┘
  per pair:
    enc_text_t(d_t→96)     : z_t_text
    enc_spk_t (d_t→96)     : z_t_spk
    proj_text_s(d_s→96)    : z_s_text
    lat_dec(96→d_t)(z_t_text + z_t_spk) : L_rec
    Orth: (z_t_text * z_t_spk).sum(1).pow(2).mean() : L_orth
    SpkCls(mean-pool(z_t_spk) → 251-way) : L_spk (CE)
    FM(z_s_text → z_t_text.detach(), K=8) : L_FM
    NoiseAdapter + Denoiser(9step)(z_s_text) → z_deno : L_DF = MSE(z_deno, z_t_text.detach())

  Total = (1-kd_alpha)*CTC + kd_alpha*LogitKD
        + grp_rec_weight * mean(L_rec) + grp_gen_weight * mean(L_FM + L_DF)
        + grp_orth_weight * mean(L_orth) + grp_spk_cls_weight * mean(L_spk)

Track A: d_t=1024, d_s=768
Track B: d_t=768,  d_s=384
```

### 3.4 `train_grp_kd.py`에서 옮겨오는 로직 매핑

`train_grp_kd.py:263-526` `DistilFlowMatchingCTCModelBPE` → `models_wav2vec.py:234` `GRPKDModule` 확장.

| `train_grp_kd.py` | wav2vec 포팅 위치 | 비고 |
| --- | --- | --- |
| `DistilFlowMatchingCTCModelBPE.__init__` (L263) `disen_mode` flag, `enc_text_t/enc_spk_t/proj_text_s/lat_dec/spk_cls` 선언 (L324-331) | `GRPKDModule.__init__`에 동일 flag/모듈 추가 | teacher_dim=1024, student_dim=768, latent=96 |
| `SpeakerClassifier` (L215-223) | `models_wav2vec.py`에 동일 클래스 추가 | 원본 그대로 복붙 |
| `_compute_v_losses_one_layer` (L405-466) `disen_mode>=1` 분기 | `GRPKDModule.forward` 안 layer-pair loop에 동일 분기 삽입 | 레이어별 동일 로직 반복 |
| `training_step` (L591-608) layer 합산 및 `total_loss`에 `orth/spk_cls` 추가 (L618-619) | `train_wav2vec.py` training_step의 GRP-KD 블록(L1037-1052)에 동일 합산 추가 | `grp_orth_weight`, `grp_spk_cls_weight` 도입 |

**Conformer와 다른 점**:

1. **spk_table 불필요** — wav2vec 파이프라인은 `batch["speaker_ids"]`가 이미 class index 형태로 제공된다 (`train_wav2vec.py:135-137`). Conformer의 `sample_id → spk_table` 룩업 로직은 이식하지 않는다.
2. **num_spk 주입** — `train_wav2vec.py`가 manifest 스캔으로 구한 `num_spk`를 모델에 이미 전달 중이므로, `GRPKDModule`에도 동일 값을 넘겨 `SpeakerClassifier(latent_dim, num_spk)`를 초기화한다.
3. **Layer 수 다름** — Track A는 24/12, Track B는 12/12. 기존 `GRPKDModule._align_layers`가 두 경우 모두를 올바르게 처리 (Track A: even-layer pick, Track B: 1:1).

### 3.5 Track별 하이퍼파라미터

```text
[공통 — Policy P1: CNN만 pretrained, transformer는 random init]
data:    LibriSpeech train-clean-100 (251 spk), dev.clean, test.clean
epochs=100, batch=4/GPU
optimizer: AdamW, lr=1e-4, weight_decay=1e-2, warmup_epochs=5, kd_warmup_epochs=10
random_init_student=True                          # transformer 부분은 random
load_pretrained_feature_extractor=True            # CNN feature extractor는 pretrained 로드
freeze_feature_extractor=True                     # 그리고 freeze
  → 이 3 flag 조합이 "CNN만 pretrained+frozen, 나머지는 random init"을 구현.
KD: kd_alpha=0.1, kd_temperature=1.0
GRP-KD: grp_latent_dim=96, grp_fm_steps=8, grp_diff_steps=9,
        grp_rec_weight=1.0, grp_gen_weight=1.0
E2 추가: grp_disen_mode=1, grp_orth_weight=1.0, grp_spk_cls_weight=1.0
random_init_student=True

[Track A]
teacher_name = facebook/wav2vec2-large-960h  (24 layers, dim=1024, heads=16, ffn=4096)
student_name = facebook/wav2vec2-base-960h   (12 layers, dim=768,  heads=12, ffn=3072)
student_hidden_size=-1 student_num_heads=-1 student_intermediate_size=-1   # base 기본값

[Track B]
teacher_name = facebook/wav2vec2-base-960h   (12 layers, dim=768,  heads=12, ffn=3072)
student_name = facebook/wav2vec2-base-960h   (config만 로드 후 scale down)
student_hidden_size=384 student_num_heads=6 student_intermediate_size=1536
→ 12 layers, dim=384, heads=6, ffn=1536
```

### 3.6 실행 GPU 구성

실험 4개 × GPU 2개 = 총 8 GPU 필요. 현재 환경은 4 GPU이므로 **두 track을 순차 실행**을 기본 계획으로 둔다.

```text
Stage 1 (동시 실행):
  A-E1: CUDA_VISIBLE_DEVICES=0,1
  A-E2: CUDA_VISIBLE_DEVICES=2,3

Stage 2 (A 완료 후, 동시 실행):
  B-E1: CUDA_VISIBLE_DEVICES=0,1
  B-E2: CUDA_VISIBLE_DEVICES=2,3
```

Track B는 student가 작아 batch를 늘리거나 1 GPU로도 돌릴 여지가 있으나, **track 간 비교의 공정성**을 위해 동일 batch=4/GPU × 2 GPU를 유지한다. 만약 GPU 가용성이 달라지거나 B를 1 GPU로 돌리고 싶으면 사용자가 최종 확정.

---

## 4. 변경할 모듈 / 파일

### 4.1 `models_wav2vec.py` 수정

1. **`SpeakerClassifier` 클래스 추가** — `train_grp_kd.py:215-223` 원본 그대로 복사. mean-pool + 단일 Linear.
2. **`GRPKDModule.__init__` 확장**
   - 생성자 인자 추가: `disen_mode=0`, `num_spk=1`, `orth_weight=1.0`, `spk_cls_weight=1.0`
   - `disen_mode >= 1`일 때 `enc_text_t / enc_spk_t / proj_text_s / lat_dec / spk_cls` 선언 (기존 `tae / sproj`와 병렬)
3. **`GRPKDModule.forward` 분기**
   - 시그니처: `forward(self, tch_feats, stu_feats, speaker_ids=None)`
   - layer-pair loop 안에서 `if self.disen_mode == 0`(E1 기존 로직) / `elif self.disen_mode >= 1`(E2 분기) 분리
   - E2 분기에서 per-layer L_rec, L_fm, L_df, L_orth, L_spk 계산 후 5-tuple 반환

### 4.2 `train_wav2vec.py` 수정

1. **argparse 인자 추가** (`--use_grp_kd` 블록 근처 L343-348):
   ```
   --grp_disen_mode       (int,  default=0)    0=E1, 1=E2(orth+spk_cls)
   --grp_orth_weight      (float, default=1.0)
   --grp_spk_cls_weight   (float, default=1.0)
   ```
2. **`Wav2VecKDModel` 생성자 호출**에 3개 인자 추가 전달 (L570-577 근처).
3. **`training_step`**의 GRP-KD 블록(L1037-1052 근처)에서 모듈이 반환하는 loss 개수를 분기 처리 및 total에 `orth_weight*L_orth + spk_cls_weight*L_spk` 추가.

### 4.3 신규 스크립트

- `scripts/train/wav2vec/grp_kd_orth/A_E1_large_base.sh` — Track A E1 baseline
- `scripts/train/wav2vec/grp_kd_orth/A_E2_large_base_orth.sh` — Track A E2 (orth+SpkCls)
- `scripts/train/wav2vec/grp_kd_orth/B_E1_base_half.sh` — Track B E1 baseline
- `scripts/train/wav2vec/grp_kd_orth/B_E2_base_half_orth.sh` — Track B E2 (orth+SpkCls)

---

## 5. 예상 리스크 및 대응

| 리스크 | 원인 | 대응 |
| --- | --- | --- |
| **per-layer 파라미터 폭증** | 12 layer pair × 4 (enc_text_t, enc_spk_t, proj_text_s, lat_dec, spk_cls) Conv1×1 = 대량의 파라미터 | 1×1 Conv는 파라미터가 `(1024+768)*96*2 + 96*1024 ≈ 440K`로 작다. 실제로 teacher forward가 더 dominant. **대신** layer-pair loop 당 `enc_text_t/enc_spk_t`를 **전역 공유 (shared)** 로 구현 — Conformer 원본과 동일 방식 |
| **Orth loss가 다른 loss를 압도** | 1024-dim teacher feature를 96-dim으로 매핑한 뒤 inner product 제곱의 평균 | Conformer에서 weight=1.0이 안정적이었던 전례 존재. 그대로 따름. 로그로 per-loss scale 모니터링 |
| **SpkCls가 수렴 안 함** (251-way) | wav2vec의 mean-pool이 Conformer 대비 정보량 차이 | Conformer에서 88~90% 도달했던 사례 있음. batch=8 effective로 충분. 실패 시 weight 하향 조정 |
| **Student text-only proj 단일경로로 KD signal 부족** | E1은 tae+sproj가 전체 latent 정렬, E2는 text 서브공간만 | Conformer E2에서 clean WER 개선 확인됨. 같은 가설 유지 |
| **2 GPU batch=4로 effective batch=8** | 기존 4GPU × 4=16 대비 절반 | lr 동일 유지 (E2가 E1 대비 공정비교라 batch 영향 상쇄). 필요 시 후속 실험에서 lr 보정 |
| **kd_warmup_epochs=10 동안 disen 모듈이 업데이트 안 됨** | E1은 KD 자체가 CTC에 포함되지만 disen의 orth/spk_cls는 GRP-KD 경로에 포함 → GRP-KD가 warmup 전에도 흘러야 함 | `training_step`을 재확인: 현재 `grpkd` 호출이 CTC와 독립적으로 warmup 여부와 무관하게 항상 합산됨 (`train_wav2vec.py:1041-1052`). 의도대로 동작 |
| **Track B teacher가 frozen base-960h인데 speaker 정보가 충분히 담겨 있지 않을 수 있음** | large-960h는 1024 dim이라 speaker 표현 여유가 있으나, base-960h는 768 dim + 학습이 content 중심이라 spk 정보가 제한적일 수 있음 | `train/grp_spk_cls`와 `train/grp_spk_acc`로 모니터링. Track B에서 spk_acc가 50% 이하면 teacher가 speaker 정보를 많이 가지고 있지 않음을 시사 → 그 경우에도 orth 제약 자체는 독립적으로 작동 가능 |
| **Track B student 파라미터 수 급감** | d=384, heads=6, ffn=1536은 base 대비 ~1/4 크기 | `grp_latent_dim=96` 고정이므로 latent space는 동일. student가 latent로 project될 때만 차원 차이 흡수. CTC 수렴 확인 필요 — sanity 체크로 10 epoch 내 val/wer 감소 추세 확인 |

---

## 6. 로그 계획

### 6.1 Loss 분해 로깅 (wandb)

```
train/ctc                — CTC loss
train/logit_kd           — logit distillation KL
train/grp_rec            — per-pair L_rec 평균
train/grp_fm             — per-pair L_FM 평균
train/grp_df             — per-pair L_DF 평균
train/grp_orth    (E2만) — per-pair L_orth 평균
train/grp_spk_cls (E2만) — per-pair L_spk CE 평균
train/grp_spk_acc (E2만) — per-pair mean(spk_cls.argmax == speaker_ids)
train/total              — 최종 total_loss
```

### 6.2 Val/Test

```
val/ctc           — CTC val loss
val/wer           — greedy decode WER
test/wer_clean    — test.clean WER (엔딩 시점)
test/wer_other    — test.other WER (엔딩 시점)
```

### 6.3 Checkpoint

- `outputs/wav2vec/e_grp_kd/` (E1), `outputs/wav2vec/e_grp_kd_disen_orth/` (E2)
- `val/wer` 기준 top-3 저장
- `last.ckpt` 별도 저장

### 6.4 Sanity 체크 포인트

- Step 1: 1 epoch 이내에 `train/total`이 감소 추세인지
- Step 2: `train/grp_orth`가 초기값 → epoch 10 시점까지 monotonic decrease
- Step 3: `train/grp_spk_acc`가 epoch 30까지 50% 이상 도달
- Step 4: `val/wer`가 E1 대비 ≥0.5%p 개선되는지 (epoch 50 기준)

---

## 7. 실행 스크립트 구성 계획

모든 스크립트는 `scripts/train/wav2vec/grp_kd_orth/` 아래에 둔다.

### 7.1 Track A — E1 (`A_E1_large_base.sh`)

```bash
#!/bin/bash
export WANDB_API_KEY=...
OUT=outputs/wav2vec/grp_kd_orth/A_E1_large_base
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 python train_wav2vec.py \
  --wandb_project DAG-KD-wav2vec \
  --wandb_run wav2vec_grp_kd_A_E1 \
  --out "$OUT" \
  --teacher_name facebook/wav2vec2-large-960h \
  --student_name facebook/wav2vec2-base-960h \
  --random_init_student True \
  --load_pretrained_feature_extractor True \
  --freeze_feature_extractor True \
  --use_ctc True --use_logit_kd True \
  --kd_alpha 0.1 --kd_temperature 1.0 \
  --use_layer_kd False --use_flow False --use_diffkd False --use_disent False \
  --use_txt_spk_probe False \
  --use_grp_kd True \
  --grp_latent_dim 96 --grp_fm_steps 8 --grp_diff_steps 9 \
  --grp_rec_weight 1.0 --grp_gen_weight 1.0 \
  --grp_disen_mode 0 \
  --batch_size 4 --epochs 100 --gpus 2 \
  --learning_rate 1e-4 --warmup_epochs 5 --kd_warmup_epochs 10 \
  2>&1 | tee "$OUT/train.log"
```

### 7.2 Track A — E2 (`A_E2_large_base_orth.sh`)

A_E1과 동일, 변경점만:
```bash
CUDA_VISIBLE_DEVICES=2,3
OUT=outputs/wav2vec/grp_kd_orth/A_E2_large_base_orth
--wandb_run wav2vec_grp_kd_A_E2
--grp_disen_mode 1 --grp_orth_weight 1.0 --grp_spk_cls_weight 1.0
```

### 7.3 Track B — E1 (`B_E1_base_half.sh`)

A_E1을 기반으로 teacher/student 변경:
```bash
CUDA_VISIBLE_DEVICES=0,1
OUT=outputs/wav2vec/grp_kd_orth/B_E1_base_half
--wandb_run wav2vec_grp_kd_B_E1
--teacher_name facebook/wav2vec2-base-960h
--student_name facebook/wav2vec2-base-960h
--student_hidden_size 384
--student_num_heads 6
--student_intermediate_size 1536
--grp_disen_mode 0
```

### 7.4 Track B — E2 (`B_E2_base_half_orth.sh`)

B_E1과 동일, 변경점만:
```bash
CUDA_VISIBLE_DEVICES=2,3
OUT=outputs/wav2vec/grp_kd_orth/B_E2_base_half_orth
--wandb_run wav2vec_grp_kd_B_E2
--grp_disen_mode 1 --grp_orth_weight 1.0 --grp_spk_cls_weight 1.0
```

---

## 8. 결과 분석 템플릿 (Step 6용)

실험 완료 후 이 섹션에 다음을 채워 넣는다:

```markdown
### Table 1: Track A — Large → Base

| ID | 방법 | dev_clean | dev_other | test_clean | test_other |
| --- | --- | --- | --- | --- | --- |
| A-E1 | GRP-KD (ver4) | ? | ? | ? | ? |
| A-E2 | + Orth + SpkCls | ? | ? | ? | ? |

### Table 2: Track B — Base → Base-half (12L, d=384)

| ID | 방법 | dev_clean | dev_other | test_clean | test_other |
| --- | --- | --- | --- | --- | --- |
| B-E1 | GRP-KD (ver4) | ? | ? | ? | ? |
| B-E2 | + Orth + SpkCls | ? | ? | ? | ? |

분석:
1. **Track 내부**: Clean split 개선폭이 Conformer E2(−1.0%p) 수준과 일치하는가
2. **Track 간**: Track A와 B에서 orth-disen 개선 방향이 같은가 (같으면 layer depth 독립적 효과 입증)
3. **Other split trade-off**: Conformer E2에서 관찰된 other 소폭 저하가 두 track 모두에서 재현되는가
4. **Loss 수렴**: orth/spk_cls가 안정적으로 감소했는가. spk_acc가 50% 이상 도달했는가 (특히 Track B teacher base가 spk 정보를 충분히 가진가)
5. **실패 시**: Track 한 쪽만 실패하면 그 track 고유 요인(layer align 또는 student scale-down) 때문. 양쪽 모두 실패면 wav2vec 구조 자체에서 orth-disen이 작동 안 하는 것 → 대안 분리 제약(CLUB, GRL) 검토
```

---

## 9. Next Steps

본 레포트 승인 후:

1. **Step 3 (코드 구현)**:
   - `models_wav2vec.py`에 `SpeakerClassifier` 추가 및 `GRPKDModule` 확장 (disen_mode 분기)
   - `train_wav2vec.py` argparse 인자/생성자/training_step 수정
   - `scripts/train/wav2vec/grp_kd_orth/` 아래 4개 스크립트 작성 (A-E1, A-E2, B-E1, B-E2)
   - dry-run (1 epoch) Track A + Track B 각각 1건씩 수행: loss/shape/checkpoint 정상 여부
   - **커밋 포인트**: (a) 모델 수정, (b) 학습 루프 수정, (c) 스크립트 추가, (d) dry-run 결과

2. **Step 4 (사용자 검토)** — 대규모 학습 착수 전 사용자 승인

3. **Step 5 (PR)** — 승인 후 `main ← exp/wav2vec-grp-kd-orth` PR 생성 및 100-epoch 학습 시작
