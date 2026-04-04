# Multi-Layer Factor KD 실험 레포트

**브랜치명**: `exp/multi-layer-factor-kd`
**실험 디렉토리**: `experiments/multi_layer_factor_kd/`

---

### Hypothesis

> 기존 DAG-KD는 Teacher의 spk factor를 layer 4 하나, txt factor를 layer 16 하나에서만 추출한다. 단일 레이어 추출은 해당 레이어의 특성에 과도하게 의존하며, Teacher 인코더의 계층적 표현 구조를 충분히 활용하지 못한다. 여러 레이어에서 factor를 추출하고 Shared AutoEncoder를 통해 동일한 latent space로 투영한 뒤, 대응하는 Student 레이어와 1:1로 KD하면 Teacher의 계층적 표현 구조가 Student에 더 풍부하게 전달되어 ASR WER이 개선될 것이다.

---

### 실험 구성

| 실험 ID | 설명 | KD 방법 | spk layers | txt layers |
| --- | --- | --- | --- | --- |
| **E1 (Baseline)** | 기존 single-layer DAG-KD | — | [4] | [16] |
| **E2** | Multi-Layer Shared AE + MSE KD | MSE | [2, 4, 6] | [12, 14, 16] |
| **E3** | Multi-Layer Shared AE + Flow/DiffKD | Flow Matching + DiffKD | [2, 4, 6] | [12, 14, 16] |

- E1은 MI_ablation best 설정(ts 단독, Phys=✓, Rec=✓, test WER 0.3223) 재사용
- E2, E3는 동일 베이스 위에서 KD 모듈만 교체

---

### 공통 설정 (E1 Baseline과 동일)

```
teacher: stt_en_conformer_ctc_small
data: LibriSpeech train-clean-100 / dev.clean / test.clean
use_ctc: True
use_logit_kd: True
use_flow: True, flow_steps: 8
use_diffkd: True
use_disent: True
disen_mi_pairs: ts
disen_lll_weight: 1.0
disen_mi_weight: 1.0
batch_size: 32, epochs: 100
```

---

### Multi-Layer Shared AE 구조

**기존 (Single-Layer)**:
```
Teacher layer4  → spk_enc(176→96, k=1) → spk_emb       (1개)
Teacher layer16 → txt_enc(176→96, k=1) → txt_emb       (1개)
```

**신규 (Multi-Layer + Shared AE)**:
```
Teacher spk_layers = [2, 4, 6]    (하위 3개)
Teacher txt_layers = [12, 14, 16] (상위 3개)

tch_feats[1] (B,176,T) ─┐
tch_feats[3] (B,176,T) ──┤→ Shared spk_enc(176→96, k=1) → spk_emb_1, spk_emb_2, spk_emb_3
tch_feats[5] (B,176,T) ─┘   Shared spk_dec(96→176, k=1) → Rec loss 평균

tch_feats[11] (B,176,T) ─┐
tch_feats[13] (B,176,T) ──┤→ Shared txt_enc(176→96, k=1) → txt_emb_1, txt_emb_2, txt_emb_3
tch_feats[15] (B,176,T) ─┘   Shared txt_dec(96→176, k=1) → Rec loss 평균
```

**1:1 KD 매핑**:
```
[E2 MSE]
stu_feats[1] → stu_spk_enc(96→96) → MSE(·, spk_emb_1)
stu_feats[3] → stu_spk_enc(96→96) → MSE(·, spk_emb_2)   (shared stu_spk_enc)
stu_feats[5] → stu_spk_enc(96→96) → MSE(·, spk_emb_3)

stu_feats[11] → stu_txt_enc(96→96) → MSE(·, txt_emb_1)
stu_feats[13] → stu_txt_enc(96→96) → MSE(·, txt_emb_2)  (shared stu_txt_enc)
stu_feats[15] → stu_txt_enc(96→96) → MSE(·, txt_emb_3)

[E3 Flow/Diff]
위 경로에서 MSE 대신 FlowMatchingModule / DiffKDModule 적용 (각 쌍 독립)
```

---

### MI 최소화

Shared AE로 모든 emb이 동일한 latent space에 있으므로 CLUB 추정기 1개(club_ts)를 공유:

```
MI_loss = mean(MI(txt_emb_i, spk_emb_i) for i=1,2,3)
LLL_loss = mean(ll_loss(txt_emb_i, spk_stat_i) for i=1,2,3)
```

spk_stat_i: 각 spk_emb_i → backbone → stats pooling → Linear(192→96)

---

### 추가 Args

```bash
--use_multi_layer_kd      # bool, default: False
--multi_kd_spk_layers     # int list, default: [2, 4, 6]
--multi_kd_txt_layers     # int list, default: [12, 14, 16]
--multi_layer_kd_type     # "mse" | "generative", default: "mse"
--multi_layer_kd_weight   # float, default: 1.0
```

---

### 평가 지표

| Metric | 타입 | 주기 | 기대 방향 |
| --- | --- | --- | --- |
| `test/wer` | scalar | epoch | ↓ |
| `probe/txt_spk_acc` | scalar | epoch | ↓ (speaker 정보 제거 확인) |
| `disen/cka_ts` | scalar | N step | ↓ (txt↔spk 유사도) |
| `train/multi_spk_kd` | scalar | epoch | 수렴 확인 |
| `train/multi_txt_kd` | scalar | epoch | 수렴 확인 |
| `train/mi_upper` | scalar | epoch | ↓ |
