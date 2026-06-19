# GRP-KD + ContentVec-style Disentanglement (v1) — 실험 설계 레포트

> **Track A 환경 (large→base)** 에서 GRP-KD baseline에 정보 분리 메커니즘을 추가하여,
> teacher representation의 **content 정보만 student로 selectively distillation** 하는 방향으로 발전시킨다.

---

## 1. 연구 배경 및 동기

### 1.1 현재까지의 결과

| 실험 | 환경 | best WER | 비고 |
|---|---|---|---|
| A_E1 (GRP-KD baseline) | large→base | **14.2%** | 정보 분리 없음 |
| A_E2 (GRP-KD + Orth) | large→base | 14.9% (-0.7%p) | orthogonality 제약 → **악화** |
| B_E1 / B_E2 (base→half) | base→half | 17.9% / 19.4% | 동일 패턴 |

A_E2의 orthogonal disentanglement가 두 트랙 모두에서 baseline 대비 악화시켰다. 이전 실험의 실패 원인 가설:
- Orthogonality 제약이 너무 강해 representation 표현력을 제한
- text/spk subspace를 임의로 분할하는 supervision이 약함 (단순 spk_cls만)
- "어떤 정보가 ASR에 필요한지" 명시적 신호 부재

### 1.2 본 실험의 차별점

이번 실험은 **명시적 supervision으로 speaker 정보를 latent에서 제거**하고, **필요할 때만 conditioning으로 주입**하는 ContentVec 방식을 GRP-KD에 통합한다.

ContentVec (ICML 2022, https://github.com/auspicious3000/contentvec) 의 3가지 disentanglement 메커니즘 중 KD 컨텍스트에 적용 가능한 2가지:

1. **Speaker Adversarial (Mechanism 1+2 변형)**: TeacherAE latent에 speaker classifier + Gradient Reversal → speaker info를 latent에서 제거
2. **Speaker Conditioning (Mechanism 3)**: FM/Diffusion meta-encoder에 speaker embedding을 condition으로 주입 → Student는 speaker info를 carrying할 필요 없음

---

## 2. 핵심 가설

> **H1**: Teacher latent에서 speaker 정보를 제거하고 student가 content-only target을 따라가면, 작은 student의 capacity가 ASR에 중요한 content 정보 학습에 집중되어 WER이 향상된다.
>
> **H2**: Speaker 정보를 conditioning으로 분리해 제공하면, FM/Diffusion pathway가 content prediction에 집중할 수 있어 더 효율적으로 학습한다.
>
> **H3 (조합)**: 두 메커니즘은 상호 보완적이다. Adversarial은 latent를 정화하고, conditioning은 KD target의 prediction 부담을 줄인다.

---

## 3. 4개 실험 설계 (변인 통제)

모든 실험은 **Track A 환경**으로 통일:
- Teacher: `facebook/wav2vec2-large-960h` (24L, dim=1024, frozen)
- Student: `facebook/wav2vec2-base-960h` config, **random init + pretrained CNN feature extractor**, freeze CNN
- Data: LibriSpeech train-clean-100 (100h)
- Batch: 8 × 2 GPUs (effective 16)
- LR: 1.5e-4, warmup 5 epoch, KD warmup 10 epoch
- Epochs: 100
- 평가: dev-clean / dev-other / test-clean WER

| 실험 | Speaker Adversarial | Speaker Conditioning | 목적 |
|---|---|---|---|
| **C_E1** | ✗ | ✗ | **Baseline** (= A_E1 재현, 14.2%) |
| **C_E2** | **✓** | ✗ | Speaker adversarial **단독** 효과 |
| **C_E3** | ✗ | **✓** | Speaker conditioning **단독** 효과 |
| **C_E4** | **✓** | **✓** | **Full** (E2+E3 조합 효과) |

### 3.1 비교 분석 (ablation)

- **C_E1 vs C_E2**: latent에서 speaker 제거의 KD 효과
- **C_E1 vs C_E3**: speaker conditioning의 KD 효과
- **C_E2/C_E3 vs C_E4**: 두 메커니즘의 시너지 또는 중복

이 4개 결과로 다음 결론을 내릴 수 있는 ablation table 완성:
- 어떤 분리 방법이 단독으로 효과적인가?
- 두 방법이 상호 보완적인가, 중복적인가?
- A_E2 (orthogonality)와 비교해 명시적 supervision이 우수한가?

---

## 4. 변경할 모듈

### 4.1 `models_wav2vec.py`

#### 4.1.1 신규 클래스
```python
class GRPSpeakerAdversarial(nn.Module):
    """Latent → Speaker classification (with GRL)
    z_t (B, L, T) → mean-pooled (B, L) → MLP → speaker logits (B, num_spk)
    GRL을 통과하면 latent로 흘러가는 gradient가 reversed → speaker info 제거 학습
    """
    def __init__(self, latent_dim, num_spk, hidden=256, alpha=1.0):
        ...
    def forward(self, z_latent_bct):
        # GRL → meanpool → 2-layer MLP → logits
        ...
```

#### 4.1.2 `GRPFlowMatchingModule` 확장
```python
class GRPFlowMatchingModule(nn.Module):
    def __init__(self, ..., use_speaker_cond=False, num_spk=0, spk_emb_dim=64):
        ...
        if use_speaker_cond:
            self.spk_embed = nn.Embedding(num_spk, spk_emb_dim)
            # meta_encoder의 입력 차원 +spk_emb_dim
            self.meta_encoder = nn.Sequential(
                nn.Linear(latent_dim + time_embed_dim + spk_emb_dim, hidden_dim),
                ...
            )

    def forward(self, s_latent_bct, t_latent_bct, speaker_ids=None, steps=None):
        ...
        if self.use_speaker_cond and speaker_ids is not None:
            # speaker embedding을 (B, T, spk_emb_dim)으로 broadcast
            spk_emb = self.spk_embed(speaker_ids).unsqueeze(1).expand(-1, T, -1)
            h = torch.cat([x, t_emb, spk_emb], dim=-1)
        else:
            h = torch.cat([x, t_emb], dim=-1)
        velocity = self.meta_encoder(h)
        ...
```

#### 4.1.3 `GRPSimpleDenoiser` 확장 (FiLM-style speaker conditioning)
```python
class GRPSimpleDenoiser(nn.Module):
    def __init__(self, ..., use_speaker_cond=False, num_spk=0, spk_emb_dim=64):
        ...
        if use_speaker_cond:
            self.spk_embed = nn.Embedding(num_spk, spk_emb_dim)
            self.spk_proj = nn.Linear(spk_emb_dim, latent_dim * 2)  # γ, β

    def forward(self, z_in, speaker_ids=None):
        # FiLM: γ * net(x) + β
        ...
```

#### 4.1.4 `GRPKDModule` 확장 (`disen_mode=2` 추가)
```python
class GRPKDModule(nn.Module):
    def __init__(self, ..., use_speaker_adv=False, use_speaker_cond=False,
                 num_spk=0, spk_adv_weight=0.1, spk_emb_dim=64):
        ...
        if use_speaker_adv:
            self.spk_adv = GRPSpeakerAdversarial(latent_dim, num_spk)
        # FM/DF에 speaker conditioning 옵션 전파
        self.fm_latent = GRPFlowMatchingModule(..., use_speaker_cond=use_speaker_cond, ...)
        self.denoiser  = GRPSimpleDenoiser(..., use_speaker_cond=use_speaker_cond, ...)

    def forward(self, tch_feats, stu_feats, speaker_ids=None):
        # baseline pipeline + 추가 loss
        for x_t_raw, x_s in pairs:
            z_t, t_rec = self.tae(x_t_raw)
            ...
            # NEW: speaker adversarial loss on z_t
            if self.use_speaker_adv and speaker_ids is not None:
                spk_logits = self.spk_adv(z_t)  # GRL inside
                spk_adv_loss += F.cross_entropy(spk_logits, speaker_ids)

            # FM/DF에 speaker_ids 전달
            fm_loss, _ = self.fm_latent(z_s, z_t.detach(), speaker_ids=speaker_ids)
            z_deno = self.denoiser(z_noisy, speaker_ids=speaker_ids)
            ...
        return L_rec, L_fm, L_df, L_spk_adv
```

#### 4.1.5 `DistilDAGKDWav2Vec2.training_step`
- `speaker_ids` (이미 batch에 있음) 를 `self.grpkd(...)` 호출에 전달
- `L_spk_adv` 를 total loss에 더함:
  ```python
  total = total + grp_rec_w * L_rec + grp_gen_w * (L_fm + L_df) + spk_adv_w * L_spk_adv
  ```

### 4.2 `train_wav2vec.py`

신규 CLI args:
- `--use_speaker_adv` (str2bool, default=False)
- `--use_speaker_cond` (str2bool, default=False)
- `--spk_adv_weight` (float, default=0.1)
- `--spk_emb_dim` (int, default=64)
- `--spk_adv_alpha` (float, default=1.0) — GRL gradient scaling

이미 `num_spk`는 `scan_speakers()`에서 산출됨 → 그대로 사용.

### 4.3 신규 helper class `GradientReversalLayer`
```python
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return _GRL.apply(x, alpha)
```

---

## 5. 실행 스크립트 구성 계획

```
experiments/grp_kd_disen_v1/
  ├── report.md                          (이 파일)
  ├── C_E1_baseline.sh                   GPU 0,1
  ├── C_E2_speaker_adv.sh                GPU 2,3
  ├── C_E3_speaker_cond.sh               GPU 0,1 (E1,E2 끝나면)
  ├── C_E4_full.sh                       GPU 2,3 (E1,E2 끝나면)
  └── run_sequential.sh                  C_E1+C_E2 동시 → C_E3+C_E4 동시
```

각 `.sh`는 `experiments/grp_kd_orth/A_*.sh`와 동일 구조, 차이점만 추가:

```bash
# C_E2 예시
... (공통 args 동일) ...
  --use_grp_kd True \
  --grp_disen_mode 0 \                 # 기존 disen_mode=1 (orth) 사용 안 함
  --use_speaker_adv True \             # NEW
  --use_speaker_cond False \           # NEW
  --spk_adv_weight 0.1 \               # NEW
  ...
```

GPU 할당:
- 8개 GPU 가용 → **2개 batch에 4개 실험 분할** (이전 base2small과 동일 방식)
- 또는 4개 동시 (각 1 GPU, batch=4) — 단 batch=8 효과 떨어짐

**제안**: 2 GPUs/실험 × 2 동시 실행 → 두 batch로 나누어 순차

---

## 6. 예상 리스크 및 대응

| 리스크 | 가능성 | 대응 |
|---|---|---|
| GRL alpha=1.0이 너무 강해 학습 불안정 | 중 | warmup 동안 alpha=0 → 1로 ramp-up (`spk_adv_alpha_max` annealing) |
| spk_adv_weight 잘못 → loss 폭주 | 중 | 0.1 시작, log로 spk_acc 모니터링하며 0.01~0.5 범위 grid sweep 가능성 |
| LibriSpeech 100h speaker 수가 적어 speaker classification 쉬움 → adversarial 효과 약함 | 중 | speaker_ids 통계 사전 확인 (보통 251 spk for clean-100). 충분함. |
| Speaker conditioning이 student를 너무 의존적으로 만듦 (test 시 spk_id 없음) | 낮 | inference 시 spk_id를 zero-vector로 (training 시에도 random dropout 적용 가능) |
| FiLM denoiser와 conditioning FM의 학습 간섭 | 낮 | 단순 concat conditioning으로 통일 (FiLM은 옵션) |
| 기존 GRP-KD baseline과의 코드 호환성 깨짐 | 낮 | `disen_mode=0`일 때 기존 동작 그대로 유지 (additive flag) |

---

## 7. 로그 계획

학습 중 wandb에 다음 메트릭 추가 로깅:

```python
self.log("train/grp_spk_adv",  L_spk_adv,  on_epoch=True)   # adversarial loss
self.log("train/grp_spk_acc",  spk_acc,    on_epoch=True)   # spk classifier accuracy
self.log("train/grp_rec",      L_rec,      on_epoch=True)   # 기존
self.log("train/grp_fm",       L_fm,       on_epoch=True)   # 기존
self.log("train/grp_df",       L_df,       on_epoch=True)   # 기존
```

분석 포인트 (학습 후):
- `grp_spk_acc`이 학습 진행에 따라 **감소**해야 함 → adversarial 동작 확인
- WER 곡선 vs spk_adv 곡선 상관관계 → "speaker 제거가 ASR에 도움"인지 검증
- 4개 실험의 best WER table → 가설 H1, H2, H3 검증

---

## 8. 단계별 작업 계획

| Step | 내용 | 산출물 |
|---|---|---|
| 1 | Worktree 생성 (`exp/wav2vec-grp-kd-disen-v1`) | ✅ 완료 |
| 2 | 본 레포트 작성 | ✅ 본 문서 |
| 3 | 사용자 승인 | **← 현재 단계** |
| 4 | 코드 수정: GRL, GRPSpeakerAdversarial, FM/DF speaker_cond, training_step | 4 files modified |
| 5 | 4개 `.sh` 스크립트 + `run_sequential.sh` 작성 | 5 scripts |
| 6 | Sanity check (1~2 epoch dry-run): tensor shape, log 정상 | 로그 |
| 7 | 사용자 코드 검토 후 → 본 실험 시작 (8 GPUs, ~3일 예상) | wandb runs |
| 8 | 결과 분석 → `experiments/grp_kd_disen_v1/results.md` | 결과 레포트 |

---

## 9. 성공 기준

- **C_E2 또는 C_E3 또는 C_E4가 C_E1(=14.2%) 보다 낮은 WER** → 가설 검증
- WER 외에 **`grp_spk_acc` 감소 확인** → speaker가 실제로 제거됨을 representation 차원에서 입증
- A_E2(14.9%, orth-only)와 비교 → 명시적 supervision의 우월성

목표 WER: **< 13.5%** (1%p 이상 개선 시 의미있는 결과로 판단)

---

## 10. 참고 문헌

1. **ContentVec** (Qian et al., ICML 2022). *ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers*.
   - Paper: https://arxiv.org/abs/2204.09224
   - Code: https://github.com/auspicious3000/contentvec

2. **GRP-KD** (ICASSP 2026 — 우리 baseline).
   - Code: https://github.com/qwer55252/KD-via-FM-in-ASR

3. **Gradient Reversal Layer** (Ganin & Lempitsky, ICML 2015). *Unsupervised Domain Adaptation by Backpropagation*.

---

**검토 요청**: 위 4개 실험 설계 및 가설을 검토해주시고, 진행해도 될지 승인 부탁드립니다.

다음 단계는 Step 4 (코드 수정 및 sanity check) 입니다.
