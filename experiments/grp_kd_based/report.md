# GRP-KD Based — Latent Disentanglement 실험

## 1. 문제 정의 — 현재 구조의 공백

이전 실험(exp/layerwise-spk-grl)에서 DAG-KD 베이스라인 위에 레이어별 GRL + CLUB disentanglement를 적용했으나 E1~E4 모두 negative result로 종료됐다. 실패 원인 분석 결과, **disentanglement 메커니즘의 문제가 아니라 KD 신호 품질 자체의 문제**로 판단했다.

```text
[DAG-KD 구조의 한계]
Student(88) → Linear(88→176) → Teacher space(176)
  - 차원 불일치: 88 → 176 직접 투영 시 정렬 불안정
  - Distillation 신호가 불충분한 상태에서 disentanglement 제약만 추가
  - 결과: KD 기반이 약해 disentanglement 효과를 측정하기 어려움

[GRP-KD 구조 (asr_train_diffm.py ver4)]
Teacher(176) → TeacherAE(176→96) → z_t (96)
                                  → TeacherAE(96→176) → Recon Loss
Student(88)  → StudentProj(88→96) → z_s (96)
  FM(pre):   z_s → [MLP velocity] → ≈ z_t        → fm_loss_pre
  Diffusion: z_s → NoiseAdapter(γ) → Denoiser(9step) → ≈ z_t  → kd_loss_post

Total = CTC + 0.1·LogitKD + Σ_layers(recon + fm_pre + kd_post)
```

GRP-KD는 Teacher/Student를 모두 96-dim latent로 압축한 뒤 동일 공간에서 정렬하여 KD 신호가 훨씬 안정적이다. 이 구조 위에서 동일한 disentanglement 아이디어를 재적용하는 것이 이번 실험의 방향이다.

미검증 질문:

- GRP-KD의 96-dim latent에 **화자(speaker) 정보가 혼재**되어 있으며, 이를 분리하면 ASR WER이 개선되는가?
- Teacher latent의 분리 제약으로 **직교(Orthogonal)와 CLUB MI** 중 어느 쪽이 더 효과적인가?

---

## 2. 가설

> **GRP-KD의 latent 공간(96-dim)에서 text/speaker subspace를 분리하고, FM+Diffusion KD를 text subspace에만 적용하면, Student가 ASR에 필요한 언어 표현만 학습하여 WER이 추가로 개선된다. 분리 제약으로는 기하학적 직교 조건이 정보이론적 CLUB MI보다 이 설정에서 더 효과적일 것이다.**

세부 가설:

1. **Teacher latent 분리**: text subspace(96)와 speaker subspace(96)로 병렬 인코딩 후 KD 대상을 text에만 한정하면 speaker noise가 제거된 KD 신호를 student가 받는다
2. **직교 제약**: `(z_t_text * z_t_spk).sum(dim=1).pow(2).mean()` 최소화만으로도 충분한 분리가 달성된다
3. **CLUB MI**: variational network 학습이 joint training에서 수렴하지 못해 직교보다 열등할 것이다

---

## 3. 공통 설정

### 공통 하이퍼파라미터

```text
데이터: LibriSpeech train-clean-100, epochs=100, batch=32, GPU×1
latent_dim=96, diffusion_steps=9, flow_steps=8
kd_alpha=0.1, kd_temperature=1.0, kd_loss_type=mse
```

### 제어 플래그

```bash
--disen_mode          # 0=E1, 1=E2(orth), 2=E3(CLUB MI), 3=E4/E5/E7(orth+GRL), 4=E6(orth+GRL×2)
                      # 6=E14(3-way GPRE), 7=E15(3-way Conv1d), 8=E16(nontxt merged), 9=E17(F0-GRL)
--orth_weight         # orth_loss 가중치 (default 1.0)
--spk_cls_weight      # speaker classifier loss 가중치 (default 1.0)
--grl_weight          # teacher GRL CE loss 가중치 (default 1.0)
--grl_alpha           # 고정 GRL alpha (default 0.1)
--grl_anneal          # True: DANN-style alpha annealing (default False)
--grl_alpha_max       # annealing 최대 alpha (default 1.0)
--grl_s_weight        # student GRL CE loss 가중치 (disen_mode=4, default 1.0)
--layer_disen_decay   # E7: orth/GRL 레이어별 감쇠율 (default 0.0=균일, 0.8=E7)
--stage1_epochs       # Two-Stage Stage 1 길이 (default=0: 비활성화)
--pros_orth_weight    # prosody orth loss 가중치 (default 1.0)
--pros_sup_weight     # prosody supervision loss 가중치 (default 1.0)
```

---

## 4. 전체 실험 결과 테이블

| ID | 방법 | 분리 제약 | SpkCls | dev_clean | dev_other | test_clean | test_other |
| --- | --- | :---: | :---: | --- | --- | --- | --- |
| E1 | GRP-KD ver4 (baseline) | ❌ | ❌ | 12.0 | 28.3 | 12.4 | 28.8 |
| E2 | E1 + Orth + SpkCls | Orthogonal | ✅ | **11.0** | 28.8 | **11.5** | 29.5 |
| E3 | E1 + CLUB MI + SpkCls | CLUB MI | ✅ | 13.1 | 31.0 | 13.3 | 32.0 |
| E4 | E2 + GRL on z_t_text | Orth + GRL | ✅ | **11.0** | **28.3** | **11.5** | **28.7** |
| E5 | E4 + GRL alpha annealing | Orth + GRL(anneal) | ✅ | 11.1 | 28.4 | 11.4 | 29.0 |
| E6 | E4 + student GRL | Orth + GRL×2 | ✅ | 11.3 | 28.8 | 11.6 | 29.0 |
| E7 | E4 + layer-selective decay | Orth + GRL(layer↓) | ✅ | ≈12.05 | - | - | - |
| E8 | E4 + CRD (InfoNCE) | Orth + GRL + CRD | ✅ | ≈12.22† | - | - | - |
| E9 | E4 + Top-K KD (k=8) | Orth + GRL | ✅ | >E4‡ | - | - | - |
| E10a | E4 + Two-Stage (s1=20) | Orth + GRL | ✅ | **10.88** | 28.49 | **11.30** | 29.02 |
| E10b | E4 + Two-Stage (s1=30) | Orth + GRL | ✅ | 10.96 | **28.21** | 11.34 | **28.88** |
| E10c | E4 + Two-Stage (s1=25) | Orth + GRL | ✅ | **10.59** | **27.80** | **11.22** | **28.23** |
| E13 | E10c + 3-Way F0 precompute | Orth + GRL + pros(Conv1d) | ✅ | 11.22 | 28.53 | 11.59 | 29.24 |
| E14 | E10c + 3-Way GPRE | Orth + GRL + pros(GPRE+F0seq) | ✅ | 11.29 | 28.28 | 11.41 | 28.39 |
| E15 | E14 + pros=Conv1d(tch_feat) + 3-orth 분리 | Orth×3 + GRL | ✅ | 11.14 | 27.93 | 11.29 | 28.05 |
| E16 | E15 → nontxt merged (spk+pros) | Orth(txt⊥nontxt) + GRL | ✅ | 11.00 | 27.77 | 11.22 | 28.14 |
| E17 | E16 + F0-GRL on text + spk-norm F0 | Orth + GRL + F0-GRL | ✅ | - | - | - | - |

† E8은 epoch 128에서 조기 중단, 미수렴 상태의 추정값  
‡ E9은 epoch 40 이전에 E4 대비 현저히 높은 WER로 조기 중단, 최종 수치 없음

---

## 5. 실험별 설계 및 분석

### Phase 1: 분리 제약 방식 비교 (E1~E6)

#### 설계 및 구현

`E1(GRP-KD ver4)` 을 베이스라인으로, E2/E3에서 분리 제약 방식만 변경하고, E4~E6에서 GRL 추가를 시도한다.  
`use_ctc=True`, `use_logit_distillation=True`, `model_version=4`는 전 실험 고정.

```text
E1 (Baseline):
  TeacherAE(176→96) + StudentProj(88→96)
  → FM(pre) + Diffusion 전체 latent에 적용
  → 분리 제약 없음

E2 (Orthogonal):
  enc_text_t(176→96) + enc_spk_t(176→96)  [병렬, teacher]
  proj_text_s(88→96)                       [student, text only]
  → Recon: (z_t_text + z_t_spk) → lat_dec(96→176)
  → Orth:  (z_t_text * z_t_spk).sum(dim=1).pow(2).mean()
  → SpkCls: z_t_spk → mean pool → fc → CE loss
  → FM(pre) + Diffusion: z_s_text ↔ z_t_text 만 적용

E3 (CLUB MI):
  E2와 동일 구조
  → Orth 대신: club.mi_upper(z_t_text, z_t_spk, K=8)
             + club.ll_loss(z_t_text, z_t_spk)  [variational net 학습]

E4 (GRL):
  E2 구조 그대로 유지 (enc_text_t + enc_spk_t + orth + SpkCls_spk)
  → 추가: z_t_text → GRL(α=0.1) → SpkCls_text → CE loss
  → GRL이 enc_text_t에 역전된 gradient를 전달 → z_t_text에서 speaker 정보 적극 제거

E5 (GRL + alpha annealing):
  E4 구조 그대로 (disen_mode=3), grl_anneal=True
  → α(p) = 0.5 × (2/(1+exp(-10p))-1), p = global_step / total_steps
  → 초반 KD 안정화 우선, 후반 adversarial pressure 점진적 증가
  → epoch 30 기준 α ≈ 0.45 (E4 고정 0.1 대비 4.5배)

E6 (GRL teacher + student):
  E4 구조 전체 유지 (disen_mode=4)
  → 추가: z_s_text → GRL(α=0.1) → SpkCls_s → CE loss
  → teacher enc_text_t + student proj_text_s 양쪽 모두 speaker-free 유도
  → KD(FM+Diffusion)는 원본 z_s_text 사용 (GRL은 별도 adversarial branch)
```

**공통 변경사항 (E2/E3)**:

- Teacher side: `TeacherAE` 제거 → `enc_text_t(176→96)` + `enc_spk_t(176→96)` 병렬 인코더
- Student side: `StudentProjector` 제거 → `proj_text_s(88→96)` 단일 인코더 (spk 인코더 없음)
  - KD 타겟(z_t_text)이 이미 speaker-clean이므로 FM+Diffusion이 자연히 text-only 표현으로 수렴
- Decoder: `lat_dec(96→176)` — z_t_text + z_t_spk 합산 후 teacher feature 재구성
- Speaker ID: NeMo의 `return_sample_id`(dataset index)를 manifest 기반 룩업 테이블로 speaker class 변환

**E3 추가 사항**:

- `ClubGaussian(x_dim=96, y_dim=96, hidden_size=128, max_samples=2048)` — models.py에서 재사용
- `club.mi_upper`: K=8 negative sampling으로 MI upper bound 추정
- `club.ll_loss`: variational network NLL 학습 (별도 weight 없이 total loss에 합산)

**E4 추가 사항**:

- `GradientReversalLayer(alpha=0.1)` — models.py에서 재사용
- `spk_cls_text`: z_t_text에 부착된 별도 speaker classifier (SpkCls_spk와 동일 구조)
- gradient flow: `z_t_text → GRL → spk_cls_text → CE` — enc_text_t에 역전된 gradient 전달
- `grl_alpha=0.1`: 보수적 설정으로 recon/FM loss와의 gradient 충돌 방지

**E5 추가 사항**:

- E4와 동일 구조, `--grl_anneal True --grl_alpha_max 0.5` 추가
- `training_step`마다 `α = 0.5 × (2/(1+exp(-10p))-1)` 계산 후 `self.grl.alpha` 동적 업데이트
- `v/grl_alpha` wandb 로깅으로 annealing 진행 추적 가능
- epoch 30 기준 α ≈ 0.45, epoch 50 이후 α ≈ 0.497로 수렴

**E6 추가 사항**:

- E4 전체 유지 (disen_mode=4), student GRL branch 추가
- `grl_s = GradientReversalLayer(alpha=0.1)` + `spk_cls_s = SpeakerClassifier(96, 251)`
- gradient flow: `z_s_text → GRL_s → spk_cls_s → CE` — proj_text_s에 역전된 gradient 전달
- KD(FM+Diffusion)는 원본 z_s_text 사용 — GRL_s는 별도 adversarial branch로 gradient만 역전

#### E1~E6 결과 분석

**E2 (Orthogonal)**: E1 대비 clean split에서 유의미한 개선(dev_clean -1.0%p, test_clean -0.9%p). Teacher latent를 text/speaker subspace로 분리하고 text subspace에만 FM+Diffusion KD를 적용하는 것이 효과적임을 확인했다. 다만 other split에서는 소폭 저하가 관찰되며, 화자 다양성이 높은 환경에서 분리 효과가 제한적임을 시사한다.

**E3 (CLUB MI)**: 전 split에서 E1 대비 성능 저하. CLUB의 variational network 학습(ll_loss)과 MI 최소화(mi_upper)가 레이어마다 16회 반복되어 학습 신호가 불안정해졌을 가능성이 있다. Variational network가 충분히 수렴하려면 100 epoch의 joint training으로는 부족했을 수 있다.

**E4 (GRL)**: E2에서 z_t_text에 GRL + SpkCls_text를 추가하여 enc_text_t가 speaker 정보를 적극적으로 제거하도록 유도했다. clean split은 E2 수준을 유지하면서 other split이 E1 baseline 수준으로 완전히 회복됐다(dev_other 28.8→28.3%, test_other 29.5→28.7%). **E5 이전까지 best.**

**E5 (GRL annealing)**: E4 대비 전 split에서 미미한 차이(dev_clean +0.1%p, dev_other +0.1%p). DANN annealing이 초반 KD 안정화에 도움이 되지만 최종 성능 차이는 미미하다. α가 0.5까지 점진적으로 증가해도 E4(고정 0.1) 대비 significant한 개선 없음.

**E6 (student GRL)**: E4 대비 전 split에서 소폭 저하(dev_clean +0.3%p, dev_other +0.5%p). student z_s_text에 GRL을 추가해도 KD 타겟인 z_t_text가 이미 speaker-clean하므로 student가 자연히 text-only 표현으로 수렴한다는 가설이 지지된다. 오히려 student-side adversarial pressure가 FM/Diffusion 학습을 방해할 수 있다.

---

### Phase 2: KD 신호 개선 시도 (E7~E9, 전부 실패)

#### E7 설계 — Layer-Selective Disentanglement

**배경**: E5/E6 결과를 통해 단순히 GRL 강도를 높이거나 student에 GRL을 추가하는 것만으로는 other split 성능을 추가 개선하기 어렵다는 것을 확인했다. E4 chained assignment 버그 분석 중 모든 16개 레이어에 동일한 강도로 orth/GRL을 적용하는 것이 비효율적일 수 있다는 점에 주목했다.

**근거**: ASR 인코더의 레이어별 역할 분담은 잘 알려진 현상이다. 하위 레이어는 음향/화자 특성(speaker identity, pitch, timbre)을 주로 담고, 상위 레이어는 음소/언어 정보를 담는다. E4에서 상위 레이어에 강한 speaker 분리 제약을 주면 linguistic 표현까지 손상될 수 있다.

**가설**: orth와 GRL 제약을 하위 레이어에 집중시키고 상위 레이어에서는 완화하면, speaker 정보는 음향 표현 단계에서 제거되고 상위 레이어의 linguistic 표현은 온전히 보존되어 WER(특히 other split)이 추가 개선된다.

```text
E7 (Layer-selective disentanglement):
  E4 구조 전체 유지 (disen_mode=3, layer_disen_decay=0.8)
  → layer_weight = 1.0 - 0.8 × (layer_idx / 15)
     layer 0(최하위): weight=1.0, layer 15(최상위): weight=0.2
  → 하위 레이어(speaker/acoustic 정보 多) → 강한 분리 제약
  → 상위 레이어(linguistic/text 정보 多) → 약한 분리 제약 (표현 보존)
  → grl_alpha=0.1 고정 (E4와 동일), annealing 없음
  → 버그 수정(chained assignment) 포함된 코드베이스에서 실행
```

#### E8 설계 — CRD (Contrastive Representation Distillation)

E4 구조에 InfoNCE 기반 CRD loss 추가. z_s_text와 z_t_text 간 contrastive alignment로 KD 신호 품질 향상 시도. `crd_weight=1.0`, K=31(배치 내 negative), 16개 레이어 전체 적용.

#### E9 설계 — Top-K Layer KD

상위 8개 레이어(8~15)에만 FM+Diffusion KD 적용. student capacity를 linguistic 표현 형성에 집중시키는 시도.

#### E7~E9 결과 분석

**E7 (Layer-selective disentanglement)**: dev_clean≈12.05%로 E1 수준으로 퇴보. 설계 의도는 상위 레이어의 linguistic 표현을 보존하기 위해 상위 레이어의 orth/GRL 가중치를 낮추는 것이었다. 그러나 enc_text_t가 모든 레이어에서 공유되는 단일 Conv1d이므로 layer_weight를 낮춰도 "레이어별 독립적 제약"이 아닌 전체 gradient 총량 감소만 일어났다. 상위 레이어에서 GRL을 20%로 줄인 결과 disentanglement가 전반적으로 약해져 speaker 정보가 z_t_text에 다시 혼입됐고 WER이 E1 수준으로 퇴보했다. **핵심 실패 원인: shared encoder 구조에서는 layer-selective 제약이 구조적으로 불가능하다.**

**E8 (CRD, Contrastive Representation Distillation)**: epoch 128에서 조기 중단, dev_clean≈12.22%로 미수렴. CRD InfoNCE loss(weight=1.0)가 16개 레이어에 걸쳐 누적되어 총 loss에서 지나치게 큰 비중을 차지했다(v/crd≈51.8→31.9 범위, fm_pre/kd_post와 동일 스케일). 배치 크기 32에서 positive 1쌍 대비 negative 31개는 충분히 다양하지 않으며, 같은 배치 내 동일 화자 발화가 false negative로 작용했을 가능성이 있다. CRD가 generative KD(FM+Diffusion)와 보완적으로 작동하려면 훨씬 낮은 weight(≤0.1)와 더 많은 negative가 필요하다.

**E9 (Top-K Layer KD, k=8)**: epoch 40 이전 조기 중단. 상위 8개 레이어(8~15)에만 KD를 적용하면 student capacity를 집중시킬 수 있다는 가설이었으나, 하위 레이어(0~7) KD 신호가 없으면 student encoder의 기초 acoustic feature 형성이 teacher와 다른 방향으로 흘러 상위 레이어 KD 자체가 효과를 잃었다. KD signal 총량이 절반으로 줄어 초반 수렴이 오히려 느려진 것도 원인이다.

**E7~E9 공통 교훈**: KD 신호를 어떻게 "선택"하거나 "추가"하는 방식으로는 E4의 벽을 넘지 못했다. 문제는 KD 신호의 질이나 양이 아니라 **CTC와 KD의 gradient 충돌**이라는 학습 dynamics 자체에 있었다.

---

### Phase 3: Two-Stage Training (E10)

#### 설계 및 동기

E5~E9까지 KD 신호 품질 개선(layer-selective, CRD, top-k)에 집중했으나 모두 E4를 넘지 못했다. 근본 원인을 재검토하면, **CTC loss와 KD loss의 gradient 충돌**이 학습 전반에 걸쳐 최적화를 방해할 수 있다:

- CTC gradient: "현재 student feature로 ASR 잘해"
- KD gradient: "teacher feature 방향으로 이동해"

두 신호가 동시에 경쟁하면 student encoder가 어중간한 지점에 수렴할 수 있다. 특히 초반 학습에서 student feature가 random initialization 상태일 때 이 충돌이 가장 심각하다.

**가설**: KD + disen loss로 student feature를 teacher 구조에 먼저 pre-initialize한 뒤, CTC loss를 추가하면 gradient 충돌 없이 ASR 최적화가 더 효과적으로 이루어져 E4 대비 WER이 개선된다.

이론적 근거: FitNets (Romero et al., 2015) — student 중간 레이어를 teacher hint로 먼저 학습(Stage 1)한 뒤 전체 KD fine-tuning(Stage 2)이 동시 학습 대비 우수함을 image classification에서 검증. 우리 student는 scratch 초기화이므로 이 이론이 직접 적용된다.

| ID | Stage 1 (KD only) | Stage 2 (Full) | 비고 |
|---|---|---|---|
| E10a | epoch 1~20 | epoch 21~100 | 짧은 pre-init |
| E10b | epoch 1~30 | epoch 31~100 | 긴 pre-init |
| E10c | epoch 1~25 | epoch 26~100 | 중간값 탐색 |

**Stage 1 losses** (CTC 없음):
```
loss = fm_loss + diff_loss + orth + spk_cls + grl + kd_alpha × logit_KD
```
- logit KD 포함: student CTC head에 teacher soft label로 방향성 제공
- CTC hard label supervision 제외: ASR gradient 차단

**Stage 2 losses** (E4 동일):
```
loss = CTC + kd_alpha × logit_KD + fm_loss + diff_loss + orth + spk_cls + grl
```

구현: `training_step`에서 `self.current_epoch < self.stage1_epochs` 조건으로 CTC loss 포함 여부 제어. 나머지 구조는 E4와 완전히 동일.

```python
if self.stage1_epochs > 0 and self.current_epoch < self.stage1_epochs:
    total_loss = kd_terms + disen_terms  # CTC 제외
else:
    total_loss = ctc_loss + kd_terms + disen_terms  # E4 동일
```

**리스크**: Stage 1에서 CTC 없이 돌면 conformer encoder가 ASR과 무관한 방향으로 drift 가능. Stage 1이 너무 길면 Stage 2에서 CTC 복구에 시간 소요.

#### E10 결과 분석

**E10a (stage1=20)**: dev_clean **10.88%**, test_clean **11.30%** — E1~E9 중 처음으로 E4(11.0%)를 넘었다. Stage 1에서 CTC 없이 KD+disen만으로 student feature를 teacher 구조에 pre-initialize한 효과가 clean split에서 명확히 나타났다. 단, dev_other 28.49%, test_other 29.02%로 E4 대비 other split이 소폭 악화됐다. Stage 1이 20 epoch으로 짧아 disen(orth/GRL)이 충분히 수렴하기 전에 CTC가 켜진 것으로 해석된다.

**E10b (stage1=30)**: dev_clean 10.96%, dev_other **28.21%**, test_other **28.88%**. Stage 1을 30 epoch으로 늘리자 disen이 더 충분히 작동하여 other split이 E4 수준을 회복했다. 반면 clean split 이득(10.96%)은 E10a(10.88%) 대비 줄었다. Stage 2에서 CTC를 학습할 수 있는 기간이 70 epoch으로 E10a(80 epoch)보다 짧아 clean 수렴이 덜 됐을 가능성이 있다.

**E10a vs E10b 트레이드오프 요약**:

| | clean 개선 | other 개선 | 판단 |
| --- | :---: | :---: | --- |
| E10a (s1=20) | ✅ +0.12%p vs E4 | ❌ -0.19%p | clean 우선 상황에서 유리 |
| E10b (s1=30) | △ +0.04%p vs E4 | ✅ +0.09%p | 균형 측면에서 유리 |

두 실험 모두 E4를 완전히 지배(모든 split 개선)하지는 못했다. stage1 길이가 clean/other 균형을 결정하는 핵심 변수임이 확인됐다.

**E10c (stage1=25)**: dev_clean **10.59%**, dev_other **27.80%**, test_clean **11.22%**, test_other **28.23%** — 4개 split 전부 E4를 넘었다. E10a의 clean 이득과 E10b의 other 이득이 동시에 실현됐다. stage1=25가 sweet spot으로 확인: 20ep은 disen 수렴 부족, 30ep은 Stage 2 CTC 기간 부족 — 25ep은 둘 다 충족했다.

E4 대비 최종 개선: dev_clean -0.41%p, dev_other -0.50%p, test_clean -0.28%p, test_other -0.47%p. **현재 best: E10c.**

**Two-Stage 유효성 결론**: CTC-KD gradient 충돌 가설이 실험적으로 지지됐다. E4까지 불가능했던 dev_clean 11% 벽 돌파(10.59%)를 달성했으며, Two-Stage 방향에서 추가 최적화 가능성이 있다.

---

### Phase 4: Prosody 분리 시도 (E13~E17)

#### E13 — 3-Way F0 Precompute (Negative Result)

**설계**: E10c 위에 3번째 disentanglement 축으로 prosody encoder 추가. teacher feature(B,176,T) → Conv1d(1×1) → z_t_pros(B,96,T), pros supervision으로 발화 단위 평균 F0 (f0_stats_train.pt, N×2) 사용.

**결과**: dev_clean 11.22%, dev_other 28.53%, test_clean 11.59%, test_other 29.24% — E10c 대비 4개 split 전부 퇴보.

**실패 원인**:

1. **pros_sup gradient 소멸** (`v/pros_sup_epoch = 0.036`): teacher feature → Conv1d → mean F0 예측은 linear path에서 trivially easy. 수 epoch 만에 수렴 후 gradient 소멸.
2. **z_t_pros = 잔차 공간**: pros_sup gradient 소멸 후 z_t_pros는 "text/spk 이후 남는 임의 잔차"로 전락.
3. **3-way 직교 제약의 이중 압박**: z_t_text가 spk와 pros 양쪽으로부터 직교 gradient를 받아 표현 용량이 압박됨.

---

#### E14 — 3-Way GPRE (Partial Positive)

**설계**: E13의 trivially easy pros supervision 문제를 해결하기 위해 pros encoder를 GlobalProsodyReferenceEncoder(mel→GPRE)로 교체.

| 항목 | E13 | E14 |
| --- | --- | --- |
| pros encoder 입력 | teacher_feat (B,176,T) | mel (B,80,T_mel) — 독립 경로 |
| pros encoder 구조 | Conv1d(1×1) | GlobalProsodyReferenceEncoder(Conv2d×3+GRU) |
| pros supervision | mean F0 (발화 단위 2 스칼라) | 프레임별 F0/energy MSE (voiced mask 포함) |
| f0 데이터 | f0_stats_train.pt (N,2) | f0_seq_train.pt (N,T_max,2) dict |

파이프라인:

```text
mel (B,80,T_mel)
  → GlobalProsodyReferenceEncoder [Conv2d×3(stride=2) + GRU]
  → ref_seq (B, T_mel/8, 96)
  → transpose + F.interpolate(size=T)
  → z_t_pros (B, 96, T)

pros_proj: Linear(96→2)
  → pred (B, T, 2) ← 프레임별 [f0, energy] 예측
  → pros_sup_loss = MSE_voiced(pred_f0, f0_target) + MSE(pred_energy, energy_target)
```

Stage 구성: Stage 1(epoch 0-24): KD+disen+pros_sup, Stage 2(epoch 25-100): +CTC

**결과**: dev_clean 11.29%, dev_other 28.28%, test_clean 11.41%, test_other 28.39%

| 지표 | 초반(ep ~10) | 후반(ep 99) | 해석 |
| --- | --- | --- | --- |
| v/pros_sup (GPRE) | ~2.28 | ~1.34 | 부분 학습됨 (random=2.0) |
| v/pros_orth | 매우 높음 | ~0.0 | 2 epoch 내 소멸 |

**성공**: GPRE pros_sup이 random 수준(2.0) 아래로 내려왔다(1.34). E13에서 실패했던 pros 학습이 mel 독립 경로 덕분에 부분적으로 달성됐다.

**실패**: pros_orth가 2 epoch 만에 0으로 소멸했다. GPRE(mel) 입력이 teacher feature와 완전히 다른 modality라 구조적 직교성이 자동으로 확보돼버린 것. pros_orth constraint는 의미 없는 상태로 학습이 진행됐다.

**결론**: GPRE 방식은 orth constraint가 trivially 만족되어 z_t_pros가 실제로 text/spk로부터 독립된 표현인지 보장할 수 없다. pros_sup 학습 자체는 됐으나 disentanglement의 의미가 없어진다.

---

#### E15 — Teacher-Feat Conv1d Pros + 3-Orth 분리 로깅

**설계 동기**: E14의 trivial orth 문제를 해결하기 위해 pros encoder 입력을 mel → teacher feature로 변경. teacher feature 기반 pros는 text/spk encoder와 동일 입력을 공유하므로 orth constraint가 non-trivial하게 작동한다. E14에서 합산 로깅되던 pros_orth를 3개(text⊥spk / text⊥pros / spk⊥pros)로 분리 기록.

```text
t_bct (B, T, 176)
  → enc_text_t  (Conv1d 1×1) → z_t_text  (B, 96, T)
  → enc_spk_t   (Conv1d 1×1) → z_t_spk   (B, 96, T)
  → enc_pros_t  (Conv1d 1×1) → z_t_pros  (B, 96, T)

recon: lat_dec(z_t_text + z_t_spk + z_t_pros) → teacher feat
orth:          (z_t_text * z_t_spk).pow(2).mean()
pros_orth:     (z_t_text * z_t_pros).pow(2).mean()
spk_pros_orth: (z_t_spk * z_t_pros).pow(2).mean()
pros_sup:      pros_proj(z_t_pros) → frame F0/energy MSE (voiced mask)
```

**결과**: dev_clean 11.14%, dev_other 27.93%, test_clean 11.29%, test_other 28.05%

| 지표 | ep 0 | ep 99 | 해석 |
| --- | --- | --- | --- |
| v/pros_sup | ~32 (≈2.0/layer) | ~32 (≈2.0/layer) | **완전히 학습 안됨** |
| v/orth (txt⊥spk) | 높음 | ~2.4 | 정상 수렴 |
| v/pros_orth (txt⊥pros) | 높음 | ~0.25 | 낮음 |
| v/spk_pros_orth (spk⊥pros) | 높음 | ~3.48 | **14배 더 얽혀있음** |

**실패 원인**: spk⊥pros orth가 3.48로 매우 높았다. teacher feature 내에서 화자 정보와 prosody 정보가 high correlation을 가져, spk⊥pros constraint가 z_t_pros에서 F0 방향을 모두 제거해버린 결과다. enc_pros_t는 text와 spk 양쪽에서 직교한 방향을 찾아야 하는데, 이 잔여 공간에 F0 신호가 없어서 pros_sup이 random level에 고착됐다.

---

#### E16 — Merged Nontxt Encoder

**설계 동기**: E15 분석에서 spk⊥pros가 pros_sup 학습을 무력화시킨다는 것을 발견했다. 사실 spk와 pros를 서로 분리할 필요는 없다 — 목표는 **text에서 spk+pros를 분리하는 것**이었다. 따라서 enc_spk_t와 enc_pros_t를 하나의 enc_nontxt_t로 합치고 spk⊥pros constraint를 제거했다.

```text
t_bct (B, T, 176)
  → enc_text_t   (Conv1d 1×1) → z_t_text   (B, 96, T)
  → enc_nontxt_t (Conv1d 1×1) → z_t_nontxt (B, 96, T)

recon:    lat_dec(z_t_text + z_t_nontxt) → teacher feat
orth:     (z_t_text * z_t_nontxt).pow(2).mean()   [text⊥nontxt만]
spk_cls:  spk_cls(z_t_nontxt) → CE
pros_sup: pros_proj(z_t_nontxt) → frame F0/energy MSE
grl:      spk_cls_text(grl(z_t_text)) → CE [reversed]
```

**결과**: dev_clean 11.00%, dev_other 27.77%, test_clean 11.22%, test_other 28.14%

| Epoch | WER(clean) | pros_sup | orth(txt⊥nontxt) | spk_cls | grl |
| --- | --- | --- | --- | --- | --- |
| 11 | 27.51% | 32.26 | 145.75 | 57.18 | 60.06 |
| 25 | - | 31.93 | 13.88 | 46.19 | 85.48 |
| 98 | 11.00% | 31.93 | 5.47 | 40.61 | 88.21 |

**결과**: pros_sup이 여전히 random level(≈2.0/layer × 16 = 32)에서 변화 없음. spk⊥pros constraint 제거에도 불구하고 pros_sup은 전혀 학습되지 않았다.

**실패 원인 재해석**: 문제는 spk⊥pros가 아니라 **enc_text_t가 F0 방향을 먼저 선점**하기 때문이었다. recon_loss가 text와 nontxt를 상보적으로 만드는 과정에서, text_enc가 F0-correlated 방향을 먼저 잡으면 orth constraint가 nontxt를 그 직교 방향으로 밀어내 nontxt에 F0 신호가 남지 않는다.

**WER 소폭 개선 원인**: pros_sup 학습 여부와 무관하게 3-way의 gradient 충돌 노이즈가 사라진 구조 단순화 효과로 해석된다.

---

#### E17 — F0-GRL on Text + Speaker-Normalized F0

**설계 동기**: E16 분석에서 근본 원인이 확인됐다: enc_text_t가 F0 정보를 선점한다. E15~E16까지 **pros encoder에서 F0를 당기는** 방향을 시도했으나 모두 실패. E17은 반대 방향, **text encoder에서 F0를 밀어내는** 방향을 시도한다. 추가로 E16에서 spk_cls와 pros_sup이 같은 화자 pitch range 정보를 두고 경쟁하는 문제를 speaker-normalized F0 target으로 해결한다.

E16(disen_mode=8) 구조에 두 가지 추가:

##### A: F0-GRL on z_t_text

```text
z_t_text → GRL(α=0.1) → f0_pred_text (Linear 96→1) → MSE(F0_target)
```

GRL이 gradient를 역전시켜 enc_text_t가 F0를 예측 못하도록 adversarial 학습. speaker GRL과 동일 GRL layer 공유, 동일 grl_weight 적용.

##### B: Speaker-normalized F0 targets

```text
pros_sup target: (F0_raw - speaker_mean_F0) / speaker_std_F0
```

글로벌 정규화 F0 → 화자별 평균/표준편차로 정규화. within-speaker pitch variation만 남겨 spk_cls(화자 ID)와 pros_sup(화자 내 pitch 변화)이 서로 다른 정보를 학습하도록 유도. per-speaker F0 통계는 f0_seq_train.pt + spk_table에서 scatter_add로 사전 계산.

**가설**: A가 text에서 F0를 밀어내면 recon_loss가 nontxt에게 F0 방향을 채우도록 당기고, B가 pros_sup target의 spk-pros correlation을 제거하여 pros_sup이 처음으로 random level(2.0/layer) 아래로 내려온다.

핵심 검증 지표: `v/pros_sup < 2.0/layer × 16 = 32`

하이퍼파라미터: E16과 동일 (`disen_mode=9, stage1_epochs=25, grl_alpha=0.1, orth_weight=1.0, spk_cls_weight=1.0, grl_weight=1.0, pros_sup_weight=1.0, kd_alpha=0.1, epochs=100, batch=32`)

**결과**: 진행 중

---

## 6. Speaker Accuracy 분석 (Linear Probe)

E2/E4 체크포인트에서 `eval_spk_probe.py`로 linear probe를 학습하여 z_t_text / z_t_spk의 speaker 정보 잔존량을 정량화했다. train-clean-100(251명)을 80/20으로 분리해 probe train/eval로 사용했다 (dev/test 화자는 train 화자와 겹치지 않아 zero-shot 평가 불가).

| 지표 | E2 (Orth) | E3 (CLUB MI) | E4 (Orth+GRL) | Random baseline |
| --- | --- | --- | --- | --- |
| z_t_spk speaker acc | 88.91% | **90.29%** | 88.63% | 0.40%|
| **z_t_text speaker acc** | 14.17% | **1.51%** | 3.56% | **0.40%** |

**z_t_spk**: 전 실험 ~89~90%로 안정적. 분리 제약 방식에 무관하게 enc_spk_t가 화자 정보를 정상적으로 담고 있음을 확인.

**z_t_text**: 분리 성능 순위는 E3(1.51%) > E4(3.56%) >> E2(14.17%). 그러나 WER 성능 순위는 E4 > E2 >> E3로 역전된다. disentanglement 수치가 낮다고 반드시 ASR 성능이 좋은 것은 아님을 보여준다.

**E3의 역설**: z_t_text speaker acc가 가장 낮음(=분리 가장 잘됨)에도 WER이 가장 나쁘다. CLUB variational network 추정 실패(`v/club_mi = -15.6`, 음수 MI는 수학적으로 불가)로 인해 z_t_text 자체가 ASR에 유용한 정보를 잃어버린 것으로 해석된다. 분리는 됐지만 표현이 붕괴된 사례.

**WER과의 연결**: z_t_text speaker acc와 WER other split 변화가 정량적으로 연결된다. E2(14.17%) → other split 악화, E4(3.56%) → other split 회복. 단, E3처럼 표현 자체가 붕괴되면 이 관계가 성립하지 않는다.
