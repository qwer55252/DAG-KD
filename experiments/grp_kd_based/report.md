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

## 3. 실험 설계

`E1(GRP-KD ver4)` 을 베이스라인으로, E2/E3에서 분리 제약 방식만 변경한다.
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
```

```text
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

E7 (Layer-selective disentanglement):
  E4 구조 전체 유지 (disen_mode=3, layer_disen_decay=0.8)
  → orth/GRL 가중치를 레이어별로 선택적 적용
  → layer_weight = 1.0 - 0.8 × (layer_idx / 15)
     layer 0(최하위): weight=1.0, layer 15(최상위): weight=0.2
  → 하위 레이어(speaker/acoustic 정보 多) → 강한 분리 제약
  → 상위 레이어(linguistic/text 정보 多) → 약한 분리 제약 (표현 보존)
  → grl_alpha=0.1 고정 (E4와 동일), annealing 없음
```

### 제어 플래그

```bash
--disen_mode          # 0=E1, 1=E2(orth), 2=E3(CLUB MI), 3=E4/E5/E7(orth+GRL), 4=E6(orth+GRL×2)
--orth_weight         # orth_loss 가중치 (default 1.0)
--spk_cls_weight      # speaker classifier loss 가중치 (default 1.0)
--grl_weight          # teacher GRL CE loss 가중치 (default 1.0)
--grl_alpha           # 고정 GRL alpha (default 0.1)
--grl_anneal          # True: DANN-style alpha annealing (default False)
--grl_alpha_max       # annealing 최대 alpha (default 1.0)
--grl_s_weight        # student GRL CE loss 가중치 (disen_mode=4, default 1.0)
--layer_disen_decay   # E7: orth/GRL 레이어별 감쇠율 (default 0.0=균일, 0.8=E7)
```

### 공통 하이퍼파라미터

```text
데이터: LibriSpeech train-clean-100, epochs=100, batch=32, GPU×1
latent_dim=96, diffusion_steps=9, flow_steps=8
kd_alpha=0.1, kd_temperature=1.0, kd_loss_type=mse
```

---

## 4. 실험 테이블

### Table 1: 분리 제약 방식 비교

| ID | 방법 | 분리 제약 | SpkCls | dev_clean | dev_other | test_clean | test_other |
| --- | --- | :---: | :---: | --- | --- | --- | --- |
| E1 | GRP-KD ver4 (baseline) | ❌ | ❌ | 12.0 | 28.3 | 12.4 | 28.8 |
| E2 | E1 + Orth + SpkCls | Orthogonal | ✅ | **11.0** | 28.8 | **11.5** | 29.5 |
| E3 | E1 + CLUB MI + SpkCls | CLUB MI | ✅ | 13.1 | 31.0 | 13.3 | 32.0 |
| E4 | E2 + GRL on z_t_text | Orth + GRL | ✅ | **11.0** | **28.3** | **11.5** | **28.7** |
| E5 | E4 + GRL alpha annealing | Orth + GRL(anneal) | ✅ | 11.1 | 28.4 | 11.4 | 29.0 |
| E6 | E4 + student GRL | Orth + GRL×2 | ✅ | 11.3 | 28.8 | 11.6 | 29.0 |
| E7 | E4 + layer-selective decay | Orth + GRL(layer↓) | ✅ | - | - | - | - |

---

## 5. 구현 세부사항

### E2/E3 공통 변경사항

- **Teacher side**: `TeacherAE` 제거 → `enc_text_t(176→96)` + `enc_spk_t(176→96)` 병렬 인코더
- **Student side**: `StudentProjector` 제거 → `proj_text_s(88→96)` 단일 인코더 (spk 인코더 없음)
  - KD 타겟(z_t_text)이 이미 speaker-clean이므로 FM+Diffusion이 자연히 text-only 표현으로 수렴
- **Decoder**: `lat_dec(96→176)` — z_t_text + z_t_spk 합산 후 teacher feature 재구성
- **Speaker ID**: NeMo의 `return_sample_id`(dataset index)를 manifest 기반 룩업 테이블로 speaker class 변환

### E3 추가 사항

- `ClubGaussian(x_dim=96, y_dim=96, hidden_size=128, max_samples=2048)` — models.py에서 재사용
- `club.mi_upper`: K=8 negative sampling으로 MI upper bound 추정
- `club.ll_loss`: variational network NLL 학습 (별도 weight 없이 total loss에 합산)

### E4 추가 사항

- `GradientReversalLayer(alpha=0.1)` — models.py에서 재사용
- `spk_cls_text`: z_t_text에 부착된 별도 speaker classifier (SpkCls_spk와 동일 구조)
- gradient flow: `z_t_text → GRL → spk_cls_text → CE` — enc_text_t에 역전된 gradient 전달
- `grl_alpha=0.1`: 보수적 설정으로 recon/FM loss와의 gradient 충돌 방지

### E5 추가 사항

- E4와 동일 구조, `--grl_anneal True --grl_alpha_max 0.5` 추가
- `training_step`마다 `α = 0.5 × (2/(1+exp(-10p))-1)` 계산 후 `self.grl.alpha` 동적 업데이트
- `v/grl_alpha` wandb 로깅으로 annealing 진행 추적 가능
- epoch 30 기준 α ≈ 0.45, epoch 50 이후 α ≈ 0.497로 수렴

### E6 추가 사항

- E4 전체 유지 (disen_mode=4), student GRL branch 추가
- `grl_s = GradientReversalLayer(alpha=0.1)` + `spk_cls_s = SpeakerClassifier(96, 251)`
- gradient flow: `z_s_text → GRL_s → spk_cls_s → CE` — proj_text_s에 역전된 gradient 전달
- KD(FM+Diffusion)는 원본 z_s_text 사용 — GRL_s는 별도 adversarial branch로 gradient만 역전
- `v/grl_s` wandb 로깅

---

## 6. 결과 분석

| Exp | 방법 | dev_clean | dev_other | test_clean | test_other |
| --- | --- | --- | --- | --- | --- |
| E1 | GRP-KD ver4 baseline | 12.0 | 28.3 | 12.4 | 28.8 |
| E2 | E1 + Orth + SpkCls | **11.0** | 28.8 | **11.5** | 29.5 |
| E3 | E1 + CLUB MI + SpkCls | 13.1 | 31.0 | 13.3 | 32.0 |
| **E4** | E2 + GRL on z_t_text | **11.0** | **28.3** | **11.5** | **28.7** |
| E5 | E4 + GRL alpha annealing | 11.1 | 28.4 | 11.4 | 29.0 |
| E6 | E4 + student GRL | 11.3 | 28.8 | 11.6 | 29.0 |

**E2 (Orthogonal)**: E1 대비 clean split에서 유의미한 개선(dev_clean -1.0%p, test_clean -0.9%p). Teacher latent를 text/speaker subspace로 분리하고 text subspace에만 FM+Diffusion KD를 적용하는 것이 효과적임을 확인했다. 다만 other split에서는 소폭 저하가 관찰되며, 화자 다양성이 높은 환경에서 분리 효과가 제한적임을 시사한다.

**E3 (CLUB MI)**: 전 split에서 E1 대비 성능 저하. CLUB의 variational network 학습(ll_loss)과 MI 최소화(mi_upper)가 레이어마다 16회 반복되어 학습 신호가 불안정해졌을 가능성이 있다. Variational network가 충분히 수렴하려면 100 epoch의 joint training으로는 부족했을 수 있다.

**E4 (GRL)**: E2에서 z_t_text에 GRL + SpkCls_text를 추가하여 enc_text_t가 speaker 정보를 적극적으로 제거하도록 유도했다. clean split은 E2 수준을 유지하면서 other split이 E1 baseline 수준으로 완전히 회복됐다(dev_other 28.8→28.3%, test_other 29.5→28.7%).

**E5 (GRL annealing)**: E4 대비 전 split에서 미미한 차이(dev_clean +0.1%p, dev_other +0.1%p). DANN annealing이 초반 KD 안정화에 도움이 되지만 최종 성능 차이는 미미하다. α가 0.5까지 점진적으로 증가해도 E4(고정 0.1) 대비 significant한 개선 없음. 다만 chained assignment 버그로 인해 grl_sum이 실제보다 2배 집계되었을 가능성이 있으므로 버그 수정 후 재실험이 필요하다.

**E6 (student GRL)**: E4 대비 전 split에서 소폭 저하(dev_clean +0.3%p, dev_other +0.5%p). student z_s_text에 GRL을 추가해도 KD 타겟인 z_t_text가 이미 speaker-clean하므로 student가 자연히 text-only 표현으로 수렴한다는 가설이 지지된다. 오히려 student-side adversarial pressure가 FM/Diffusion 학습을 방해할 수 있다. 동일한 chained assignment 버그 영향 존재.

---

## 7. Speaker Accuracy 분석 (Linear Probe)

E2/E4 체크포인트에서 `eval_spk_probe.py`로 linear probe를 학습하여 z_t_text / z_t_spk의 speaker 정보 잔존량을 정량화했다. train-clean-100(251명)을 80/20으로 분리해 probe train/eval로 사용했다 (dev/test 화자는 train 화자와 겹치지 않아 zero-shot 평가 불가).

| 지표 | E2 (Orth) | E3 (CLUB MI) | E4 (Orth+GRL) | Random baseline |
| --- | --- | --- | --- | --- |
| z_t_spk speaker acc | 88.91% | **90.29%** | 88.63% | 0.40% |
| **z_t_text speaker acc** | 14.17% | **1.51%** | 3.56% | **0.40%** |

**z_t_spk**: 전 실험 ~89~90%로 안정적. 분리 제약 방식에 무관하게 enc_spk_t가 화자 정보를 정상적으로 담고 있음을 확인.

**z_t_text**: 분리 성능 순위는 E3(1.51%) > E4(3.56%) >> E2(14.17%). 그러나 WER 성능 순위는 E4 > E2 >> E3로 역전된다. disentanglement 수치가 낮다고 반드시 ASR 성능이 좋은 것은 아님을 보여준다.

**E3의 역설**: z_t_text speaker acc가 가장 낮음(=분리 가장 잘됨)에도 WER이 가장 나쁘다. CLUB variational network 추정 실패(`v/club_mi = -15.6`, 음수 MI는 수학적으로 불가)로 인해 z_t_text 자체가 ASR에 유용한 정보를 잃어버린 것으로 해석된다. 분리는 됐지만 표현이 붕괴된 사례.

**WER과의 연결**: z_t_text speaker acc와 WER other split 변화가 정량적으로 연결된다. E2(14.17%) → other split 악화, E4(3.56%) → other split 회복. 단, E3처럼 표현 자체가 붕괴되면 이 관계가 성립하지 않는다.

**결론**: E4가 현재 best. E5/E6 모두 E4를 유의미하게 개선하지 못했다. 다음 방향으로 E7(layer-selective disentanglement)을 시도한다.

---

## 8. E7 설계 근거 — layer-selective disentanglement

**배경**: E5/E6 결과를 통해 단순히 GRL 강도를 높이거나 student에 GRL을 추가하는 것만으로는 other split 성능을 추가 개선하기 어렵다는 것을 확인했다. 한편 E4 chained assignment 버그 분석 중 모든 16개 레이어에 동일한 강도로 orth/GRL을 적용하는 것이 비효율적일 수 있다는 점에 주목했다.

**근거**: ASR 인코더의 레이어별 역할 분담은 잘 알려진 현상이다. 하위 레이어는 음향/화자 특성(speaker identity, pitch, timbre)을 주로 담고, 상위 레이어는 음소/언어 정보를 담는다. E4에서 상위 레이어에 강한 speaker 분리 제약을 주면 linguistic 표현까지 손상될 수 있다.

**E7 가설**: orth와 GRL 제약을 하위 레이어에 집중시키고 상위 레이어에서는 완화하면, speaker 정보는 음향 표현 단계에서 제거되고 상위 레이어의 linguistic 표현은 온전히 보존되어 WER(특히 other split)이 추가 개선된다.

**구현**: `layer_weight = 1.0 - 0.8 × (layer_idx / 15)`

- layer 0: weight=1.0 (E4와 동일)
- layer 7: weight=0.47
- layer 15: weight=0.2 (20% 강도만 적용)
- 버그 수정(chained assignment) 포함된 코드베이스에서 실행
