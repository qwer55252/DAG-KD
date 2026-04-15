# GRP-KD Based — Latent Disentanglement 실험

## 1. 문제 정의 — 현재 구조의 공백

이전 실험(exp/layerwise-spk-grl)에서 DAG-KD 베이스라인 위에 레이어별 GRL + CLUB disentanglement를 적용했으나 E1~E4 모두 negative result로 종료됐다. 실패 원인 분석 결과, **disentanglement 메커니즘의 문제가 아니라 KD 신호 품질 자체의 문제**로 판단했다.

```
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

```
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

### 제어 플래그

```bash
--disen_mode    # 0=E1, 1=E2(orth), 2=E3(CLUB MI)
--orth_weight   # orth_loss 또는 club_mi_loss 가중치 (default 1.0)
--spk_cls_weight  # speaker classifier loss 가중치 (default 1.0)
```

### 공통 하이퍼파라미터

```
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

---

## 6. 결과 분석

| Exp | 방법 | dev_clean | dev_other | test_clean | test_other |
| --- | --- | --- | --- | --- | --- |
| E1 | GRP-KD ver4 baseline | 12.0 | 28.3 | 12.4 | 28.8 |
| E2 | E1 + Orth + SpkCls | **11.0** | 28.8 | **11.5** | 29.5 |
| E3 | E1 + CLUB MI + SpkCls | 13.1 | 31.0 | 13.3 | 32.0 |

**E2 (Orthogonal)**: E1 대비 clean split에서 유의미한 개선(dev_clean -1.0%p, test_clean -0.9%p). Teacher latent를 text/speaker subspace로 분리하고 text subspace에만 FM+Diffusion KD를 적용하는 것이 효과적임을 확인했다. 다만 other split에서는 소폭 저하가 관찰되며, 화자 다양성이 높은 환경에서 분리 효과가 제한적임을 시사한다.

**E3 (CLUB MI)**: 전 split에서 E1 대비 성능 저하. CLUB의 variational network 학습(ll_loss)과 MI 최소화(mi_upper)가 레이어마다 16회 반복되어 학습 신호가 불안정해졌을 가능성이 있다. Variational network가 충분히 수렴하려면 100 epoch의 joint training으로는 부족했을 수 있다.

**결론**: 직교 제약(E2)이 CLUB MI(E3)보다 이 설정에서 더 효과적이다. E2를 기반으로 후속 실험을 진행한다.
