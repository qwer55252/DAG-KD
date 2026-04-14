# GRP-KD Based 실험 레포트

## 배경 및 동기

이전 실험(exp/layerwise-spk-grl)에서 DAG-KD 베이스라인 위에 레이어별 GRL + CLUB 기반 disentanglement를 적용했으나 E1~E4 모두 성능 개선 없이 negative result로 종료됐다.

실패 원인 분석:

- DAG-KD는 Student(88) → Teacher(176)으로 직접 투영하는 방식이라 차원 불일치 및 정렬 불안정 문제가 있었다
- Distillation 신호 자체가 불충분한 상태에서 disentanglement 제약만 추가된 구조
- **결론**: disentanglement 아이디어 자체의 문제가 아니라 KD 기반 품질의 문제

이에 GRP-KD(asr_train_diffm.py)를 새 베이스라인으로 채택한다. GRP-KD는 Teacher/Student를 모두 96-dim latent로 압축한 뒤 Flow Matching + Diffusion으로 정렬하여 KD 신호 품질 자체가 더 안정적이다. 동일한 disentanglement 아이디어를 더 강한 KD 기반 위에 재적용하는 것이 이번 실험의 방향이다.

---

## GRP-KD 파이프라인 (ver4)

각 encoder layer에서:

```
Teacher(176) → TeacherAE enc(176→96) → z_t (96)
             → TeacherAE dec(96→176) → recon → Recon Loss (MSE)

Student(88)  → StudentProjector(88→96) → z_s (96)

FM(pre):   z_s → [MLP velocity] → z_s_flow ≈ z_t  →  fm_loss_pre
Noise:     z_s → NoiseAdapter(γ) → Z_noisy = γ·z_s + (1-γ)·ε
Denoiser:  Z_noisy → SimpleDenoiser(9 steps) → z_deno ≈ z_t  →  kd_loss_post (MSE)
```

총 Loss (16 레이어 합산):
```
Total = CTC + 0.1·LogitKD + Σ_layers(recon + fm_pre + kd_post)
```

DAG-KD 베이스라인과의 차이:

| 항목 | DAG-KD | GRP-KD ver4 |
|------|--------|-------------|
| KD 공간 | Student(88) → Teacher(176) 직접 투영 | 둘 다 latent(96)으로 압축 후 정렬 |
| Flow Matching | 88-dim에서 velocity 예측 후 176으로 확장 | 96-dim 동일 공간에서 정렬 |
| Diffusion | DiffKDModule (teacher AE + denoiser) | NoiseAdapter(학습 가능 γ) + SimpleDenoiser |
| 레이어 적용 | 선택적 | 전체 16개 레이어 합산 |
| 표현 분리 | CLUB, GRL, prosody predictor | 없음 (순수 KD) |

---

## 실험 테이블

| Exp | 방법 | 가정 | 스크립트 |
|-----|------|------|----------|
| E1 | GRP-KD ver4 (baseline) | GRP-KD 자체 재현 | E1_ver4_baseline.sh |
| E2 | E1 + Latent Disentanglement | teacher latent를 text/spk로 분리하고 text만 KD하면 ASR 성능 개선 | E2_disen_orth.sh |

---

## E1: GRP-KD ver4 Baseline

### 설정

- `model_version=4`: AE + FM(pre) + NoiseAdapter + Diffusion + KD(post)
- `latent_dim=96`, `diffusion_steps=9`, `flow_steps=8`
- `kd_alpha=0.1`, `kd_loss_type=mse`
- `epochs=100`, `batch_size=32`
- wandb: `GRP-based / grp_kd_E1_ver4`

### 결과

| Split | WER (%) |
|-------|---------|
| dev_clean | - |
| dev_other | - |
| test_clean | - |
| test_other | - |

---

## E2: Latent Disentanglement (Orthogonal + Speaker Classifier)

### 가정

GRP-KD의 96-dim latent z_t에는 언어(text) 정보뿐 아니라 화자(speaker) 정보가 혼재되어 있다. Teacher latent를 text/speaker subspace로 분리하고, FM + Diffusion KD를 text subspace에만 적용하면, Student가 ASR에 필요한 언어 표현만 학습하여 WER이 추가로 개선된다.

### 구조

```
Teacher(176) → enc_text_t(176→96) → z_t_text (96)  ─┐
             → enc_spk_t(176→96)  → z_t_spk  (96)  ─┼→ lat_dec(96→176) → Recon Loss
                                                      │
z_t_text ⊥ z_t_spk                                  │  Orthogonality Loss
z_t_spk  → SpeakerClassifier → CE Loss              │

Student(88) → proj_text_s(88→96) → z_s_text (96)
                    │
        FM(pre): z_s_text → z_t_text  (text subspace 정렬)
        Diffusion: NoiseAdapter(z_s_text) → Denoiser → z_t_text
```

DAG-KD 측 disentanglement와의 차이:
- DAG-KD: 176-dim raw feature 공간에서 분리 시도 → 고차원, 불안정
- E2: 96-dim compressed latent에서 분리 → 저차원, 정렬 이미 완료된 공간

### 설정

- `disen_mode=1`
- `orth_weight=1.0`, `spk_cls_weight=1.0`
- 나머지 하이퍼파라미터 E1과 동일
- wandb: `GRP-based / grp_kd_E2_disen_orth`

### 구현 세부사항

- Student 측 spk 인코더 없음: KD 타겟(z_t_text)이 이미 speaker-clean이므로 FM+Diffusion이 z_s_text를 z_t_text로 정렬하면 자연히 text-only 표현이 됨
- Orthogonality loss: teacher side만 적용 — `(z_t_text * z_t_spk).sum(dim=1).pow(2).mean()`
- Speaker ID: NeMo의 `return_sample_id`가 반환하는 dataset index를 manifest 기반 룩업 테이블로 speaker class로 변환

### 결과

| Split | WER (%) |
|-------|---------|
| dev_clean | - |
| dev_other | - |
| test_clean | - |
| test_other | - |

---

## 결과 분석

*(실험 완료 후 작성)*
