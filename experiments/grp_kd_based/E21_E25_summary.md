# Disentanglement 실험 정리 E21–E25

> **Baseline**: E10c — Orth + GRL + SpkCls, `stage1_epochs=25`, dev.clean **10.84%**

---

## E21 — Instance Normalization on z_t_text

### 인용 논문
| 논문 | 인용한 부분 |
|------|------------|
| Huang & Belongie, ICCV 2017 *Arbitrary Style Transfer in Real-time with AdaIN* | "style = feature map의 채널별 mean & variance" → IN이 이를 제거해 content만 보존 |
| Kaneko et al., Interspeech 2019 *StarGAN-VC2* | 음성 변환에서 인코더 feature에 IN 적용 → speaker-independent content 표현 |

### Instance Normalization이란?
텐서 shape: `z_t_text = (B, D, T)` — B=배치, D=차원, T=시간 프레임

InstanceNorm은 **(b, d) 쌍마다** T축 전체의 mean과 std를 계산해 정규화한다:

```python
z_norm[b, d, t] = (z[b, d, t] - mean_t(z[b, d, :])) / std_t(z[b, d, :])
```

즉 **"utterance b의 d번째 차원이 시간 전반에 걸쳐 가지는 평균 수준"** 을 제거한다.

**직관**: 화자 특성(성문, 성도 구조, 음색)은 한 발화 내내 일정하게 유지된다 → 시간에 관계없이 특정 차원이 지속적으로 높거나 낮은 패턴 = speaker bias. 반면 언어 내용(음소, 단어)은 시간에 따라 계속 변한다. IN은 time-constant한 패턴을 제거하고, time-varying한 패턴만 남긴다.

예: 저음 화자는 특정 차원 d가 T 전체에 걸쳐 +3.0 수준을 유지할 수 있다. IN은 이 +3.0 offset을 subtract해서 0으로 만든다 → 화자 bias 소거. 반면 /p/ → /a/ → /t/ 처럼 시간에 따라 바뀌는 음소 패턴은 각 t에서 평균을 빼도 relative 차이가 보존된다.

### 핵심 메소드
`z_t_text (B, D, T)`에 `F.instance_norm()` 적용 → 채널별 utterance-level mean/var 제거 → speaker style 구조적 소거

### 바뀐 파이프라인
```
z_t_text = enc_text_t(t_bct)
z_t_text = F.instance_norm(z_t_text)   ← 추가: 각 채널의 시간축 통계 정규화
recon = dec(z_t_text + z_t_spk)
```
E10c 대비 `--use_text_in True` 1개 플래그 추가. 나머지 동일.

**결과**: 실패. fm_pre 36배 폭등 (0.12 → 4.33). IN이 z_t_text 분포를 근본적으로 바꿔 KD 타깃 붕괴.

---

## E22 — Variational Information Bottleneck (VIB)

### 인용 논문
| 논문 | 인용한 부분 |
|------|------------|
| Alemi et al., ICLR 2017 *Deep Variational Information Bottleneck* | IB 목표 `max I(Z;Y) - β·I(X;Z)`를 variational bound로 근사: `E[-log p(y|z)] + β·KL(N(μ,σ²)‖N(0,I))` |

### Variational Information Bottleneck이란?

**구조**: z_t_text를 고정 벡터가 아닌 가우시안 분포 N(μ, σ²)로 모델링하고, 학습 시 그 분포에서 샘플링한 값을 사용한다. KL loss는 이 분포가 N(0,I)에서 멀어지는 것을 패널티로 준다.

**KL을 N(0,I)에 거는 이유**: N(0,I)는 입력에 대한 정보가 전혀 없는 "무정보 분포"다. KL(N(μ,σ²) ‖ N(0,I)) = 0.5·(μ² + σ² - 1 - log σ²)인데, 이 값이 낮으려면 μ≈0, σ≈1이어야 한다. 즉 **μ가 입력에 따라 크게 달라지면 KL 비용이 올라간다** — 이게 정보량 패널티다. 특정 분포가 정규분포여야 한다는 의미가 아니라, 입력 정보를 많이 인코딩할수록 비용을 치르게 만드는 장치다.

**Disentanglement 메커니즘 (솔직하게)**: VIB는 speaker 정보를 "직접 지우거나" "z_t_spk로 위임시키는" 메커니즘이 아니다. 이 점이 중요하다.

VIB는 z_t_text의 **전체 정보량을 줄이는** 압축 기법이다:

- μ가 입력과 무관할수록 KL이 낮아지므로, 네트워크는 "KL 비용을 지불할 만큼 유용한 정보"만 μ에 담는다
- CTC loss가 있으므로 언어 정보는 유용 → μ에 유지
- speaker 정보는 ASR에 불필요 → KL 비용 대비 이득 없음 → 이론상 압축됨

**한계**: 그러나 z_t_text의 μ(평균)는 여전히 speaker 정보를 담을 수 있다. 네트워크가 μ에 speaker 정보를 인코딩하면서 σ를 작게 유지하면 KL 비용을 최소화하면서 speaker 정보도 보유 가능하다. 즉 VIB만으로는 speaker → z_t_spk 라우팅을 보장하지 못하고, orth/grl 같은 명시적 제약 없이는 disentanglement가 약하다. E22가 E10c를 이기지 못한 이유다.

### 핵심 메소드
```
μ, log_var = vib_mu(z_t_text), vib_logvar(z_t_text)
z_t_text = μ + ε·exp(0.5·log_var)   (학습 시: 분포에서 샘플링)
z_t_text = μ                          (추론 시: 분포의 중심값 사용)
KL loss = -0.5·(1 + log_var - μ² - exp(log_var)).mean()
```

### 바뀐 파이프라인
```
z_t_text = enc_text_t(t_bct)
z_t_text → vib_mu, vib_logvar → reparameterize → z_t_text (VIB)   ← 추가
recon = dec(z_t_text_VIB + z_t_spk)
```
E10c 대비: `orth_weight=0, grl_weight=0` 제거, `vib_beta=0.01` 추가. Orth/GRL을 VIB 단독으로 대체.

**결과**: epoch 99 완료 후 segfault로 WER 미수집. Wandb 추이상 E10c 미달 예상.

---

## E23 — Cross-Covariance Decorrelation

### 인용 논문
| 논문 | 인용한 부분 |
|------|------------|
| Zbontar et al., ICML 2021 *Barlow Twins* | 두 표현의 cross-correlation 행렬을 단위행렬에 가깝게 → D² 차원 간 중복성 제거 |
| Bardes et al., ICLR 2022 *VICReg* | Covariance regularization: `‖C‖_F²` off-diagonal 최소화 수식 및 `/D` 정규화 방식 |

### Cross-Covariance Decorrelation이란?

#### 기존 Orth의 한계 — 1개 scalar 제약

기존 Orth loss `(z_t_text · z_t_spk)²`는 두 벡터 전체의 내적을 0으로 만든다. 이는 "전체 벡터 방향"만 보는 것이라, 특정 차원들 사이의 세밀한 상관관계를 놓칠 수 있다.

예: D=3인 경우
```
z_t_text = [+1.0,  0.0,  0.0]
z_t_spk  = [ 0.0, +1.0,  0.0]
dot product = 0 → orth loss = 0 (제약 만족)
```
그런데 여기서 z_t_text의 0번 차원과 z_t_spk의 1번 차원이 배치 전반에 걸쳐 같이 움직인다면? → 내적은 0이지만 차원 간 상관관계는 존재. Orth loss가 감지 못한다.

#### Cross-Covariance — D×D 행렬로 확장

배치(B개 발화)를 기준으로 "어떤 차원 쌍이 함께 움직이는가"를 행렬로 만든다.

구체적 예시: D=2, B=3 (배치 3개 발화)

```
발화별 z_t_text (배치 정규화 후):    z_t_spk (배치 정규화 후):
  화자A: [+1.1, +0.2]                   [+1.0, -0.3]
  화자A: [+0.9, -0.1]                   [+0.9, +0.2]
  화자B: [-1.0, +0.3]                   [-0.8, +0.1]
```

C = z_t_text.T @ z_t_spk / (B-1) 계산:

```
C[0,0] = (+1.1×+1.0 + +0.9×+0.9 + -1.0×-0.8) / 2 = (1.1+0.81+0.8)/2 ≈ +1.36  ← text dim0 ↔ spk dim0 강한 상관
C[0,1] = (+1.1×-0.3 + +0.9×+0.2 + -1.0×+0.1) / 2 = (-0.33+0.18-0.1)/2 ≈ -0.12
C[1,0] = (+0.2×+1.0 + -0.1×+0.9 + +0.3×-0.8) / 2 = (0.2-0.09-0.24)/2 ≈ -0.07
C[1,1] = (+0.2×-0.3 + -0.1×+0.2 + +0.3×+0.1) / 2 = (-0.06-0.02+0.03)/2 ≈ -0.02

     → C ≈ [[+1.36, -0.12],
             [-0.07, -0.02]]
```

`C[0,0] ≈ 1.36` → "text의 0번 차원과 spk의 0번 차원이 화자에 따라 같이 오르내린다" = 화자 정보가 text에 남아있다는 신호. loss = ‖C‖_F²/D는 이 값을 0으로 당긴다.

**논문 근거**: Barlow Twins (Zbontar et al., ICML 2021)은 같은 이미지의 두 augmented view 사이의 cross-correlation 행렬을 단위행렬로 만들어 표현의 중복성을 제거했다. VICReg (Bardes et al., ICLR 2022)는 동일한 아이디어를 covariance regularization으로 정식화했다. E23은 이 원리를 "동일 발화의 두 augmented view" 대신 "동일 발화의 text/speaker 표현 쌍"에 적용했다 — 두 표현이 서로 독립적이 되도록 D×D 제약을 부여.

### 핵심 메소드
```
zt = z_t_text.mean(dim=2)               # (B, D) utterance-level pool (시간축 평균)
zs = z_t_spk.mean(dim=2)
zt = (zt - zt.mean(0)) / (zt.std(0)+ε)  # batch 정규화 (스케일 통일)
zs = (zs - zs.mean(0)) / (zs.std(0)+ε)
C  = zt.T @ zs / (B-1)                  # (D, D) cross-covariance 행렬
loss = ‖C‖_F² / D                        # 모든 차원 쌍의 상관관계 합
```

### 바뀐 파이프라인
```
기존: orth_loss = (z_t_text · z_t_spk)².mean()           ← 1개 scalar 제약
E23: cross_cov_loss = ‖C(pool(z_t_text), pool(z_t_spk))‖_F² / D   ← D×D 행렬 제약
```
E10c 대비: `orth_weight=0.0`, `cross_cov_weight=1.0` 추가. GRL/SpkCls 유지.

**결과**: 실패. fm_pre 46배 폭등 (0.15 → 7.12), orth_epoch 400,000배 폭발. Utterance-level pooling이 frame-level disentanglement 신호를 잃어 orth 구조 붕괴.

---

## E24 — Orthogonal Reconstruction (Gram-Schmidt in recon path)

### 인용 논문
| 논문 | 인용한 부분 |
|------|------------|
| Ravfogel et al., ACL 2020 *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (INLP)* | 보호 속성(speaker) 방향을 null-space projection으로 제거하는 원리. INLP는 post-hoc iterative 적용이지만, E24는 학습 중 forward pass에서 동적으로 적용. |

### Gram-Schmidt Projection (Orthogonal Reconstruction)이란?
기존 파이프라인에서 `recon = dec(z_t_text + z_t_spk)`의 recon loss는 z_t_text에 화자 정보를 담도록 gradient를 흘릴 수 있다 — 화자 정보를 z_t_text에도 넣어두면 recon이 더 쉽기 때문이다.

Gram-Schmidt projection은 벡터에서 특정 방향 성분을 수학적으로 제거한다:
- `z_t_spk` 방향의 단위벡터 `û_spk`를 구한다
- `z_t_text`에서 `û_spk` 방향 성분(projection)을 뺀다
- 결과 `z_t_text_orth`는 `z_t_spk`와 **수직(직교)** 이 보장됨

이를 recon path에만 적용하면:
- **recon gradient**: `z_t_text_orth`를 통해 역전파 → speaker 방향으로 절대 밀 수 없음 (수학적 보장)
- **KD target**: `z_t_text.detach()`는 projection 이전 원본 사용 → 분포 변화 없음

### 핵심 메소드
```
û_spk          = z_t_spk / (‖z_t_spk‖ + ε)              # speaker 방향 단위벡터
proj           = (z_t_text · û_spk) · û_spk               # z_t_text의 speaker 방향 성분
z_t_text_orth  = z_t_text - proj                           # speaker 성분 제거 → z_t_spk와 직교

# gradient 보장:
# ∂L_recon/∂z_t_text = (I - û_spk û_spk^T) · ∂L/∂z_t_text_orth
# → speaker 방향 성분은 항상 0으로 소거되어 역전파
```

### 바뀐 파이프라인
```
기존: recon = dec(z_t_text + z_t_spk)

E24: z_t_text_orth = z_t_text - (z_t_text·û_spk)·û_spk   ← 추가: speaker 방향 제거
     recon = dec(z_t_text_orth + z_t_spk)                 ← 변경: 직교화된 벡터 사용
     KD: FM(z_s_text → z_t_text_d=z_t_text.detach())      ← 변경 없음: 원본 보존
```
E10c 대비: `--orth_recon True` 1개 플래그 추가. 나머지 동일.

**결과**: dev.clean 11.04% (E10c 10.84% 대비 소폭 악화). fm_pre **10배 감소** (0.153 → 0.014) — 강한 긍정 신호. train_loss 개선 (314 → 299).

---

## E25 — Cosine Orth

### 인용 논문
| 논문 | 인용한 부분 |
|------|------------|
| Wang et al., NeurIPS 2020 *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere* | unit hypersphere 위에서 각도 거리가 표현 학습의 자연스러운 척도임을 이론적으로 정립 — 정규화 후 cosine이 scale-invariant 제약임을 뒷받침 |
| Schroff et al., CVPR 2015 *FaceNet* | L2 정규화 임베딩에서 cosine/angular distance가 metric learning의 핵심 거리 척도 |

### Cosine Orth란?
기존 Orth loss는 실제로 다음과 같이 분해된다:
```
(z_t_text · z_t_spk)² = ‖z_t_text‖² · ‖z_t_spk‖² · cos²(θ)
```
즉 **벡터 크기와 각도의 곱**이다. 모델이 이 loss를 줄이는 방법은 두 가지:
1. 실제로 두 벡터의 방향을 수직으로 만든다 (cos²(θ) → 0) — 우리가 원하는 것
2. 벡터 크기를 줄인다 (‖z‖ → 0) — **크기 우회(shortcut)**

크기를 줄이면 orth loss가 0에 가까워지지만 실제로 speaker 정보가 z_t_text에서 빠지지 않는다. 모델이 이 shortcut을 학습하면 disentanglement가 무의미해진다.

Cosine Orth는 두 벡터를 먼저 **L2 정규화(unit vector)**로 만든 뒤 내적을 계산한다:
```
cos²(θ) = (zt_n · zs_n)²    where zt_n, zs_n은 unit vector
```
이 경우 크기는 항상 1이므로 오직 **방향(각도)** 만이 loss에 영향을 준다. 크기 우회 불가능.

### 핵심 메소드
```
기존: orth_loss = (z_t_text · z_t_spk).sum(dim=1).pow(2).mean()
                = ‖z_t_text‖·‖z_t_spk‖·cos²(θ)   ← 크기 우회 가능

E25: zt_n = z_t_text / (‖z_t_text‖ + ε)            ← L2 정규화 → unit vector
     zs_n = z_t_spk  / (‖z_t_spk‖  + ε)
     orth_loss = (zt_n · zs_n).sum(dim=1).pow(2).mean()
              = cos²(θ)                              ← 방향만 제어, 크기 무관
```
**스케일 보정**: cosine orth loss ∈ [0,1]이므로 raw orth (~5.47)와 gradient 규모를 맞추려면 `orth_weight=100` 필요 (raw orth ≈ D × cosine orth, D=96).

### 바뀐 파이프라인
```
E24: orth_recon=True (recon gradient 차단)
   + z_t_text_orth = z_t_text - (z_t_text·û_spk)·û_spk
   + recon = dec(z_t_text_orth + z_t_spk)

E25 추가:
   + zt_n = normalize(z_t_text)    ← orth 계산 전 L2 정규화
   + zs_n = normalize(z_t_spk)
   + orth_loss = (zt_n · zs_n)².mean() * orth_weight(100)
```
E24 대비: `--cosine_orth True`, `--orth_weight 100.0` 변경. 나머지 동일.

**결과**: 실행 중.
