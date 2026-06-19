# DAG-KD 인수인계 정리 (2026-06-10)

## 1. 한 줄 요약

DAG-KD는 ASR teacher representation을 그대로 모방하지 않고, text/content와 speaker/prosody/non-content 정보를 분리한 뒤 작은 student에 유효한 정보만 전달하려는 Knowledge Distillation 프로젝트이다.

현재 결론은 "주제가 틀렸다"가 아니라, 효과가 나는 조건과 나지 않는 조건이 꽤 선명해진 상태다.

- Conformer/GRP-KD 계열에서는 teacher latent를 text/speaker subspace로 나누고 GRL을 더한 설정이 가장 유망했다.
- wav2vec2 계열, 특히 SSL-pretrained student와 960h 학습에서는 CTC fine-tuning baseline이 강해서 GRP-KD가 오히려 성능을 해쳤다.
- hard orthogonal disentanglement는 wav2vec2에서 반복적으로 손해를 냈고, soft/MI/GRL 기반 제약 또는 약한 student 조건에서 다시 봐야 한다.

## 2. 프로젝트 목표와 핵심 아이디어

### 문제의식

일반적인 KD는 teacher hidden state나 logit을 통째로 student가 따라 하게 만든다. 하지만 작은 ASR student는 capacity가 제한적이므로, teacher representation에 포함된 speaker, prosody, channel, style 같은 non-linguistic 정보를 그대로 따라 하는 것이 오히려 낭비가 될 수 있다.

### DAG-KD의 주장

ASR representation 안의 정보를 분리하고, ASR 인식에 직접적으로 유효한 content/text 중심 정보만 student에게 전달하면 더 효율적인 student를 만들 수 있다.

### 구현된 주요 모듈

- `FlowMatchingModule`: student feature를 teacher feature 분포로 정렬하는 rectified-flow 계열 KD.
- `DiffKDModule`: diffusion/denoising 기반 feature KD.
- `ARClubGaussian`, `ClubGaussian`: CLUB 기반 mutual information upper bound 추정.
- `GlobalProsodyReferenceEncoder`, `GlobalStyleTokenLayer`: prosody/style representation.
- `SpeakerClassifier`, `GradientReversalLayer`: speaker 정보 분리/제거를 위한 classifier 및 GRL.
- `GRPKDModule`: multi-layer latent autoencoder + flow matching + diffusion 기반 KD.
- `eval_spk_probe.py`: z_t_text/z_t_spk에 남은 speaker 정보를 linear probe로 측정.

## 3. 코드 맵

### Conformer/NeMo 라인

- `train.py`: 기본 DAG-KD 학습 스크립트.
- `models.py`: Conformer 기반 `DistilDAGKDCTCModelBPE`, Flow/DiffKD/MI/GRL/Probe 주요 구현.
- `train_grp_kd.py`: GRP-KD 기반 latent disentanglement 실험 라인.
- `eval_spk_probe.py`: GRP-KD 체크포인트에서 speaker probe 평가.
- `librispeech_asr.py`, `gigaspeech_asr.py`: HF dataset loader.
- `utils.py`: manifest, speaker mapping, WER, phys cache, logging utility.

### wav2vec2 라인

- `train_wav2vec.py`: HuggingFace wav2vec2 기반 학습 스크립트.
- `models_wav2vec.py`: wav2vec2용 DAG-KD/GRP-KD/S-DisKD 구현.
- `scripts/train/wav2vec/`: wav2vec2 실험 실행 스크립트.
- `experiments/wav2vec_*`: wav2vec2 실험 설계 및 결과 리포트.

### 실험 리포트 위치

- `experiments/MI_ablation/report.md`
- `experiments/cyclic_disen/report.md`
- `experiments/layerwise_spk_grl/report.md`
- `experiments/grp_kd_based/report.md`
- `experiments/s-diskd/report.md`
- `experiments/base2small/report.md`
- `experiments/wav2vec_grp_kd_orth/report.md`
- `experiments/wav2vec_grp_kd_960h/report.md`

## 4. 실험별 핵심 결과

### 4.1 MI Ablation

위치: `experiments/MI_ablation/report.md`

목적: DAG-KD의 MI 최소화 구성요소 중 어떤 pair가 실제로 도움이 되는지 확인.

공통 조건:

- LibriSpeech train-clean-100.
- Teacher: `stt_en_conformer_ctc_small`.
- CTC + LogitKD + Flow + DiffKD + Speaker CE.
- spk layer 4, txt layer 16.

핵심 결과:

- `txt-speaker(ts)` MI 단독이 가장 안정적이었다. test WER 0.3223으로 baseline 0.3242보다 소폭 개선.
- `txt-prosody(tp)`, `prosody-speaker(ps)`를 추가할수록 성능이 악화했다.
- Phys loss(F0/Energy/VUV)는 prosody 표현을 과하게 구속해 성능에 부정적이었다.
- Text reconstruction은 `ts` 단독에서는 유효하지만, prosody 관련 MI가 들어가면 충돌하는 경향이 있었다.

해석:

- "모든 정보를 다 분리할수록 좋다"는 가설은 기각.
- ASR에는 speaker와 text의 분리만 조심스럽게 쓰는 편이 낫고, prosody를 독립적으로 밀어내는 것은 위험하다.

### 4.2 Cyclic Disentanglement

위치: `experiments/cyclic_disen/report.md`

목적: CLUB MI 대신 cyclic adversarial 구조로 표현 분리를 더 안정적으로 만들 수 있는지 확인.

핵심 결과:

- `cyclic ts`: test_clean 14.35%, test_other 32.60%로 baseline보다 악화.
- `cyclic tp`: test_clean 13.96%, test_other 32.46%로 baseline 14.08%/32.56% 대비 소폭 개선.

해석:

- cyclic adversarial은 `ts` 분리에는 충분하지 않았다.
- 다만 `tp`에서는 직접 MI 최소화보다 간접 cyclic 구조가 덜 해로운 가능성을 보였다.

후속 후보:

- `CLUB ts + cyclic tp` 혼합.
- cyclic weight 및 GRL alpha sweep.

### 4.3 Layerwise Speaker GRL

위치: `experiments/layerwise_spk_grl/report.md`

목적: teacher layer feature에서 speaker 정보를 제거한 뒤, 그 spk-free feature를 student KD target으로 쓰면 성능이 개선되는지 확인.

핵심 결과:

- Layer KD baseline: test_clean 13.94%, test_other 31.90%로 가장 좋았다.
- Teacher feature를 enc_i로 가공해 speaker를 제거한 E2/E3/E4는 baseline을 넘지 못했다.
- E3는 speaker 제거와 student KD loss 수렴에는 성공했지만, WER는 test_other 32.73%로 나빠졌다.

해석:

- teacher feature를 인위적으로 바꿔 KD target으로 쓰면 정보 손실이 크다.
- 올바른 방향은 teacher feature 자체를 변형하는 것이 아니라, student-side disentanglement가 teacher의 계층 구조를 받아들이게 하는 방식이다.

### 4.4 GRP-KD Based Latent Disentanglement

위치: `experiments/grp_kd_based/report.md`

목적: 기존 DAG-KD의 약한 KD 신호 문제를 피하기 위해 GRP-KD latent space 위에서 text/speaker 분리를 다시 검증.

핵심 결과:

| Exp | 방법 | dev_clean | dev_other | test_clean | test_other |
| --- | --- | ---: | ---: | ---: | ---: |
| E1 | GRP-KD baseline | 12.0 | 28.3 | 12.4 | 28.8 |
| E2 | E1 + Orth + SpkCls | 11.0 | 28.8 | 11.5 | 29.5 |
| E3 | E1 + CLUB MI + SpkCls | 13.1 | 31.0 | 13.3 | 32.0 |
| E4 | E2 + GRL on z_t_text | 11.0 | 28.3 | 11.5 | 28.7 |

가장 중요한 결론:

- E2는 clean split을 크게 개선했지만 other split을 해쳤다.
- E4는 clean 개선을 유지하면서 other split도 baseline 수준으로 회복했다.
- 현재 Conformer 라인에서 가장 논문화 가능성이 있는 결과는 E4다.

Speaker probe:

| 지표 | E2 Orth | E3 CLUB MI | E4 Orth+GRL | Random |
| --- | ---: | ---: | ---: | ---: |
| z_t_spk speaker acc | 88.91% | 90.29% | 88.63% | 0.40% |
| z_t_text speaker acc | 14.17% | 1.51% | 3.56% | 0.40% |

해석:

- E3는 z_t_text speaker acc가 가장 낮지만 WER가 가장 나쁘다. 즉, "speaker 정보가 적다"만으로 좋은 representation이 아니다.
- 좋은 분리는 speaker 정보 제거와 ASR 정보 보존이 동시에 되어야 한다.
- E4가 현재 가장 균형이 좋다.

### 4.5 S-DisKD

위치: `experiments/s-diskd/report.md`

목적: teacher-side factor를 student 중간 레이어와 직접 정렬하는 방식 검증.

핵심 결과:

| Exp | 방법 | test_clean | test_other |
| --- | --- | ---: | ---: |
| E1 | baseline | 14.08 | 32.56 |
| E2 | student text KD | 14.34 | 32.82 |
| E3 | student speaker KD | 13.55 | 31.23 |
| E4 | text + speaker KD | 13.75 | 31.25 |
| E5 | full + student CLUB | 13.77 | 31.66 |

해석:

- speaker factor KD만 적용한 E3가 최고.
- text factor KD는 CTC와 충돌해 성능을 해치는 경향.
- student-side CLUB 추가 이득은 없었다.

후속 후보:

- speaker factor KD의 layer/weight/loss sweep.
- text factor는 직접 MSE보다 CTC-compatible한 방식으로 재설계.

### 4.6 wav2vec2 base2small

위치: `experiments/base2small/report.md`

목적: wav2vec2-base-960h teacher에서 half-size random-init student로 KD 방식별 효과 비교.

핵심 결과:

| Exp | 방법 | test_clean | test_other |
| --- | --- | ---: | ---: |
| E-A | CTC only | 25.32 | 55.96 |
| E-B | Logit KD | 24.77 | 55.31 |
| E-C | GRP-KD | 22.93 | 53.56 |
| E-D | DAG-KD | 27.33 | 58.53 |

해석:

- random-init half-size student에서는 GRP-KD가 유의미하게 개선.
- DAG-KD disentanglement는 random-init student에 너무 불안정했다.
- weak/random student 조건에서는 generative KD가 도움이 될 가능성이 있다.

### 4.7 wav2vec2 GRP-KD + Orthogonal Disentanglement

위치: `experiments/wav2vec_grp_kd_orth/report.md`

목적: Conformer GRP-KD에서 유효했던 orthogonal disentanglement가 wav2vec2에서도 재현되는지 확인.

Track A: large -> base

| 방법 | test_clean | test_other |
| --- | ---: | ---: |
| A-E1 GRP-KD baseline | 14.92 | 40.36 |
| A-E2 + Orth + SpkCls | 15.45 | 42.05 |

Track B: base -> base-half

| 방법 | test_clean | test_other |
| --- | ---: | ---: |
| B-E1 GRP-KD baseline | 18.52 | 46.32 |
| B-E2 + Orth + SpkCls | 20.28 | 48.26 |

해석:

- 8개 split 모두에서 orth + speaker classifier가 baseline보다 악화.
- 작은 student일수록 손해가 커졌다.
- Conformer에서 작동한 hard orthogonal disentanglement가 wav2vec2 latent space에는 맞지 않는다.
- wav2vec2는 SSL/CTC 표현 구조가 이미 text 중심이거나 sparse해서 추가 hard 분리가 over-regularization일 가능성이 크다.

### 4.8 wav2vec2 GRP-KD on LibriSpeech 960h (SSL Student)

위치: `experiments/wav2vec_grp_kd_960h/report.md`

목적: reviewer pushback을 피하기 위해 100h random-init의 14% WER 영역이 아니라, SSL-pretrained student + 960h로 single-digit WER 영역에서 KD 효과를 검증.

설계:

- Student: `facebook/wav2vec2-base` SSL-only, CTC head random init.
- Teacher: `facebook/wav2vec2-large-960h`.
- Data: LibriSpeech 960h.
- A-E0: CTC only.
- A-E1: CTC + logit KD + GRP-KD.
- A-E2: A-E1 + orth + speaker CE.

완료 결과:

| Split | A-E0 CTC | A-E1 GRP-KD | A-E2 GRP-KD+Orth |
| --- | ---: | ---: | ---: |
| dev_clean | 4.63 | 5.68 | 5.31 |
| dev_other | 12.55 | 13.81 | 13.22 |
| test_clean | 4.82 | 5.84 | 5.30 |
| test_other | 13.50 | 14.81 | 13.81 |

해석:

- A-E0가 test_clean 4.82%로 single-digit 목표는 달성했다.
- 하지만 GRP-KD는 strong SSL student 위에서 모든 split을 악화했다.
- Orth + speaker CE는 A-E1의 손상을 일부 회복했지만, CTC baseline보다 좋지는 않다.
- 결론: strong SSL-pretrained student에는 teacher-side GRP constraint 자체가 과도하게 작동한다.

## 5. 최근 코드 변경 상태

현재 git diff 기준 수정된 파일:

- `experiments/wav2vec_grp_kd_960h/report.md`
- `train_wav2vec.py`
- `models_wav2vec.py`
- `scripts/train/wav2vec/grp_kd_960h/A_E0_no_kd.sh`
- `scripts/train/wav2vec/grp_kd_960h/A_E1_kd.sh`
- `scripts/train/wav2vec/grp_kd_960h/A_E2_kd_orth.sh`

변경 내용:

- 960h 실험 결과를 리포트에 추가.
- `train_wav2vec.py`에 `--accumulate_grad_batches` 추가.
- A-E1/A-E2 OOM 대응: batch 16 -> 8, gradient accumulation 2로 effective batch 64 유지.
- speaker mapping cache를 먼저 로드해 960h speaker scan 비용 감소.
- `spk_idx < 0` 샘플을 `speaker_id_mapping.json`과 `spk_id`로 복구.
- GRP-KD speaker CE에서 invalid speaker id를 valid mask로 제외.
- `use_disent=False`일 때 phys cache 생성을 skip.
- 960h shell scripts는 `last.ckpt`가 있으면 resume하고, `train.log`는 append하도록 변경.

주의:

- shell script에 W&B key가 직접 들어가 있다. 공유/공개 전 제거하거나 환경변수 주입 방식으로 바꾸는 것이 좋다.
- 현재 working tree에는 untracked script/report 파일이 많다. 후배에게 넘기기 전 `git status --short`를 기준으로 "필수 파일"과 "폐기 가능한 파일"을 한 번 정리하는 것이 좋다.

## 6. 후배가 이어받을 때 추천 우선순위

### 1순위: Conformer GRP-KD E4 재현 및 강화

가장 유망한 출발점은 `experiments/grp_kd_based/report.md`의 E4다.

해야 할 일:

- E1/E2/E4를 동일 환경에서 재현.
- 최소 3 seed로 WER 평균/표준편차 확보.
- E4의 GRL alpha, orth weight sweep.
- E5 alpha annealing 결과 확인 또는 재실행.
- linear probe뿐 아니라 CKA, layer-wise feature similarity, loss curve를 같이 보고 "분리 수치와 ASR 성능의 관계"를 정리.

논문화 포인트:

- speaker 정보 제거 자체가 목적이 아니라, ASR 정보 보존과 speaker leakage 감소의 균형이 중요하다는 메시지.
- CLUB MI는 분리를 강하게 만들 수 있지만 representation collapse 위험이 있다는 negative evidence.

### 2순위: S-DisKD의 speaker factor KD 정리

`experiments/s-diskd/report.md`의 E3는 간단하고 설득력 있는 결과다.

해야 할 일:

- speaker factor KD layer sweep: layer 2, 4, 6, 8.
- weight sweep: 0.1, 0.5, 1.0, 2.0.
- cosine/contrastive loss와 MSE 비교.
- text factor KD는 직접 MSE 대신 CTC-compatible target으로 재설계.

### 3순위: wav2vec2는 "hard disentanglement 실패 조건"으로 정리

wav2vec2 결과는 논문 메인 성공 스토리보다는 boundary condition으로 쓰는 것이 좋다.

핵심 메시지:

- weak/random student에서는 GRP-KD가 도움이 될 수 있다.
- strong SSL-pretrained student에서는 CTC fine-tuning baseline이 강하고, teacher-side GRP constraint가 오히려 over-regularization이 된다.
- hard orthogonal disentanglement는 wav2vec2 latent에서는 일관되게 성능을 해친다.

후속 후보:

- hard orth 대신 soft GRL/MI를 약하게 적용.
- `grp_rec_weight`, `grp_gen_weight`를 낮춰 CTC 주도 학습을 유지.
- SSL teacher vs ASR fine-tuned teacher 비교.
- small student 조건에서만 KD를 다시 검증.

## 7. SPL 투고 관점 판단

현재 상태만으로 SPL에 바로 넣기에는 리스크가 있다.

이유:

- 가장 최신의 960h/wav2vec2 실험에서는 제안 KD가 CTC baseline보다 낮다.
- Conformer 쪽 positive result는 유망하지만 seed 반복과 baseline 정리가 더 필요하다.
- 실험 라인이 많아졌기 때문에 논문 메시지를 하나로 압축해야 한다.

그래도 버릴 주제는 아니다. 오히려 후속자가 가져가기에 좋은 이유:

- 구현이 이미 넓게 되어 있다.
- negative result가 많아서 잘못된 방향을 피할 수 있다.
- Conformer GRP-KD E4와 S-DisKD speaker KD라는 positive signal이 있다.
- "언제 disentanglement KD가 도움이 되고, 언제 해로운가"라는 더 성숙한 연구 질문으로 바꿀 수 있다.

## 8. 추천 논문 메시지 후보

### 후보 A: 긍정 결과 중심

"ASR teacher representation에서 speaker-related latent를 명시적으로 분리하고, text-preserving adversarial constraint를 함께 쓰면 small Conformer student의 KD 성능을 개선할 수 있다."

필요한 추가 실험:

- Conformer E1/E2/E4 multi-seed.
- logit KD/layer KD/GRP-KD baseline 정리.
- speaker probe와 WER correlation 분석.

### 후보 B: 조건부 효과 중심

"Disentanglement-aware KD is beneficial only when the KD signal and representation space are compatible; hard disentanglement can harm strong SSL-pretrained ASR students."

필요한 추가 실험:

- Conformer positive result와 wav2vec2 negative result를 같은 프레임으로 비교.
- student strength/random-init/SSL-pretrained에 따른 효과 분석.
- hard orth vs soft GRL/MI 비교.

### 후보 C: 후배용 안전한 석사/학부 연구 확장

"Speaker factor alignment improves ASR student training, while direct text factor alignment conflicts with CTC."

필요한 추가 실험:

- S-DisKD E3 중심으로 layer/weight/loss sweep.
- 구현 부담이 상대적으로 낮고 결과 해석이 쉽다.

## 9. 지도교수님께 말할 때 추천 표현

핵심은 "그만두겠다"가 아니라 "제가 만든 기반을 후속자가 더 잘 살릴 수 있게 넘기고, 저는 현재 우선순위가 더 높은 연구에 집중하겠다"로 말하는 것이다.

추천 문장:

> DAG-KD를 버리려는 것은 아닙니다. 지금까지 구현과 실험을 통해 어떤 조건에서 효과가 있고 어떤 조건에서 깨지는지가 꽤 정리됐습니다. 다만 SPL로 바로 밀기에는 최신 960h/wav2vec2 결과가 CTC baseline을 넘지 못해서, 제가 계속 붙잡기보다는 후속자가 Conformer GRP-KD와 S-DisKD 쪽 positive signal을 중심으로 재현/seed/ablation을 쌓는 편이 더 효율적이라고 판단했습니다.

조금 더 단단한 버전:

> 이 주제는 실패한 게 아니라 연구 질문이 바뀐 상태라고 봅니다. 처음에는 "disentanglement KD가 항상 student를 개선하는가"였지만, 지금 결과는 "어떤 student/teacher representation 조건에서 disentanglement KD가 도움이 되고, 언제 over-regularization이 되는가"로 좁혀졌습니다. 저는 여기까지의 코드, 리포트, 실행 스크립트, negative result를 정리해서 후배가 바로 이어갈 수 있게 넘기고, 제 시간은 현재 더 논문화 가능성이 높은 다른 연구에 집중하고 싶습니다.

짧은 회의용 버전:

> DAG-KD는 포기라기보다 이관이 맞는 것 같습니다. 제가 만든 실험 기반과 결과는 충분히 후속 연구 가치가 있지만, SPL 단기 투고로는 추가 seed와 방향 정리가 필요합니다. 저는 현재 다른 연구에 집중하고, DAG-KD는 후배가 Conformer positive line을 중심으로 이어가면 좋겠습니다.

## 10. 후배에게 말할 때 추천 표현

추천 문장:

> 이 프로젝트는 처음부터 다시 시작할 필요는 없어. 내가 해본 결과, 무작정 MI나 orthogonal constraint를 넣으면 안 되고, Conformer GRP-KD E4와 S-DisKD speaker KD가 가장 좋은 출발점이야. wav2vec2 960h는 CTC baseline이 너무 강해서 KD가 오히려 손해였고, 그건 실패라기보다 피해야 할 조건을 확인한 결과라고 보면 돼.

이어받을 때 첫 작업:

1. `experiments/grp_kd_based/report.md`와 `experiments/s-diskd/report.md` 먼저 읽기.
2. `train_grp_kd.py`, `models.py`, `eval_spk_probe.py` 확인.
3. E1/E2/E4 스크립트가 현재 환경에서 재현되는지 dry-run.
4. seed 반복과 weight sweep 계획 세우기.
5. wav2vec2 라인은 나중에 boundary condition/negative result로 정리.

## 11. 내 입장 정리 문장

개인적으로 말할 때는 이렇게 정리하면 좋다.

> DAG-KD를 접는 것이 아니라, 제가 할 수 있는 탐색 단계는 충분히 했고 이제는 재현, seed, ablation을 꾸준히 쌓아야 하는 단계라고 봅니다. 그 작업은 후배가 이어받기에 좋고, 저는 지금 제 연구 포트폴리오에서 더 집중해야 할 주제로 시간을 옮기고 싶습니다.

더 부드러운 버전:

> 이 주제가 아깝지 않아서 넘기는 겁니다. 그냥 멈추면 사라지지만, 지금처럼 코드와 결과, 실패한 방향까지 정리해서 넘기면 후배가 훨씬 빠르게 논문화 가능한 형태로 가져갈 수 있습니다.

## 12. 최종 판단

DAG-KD는 "폐기"가 아니라 "책임 있는 인수인계"가 맞다.

현재 가장 좋은 판단:

- SPL 단기 투고는 보류.
- 후배에게는 Conformer GRP-KD E4와 S-DisKD speaker KD를 중심으로 넘긴다.
- wav2vec2 960h 결과는 "strong SSL student에서는 KD 제약이 과해질 수 있다"는 중요한 negative/boundary result로 보존한다.
- 본인은 다른 연구에 집중하되, DAG-KD는 필요하면 방향성/리뷰 정도로 계속 도와주는 형태가 좋다.
