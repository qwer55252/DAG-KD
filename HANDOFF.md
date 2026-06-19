# DAG-KD 인수인계 문서

작성일: 2026-06-19

## 0. 먼저 읽을 결론

DAG-KD는 폐기할 주제가 아니라, 탐색 단계가 끝나고 후속자가 정리/재현/ablation을 쌓아야 하는 단계에 온 주제다.

핵심 결론은 세 가지다.

1. Conformer/NeMo 기반에서는 `GRP-KD + latent disentanglement + GRL` 라인이 가장 유망하다.
2. `speaker factor KD`는 실제 WER 개선 신호가 있고, `text factor KD`는 CTC와 충돌해 성능을 해치는 경향이 있다.
3. wav2vec2/SSL-pretrained student/960h 조건에서는 CTC fine-tuning baseline이 강해서 GRP-KD가 오히려 over-regularization으로 작동했다.

즉, 연구 질문은 처음의 “disentanglement KD가 항상 좋은가?”에서 다음으로 바뀌었다.

> 어떤 teacher/student representation 조건에서 disentanglement-aware KD가 도움이 되고, 언제 over-regularization이 되는가?

SPL 단기 투고로 바로 밀기에는 아직 위험하다. 하지만 후속자가 Conformer positive line을 중심으로 seed 반복과 ablation을 쌓으면 논문화 가능성은 남아 있다.

## 1. 연구 목표

일반적인 ASR Knowledge Distillation은 teacher의 hidden representation이나 logit을 통째로 student가 따라 하게 만든다. DAG-KD의 문제의식은 다음과 같다.

- Teacher representation에는 text/content 정보뿐 아니라 speaker, prosody, channel, recording condition 같은 non-content 정보도 섞여 있다.
- 작은 student는 capacity가 제한적이므로, teacher의 모든 정보를 무차별적으로 모방하면 ASR에 덜 중요한 정보를 재현하느라 capacity를 낭비할 수 있다.
- 따라서 teacher/student representation을 text/content와 speaker/prosody/non-content로 분리하고, ASR에 유효한 정보만 distill하는 것이 목표다.

한 문장으로 정리하면:

> ASR teacher representation에서 학습에 필요한 정보와 방해되는 정보를 구분하고, 작은 student가 필요한 정보만 효율적으로 받도록 만드는 KD 프레임워크.

## 2. 저장소 구조

### 메인 코드

| 파일 | 역할 |
| --- | --- |
| `train.py` | Conformer/NeMo 기반 DAG-KD 메인 학습 스크립트 |
| `models.py` | Conformer용 `DistilDAGKDCTCModelBPE`, Flow/DiffKD/MI/GRL/Probe 구현 |
| `train_grp_kd.py` | Conformer GRP-KD latent disentanglement 실험 라인 |
| `eval_spk_probe.py` | 체크포인트에서 z_t_text/z_t_spk speaker leakage probe |
| `train_wav2vec.py` | HuggingFace wav2vec2 기반 학습 스크립트 |
| `models_wav2vec.py` | wav2vec2용 DAG-KD/GRP-KD/S-DisKD 구현 |
| `utils.py` | manifest, speaker mapping, WER, phys cache, source snapshot 등 |
| `inference.py` | 평가/추론 |

### 주요 실험 리포트

| 리포트 | 내용 |
| --- | --- |
| `experiments/MI_ablation/report.md` | MI pair, Phys loss, reconstruction ablation |
| `experiments/cyclic_disen/report.md` | cyclic adversarial disentanglement |
| `experiments/layerwise_spk_grl/report.md` | teacher layer feature에서 speaker 제거 후 KD |
| `experiments/grp_kd_based/report.md` | Conformer GRP-KD latent disentanglement 핵심 positive line |
| `experiments/s-diskd/report.md` | student-side factor KD, speaker KD positive |
| `experiments/base2small/report.md` | wav2vec2 base -> half random-init student |
| `experiments/wav2vec_grp_kd_orth/report.md` | wav2vec2 hard orthogonal disentanglement negative result |
| `experiments/wav2vec_grp_kd_960h/report.md` | wav2vec2 SSL student + LibriSpeech 960h final validation |
| `experiments/DAG_KD_HANDOFF_2026-06-10.md` | 더 상세한 이전 인수인계 초안 |

### 실행 스크립트

- Conformer 계열: `scripts/train/*.sh`, `experiments/grp_kd_based/*.sh`, `scripts/train/s-diskd/*.sh`
- wav2vec2 계열: `scripts/train/wav2vec/**`
- 960h wav2vec2 최종 실험: `scripts/train/wav2vec/grp_kd_960h/*.sh`

주의: 일부 shell script에 W&B API key가 직접 들어 있다. 외부 공유 전 반드시 제거하고 환경변수 방식으로 바꾸는 것이 좋다.

## 3. 핵심 모듈 개념

| 모듈 | 의미 |
| --- | --- |
| Flow Matching KD | student feature를 teacher feature/latent로 부드럽게 이동시키는 generative KD |
| DiffKD | diffusion/denoising 방식으로 teacher feature 구조를 전달 |
| GRP-KD | teacher/student를 96-dim latent로 압축한 뒤 reconstruction + flow + diffusion으로 정렬 |
| CLUB MI | text/speaker/prosody 표현 간 mutual information upper bound를 낮추는 방식 |
| Orthogonal Disentanglement | z_text와 z_spk의 내적을 줄여 subspace 분리 |
| GRL | text latent에서 speaker 정보를 제거하기 위한 adversarial classifier |
| Speaker Probe | 분리된 latent에 speaker 정보가 얼마나 남았는지 측정 |
| S-DisKD | teacher factor를 student layer/factor에 직접 정렬하는 실험 라인 |

중요한 교훈:

- speaker 정보가 적다고 항상 좋은 text representation은 아니다.
- CLUB MI는 speaker leakage를 강하게 낮출 수 있지만 representation collapse를 만들 수 있다.
- hard orthogonal constraint는 Conformer에서는 일부 유효했지만 wav2vec2에서는 일관되게 손해였다.

## 4. 실험별 정리

### 4.1 MI Ablation

리포트: `experiments/MI_ablation/report.md`

목표:

- text/speaker/prosody MI pair 중 어떤 분리가 실제로 도움이 되는지 확인.

조건:

- LibriSpeech train-clean-100.
- Teacher: `stt_en_conformer_ctc_small`.
- CTC + Logit KD + Flow + DiffKD + SpkCE.
- spk layer 4, txt layer 16.

결과:

| 설정 | test WER |
| --- | ---: |
| baseline, MI 없음 | 0.3242 |
| MI ts | 0.3223 |
| MI tp | 0.3302 |
| MI ts,tp | 0.3271 |
| MI ts,tp,ps | 0.3303 |

결론:

- `txt-speaker(ts)` MI 단독만 소폭 유효.
- `txt-prosody(tp)`와 `prosody-speaker(ps)`는 대체로 역효과.
- Phys loss는 prosody를 F0/Energy/VUV에 과하게 묶어 ASR과 충돌하는 경향.
- Rec loss는 `ts` 단독에서는 유효하지만, prosody 관련 MI가 들어가면 불안정.

다음 사람이 할 일:

- `ts`만 유지하고 Phys를 제거한 조합을 추가 검증.
- MI loss scale과 CLUB 수렴 상태를 반드시 같이 확인.

### 4.2 Cyclic Disentanglement

리포트: `experiments/cyclic_disen/report.md`

목표:

- 직접 CLUB MI 대신 cyclic/adversarial 구조로 더 안정적인 분리를 만들 수 있는지 확인.

결과:

| 설정 | test_clean | test_other |
| --- | ---: | ---: |
| DAG-KD baseline | 14.08 | 32.56 |
| cyclic ts | 14.35 | 32.60 |
| cyclic tp | 13.96 | 32.46 |

결론:

- cyclic `ts`는 CLUB `ts`보다 약하고 baseline보다도 나쁘다.
- cyclic `tp`는 CLUB `tp`와 달리 소폭 개선이 있어, prosody 관련 분리는 직접 MI보다 간접 구조가 나을 수 있다.

다음 사람이 할 일:

- `CLUB ts + cyclic tp` 조합 검증.
- cyclic weight/GRL alpha sweep.

### 4.3 Layerwise Speaker GRL

리포트: `experiments/layerwise_spk_grl/report.md`

목표:

- teacher layer feature에서 speaker 정보를 제거한 spk-free target을 만들고, 이를 student layer KD target으로 쓰면 좋은지 확인.

결과:

| 설정 | test_clean | test_other | 해석 |
| --- | ---: | ---: | --- |
| E1 Layer KD baseline | 13.94 | 31.90 | 가장 좋음 |
| E3 normalized student KD | 13.81 | 32.73 | KD는 동작하지만 WER 악화 |
| E4 teacher-space KD | 13.74 | 32.18 | speaker 제거 실패 |

결론:

- teacher feature를 인위적으로 가공해 spk-free target으로 만드는 방식은 한계가 있다.
- teacher가 실제로 만든 적 없는 feature를 student가 따라 하게 되어 정보 손실/target mismatch가 생긴다.
- 방향은 teacher feature를 바꾸는 것이 아니라, student-side factor가 teacher의 계층 구조를 받아들이게 하는 쪽이 낫다.

### 4.4 GRP-KD Based Latent Disentanglement

리포트: `experiments/grp_kd_based/report.md`

이 프로젝트에서 가장 중요한 positive line이다.

목표:

- 기존 DAG-KD의 KD 신호가 약하다는 문제를 피하기 위해 GRP-KD latent space 위에서 text/speaker 분리를 검증.

결과:

| Exp | 방법 | dev_clean | dev_other | test_clean | test_other |
| --- | --- | ---: | ---: | ---: | ---: |
| E1 | GRP-KD baseline | 12.0 | 28.3 | 12.4 | 28.8 |
| E2 | E1 + Orth + SpkCls | 11.0 | 28.8 | 11.5 | 29.5 |
| E3 | E1 + CLUB MI + SpkCls | 13.1 | 31.0 | 13.3 | 32.0 |
| E4 | E2 + GRL on z_t_text | 11.0 | 28.3 | 11.5 | 28.7 |

Speaker probe:

| 지표 | E2 Orth | E3 CLUB MI | E4 Orth+GRL | Random |
| --- | ---: | ---: | ---: | ---: |
| z_t_spk speaker acc | 88.91 | 90.29 | 88.63 | 0.40 |
| z_t_text speaker acc | 14.17 | 1.51 | 3.56 | 0.40 |

결론:

- E2는 clean split을 크게 개선하지만 other split은 악화.
- E4는 clean 개선을 유지하면서 other split을 baseline 수준으로 회복.
- E3는 speaker leakage가 가장 낮지만 WER가 가장 나쁘다. 분리 자체보다 ASR 정보 보존이 더 중요하다.
- 현재 가장 유망한 논문 출발점은 E4다.

다음 사람이 할 일:

- E1/E2/E4를 같은 환경에서 재현.
- 최소 3 seed 평균/표준편차 확보.
- E4의 `grl_alpha`, `orth_weight`, `spk_cls_weight` sweep.
- E5 alpha annealing과 E6 student GRL branch 확인.
- speaker probe뿐 아니라 WER와 latent quality를 함께 분석.

### 4.5 S-DisKD

리포트: `experiments/s-diskd/report.md`

목표:

- teacher-side factor를 student 중간 레이어와 직접 정렬하면 성능이 좋아지는지 확인.

결과:

| Exp | 방법 | test_clean | test_other |
| --- | --- | ---: | ---: |
| E1 | baseline | 14.08 | 32.56 |
| E2 | student text factor KD | 14.34 | 32.82 |
| E3 | student speaker factor KD | 13.55 | 31.23 |
| E4 | text + speaker factor KD | 13.75 | 31.25 |
| E5 | full + student CLUB | 13.77 | 31.66 |

결론:

- speaker factor KD만 적용한 E3가 최고.
- text factor KD는 CTC와 충돌해 오히려 성능을 해친다.
- student-side CLUB 추가 효과는 거의 없다.

다음 사람이 할 일:

- speaker factor KD를 중심으로 layer sweep: 2, 4, 6, 8.
- weight sweep: 0.1, 0.5, 1.0, 2.0.
- MSE, cosine, contrastive loss 비교.
- text factor KD는 직접 정렬보다 CTC-compatible 방식으로 재설계.

### 4.6 wav2vec2 base2small

리포트: `experiments/base2small/report.md`

목표:

- wav2vec2-base-960h teacher에서 half-size random-init student로 KD 방식별 효과 비교.

결과:

| Exp | 방법 | test_clean | test_other |
| --- | --- | ---: | ---: |
| E-A | CTC only | 25.32 | 55.96 |
| E-B | Logit KD | 24.77 | 55.31 |
| E-C | GRP-KD | 22.93 | 53.56 |
| E-D | DAG-KD | 27.33 | 58.53 |

결론:

- random-init half-size student에서는 GRP-KD가 확실히 도움이 된다.
- DAG-KD disentanglement는 random-init student에는 너무 불안정하다.
- weak student 조건에서는 generative KD가 의미가 있다.

### 4.7 wav2vec2 GRP-KD + Orthogonal Disentanglement

리포트: `experiments/wav2vec_grp_kd_orth/report.md`

목표:

- Conformer GRP-KD에서 유효했던 orthogonal disentanglement가 wav2vec2에서도 재현되는지 확인.

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

결론:

- 8개 split 모두에서 orth + speaker classifier가 성능을 악화.
- 작은 student일수록 손해가 커진다.
- Conformer에서 작동한 hard orthogonal disentanglement를 wav2vec2에 그대로 옮기면 안 된다.
- wav2vec2 latent는 이미 text 중심이거나 SSL 구조상 hard 분리가 over-regularization이 되는 듯하다.

### 4.8 wav2vec2 960h SSL Student

리포트: `experiments/wav2vec_grp_kd_960h/report.md`

목표:

- 100h random-init의 높은 WER 영역이 아니라, SSL-pretrained student + LibriSpeech 960h의 single-digit WER 영역에서 KD 효과를 검증.

설계:

- Student: `facebook/wav2vec2-base` SSL-only, CTC head random init.
- Teacher: `facebook/wav2vec2-large-960h`.
- Data: LibriSpeech 960h.
- A-E0: CTC only.
- A-E1: CTC + logit KD + GRP-KD.
- A-E2: A-E1 + orth + speaker CE.

결과:

| Split | A-E0 CTC | A-E1 GRP-KD | A-E2 GRP-KD+Orth |
| --- | ---: | ---: | ---: |
| dev_clean | 4.63 | 5.68 | 5.31 |
| dev_other | 12.55 | 13.81 | 13.22 |
| test_clean | 4.82 | 5.84 | 5.30 |
| test_other | 13.50 | 14.81 | 13.81 |

결론:

- A-E0가 test_clean 4.82%로 single-digit 목표는 달성.
- 하지만 GRP-KD는 strong SSL student 위에서 모든 split을 악화.
- Orth + speaker CE는 A-E1 대비 회복 효과는 있지만 CTC baseline보다 좋지 않다.
- strong SSL-pretrained student에는 teacher-side GRP constraint 자체가 과한 제약이다.

후속자가 이 라인을 잡는다면:

- `grp_rec_weight`, `grp_gen_weight`를 크게 낮춰 CTC-dominant 학습 유지.
- SSL teacher vs ASR fine-tuned teacher 비교.
- hard orth 대신 매우 약한 GRL/MI만 적용.
- 하지만 논문 메인 라인보다는 boundary/negative result로 쓰는 편이 안전하다.

## 5. 현재 코드 상태

`git status --short` 기준으로 현재 작업 트리는 깨끗하지 않다.

수정된 tracked 파일:

- `experiments/wav2vec_grp_kd_960h/report.md`
- `models_wav2vec.py`
- `scripts/train/wav2vec/grp_kd_960h/A_E0_no_kd.sh`
- `scripts/train/wav2vec/grp_kd_960h/A_E1_kd.sh`
- `scripts/train/wav2vec/grp_kd_960h/A_E2_kd_orth.sh`
- `train_wav2vec.py`

중요한 최근 변경:

- `train_wav2vec.py`에 `--accumulate_grad_batches` 추가.
- A-E1/A-E2 OOM 대응으로 batch 16 -> 8, gradient accumulation 2 적용.
- speaker mapping cache를 먼저 로드하도록 수정해 960h speaker scan 비용 감소.
- `spk_idx < 0`이면 `speaker_id_mapping.json`과 `spk_id`로 복구 시도.
- GRP-KD speaker CE에서 invalid speaker id를 mask.
- `use_disent=False`면 phys cache 생성을 skip.
- 960h 스크립트는 `last.ckpt`가 있으면 resume하고 `train.log`에 append.

untracked 파일도 많다. 후속자에게 넘기기 전에는 다음 기준으로 정리하는 것이 좋다.

- 리포트/실험 스크립트/재현에 필요한 파일은 commit.
- 임시 파일, 중복 스크립트, 개인 설정 디렉토리는 별도 확인 후 제외.
- `.claude/`, `CLAUDE_tmp.md`, W&B key가 들어간 스크립트는 공유 전 정리.

## 6. 후속자가 읽는 순서

처음 받는 사람이 하루 안에 맥락을 잡게 하려면 이 순서를 추천한다.

1. `HANDOFF.md`
2. `experiments/grp_kd_based/report.md`
3. `experiments/s-diskd/report.md`
4. `experiments/wav2vec_grp_kd_960h/report.md`
5. `models.py`, `train_grp_kd.py`, `eval_spk_probe.py`
6. `models_wav2vec.py`, `train_wav2vec.py`
7. 필요한 스크립트만 골라서 dry-run

## 7. 후속 연구 우선순위

### 1순위: Conformer GRP-KD E4 재현 및 강화

이 프로젝트에서 가장 논문화 가능성이 높은 축이다.

해야 할 일:

- E1/E2/E4를 같은 환경에서 재현.
- 3 seed 이상 반복.
- WER 평균/표준편차 보고.
- `grl_alpha`, `orth_weight`, `spk_cls_weight` sweep.
- E5 alpha annealing 확인.
- speaker probe와 WER를 같이 분석.

논문 메시지:

> speaker information removal alone is not sufficient; text-preserving adversarial disentanglement improves KD only when it preserves ASR-relevant latent quality.

### 2순위: S-DisKD speaker factor KD

구현 부담이 작고 결과가 직관적이다.

해야 할 일:

- E3 재현.
- speaker layer sweep.
- KD weight sweep.
- MSE/cosine/contrastive 비교.
- text factor KD가 왜 CTC와 충돌하는지 loss curve 및 layer representation으로 설명.

논문 메시지:

> low-level speaker/acoustic factor alignment can indirectly improve ASR student learning, while direct high-level text factor alignment may conflict with CTC optimization.

### 3순위: wav2vec2 negative/boundary condition 정리

메인 성공 스토리보다는 “언제 안 되는가”를 보여주는 보조 실험으로 쓰는 편이 좋다.

메시지:

- random-init weak student: GRP-KD improves.
- SSL-pretrained strong student: CTC baseline dominates and GRP-KD harms.
- hard orth disentanglement consistently harms wav2vec2.

## 8. SPL 관점 판단

현재 그대로 SPL에 넣기는 어렵다.

이유:

- 최신 960h/wav2vec2 결과에서 제안 KD가 CTC baseline을 넘지 못했다.
- positive result가 Conformer 쪽에 있지만 seed 반복과 ablation이 부족하다.
- 실험 라인이 많아 논문 메시지가 흩어져 있다.

하지만 좋은 방향은 있다.

SPL 후보 메시지는 다음 중 하나로 좁히는 것이 좋다.

### 후보 A: Conformer positive 중심

> Disentanglement-aware GRP-KD with text-preserving adversarial regularization improves small Conformer ASR student training.

필수 추가:

- E1/E2/E4 multi-seed.
- logit/layer/GRP-KD baseline 정리.
- speaker probe와 WER correlation.

### 후보 B: 조건부 효과 중심

> Disentanglement-aware KD is effective only under compatible teacher/student representation conditions; hard disentanglement can over-regularize strong SSL-pretrained ASR students.

필수 추가:

- Conformer positive와 wav2vec2 negative를 같은 프레임으로 비교.
- student strength에 따른 효과 분석.

### 후보 C: 후배가 안전하게 이어가기 좋은 주제

> Speaker factor alignment improves ASR student training, whereas direct text factor alignment conflicts with CTC.

필수 추가:

- S-DisKD E3 중심 layer/weight/loss sweep.

## 9. 인수인계할 때 말할 핵심

교수님께:

> DAG-KD를 버리려는 것은 아닙니다. 지금까지 구현과 실험을 통해 어떤 조건에서 효과가 있고 어떤 조건에서 깨지는지가 꽤 정리됐습니다. 다만 SPL로 바로 밀기에는 최신 960h/wav2vec2 결과가 CTC baseline을 넘지 못해서, 제가 계속 붙잡기보다는 후속자가 Conformer GRP-KD와 S-DisKD 쪽 positive signal을 중심으로 재현, seed, ablation을 쌓는 편이 더 효율적이라고 판단했습니다. 저는 이 기반을 정리해서 넘기고, 현재 더 우선순위가 높은 다른 연구에 집중하고 싶습니다.

후배에게:

> 처음부터 다시 시작할 필요는 없어. 내가 해본 결과, 무작정 MI나 orthogonal constraint를 넣으면 안 되고, Conformer GRP-KD E4와 S-DisKD speaker KD가 가장 좋은 출발점이야. wav2vec2 960h는 CTC baseline이 너무 강해서 KD가 오히려 손해였고, 그건 실패라기보다 피해야 할 조건을 확인한 결과라고 보면 돼.

내 입장 정리:

> DAG-KD를 접는 것이 아니라, 내가 할 수 있는 탐색 단계는 충분히 했고 이제는 재현, seed, ablation을 꾸준히 쌓아야 하는 단계라고 본다. 그 작업은 후배가 이어받기에 좋고, 나는 지금 내 연구 포트폴리오에서 더 집중해야 할 주제로 시간을 옮기고 싶다.

## 10. 넘기기 전 체크리스트

- [ ] `HANDOFF.md`와 각 `experiments/*/report.md`를 후배에게 읽는 순서대로 안내.
- [ ] `git status --short` 기준으로 tracked/untracked 파일 정리.
- [ ] W&B API key가 들어간 script 제거 또는 환경변수화.
- [ ] 가장 중요한 checkpoint/output 경로를 따로 공유.
- [ ] Conformer GRP-KD E1/E2/E4 재현 명령을 확정.
- [ ] S-DisKD E1/E3 재현 명령을 확정.
- [ ] wav2vec2 960h 결과는 negative/boundary result로 분류.
- [ ] 교수님께는 "포기"가 아니라 "책임 있는 이관"으로 설명.

## 11. 최종 판단

DAG-KD는 지금 멈추면 아까운 주제다. 다만 지금 단계에서 필요한 일은 새로운 아이디어를 더 얹는 것이 아니라, 이미 나온 positive/negative signal을 좁히고 재현성을 쌓는 작업이다.

후속자에게는 Conformer GRP-KD E4와 S-DisKD speaker KD를 중심으로 넘기고, wav2vec2 라인은 “strong SSL student에서는 KD/disentanglement가 오히려 해가 될 수 있다”는 boundary evidence로 보존하는 것이 가장 좋다.
