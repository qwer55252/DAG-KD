# wav2vec2 GRP-KD on LibriSpeech 960h (SSL Student)

## 1. 배경 및 문제 인식

선행 실험 (`exp/wav2vec-grp-kd-orth`, 100h, random-init transformer)에서:
- A_E1 baseline test_clean **14.92%**, A_E2 +orth 15.45%, B-track ~18% — single-digit 미달
- 14% WER는 standard wav2vec2-base-960h (3.4%) 대비 너무 높아 KD 변종 우열을 따지기 전에 "왜 이렇게 낮냐"는 reviewer pushback이 예상됨

**가설 검증을 단자릿수 WER 영역에서 수행**해야 결과 신뢰도 확보 가능.

## 2. "Who breaks less" 함정과 회피

ASR fine-tuned 모델(예: `wav2vec2-base-960h`)을 학생으로 쓰면 시작 WER ~3%인데 KD는 이를 깎아내리기만 함 — KD가 도움된 게 아니라 덜 망친 것을 측정하게 됨.

**해결**: **SSL-only 모델 (`facebook/wav2vec2-base`)** 을 학생으로 사용.
- SSL pretraining 으로 풍부한 latent 표현 보유
- CTC 헤드는 random init → ASR 능력 0에서 출발
- E0 baseline (CTC fine-tuning만, no KD)이 ~3.5–4.5% 도달 예상
- KD 변종은 **E0를 추가로 깎아야** 효과 인정

## 3. 실험 설계

### Track A only (Track B는 결과 보고 결정)

| ID | Student init | Loss 구성 | 측정 의미 |
| --- | --- | --- | --- |
| **A-E0** | `wav2vec2-base` SSL | CTC | Fine-tuning 단독 anchor (single-digit 검증) |
| **A-E1** | 동상 | CTC + logit-KD + **GRP-KD (disen=0)** | KD가 fine-tuning 위에 +α 주는지 |
| **A-E2** | 동상 | E1 + **orth + spk_cls (disen=1)** | disen이 GRP-KD 위에 +α 주는지 |

Teacher: `facebook/wav2vec2-large-960h` 고정 (24L, d=1024).
Layer alignment: 균등 stride (teacher 짝수층 {2,4,...,24} ↔ student {1..12}).

### 공통 하이퍼파라미터
| 항목 | 값 | 이전 100h 대비 변경 사유 |
| --- | --- | --- |
| Data | LibriSpeech 960h (train.clean.100 + clean.360 + other.500) | 9.6× 데이터로 single-digit 도달 |
| Student | `facebook/wav2vec2-base` (SSL only, 12L d=768, 95M) | random init 대신 SSL pretraining 활용 |
| Processor | `facebook/wav2vec2-base-960h` | SSL 체크포인트엔 tokenizer 없음 |
| `--random_init_student` | False | 전체 SSL 모델을 student로 로드 |
| `--freeze_feature_extractor` | True | P1 정책 유지 (CNN pretrained+frozen) |
| Batch | 16 / GPU | 4-GPU dry-run에서 OOM 안전 마진 + 90%+ throughput |
| GPUs / run | 4 (sequential) | 첫 결과 빠르게, fail-fast 가능 |
| Effective batch | 64 | 4 GPU × 16 |
| LR | 1e-4 | Pretrained SSL fine-tuning 권장 범위, 큰 batch 대응 안전치 |
| Warmup | 1 epoch | Pretrained 학생이라 짧게 |
| KD warmup | 1 epoch | 동상 |
| Epochs | 10 | 960h 1 epoch ≈ 100h 9.6 epoch, 100h 100 epoch과 비슷한 step 수 |

### 4-GPU 실행 전략
Sequential 단독 실행 (each run uses all 4 GPUs):
- Stage 1: A-E0 (~18h) → Stage 2: A-E1 (~18h) → Stage 3: A-E2 (~18h)
- 총 **~57h** wall-clock
- 첫 결과 도달이 가장 빠름 → E0 결과 보고 후속 단계 디버그 가능
- vs 2-GPU 동시 (Strategy B): 총 시간은 같지만 첫 결과 36h 후

### Dry-run 검증 결과 (`scripts/train/wav2vec/grp_kd_960h/dry_run_100h.sh`)
- 100h 데이터 + 4-GPU + batch=10/14/18 모두 OOM 없이 학습 진입
- batch=14→18에서 throughput 3% 증가에 그쳐 diminishing returns 확인
- batch=16 채택 (안전 마진 + 거의 max throughput)
- Throughput: 1.0 it/s × effective batch 64 → 960h 1 epoch ≈ 1.9h → 10 epoch ≈ **19h/run**

## 4. 가설

> **SSL-pretrained 학생에 추가되는 GRP-KD는 단순 CTC fine-tuning baseline 대비 의미 있는 WER 개선 (≥0.3 pp on test_clean) 을 제공한다. 추가로 orth disen은 GRP-KD 위에 또 다른 개선을 가져온다.**

세부:
1. **E0 single-digit 도달**: `wav2vec2-base` SSL + 960h CTC FT로 test_clean ≤ 5% 도달 (예상 3.5–4.5%)
2. **E1 > E0**: GRP-KD 추가로 WER 0.3–1.0 pp 개선 (만약 차이 없으면 KD 효과는 단순 fine-tuning과 구분 안 되는 수준)
3. **E2 vs E1**: 100h 실험에서 orth는 baseline보다 손해였으나, single-digit 영역 + SSL 학생이라 새로운 dynamics 가능

## 5. 예상 리스크

- **KD 마진 축소**: SSL 학생은 100h random-init 때보다 강력 → KD가 줄 수 있는 마진이 작을 수 있음 (0.3 pp 미만이면 noise 구분 어려움). 통계적 유의성 위해 다중 seed 권장하지만 시간 부담.
- **GRP-KD 무용지물 가능성**: SSL latent이 이미 충분히 풍부해 teacher latent로부터 reconstruction이 redundant할 수 있음. E1-E0 차이 ≤0.2 pp면 GRP-KD 설계 자체 재고 필요.
- **디스크**: 91GB free. Phys_cache 960h용 추정 ~3GB. HF datasets cache는 이미 304GB 존재 (cleanup 필요시 별도 작업).
- **OOM 위험**: 긴 utterance (>25s) 등장 시 batch=16에서 OOM 가능성 — 발생 시 batch=14로 fallback.

## 6. 로그 계획

- WandB project: `DAG-KD-wav2vec`, runs `wav2vec_960h_A_E{0,1,2}_*`
- 추적 metric: train/total, train/ctc, train/kd_logit, train/grp_rec, train/grp_fm, train/grp_df, train/grp_orth, train/grp_spk_cls, val/wer
- Final eval 4-split: dev_clean, dev_other, test_clean, test_other

## 7. 후속 결정 분기

- **E0 < 5% (성공)**: E1 launch → E1 vs E0 비교 → E2 launch
- **E0 5–7%**: 디버그 (LR/epoch 조정), E1 후속 보류
- **E0 > 7%**: 설계 결함 — manifest/processor/optimizer 확인. 전체 중단.
- **E0 → E1 개선 ≥0.3 pp**: orth (E2) 의미 — launch
- **E1 ≈ E0**: orth 의미 약화 — 결과 보고 후 사용자 판단

## 8. 실행 스크립트 위치

- `scripts/train/wav2vec/grp_kd_960h/A_E0_no_kd.sh`
- `scripts/train/wav2vec/grp_kd_960h/A_E1_kd.sh`
- `scripts/train/wav2vec/grp_kd_960h/A_E2_kd_orth.sh`
- `scripts/train/wav2vec/grp_kd_960h/dry_run_100h.sh` (개발용)

## 9. 2026-05-27 재개 상태

- A-E0는 `outputs/wav2vec/grp_kd_960h/A_E0_no_kd/checkpoints/last.ckpt`에서 재개.
- 실행 세션: `tmux attach -t dagkd_A_E0_960h`
- 재개 직후 확인: `Epoch 5` 학습 루프 진입, 4 GPU 사용 확인.
- A-E1/A-E2는 A-E0의 final 4-split eval 확인 후 순차 실행.
- A-E0/A-E1/A-E2 스크립트는 checkpoint가 있으면 `--resume_ckpt_path`를 자동 전달하고 `train.log`에 append하도록 수정.

## 10. 2026-05-28 A-E0 완료 결과

| Split | WER |
| --- | ---: |
| dev_clean | 4.63% |
| dev_other | 12.55% |
| test_clean | 4.82% |
| test_other | 13.50% |

- A-E0는 `max_epochs=10`까지 정상 완료.
- clean 기준 single-digit 목표는 달성. `test_clean <= 5%` 기준으로 후속 A-E1 실행 조건 충족.
- 다음 실행: A-E1 (`CTC + logit-KD + GRP-KD`, `grp_disen_mode=0`).

## 11. 2026-05-28 A-E1 실행 상태

- A-E1 실행 시작: `tmux attach -t dagkd_A_E1_960h`
- W&B run: `wav2vec_960h_A_E1_kd`
- 1차 실행은 `batch_size=16`에서 teacher forward 중 GPU 0 OOM으로 중단.
- 대응: `train_wav2vec.py`에 `--accumulate_grad_batches` 옵션 추가.
- A-E1/A-E2는 `batch_size=8`, `accumulate_grad_batches=2`로 변경해 effective batch 64 유지.
- 재실행 W&B run: `wav2vec_960h_A_E1_kd` (`a7tpmx9n`)
- 재실행 초기 확인: sanity check 통과, `Epoch 0` 학습 루프 진입, 4 GPU 모두 사용 확인.

## 12. 2026-06-02 A-E1 완료 결과

| Split | A-E0 CTC | A-E1 GRP-KD | Delta |
| --- | ---: | ---: | ---: |
| dev_clean | 4.63% | 5.68% | +1.05 pp |
| dev_other | 12.55% | 13.81% | +1.26 pp |
| test_clean | 4.82% | 5.84% | +1.02 pp |
| test_other | 13.50% | 14.81% | +1.31 pp |

- A-E1는 `max_epochs=10`까지 정상 완료.
- GRP-KD는 SSL-pretrained student + 960h CTC baseline 위에서 개선을 주지 못했고, 모든 split에서 WER가 악화.
- 결정: 기존 분기 기준(`E0 -> E1 개선 >= 0.3 pp`)을 만족하지 못하므로 A-E2는 자동 launch하지 않음.
- 해석: random-init/약한 student에서 유효했던 GRP-KD가 strong SSL student에는 과한 teacher constraint로 작동했을 가능성이 큼.

## 13. 2026-06-02 A-E2 실행 상태

- 사용자 요청으로 A-E2 (`CTC + logit-KD + GRP-KD + orth + speaker CE`) 실행.
- 1차 A-E2 실행은 GRP-KD speaker classifier CE에서 `spk_idx=-1` label이 들어가 device-side assert로 실패.
- 원인: train manifest에 `spk_idx=-1` 샘플이 다수 존재하고, GRP-KD E2 speaker CE에는 기존 DAG speaker CE와 달리 valid-mask가 없었음.
- 수정:
  - `ManifestDataset`에서 `spk_idx < 0`이면 인접한 `speaker_id_mapping.json`과 `spk_id`로 복구 시도.
  - `GRPKDModule._forward_disen`에서 speaker CE를 `0 <= speaker_id < num_spk`인 샘플에만 적용.
- 재실행 세션: `tmux attach -t dagkd_A_E2_960h`
- W&B run: `wav2vec_960h_A_E2_kd_orth` (`b9eilghh`)
- 재실행 초기 확인: sanity check 통과, `Epoch 0` 학습 루프 진입, 4 GPU 모두 사용 확인.

## 14. 2026-06-04 A-E2 완료 결과

| Split | A-E0 CTC | A-E1 GRP-KD | A-E2 GRP-KD+Orth | E2 - E0 | E2 - E1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev_clean | 4.63% | 5.68% | 5.31% | +0.68 pp | -0.37 pp |
| dev_other | 12.55% | 13.81% | 13.22% | +0.67 pp | -0.59 pp |
| test_clean | 4.82% | 5.84% | 5.30% | +0.48 pp | -0.54 pp |
| test_other | 13.50% | 14.81% | 13.81% | +0.31 pp | -1.00 pp |

- A-E2는 `max_epochs=10`까지 정상 완료.
- Orth + speaker CE는 A-E1 대비 모든 split에서 WER를 회복시킴.
- 하지만 A-E0 CTC baseline 대비로는 여전히 모든 split에서 악화.
- 해석: disentanglement는 GRP-KD의 손상을 완화하지만, strong SSL-pretrained student에서는 teacher-side GRP constraint 자체가 CTC fine-tuning보다 유리하지 않음.
