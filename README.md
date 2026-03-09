# DAG-KD: Disentanglement-Aware Generative Knowledge Distillation for ASR

## 개요

DAG-KD는 자동 음성 인식(ASR) 분야에서 **Teacher-Student Knowledge Distillation** 프레임워크입니다.
NeMo의 Conformer CTC 모델을 Teacher로 사용하고, 더 작은 Student 모델을 효율적으로 학습시키는 것을 목표로 합니다.

핵심 아이디어:
- **ASR 표현의 계층적 성질** (하위 레이어: speaker/prosody, 상위 레이어: linguistic)을 활용한 disentanglement
- **Generative KD** (Flow Matching, DiffKD)로 Teacher feature 분포를 Student에 전이
- **상호정보량(MI) 최소화** (CLUB)로 Text / Speaker / Prosody 표현을 서로 독립적으로 분리

---

## 아키텍처

```
Teacher (stt_en_conformer_ctc_small, frozen)
  └── Encoder (N layers, d_model=256, n_heads=8)
      ├── [Lower layers] → Speaker representation
      └── [Upper layers] → Text representation

Student (Teacher 절반 크기: d_model=128, n_heads=4)
  └── Encoder (N layers)
      ├── Text Encoder  (Conv1x1, latent_dim=96)
      ├── Speaker Encoder (Conv1x1 + TDNN backbone)
      └── Prosody Encoder (Global Style Token, GST)
```

### 학습 손실 구성

| Loss | 설명 |
|------|------|
| **CTC Loss** | 기본 ASR 학습 손실 |
| **Logit KD** | Teacher/Student CTC 로짓 간 KL divergence |
| **Layer KD** | 레이어별 특징 MSE (옵션) |
| **Flow Matching KD** | Student → Teacher feature를 Rectified Flow로 전이 |
| **DiffKD** | Diffusion 방식의 iterative denoising KD |
| **Reconstruction Loss** | Text/Speaker/Prosody 인코더 재구성 손실 |
| **MI Loss (CLUB)** | Text-Speaker, Text-Prosody, Speaker-Prosody 상호정보량 상한 최소화 |
| **Speaker CE** | Speaker 분류기 Cross-Entropy |
| **Prosody Physical Loss** | F0, Energy, Voicing 예측 손실 |
| **Text Speaker Probe** | Text 표현에서 Speaker 정보 제거 확인용 Adversarial Probe |

---

## 프로젝트 구조

```
DAG-KD/
├── train.py                  # 메인 학습 스크립트 (Student KD 학습)
├── train_teacher.py          # Teacher 단독 학습 + disentanglement
├── train_ver2.py             # 학습 스크립트 ver2
├── train_ver3.py             # 학습 스크립트 ver3
├── train_student.py          # Student 단독 학습 스크립트
├── models.py                 # 핵심 모델 (DistilDAGKDCTCModelBPE)
├── models_ver2.py            # 모델 ver2 (TeacherASRWithDisent 등)
├── models_ver3.py            # 모델 ver3
├── utils.py                  # 데이터 처리 / manifest 생성 / 유틸
├── inference.py              # 추론 및 WER 평가 스크립트
├── librispeech_asr.py        # LibriSpeech HuggingFace 데이터셋 스크립트
├── gigaspeech_asr.py         # GigaSpeech HuggingFace 데이터셋 스크립트
├── scripts/
│   ├── train/                # 실험별 학습 실행 스크립트 (.sh)
│   │   ├── dag-kd_baseline.sh
│   │   ├── train_teacher.sh
│   │   ├── train_student.sh
│   │   ├── disen_*/          # Disentanglement 실험 스크립트들
│   │   └── ...
│   └── inference/
│       └── inference.sh      # 추론 실행 스크립트
├── data/                     # LibriSpeech 데이터셋 캐시
└── outputs/                  # 학습 결과, 체크포인트, XAI 결과
```

---

## 주요 모듈

### `DistilDAGKDCTCModelBPE` ([models.py](models.py))

`nemo_asr.models.EncDecCTCModelBPE`를 상속한 Student 모델. 주요 컴포넌트:

- **FlowMatchingModule**: Student → Teacher feature를 Rectified Flow로 정렬
- **DiffKDModule**: Diffusion 방식의 iterative denoising으로 feature KD
- **GlobalProsodyReferenceEncoder + GlobalStyleTokenLayer**: 운율(prosody) 임베딩 추출
- **ARClubGaussian / ClubGaussian**: 상호정보량 상한 추정 (CLUB 기반)
- **Speaker Backbone (TDNN)**: 화자 임베딩 + 분류기
- **Text Speaker Probe**: Text 표현의 화자 불변성 검증

### Forward Hook 기반 Feature 캡처

학습 중 Teacher/Student 각 레이어의 중간 출력을 hook으로 캡처하여 KD에 활용합니다.

---

## 환경 설정

### 필수 패키지

```bash
pip install nemo_toolkit[asr]
pip install lightning
pip install wandb
pip install datasets
pip install librosa soundfile
pip install omegaconf regex
```

### 데이터셋

LibriSpeech는 HuggingFace `datasets`를 통해 자동으로 다운로드됩니다.

```
train split: train-clean-100 (100시간)
val split:   dev-clean
test split:  test-clean, test-other
```

---

## 학습

### Teacher 학습 (선택)

```bash
bash scripts/train/train_teacher.sh
```

또는 직접 실행:

```bash
python train_teacher.py \
  --wandb_run my_teacher \
  --out outputs/teacher \
  --teacher_name stt_en_conformer_ctc_small \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --batch_size 16 \
  --epochs 100 \
  --gpus 1
```

### Student KD 학습 (메인)

```bash
bash scripts/train/dag-kd_baseline.sh
```

또는 직접 실행:

```bash
python train.py \
  --wandb_run dag_kd_run \
  --out outputs/dag-kd/run \
  --teacher_name stt_en_conformer_ctc_small \
  --data_cfg train_100 \
  --train_split train.clean.100 \
  --use_ctc True \
  --use_logit_kd True \
  --use_flow True \
  --use_diffkd True \
  --use_disent True \
  --disent_spk_layers "4" \
  --disent_txt_layers "16" \
  --disen_mi_pairs "ts,tp,ps" \
  --disen_lll_weight 1.0 \
  --disen_mi_weight 1.0 \
  --batch_size 32 \
  --epochs 100 \
  --gpus 1
```

### 주요 학습 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--teacher_name` | `stt_en_conformer_ctc_small` | NeMo Pretrained Teacher 모델 |
| `--use_ctc` | `True` | CTC 손실 사용 여부 |
| `--use_logit_kd` | `True` | Logit KD (CTC KL divergence) 사용 여부 |
| `--kd_alpha` | `0.5` | KD 손실 가중치 |
| `--kd_temperature` | `1.0` | Knowledge Distillation 온도 |
| `--use_flow` | `False` | Flow Matching KD 사용 여부 |
| `--flow_steps` | `8` | Flow Matching 스텝 수 |
| `--use_diffkd` | `False` | DiffKD 사용 여부 |
| `--use_disent` | `True` | Disentanglement 사용 여부 |
| `--disent_spk_layers` | `[1,2]` | Speaker 표현 추출 레이어 (하위) |
| `--disent_txt_layers` | `[15,16]` | Text 표현 추출 레이어 (상위) |
| `--disen_mi_pairs` | `ts,tp,ps` | MI 최소화 쌍 (t=text, s=speaker, p=prosody) |
| `--disen_mi_weight` | `1e-3` | MI 손실 가중치 λ_MI |
| `--neg_K` | `8` | CLUB 추정을 위한 Negative sample 수 |
| `--mi_warmup_steps` | `5000` | CLUB만 학습하는 warmup 스텝 |
| `--use_txt_spk_probe` | `True` | Text Speaker Probe 사용 여부 |
| `--test_mode` | `False` | 소규모 테스트 모드 (200 샘플) |

---

## 추론 및 평가

```bash
bash scripts/inference/inference.sh
```

또는 직접 실행:

```bash
python inference.py \
  --ckpt_path outputs/dag-kd/run/checkpoints/last.ckpt \
  --gpus 1 \
  --batch_size 8 \
  --data_dir data \
  --data_cfg train_100 \
  --eval_data librispeech
```

평가 지표:
- WER (Word Error Rate), 평균 ± 표준편차
- WER 히스토그램 / Boxplot 자동 저장 (`outputs/xai/wer_plots/`)

---

## 실험 구성

`scripts/train/` 디렉토리에 다양한 실험 구성이 포함되어 있습니다:

| 스크립트 패턴 | 설명 |
|--------------|------|
| `baseline_*.sh` | KD 없이 단순 학습 베이스라인 |
| `disen_1_*.sh` | Disentanglement v1 (상위 1레이어 spk/txt) |
| `disen_2_*.sh` | Disentanglement v2 (상위 2레이어) |
| `disen_4_*.sh` | Disentanglement v4 (다층 disentanglement) |
| `*_ver2~4*.sh` | 버전별 변형 실험 |
| `*_spkadv_*.sh` | Speaker Adversarial 방식 실험 |
| `*LayerwiseDisen*.sh` | Layer-wise Disentanglement 실험 |
| `*_mi*.sh` | MI 가중치 변형 실험 |

---

## 학습 로깅

[Weights & Biases (W&B)](https://wandb.ai)를 통해 학습 과정을 모니터링합니다.

주요 로깅 지표:
- `train/mi_upper`: Text-Speaker-Prosody 상호정보량 상한
- `train/lll`: Layer Locality Loss
- `train/rec_txt`, `train/rec_spk`, `train/rec_pros`: 재구성 손실
- `train/spk_ce`, `train/spk_acc`: 화자 분류 손실/정확도
- `train/phys_loss`: Prosody 물리량 예측 손실 (F0, Energy, Voicing)

---

## 핵심 기술 참고

- **NeMo Conformer CTC**: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- **CLUB (Contrastive Log-ratio Upper Bound)**: MI 상한 추정
- **Global Style Token (GST)**: 운율 표현 추출
- **Flow Matching**: Rectified Flow 기반 생성형 KD
- **LibriSpeech**: [openslr.org/12](http://www.openslr.org/12)
