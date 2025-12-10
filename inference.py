#!/usr/bin/env python3
# inference_dagkd_splits.py
"""
LibriSpeech / TEDLIUM2 / CommonVoice / GigaSpeech 4 splits에 대해
DAG-KD( DistilDAGKDCTCModelBPE ) Student 모델로 inference 및 평가.

- 출력 형식은 FlowMatching용 inference_splits.py와 동일:
    ===== Evaluating split: dev.clean =====
    results: [...]
    → dev.clean | loss = 0.1234 | WER = 5.67%
    ...
    ===== Final Summary =====
    dev.clean   → Loss: ..., WER ...
"""

import os
import re
import regex as re_u
import json
import uuid
import argparse
import glob
import unicodedata

import torch
import lightning as pl
import soundfile as sf
import nemo.collections.asr as nemo_asr

from datasets import load_dataset, DownloadConfig, config as hf_config
from copy import deepcopy
import aiohttp  # train.py와 동일하게 사용

# ─────────────────────────────────────────────────────────────
# 프로젝트 구조에 맞게 import 경로 수정해줘
from utils import (
    release_nemoAPI,
    build_manifest_from_hf,   # 기존 FlowMatching에서 쓰던 것과 동일 인터페이스라고 가정
    build_manifest_from_hf_with_meta,
)
from models import DistilDAGKDCTCModelBPE
# GigaSpeech용 manifest 함수는 이 파일 안에 구현
# ─────────────────────────────────────────────────────────────


# ====================== 텍스트 정규화 & GigaSpeech manifest ======================

def normalize_text_cv(s: str, keep_punct: bool = False) -> str:
    # 0) 유니코드 정규화 + 소문자
    s = unicodedata.normalize("NFKC", s or "").strip().lower()

    # 1) 흔한 특수문자 매핑 및 제거
    for k, v in {
        "\u2047": " ", "“": '"', "”": '"', "„": '"',
        "‘": "'", "’": "'",
        "–": "-", "—": "-",
        "…": " ",
        "‹": " ", "›": " ", "«": " ", "»": " ",
    }.items():
        s = s.replace(k, v)

    # 2) 바깥 큰따옴표만 한 쌍이면 제거
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]

    # 3) CV 특유의 공백+아포스트로피 정리: "men 's" → "men's"
    s = re.sub(r"\s+'\s*s\b", "'s", s)

    # 4) 평가용: 구두점 제거 권장(문자/숫자/공백/아포스트로피/하이픈만 유지)
    if not keep_punct:
        s = re_u.sub(r"[^\p{L}\p{N}\s'\-]", " ", s)

    # 5) 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_manifest_from_hf_gigaspeech(ds, manifest_path: str, cache_dir: str):
    """
    HF GigaSpeech Dataset -> NeMo manifest(JSONL)
      { "audio_filepath": ..., "duration": ..., "text": ... }
    """
    import re as _re

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    extract_root  = os.path.join(cache_dir, "extracted")
    tmp_audio_dir = os.path.join(cache_dir, "tmp_manifest_audio")
    os.makedirs(tmp_audio_dir, exist_ok=True)

    # 레퍼런스 내 특수 태그 목록 (대소문자 무시)
    BANNED_TAGS = {
        "<MUSIC>", "<COMMA>", "<NOISE>", "<VOCALIZED_NOISE>", "<LAUGHTER>",
        "<SPOKEN_NOISE>", "<PERIOD>", "<QUESTION_MARK>", "<EXCLAMATION_MARK>",
        "<SEMICOLON>", "<COLON>", "<DASH>", "<ELLIPSIS>", "<SIL>", "<OTHER>",
    }
    _TAGS_RE = _re.compile(
        r"(?:%s)" % "|".join(_re.escape(t) for t in BANNED_TAGS), _re.IGNORECASE
    )

    def _strip_special_tags(text: str):
        if not text:
            return "", True
        no_tags = _TAGS_RE.sub(" ", text)
        no_tags = _re.sub(r"\s+", " ", no_tags).strip()
        is_tag_only = (len(no_tags) == 0)
        return no_tags, is_tag_only

    def _resolve_audio_path(sample) -> tuple[str, float]:
        """오디오 파일 실제 경로와 duration(sec) 반환"""
        audio = sample["audio"]
        orig_path = audio.get("path", None)
        # 1) path가 실제 파일이면 그대로 사용
        if isinstance(orig_path, str) and os.path.isfile(orig_path):
            arr = audio.get("array", None)
            sr  = audio.get("sampling_rate", 16000)
            if arr is not None:
                dur = float(len(arr)) / float(sr)
            else:
                try:
                    info = sf.info(orig_path)
                    dur = float(info.frames) / float(info.samplerate)
                except Exception:
                    dur = 0.0
            return orig_path, dur

        # 2) cache/extracted 아래에서 파일명으로 재귀 검색
        cand_name = None
        if isinstance(orig_path, str) and len(orig_path) > 0:
            cand_name = os.path.basename(orig_path)
        elif isinstance(sample.get("path", None), str):
            cand_name = os.path.basename(sample["path"])

        if cand_name:
            matches = glob.glob(os.path.join(extract_root, "**", cand_name), recursive=True)
            if matches:
                found = matches[0]
                try:
                    info = sf.info(found)
                    dur = float(info.frames) / float(info.samplerate)
                except Exception:
                    arr = audio.get("array", None)
                    sr  = audio.get("sampling_rate", 16000)
                    dur = float(len(arr)) / float(sr) if arr is not None else 0.0
                return found, dur

        # 3) bytes가 있으면 tmp에 저장
        if audio.get("bytes", None) is not None:
            ext = ".wav"
            if isinstance(orig_path, str) and "." in os.path.basename(orig_path):
                ext = os.path.splitext(orig_path)[1] or ".wav"
            out_path = os.path.join(tmp_audio_dir, f"hf_{uuid.uuid4().hex}{ext}")
            with open(out_path, "wb") as f:
                f.write(audio["bytes"])
            try:
                info = sf.info(out_path)
                dur = float(info.frames) / float(info.samplerate)
            except Exception:
                arr = audio.get("array", None)
                sr  = audio.get("sampling_rate", 16000)
                dur = float(len(arr)) / float(sr) if arr is not None else 0.0
            return out_path, dur

        # 4) array+sr로 WAV 저장
        arr = audio.get("array", None)
        sr  = audio.get("sampling_rate", 16000)
        if arr is not None:
            out_path = os.path.join(tmp_audio_dir, f"hf_{uuid.uuid4().hex}.wav")
            sf.write(out_path, arr, sr)
            dur = float(len(arr)) / float(sr)
            return out_path, dur

        raise FileNotFoundError("오디오 경로/바이트/배열 중 어느 것도 사용할 수 없습니다.")

    n_written, n_skipped_short, n_skipped_tagonly = 0, 0, 0
    min_sec = 1.0
    with open(manifest_path, "w", encoding="utf-8") as fout:
        for sample in ds:
            audio_path, duration = _resolve_audio_path(sample)

            # 너무 짧은 샘플 스킵
            if duration < min_sec:
                n_skipped_short += 1
                continue

            raw_text = sample.get("sentence", None)
            if raw_text is None:
                raw_text = sample.get("text", "")

            tag_stripped, is_tag_only = _strip_special_tags(raw_text)
            if is_tag_only:
                n_skipped_tagonly += 1
                continue

            text = normalize_text_cv(tag_stripped, keep_punct=False)

            fout.write(
                json.dumps(
                    {
                        "audio_filepath": audio_path,
                        "duration": float(duration),
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    print(
        f"[manifest] wrote {n_written} lines "
        f"(skipped {n_skipped_short} < {min_sec}s, "
        f"skipped {n_skipped_tagonly} tag-only refs) → {manifest_path}"
    )


# ====================== DAG-KD용 config helper ======================

def make_dagkd_student_config(teacher_model, args, train_manifest, val_manifest, test_manifest):
    """
    DAG-KD train.py에서 하던 cfg 설정을 inference용으로 축약한 버전.
    - Teacher cfg를 복사
    - hidden dim / head 수 절반
    - ds 관련 필드 최소한 셋팅
    - DAG-KD 관련 필드( latent_dim 등 ) 기본값 세팅
    """
    from omegaconf import OmegaConf
    stu_cfg = deepcopy(teacher_model.cfg)

    # ── 데이터셋 설정 (train/val/test manifest는 일단 placeholder)
    stu_cfg.train_ds.is_tarred = False
    stu_cfg.train_ds.manifest_filepath = train_manifest
    stu_cfg.train_ds.sample_rate = args.data_sample_rate
    stu_cfg.train_ds.batch_size = args.batch_size

    stu_cfg.validation_ds.is_tarred = False
    stu_cfg.validation_ds.manifest_filepath = val_manifest
    stu_cfg.validation_ds.sample_rate = args.data_sample_rate
    stu_cfg.validation_ds.batch_size = args.batch_size

    stu_cfg.test_ds.is_tarred = False
    stu_cfg.test_ds.manifest_filepath = test_manifest
    stu_cfg.test_ds.sample_rate = args.data_sample_rate
    stu_cfg.test_ds.batch_size = args.batch_size

    # ── encoder/decoder 스케일 절반 (train.py와 동일 로직)
    stu_cfg.encoder.d_model = max(8, teacher_model.cfg.encoder.d_model // 2)
    stu_cfg.encoder.n_heads = max(1, teacher_model.cfg.encoder.n_heads // 2)
    stu_cfg.decoder.feat_in = max(8, teacher_model.cfg.decoder.feat_in // 2)

    # ── NeMo dataconfig에 sample_id 반환 옵션
    OmegaConf.set_struct(stu_cfg, False)
    OmegaConf.set_struct(stu_cfg.train_ds, False)
    OmegaConf.set_struct(stu_cfg.validation_ds, False)
    OmegaConf.set_struct(stu_cfg.test_ds, False)
    stu_cfg.train_ds.return_sample_id = True
    stu_cfg.validation_ds.return_sample_id = False
    stu_cfg.test_ds.return_sample_id = False

    # ── DAG-KD 관련 필드 (train.py 기본값 복사)
    stu_cfg.latent_dim = 96
    stu_cfg.disen_mi_weight = 1e-3              # λ_MI
    stu_cfg.rec_txt_lambda = 0.1
    stu_cfg.rec_spk_lambda = 0.1
    stu_cfg.disen_mi_pairs  = "ts,tp,ps"        # 사용 쌍
    stu_cfg.disen_gst_tokens = 10
    stu_cfg.disen_gst_heads  = 4
    stu_cfg.disen_gst_token_dim = 96
    stu_cfg.disen_gst_ref_dim = 96
    stu_cfg.disen_spk_ce_lambda = 1.0
    stu_cfg.disen_global_style = True

    # speaker/lang head는 어차피 WER 평가에는 직접 영향 X → 대략값
    stu_cfg.num_spk = getattr(stu_cfg, "num_spk", 251)
    stu_cfg.num_lang = getattr(stu_cfg, "num_lang", 0)

    # 기타
    stu_cfg.out_dir = getattr(stu_cfg, "out_dir", args.data_dir)
    stu_cfg.disen_vis_enable = getattr(stu_cfg, "disen_vis_enable", False)

    return stu_cfg


# ====================== 공용 argparser ======================

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LibriSpeech 4 splits에 대해 DAG-KD 모델 inference 및 평가"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="학습된 체크포인트 경로 (예: outputs/.../checkpoints/last.ckpt)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="훈련 시 사용한 data root (manifests 폴더가 있어야 함)",
    )
    parser.add_argument("--data_cfg", type=str, default="train_100")
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="사용할 GPU 개수",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="추론 시 배치 크기",
    )
    parser.add_argument(
        "--data_script_path",
        type=str,
        default="./librispeech_asr.py",
        help="HF 데이터 스크립트 경로 [./librispeech_asr.py, ./tedlium_asr.py, ./commonvoice_asr.py, ./gigaspeech.py]",
    )
    parser.add_argument(
        "--data_config_name",
        type=str,
        default="train_100",
        help="HF 데이터 config 이름 (train_100 등)",
    )
    # 아래 KD 관련 플래그들은 DistilDAGKDCTCModelBPE __init__ 인자와 매핑
    parser.add_argument(
        "--use_ctc",
        type=str2bool,
        default=True,
        help="CTC loss 사용 여부",
    )
    parser.add_argument(
        "--use_logit_distillation",
        type=str2bool,
        default=True,
        help="Teacher logits와의 KL loss 사용 여부 (DAG-KD의 use_logit_kd로 전달)",
    )
    parser.add_argument(
        "--use_layerwise_distillation",
        type=str2bool,
        default=False,
        help="레이어 단위 KD 사용 여부 (DAG-KD의 use_layer_kd로 전달)",
    )
    parser.add_argument(
        "--use_flow_matching",
        type=str2bool,
        default=False,
        help="Flow Matching KD 사용 여부 (DAG-KD의 use_flow로 전달)",
    )
    parser.add_argument(
        "--flow_steps",
        type=int,
        default=8,
        help="Flow Matching 단계 수 (DAG-KD의 flow_steps로 전달)",
    )
    parser.add_argument(
        "--flow_schedule",
        type=str,
        default="rectified",
        choices=["rectified", "vp_ode", "ve_ode"],
        help="(DAG-KD에서는 현재 직접 쓰지 않지만 CLI 호환을 위해 유지)",
    )
    parser.add_argument(
        "--flow_weight",
        type=float,
        default=1.0,
        help="Flow KD loss 가중치",
    )
    parser.add_argument(
        "--use_diffkd",
        type=str2bool,
        default=False,
        help="DiffKD 기법 사용 여부",
    )
    parser.add_argument(
        "--kd_temperature",
        type=float,
        default=1.0,
        help="logit distillation 온도",
    )
    parser.add_argument(
        "--kd_alpha",
        type=float,
        default=0.5,
        help="logit distillation 가중치",
    )
    parser.add_argument(
        "--layer_kd_alpha",
        type=float,
        default=0.5,
        help="레이어 단위 KD 가중치",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="librispeech",
        help="평가할 데이터셋 (librispeech, tedlium2, commonvoice, gigaspeech)",
    )
    parser.add_argument(
        "--is_teacher",
        type=str2bool,
        default=False,
        help="teacher 모델로 평가할지 여부 (DAG-KD에서는 보통 False)",
    )
    parser.add_argument(
        "--data_sample_rate",
        type=int,
        default=16000,
        help="샘플링 주파수",
    )
    return parser.parse_args()


# ====================== main ======================

def main():
    args = parse_args()

    # 1) Trainer 세팅
    trainer = pl.Trainer(accelerator="gpu", devices=args.gpus)

    # 2) Teacher 모델 로드
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name="stt_en_conformer_ctc_small",
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
        trainer=trainer,
    )

    # train.py와 동일하게 .nemo 압축 해제
    nemo_archive_dir = os.path.join(args.data_dir, "nemo_archive")
    os.makedirs(nemo_archive_dir, exist_ok=True)
    release_nemoAPI(teacher, out_folder=nemo_archive_dir)

    # 3) Student config 생성 (DAG-KD용)
    manifest_dir = os.path.join(args.data_dir, args.data_cfg, "manifests")
    cache_dir = os.path.join(args.data_dir, args.data_cfg, "cache")
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    hf_config.HF_DATASETS_CACHE = cache_dir

    # train.py에서 만든 LibriSpeech manifest 경로들
    librispeech_manifest_map = {
        "dev.clean":  os.path.join(manifest_dir, "dev_clean.json"),
        "dev.other":  os.path.join(manifest_dir, "dev_other.json"),
        "test.clean": os.path.join(manifest_dir, "test_clean.json"),
        "test.other": os.path.join(manifest_dir, "test_other.json"),
    }

    if args.eval_data == "librispeech":
        train_m = os.path.join(manifest_dir, "train.json")
        val_m   = librispeech_manifest_map["dev.clean"]
        test_m  = librispeech_manifest_map["test.clean"]
    else:
        print(f'[Error] eval_data={args.eval_data}인 경우 train/val/test json 경로 설정 필요')
        return

    if args.is_teacher:
        # Teacher만 평가하고 싶으면 teacher.cfg 그대로 활용
        model_cfg = deepcopy(teacher.cfg)
    else:
        model_cfg = make_dagkd_student_config(
            teacher_model=teacher,
            args=args,
            train_manifest=train_m,
            val_manifest=val_m,
            test_manifest=test_m,
        )

    # 4) DAG-KD Student 모델 인스턴스 생성
    model = DistilDAGKDCTCModelBPE(
        cfg=model_cfg,
        trainer=trainer,
        teacher_model=teacher,
        use_ctc=args.use_ctc,
        use_logit_kd=args.use_logit_distillation,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
        use_layer_kd=args.use_layerwise_distillation,
        layer_kd_alpha=args.layer_kd_alpha,
        use_flow=args.use_flow_matching,
        flow_steps=args.flow_steps,
        flow_weight=args.flow_weight,
        use_diffkd=args.use_diffkd,
        diffkd_steps=5,              # 필요하면 argparse로 노출 가능
        use_disent=True,             # 필요 시 argparse로 뺄 수 있음
        disent_spk_layers=[4],
        disent_txt_layers=[16],
    )
    model.eval()

    # 5) 체크포인트 로드
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load_state_dict] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("  missing (first 10):", missing[:10])
    if unexpected:
        print("  unexpected (first 10):", unexpected[:10])

    # 6) DownloadConfig (train.py와 동일하게)
    dl_cfg = DownloadConfig(
        cache_dir=cache_dir,
        force_download=False,
        resume_download=True,
        max_retries=8,
        disable_tqdm=False,
        download_desc="Downloading dataset",
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=72000)}},
        delete_extracted=False,
        extract_compressed_file=True,
        force_extract=True,
    )

    # 7) 평가 대상 split 및 HF 스크립트 설정
    if args.eval_data == "tedlium2":
        split_names = ["validation", "test"]
        script_path = "./tedlium_asr.py"
        config_name = "release2"
    elif args.eval_data == "librispeech":
        split_names = ["dev.clean", "dev.other", "test.clean", "test.other"]
        script_path = args.data_script_path
        config_name = args.data_config_name
    elif args.eval_data == "commonvoice":
        split_names = ["validation", "test"]
        script_path = "./commonvoice_asr.py"
        config_name = "en"
    elif args.eval_data == "gigaspeech":
        split_names = ["validation", "test"]
        script_path = "./gigaspeech.py"
        config_name = "dev"
    else:
        raise ValueError("지원하는 eval_data: librispeech, tedlium2, commonvoice, gigaspeech")

    all_metrics = {}
    for split in split_names:
        print(f"\n===== Evaluating split: {split} =====")

        # ── LibriSpeech의 경우: train.py에서 만든 manifest 직접 사용 (재다운로드 X)
        if args.eval_data == "librispeech" and split in librispeech_manifest_map \
           and os.path.isfile(librispeech_manifest_map[split]):
            manifest_i = librispeech_manifest_map[split]
            print(f"[INFO] Using existing manifest (no HF download): {manifest_i}")
        else:
            # 그 외 데이터셋 또는 manifest 없는 경우 → HF에서 로드 후 manifest 생성
            ds = load_dataset(
                script_path,
                config_name,
                split=split,
                trust_remote_code=True,
                download_config=dl_cfg,
                cache_dir=cache_dir,
            )
            json_name = split.replace(".", "_") + ".json"
            manifest_i = os.path.join(manifest_dir, json_name)

            if args.eval_data == "gigaspeech":
                build_manifest_from_hf_gigaspeech(ds, manifest_i, cache_dir)
            else:
                # 시그니처 맞추기 위해 spk2idx=None 전달
                build_manifest_from_hf_with_meta(ds, manifest_i, cache_dir, spk2idx=None)

        # ── test config 설정 및 평가
        test_cfg = deepcopy(model.cfg.test_ds)
        test_cfg.manifest_filepath = manifest_i
        test_cfg.shuffle = False
        model.setup_test_data(test_cfg)
        dl = model.test_dataloader()

        results = trainer.test(
            model=model,
            dataloaders=[dl],
            verbose=False,
        )
        print(f"results: {results}")

        res  = results[0]
        wer  = res.get("test_wer", res.get("wer", None))
        loss = res.get("test_loss", res.get("loss", None))
        print(f"→ {split} | loss = {loss:.4f} | WER = {wer:.2%}")

        all_metrics[split] = {"loss": loss, "wer": wer}

    print("\n===== Final Summary =====")
    for split, m in all_metrics.items():
        print(f"{split:10s} → Loss: {m['loss']:.4f}, WER: {m['wer']:.2%}")


if __name__ == "__main__":
    main()
