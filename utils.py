import re, os, json, uuid, glob, torch, argparse, librosa, unicodedata
import numpy as np
import regex as re_u
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from collections import defaultdict
from typing import Optional, Dict, Any
from torch.utils.data import ConcatDataset
from nemo.collections import asr as nemo_asr
from nemo.utils.app_state import AppState
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector


# 파일 경로에서 speaker id 파싱: {spk}-{chapter}-{idx}.wav → spk
def _speaker_from_filepath(path: str) -> int:
    """
    path: ".../1743-142912-0035.wav" 형태의 파일 경로
    return: 1743 (int) 또는 파싱 실패 시 -1
    """
    if not path:
        return -1
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)  # "1743-142912-0035"
    parts = stem.split("-")
    if len(parts) >= 1 and parts[0].isdigit():
        return int(parts[0])
    return -1


def save_speaker_mapping(spk2idx, idx2spk, out_path: str):
    """
    spk2idx / idx2spk 매핑 정보를 JSON으로 저장.
    - spk2idx: {원래 speaker id(int) -> index(int)}
    - idx2spk: {index(int) -> 원래 speaker id(int)}
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {
        "num_speakers": len(spk2idx),
        # 원래 speaker id -> index
        "spk2idx": {str(spk): int(idx) for spk, idx in spk2idx.items()},
        # index -> 원래 speaker id
        "idx2spk": {str(idx): int(spk) for idx, spk in idx2spk.items()},
        # 이름 정보가 따로 없으니, 일단 placeholder 형태로 추가
        "speakers": [
            {
                "idx": int(idx),
                "orig_id": int(spk),
                "name": f"spk_{spk}",
            }
            for spk, idx in spk2idx.items()
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[SPEAKER MAP] saved id<->idx mapping to {out_path}")
    

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"): return True
    if v in ("no", "false","f", "n", "0"): return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def _clean_text(s: str) -> str:
    s = re_u.sub(r"\{.*?\}", "", s)
    s = s.replace("<sil>", " ")
    s = re_u.sub(r"\s+", " ", s).strip()
    return s.lower()


def scan_speakers(ds):
    spk_set = set()
    for ex in ds:
        spk_val = None

        # 1) 우선 메타 필드 쪽에서 찾기
        if "speaker_id" in ex:
            spk_val = int(ex["speaker_id"])
        elif "speaker" in ex and str(ex["speaker"]).isdigit():
            spk_val = int(ex["speaker"])
        else:
            # 2) 없으면 파일명에서 추출
            f = ex.get("file", None)
            audio = ex.get("audio", {})
            audio_path = audio.get("path", None)

            candidate = f or audio_path
            sid = _speaker_from_filepath(candidate) if candidate else -1
            if sid != -1:
                spk_val = sid

        if spk_val is not None:
            spk_set.add(spk_val)

    spk_list = sorted(list(spk_set))
    spk2idx = {spk: i for i, spk in enumerate(spk_list)}
    idx2spk = {i: spk for i, spk in enumerate(spk_list)}
    return spk2idx, idx2spk

# >>>>>>>>>>>>>>>>>>>>>>>>>>

# <<<<<<<<<<<<<<<<<<<<<<<<<< HF -> NeMo manifest
def build_manifest_from_hf_with_meta(ds, manifest_path: str, cache_dir: str,
                                     spk2idx=None):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp_audio_dir = os.path.join(cache_dir, "tmp_audio", "hf2wav")
    os.makedirs(tmp_audio_dir, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as fout:
        for ex in ds:
            audio = ex["audio"]; arr = audio.get("array", None)
            sr = audio.get("sampling_rate", 16000)
            candidates = []
            if isinstance(audio.get("path"), str): candidates.append(audio["path"])
            if isinstance(ex.get("file"), str):    candidates.append(ex["file"])
            wav_path = None
            for c in candidates:
                if c and os.path.isfile(c): wav_path = c; break
            if wav_path is None:
                if arr is None: continue
                base = os.path.splitext(os.path.basename(ex.get("file","")))[0] or str(ex.get("id","sample"))
                wav_path = os.path.join(tmp_audio_dir, base + ".wav")
                if os.path.exists(wav_path):
                    wav_path = os.path.join(tmp_audio_dir, base + f"_{uuid.uuid4().hex[:6]}.wav")
                sf.write(wav_path, arr, sr)

            if arr is not None:
                duration = float(len(arr))/float(sr)
            else:
                try:
                    info = sf.info(wav_path)
                    duration = float(info.frames)/float(info.samplerate) if info.samplerate else 0.0
                except:
                    duration = 0.0
            if duration <= 0: continue

            text = ex.get("text","").strip().upper()
            full_id = ex.get("id", os.path.splitext(os.path.basename(wav_path))[0])
            spk_id = ex.get("speaker_id")
            if spk_id in spk2idx:
                spk_idx = spk2idx[spk_id]
            else:
                spk_idx = -1 # unknown speaker (not in train set)

            item = {
                "audio_filepath": wav_path,
                "duration": duration,
                "text": text,
                "full_id": str(full_id),
                "spk_id": int(spk_id),
                "spk_idx": int(spk_idx),
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

def build_manifest_from_hf(ds, manifest_path: str, cache_dir: str, lang_key: str = "language", spk_key: str = "speaker"):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp_audio_dir = os.path.join(cache_dir, "tmp_audio", "dag_kd")
    os.makedirs(tmp_audio_dir, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as fout:
        for sample in ds:
            audio = sample.get("audio", {})
            arr = audio.get("array")
            sr  = audio.get("sampling_rate", 16000)

            # 후보 경로
            candidates = []
            p = audio.get("path")
            if isinstance(p, str) and p:
                candidates.append(p)
            f = sample.get("file")
            if isinstance(f, str) and f:
                candidates.append(f)

            orig_path = None
            for c in candidates:
                if c and os.path.isfile(c):
                    orig_path = c
                    break

            if orig_path is None:
                if arr is None:
                    # 스킵
                    continue
                base = None
                if isinstance(f, str) and f:
                    base = os.path.splitext(os.path.basename(f))[0]
                if not base:
                    base = str(sample.get("id", uuid.uuid4().hex))
                wav_path = os.path.join(tmp_audio_dir, base + ".wav")
                if os.path.exists(wav_path):
                    wav_path = os.path.join(tmp_audio_dir, base + f"_{uuid.uuid4().hex[:8]}.wav")
                sf.write(wav_path, arr, sr)
                orig_path = wav_path

            # duration
            if arr is not None:
                duration = float(len(arr)) / float(sr)
            else:
                try:
                    info = sf.info(orig_path)
                    duration = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
                except Exception:
                    duration = 0.0

            text = _clean_text(sample.get("text", ""))

            # optional labels
            lang = sample.get(lang_key, None)
            spk  = sample.get(spk_key,  None)

            # 파일명에서 speaker id 파싱
            spk_from_path = _speaker_from_filepath(orig_path)

            if duration > 0 and orig_path:
                entry = {"audio_filepath": orig_path, "duration": duration, "text": text}
                if lang is not None:
                    entry["lang"] = str(lang)

                # 우선순위: 파일명 기반 speaker → 샘플 내 speaker 필드
                if spk_from_path != -1:
                    entry["speaker"] = int(spk_from_path)
                elif spk is not None:
                    entry["speaker"] = str(spk)

                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

def build_manifest_from_hf_gigaspeech(ds, manifest_path: str, cache_dir: str):
    """
    HF Dataset -> NeMo manifest(JSONL)
      { "audio_filepath": ..., "duration": ..., "text": ... }
    - path 우선 사용, 안되면 cache/extracted 재귀탐색,
      그래도 안되면 bytes/array를 임시 파일로 저장
    """
    import io
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    extract_root  = os.path.join(cache_dir, "extracted")
    tmp_audio_dir = os.path.join(cache_dir, "tmp_manifest_audio")
    os.makedirs(tmp_audio_dir, exist_ok=True)

    # [ADDED] 레퍼런스 내 특수 태그 목록 (대소문자 무시)
    BANNED_TAGS = {
        "<MUSIC>", "<COMMA>", "<NOISE>", "<VOCALIZED_NOISE>", "<LAUGHTER>",
        "<SPOKEN_NOISE>", "<PERIOD>", "<QUESTION_MARK>", "<EXCLAMATION_MARK>",
        "<SEMICOLON>", "<COLON>", "<DASH>", "<ELLIPSIS>", "<SIL>", "<OTHER>"
    }
    # [ADDED] 태그 스트립용 정규식 (대소문자 무시)
    import re as _re
    _TAGS_RE = _re.compile(r"(?:%s)" % "|".join(_re.escape(t) for t in BANNED_TAGS), _re.IGNORECASE)

    # [ADDED] 태그만 제거하되, 제거 후 빈 문자열이면 "태그만 존재"로 판단
    def _strip_special_tags(text: str) -> tuple[str, bool]:
        """
        Returns: (tags_removed_text, is_tag_only)
        - 텍스트에서 BANNED_TAGS에 해당하는 토큰을 모두 제거
        - 제거 결과가 빈 문자열이면 '태그만 있는' 케이스로 간주
        """
        if not text:
            return "", True
        no_tags = _TAGS_RE.sub(" ", text)
        no_tags = re_u.sub(r"\s+", " ", no_tags).strip()
        is_tag_only = (len(no_tags) == 0)
        return no_tags, is_tag_only

    def _resolve_audio_path(sample) -> tuple[str, float]:
        """오디오 파일 실제 경로와 duration(sec) 반환"""
        audio = sample["audio"]
        # 1) path가 실제 파일이면 그대로 사용
        orig_path = audio.get("path", None)
        if isinstance(orig_path, str) and os.path.isfile(orig_path):
            # duration이 없으면 soundfile로 계산
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
                    # array가 있으면 fallback
                    arr = audio.get("array", None)
                    sr  = audio.get("sampling_rate", 16000)
                    dur = float(len(arr)) / float(sr) if arr is not None else 0.0
                return found, dur

        # 3) 원본 파일 바이트가 있으면 그대로 저장(확장자 추정)
        if audio.get("bytes", None) is not None:
            # 확장자 추정: path에서 따오거나 기본 .wav
            ext = ".wav"
            if isinstance(orig_path, str) and "." in os.path.basename(orig_path):
                ext = os.path.splitext(orig_path)[1] or ".wav"
            out_path = os.path.join(tmp_audio_dir, f"hf_{uuid.uuid4().hex}{ext}")
            with open(out_path, "wb") as f:
                f.write(audio["bytes"])
            # duration 계산 시도
            try:
                info = sf.info(out_path)
                dur = float(info.frames) / float(info.samplerate)
            except Exception:
                # array가 있으면 fallback
                arr = audio.get("array", None)
                sr  = audio.get("sampling_rate", 16000)
                dur = float(len(arr)) / float(sr) if arr is not None else 0.0
            return out_path, dur

        # 4) 마지막 수단: array+sr로 WAV 저장
        arr = audio.get("array", None)
        sr  = audio.get("sampling_rate", 16000)
        if arr is not None:
            out_path = os.path.join(tmp_audio_dir, f"hf_{uuid.uuid4().hex}.wav")
            sf.write(out_path, arr, sr)
            dur = float(len(arr)) / float(sr)
            return out_path, dur

        raise FileNotFoundError("오디오 경로/바이트/배열 중 어느 것도 사용할 수 없습니다.")

    n_written, n_skipped_short, n_skipped_tagonly = 0, 0, 0  # [CHANGED] 카운터 추가
    min_sec = 1.0
    with open(manifest_path, "w", encoding="utf-8") as fout:
        for sample in ds:
            audio_path, duration = _resolve_audio_path(sample)

            # 너무 짧은 샘플 스킵
            if duration < min_sec:
                n_skipped_short += 1
                continue

            # GigaSpeech는 보통 'text', 일부 스크립트는 'sentence'
            raw_text = sample.get("sentence", None)
            if raw_text is None:
                raw_text = sample.get("text", "")

            # [CHANGED] "태그가 있으면 스킵" → "태그는 제거만, 태그만 있으면 스킵"
            tag_stripped, is_tag_only = _strip_special_tags(raw_text)
            if is_tag_only:
                n_skipped_tagonly += 1
                continue

            # 이후 일반 정규화
            text = normalize_text_cv(tag_stripped, keep_punct=False)

            fout.write(json.dumps({
                "audio_filepath": audio_path,
                "duration": float(duration),
                "text": text,
            }, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[manifest] wrote {n_written} lines "
          f"(skipped {n_skipped_short} < {min_sec}s, "
          f"skipped {n_skipped_tagonly} tag-only refs) → {manifest_path}")
# >>>>>>>>>>>>>>>>>>>>>>>>>>

# <<<<<<<<<<<<<<<<<<<<<<<<<< NeMo .nemo unpack helper
# def release_nemoAPI(model, out_folder: str):
#     meta = AppState().get_model_metadata_from_guid(model.model_guid)
#     nemo_file = meta.restoration_path
#     connector = SaveRestoreConnector()
#     connector._unpack_nemo_file(nemo_file, out_folder=out_folder)
#     model._save_restore_connector.model_extracted_dir = out_folder
#     AppState().nemo_file_folder = out_folder

# def release_nemoAPI(teacher_model, out_folder: str = "/workspace/outputs/nemo_archive"):
#     # 1) .nemo 실제 경로 조회
#     meta = AppState().get_model_metadata_from_guid(teacher_model.model_guid)
#     nemo_file = meta.restoration_path
#     os.makedirs(out_folder, exist_ok=True)

#     # 2) 압축 풀기
#     connector = SaveRestoreConnector()
#     connector._unpack_nemo_file(nemo_file, out_folder=out_folder)

#     # 3) 다음 복원 때 재활용할 디렉토리 지정
#     teacher_model._save_restore_connector.model_extracted_dir = out_folder
#     AppState().nemo_file_folder = out_folder

def release_nemoAPI(model=None, out_folder: str = "/workspace/outputs/nemo_archive", nemo_file: str | None = None):
    os.makedirs(out_folder, exist_ok=True)

    # (A) out_folder에 이미 풀려있는지 체크
    already_extracted = (
        os.path.isfile(os.path.join(out_folder, "model_config.yaml")) and
        os.path.isfile(os.path.join(out_folder, "model_weights.ckpt"))
    )

    # (B) nemo_file이 없으면 model_guid에서 restoration_path를 시도
    if nemo_file is None and model is not None and hasattr(model, "model_guid"):
        try:
            meta = AppState().get_model_metadata_from_guid(model.model_guid)
            nemo_file = meta.restoration_path
        except Exception:
            nemo_file = None

    # (C) 필요하면 압축 풀기 (restoration_path가 있을 때만)
    if (not already_extracted) and (nemo_file is not None):
        connector = SaveRestoreConnector()
        connector._unpack_nemo_file(nemo_file, out_folder=out_folder)

    # (D) 핵심: nemo: artifact 해석용 base dir 세팅
    AppState().nemo_file_folder = out_folder

    # (E) 선택: model이 있으면 extracted_dir도 잡아주기 (restore_from 최적화용)
    if model is not None and hasattr(model, "_save_restore_connector"):
        model._save_restore_connector.model_extracted_dir = out_folder


def _strip_module_prefix(sd: dict) -> dict:
    # DataParallel 등에 의해 앞에 'module.' 있을 때 제거
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def _extract_gst_subdict(full_state: dict) -> dict:
    """
    ckpt의 'gst.encoder.*', 'gst.stl.*' 키만 뽑아서
    현재 클래스 구조('encoder.*', 'stl.*')에 맞게 prefix를 제거해 리턴
    """
    sub = {}
    for k, v in full_state.items():
        if k.startswith("gst.encoder."):
            sub[k.replace("gst.", "", 1)] = v        # -> 'encoder.xxx'
        elif k.startswith("gst.stl."):
            sub[k.replace("gst.", "", 1)] = v        # -> 'stl.xxx'
    return sub

'''
def load_gst_pretrained(gst_module: GST, ckpt_path: str, strict: bool = True):
    """
    gst_module: 위에 정의한 GST 클래스 인스턴스
    ckpt_path : '/workspace/DAG-KD/pretrained_weights/GST Tacotron Epoch 100.pt'
    """
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("state_dict", raw)
    state = _strip_module_prefix(state)
    sub = _extract_gst_subdict(state)

    # 차원 호환성 체크 (GRU hidden = hp.E//2)
    # encoder.gru.weight_hh_l0 shape 확인해서 hidden_dim 유추
    w_hh = sub.get("encoder.gru.weight_hh_l0", None)
    if w_hh is not None:
        hidden_ckpt = w_hh.shape[1] // 3  # GRU weight_hh_l0의 shape(3*hidden, hidden)
        hidden_cfg  = getattr(gst_module.encoder.gru, "hidden_size", None)
        if hidden_cfg is not None and hidden_ckpt != hidden_cfg:
            raise RuntimeError(
                f"[GST] GRU hidden mismatch: ckpt={hidden_ckpt}, code(hp.E//2)={hidden_cfg}. "
                f"hp.E 값을 {hidden_ckpt*2} 로 맞춰주세요."
            )

    missing, unexpected = gst_module.load_state_dict(sub, strict=strict)
    # torch 2.4: load_state_dict return 값이 None이므로 직접 체크
    # strict=True면 mismatch 시 예외가 나므로 위 라인은 통과되면 정상.
    print("[GST] pretrained loaded from:", ckpt_path)
    if not strict:
        print("[GST] (strict=False) load finished. Check missing/unexpected keys manually if needed.")
'''



def _edit_distance(ref_words, hyp_words):
    """
    단어 단위 edit distance (Levenshtein)
    ref_words, hyp_words: list[str]
    """
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]

def compute_sample_wers(ref_texts, hyp_texts):
    """
    ref_texts, hyp_texts: list[str]
    각 샘플별 WER 리스트 반환
    """
    wers = []
    for ref, hyp in zip(ref_texts, hyp_texts):
        # hyp이 Hypothesis 객체인 경우 text 필드로 변환
        if not isinstance(hyp, str):
            if hasattr(hyp, "text"):
                hyp = hyp.text
            elif hasattr(hyp, "transcript"):
                hyp = hyp.transcript
            else:
                hyp = str(hyp)
        
        ref = ref.lower().strip()
        hyp = hyp.lower().strip()
        
        ref_words = ref.split()
        hyp_words = hyp.split()
        
        if len(ref_words) == 0:
            wers.append(0.0)
            continue
        dist = _edit_distance(ref_words, hyp_words)
        wers.append(dist / len(ref_words))
    return wers

def compute_speaker_durations(manifest_path: str, out_path: str = None):
    """
    manifest(jsonl)에서 speaker별 duration 합계 계산 후
    - 콘솔에 간단 요약 출력
    - out_path가 주어지면 JSON 파일로 저장
    """
    spk_dur = defaultdict(float)

    if not os.path.isfile(manifest_path):
        print(f"[SPEAKER DUR] manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            spk = int(obj.get("speaker", -1))
            dur = float(obj.get("duration", 0.0))
            if spk < 0:
                continue
            spk_dur[spk] += dur

    print(f"[SPEAKER DUR] from {manifest_path}")
    print(f"  num_speakers_with_dur = {len(spk_dur)}")

    # duration이 큰 순서로 상위 몇 개만 출력 (원하면 개수 조절)
    for spk, dur in sorted(spk_dur.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  speaker={spk:4d} total_dur={dur/3600:.2f} h ({dur:.1f} s)")

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "speaker_durations_sec": {str(k): v for k, v in spk_dur.items()}
        }
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(payload, fout, ensure_ascii=False, indent=2)
        print(f"[SPEAKER DUR] saved to {out_path}")

def int_list_arg(s):
    # 예: "1,2,3" → [1,2,3]
    return [int(x) for x in s.split(",") if x.strip()]

def save_mel_examples_from_manifest(
    manifest_path: str,
    model: nemo_asr.models.EncDecCTCModelBPE,
    out_dir: str,
    num_examples: int = 4,
    split_name: str = "train",
):
    """
    manifest에 적힌 audio 몇 개를 로드해서
    model.preprocessor로 멜 스펙트로그램을 만든 뒤 PNG로 저장.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 샘플레이트: model cfg 또는 preprocessor에서 가져오기
    # (둘 다 결국 16000일 것)
    if hasattr(model, "cfg") and hasattr(model.cfg, "sample_rate"):
        target_sr = model.cfg.sample_rate
    else:
        # NeMo의 AudioToMelSpectrogramPreprocessor는 _sample_rate를 갖고 있음
        target_sr = getattr(getattr(model, "preprocessor", object()), "_sample_rate", 16000)

    # manifest 읽기
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    # 앞에서부터 num_examples개만 사용
    for idx, e in enumerate(entries[:num_examples]):
        wav_path = e["audio_filepath"]
        text = e.get("text", "")

        # 1) 오디오 로드 (soundfile)
        audio, sr = sf.read(wav_path)  # audio: (T,) or (T, C)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono

        # 2) 샘플레이트 맞추기
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # 3) torch tensor로 변환
        signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # (1, T)
        length = torch.tensor([signal.shape[1]], dtype=torch.int64)    # (1,)

        # 4) 멜 스펙트로그램 추출
        model.preprocessor.eval()
        with torch.no_grad():
            mel, mel_len = model.preprocessor(
                input_signal=signal,
                length=length,
            )  # mel: (1, n_mels, T_mel)

        mel_np = mel[0].cpu().numpy()  # (n_mels, T_mel)

        # 5) 그림으로 저장
        plt.figure(figsize=(10,5))
        plt.imshow(mel_np, origin="lower", aspect="auto")
        plt.xlabel("Time frames")
        plt.ylabel("Mel bins")
        plt.colorbar()
        plt.title(f"{split_name} example {idx}\n{text[:80]}")  # 텍스트 앞 부분만
        plt.tight_layout()

        save_path = os.path.join(out_dir, f"mel_{split_name}_{idx}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"[MEL] Saved mel example: {save_path}")

def materialize_nemo_artifacts_in_cfg(cfg, nemo_archive_dir: str):
    nemo_archive_dir = os.path.abspath(nemo_archive_dir)
    OmegaConf.set_struct(cfg, False)

    # tokenizer 쪽 nemo: 경로를 로컬 절대경로로 치환
    if "tokenizer" in cfg and cfg.tokenizer is not None:
        for k in ["model_path", "vocab_path", "spe_tokenizer_vocab"]:
            if k in cfg.tokenizer and isinstance(cfg.tokenizer[k], str) and cfg.tokenizer[k].startswith("nemo:"):
                rel = cfg.tokenizer[k][5:]  # remove "nemo:"
                abs_path = os.path.join(nemo_archive_dir, rel)
                cfg.tokenizer[k] = os.path.abspath(abs_path)

    # 안전 체크: 파일 존재 확인
    tok = cfg.tokenizer
    for k in ["model_path", "vocab_path", "spe_tokenizer_vocab"]:
        p = tok.get(k, None)
        if p is not None and isinstance(p, str):
            if not os.path.exists(p):
                raise FileNotFoundError(f"[materialize] tokenizer.{k} not found: {p}")

    return cfg

def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    lengths: (B,)
    return: mask (B, max_len, 1) where valid=1, pad=0
    """
    device = lengths.device
    B = lengths.size(0)
    arange = torch.arange(max_len, device=device).unsqueeze(0).expand(B, max_len)
    mask = (arange < lengths.unsqueeze(1)).float().unsqueeze(-1)
    return mask

def masked_mse(a: torch.Tensor, b: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    a,b: (B,T,D)
    lengths: (B,) optional
    """
    diff2 = (a - b).pow(2)
    if lengths is None:
        return diff2.mean()
    mask = make_pad_mask(lengths, a.size(1))  # (B,T,1)
    diff2 = diff2 * mask
    denom = mask.sum() * a.size(-1) + 1e-8
    return diff2.sum() / denom

def _to_long(x, device):
    if torch.is_tensor(x):
        return x.to(device).long()
    return torch.as_tensor(x, device=device, dtype=torch.long)

def _get_item_by_global_index(ds, idx: int):
    if isinstance(ds, ConcatDataset):
        for sub_i, csz in enumerate(ds.cumulative_sizes):
            if idx < csz:
                prev = 0 if sub_i == 0 else ds.cumulative_sizes[sub_i - 1]
                return ds.datasets[sub_i][idx - prev]
        raise IndexError(idx)
    return ds[idx]

def extract_speaker_ids_from_batch(batch, spk_table: torch.Tensor, device):
    # batch: (signal, signal_len, tokens, tokens_len, sample_id)
    if not (isinstance(batch, (tuple, list)) and len(batch) >= 5):
        return None
    sample_id = batch[4]
    if not torch.is_tensor(sample_id) or sample_id.dim() != 1:
        return None

    sample_id = sample_id.to(device).long()
    spk_table = spk_table.to(device, non_blocking=True)
    return spk_table[sample_id]  # (B,)

def load_speaker_table_from_manifest(manifest_path: str, key_candidates=("spk_idx", "speaker", "spk_id")):
    table = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            v = -1
            for k in key_candidates:
                if k in j:
                    v = int(j[k])
                    break
            table.append(v)
    # CPU tensor로 들고 있다가, 필요할 때 device로 옮겨서 take/indexing
    return torch.tensor(table, dtype=torch.long)  # (N,)

def rotate_last_ckpts(ckpt_dir: str, keep: int = 50):
    p = Path(ckpt_dir)
    ckpts = sorted(p.glob("last*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not ckpts:
        return

    latest = ckpts[0]
    last = p / "last.ckpt"

    # 1) 최신을 last.ckpt로 만들기
    if latest != last:
        if last.exists():
            # 기존 last는 백업으로 이동
            backup = p / f"last-prev.ckpt"
            if backup.exists():
                backup.unlink()
            last.rename(backup)
        latest.rename(last)

    # 2) 나머지 오래된 애들은 last-v1, last-v2 ...로 정렬 (원하면)
    others = sorted([c for c in p.glob("last*.ckpt") if c.name != "last.ckpt"],
                    key=lambda x: x.stat().st_mtime, reverse=True)

    for i, f in enumerate(others, start=1):
        target = p / f"last-v{i}.ckpt"
        if f != target:
            if target.exists():
                target.unlink()
            f.rename(target)

    # 3) 너무 많으면 prune
    all_ckpts = sorted(p.glob("last*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
    for f in all_ckpts[keep:]:
        f.unlink()

def normalize_text_cv(s: str, keep_punct: bool = False) -> str:
    # 0) 유니코드 정규화 + 소문자
    s = unicodedata.normalize("NFKC", s or "").strip().lower()

    # 1) 흔한 특수문자 매핑 및 제거
    for k, v in {"\u2047": " ","“": '"', "”": '"', "„": '"',"‘": "'", "’": "'","–": "-", "—": "-","…": " ", "‹": " ", "›": " ", "«": " ", "»": " ",}.items():   # DOUBLE QUESTION MARK → 제거(공백)
        s = s.replace(k, v)

    # 2) 바깥 큰따옴표만 한 쌍이면 제거
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]

    # 3) CV 특유의 공백+아포스트로피 정리: "men 's" → "men's"
    s = re_u.sub(r"\s+'\s*s\b", "'s", s)

    # 4) 평가용: 구두점 제거 권장(문자/숫자/공백/아포스트로피/하이픈만 유지)
    if not keep_punct:
        s = re_u.sub(r"[^\p{L}\p{N}\s'\-]", " ", s)

    # 5) 공백 정리
    s = re_u.sub(r"\s+", " ", s).strip()
    return s

def head_manifest(src_path: str, dst_path: str, n_lines: int) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    wrote = 0
    with open(src_path, "r", encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            wrote += 1
            if wrote >= n_lines:
                break
    return dst_path, wrote

def ensure_BCT(self, x, C_expected=None):
    # x: (B,T,C) or (B,C,T) -> (B,C,T)
    if x.dim() != 3:
        raise ValueError("expected 3D")
    if C_expected is not None:
        if x.size(1) == C_expected:      # (B,C,T)
            return x
        if x.size(2) == C_expected:      # (B,T,C)
            return x.transpose(1, 2)
    # 기대 채널을 모르겠으면 마지막 축이 채널일 가능성이 높다고 가정
    # (NeMo encoder output은 보통 마지막이 C)
    return x.transpose(1, 2) if x.size(1) > x.size(2) else x

def make_pad_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """
    lengths: (B,) int
    return: (B, T) bool
    """
    device = lengths.device
    ar = torch.arange(T, device=device)[None, :]
    return ar < lengths[:, None]


def masked_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    pred,tgt: (B,T) or (B,C,T)
    mask:     (B,T) or (B,1,T)
    """
    loss = (pred - tgt).abs()
    loss = loss * mask
    denom = mask.sum().clamp_min(eps)
    return loss.sum() / denom


def masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, eps=1e-8) -> torch.Tensor:
    loss = (pred - tgt) ** 2
    loss = loss * mask
    denom = mask.sum().clamp_min(eps)
    return loss.sum() / denom

def compute_pitch_features_librosa(
    signal: torch.Tensor,
    sample_rate: int = 16000,
    frame_shift_ms: float = 10.0, # HOP_MS
    frame_length_ms: float = 25.0, # WIN_MS
    fmin: float = 50.0,
    fmax: float = 400.0,
    device=None,
):
    """
    Returns: Tensor [num_frames, 2] where
      - [:,0] = f0 in Hz (0 if unvoiced)
      - [:,1] = voiced_prob in [0,1]
    """
    import librosa

    device = device or signal.device

    # signal: (T,) float tensor
    y = signal.detach().float().cpu().numpy()

    hop_length = int(round(sample_rate * frame_shift_ms / 1000.0))      # e.g. 160
    win_length = int(round(sample_rate * frame_length_ms / 1000.0))     # e.g. 400

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=win_length,
        hop_length=hop_length,
        center=True,
    )

    # f0: (num_frames,) with np.nan for unvoiced
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0).astype(np.float32)

    feats = np.stack([f0, voiced_prob], axis=-1)  # [N,2]
    return torch.from_numpy(feats).to(device)

def compute_phys_for_wav(wav_path: str, SR=16000, HOP_MS=10.0, WIN_MS=25.0) -> Dict[str, np.ndarray]:
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        # 여기서는 속도/의존성 최소화를 위해 resample을 생략. SR mismatch면 경고 후 그대로 진행.
        # 실제 실험에선 librosa.resample이나 torchaudio resample 권장.
        print(f"[WARN] sample_rate mismatch: {wav_path} sr={sr} (expected {SR})")

    # ===== F0/VUV (librosa 기반) =====
    # utils.compute_pitch_features_librosa(wav, ...) 가 (T,2) [vuv_prob, f0] 형태라고 가정
    # (너 코드 기준: pitch_features[...,0]=voicing, [...,1]=f0)
    pitch = compute_pitch_features_librosa(
        torch.from_numpy(wav).unsqueeze(0),   # (1, N) torch CPU
        sample_rate=SR,
        frame_shift_ms=HOP_MS,
        frame_length_ms=WIN_MS,
        device="cpu",
    ).squeeze(0).cpu().numpy()  # (T_pitch, 2)

    vuv = pitch[:, 0].astype(np.float32)  # (T,)
    f0  = pitch[:, 1].astype(np.float32)  # (T,)

    # ===== Energy =====
    # mel을 여기서 다시 만들려면 비용이 큼. 대신 waveform 기반 energy proxy로도 OK.
    # hop/window 기반으로 프레임 에너지 계산(간단 RMS)
    hop = int(SR * (HOP_MS / 1000.0))
    win = int(SR * (WIN_MS / 1000.0))
    T = int(np.ceil(len(wav) / hop))
    energy = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        s = t * hop
        e = min(len(wav), s + win)
        frame = wav[s:e]
        if frame.size == 0:
            energy[t] = 0.0
        else:
            energy[t] = float(np.sqrt(np.mean(frame * frame) + 1e-8))

    # f0/vuv 길이와 energy 길이 맞추기(보간)
    if len(f0) != len(energy):
        x_src = np.linspace(0, 1, num=len(f0), dtype=np.float32)
        x_tgt = np.linspace(0, 1, num=len(energy), dtype=np.float32)
        f0  = np.interp(x_tgt, x_src, f0).astype(np.float32)
        vuv = np.interp(x_tgt, x_src, vuv).astype(np.float32)

    return f0, vuv, energy

def build_phys_cache_for_manifest(
    manifest_path: str,
    split_name: str,
    max_items=None,
    phys_cache_root: Path = None,
    HOP_MS: float = 10.0,
    WIN_MS: float = 25.0,
    SR: int = 16000,
):
    """
    저장 포맷: <phys_cache_root>/<split_name>/<manifest_id>.npy
    manifest_id = manifest line index + 1 (NeMo item['id']와 동일 컨벤션)
    배열 shape: (3, T)  [f0, energy, vuv]
    dtype: float16 (IO 감소)
    """
    phys_cache_root = Path(phys_cache_root)
    out_dir = phys_cache_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 총 라인 수(진행률 정확히)
    with open(manifest_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    if max_items is not None:
        total = min(total, int(max_items))

    n_new, n_skip, n_fail = 0, 0, 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        pbar = tqdm(
            enumerate(f),
            total=total,
            desc=f"[phys-cache] {split_name}",
            unit="utt",
            dynamic_ncols=True,
        )

        for line_idx, line in pbar:
            if max_items is not None and line_idx >= max_items:
                break

            manifest_id = line_idx + 1
            out_path = out_dir / f"{manifest_id}.npy"
            if out_path.exists():
                n_skip += 1
                pbar.set_postfix(new=n_new, skip=n_skip, fail=n_fail)
                continue

            try:
                obj = json.loads(line)
                wav_path = obj["audio_filepath"]

                # 반환: f0 (T,), vuv (T,), energy (T,)
                phys_dict = compute_phys_for_wav(
                    wav_path,
                    SR=SR,
                    HOP_MS=HOP_MS,
                    WIN_MS=WIN_MS,
                )
                f0 = phys_dict["f0"]
                vuv = phys_dict["vuv"]
                energy = phys_dict["energy"]

                phys = np.stack([f0, energy, vuv], axis=0).astype(np.float16)  # (3,T)
                np.save(out_path, phys)

                n_new += 1
                pbar.set_postfix(new=n_new, skip=n_skip, fail=n_fail)
            except Exception as e:
                n_fail += 1
                # 너무 길어지지 않게 예외 메시지 짧게
                pbar.set_postfix(new=n_new, skip=n_skip, fail=n_fail, err=str(e)[:60])

    print(f"[phys-cache] {split_name}: created={n_new}, skipped={n_skip}, failed={n_fail}, out_dir={out_dir}")