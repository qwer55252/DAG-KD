import re, os, json, uuid, torch, argparse, librosa
import soundfile as sf
import matplotlib.pyplot as plt
from collections import defaultdict
from nemo.utils.app_state import AppState
from nemo.collections import asr as nemo_asr
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
    s = re.sub(r"\{.*?\}", "", s)
    s = s.replace("<sil>", " ")
    s = re.sub(r"\s+", " ", s).strip()
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
            sample_id = ex.get("id", os.path.splitext(os.path.basename(wav_path))[0])

            # 1) 파일명에서 speaker id 파싱
            spk_from_path = _speaker_from_filepath(wav_path)

            # 2) 필요하면 meta에서도 한번 더 보되, 파일명 파싱이 우선
            if spk_from_path != -1:
                spk_raw = spk_from_path
            else:
                spk_raw = ex.get("speaker_id", ex.get("speaker", -1))

            # 3) spk2idx가 있으면 index로 매핑 (0 ~ num_spk-1), 없으면 값 그대로 쓰거나 -1
            if spk2idx and isinstance(spk_raw, int) and spk_raw in spk2idx:
                spk = spk2idx[spk_raw]
            elif spk2idx and isinstance(spk_raw, str) and spk_raw.isdigit() and int(spk_raw) in spk2idx:
                spk = spk2idx[int(spk_raw)]
            else:
                # spk2idx가 없거나, 매칭 안 되면 원래 speaker id 그대로 넣거나 -1
                spk = spk_raw if isinstance(spk_raw, int) and spk_raw >= 0 else -1

            item = {
                "audio_filepath": wav_path,
                "duration": duration,
                "text": text,
                "id": str(sample_id),
                "speaker": int(spk),
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
# >>>>>>>>>>>>>>>>>>>>>>>>>>

# <<<<<<<<<<<<<<<<<<<<<<<<<< NeMo .nemo unpack helper
# def release_nemoAPI(model, out_folder: str):
#     meta = AppState().get_model_metadata_from_guid(model.model_guid)
#     nemo_file = meta.restoration_path
#     connector = SaveRestoreConnector()
#     connector._unpack_nemo_file(nemo_file, out_folder=out_folder)
#     model._save_restore_connector.model_extracted_dir = out_folder
#     AppState().nemo_file_folder = out_folder

def release_nemoAPI(teacher_model, out_folder: str = "/workspace/outputs/nemo_archive"):
    # 1) .nemo 실제 경로 조회
    meta = AppState().get_model_metadata_from_guid(teacher_model.model_guid)
    nemo_file = meta.restoration_path
    os.makedirs(out_folder, exist_ok=True)

    # 2) 압축 풀기
    connector = SaveRestoreConnector()
    connector._unpack_nemo_file(nemo_file, out_folder=out_folder)

    # 3) 다음 복원 때 재활용할 디렉토리 지정
    teacher_model._save_restore_connector.model_extracted_dir = out_folder
    AppState().nemo_file_folder = out_folder

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

