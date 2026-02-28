import os
import math
import json
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
import nemo.collections.asr as nemo_asr
from collections import OrderedDict
from utils import (ensure_BCT)

class DiffKDModule(nn.Module):
    """간단한 Linear AE + 1D CNN denoiser 버전"""
    def __init__(self, teacher_dim, student_dim, latent_dim=None, steps=5):
        super().__init__()
        self.steps = steps
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.latent_dim = latent_dim or min(teacher_dim, student_dim)
        self.enc = nn.Conv1d(teacher_dim, self.latent_dim, 1)
        self.dec = nn.Conv1d(self.latent_dim, teacher_dim, 1)
        self.proj= nn.Conv1d(student_dim, self.latent_dim, 1)
        self.denoiser = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.latent_dim, self.latent_dim, 3, padding=1),
        )
        self.mse = nn.MSELoss()
    
    def forward(self, stu_feat, tch_feat):
        stu_feat = self._to_BCT(stu_feat, self.student_dim)
        tch_feat = self._to_BCT(tch_feat, self.teacher_dim)
        # stu_feat/tch_feat: (B, C, T)
        z_t = self.enc(tch_feat)
        ae = self.mse(self.dec(z_t), tch_feat)
        z_s = self.proj(stu_feat)
        x = z_s
        for _ in range(self.steps):
            noise = self.denoiser(x)
            x = x - noise / self.steps
        distill = self.mse(x, z_t)
        return ae + distill
    
    def _to_BCT(self, x, C_expected):
        if x.size(1) == C_expected: return x
        if x.size(2) == C_expected: return x.transpose(1,2)
        return x  # 마지막 fallback

class FlowMatchingModule(nn.Module):
    """간단화된 Feature-space Flow Matching (Rectified schedule)"""
    def __init__(self, feat_dim_s: int, feat_dim_t: int, hidden=128, time_dim=32, steps=8, loss_weight=1.0):
        super().__init__()
        self.feat_dim_s = feat_dim_s
        self.feat_dim_t = feat_dim_t
        self.steps = steps
        self.loss_weight = loss_weight
        self.time = nn.Linear(1, time_dim)
        self.net = nn.Sequential(
            nn.Linear(feat_dim_s + time_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim_s),
        )
        self.shape = nn.Linear(feat_dim_s, feat_dim_t)  # s-shape -> t-shape
        self.mse = nn.MSELoss()
    
    def forward(self, stu_feat, tch_feat):
        # (B,C,T) -> (B,T,C)
        s = self._to_BTC(stu_feat, self.feat_dim_s)
        t = self._to_BTC(tch_feat, self.feat_dim_t).detach()
        x = s
        for i in range(self.steps, 0, -1):
            tt = torch.full((x.size(0), x.size(1), 1), i / self.steps, device=x.device)
            te = self.time(tt)
            h = torch.cat([x, te], dim=-1)
            v = self.net(h)
            x = x - v / self.steps
        # match teacher (shape-align)
        pred = self.shape(x)
        loss = self.mse(pred, t)
        return self.loss_weight * loss
    
    def _to_BTC(self, x, feat_dim):
        # x: (B,C,T) or (B,T,C)  -> (B,T,C)
        if x.dim() != 3:
            raise ValueError("expected 3D tensor")
        if x.size(-1) == feat_dim:       # already (B,T,C)
            return x
        if x.size(1) == feat_dim:        # (B,C,T)
            return x.transpose(1, 2)
        # fallback(애매하면 last dim이 time일 가능성이 큼): (B,C,T)로 간주
        return x.transpose(1, 2)

class DistilDAGKDCTCModelBPE(nemo_asr.models.EncDecCTCModelBPE):
    def __init__(
        self,
        cfg,
        trainer,
        teacher_model,
        # KD
        use_ctc: bool = True,
        use_logit_kd: bool = True,
        kd_alpha: float = 0.5,
        kd_temperature: float = 1.0,
        # Layerwise metric KD (옵션: 필요 없으면 끄기)
        use_layer_kd: bool = False,
        layer_kd_alpha: float = 0.5,
        # Generative KD
        use_flow: bool = False,
        flow_steps: int = 8,
        flow_weight: float = 1.0,
        use_diffkd: bool = False,
        diffkd_steps: int = 5,
        # Disentanglement (기존 GRL 기반은 사용 X, MI 기반 factorization만 사용)
        use_disent: bool = False,
        disent_spk_layers: list = [1,2],
        disent_txt_layers: list = [15,16],
        
        # 기타
    ):
        super().__init__(cfg=cfg, trainer=trainer)
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.use_ctc = use_ctc
        self.use_logit_kd = use_logit_kd
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self.use_layer_kd = use_layer_kd
        self.layer_kd_alpha = layer_kd_alpha
        self.use_flow = use_flow
        self.use_diffkd = use_diffkd
        self.use_disent = use_disent
        self.disent_spk_layers = disent_spk_layers
        self.disent_txt_layers = disent_txt_layers
        self.latent_dim = cfg.latent_dim # 96

        # --- Feature capture (hook) ---
        self.stu_feats = []
        self.tch_feats = []
        for lyr in self.encoder.layers:
            lyr.register_forward_hook(self._cap_stu)
        for lyr in self.teacher.encoder.layers:
            lyr.register_forward_hook(self._cap_tch)

        # 차원
        self.dim_s = cfg.encoder.d_model
        self.dim_t = teacher_model.cfg.encoder.d_model
        self.latent_dim = cfg.latent_dim
        
        # Projection for metric KD (student->teacher) - 옵션
        self.stu_to_tea_proj = nn.Conv1d(self.dim_s, self.dim_t, kernel_size=1, bias=True)

        # Text Encoder (Conv1x1 → Conv1x1, Rec loss)
        self.txt_enc = nn.Conv1d(self.dim_t, self.latent_dim, kernel_size=1) # (B, 96, T)
        self.txt_dec = nn.Conv1d(self.latent_dim, self.dim_t, kernel_size=1) # (B, 96, T)

        # Speaker Encoder (Conv1x1 → Conv1x1, Rec loss)
        self.spk_enc = nn.Conv1d(self.dim_t, self.latent_dim, kernel_size=1) # (B, 96, T)
        self.spk_dec = nn.Conv1d(self.latent_dim, self.dim_t, kernel_size=1) # (B, 96, T)

        # Speaker classifier (파란 박스 - teacher speaker embedding CE)
        self.num_spk = getattr(cfg, "num_spk", 0)
        if self.num_spk > 1:
            # TDNN-like speaker backbone: (B, latent_dim, T) -> (B, latent_dim, T)
            # TODO: spk_backbone 사이즈가 너무 큰게 아닌지?
            self.spk_backbone = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, self.latent_dim),
                nn.ReLU(inplace=True),

                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, dilation=2, padding=2, bias=False),
                nn.GroupNorm(8, self.latent_dim),
                nn.ReLU(inplace=True),

                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, dilation=3, padding=3, bias=False),
                nn.GroupNorm(8, self.latent_dim),
                nn.ReLU(inplace=True),
            )

            # stats pooling: mean + std → 2 * latent_dim
            spk_cls_hidden = getattr(cfg, "spk_cls_hidden", self.latent_dim * 2)

            self.spk_cls = nn.Sequential(
                nn.Linear(self.latent_dim * 2, spk_cls_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=getattr(cfg, "spk_cls_dropout", 0.3)),
                nn.Linear(spk_cls_hidden, self.num_spk),
            )
        else:
            self.spk_backbone = None
            self.spk_cls = None
        self.disen_spk_ce_lambda = getattr(cfg, "disen_spk_ce_lambda", 1.0)

        # Reconstruction loss weight
        self.rec_txt_lambda = getattr(cfg, "rec_txt_lambda", 1.0)
        self.rec_spk_lambda = getattr(cfg, "rec_spk_lambda", 1.0)
        self.rec_pros_lambda = getattr(cfg, "rec_pros_lambda", 1.0)

        # ===== Prosody (GST) =====
        n_mels = getattr(getattr(cfg, "preprocessor", type("x", (object,), {})()), "features", 80)
        self.pros_ref = GlobalProsodyReferenceEncoder(
            n_mels=n_mels, channels=(32, 64, 128), gru_dim=96
        )
        self.pros_gst = GlobalStyleTokenLayer(
            num_tokens=getattr(cfg, "disen_gst_tokens", 10),
            token_dim=getattr(cfg, "disen_gst_token_dim", 96),
            num_heads=getattr(cfg, "disen_gst_heads", 4),
            ref_dim=getattr(cfg, "disen_ref_dim", 96),
        )
        self.pros_proj = nn.Linear(getattr(cfg, "disen_gst_token_dim", 96), self.latent_dim)
        
        # ===== Prosody Supervision Heads =====
        # 1. Mel Reconstruction Decoder (pros_emb -> mel_spectrogram)
        # 매우 가벼운 구조 (파라미터 수 최소화)
        self.mel_dec = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.latent_dim, n_mels, kernel_size=1)
        )
        
        # 2. Physical Quantity Predictor (F0, energy, voicing, duration)
        # 4개의 물리량을 예측하는 가벼운 컨볼루션 헤드
        self.prosody_predictor = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.latent_dim, 3, kernel_size=1) 
        )
        # --- phys cache 옵션 ---
        self.phys_cache_ext = getattr(cfg, "phys_cache_ext", ".npy")  # ".npy" 권장
        self.phys_cache_lru = int(getattr(cfg, "phys_cache_lru", 2048))  # 최근 2048개 샘플 캐싱
        self._phys_lru = OrderedDict()  # (split, manifest_id) -> np.ndarray(3,T) or torch.Tensor
        
        # ===== MI 추정기 (vCLUB) =====
        self.mi_weight = getattr(cfg, "disen_mi_weight", 1.0)        # λ_MI
        self.mi_pairs = getattr(cfg, "disen_mi_pairs", "ts,tp,ps")   # 사용 쌍
        # self.club_tp = ClubGaussian(u_dim=self.latent_dim, v_dim=self.latent_dim)
        self.club_ts = ARClubGaussian(u_dim=self.latent_dim, v_dim=self.latent_dim, hidden=getattr(cfg, "club_hidden", 128))
        self.club_ps = ARClubGaussian(u_dim=self.latent_dim, v_dim=self.latent_dim, hidden=getattr(cfg, "club_hidden", 128))
        club_hidden = int(getattr(cfg, "club_hidden", 128))
        max_samp = int(getattr(cfg, "club_tp_max_samples", 2048))  # 새 cfg 옵션(없으면 2048)
        self.club_tp = ClubGaussian(x_dim=self.latent_dim,y_dim=self.latent_dim,hidden_size=club_hidden, max_samples=max_samp)

        # LLL 가중치 (논문 식(7)에서 IvCLUB + LLL 같이 들어감)
        self.lll_weight = getattr(cfg, "disen_lll_weight", 1.0)
        
        self.spk_stat_proj = nn.Linear(self.latent_dim * 2, self.latent_dim)


        # ===== Generative KD 모듈 (Student last layer ↔ Teacher Text feature) =====
        self.flow = FlowMatchingModule(
            self.dim_s, self.latent_dim, hidden=self.latent_dim, steps=flow_steps, loss_weight=flow_weight
        ) if use_flow else None

        self.diffkd = DiffKDModule(
            teacher_dim=self.latent_dim, latent_dim=self.latent_dim, student_dim=self.dim_s, steps=diffkd_steps
        ) if use_diffkd else None
        
        # forward 중간 결과 저장
        self._last_mel = None        # Student preprocessor output (B, n_mels, T)
        self._last_enc = None        # Student encoder output (B, T, C_s)
        self._tch_last = None        # Teacher last layer feature (B, C_t, T)
        self._txt_emb = None       # Teacher Text Encoder feature (B, C_t, T)
        self._tch_logp = None        # Teacher log-probs (B, T, V)
        
        # XAI / visualization 설정
        self.vis_enable = bool(getattr(cfg, "disen_vis_enable", True))
        self.vis_interval = int(getattr(cfg, "disen_vis_interval", 500))
        self.vis_max_samples = int(getattr(cfg, "disen_vis_max_samples", 1))
        out_dir = getattr(cfg, "out_dir", "./outputs")
        self.vis_dir = Path(out_dir) / "xai"
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # === Text speaker probe (for invariance check) ===
        self.use_txt_spk_probe = bool(getattr(cfg, "use_txt_spk_probe", True))
        self.txt_probe_lambda  = float(getattr(cfg, "txt_probe_lambda", 1.0))

        if self.num_spk > 1 and self.use_txt_spk_probe:
            # probe capacity는 spk_backbone 정도면 충분 (ver2와 동일)
            self.txt_probe_backbone = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, self.latent_dim),
                nn.ReLU(inplace=True),

                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, dilation=2, padding=2, bias=False),
                nn.GroupNorm(8, self.latent_dim),
                nn.ReLU(inplace=True),

                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, dilation=3, padding=3, bias=False),
                nn.GroupNorm(8, self.latent_dim),
                nn.ReLU(inplace=True),
            )

            txt_probe_hidden = getattr(cfg, "txt_probe_hidden", self.latent_dim * 2)
            self.txt_spk_probe_cls = nn.Sequential(
                nn.Linear(self.latent_dim * 2, txt_probe_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=getattr(cfg, "txt_probe_dropout", 0.3)),
                nn.Linear(txt_probe_hidden, self.num_spk),
            )
        else:
            self.txt_probe_backbone = None
            self.txt_spk_probe_cls = None        
    
        train_manifest = self.cfg.train_ds.manifest_filepath
        self._load_manifest_speakers(train_manifest)
        L = self.cfg.encoder.n_layers
        self.layer_list_for_disent = self._prepare_layer_indices(getattr(cfg, "layer_list_for_disent", [4,8,12,16]), L, default_low=True)  # 1-based index
        self.neg_K = int(getattr(cfg, "neg_K", 8))

    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        # ------ Forward (Student path) ------
        # hook buffer 초기화
        self.stu_feats.clear()
        # teacher hook은 _run_teacher에서 초기화

        has_input = input_signal is not None and input_signal_length is not None
        has_proc = processed_signal is not None and processed_signal_length is not None
        if (has_input ^ has_proc) is False:
            raise ValueError("Provide either raw input or processed input, not both.")

        if not has_proc:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_length
            )

        # Student encode
        s_enc, s_len = self.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )  # (B, T, C_s)
        logp = self.decoder(encoder_output=s_enc)
        greedy = logp.argmax(-1)

        # 캐시
        self._last_mel = processed_signal          # (B, n_mels, T)
        self._last_mel_len = processed_signal_length
        self._last_enc = s_enc                     # (B, T, C_s)

        return logp, s_len, greedy

    def training_step(self, batch, batch_idx):
        """
        그림과 맞게 학습 순서:
        1) Student forward
        2) Teacher forward (_run_teacher)
        3) Text/Speaker/Prosody factorization + MI, Rec, Speaker CE
        4) Generative KD(FM/DF) : Student last vs Teacher Text feature
        5) CTC, Logit KD 등 합산
        """
        # NeMo ASR default batch: (signal, signal_len, transcript, transcript_len)
        if len(batch) == 5:
            signal, sig_len, y, ylen, sample_ids = batch
            sample_ids = sample_ids.long()
            if getattr(self, "sampleid_to_manifest_id", None) is not None:
                # dataset index -> manifest_id(1-based)
                manifest_ids = self.sampleid_to_manifest_id.to(sample_ids.device)[sample_ids]  # (B,)
                # manifest_speakers는 manifest 파일 라인 순서대로 읽은 리스트이므로 (manifest_id - 1)로 접근
                speaker_ids = self.manifest_speakers.to(sample_ids.device)[manifest_ids - 1]
            else:
                # fallback(필터링 없다는 가정)
                speaker_ids = self.manifest_speakers.to(sample_ids.device)[sample_ids]

        elif len(batch) == 4:
            signal, sig_len, y, ylen = batch
            speaker_ids = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        
        if speaker_ids is not None:
            assert speaker_ids.min().item() >= -1
            assert speaker_ids.max().item() < self.num_spk
        
        # 1) Student forward
        logp, enc_len, _ = self.forward(
            input_signal=signal, input_signal_length=sig_len
        )

        # 2) Teacher forward (한 번만)
        self._run_teacher(self._last_mel, self._last_mel_len) # student forward 때 캐시한 mel 사용

        total = torch.tensor(0.0, device=logp.device)
        
        # 3) Factorization embeddings + MI/Rec/Speaker CE
        if self.use_disent:
            phys_targets = self._get_phys_targets(sample_ids, T_mel=self._last_mel.size(-1), split_name="train")
            embs = self._make_embeddings(speaker_ids, phys_targets=phys_targets)
            
            # MI term
            mi_upper, lll, mi_terms, lll_terms = self._mi_loss(
                txt_emb=embs["txt_emb"],
                pros_emb=embs["pros_emb"],
                spk_stat=embs["spk_stat"],
                enc_len=getattr(self, "t_enc_len", None),  # ✅ teacher length 사용
            )
            mi_upper = torch.clamp(mi_upper, min=0.0)
            self.log("train/mi_upper", mi_upper, on_epoch=True)
            self.log("train/lll", lll, on_epoch=True)
            total = total + self.mi_weight * mi_upper + self.lll_weight * lll

            # Reconstruction loss
            rec_txt = embs["rec_txt"]
            rec_spk = embs["rec_spk"]
            rec_pros = embs["rec_pros"]
            self.log("train/rec_txt", rec_txt, on_epoch=True)
            self.log("train/rec_spk", rec_spk, on_epoch=True)
            self.log("train/rec_pros", rec_pros, on_epoch=True)
            for k, v in mi_terms.items():
                self.log(f"train/mi_{k}", v, on_epoch=True)
            for k, v in lll_terms.items():
                self.log(f"train/lll_{k}", v, on_epoch=True)
            total = total + self.rec_txt_lambda * rec_txt + self.rec_spk_lambda * rec_spk + self.rec_pros_lambda * rec_pros

            # Speaker CE & ACC
            spk_ce = embs["spk_ce"]
            spk_acc = embs.get("spk_acc", None)

            if spk_ce is not None and torch.is_tensor(spk_ce):
                self.log("train/spk_ce", spk_ce, on_step=False, on_epoch=True)
                total = total + self.disen_spk_ce_lambda * spk_ce

            if spk_acc is not None and torch.is_tensor(spk_acc):
                self.log("train/spk_acc", spk_acc, on_epoch=True)

            # Prosody physical quantity supervision
            phys_loss = embs.get("phys_loss", None)
            if phys_loss is not None and torch.is_tensor(phys_loss):
                self.log("train/phys_loss", phys_loss, on_step=True, on_epoch=True, prog_bar=False)
                total = total + self.rec_pros_lambda * phys_loss
        else:
            self._txt_emb = None

        # 4) Generative KD (FM / DF)
        flow_loss = torch.tensor(0.0, device=self.device)
        diff_loss = torch.tensor(0.0, device=self.device)
        if self._txt_emb is not None and self._last_enc is not None:
            stu_feat = ensure_BCT(self._last_enc, C_expected=self.dim_s)
            tch_feat = ensure_BCT(self._txt_emb.detach(), C_expected=self.latent_dim)
            if self.use_flow:
                flow_loss = self.flow(stu_feat, tch_feat)
            if self.use_diffkd:
                diff_loss = self.diffkd(stu_feat, tch_feat)
                
        self.log("train/flow_loss", flow_loss, on_step=False, on_epoch=True)
        self.log("train/diff_loss", diff_loss, on_step=False, on_epoch=True)
        total = total + flow_loss + diff_loss

        # 5) CTC
        if self.use_ctc:
            ctc = self._ctc_loss(logp, enc_len, y, ylen)
            self.log("train/ctc", ctc, on_step=False, on_epoch=True)
            total = total + ctc

        # 6) Logit KD
        if self.use_logit_kd:
            kd_logit = self._logit_kd(logp)
            self.log("train/logit_kd", kd_logit, on_step=False, on_epoch=True)
            total = total + self.kd_alpha * kd_logit

        # 7) Layer-wise metric KD
        if self.use_layer_kd:
            kd_layer = self._layer_metric_kd()
            self.log("train/layer_kd", kd_layer, on_step=False, on_epoch=True)
            total = total + self.layer_kd_alpha * kd_layer

        if self.use_disent and self.use_txt_spk_probe and (embs is not None):
            probe_ce = embs.get("txt_probe_ce", None)
            probe_acc = embs.get("txt_probe_acc", None)

            if probe_ce is not None:
                self.log("probe/txt_spk_ce", probe_ce, on_step=False, on_epoch=True)
                total = total + self.txt_probe_lambda * probe_ce

            if probe_acc is not None:
                self.log("probe/txt_spk_acc", probe_acc, on_step=False, on_epoch=True)
        
        self.log("train/total", total, on_step=False, on_epoch=True, prog_bar=True)
        return total
        
    def _make_embeddings(self, speaker_ids, phys_targets=None):
        """
        ===== Teacher Text/Speaker/Prosody embedding 생성 =====
        반환: dict(
          txt_emb   = text embedding (B, E),
          spk_emb   = speaker embedding (B, E),
          pros_emb  = prosody embedding (B, E),
          spk_ce    = speaker CE loss,
          rec_txt   = text rec loss,
          rec_spk   = speaker rec loss,
        )
        """
        if self._tch_last is None:
            return None

        B, C_t, T_t = self._tch_last.shape
        last = self._tch_last                       # X_L^T  (B, C_t, T)
        
        # layer 리스트 기반 speaker/text representation 추출
        spk_rep, txt_rep = self._get_spk_txt_reps_from_layers()
        if spk_rep is None or txt_rep is None:
            # hook이 비어 있는 이상한 상황이면 그냥 last로 fallback
            spk_rep = last
            txt_rep = last

        # ----- Text Encoder -----
        txt_emb = self.txt_enc(txt_rep)     # (B, 196, T) -> (B, 96, T)
        txt_rec = self.txt_dec(txt_emb)     # (B, 96, T) -> (B, 196, T)
        rec_txt = F.mse_loss(txt_rec, txt_rep)
        
        # === text speaker probe ===
        txt_probe_ce, txt_probe_acc = self._text_spk_probe(txt_emb, speaker_ids)

        # Generative KD용 Text feature 캐시 (B, C_t, T)
        self._txt_emb = txt_emb

        # ----- Speaker Encoder -----
        spk_emb = self.spk_enc(spk_rep)     # (B, 196, T) -> (B, 96, T)
        spk_rec = self.spk_dec(spk_emb)     # (B, 96, T) -> (B, 196, T)
        rec_spk = F.mse_loss(spk_rec, spk_rep)         # Speaker Encoder도 X_L^T를 재구성하도록 학습 (그림의 Rec loss)

        # ====== Speaker static (B,96) 만들기 ======
        # backbone 통과 후 stats pooling 추천
        # TODO: backbone이 아니라 그냥 바로 spk_emb에서 스피커 예측하도록 하는게 좋을 듯?
        spk_feat = self.spk_backbone(spk_emb) if self.spk_backbone is not None else spk_emb  # (B,96,T)
        spk_mean = spk_feat.mean(dim=-1)   # (B,96)
        spk_std  = self.safe_std(spk_feat, dim=-1)    # (B,96)
        spk_stat = torch.cat([spk_mean, spk_std], dim=-1)  # (B,192) # TODO: 이게 맞아? 이게 필요할까? MI를 왜 이걸로? 어차피 static인데 
        spk_stat = self.spk_stat_proj(spk_stat)            # (B,96)
        
        # ----- Prosody (frame-level GST) -----
        # TODO: utterance-level prosody를 다루지 않을거임. frame-wise한 prosody embedding을 만들고, text embedding과 MI 측정할 예정
        T = txt_emb.size(-1)
        
        mel = self._last_mel.to(self.device) # (B, n_mels, T_mel)
        ref_seq = self.pros_ref(mel, return_seq=True) # (B, T', gru_dim)                # 1. Frame-level Reference Sequence (return_seq=True)
        style_seq = self.pros_gst(ref_seq) # (B, T', token_dim)                         # 2. Frame-level Style Tokens
        pros_emb = self.pros_proj(style_seq).transpose(1, 2)                            # 3. Projection & Transpose -> (B, latent_dim, T')
        pros_emb = F.interpolate(pros_emb, size=T, mode='linear', align_corners=False)  # 4. Target 시간 축(T_target)에 맞게 보간(Interpolate)
        
        pros_stat = pros_emb.mean(dim=-1)  # (B, 96) pros_stat for MI (utterance-level)

        # ===== Prosody Positive Supervision =====
        # 1. Mel Reconstruction Loss
        mel_target = self._last_mel # (B, 80, T_mel)
        pros_emb_for_mel = F.interpolate(pros_emb, size=mel_target.size(-1), mode='linear', align_corners=False)
        mel_pred = self.mel_dec(pros_emb_for_mel)
        rec_pros = F.mse_loss(mel_pred, mel_target)

        # 2. Physical Quantity Loss
        phys_loss = torch.tensor(0.0, device=pros_emb.device)
        if phys_targets is not None:
            phys_pred = self.prosody_predictor(pros_emb) # (B, 3, T_target)
            
            # target 길이를 맞춰줌 (보통 T와 T_mel이 같겠지만 만약을 대비)
            if phys_pred.size(-1) != phys_targets.size(-1):
                 phys_pred = F.interpolate(phys_pred, size=phys_targets.size(-1), mode='linear', align_corners=False)
                 
            phys_loss = F.mse_loss(phys_pred, phys_targets)
            
        # ----- Speaker CE & ACC -----
        spk_ce = torch.tensor(0.0, device=txt_emb.device)
        spk_acc = None
        if self.spk_cls is not None:
            valid_mask = (speaker_ids is not None) & (speaker_ids >= 0) & (speaker_ids < self.num_spk)
            if valid_mask.any():
                # spk_emb: (B, latent_dim, T)

                # (1) TDNN-style backbone 통과
                if self.spk_backbone is not None:
                    spk_feat = self.spk_backbone(spk_emb)      # (B, latent_dim, T)
                else:
                    spk_feat = spk_emb

                # (2) stats pooling: mean + std → utterance-level embedding
                spk_mean = spk_feat.mean(dim=-1)              # (B, latent_dim)
                spk_std  = self.safe_std(spk_feat, dim=-1)               # (B, latent_dim)
                spk_utt  = torch.cat([spk_mean, spk_std], dim=-1)  # (B, 2*latent_dim)

                spk_utt_valid = spk_utt[valid_mask]           # (B_valid, 2*latent_dim)
                target_valid = speaker_ids[valid_mask]
                target_valid = target_valid.clamp(min=0).long()   # ★ int64 로 변환 ★
                
                # (선택) 디버그용 안전 체크
                if torch.any(target_valid < 0) or torch.any(target_valid >= self.num_spk):
                    raise RuntimeError(
                        f"[BUG] speaker_ids out of range: "
                        f"min={int(target_valid.min())}, max={int(target_valid.max())}, "
                        f"num_spk={self.num_spk}"
                    )
                
                # (3) classifier: (B_valid, num_spk)
                logits_utt = self.spk_cls(spk_utt_valid)      # (B_valid, num_spk)

                # (4) CE loss
                spk_ce = F.cross_entropy(logits_utt, target_valid)

                # (5) accuracy (전체 배치 기준)
                all_logits = self.spk_cls(spk_utt)            # (B, num_spk)
                preds = all_logits.argmax(dim=-1)             # (B,) long
                spk_acc = (preds[valid_mask] == target_valid).float().mean()
        
        # XAI: 시각화
        if self.vis_enable:
            self._xai_visualize(
                txt_emb=txt_emb.detach(),
                spk_emb=spk_emb.detach(),
                pros_emb=pros_emb.detach(),
                txt_rep=txt_rep.detach(),
                spk_rep=spk_rep.detach(),
                rec_txt=rec_txt.detach(),
                rec_spk=rec_spk.detach(),
                speaker_ids=speaker_ids.detach() if speaker_ids is not None else None,
                enc_len=getattr(self, "t_enc_len", None),
            )
        
        return {
            "txt_emb": txt_emb,         # (B,96,T)
            "spk_emb": spk_emb,         # (B,96)
            "pros_emb": pros_emb,       # (B,96,T)

            "spk_stat": spk_stat,       # (B,96) static (MI용)
            "pros_stat": pros_stat,     # (B,96) static (MI용)

            "spk_ce": spk_ce,
            "spk_acc": spk_acc,
            
            "rec_txt": rec_txt,
            "rec_spk": rec_spk,
            "rec_pros": rec_pros,
            
            "txt_probe_ce": txt_probe_ce,
            "txt_probe_acc": txt_probe_acc,
            
            "pros_stat": pros_stat,     # (B,96) static (MI용)
            "phys_loss": phys_loss
        }
    
    def _mi_loss(self, txt_emb, pros_emb, spk_stat, enc_len=None):
        """
        txt_emb: (B,D,T) dynamic
        pros_emb:(B,D,T) dynamic
        spk_stat:(B,D)   static
        enc_len : (B,)   valid length for T (teacher length 권장)

        return:
          mi_sum, lll_sum, mi_terms, lll_terms
        """
        txt = txt_emb
        pros = pros_emb
        spk = spk_stat

        pairs = set([t.strip() for t in self.mi_pairs.split(",") if t.strip()])

        lll_terms = {}
        mi_terms = {}

        lll_sum = torch.tensor(0.0, device=self.device)
        mi_sum  = torch.tensor(0.0, device=self.device)

        # ---- length mask 생성 ----
        mask = None
        if enc_len is None:
            enc_len = getattr(self, "t_enc_len", None)
        if enc_len is not None:
            T = txt.size(-1)
            mask = self._make_len_mask(enc_len, T, device=txt.device, dtype=torch.float32)  # (B,T)

        # reduce_dim: 학습용은 일단 "mean" 유지 권장(스케일 안정)
        reduce_dim = "mean"

        # tp: dynamic-dynamic (ClubGaussian)  ✅ K negative 지원
        if "tp" in pairs:
            lll_terms["tp"] = self.club_tp.ll_loss(
                txt.detach(), pros.detach(),
                mask=mask,
                reduce_dim=reduce_dim
            )
            lll_sum = lll_sum + lll_terms["tp"]

            self._freeze_params(self.club_tp, True)
            mi_terms["tp"] = self.club_tp.mi_upper(
                txt, pros,
                K=self.neg_K,            # ✅ tp도 K negative
                mask=mask,
                reduce_dim=reduce_dim
            )
            mi_sum = mi_sum + mi_terms["tp"]
            self._freeze_params(self.club_tp, False)

        # ts: dynamic-static (ARClubGaussian)
        if "ts" in pairs:
            lll_terms["ts"] = self.club_ts.ll_loss(
                txt.detach(), spk.detach(),
                mask=mask,
                reduce_dim=reduce_dim
            )
            lll_sum = lll_sum + lll_terms["ts"]

            self._freeze_params(self.club_ts, True)
            mi_terms["ts"] = self.club_ts.mi_upper(
                txt, spk,
                K=self.neg_K,
                mask=mask,
                reduce_dim=reduce_dim
            )
            mi_sum = mi_sum + mi_terms["ts"]
            self._freeze_params(self.club_ts, False)

        # ps: dynamic-static (ARClubGaussian)
        if "ps" in pairs:
            lll_terms["ps"] = self.club_ps.ll_loss(
                pros.detach(), spk.detach(),
                mask=mask,
                reduce_dim=reduce_dim
            )
            lll_sum = lll_sum + lll_terms["ps"]

            self._freeze_params(self.club_ps, True)
            mi_terms["ps"] = self.club_ps.mi_upper(
                pros, spk,
                K=self.neg_K,
                mask=mask,
                reduce_dim=reduce_dim
            )
            mi_sum = mi_sum + mi_terms["ps"]
            self._freeze_params(self.club_ps, False)

        return mi_sum, lll_sum, mi_terms, lll_terms

    def _get_phys_targets(self, sample_ids: torch.Tensor, T_mel: int, split_name: str = "train"):
        """
        sample_ids: (B,) long  (NeMo return_sample_id=True일 때 dataset index)
        returns: (B,3,T_mel) float32 (f0, energy, vuv)
        """
        if sample_ids is None:
            return None

        root = Path(getattr(self.cfg, "phys_cache_root", ""))
        if not root.exists():
            return None

        # dataset index -> manifest_id(1-based)로 변환 (필터링 대응)
        if getattr(self, "sampleid_to_manifest_id", None) is not None:
            manifest_ids = self.sampleid_to_manifest_id.to(sample_ids.device)[sample_ids]  # (B,)
        else:
            # fallback: 필터링 없다고 가정
            manifest_ids = sample_ids + 1

        B = int(sample_ids.numel())

        # (중요) GPU 텐서를 루프에서 개별 대입하면 작은 H2D copy가 B번 생김
        # => CPU에서 만들고 한 번에 GPU로 올리는 게 보통 더 빠름
        out_cpu = torch.zeros((B, 3, T_mel), dtype=torch.float32, device="cpu")

        for b in range(B):
            mid = int(manifest_ids[b].item())
            key = (split_name, mid)

            arr = self._phys_lru_get(key)
            if arr is None:
                p = root / split_name / f"{mid}{self.phys_cache_ext}"
                if not p.exists():
                    continue

                if self.phys_cache_ext == ".npy":
                    # mmap 가능
                    arr = np.load(p, mmap_mode="r")  # (3,T_src) float16
                else:
                    # (구형 npz를 유지한다면) 여기서 np.load(p) 후 key로 읽기
                    data = np.load(p)
                    f0 = data["f0"].astype(np.float32)
                    eng = data["energy"].astype(np.float32)
                    vuv = data["vuv"].astype(np.float32)
                    arr = np.stack([f0, eng, vuv], axis=0).astype(np.float32)

                self._phys_lru_put(key, arr)

            # numpy(memmap) -> torch
            phys = torch.from_numpy(np.array(arr, copy=False))  # float16 or float32
            if phys.dtype != torch.float32:
                phys = phys.float()

            # 길이 맞추기 (T_src -> T_mel)
            # torch interpolate로 1번에 처리 (np.interp 3번보다 보통 낫습니다)
            T_src = phys.size(-1)
            if T_src != T_mel:
                phys = F.interpolate(
                    phys.unsqueeze(0),  # (1,3,T_src)
                    size=T_mel,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)

            out_cpu[b] = phys

        # TODO: 정규화 한 버전 vs 안 한 버전 비교하기
        # (선택) 정규화: 기존과 동일하게 frame축 기준 z-score
        # mean = out_cpu.mean(dim=-1, keepdim=True)
        # std = out_cpu.std(dim=-1, keepdim=True).clamp_min(1e-5)
        # out_cpu = (out_cpu - mean) / std

        # GPU로 한 번에 올림
        out = out_cpu.to(self.device, non_blocking=True)
        return out

    def _phys_lru_get(self, key):
        if key not in self._phys_lru:
            return None
        val = self._phys_lru.pop(key)
        self._phys_lru[key] = val
        return val

    def _phys_lru_put(self, key, val):
        self._phys_lru[key] = val
        if len(self._phys_lru) > self.phys_cache_lru:
            self._phys_lru.popitem(last=False)
    
    def _run_teacher(self, preprocessed_signal, processed_signal_length):
        """그림 기준 왼쪽 Teacher Encoder + Decoder 한 번만 실행."""
        self.tch_feats.clear()
        with torch.no_grad():
            t_enc, t_enc_len = self.teacher.encoder(
                audio_signal=preprocessed_signal, length=processed_signal_length
            )  # (B, T, C_t)
            self.t_enc_len = t_enc_len
            t_logp = self.teacher.decoder(encoder_output=t_enc)
        # 캐시
        self._tch_logp = t_logp
        self._tch_last = self._to_BCT(t_enc)       # (B, C_t, T)

    def _to_BCT(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        if x.dim() == 3 and (x.size(2) == self.dim_s or x.size(2) == self.dim_t):
            x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        return x

    def _cap_stu(self, module, inp, out):
        self.stu_feats.append(self._to_BCT(out))

    def _cap_tch(self, module, inp, out):
        self.tch_feats.append(self._to_BCT(out))

    def _prepare_layer_indices(self, layer_list, L, default_low: bool):
        """
        layer_list: 사용자가 넘긴 리스트 (1-based index라고 가정, 예: [1,2,3])
        L: 전체 layer 수 (len(self.tch_feats))

        반환: 사용할 0-based index 리스트 (비어 있으면 fallback로 low/high를 선택)
        """
        idxs = []

        if isinstance(layer_list, (list, tuple)) and len(layer_list) > 0:
            for x in layer_list:
                try:
                    i = int(x)
                except Exception:
                    continue
                # 1-based → 0-based
                i0 = i - 1
                if 0 <= i0 < L:
                    idxs.append(i0)

        if len(idxs) > 0:
            return sorted(set(idxs))

        # fallback: 사용자가 준 index가 전부 out-of-range거나 비어 있으면
        if default_low:
            # 하위 1/3 layer
            k = max(1, L // 3)
            return list(range(0, k))
        else:
            # 상위 1/3 layer
            k = max(1, L // 3)
            return list(range(L - k, L))

    def _get_spk_txt_reps_from_layers(self):
        """
        self.tch_feats: List[(B, C_t, T)], hook로 캡쳐한 teacher encoder 각 layer 출력.
        disent_spk_layers, disent_txt_layers에 지정된 layer만 골라 평균.

        반환: (spk_rep, txt_rep)  둘 다 (B, C_t, T)
        """
        if not self.tch_feats:
            return None, None

        L = len(self.tch_feats)

        # speaker용 layer index
        spk_idxs = self._prepare_layer_indices(
            self.disent_spk_layers, L, default_low=True
        )
        # text용 layer index
        txt_idxs = self._prepare_layer_indices(
            self.disent_txt_layers, L, default_low=False
        )

        spk_stack = torch.stack([self.tch_feats[i] for i in spk_idxs], dim=0)  # (K_s,B,C,T)
        txt_stack = torch.stack([self.tch_feats[i] for i in txt_idxs], dim=0)  # (K_t,B,C,T)

        spk_rep = spk_stack.mean(dim=0)  # (B,176,T)
        txt_rep = txt_stack.mean(dim=0)  # (B,176,T)
        return spk_rep, txt_rep
    
    def _ctc_loss(self, logp, enc_len, y, ylen):
        return self.loss(
            log_probs=logp,
            targets=y,
            input_lengths=enc_len,
            target_lengths=ylen,
        )

    def _logit_kd(self, logp_s):
        if self._tch_logp is None:
            return torch.tensor(0.0, device=logp_s.device)
        T = self.kd_temperature
        with torch.no_grad():
            t_logp = self._tch_logp
            p_t = F.softmax(t_logp / T, dim=-1)
        logp_s_T = F.log_softmax(logp_s / T, dim=-1)
        return F.kl_div(logp_s_T, p_t, reduction="batchmean") * (T * T)

    def _layer_metric_kd(self):
        # 기존 구현 유지
        if not self.stu_feats or not self.tch_feats:
            return torch.tensor(0.0, device=self.device)

        losses = []
        L = min(len(self.stu_feats), len(self.tch_feats))
        for i in range(L):
            s = self.stu_feats[i]              # (B, C_s, T)
            t = self.tch_feats[i].detach()     # (B, C_t, T)
            s_proj = self.stu_to_tea_proj(s)              # (B, C_t, T)
            losses.append(F.mse_loss(s_proj, t))
        return sum(losses) / L

    def _generative_kd(self):
        """
        - Student 마지막 레이어 feature (encoder 출력)
        - Teacher Text Encoder로 얻은 Text feature
        사이에 Flow Matching / DiffKD 적용.
        """
        if (not self.use_flow and not self.use_diffkd) or (self._last_enc is None):
            z = torch.tensor(0.0, device=self.device)
            return z, z

        if self._txt_emb is None:
            # Text Encoder가 아직 안 돌았다면 0
            z = torch.tensor(0.0, device=self.device)
            return z, z

        # Student last: (B, T, C_s) -> (B, C_s, T)
        s_raw = self._last_enc
        s = ensure_BCT(s_raw, C_expected=self.dim_s)   # (B,C_s,T)
        # Teacher text feature: (B, C_t, T)
        t_raw = self._txt_emb.detach()
        t = ensure_BCT(t_raw, C_expected=self.latent_dim)  # (B,96,T) or (B,dim_t,T) depending on what you store

        flow_loss = torch.tensor(0.0, device=self.device)
        diff_loss = torch.tensor(0.0, device=self.device)
        if self.use_flow and self.flow is not None:
            flow_loss = self.flow(s, t)
        if self.use_diffkd and self.diffkd is not None:
            diff_loss = self.diffkd(s, t)
        
        return flow_loss, diff_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 4:
            signal, sig_len, y, ylen = batch
        else:
            signal, sig_len, y, ylen, _ = batch
        logp, enc_len, hyp = self.forward(
            input_signal=signal, input_signal_length=sig_len
        )
        transcribed = self.wer.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=logp,
            decoder_lengths=enc_len,
            return_hypotheses=False,
        )
        return transcribed

    def _save_fig(self, fig, name: str):
        path = self.vis_dir / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _plot_heatmap(self, mat, title, fname, xlabel=None, ylabel=None):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(mat, aspect="auto", origin="lower")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        self._save_fig(fig, fname)

    def _plot_line(self, vec, title, fname, xlabel="index"):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(vec)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        self._save_fig(fig, fname)

    def _xai_visualize(self, txt_emb, spk_emb, pros_emb, txt_rep, spk_rep, rec_txt, rec_spk):
        """
        txt_emb, spk_emb, pros_emb: (B, 96, T)
        txt_rep, spk_rep: (B, C_t, T)
        """
        # step/epoch 기반 prefix
        step = int(getattr(self, "global_step", 0))
        if (step % self.vis_interval) != 0:
            return

        B = txt_emb.size(0)
        num = min(self.vis_max_samples, B)

        # 1) mel & pros_ref heatmap
        if self._last_mel is not None:
            mel = self._last_mel[:num].detach().cpu()   # (num, n_mels, T)
            for b in range(num):
                self._plot_heatmap(
                    mel[b],
                    title=f"Mel (sample {b}, step {step})",
                    fname=f"step{step:06d}_s{b}_mel.png",
                    xlabel="time",
                    ylabel="mel-bin",
                )

        if hasattr(self.pros_ref, "last_out") and self.pros_ref.last_out is not None:
            pref = self.pros_ref.last_out[:num]         # (num,96,T)
            for b in range(num):
                self._plot_heatmap(
                    pref[b],
                    title=f"Prosody ref (sample {b}, step {step})",
                    fname=f"step{step:06d}_s{b}_pros_ref.png",
                    xlabel="time",
                    ylabel="pros-dim",
                )

        # 2) GST attention (token × head)
        if hasattr(self.pros_gst, "last_attn") and self.pros_gst.last_attn is not None:
            attn = self.pros_gst.last_attn              # (B, N, H)
            for b in range(num):
                a = attn[b]                             # (N,H)
                # head-mean
                mean_token = a.mean(dim=-1).numpy()
                self._plot_line(
                    mean_token,
                    title=f"GST token weights (sample {b}, step {step})",
                    fname=f"step{step:06d}_s{b}_gst_tokens.png",
                    xlabel="token-id",
                )
                # full heatmap N×H
                self._plot_heatmap(
                    a.numpy(),
                    title=f"GST attn (token×head) (s{b}, step {step})",
                    fname=f"step{step:06d}_s{b}_gst_attn.png",
                    xlabel="head",
                    ylabel="token",
                )

        # 3) txt/spk/pros 임베딩 norm over time & similarity
        for b in range(num):
            t = txt_emb[b].detach().cpu()   # (E,T)
            s = spk_emb[b].detach().cpu()
            p = pros_emb[b].detach().cpu()

            # L2 norm over channels per time
            def time_norm(x):
                return x.norm(dim=0).numpy()  # (T,)

            self._plot_line(
                time_norm(t),
                title=f"||txt_emb|| over time (s{b}, step {step})",
                fname=f"step{step:06d}_s{b}_txt_norm.png",
                xlabel="time"
            )
            self._plot_line(
                time_norm(s),
                title=f"||spk_emb|| over time (s{b}, step {step})",
                fname=f"step{step:06d}_s{b}_spk_norm.png",
                xlabel="time"
            )
            self._plot_line(
                time_norm(p),
                title=f"||pros_emb|| over time (s{b}, step {step})",
                fname=f"step{step:06d}_s{b}_pros_norm.png",
                xlabel="time"
            )

            # time-mean embedding for cosine similarity
            t_bar = t.mean(dim=-1)   # (E,)
            s_bar = s.mean(dim=-1)
            p_bar = p.mean(dim=-1)

            def cos(a, b):
                return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

            sim_mat = torch.tensor([
                [1.0,       cos(t_bar, s_bar), cos(t_bar, p_bar)],
                [cos(s_bar, t_bar), 1.0,       cos(s_bar, p_bar)],
                [cos(p_bar, t_bar), cos(p_bar, s_bar), 1.0],
            ])
            self._plot_heatmap(
                sim_mat.numpy(),
                title=f"Cosine sim (txt/spk/pros) (s{b}, step {step})",
                fname=f"step{step:06d}_s{b}_cos_sim.png",
                xlabel="emb",
                ylabel="emb",
            )

        # 4) 재구성 오차 heatmap (1개 샘플만)
        t_err = (rec_txt.detach().cpu() - txt_rep.detach().cpu()).abs()
        s_err = (rec_spk.detach().cpu() - spk_rep.detach().cpu()).abs()
        self._plot_heatmap(
            t_err[0],
            title=f"Text rec abs error (step {step})",
            fname=f"step{step:06d}_rec_txt_err.png",
            xlabel="time",
            ylabel="channel",
        )
        self._plot_heatmap(
            s_err[0],
            title=f"Spk rec abs error (step {step})",
            fname=f"step{step:06d}_rec_spk_err.png",
            xlabel="time",
            ylabel="channel",
        )
        # 5) 2D projection (PCA / TSNE) + label coloring ===============
        # mask로 pooling
        T = txt_emb.size(-1)
        if enc_len is None:
            enc_len = getattr(self, "t_enc_len", None)
        mask = None
        if enc_len is not None:
            mask = self._make_len_mask(enc_len, T, device=txt_emb.device, dtype=torch.float32)  # (B,T)

        def masked_mean_BCT(x):
            # x: (B,C,T)
            if mask is None:
                return x.mean(dim=-1)
            denom = mask.sum(dim=1).clamp(min=1.0)  # (B,)
            return (x * mask[:, None, :]).sum(dim=-1) / denom[:, None]

        # utterance-level stats for projection
        txt_stat  = masked_mean_BCT(txt_emb).detach()
        spk_stat  = masked_mean_BCT(spk_emb).detach()   # (B,C)  (spk_stat을 따로 넘기면 더 좋지만, 여기선 mean 사용)
        pros_stat = masked_mean_BCT(pros_emb).detach()

        # subsample for plotting if huge batch
        max_pts = int(getattr(self.cfg, "disen_vis_proj_max_points", 256))
        if B > max_pts:
            idx = torch.randperm(B, device=txt_stat.device)[:max_pts]
            txt_stat, spk_stat, pros_stat = txt_stat[idx], spk_stat[idx], pros_stat[idx]
            if speaker_ids is not None:
                speaker_ids = speaker_ids[idx]
            B = max_pts

        # projection method
        method = str(getattr(self.cfg, "disen_vis_proj_method", "pca")).lower()

        def proj2d(X, method="pca"):
            # X: (N,D) torch
            X = X.float()
            X = X - X.mean(dim=0, keepdim=True)
            if X.size(0) < 2:
                return X.new_zeros((X.size(0), 2)).cpu().numpy()

            if method == "tsne":
                try:
                    from sklearn.manifold import TSNE
                    Xn = X.detach().cpu().numpy()
                    # perplexity는 N에 맞춰야 함
                    N = Xn.shape[0]
                    perp = float(min(30, max(2, (N - 1) // 3)))
                    Z = TSNE(
                        n_components=2,
                        init="pca",
                        learning_rate="auto",
                        perplexity=perp,
                        random_state=0
                    ).fit_transform(Xn)
                    return Z
                except Exception:
                    method = "pca"

            # PCA (torch SVD)
            # X: (N,D) -> Vh: (k,D)
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            W = Vh[:2].T  # (D,2)
            Z = (X @ W).detach().cpu().numpy()
            return Z

        Z_txt  = proj2d(txt_stat, method=method)
        Z_spk  = proj2d(spk_stat, method=method)
        Z_pros = proj2d(pros_stat, method=method)

        # colors
        if speaker_ids is not None:
            c = speaker_ids.detach().cpu().numpy()
        else:
            c = np.arange(B)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        scat0 = axes[0].scatter(Z_txt[:, 0],  Z_txt[:, 1],  c=c, s=14)
        axes[0].set_title("txt_stat 2D (colored by speaker)")
        axes[0].set_xticks([]); axes[0].set_yticks([])

        axes[1].scatter(Z_spk[:, 0],  Z_spk[:, 1],  c=c, s=14)
        axes[1].set_title("spk_stat 2D (colored by speaker)")
        axes[1].set_xticks([]); axes[1].set_yticks([])

        axes[2].scatter(Z_pros[:, 0], Z_pros[:, 1], c=c, s=14)
        axes[2].set_title("pros_stat 2D (colored by speaker)")
        axes[2].set_xticks([]); axes[2].set_yticks([])

        # 공통 colorbar (speaker id가 많아도 대충 추세 확인 가능)
        fig.colorbar(scat0, ax=axes, fraction=0.02, pad=0.02)
        fig.suptitle(f"2D projection ({method}) @ step {step}")
        self._save_fig(fig, f"step{step:06d}_proj_{method}_speaker.png")

    def _text_spk_probe(self, txt_emb, speaker_ids):
        if (self.txt_spk_probe_cls is None) or (speaker_ids is None):
            return None, None

        valid_mask = (speaker_ids >= 0) & (speaker_ids < self.num_spk)
        if not valid_mask.any():
            return None, None

        x = txt_emb.detach()

        if self.txt_probe_backbone is not None:
            x = self.txt_probe_backbone(x)

        mean = x.mean(dim=-1)
        std  = self.safe_std(x, dim=-1)
        utt  = torch.cat([mean, std], dim=-1)

        logits = self.txt_spk_probe_cls(utt)
        target = speaker_ids.clamp(min=0).long()

        probe_ce = F.cross_entropy(logits[valid_mask], target[valid_mask])
        preds = logits.argmax(dim=-1)
        probe_acc = (preds[valid_mask] == target[valid_mask]).float().mean()
        return probe_ce, probe_acc

    def _freeze_params(self, module: nn.Module, freeze: bool):
        for p in module.parameters():
            p.requires_grad = (not freeze)
    
    def _load_manifest_speakers(self, manifest_path: str):
        spk = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                spk.append(int(obj.get("spk_idx", -1)))
        # (N,) 텐서로 들고 있기
        self.manifest_speakers = torch.tensor(spk, dtype=torch.long)  # CPU에 둬도 OK

    @staticmethod
    def safe_std(x, dim=-1, eps=1e-5):
        # unbiased=False로 분모 0 방지 + eps로 0분산 안정화
        var = torch.var(x, dim=dim, unbiased=False)
        return torch.sqrt(var + eps)

    def _ramp(self, step, warmup, ramp):
        # warmup 동안 0, 그 이후 ramp 동안 0->1 선형
        if step < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        x = (step - warmup) / float(ramp)
        return float(min(1.0, max(0.0, x)))

    def setup_training_data(self, train_data_config):
        super().setup_training_data(train_data_config)

        # NeMo가 만든 train dataset에서 dataset index -> manifest_id(1-based) 매핑 생성
        try:
            ds = self._train_dl.dataset  # NeMo가 내부에 생성한 DataLoader
            col = ds.manifest_processor.collection
            manifest_ids = [int(col[i].id) for i in range(len(col))]  # 1-based
            self.sampleid_to_manifest_id = torch.tensor(manifest_ids, dtype=torch.long)  # CPU 텐서
        except Exception as e:
            # fallback: 매핑 생성 실패하면 기존 가정(sample_id+1)으로만 동작
            self.sampleid_to_manifest_id = None
            print(f"[WARN] could not build sampleid_to_manifest_id mapping: {e}")

    @staticmethod
    def _make_len_mask(lengths: torch.Tensor, T: int, device=None, dtype=torch.float32):
        """
        lengths: (B,)  각 샘플의 유효 길이
        return : (B,T) 0/1 mask (1=valid)
        """
        if lengths is None:
            return None
        if device is None:
            device = lengths.device
        lengths = lengths.to(device)
        t = torch.arange(T, device=device).unsqueeze(0)  # (1,T)
        mask = (t < lengths.unsqueeze(1)).to(dtype=dtype)  # (B,T)
        return mask

class GlobalStyleTokenLayer(nn.Module):
    def __init__(self, num_tokens=10, token_dim=96, num_heads=4, ref_dim=96):
        super().__init__()
        assert token_dim % num_heads == 0
        self.token_dim = token_dim
        self.num_heads = num_heads

        self.tokens = nn.Parameter(torch.randn(num_tokens, token_dim))
        self.query_proj = nn.Linear(ref_dim, token_dim, bias=False)
        self.key_proj   = nn.Linear(token_dim, token_dim, bias=False)
        self.v = nn.Linear(token_dim, num_heads, bias=False)

        # XAI: (B, N, H) - time-mean attention (기존 호환)
        self.last_attn = None
        # XAI/debug: (B, T', N, H) - frame-level attention
        self.last_attn_seq = None

    def forward(self, ref_emb):
        """
        ref_emb:
          - (B, ref_dim)          : global query 1개 → (B, token_dim)
          - (B, T', ref_dim)      : frame-level queries → (B, T', token_dim)
        """
        k = torch.tanh(self.key_proj(self.tokens))  # (N, D)
        N, D = k.size()

        if ref_emb.dim() == 2:
            # ----- 기존 global GST -----
            q = torch.tanh(self.query_proj(ref_emb))          # (B, D)
            B = q.size(0)

            q_exp = q.unsqueeze(1).expand(B, N, D)            # (B, N, D)
            k_exp = k.unsqueeze(0).expand(B, N, D)            # (B, N, D)
            s = torch.tanh(q_exp + k_exp)                     # (B, N, D)

            logits = self.v(s)                                # (B, N, H)
            attn = torch.softmax(logits, dim=1)               # (B, N, H)

            self.last_attn = attn.detach().cpu()
            self.last_attn_seq = None

            style = torch.einsum("bnh,nd->bdh", attn, self.tokens)  # (B, D, H)
            style = style.mean(dim=-1)                               # (B, D)
            return style

        if ref_emb.dim() == 3:
            # ----- frame-level GST -----
            # ref_emb: (B, T', ref_dim)
            q = torch.tanh(self.query_proj(ref_emb))          # (B, T', D)
            B, Tp, _ = q.size()

            # (B, T', 1, D) + (1, 1, N, D) -> (B, T', N, D)
            s = torch.tanh(q[:, :, None, :] + k[None, None, :, :])   # (B, T', N, D)

            logits = self.v(s)                                # (B, T', N, H)
            attn = torch.softmax(logits, dim=2)               # softmax over tokens N

            # 저장: frame-level & time-mean
            self.last_attn_seq = attn.detach().cpu()          # (B, T', N, H)
            self.last_attn = attn.mean(dim=1).detach().cpu()  # (B, N, H)  (기존 호환)

            style = torch.einsum("btnh,nd->btdh", attn, self.tokens) # (B, T', D, H)
            style = style.mean(dim=-1)                                # (B, T', D)
            return style

        raise ValueError(f"ref_emb must be 2D or 3D, got {ref_emb.dim()}D")

class GlobalProsodyReferenceEncoder(nn.Module):
    def __init__(self, n_mels=80, channels=(32,64,128), gru_dim=96):
        super().__init__()
        self.n_mels = n_mels
        self.channels = channels
        self.K = len(channels)

        convs = []
        in_ch = 1
        for out_ch in channels:
            convs += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*convs)

        # 80 -> 40 -> 20 으로 n_mels(Frequency)가 1/4로 줄어듦
        reduced_mels = self.n_mels // 8
        self.gru = nn.GRU(
            input_size=channels[-1] * reduced_mels,
            hidden_size=gru_dim,
            batch_first=True
        )

        # XAI
        self.last_out = None   # (B, gru_dim)
        self.last_seq = None   # (B, gru_dim, T')

    def forward(self, mel, return_seq: bool = False):
        """
        mel: (B, n_mels, T)
        return:
          - return_seq=False: global ref (B, gru_dim)
          - return_seq=True : frame-level ref seq (B, T', gru_dim)
        """
        B, n_mels, T = mel.shape

        x = mel.transpose(1, 2).unsqueeze(1)      # (B,1,T,n_mels)
        z = self.conv(x)                          # (B,C,T',F')
        B, C, Tp, Fp = z.shape

        z = z.permute(0, 2, 1, 3).contiguous()    # (B,T',C,F')
        z = z.view(B, Tp, C * Fp)                 # (B,T',C*F')

        out_seq, h = self.gru(z)                  # out_seq: (B,T',gru_dim), h: (1,B,gru_dim)
        ref_global = h[-1]                        # (B,gru_dim)

        # XAI 저장
        self.last_out = ref_global.detach().cpu()
        self.last_seq = out_seq.transpose(1, 2).detach().cpu()   # (B,gru_dim,T')

        return out_seq if return_seq else ref_global

'''
class ClubGaussian(nn.Module):
    """
    vCLUB for dynamic-dynamic: U_{1:T}, V_{1:T}
    I(U;V) <= E_{p(u,v)}[log q(u|v)] - E_{p(u)p(v)}[log q(u|v)]

    q(u_t | v_t) = N(u_t | mu(v_t), diag(sigma^2(v_t)))
    """
    def __init__(self, u_dim=96, v_dim=96, hidden=128):
        super().__init__()
        self.u_dim = int(u_dim)
        self.v_dim = int(v_dim)
        self.net = nn.Sequential(
            nn.Linear(v_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )
        self.mu = nn.Linear(hidden, u_dim)
        self.logvar = nn.Linear(hidden, u_dim)

    def log_q(self, u, v, reduce_time="mean", mask=None, reduce_dim="mean"):
        """
        u: (B,Du,T), v: (B,Dv,T)
        mask: (B,T) 0/1 (1=valid)
        reduce_dim: "mean" or "sum"  (Du 차원 축약 방식)
        return: (B,) aggregated log q(u|v)
        """
        if u.dim() != 3 or v.dim() != 3:
            raise ValueError("u and v must be 3D tensors (B,D,T)")
        if u.size(0) != v.size(0) or u.size(2) != v.size(2):
            raise ValueError(f"shape mismatch: u={tuple(u.shape)}, v={tuple(v.shape)}")

        B, Du, T = u.shape
        u_seq = u.transpose(1, 2)  # (B,T,Du)
        v_seq = v.transpose(1, 2)  # (B,T,Dv)

        h = self.net(v_seq)            # (B,T,H)
        mu = self.mu(h)                # (B,T,Du)
        logvar = self.logvar(h).clamp(-8.0, 8.0)  # (B,T,Du)

        ll = -0.5 * (math.log(2 * math.pi) + logvar) - 0.5 * ((u_seq - mu) ** 2) / logvar.exp()
        # ll: (B,T,Du)

        # --- dim reduce ---
        if reduce_dim == "sum":
            ll = ll.sum(dim=-1)   # (B,T)
        else:
            ll = ll.mean(dim=-1)  # (B,T)

        # --- time mask ---
        if mask is not None:
            mask_f = mask.to(ll.device).float()
            if mask_f.shape != (B, T):
                raise ValueError(f"mask must be (B,T), got {tuple(mask_f.shape)} vs {(B,T)}")
            ll = ll * mask_f
            denom = mask_f.sum(dim=1).clamp(min=1.0)  # (B,)
        else:
            denom = ll.new_full((B,), float(T))

        # --- time reduce ---
        if reduce_time == "sum":
            out = ll.sum(dim=1)               # (B,)
        else:  # mean
            out = ll.sum(dim=1) / denom       # (B,)

        return out

    def ll_loss(self, u, v, reduce_time="mean", mask=None, reduce_dim="mean"):
        # LLL = - E_{p(u,v)}[log q(u|v)]
        return -self.log_q(u, v, reduce_time=reduce_time, mask=mask, reduce_dim=reduce_dim).mean()

    def mi_upper(self, u, v, K=8, reduce_time="mean", mask=None, reduce_dim="mean"):
        """
        K-negative sampled vCLUB upper bound
        u: (B,Du,T), v: (B,Dv,T)
        mask: (B,T)
        """
        if u.size(0) < 2:
            return u.new_tensor(0.0)

        B, _, T = u.shape
        device = u.device
        K = max(1, int(K))

        # positive (u_i, v_i)
        pos = self.log_q(u, v, reduce_time=reduce_time, mask=mask, reduce_dim=reduce_dim).mean()

        # negative: (u_i, v_j)
        idx = torch.randint(0, B, (B, K), device=device)
        i_idx = torch.arange(B, device=device).unsqueeze(1)
        idx = torch.where(idx == i_idx, (idx + 1) % B, idx)  # avoid j==i

        # v_neg: (B*K, Dv, T)
        v_neg = v[idx.reshape(-1)]

        # u_rep: (B*K, Du, T)
        u_rep = u.unsqueeze(1).expand(B, K, u.size(1), T).reshape(-1, u.size(1), T)

        # --- mask 교집합 (u_i length ∩ v_j length) ---
        if mask is not None:
            mask_u = mask.to(device).float()                 # (B,T)
            mask_u_rep = mask_u.unsqueeze(1).expand(B, K, T).reshape(-1, T)  # (B*K,T)
            mask_v_neg = mask_u[idx.reshape(-1)]             # (B*K,T)  (mask of v_j)
            pair_mask = mask_u_rep * mask_v_neg              # intersection
        else:
            pair_mask = None

        neg = self.log_q(
            u_rep, v_neg,
            reduce_time=reduce_time,
            mask=pair_mask,
            reduce_dim=reduce_dim
        ).mean()

        return pos - neg
'''

class ClubGaussian(nn.Module):
    """
    Vector CLUB q(y|x) with Gaussian assumption.
    - base API: forward(x_samples, y_samples), learning_loss(x_samples, y_samples)
      expects x_samples,y_samples: (N, Dx/Dy)

    Added for this project:
    - mi_upper(u, v, K=..., mask=..., reduce_dim=...)
    - ll_loss(u, v, mask=..., reduce_dim=...)
      expects u,v: (B, D, T) for dynamic-dynamic (tp)
      where we interpret q(u|v) => x=v, y=u
    """
    def __init__(self, x_dim, y_dim, hidden_size, max_samples=2048):
        super().__init__()
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.max_samples = int(max_samples)

        self.p_mu = nn.Sequential(
            nn.Linear(self.x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.y_dim),
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(self.x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.y_dim),
            nn.Tanh(),  # logvar in (-1,1) for stability (original impl)
        )

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    # ---------- original CLUB style ----------
    def forward(self, x_samples, y_samples):
        """
        returns scalar MI upper estimate (all-pairs negative, O(N^2))
        NOTE: kept for compatibility; not used in training here (we use K-negative).
        """
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = - (mu - y_samples) ** 2 / 2.0 / logvar.exp()  # (N,Dy)

        prediction_1 = mu.unsqueeze(1)          # (N,1,Dy)
        y_samples_1  = y_samples.unsqueeze(0)   # (1,N,Dy)

        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.0 / logvar.exp()  # (N,Dy)

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):
        """
        unnormalized loglikelihood (original)
        """
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)

    # ---------- project helpers ----------
    def _seq_to_samples(self, x, y, mask=None):
        """
        x,y: (B,D,T) or already (N,D)
        mask: (B,T) float/bool, 1=valid (optional)

        returns:
          x_s, y_s : (N,D)
        """
        if x.dim() == 3:
            # (B,D,T)->(B,T,D)->(N,D)
            B, Dx, T = x.shape
            x_s = x.transpose(1, 2).reshape(-1, Dx)
            y_s = y.transpose(1, 2).reshape(-1, y.size(1))
            if mask is not None:
                m = mask.reshape(-1).bool()
                x_s = x_s[m]
                y_s = y_s[m]
        elif x.dim() == 2:
            x_s, y_s = x, y
        else:
            raise ValueError(f"expected x dim 2 or 3, got {x.dim()}")

        # subsample to avoid explosion
        N = x_s.size(0)
        if self.max_samples and N > self.max_samples:
            idx = torch.randperm(N, device=x_s.device)[: self.max_samples]
            x_s = x_s[idx]
            y_s = y_s[idx]

        return x_s, y_s

    def ll_loss(self, u, v, mask=None, reduce_dim="mean"):
        """
        LLL term: -E[log q(u|v)]  (train CLUB parameters)
        u,v: (B,D,T) for tp (dynamic-dynamic)
        """
        # q(u|v) => x=v, y=u
        x_s, y_s = self._seq_to_samples(v, u, mask=mask)
        loss = self.learning_loss(x_s, y_s)  # scalar

        # scale option (to align with your reduce_dim="mean" convention)
        if reduce_dim == "mean":
            loss = loss / float(self.y_dim)
        return loss

    def mi_upper(self, u, v, K=8, mask=None, reduce_dim="mean"):
        """
        MI upper estimate with K-negative sampling (O(N*K)).
        u,v: (B,D,T)
        """
        x_s, y_s = self._seq_to_samples(v, u, mask=mask)  # q(u|v) => x=v, y=u
        N = x_s.size(0)
        if N < 2:
            return x_s.new_tensor(0.0)

        mu, logvar = self.get_mu_logvar(x_s)  # (N,Dy)
        var = logvar.exp()

        # positive: (x_i, y_i)
        pos = - (mu - y_s) ** 2 / 2.0 / var  # (N,Dy)

        # negative: (x_i, y_j)
        K = max(1, int(K))
        idx = torch.randint(0, N, (N, K), device=x_s.device)
        i_idx = torch.arange(N, device=x_s.device).unsqueeze(1)
        idx = torch.where(idx == i_idx, (idx + 1) % N, idx)

        y_neg = y_s[idx]                 # (N,K,Dy)
        mu_e  = mu.unsqueeze(1)          # (N,1,Dy)
        var_e = var.unsqueeze(1)         # (N,1,Dy)

        neg = - (y_neg - mu_e) ** 2 / 2.0 / var_e  # (N,K,Dy)
        neg = neg.mean(dim=1)                      # (N,Dy)  평균 over K

        # reduce over dim
        if reduce_dim == "mean":
            pos_r = pos.mean(dim=-1)   # (N,)
            neg_r = neg.mean(dim=-1)   # (N,)
        else:
            pos_r = pos.sum(dim=-1)
            neg_r = neg.sum(dim=-1)

        return (pos_r - neg_r).mean()


class ARClubGaussian(nn.Module):
    """
    Autoregressive vCLUB for I(U_{1:T}; V_static)

    q(u_{1:T} | v) = Π_t N(u_t | mu_t, diag(sigma_t^2))
    mu_t, sigma_t from a causal RNN that sees (u_{<t}, v)

    MI upper:
      I(U;V) ≤ E_{p(u,v)}[log q(u|v)] - E_{p(u)p(v)}[log q(u|v)]

    LLL (식 5):
      LLL = - E_{p(u,v)}[log q(u|v)]
    """
    def __init__(self, u_dim=96, v_dim=96, hidden=128):
        super().__init__()
        self.v_to_h0 = nn.Sequential(
            nn.Linear(v_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.gru = nn.GRU(
            input_size=u_dim,
            hidden_size=hidden,
            batch_first=True,
        )
        self.mu = nn.Linear(hidden, u_dim)
        self.logvar = nn.Linear(hidden, u_dim)

    def _shift_right(self, u_seq):
        # u_seq: (B, T, Du)
        z0 = torch.zeros_like(u_seq[:, :1, :])
        return torch.cat([z0, u_seq[:, :-1, :]], dim=1)  # (B,T,Du)

    def log_q(self, u, v, reduce_time="mean", mask=None, reduce_dim="mean"):
        """
        u: (B, Du, T)
        v: (B, Dv)   (static)
        return: (B,) aggregated log q(u_{1:T} | v)
        """
        u_seq = u.transpose(1, 2)  # (B,T,Du)

        h0 = self.v_to_h0(v).unsqueeze(0)  # (1,B,H)
        causal_u = self._shift_right(u_seq)    # (B,T,Du)
        h, _ = self.gru(causal_u, h0)          # (B,T,H)

        mu = self.mu(h)                            # (B,T,Du)
        logvar = self.logvar(h).clamp(-8.0, 8.0)   # (B,T,Du)

        ll = -0.5 * (math.log(2 * math.pi) + logvar) - 0.5 * ((u_seq - mu) ** 2) / logvar.exp()
        # ll: (B,T,Du)

        # --- dim reduce ---
        if reduce_dim == "sum":
            ll = ll.sum(dim=-1)   # (B,T)
        else:
            ll = ll.mean(dim=-1)  # (B,T)

        # 길이 마스크
        if mask is not None:
            mask_f = mask.to(ll.device).float()      # (B,T)
            ll = ll * mask_f
            denom = mask_f.sum(dim=1).clamp(min=1.0)  # (B,)
        else:
            denom = ll.new_full((ll.size(0),), ll.size(1))  # (B,) = T

        if reduce_time == "sum":
            out = ll.sum(dim=1)               # (B,)
        else:
            out = ll.sum(dim=1) / denom       # (B,)

        return out

    def ll_loss(self, u, v, reduce_time="mean", mask=None, reduce_dim="mean"):
        return -self.log_q(u, v, reduce_time=reduce_time, mask=mask, reduce_dim=reduce_dim).mean()

    def mi_upper(self, u, v, K=8, reduce_time="mean", mask=None, reduce_dim="mean"):
        """
        u: (B,Du,T)
        v: (B,Dv) static
        mask: (B,T) for u
        """
        B = u.size(0)
        if B < 2:
            return u.new_tensor(0.0)

        device = u.device
        K = max(1, int(K))

        # positive: (u_i, v_i)
        pos = self.log_q(u, v, reduce_time=reduce_time, mask=mask, reduce_dim=reduce_dim).mean()

        # negative: (u_j, v_i)
        idx = torch.randint(0, B, (B, K), device=device)
        i_idx = torch.arange(B, device=device).unsqueeze(1)
        idx = torch.where(idx == i_idx, (idx + 1) % B, idx)

        u_neg = u[idx.reshape(-1)]  # (B*K,Du,T)
        v_rep = v.unsqueeze(1).expand(B, K, v.size(1)).reshape(-1, v.size(1))  # (B*K,Dv)

        # mask는 반드시 u_neg에 맞춰야 함
        if mask is not None:
            mask_neg = mask.to(device)[idx.reshape(-1)]  # (B*K,T)
        else:
            mask_neg = None

        neg = self.log_q(u_neg, v_rep, reduce_time=reduce_time, mask=mask_neg, reduce_dim=reduce_dim).mean()
        return pos - neg
