import os
import math
import json
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
import nemo.collections.asr as nemo_asr
from typing import Optional, Dict

class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class SimpleClsHead(nn.Module):
    """시간축 평균 후 분류. 입력: (B, C, T) or (B, T, C)"""
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, feat):
        # (B,C,T) 또는 (B,T,C) 허용
        if feat.dim() != 3:
            raise ValueError("feat must be 3D")
        if feat.size(1) < feat.size(2):
            # (B, C, T) 가정
            x = feat.mean(dim=-1)  # (B, C)
        else:
            # (B, T, C)
            x = feat.mean(dim=1)   # (B, C)
        return self.net(x)


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
        z_t = self.enc(tch_feat).detach()
        rec = self.dec(z_t)
        ae = self.mse(rec, tch_feat)
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
        use_disent: bool = True,
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
        self.stu_proj = nn.Conv1d(self.dim_s, self.dim_t, kernel_size=1, bias=True)

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
            self.spk_backbone = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(self.latent_dim),
                nn.ReLU(inplace=True),

                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, dilation=2, padding=2, bias=False),
                nn.BatchNorm1d(self.latent_dim),
                nn.ReLU(inplace=True),

                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, dilation=3, padding=3, bias=False),
                nn.BatchNorm1d(self.latent_dim),
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

        # ===== Prosody (GST) =====
        n_mels = getattr(getattr(cfg, "preprocessor", type("x", (object,), {})()), "features", 80)
        self.use_prosody = True
        self.pros_ref = GlobalProsodyReferenceEncoder(
            n_mels=n_mels, channels=(32, 32, 64, 64, 128, 128), gru_dim=96
        )
        self.pros_gst = GlobalStyleTokenLayer(
            num_tokens=getattr(cfg, "disen_gst_tokens", 10),
            token_dim=getattr(cfg, "disen_gst_token_dim", 96),
            num_heads=getattr(cfg, "disen_gst_heads", 4),
            ref_dim=getattr(cfg, "disen_ref_dim", 96),
        )
        self.pros_proj = nn.Linear(getattr(cfg, "disen_gst_token_dim", 96), self.latent_dim)

        # ===== MI 추정기 (vCLUB) =====
        self.mi_weight = getattr(cfg, "disen_mi_weight", 1.0)        # λ_MI
        self.mi_pairs = getattr(cfg, "disen_mi_pairs", "ts,tp,ps")   # 사용 쌍

        self.club_zs = ClubGaussian(u_dim=self.latent_dim, v_dim=self.latent_dim)
        self.club_zp = ClubGaussian(u_dim=self.latent_dim, v_dim=self.latent_dim)
        self.club_ps = ClubGaussian(u_dim=self.latent_dim, v_dim=self.latent_dim)
        
        # ===== MI 추정기 (AR-vCLUB) =====
        self.club_ts = ARClubGaussian(u_dim=self.latent_dim, v_dim=self.latent_dim, hidden=getattr(cfg, "club_hidden", 96))
        self.club_tp = ARClubGaussian(u_dim=self.latent_dim, v_dim=self.latent_dim, hidden=getattr(cfg, "club_hidden", 96))

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
                nn.BatchNorm1d(self.latent_dim),
                nn.ReLU(inplace=True),

                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, dilation=2, padding=2, bias=False),
                nn.BatchNorm1d(self.latent_dim),
                nn.ReLU(inplace=True),

                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, dilation=3, padding=3, bias=False),
                nn.BatchNorm1d(self.latent_dim),
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
        
        # === Student speaker adversary (GRL) ===
        self.use_stu_spk_adv = bool(getattr(cfg, "use_stu_spk_adv", True))
        self.stu_spk_adv_lambda_max = float(getattr(cfg, "stu_spk_adv_lambda_max", 0.1))
        self.stu_spk_adv_warmup_steps = int(getattr(cfg, "stu_spk_adv_warmup_steps", 2000))

        # 상위 레이어만 쓰고 싶으면 hook 기반으로 layer index 사용 (옵션)
        self.stu_adv_layers = getattr(cfg, "stu_adv_layers", None)  # 예: [15,16] (1-based)

        if self.num_spk > 1 and self.use_stu_spk_adv:
            # 가벼운 head 권장 (너무 세면 CTC 망가짐)
            self.stu_spk_adv_backbone = nn.Sequential(
                nn.Conv1d(self.dim_s, self.dim_s, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(self.dim_s),
                nn.ReLU(inplace=True),
            )
            hidden = int(getattr(cfg, "stu_spk_adv_hidden", self.dim_s))
            self.stu_spk_adv_cls = nn.Sequential(
                nn.Linear(self.dim_s * 2, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(getattr(cfg, "stu_spk_adv_dropout", 0.2))),
                nn.Linear(hidden, self.num_spk),
            )
        else:
            self.stu_spk_adv_backbone = None
            self.stu_spk_adv_cls = None
    
        train_manifest = self.cfg.train_ds.manifest_filepath
        self._load_manifest_speakers(train_manifest)

    # ------ Feature hooks ------
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
    
    # ------ Forward (Student path) ------
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
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
        self._last_enc = s_enc                     # (B, T, C_s)

        return logp, s_len, greedy

    # ------ Teacher one-pass (encoder + decoder) ------
    def _run_teacher(self, signal, sig_len):
        """그림 기준 왼쪽 Teacher Encoder + Decoder 한 번만 실행."""
        self.tch_feats.clear()
        with torch.no_grad():
            t_proc, t_len = self.teacher.preprocessor(
                input_signal=signal, length=sig_len
            )
            t_enc, _ = self.teacher.encoder(
                audio_signal=t_proc, length=t_len
            )  # (B, T, C_t)
            t_logp = self.teacher.decoder(encoder_output=t_enc)
        # 캐시
        self._tch_logp = t_logp
        self._tch_last = self._to_BCT(t_enc)       # (B, C_t, T)

    # ------ Loss helpers ------
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
            s_proj = self.stu_proj(s)              # (B, C_t, T)
            losses.append(F.mse_loss(s_proj, t))
        return sum(losses) / L

    def _generative_kd(self):
        """
        그림 기준:
        - Student 마지막 레이어 feature (encoder 출력)
        - Teacher Text Encoder로 얻은 Text feature
        사이에 Flow Matching / DiffKD 적용.
        """
        if (not self.use_flow and not self.use_diffkd) or (self._last_enc is None):
            return torch.tensor(0.0, device=self.device)

        if self._txt_emb is None:
            # Text Encoder가 아직 안 돌았다면 0
            return torch.tensor(0.0, device=self.device)

        # Student last: (B, T, C_s) -> (B, C_s, T)
        s_raw = self._last_enc
        s = self._ensure_BCT(s_raw, C_expected=self.dim_s)   # (B,C_s,T)
        # Teacher text feature: (B, C_t, T)
        t_raw = self._txt_emb.detach()
        t = self._ensure_BCT(t_raw, C_expected=self.latent_dim)  # (B,96,T) or (B,dim_t,T) depending on what you store

        flow_loss = torch.tensor(0.0, device=self.device)
        diff_loss = torch.tensor(0.0, device=self.device)
        if self.use_flow and self.flow is not None:
            flow_loss = self.flow(s, t)
        if self.use_diffkd and self.diffkd is not None:
            diff_loss = self.diffkd(s, t)
        
        return flow_loss, diff_loss

    # ===== Teacher Text/Speaker/Prosody embedding 생성 =====
    def _make_embeddings(self, speaker_ids):
        """
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
        spk_feat = self.spk_backbone(spk_emb) if self.spk_backbone is not None else spk_emb  # (B,96,T)
        spk_mean = spk_feat.mean(dim=-1)   # (B,96)
        spk_std  = spk_feat.std(dim=-1)    # (B,96)
        spk_stat = torch.cat([spk_mean, spk_std], dim=-1)  # (B,192) # TODO: 이게 맞아? 
        spk_stat = self.spk_stat_proj(spk_stat)            # (B,96)
        
        # ----- Prosody (GST) -----
        T = txt_emb.size(-1)
        if self.use_prosody and (self._last_mel is not None):   # (B, mel_dim=80, T)
            # utterance-level의 style로 구현한 상태
            # 1) 멜에서 글로벌 ref 추출: (B, 96)
            ref = self.pros_ref(self._last_mel)          # (B, ref_dim=96)
            # 2) GST로 스타일 토큰 조합: (B, token_dim=96)
            p_raw = self.pros_gst(ref)                   # (B, token_dim=96)
            # 3) latent_dim으로 proj: (B, 96)
            pros_stat = self.pros_proj(p_raw)            # (B, latent_dim=96) <-- 이게 진짜 static
            # 4) frame-wise로 broadcast: (B,96,T)
            pros_emb = pros_stat.unsqueeze(-1).expand(-1, -1, T)

        else:
            pros_stat = torch.zeros((txt_emb.size(0), self.latent_dim), device=txt_emb.device)
            pros_emb = torch.zeros_like(txt_emb) # (B, 96, T)

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
                spk_std  = spk_feat.std(dim=-1)               # (B, latent_dim)
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
            )
        
        return {
            "txt_emb": txt_emb,         # (B,96,T) dynamic
            "spk_emb": spk_emb,         # (B,96,T) 기존 유지(CE/XAI용)
            "pros_emb": pros_emb,       # (B,96,T) 기존 유지(XAI용)

            "spk_stat": spk_stat,       # (B,96) static (MI용)
            "pros_stat": pros_stat,     # (B,96) static (MI용)

            "spk_ce": spk_ce,
            "spk_acc": spk_acc,
            "rec_txt": rec_txt,
            "rec_spk": rec_spk,
            "txt_probe_ce": txt_probe_ce,
            "txt_probe_acc": txt_probe_acc,
        }
    '''
    def _mi_loss(self, embs):
        """
        vCLUB 상한 합: I(z;s)+I(z;p)+I(p;s)
        """
        if embs is None:
            return torch.tensor(0.0, device=self.device)
        txt_emb, spk_emb, pros_emb = embs["txt_emb"], embs["spk_emb"], embs["pros_emb"]
        pairs = set([t.strip() for t in self.mi_pairs.split(",") if t.strip()])
        mi = torch.tensor(0.0, device=txt_emb.device)
        cnt = 0

        if "ts" in pairs:
            mi = mi + self.club_zs.mi_upper(u=txt_emb, v=spk_emb); cnt += 1
        if "tp" in pairs:
            mi = mi + self.club_zp.mi_upper(u=txt_emb, v=pros_emb); cnt += 1
        if "ps" in pairs:
            mi = mi + self.club_ps.mi_upper(u=pros_emb, v=spk_emb); cnt += 1

        if cnt > 0:
            mi = mi / cnt
        return mi
    '''
    
    def _mi_loss(self, embs):
        """
        return:
        mi_upper_avg: 평균 MI upper (식 4 추정치)
        lll_avg: 평균 LLL (식 5)
        """
        if embs is None:
            z = torch.tensor(0.0, device=self.device)
            return z, z

        txt = embs["txt_emb"]        # (B,96,T) dynamic
        spk = embs["spk_stat"]       # (B,96)   static
        pros = embs["pros_stat"]     # (B,96)   static

        pairs = set([t.strip() for t in self.mi_pairs.split(",") if t.strip()])

        mi_sum  = torch.tensor(0.0, device=txt.device)
        lll_sum = torch.tensor(0.0, device=txt.device)
        cnt = 0

        # --- ts: I(txt_{1:T} ; spk_static) ---
        if "ts" in pairs:
            # (5) LLL: CLUB(q) 네트워크 학습용 (u,v는 detach)
            lll_sum = lll_sum + self.club_ts.ll_loss(txt.detach(), spk.detach())

            # (4) MI upper: 표현(txt,spk)이 덜 얽히게 만드는 loss
            # 이때 CLUB 파라미터는 고정하고(gradient 막고) txt/spk 쪽으로만 gradient 흐르게
            self._freeze_params(self.club_ts, True)
            mi_sum = mi_sum + self.club_ts.mi_upper(txt, spk)
            self._freeze_params(self.club_ts, False)

            cnt += 1

        # --- tp: I(txt_{1:T} ; pros_static) ---
        if "tp" in pairs:
            lll_sum = lll_sum + self.club_tp.ll_loss(txt.detach(), pros.detach())

            self._freeze_params(self.club_tp, True)
            mi_sum = mi_sum + self.club_tp.mi_upper(txt, pros)
            self._freeze_params(self.club_tp, False)

            cnt += 1

        if cnt > 0:
            mi_sum  = mi_sum / cnt
            lll_sum = lll_sum / cnt

        return mi_sum, lll_sum

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
            
            # sample_id -> speaker_id lookup
            speaker_ids = self.manifest_speakers.to(sample_ids.device)[sample_ids]
        elif len(batch) == 4:
            signal, sig_len, y, ylen = batch
            speaker_ids = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        
        assert speaker_ids.min().item() >= -1
        assert speaker_ids.max().item() < self.num_spk
        
        # 1) Student forward
        logp, enc_len, _ = self.forward(
            input_signal=signal, input_signal_length=sig_len
        )

        # 2) Teacher forward (한 번만)
        self._run_teacher(signal, sig_len)

        total = torch.tensor(0.0, device=logp.device)
        
        # (추가) Student speaker adversary
        if self.use_stu_spk_adv and (self.num_spk > 1):
            adv_ce_total = torch.tensor(0.0, device=logp.device)
            adv_acc_total = torch.tensor(0.0, device=logp.device)
            for stu_feat in self.stu_feats:
                adv_ce, adv_acc = self._student_spk_adv_loss(stu_feat, speaker_ids)    
                adv_acc_total += adv_acc
                adv_ce_total += adv_ce            
            adv_acc_mean = adv_acc_total / len(self.stu_feats)
            adv_ce_mean = adv_ce_total / len(self.stu_feats)
            if adv_ce_mean is not None:
                self.log("adv/stu_spk_ce_mean", adv_ce_mean, on_step=False, on_epoch=True)
                self.log("adv/stu_spk_acc_mean", adv_acc_mean, on_step=False, on_epoch=True)
                self.log("adv/grl_lambda", torch.tensor(self._grl_lambda(), device=logp.device),
                        on_step=False, on_epoch=True)

                # GRL이 이미 lam을 적용하므로 보통은 그냥 더함
                # 필요하면 cfg로 loss_weight 따로 두고 작게(예: 0.1) 곱해라
                total = total + adv_ce_mean

        # 3) Factorization embeddings + MI/Rec/Speaker CE
        embs = self._make_embeddings(speaker_ids)              # {'txt_emb','spk_emb','pros_emb','spk_ce','rec_*'}
        if self.use_disent:
            # MI term
            mi_upper, lll = self._mi_loss(embs)
            # mi_loss = torch.clamp(mi_raw, min=0.0)
            self.log("train/mi_upper", mi_upper, on_epoch=True)
            # self.log("train/mi_upper", mi_loss, on_epoch=True)
            self.log("train/lll", lll, on_epoch=True)
            total = total + self.mi_weight * (mi_upper + self.lll_weight * lll)


            # Reconstruction loss
            rec_txt = embs["rec_txt"]
            rec_spk = embs["rec_spk"]
            self.log("train/rec_txt", rec_txt, on_epoch=True)
            self.log("train/rec_spk", rec_spk, on_epoch=True)
            total = total + self.rec_txt_lambda * rec_txt + self.rec_spk_lambda * rec_spk

            # Speaker CE & ACC
            spk_ce = embs["spk_ce"]
            spk_acc = embs.get("spk_acc", None)

            if spk_ce is not None and torch.is_tensor(spk_ce):
                self.log("train/spk_ce", spk_ce, on_step=False, on_epoch=True)
                total = total + self.disen_spk_ce_lambda * spk_ce

            if spk_acc is not None and torch.is_tensor(spk_acc):
                self.log("train/spk_acc", spk_acc, on_epoch=True)
        else:
            # disent off일 때는 관련 loss를 0으로만 로깅 (원하면 생략해도 됨)
            zero = torch.tensor(0.0, device=logp.device)
            self.log("train/mi_upper", zero, on_step=False, on_epoch=True)
            self.log("train/rec_txt", zero, on_step=False, on_epoch=True)
            self.log("train/rec_spk", zero, on_step=False, on_epoch=True)
            self.log("train/spk_ce", zero, on_step=False, on_epoch=True)
            # self._txt_emb 도 이때는 사용 안 하도록 보장하고 싶으면 명시적으로 초기화
            self._txt_emb = None
        
        # 4) Generative KD (FM / DF)
        if self.use_flow or self.use_diffkd:
            flow_loss, diff_loss = self._generative_kd()
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
        std  = x.std(dim=-1)
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
    
    def _ensure_BCT(self, x, C_expected=None):
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
    
    def _grl_lambda(self):
        # 0 -> lambda_max 선형 warmup
        if self.stu_spk_adv_warmup_steps <= 0:
            return self.stu_spk_adv_lambda_max
        step = int(getattr(self, "global_step", 0))
        r = min(1.0, step / float(self.stu_spk_adv_warmup_steps))
        return self.stu_spk_adv_lambda_max * r

    def _get_student_adv_rep(self, s_enc):
        """
        반환: rep (B, C_s, T)
        - 기본: s_enc를 사용
        - stu_adv_layers가 있으면 hook된 self.stu_feats에서 layer 평균 사용
        """
        # 기본: last output
        rep = self._ensure_BCT(s_enc, C_expected=self.dim_s)  # (B,C,T)

        if self.stu_adv_layers is None:
            return rep

        if not self.stu_feats:
            return rep

        L = len(self.stu_feats)
        idxs = self._prepare_layer_indices(self.stu_adv_layers, L, default_low=False)
        stack = torch.stack([self.stu_feats[i] for i in idxs], dim=0)  # (K,B,C,T)
        rep = stack.mean(dim=0)  # (B,C,T)
        return rep

    def _student_spk_adv_loss(self, stu_feat, speaker_ids):
        if (self.stu_spk_adv_cls is None) or (speaker_ids is None):
            return None, None

        valid_mask = (speaker_ids >= 0) & (speaker_ids < self.num_spk)
        if not valid_mask.any():
            return None, None

        lam = self._grl_lambda()
        
        stu_feat = self._ensure_BCT(stu_feat, C_expected=self.dim_s)  # (B,C,T)
    

        # GRL 적용: speaker CE가 student rep를 "speaker 못맞추게" 업데이트하도록
        rep_grl = GradReverseFn.apply(stu_feat, lam)
        if self.stu_spk_adv_backbone is not None:
            rep_grl = self.stu_spk_adv_backbone(rep_grl)  # (B,C,T)

        mean = rep_grl.mean(dim=-1)  # (B,C)
        std  = rep_grl.std(dim=-1)   # (B,C)
        utt  = torch.cat([mean, std], dim=-1)  # (B,2C)

        logits = self.stu_spk_adv_cls(utt)  # (B,num_spk)
        target = speaker_ids.clamp(min=0).long()

        ce = F.cross_entropy(logits[valid_mask], target[valid_mask])
        preds = logits.argmax(dim=-1)
        acc = (preds[valid_mask] == target[valid_mask]).float().mean()
        return ce, acc

    def _load_manifest_speakers(self, manifest_path: str):
        spk = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                spk.append(int(obj.get("speaker", -1)))
        # (N,) 텐서로 들고 있기
        self.manifest_speakers = torch.tensor(spk, dtype=torch.long)  # CPU에 둬도 OK


class StyleTokenLayer(nn.Module):
    def __init__(self, num_tokens=10, token_dim=96, num_heads=4, ref_dim=96):
        super().__init__()
        assert token_dim % num_heads == 0
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.tokens = nn.Parameter(torch.randn(num_tokens, token_dim))
        self.query_proj = nn.Linear(ref_dim, token_dim, bias=False)
        self.key_proj   = nn.Linear(token_dim, token_dim, bias=False)
        self.v = nn.Linear(token_dim, num_heads, bias=False)
        
        self.last_attn = None  # (B, N_tokens, H)

    def forward(self, ref_seq):   # ref_seq: (B, D, T)
        B, D, T = ref_seq.shape
        # (B,T,D)
        ref_flat = ref_seq.transpose(1, 2).contiguous().view(B * T, D)

        q = torch.tanh(self.query_proj(ref_flat))      # (B*T, token_dim)
        k = torch.tanh(self.key_proj(self.tokens))     # (N, token_dim)

        N = k.size(0)
        q_exp = q.unsqueeze(1).expand(B*T, N, self.token_dim)  # (B*T,N,D)
        k_exp = k.unsqueeze(0).expand(B*T, N, self.token_dim)  # (B*T,N,D)
        s = torch.tanh(q_exp + k_exp)                           # (B*T,N,D)
        logits = self.v(s)                                      # (B*T,N,H)
        attn = torch.softmax(logits, dim=1)                     # (B*T,N,H)
        
        # XAI: 저장 (detach + cpu)
        self.last_attn = attn.detach().cpu()

        style = torch.einsum("bnh,nd->bdh", attn, self.tokens)  # (B*T,D,H)
        style = style.mean(dim=-1)                              # (B*T,D)

        # 다시 (B,D,T)로 복원
        style = style.view(B, T, D).transpose(1, 2).contiguous()  # (B,D,T)
        return style


class ProsodyReferenceEncoder(nn.Module): # TODO 이거 이상함 점검 필요 차원수 잘 봐보셈
    def __init__(self, n_mels=80, channels=(32,32,64,64,128,128), gru_dim=96):
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

        # n_mels 방향 축소량 계산 (stride=2, k=3, pad=1 기준)
        def shrink(L, k=3, s=2, p=1, n=self.K):
            for _ in range(n):
                L = (L - k + 2*p)//s + 1
            return L
        self.mels_after = shrink(n_mels)  # F' (= n_mels // 2^K 근사)

        self.gru = nn.GRU(
            input_size=channels[-1] * self.mels_after,  # C * F'
            hidden_size=gru_dim,
            batch_first=True
        )
        self.last_out = None   # (B, 96, T) 저장용

    def forward(self, mel):  # mel: (B, n_mels, T)
        B, n_mels, T = mel.shape

        # 1) 2D conv
        x = mel.transpose(1, 2).unsqueeze(1)      # (B,1,T,n_mels)
        z = self.conv(x)                          # (B,C,T',F')
        B, C, Tp, Fp = z.shape

        # 2) (B,T',C*F')로 reshape
        z = z.permute(0, 2, 1, 3).contiguous()    # (B,T',C,F')
        z = z.view(B, Tp, C * Fp)                 # (B,T',C*F')

        # 3) GRU sequence output 사용
        out, _ = self.gru(z)                      # out: (B,T',gru_dim=96)

        # 4) (B,96,T') → (B,96,T) 로 upsample
        out = out.transpose(1, 2)                 # (B,96,T')
        if Tp != T:
            out = F.interpolate(out, size=T, mode="linear", align_corners=False)    # (B,96,T)
        
        self.last_out = out.detach().cpu()        # XAI 저장
        return out                                # (B,96,T)


class GlobalStyleTokenLayer(nn.Module):
    def __init__(self, num_tokens=10, token_dim=96, num_heads=4, ref_dim=96):
        super().__init__()
        assert token_dim % num_heads == 0
        self.token_dim = token_dim
        self.num_heads = num_heads

        # 스타일 토큰 뱅크 (N_tokens, D)
        self.tokens = nn.Parameter(torch.randn(num_tokens, token_dim))

        # ref_emb (B, ref_dim) → query
        self.query_proj = nn.Linear(ref_dim, token_dim, bias=False)
        # tokens (N, token_dim) → key
        self.key_proj   = nn.Linear(token_dim, token_dim, bias=False)

        # head별 스칼라 점수 v_h
        self.v = nn.Linear(token_dim, num_heads, bias=False)

        # XAI용: 마지막 attention 저장 (B, N_tokens, H)
        self.last_attn = None

    def forward(self, ref_emb):  # ref_emb: (B, ref_dim=96)
        # 1) query / key 생성
        q = torch.tanh(self.query_proj(ref_emb))      # (B, token_dim)
        k = torch.tanh(self.key_proj(self.tokens))    # (N, token_dim)

        B, D = q.size()
        N = k.size(0)

        # 2) additive attention (global, time 축 없음)
        # (B,1,D) + (1,N,D) → (B,N,D)
        q_exp = q.unsqueeze(1).expand(B, N, D)
        k_exp = k.unsqueeze(0).expand(B, N, D)
        s = torch.tanh(q_exp + k_exp)                 # (B,N,D)

        # (B,N,D) → (B,N,H)
        logits = self.v(s)                            # (B, N, H)
        attn = torch.softmax(logits, dim=1)           # (B, N, H)

        # XAI: token×head attention 저장
        self.last_attn = attn.detach().cpu()          # (B,N,H)

        # 3) 토큰 가중합 → head-wise 스타일 벡터
        # tokens: (N,D), attn: (B,N,H)
        style = torch.einsum("bnh,nd->bdh", attn, self.tokens)  # (B,D,H)
        style = style.mean(dim=-1)  # head 평균 → (B,D=token_dim)

        return style  # (B, token_dim)


class GlobalProsodyReferenceEncoder(nn.Module):
    def __init__(self, n_mels=80, channels=(32,32,64,64,128,128), gru_dim=96):
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

        # n_mels 방향 축소량 계산 (stride=2, k=3, pad=1 기준) → F'
        def shrink(L, k=3, s=2, p=1, n=self.K):
            for _ in range(n):
                L = (L - k + 2*p)//s + 1
            return L
        self.mels_after = shrink(n_mels)  # F'

        # GRU는 (time', C*F')를 입력받아서 마지막 hidden을 스타일 ref로 사용
        self.gru = nn.GRU(
            input_size=channels[-1] * self.mels_after,  # C * F'
            hidden_size=gru_dim,                        # ref_dim = 96
            batch_first=True
        )

        # XAI용: 마지막 글로벌 ref 벡터 저장 (B, gru_dim)
        self.last_out = None

    def forward(self, mel):  # mel: (B, n_mels, T)
        """
        출력: ref_emb (B, gru_dim=96)
        - T는 conv+GRU 안에서 요약되고, time 축은 사라짐 (글로벌 prosody)
        """
        B, n_mels, T = mel.shape

        # 1) 2D conv: time × freq → downsample된 time' × freq'
        x = mel.transpose(1, 2).unsqueeze(1)      # (B,1,T,n_mels)
        z = self.conv(x)                          # (B,C,T',F')
        B, C, Tp, Fp = z.shape

        # 2) (B,T',C*F')로 reshape 해서 GRU 인풋으로 사용
        z = z.permute(0, 2, 1, 3).contiguous()    # (B,T',C,F')
        z = z.view(B, Tp, C * Fp)                 # (B,T',C*F')

        # 3) GRU 마지막 hidden을 ref로 사용 (global prosody)
        _, h = self.gru(z)                        # h: (1,B,gru_dim)
        ref_emb = h[-1]                           # (B,gru_dim=96)

        # XAI 저장
        self.last_out = ref_emb.detach().cpu()

        return ref_emb                            # (B,96)


class ClubGaussian(nn.Module):
    """
    I(U;V) ≤ E_{p(u,v)}[ log q(u|v) ] - E_{p(u)p(v)}[ log q(u|v) ]
    q(u|v) = N(u | mu(v), diag(sigma^2(v)))
    """
    def __init__(self, u_dim=96, v_dim=96, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(v_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )
        self.mu = nn.Linear(hidden, u_dim)
        self.logvar = nn.Linear(hidden, u_dim)

    def log_q(self, u, v):
        # u,v shape: (B, latent_dim=96, T)
        v = v.transpose(1, 2)
        u = u.transpose(1, 2)
        # u: (B,Du), v:(B,Dv)  → log N(u|μ(v), Σ(v))
        h = self.net(v)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-8.0, max=8.0)
        # 로그우도(대각가우시안)
        ll = -0.5 * (math.log(2*math.pi) + logvar) - 0.5 * ((u - mu)**2) / logvar.exp()
        return ll.sum(dim=-1)  # (B, T, latent_dim=96)

    def mi_upper(self, u, v):
        # positive
        log_q_pos = self.log_q(u, v).mean()
        # negative with shuffled v
        v_shuffle = v[torch.randperm(v.size(0), device=v.device)]
        log_q_neg = self.log_q(u, v_shuffle).mean()
        return (log_q_pos - log_q_neg)

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
    def __init__(self, u_dim=96, v_dim=96, hidden=256):
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

    def log_q(self, u, v):
        """
        u: (B, Du, T)
        v: (B, Dv)   (static)
        return: (B, T)  log q(u_t | u_{<t}, v) summed over Du
        """
        # (B,Du,T) -> (B,T,Du)
        u_seq = u.transpose(1, 2)

        # init hidden from v
        h0 = self.v_to_h0(v).unsqueeze(0)  # (1,B,H)

        # GRU input is shifted u (so that at t, it has u_<t)
        causal_u = self._shift_right(u_seq)    # (B,T,Du)
        h, _ = self.gru(causal_u, h0)          # (B,T,H)

        mu = self.mu(h)                            # (B,T,Du)
        logvar = self.logvar(h).clamp(-8.0, 8.0)   # (B,T,Du)

        ll = -0.5 * (math.log(2 * math.pi) + logvar) - 0.5 * ((u_seq - mu) ** 2) / logvar.exp()
        ll = ll.sum(dim=-1)  # (B,T)
        return ll

    def ll_loss(self, u, v):
        # LLL = - E[log q(u|v)]
        return -self.log_q(u, v).mean()

    def mi_upper(self, u, v):
        # positive term
        log_q_pos = self.log_q(u, v).mean()

        # negative term: shuffle v across batch to approximate p(u)p(v)
        v_shuf = v[torch.randperm(v.size(0), device=v.device)]
        log_q_neg = self.log_q(u, v_shuf).mean()

        return (log_q_pos - log_q_neg)

'''
class ClubGaussian(nn.Module): # 지금 스피커에서 텍스트 뽑고 있음.... # v: speaker, u: text
    """
    I(U;V) ≤ E_{p(u,v)}[ log q(u|v) ] - E_{p(u)p(v)}[ log q(u|v) ]
    q(u|v) = N(u | mu(v), diag(sigma^2(v)))
    """
    def __init__(self, u_dim=96, v_dim=96, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(v_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )
        self.mu = nn.Linear(hidden, u_dim) # <- 가우시안으로 두면 이거 필요없이 mean 취하면 됨.
        self.logvar = nn.Linear(hidden, u_dim)

    def log_q(self, u, v): # 텍스트에서 스피커를 뽑아내는게 더 자연스러움
        # u,v shape: (B, latent_dim=96, T)
        v = v.transpose(1, 2)
        u = u.transpose(1, 2)
        # u: (B,Du), v:(B,Dv)  → log N(u|μ(v), Σ(v))
        h = self.net(v) # <- 이런식으로 과거 정보도 넣어줘야 함
        mu = self.mu(h) # <- 이거 필요 없이 mean 취한게...
        logvar = self.logvar(h).clamp(min=-8.0, max=8.0) #<- 이것도 직접 측정해야지 왜 네트워크 씀
        # 로그우도(대각가우시안)
        ll = -0.5 * (math.log(2*math.pi) + logvar) - 0.5 * ((u - mu)**2) / logvar.exp()
        return ll.sum(dim=-1)  # (B, T, latent_dim=96)

    def mi_upper(self, u, v):
        # positive
        log_q_pos = self.log_q(u, v).mean()
        # negative with shuffled v
        v_shuffle = v[torch.randperm(v.size(0), device=v.device)]
        log_q_neg = self.log_q(u, v_shuffle).mean()
        return (log_q_pos - log_q_neg)


class ClubMonteCarlo(nn.Module): # 지금 스피커에서 텍스트 뽑고 있음.... # v: speaker, u: text
    """
    I(U;V) ≤ E_{p(u,v)}[ log q(u|v) ] - E_{p(u)p(v)}[ log q(u|v) ]
    q(u|v) = N(u | mu(v), diag(sigma^2(v)))
    """
    def __init__(self, u_dim=96, v_dim=96, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(v_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True)
        )
        self.mu = nn.Linear(hidden, u_dim) # <- 가우시안으로 두면 이거 필요없이 mean 취하면 됨.
        self.logvar = nn.Linear(hidden, u_dim)

    def log_q(self, u, v): # 텍스트에서 스피커를 뽑아내는게 더 자연스러움
        # u,v shape: (B, latent_dim=96, T)
        v = v.transpose(1, 2)
        u = u.transpose(1, 2)
        # u: (B,Du), v:(B,Dv)  → log N(u|μ(v), Σ(v))
        h = self.net(v, u[:t]) # <- 이런식으로 과거 정보도 넣어줘야 함
        mu = self.mu(h) # <- 이거 필요 없이 mean 취한게...
        logvar = self.logvar(h).clamp(min=-8.0, max=8.0) #<- 이것도 직접 측정해야지 왜 네트워크 씀
        # 로그우도(대각가우시안)
        ll = -0.5 * (math.log(2*math.pi) + logvar) - 0.5 * ((u - mu)**2) / logvar.exp()
        return ll.sum(dim=-1)  # (B, T, latent_dim=96)

    def mi_upper(self, u, v):
        # positive
        log_q_pos = self.log_q(u, v).mean()
        # negative with shuffled v
        v_shuffle = v[torch.randperm(v.size(0), device=v.device)]
        log_q_neg = self.log_q(u, v_shuffle).mean()
        return (log_q_pos - log_q_neg)
'''    