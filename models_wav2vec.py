#!/usr/bin/env python3
"""
DAG-KD: wav2vec2.0 기반 Teacher-Student KD 모델
- Teacher : facebook/wav2vec2-large-960h  (frozen, d=1024, 24 layers)
- Student : facebook/wav2vec2-base-960h   (trainable, d=768, 12 layers)
- models.py의 DAG-KD 로직(Factorization, MI-CLUB, Generative KD, S-DisKD)을 그대로 유지.
- NeMo 의존성 제거 → pure PyTorch Lightning + HuggingFace Transformers.
"""

import os
import math
import json
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
import lightning as pl
import torchaudio
from collections import OrderedDict
from transformers import Wav2Vec2ForCTC, AutoConfig
from utils import ensure_BCT


# ============================================================
# 보조 모듈 (models.py에서 그대로 복사 — framework-agnostic)
# ============================================================

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
        self.proj = nn.Conv1d(student_dim, self.latent_dim, 1)
        self.denoiser = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.latent_dim, self.latent_dim, 3, padding=1),
        )
        self.mse = nn.MSELoss()

    def forward(self, stu_feat, tch_feat):
        stu_feat = self._to_BCT(stu_feat, self.student_dim)
        tch_feat = self._to_BCT(tch_feat, self.teacher_dim)
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
        if x.size(2) == C_expected: return x.transpose(1, 2)
        return x


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
        self.shape = nn.Linear(feat_dim_s, feat_dim_t)
        self.mse = nn.MSELoss()

    def forward(self, stu_feat, tch_feat):
        s = self._to_BTC(stu_feat, self.feat_dim_s)
        t = self._to_BTC(tch_feat, self.feat_dim_t).detach()
        x = s
        for i in range(self.steps, 0, -1):
            tt = torch.full((x.size(0), x.size(1), 1), i / self.steps, device=x.device)
            te = self.time(tt)
            h = torch.cat([x, te], dim=-1)
            v = self.net(h)
            x = x - v / self.steps
        pred = self.shape(x)
        loss = self.mse(pred, t)
        return self.loss_weight * loss

    def _to_BTC(self, x, feat_dim):
        if x.dim() != 3:
            raise ValueError("expected 3D tensor")
        if x.size(-1) == feat_dim:
            return x
        if x.size(1) == feat_dim:
            return x.transpose(1, 2)
        return x.transpose(1, 2)


class GlobalStyleTokenLayer(nn.Module):
    def __init__(self, num_tokens=10, token_dim=96, num_heads=4, ref_dim=96):
        super().__init__()
        assert token_dim % num_heads == 0
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.tokens = nn.Parameter(torch.randn(num_tokens, token_dim))
        self.query_proj = nn.Linear(ref_dim, token_dim, bias=False)
        self.key_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.v = nn.Linear(token_dim, num_heads, bias=False)
        self.last_attn = None
        self.last_attn_seq = None

    def forward(self, ref_emb):
        k = torch.tanh(self.key_proj(self.tokens))  # (N, D)
        N, D = k.size()

        if ref_emb.dim() == 2:
            q = torch.tanh(self.query_proj(ref_emb))          # (B, D)
            B = q.size(0)
            q_exp = q.unsqueeze(1).expand(B, N, D)
            k_exp = k.unsqueeze(0).expand(B, N, D)
            s = torch.tanh(q_exp + k_exp)
            logits = self.v(s)
            attn = torch.softmax(logits, dim=1)
            self.last_attn = attn.detach().cpu()
            self.last_attn_seq = None
            style = torch.einsum("bnh,nd->bdh", attn, self.tokens)
            style = style.mean(dim=-1)
            return style

        if ref_emb.dim() == 3:
            q = torch.tanh(self.query_proj(ref_emb))          # (B, T', D)
            B, Tp, _ = q.size()
            s = torch.tanh(q[:, :, None, :] + k[None, None, :, :])  # (B, T', N, D)
            logits = self.v(s)
            attn = torch.softmax(logits, dim=2)
            self.last_attn_seq = attn.detach().cpu()
            self.last_attn = attn.mean(dim=1).detach().cpu()
            style = torch.einsum("btnh,nd->btdh", attn, self.tokens)  # (B, T', D, H)
            style = style.mean(dim=-1)                                  # (B, T', D)
            return style

        raise ValueError(f"ref_emb must be 2D or 3D, got {ref_emb.dim()}D")


class GlobalProsodyReferenceEncoder(nn.Module):
    def __init__(self, n_mels=80, channels=(32, 64, 128), gru_dim=96):
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
        reduced_mels = self.n_mels // 8
        self.gru = nn.GRU(
            input_size=channels[-1] * reduced_mels,
            hidden_size=gru_dim,
            batch_first=True,
        )
        self.last_out = None
        self.last_seq = None

    def forward(self, mel, return_seq: bool = False):
        B, n_mels, T = mel.shape
        x = mel.transpose(1, 2).unsqueeze(1)       # (B,1,T,n_mels)
        z = self.conv(x)                           # (B,C,T',F')
        B, C, Tp, Fp = z.shape
        z = z.permute(0, 2, 1, 3).contiguous()     # (B,T',C,F')
        z = z.view(B, Tp, C * Fp)                  # (B,T',C*F')
        out_seq, h = self.gru(z)
        ref_global = h[-1]
        self.last_out = ref_global.detach().cpu()
        self.last_seq = out_seq.transpose(1, 2).detach().cpu()
        return out_seq if return_seq else ref_global


class ClubGaussian(nn.Module):
    """Vector CLUB q(y|x) with Gaussian assumption. (dynamic-dynamic용)"""
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
            nn.Tanh(),
        )

    def get_mu_logvar(self, x_samples):
        return self.p_mu(x_samples), self.p_logvar(x_samples)

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        positive = -(mu - y_samples) ** 2 / 2.0 / logvar.exp()
        prediction_1 = mu.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)
        negative = -((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.0 / logvar.exp()
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)

    def _seq_to_samples(self, x, y, mask=None):
        if x.dim() == 3:
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
        N = x_s.size(0)
        if self.max_samples and N > self.max_samples:
            idx = torch.randperm(N, device=x_s.device)[: self.max_samples]
            x_s = x_s[idx]
            y_s = y_s[idx]
        return x_s, y_s

    def ll_loss(self, u, v, mask=None, reduce_dim="mean"):
        x_s, y_s = self._seq_to_samples(v, u, mask=mask)
        loss = self.learning_loss(x_s, y_s)
        if reduce_dim == "mean":
            loss = loss / float(self.y_dim)
        return loss

    def mi_upper(self, u, v, K=8, mask=None, reduce_dim="mean"):
        x_s, y_s = self._seq_to_samples(v, u, mask=mask)
        N = x_s.size(0)
        if N < 2:
            return x_s.new_tensor(0.0)
        mu, logvar = self.get_mu_logvar(x_s)
        var = logvar.exp()
        pos = -(mu - y_s) ** 2 / 2.0 / var
        K = max(1, int(K))
        idx = torch.randint(0, N, (N, K), device=x_s.device)
        i_idx = torch.arange(N, device=x_s.device).unsqueeze(1)
        idx = torch.where(idx == i_idx, (idx + 1) % N, idx)
        y_neg = y_s[idx]
        mu_e = mu.unsqueeze(1)
        var_e = var.unsqueeze(1)
        neg = -(y_neg - mu_e) ** 2 / 2.0 / var_e
        neg = neg.mean(dim=1)
        if reduce_dim == "mean":
            pos_r = pos.mean(dim=-1)
            neg_r = neg.mean(dim=-1)
        else:
            pos_r = pos.sum(dim=-1)
            neg_r = neg.sum(dim=-1)
        return (pos_r - neg_r).mean()


class ARClubGaussian(nn.Module):
    """Autoregressive vCLUB for I(U_{1:T}; V_static)"""
    def __init__(self, u_dim=96, v_dim=96, hidden=128):
        super().__init__()
        self.v_to_h0 = nn.Sequential(
            nn.Linear(v_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.gru = nn.GRU(input_size=u_dim, hidden_size=hidden, batch_first=True)
        self.mu = nn.Linear(hidden, u_dim)
        self.logvar = nn.Linear(hidden, u_dim)

    def _shift_right(self, u_seq):
        z0 = torch.zeros_like(u_seq[:, :1, :])
        return torch.cat([z0, u_seq[:, :-1, :]], dim=1)

    def log_q(self, u, v, reduce_time="mean", mask=None, reduce_dim="mean"):
        u_seq = u.transpose(1, 2)
        h0 = self.v_to_h0(v).unsqueeze(0)
        causal_u = self._shift_right(u_seq)
        h, _ = self.gru(causal_u, h0)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 8.0)
        ll = -0.5 * (math.log(2 * math.pi) + logvar) - 0.5 * ((u_seq - mu) ** 2) / logvar.exp()
        if reduce_dim == "sum":
            ll = ll.sum(dim=-1)
        else:
            ll = ll.mean(dim=-1)
        if mask is not None:
            mask_f = mask.to(ll.device).float()
            ll = ll * mask_f
            denom = mask_f.sum(dim=1).clamp(min=1.0)
        else:
            denom = ll.new_full((ll.size(0),), ll.size(1))
        if reduce_time == "sum":
            out = ll.sum(dim=1)
        else:
            out = ll.sum(dim=1) / denom
        return out

    def ll_loss(self, u, v, reduce_time="mean", mask=None, reduce_dim="mean"):
        return -self.log_q(u, v, reduce_time=reduce_time, mask=mask, reduce_dim=reduce_dim).mean()

    def mi_upper(self, u, v, K=8, reduce_time="mean", mask=None, reduce_dim="mean"):
        B = u.size(0)
        if B < 2:
            return u.new_tensor(0.0)
        device = u.device
        K = max(1, int(K))
        pos = self.log_q(u, v, reduce_time=reduce_time, mask=mask, reduce_dim=reduce_dim).mean()
        idx = torch.randint(0, B, (B, K), device=device)
        i_idx = torch.arange(B, device=device).unsqueeze(1)
        idx = torch.where(idx == i_idx, (idx + 1) % B, idx)
        u_neg = u[idx.reshape(-1)]
        v_rep = v.unsqueeze(1).expand(B, K, v.size(1)).reshape(-1, v.size(1))
        if mask is not None:
            mask_neg = mask.to(device)[idx.reshape(-1)]
        else:
            mask_neg = None
        neg = self.log_q(u_neg, v_rep, reduce_time=reduce_time, mask=mask_neg, reduce_dim=reduce_dim).mean()
        return pos - neg


# ============================================================
# 메인 모델
# ============================================================

class DistilDAGKDWav2Vec2(pl.LightningModule):
    """
    wav2vec2.0 기반 DAG-KD Student 모델.
    models.py의 DistilDAGKDCTCModelBPE 로직을 최대한 그대로 유지하며,
    NeMo → HuggingFace Transformers + PyTorch Lightning 으로 재구현.
    """

    def __init__(
        self,
        teacher_name: str,
        student_name: str,
        num_spk: int,
        phys_cache_root: str,
        out_dir: str,
        train_manifest: str,
        blank_id: int = 0,
        # KD
        use_ctc: bool = True,
        use_logit_kd: bool = True,
        kd_alpha: float = 0.5,
        kd_temperature: float = 1.0,
        use_layer_kd: bool = False,
        layer_kd_alpha: float = 0.5,
        # Generative KD
        use_flow: bool = False,
        flow_steps: int = 8,
        flow_weight: float = 1.0,
        use_diffkd: bool = False,
        diffkd_steps: int = 5,
        # Disentanglement
        use_disent: bool = True,
        # Teacher 레이어 선택 (1-based, Factorization용)
        tch_spk_layers: list = None,   # teacher에서 speaker rep 뽑을 레이어 (기본: 하위 2개)
        tch_txt_layers: list = None,   # teacher에서 text rep 뽑을 레이어 (기본: 상위 2개)
        # Student 레이어 선택 (1-based, S-DisKD용)
        stu_spk_layers: list = None,   # student에서 speaker rep 뽑을 레이어 (기본: 하위 2개)
        stu_txt_layers: list = None,   # student에서 text rep 뽑을 레이어 (기본: 상위 2개)
        # Hyper-params
        latent_dim: int = 96,
        disen_mi_weight: float = 1e-3,
        disen_lll_weight: float = 1.0,
        disen_mi_pairs: str = "ts,tp,ps",
        disen_gst_tokens: int = 10,
        disen_gst_heads: int = 4,
        disen_gst_token_dim: int = 96,
        disen_spk_ce_lambda: float = 1.0,
        rec_txt_lambda: float = 0.1,
        rec_spk_lambda: float = 0.1,
        rec_pros_lambda: float = 1.0,
        neg_K: int = 8,
        mi_warmup_steps: int = 5000,
        mi_ramp_steps: int = 20000,
        mi_lambda_max: float = 0.01,
        lll_lambda_max: float = 0.01,
        mi_clamp_min0: bool = True,
        club_hidden: int = 128,
        # S-DisKD
        use_stu_txt_kd: bool = False,
        use_stu_spk_kd: bool = False,
        use_stu_club: bool = False,
        stu_txt_kd_weight: float = 1.0,
        stu_spk_kd_weight: float = 1.0,
        stu_club_weight: float = 1e-3,
        # Text probe
        use_txt_spk_probe: bool = True,
        txt_probe_lambda: float = 1.0,
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_epochs: int = 0,
        freeze_feature_extractor: bool = False,
        # Student 초기화
        random_init_student: bool = False,
        # Vis
        disen_vis_enable: bool = False,
        # Audio
        sample_rate: int = 16000,
        n_mels: int = 80,
        mel_hop_length: int = 160,   # 10ms @ 16kHz
        mel_win_length: int = 400,   # 25ms @ 16kHz
    ):
        super().__init__()
        # mutable default args
        if tch_spk_layers is None:
            tch_spk_layers = [1, 2]
        if tch_txt_layers is None:
            tch_txt_layers = [23, 24]
        if stu_spk_layers is None:
            stu_spk_layers = [1, 2]
        if stu_txt_layers is None:
            stu_txt_layers = [11, 12]

        self.save_hyperparameters()

        # teacher가 필요한지 여부 (어떤 KD/disentangle도 없으면 불필요)
        _need_teacher = (
            use_logit_kd or use_layer_kd or use_flow or use_diffkd or use_disent
        )

        # ---- Teacher (frozen) ----
        if _need_teacher:
            self.teacher = Wav2Vec2ForCTC.from_pretrained(teacher_name)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
        else:
            self.teacher = None

        # ---- Student ----
        if random_init_student:
            # config만 로드하고 가중치는 random 초기화
            _cfg = AutoConfig.from_pretrained(student_name)
            self.student = Wav2Vec2ForCTC(_cfg)
        else:
            self.student = Wav2Vec2ForCTC.from_pretrained(student_name)
        if freeze_feature_extractor:
            self.student.freeze_feature_encoder()

        # vocab 크기 통일 (teacher 기준 또는 student 자체)
        if self.teacher is not None:
            t_vocab = self.teacher.config.vocab_size
        else:
            t_vocab = self.student.config.vocab_size
        s_vocab = self.student.config.vocab_size
        if t_vocab != s_vocab:
            self.student.lm_head = nn.Linear(
                self.student.config.hidden_size, t_vocab, bias=True
            )
            self.student.config.vocab_size = t_vocab
        self.vocab_size = t_vocab
        self._blank_id = blank_id

        # ---- 차원 ----
        self.dim_s = self.student.config.hidden_size   # e.g. 768
        self.dim_t = self.teacher.config.hidden_size if self.teacher is not None else self.dim_s
        self.latent_dim = latent_dim

        # ---- Feature capture hooks ----
        # wav2vec2 encoder layer는 항상 (B, T, hidden_size) 출력 → (B, C, T)로 변환
        self.stu_feats: list = []
        self.tch_feats: list = []
        for lyr in self.student.wav2vec2.encoder.layers:
            lyr.register_forward_hook(self._cap_stu)
        if self.teacher is not None:
            for lyr in self.teacher.wav2vec2.encoder.layers:
                lyr.register_forward_hook(self._cap_tch)

        # ---- KD flags ----
        self.use_ctc = use_ctc
        self.use_logit_kd = use_logit_kd
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self.use_layer_kd = use_layer_kd
        self.layer_kd_alpha = layer_kd_alpha
        self.use_flow = use_flow
        self.use_diffkd = use_diffkd
        self.use_disent = use_disent
        self.tch_spk_layers = tch_spk_layers
        self.tch_txt_layers = tch_txt_layers
        self.stu_spk_layers = stu_spk_layers
        self.stu_txt_layers = stu_txt_layers

        # ---- Loss weights ----
        self.mi_weight = disen_mi_weight
        self.lll_weight = disen_lll_weight
        self.mi_pairs = disen_mi_pairs
        self.rec_txt_lambda = rec_txt_lambda
        self.rec_spk_lambda = rec_spk_lambda
        self.rec_pros_lambda = rec_pros_lambda
        self.disen_spk_ce_lambda = disen_spk_ce_lambda
        self.neg_K = neg_K
        self.mi_warmup_steps = mi_warmup_steps
        self.mi_ramp_steps = mi_ramp_steps
        self.mi_lambda_max = mi_lambda_max
        self.lll_lambda_max = lll_lambda_max
        self.mi_clamp_min0 = mi_clamp_min0
        self.txt_probe_lambda = txt_probe_lambda
        self.use_txt_spk_probe = use_txt_spk_probe
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.use_stu_txt_kd = use_stu_txt_kd
        self.use_stu_spk_kd = use_stu_spk_kd
        self.use_stu_club = use_stu_club
        self.stu_txt_kd_weight = stu_txt_kd_weight
        self.stu_spk_kd_weight = stu_spk_kd_weight
        self.stu_club_weight = stu_club_weight

        # ---- Mel spectrogram (prosody용) ----
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=mel_win_length,
            win_length=mel_win_length,
            hop_length=mel_hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.n_mels = n_mels

        # ---- Projection: Student → Teacher (layer KD용) ----
        self.stu_to_tea_proj = nn.Conv1d(self.dim_s, self.dim_t, kernel_size=1, bias=True)

        # ---- Text/Speaker Encoder (Conv1x1) ----
        self.txt_enc = nn.Conv1d(self.dim_t, self.latent_dim, kernel_size=1)
        self.txt_dec = nn.Conv1d(self.latent_dim, self.dim_t, kernel_size=1)
        self.spk_enc = nn.Conv1d(self.dim_t, self.latent_dim, kernel_size=1)
        self.spk_dec = nn.Conv1d(self.latent_dim, self.dim_t, kernel_size=1)

        # ---- Speaker Classifier (TDNN-style backbone + stats pooling) ----
        self.num_spk = num_spk
        if self.num_spk > 1:
            self.spk_backbone = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, self.latent_dim), nn.ReLU(inplace=True),
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, dilation=2, padding=2, bias=False),
                nn.GroupNorm(8, self.latent_dim), nn.ReLU(inplace=True),
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, dilation=3, padding=3, bias=False),
                nn.GroupNorm(8, self.latent_dim), nn.ReLU(inplace=True),
            )
            self.spk_cls = nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(self.latent_dim * 2, self.num_spk),
            )
        else:
            self.spk_backbone = None
            self.spk_cls = None
        self.spk_stat_proj = nn.Linear(self.latent_dim * 2, self.latent_dim)

        # ---- Prosody (GST) ----
        self.pros_ref = GlobalProsodyReferenceEncoder(
            n_mels=n_mels, channels=(32, 64, 128), gru_dim=96
        )
        self.pros_gst = GlobalStyleTokenLayer(
            num_tokens=disen_gst_tokens,
            token_dim=disen_gst_token_dim,
            num_heads=disen_gst_heads,
            ref_dim=96,
        )
        self.pros_proj = nn.Linear(disen_gst_token_dim, self.latent_dim)

        # Mel reconstruction decoder
        self.mel_dec = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.latent_dim, n_mels, kernel_size=1),
        )

        # Physical quantity predictor (F0, energy, voicing)
        self.prosody_predictor = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, self.latent_dim), nn.ReLU(inplace=True),
            nn.Conv1d(self.latent_dim, 3, kernel_size=1),
        )

        # ---- Phys cache ----
        self.phys_cache_root = Path(phys_cache_root)
        self.phys_cache_ext = ".npy"
        self.phys_cache_lru = 2048
        self._phys_lru: OrderedDict = OrderedDict()

        # ---- MI 추정기 (CLUB) ----
        self.club_ts = ARClubGaussian(u_dim=self.latent_dim, v_dim=self.latent_dim, hidden=club_hidden)
        self.club_ps = ARClubGaussian(u_dim=self.latent_dim, v_dim=self.latent_dim, hidden=club_hidden)
        self.club_tp = ClubGaussian(
            x_dim=self.latent_dim, y_dim=self.latent_dim,
            hidden_size=club_hidden, max_samples=2048,
        )

        # ---- S-DisKD ----
        self.stu_txt_enc = nn.Conv1d(self.dim_s, self.latent_dim, kernel_size=1)
        self.stu_spk_enc = nn.Conv1d(self.dim_s, self.latent_dim, kernel_size=1)
        if self.use_stu_club:
            self.stu_club_ts = ClubGaussian(
                x_dim=self.latent_dim, y_dim=self.latent_dim,
                hidden_size=club_hidden, max_samples=2048,
            )
        else:
            self.stu_club_ts = None

        # ---- Generative KD ----
        self.flow = FlowMatchingModule(
            self.dim_s, self.latent_dim,
            hidden=self.latent_dim, steps=flow_steps, loss_weight=flow_weight,
        ) if use_flow else None
        self.diffkd = DiffKDModule(
            teacher_dim=self.latent_dim, latent_dim=self.latent_dim,
            student_dim=self.dim_s, steps=diffkd_steps,
        ) if use_diffkd else None

        # ---- Text speaker probe ----
        if self.num_spk > 1 and use_txt_spk_probe:
            self.txt_probe_backbone = nn.Sequential(
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, self.latent_dim), nn.ReLU(inplace=True),
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, dilation=2, padding=2, bias=False),
                nn.GroupNorm(8, self.latent_dim), nn.ReLU(inplace=True),
                nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, dilation=3, padding=3, bias=False),
                nn.GroupNorm(8, self.latent_dim), nn.ReLU(inplace=True),
            )
            self.txt_spk_probe_cls = nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(self.latent_dim * 2, self.num_spk),
            )
        else:
            self.txt_probe_backbone = None
            self.txt_spk_probe_cls = None

        # ---- XAI ----
        self.vis_enable = disen_vis_enable
        self.vis_interval = 500
        self.vis_max_samples = 1
        self.vis_dir = Path(out_dir) / "xai"
        os.makedirs(self.vis_dir, exist_ok=True)

        # ---- 내부 상태 캐시 ----
        self._last_mel = None
        self._last_enc = None
        self._tch_last = None
        self._txt_emb = None
        self._tch_logp = None
        self.t_enc_len = None

        # ---- manifest speaker 로딩 ----
        self._load_manifest_speakers(train_manifest)
        L_t = self.teacher.config.num_hidden_layers if self.teacher is not None else self.student.config.num_hidden_layers
        self.layer_list_for_disent = self._prepare_layer_indices(
            [4, 8, 12, 16], L_t, default_low=True
        )

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, input_values, attention_mask=None):
        """Student forward. 호출 후 self._last_mel, self._last_enc 갱신됨."""
        self.stu_feats.clear()

        outputs = self.student(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (B, T_enc, V)

        # Encoder output length
        enc_len = self._get_enc_len(self.student, input_values, attention_mask)

        # Last encoder feature (B, dim_s, T_enc) — hook에서 채워짐
        self._last_enc = self.stu_feats[-1] if self.stu_feats else None

        # Mel (prosody용) — log-mel
        with torch.no_grad():
            mel_raw = self.mel_transform(input_values)          # (B, n_mels, T_mel)
            self._last_mel = torch.log(mel_raw.clamp(min=1e-9)) # log-mel

        return logits, enc_len

    # ============================================================
    # Training Step (models.py의 training_step과 동일 로직)
    # ============================================================

    def training_step(self, batch, batch_idx):
        input_values = batch["input_values"]       # (B, T_wav)
        attention_mask = batch["attention_mask"]   # (B, T_wav)
        labels = batch["labels"]                   # (B, T_label), -100 = padding
        speaker_ids = batch["speaker_ids"]         # (B,)
        manifest_ids = batch["manifest_ids"]       # (B,) 1-based

        # 1) Student forward
        logits, enc_len = self.forward(input_values, attention_mask)

        # 2) Teacher forward
        self._run_teacher(input_values, attention_mask)

        total = torch.tensor(0.0, device=logits.device)
        embs = None

        # 3) Factorization embeddings + MI / Rec / Speaker CE
        if self.use_disent:
            phys_targets = self._get_phys_targets(
                manifest_ids, T_mel=self._last_mel.size(-1), split_name="train"
            )
            embs = self._make_embeddings(speaker_ids, phys_targets=phys_targets)

            if embs is not None:
                mi_upper, lll, mi_terms, lll_terms = self._mi_loss(
                    txt_emb=embs["txt_emb"],
                    pros_emb=embs["pros_emb"],
                    spk_stat=embs["spk_stat"],
                    enc_len=self.t_enc_len,
                )
                mi_upper = torch.clamp(mi_upper, min=0.0)
                self.log("train/mi_upper", mi_upper, on_epoch=True)
                self.log("train/lll", lll, on_epoch=True)
                total = total + self.mi_weight * mi_upper + self.lll_weight * lll

                for k, v in mi_terms.items():
                    self.log(f"train/mi_{k}", v, on_epoch=True)
                for k, v in lll_terms.items():
                    self.log(f"train/lll_{k}", v, on_epoch=True)

                rec_txt = embs["rec_txt"]
                rec_spk = embs["rec_spk"]
                rec_pros = embs["rec_pros"]
                self.log("train/rec_txt", rec_txt, on_epoch=True)
                self.log("train/rec_spk", rec_spk, on_epoch=True)
                self.log("train/rec_pros", rec_pros, on_epoch=True)
                total = total + self.rec_txt_lambda * rec_txt + self.rec_spk_lambda * rec_spk + self.rec_pros_lambda * rec_pros

                spk_ce = embs["spk_ce"]
                spk_acc = embs.get("spk_acc", None)
                if spk_ce is not None and torch.is_tensor(spk_ce):
                    self.log("train/spk_ce", spk_ce, on_step=False, on_epoch=True)
                    total = total + self.disen_spk_ce_lambda * spk_ce
                if spk_acc is not None and torch.is_tensor(spk_acc):
                    self.log("train/spk_acc", spk_acc, on_epoch=True)

                phys_loss = embs.get("phys_loss", None)
                if phys_loss is not None and torch.is_tensor(phys_loss):
                    self.log("train/phys_loss", phys_loss, on_step=True, on_epoch=True)
                    total = total + self.rec_pros_lambda * phys_loss
        else:
            self._txt_emb = None

        # 4) Generative KD (FM / DiffKD)
        flow_loss = torch.tensor(0.0, device=self.device)
        diff_loss = torch.tensor(0.0, device=self.device)
        if self._txt_emb is not None and self._last_enc is not None:
            stu_feat = ensure_BCT(self._last_enc, C_expected=self.dim_s)
            tch_feat = ensure_BCT(self._txt_emb.detach(), C_expected=self.latent_dim)
            # T축 정렬 (teacher/student T_enc는 동일하지만 혹시 모르니 맞춰줌)
            if stu_feat.size(-1) != tch_feat.size(-1):
                tch_feat = F.interpolate(tch_feat, size=stu_feat.size(-1), mode="linear", align_corners=False)
            if self.use_flow and self.flow is not None:
                flow_loss = self.flow(stu_feat, tch_feat)
            if self.use_diffkd and self.diffkd is not None:
                diff_loss = self.diffkd(stu_feat, tch_feat)

        self.log("train/flow_loss", flow_loss, on_step=False, on_epoch=True)
        self.log("train/diff_loss", diff_loss, on_step=False, on_epoch=True)
        total = total + flow_loss + diff_loss

        # 5) CTC
        if self.use_ctc:
            ctc = self._ctc_loss(logits, enc_len, labels)
            self.log("train/ctc", ctc, on_step=False, on_epoch=True)
            total = total + ctc

        # 6) Logit KD
        if self.use_logit_kd:
            kd_logit = self._logit_kd(logits)
            self.log("train/logit_kd", kd_logit, on_step=False, on_epoch=True)
            total = total + self.kd_alpha * kd_logit

        # 7) Layer-wise metric KD
        if self.use_layer_kd:
            kd_layer = self._layer_metric_kd()
            self.log("train/layer_kd", kd_layer, on_step=False, on_epoch=True)
            total = total + self.layer_kd_alpha * kd_layer

        # 8) S-DisKD
        if (self.use_stu_txt_kd or self.use_stu_spk_kd) and self.stu_feats and embs is not None:
            L_stu = len(self.stu_feats)
            spk_idxs = self._prepare_layer_indices(self.stu_spk_layers, L_stu, default_low=True)
            txt_idxs = self._prepare_layer_indices(self.stu_txt_layers, L_stu, default_low=False)

            stu_txt_factor = None
            stu_spk_factor = None
            stu_txt_kd_loss = torch.tensor(0.0, device=self.device)
            stu_spk_kd_loss = torch.tensor(0.0, device=self.device)

            if self.use_stu_txt_kd:
                stu_txt_rep = torch.stack([self.stu_feats[i] for i in txt_idxs], dim=0).mean(0)
                stu_txt_factor = self.stu_txt_enc(stu_txt_rep)
                tch_txt_target = embs["txt_emb"].detach()
                if stu_txt_factor.size(-1) != tch_txt_target.size(-1):
                    stu_txt_factor = F.interpolate(stu_txt_factor, size=tch_txt_target.size(-1), mode="linear", align_corners=False)
                stu_txt_kd_loss = F.mse_loss(stu_txt_factor, tch_txt_target)

            if self.use_stu_spk_kd:
                stu_spk_rep = torch.stack([self.stu_feats[i] for i in spk_idxs], dim=0).mean(0)
                stu_spk_factor = self.stu_spk_enc(stu_spk_rep)
                tch_spk_target = embs["spk_emb"].detach()
                if stu_spk_factor.size(-1) != tch_spk_target.size(-1):
                    stu_spk_factor = F.interpolate(stu_spk_factor, size=tch_spk_target.size(-1), mode="linear", align_corners=False)
                stu_spk_kd_loss = F.mse_loss(stu_spk_factor, tch_spk_target)

            self.log("train/stu_txt_kd", stu_txt_kd_loss, on_step=False, on_epoch=True)
            self.log("train/stu_spk_kd", stu_spk_kd_loss, on_step=False, on_epoch=True)
            total = total + self.stu_txt_kd_weight * stu_txt_kd_loss + self.stu_spk_kd_weight * stu_spk_kd_loss

            if self.use_stu_club and self.stu_club_ts is not None and stu_txt_factor is not None and stu_spk_factor is not None:
                stu_lll = self.stu_club_ts.ll_loss(stu_txt_factor.detach(), stu_spk_factor.detach())
                self._freeze_params(self.stu_club_ts, True)
                stu_mi = self.stu_club_ts.mi_upper(stu_txt_factor, stu_spk_factor, K=self.neg_K)
                self._freeze_params(self.stu_club_ts, False)
                self.log("train/stu_mi_ts", stu_mi, on_step=False, on_epoch=True)
                self.log("train/stu_lll_ts", stu_lll, on_step=False, on_epoch=True)
                total = total + self.stu_club_weight * stu_mi + self.lll_weight * stu_lll

        # Text speaker probe
        if self.use_disent and self.use_txt_spk_probe and embs is not None:
            probe_ce = embs.get("txt_probe_ce", None)
            probe_acc = embs.get("txt_probe_acc", None)
            if probe_ce is not None:
                self.log("probe/txt_spk_ce", probe_ce, on_step=False, on_epoch=True)
                total = total + self.txt_probe_lambda * probe_ce
            if probe_acc is not None:
                self.log("probe/txt_spk_acc", probe_acc, on_step=False, on_epoch=True)

        self.log("train/total", total, on_step=False, on_epoch=True, prog_bar=True)
        return total

    # ============================================================
    # Validation Step
    # ============================================================

    def validation_step(self, batch, batch_idx):
        input_values = batch["input_values"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        self.stu_feats.clear()
        outputs = self.student(input_values=input_values, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)
        enc_len = self._get_enc_len(self.student, input_values, attention_mask)

        # CTC val loss
        if self.use_ctc:
            val_ctc = self._ctc_loss(logits, enc_len, labels)
            self.log("val/ctc", val_ctc, on_epoch=True, prog_bar=False, sync_dist=True)

        # Greedy decode → WER
        pred_ids = logits.argmax(dim=-1)  # (B, T)
        processor = getattr(self, "processor", None)
        if processor is not None:
            # CTC collapse 후 processor.decode로 문자열 변환
            pred_id_lists = self._ctc_decode_batch(pred_ids)
            pred_strs = [processor.decode(ids) for ids in pred_id_lists]
            ref_strs = [
                processor.decode([t for t in row.tolist() if t != -100 and t != self._blank_id])
                for row in labels
            ]
            from utils import compute_sample_wers
            wers = compute_sample_wers(ref_strs, pred_strs)
        else:
            wers = []
        wer_mean = float(sum(wers) / len(wers)) if wers else 0.0
        self.log("val/wer", wer_mean, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"val/wer": wer_mean}

    # ============================================================
    # Optimizer
    # ============================================================

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=1e-2)
        T_max = self.trainer.max_epochs
        warmup = self.warmup_epochs

        if warmup > 0:
            # Linear warmup → Cosine annealing
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup
            )
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(T_max - warmup, 1), eta_min=1e-6
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_warmup, scheduler_cosine],
                milestones=[warmup],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=1e-6
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    # ============================================================
    # Teacher forward
    # ============================================================

    def _run_teacher(self, input_values, attention_mask):
        """Teacher encoder + decoder 한 번 실행. tch_feats / _tch_last / _tch_logp 갱신."""
        if self.teacher is None:
            self._tch_logp = None
            self._tch_last = None
            self.t_enc_len = None
            return
        self.tch_feats.clear()
        with torch.no_grad():
            outputs = self.teacher(
                input_values=input_values,
                attention_mask=attention_mask,
            )
        self._tch_logp = outputs.logits                                 # (B, T, V)
        self._tch_last = self.tch_feats[-1] if self.tch_feats else None  # (B, dim_t, T)
        self.t_enc_len = self._get_enc_len(self.teacher, input_values, attention_mask)

    # ============================================================
    # Embeddings & MI (models.py의 _make_embeddings / _mi_loss 그대로)
    # ============================================================

    def _make_embeddings(self, speaker_ids, phys_targets=None):
        if self._tch_last is None:
            return None

        B, C_t, T_t = self._tch_last.shape
        last = self._tch_last

        spk_rep, txt_rep = self._get_spk_txt_reps_from_layers()
        if spk_rep is None or txt_rep is None:
            spk_rep = last
            txt_rep = last

        # Text Encoder
        txt_emb = self.txt_enc(txt_rep)
        txt_rec = self.txt_dec(txt_emb)
        rec_txt = F.mse_loss(txt_rec, txt_rep)
        txt_probe_ce, txt_probe_acc = self._text_spk_probe(txt_emb, speaker_ids)
        self._txt_emb = txt_emb

        # Speaker Encoder
        spk_emb = self.spk_enc(spk_rep)
        spk_rec = self.spk_dec(spk_emb)
        rec_spk = F.mse_loss(spk_rec, spk_rep)

        spk_feat = self.spk_backbone(spk_emb) if self.spk_backbone is not None else spk_emb
        spk_mean = spk_feat.mean(dim=-1)
        spk_std = self.safe_std(spk_feat, dim=-1)
        spk_stat = torch.cat([spk_mean, spk_std], dim=-1)
        spk_stat = self.spk_stat_proj(spk_stat)

        # Prosody (frame-level GST)
        T = txt_emb.size(-1)
        mel = self._last_mel.to(self.device)
        ref_seq = self.pros_ref(mel, return_seq=True)
        style_seq = self.pros_gst(ref_seq)
        pros_emb = self.pros_proj(style_seq).transpose(1, 2)
        pros_emb = F.interpolate(pros_emb, size=T, mode="linear", align_corners=False)
        pros_stat = pros_emb.mean(dim=-1)

        # Mel Reconstruction
        mel_target = self._last_mel
        pros_emb_for_mel = F.interpolate(pros_emb, size=mel_target.size(-1), mode="linear", align_corners=False)
        mel_pred = self.mel_dec(pros_emb_for_mel)
        rec_pros = F.mse_loss(mel_pred, mel_target)

        # Physical Quantity Supervision
        phys_loss = torch.tensor(0.0, device=pros_emb.device)
        if phys_targets is not None:
            phys_pred = self.prosody_predictor(pros_emb)
            if phys_pred.size(-1) != phys_targets.size(-1):
                phys_pred = F.interpolate(phys_pred, size=phys_targets.size(-1), mode="linear", align_corners=False)
            phys_loss = F.mse_loss(phys_pred, phys_targets)

        # Speaker CE & ACC
        spk_ce = torch.tensor(0.0, device=txt_emb.device)
        spk_acc = None
        if self.spk_cls is not None and speaker_ids is not None:
            valid_mask = (speaker_ids >= 0) & (speaker_ids < self.num_spk)
            if valid_mask.any():
                if self.spk_backbone is not None:
                    spk_feat = self.spk_backbone(spk_emb)
                else:
                    spk_feat = spk_emb
                spk_mean = spk_feat.mean(dim=-1)
                spk_std = self.safe_std(spk_feat, dim=-1)
                spk_utt = torch.cat([spk_mean, spk_std], dim=-1)
                spk_utt_valid = spk_utt[valid_mask]
                target_valid = speaker_ids[valid_mask].clamp(min=0).long()
                logits_utt = self.spk_cls(spk_utt_valid)
                spk_ce = F.cross_entropy(logits_utt, target_valid)
                all_logits = self.spk_cls(spk_utt)
                preds = all_logits.argmax(dim=-1)
                spk_acc = (preds[valid_mask] == target_valid).float().mean()

        # XAI
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
                enc_len=self.t_enc_len,
            )

        return {
            "txt_emb": txt_emb,
            "spk_emb": spk_emb,
            "pros_emb": pros_emb,
            "spk_stat": spk_stat,
            "pros_stat": pros_stat,
            "spk_ce": spk_ce,
            "spk_acc": spk_acc,
            "rec_txt": rec_txt,
            "rec_spk": rec_spk,
            "rec_pros": rec_pros,
            "txt_probe_ce": txt_probe_ce,
            "txt_probe_acc": txt_probe_acc,
            "phys_loss": phys_loss,
        }

    def _mi_loss(self, txt_emb, pros_emb, spk_stat, enc_len=None):
        txt = txt_emb
        pros = pros_emb
        spk = spk_stat
        pairs = set(t.strip() for t in self.mi_pairs.split(",") if t.strip())

        lll_terms = {}
        mi_terms = {}
        lll_sum = torch.tensor(0.0, device=self.device)
        mi_sum = torch.tensor(0.0, device=self.device)

        mask = None
        if enc_len is not None:
            T = txt.size(-1)
            mask = self._make_len_mask(enc_len, T, device=txt.device, dtype=torch.float32)

        reduce_dim = "mean"

        if "tp" in pairs:
            lll_terms["tp"] = self.club_tp.ll_loss(txt.detach(), pros.detach(), mask=mask, reduce_dim=reduce_dim)
            lll_sum = lll_sum + lll_terms["tp"]
            self._freeze_params(self.club_tp, True)
            mi_terms["tp"] = self.club_tp.mi_upper(txt, pros, K=self.neg_K, mask=mask, reduce_dim=reduce_dim)
            mi_sum = mi_sum + mi_terms["tp"]
            self._freeze_params(self.club_tp, False)

        if "ts" in pairs:
            lll_terms["ts"] = self.club_ts.ll_loss(txt.detach(), spk.detach(), mask=mask, reduce_dim=reduce_dim)
            lll_sum = lll_sum + lll_terms["ts"]
            self._freeze_params(self.club_ts, True)
            mi_terms["ts"] = self.club_ts.mi_upper(txt, spk, K=self.neg_K, mask=mask, reduce_dim=reduce_dim)
            mi_sum = mi_sum + mi_terms["ts"]
            self._freeze_params(self.club_ts, False)

        if "ps" in pairs:
            lll_terms["ps"] = self.club_ps.ll_loss(pros.detach(), spk.detach(), mask=mask, reduce_dim=reduce_dim)
            lll_sum = lll_sum + lll_terms["ps"]
            self._freeze_params(self.club_ps, True)
            mi_terms["ps"] = self.club_ps.mi_upper(pros, spk, K=self.neg_K, mask=mask, reduce_dim=reduce_dim)
            mi_sum = mi_sum + mi_terms["ps"]
            self._freeze_params(self.club_ps, False)

        return mi_sum, lll_sum, mi_terms, lll_terms

    # ============================================================
    # Loss functions
    # ============================================================

    def _ctc_loss(self, logits, enc_len, labels):
        """
        logits: (B, T, V)
        enc_len: (B,)
        labels: (B, T_label), -100 = padding
        """
        label_lengths = (labels != -100).sum(-1).long()      # (B,)
        labels_clean = labels.clone()
        labels_clean[labels_clean < 0] = 0                   # -100 → 0 (ignored by label_lengths)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
        return F.ctc_loss(
            log_probs,
            labels_clean,
            enc_len.long(),
            label_lengths,
            blank=self._blank_id,
            reduction="mean",
            zero_infinity=True,
        )

    def _logit_kd(self, logits_s):
        if self._tch_logp is None:
            return torch.tensor(0.0, device=logits_s.device)
        T_s = logits_s.size(1)
        T_t = self._tch_logp.size(1)
        tch_logp = self._tch_logp
        if T_s != T_t:
            tch_logp = tch_logp.transpose(1, 2)
            tch_logp = F.interpolate(tch_logp, size=T_s, mode="linear", align_corners=False)
            tch_logp = tch_logp.transpose(1, 2)
        T = self.kd_temperature
        with torch.no_grad():
            p_t = F.softmax(tch_logp / T, dim=-1)
        logp_s_T = F.log_softmax(logits_s / T, dim=-1)
        # batchmean은 B로만 나누므로 T(시퀀스 길이)로 추가 정규화 → per-token KL
        return F.kl_div(logp_s_T, p_t, reduction="batchmean") * (T * T) / T_s

    def _layer_metric_kd(self):
        if not self.stu_feats or not self.tch_feats:
            return torch.tensor(0.0, device=self.device)
        losses = []
        L = min(len(self.stu_feats), len(self.tch_feats))
        for i in range(L):
            s = self.stu_feats[i]           # (B, dim_s, T)
            t = self.tch_feats[i].detach()  # (B, dim_t, T)
            s_proj = self.stu_to_tea_proj(s)
            if s_proj.size(-1) != t.size(-1):
                s_proj = F.interpolate(s_proj, size=t.size(-1), mode="linear", align_corners=False)
            losses.append(F.mse_loss(s_proj, t))
        return sum(losses) / L

    # ============================================================
    # Phys cache helpers
    # ============================================================

    def _get_phys_targets(self, manifest_ids: torch.Tensor, T_mel: int, split_name: str = "train"):
        if manifest_ids is None:
            return None
        if not self.phys_cache_root.exists():
            return None

        B = int(manifest_ids.numel())
        out_cpu = torch.zeros((B, 3, T_mel), dtype=torch.float32, device="cpu")

        for b in range(B):
            mid = int(manifest_ids[b].item())
            key = (split_name, mid)
            arr = self._phys_lru_get(key)
            if arr is None:
                p = self.phys_cache_root / split_name / f"{mid}{self.phys_cache_ext}"
                if not p.exists():
                    continue
                arr = np.load(p, mmap_mode="r")
                self._phys_lru_put(key, arr)

            phys = torch.from_numpy(np.array(arr, copy=False))
            if phys.dtype != torch.float32:
                phys = phys.float()
            T_src = phys.size(-1)
            if T_src != T_mel:
                phys = F.interpolate(
                    phys.unsqueeze(0), size=T_mel, mode="linear", align_corners=False
                ).squeeze(0)
            out_cpu[b] = phys

        return out_cpu.to(self.device, non_blocking=True)

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

    # ============================================================
    # Hook callbacks
    # ============================================================

    def _cap_stu(self, module, inp, out):
        """wav2vec2 encoder layer는 tuple(hidden_states, ...) 반환.
        hidden_states: (B, T, C) → (B, C, T)로 변환 후 저장."""
        if isinstance(out, (tuple, list)):
            out = out[0]
        self.stu_feats.append(out.transpose(1, 2).contiguous())

    def _cap_tch(self, module, inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        self.tch_feats.append(out.transpose(1, 2).contiguous())

    # ============================================================
    # Layer helpers
    # ============================================================

    def _get_spk_txt_reps_from_layers(self):
        if not self.tch_feats:
            return None, None
        L = len(self.tch_feats)
        spk_idxs = self._prepare_layer_indices(self.tch_spk_layers, L, default_low=True)
        txt_idxs = self._prepare_layer_indices(self.tch_txt_layers, L, default_low=False)
        spk_rep = torch.stack([self.tch_feats[i] for i in spk_idxs], dim=0).mean(0)
        txt_rep = torch.stack([self.tch_feats[i] for i in txt_idxs], dim=0).mean(0)
        return spk_rep, txt_rep

    def _prepare_layer_indices(self, layer_list, L, default_low: bool):
        idxs = []
        if isinstance(layer_list, (list, tuple)) and len(layer_list) > 0:
            for x in layer_list:
                try:
                    i0 = int(x) - 1  # 1-based → 0-based
                except Exception:
                    continue
                if 0 <= i0 < L:
                    idxs.append(i0)
        if idxs:
            return sorted(set(idxs))
        k = max(1, L // 3)
        return list(range(0, k)) if default_low else list(range(L - k, L))


    # ============================================================
    # Misc helpers (models.py에서 그대로)
    # ============================================================

    def _text_spk_probe(self, txt_emb, speaker_ids):
        if self.txt_spk_probe_cls is None or speaker_ids is None:
            return None, None
        valid_mask = (speaker_ids >= 0) & (speaker_ids < self.num_spk)
        if not valid_mask.any():
            return None, None
        x = txt_emb.detach()
        if self.txt_probe_backbone is not None:
            x = self.txt_probe_backbone(x)
        mean = x.mean(dim=-1)
        std = self.safe_std(x, dim=-1)
        utt = torch.cat([mean, std], dim=-1)
        logits = self.txt_spk_probe_cls(utt)
        target = speaker_ids.clamp(min=0).long()
        probe_ce = F.cross_entropy(logits[valid_mask], target[valid_mask])
        preds = logits.argmax(dim=-1)
        probe_acc = (preds[valid_mask] == target[valid_mask]).float().mean()
        return probe_ce, probe_acc

    def _freeze_params(self, module: nn.Module, freeze: bool):
        for p in module.parameters():
            p.requires_grad = not freeze

    def _load_manifest_speakers(self, manifest_path: str):
        spk = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                spk.append(int(obj.get("spk_idx", -1)))
        self.manifest_speakers = torch.tensor(spk, dtype=torch.long)

    @staticmethod
    def safe_std(x, dim=-1, eps=1e-5):
        var = torch.var(x, dim=dim, unbiased=False)
        return torch.sqrt(var + eps)

    @staticmethod
    def _make_len_mask(lengths: torch.Tensor, T: int, device=None, dtype=torch.float32):
        if lengths is None:
            return None
        if device is None:
            device = lengths.device
        lengths = lengths.to(device)
        t = torch.arange(T, device=device).unsqueeze(0)
        return (t < lengths.unsqueeze(1)).to(dtype=dtype)

    def _get_enc_len(self, model, input_values, attention_mask):
        """wav2vec2 feature extractor 이후 encoder 시퀀스 길이 계산."""
        if attention_mask is not None:
            wav_len = attention_mask.sum(-1).long()
        else:
            B, T_wav = input_values.shape
            wav_len = torch.full((B,), T_wav, dtype=torch.long, device=input_values.device)
        return model.wav2vec2._get_feat_extract_output_lengths(wav_len).long()

    def _ctc_decode_batch(self, ids: torch.Tensor, collapse: bool = True) -> list:
        """
        ids: (B, T) int64 — token IDs (greedy)
        CTC blank/repeat collapse 후 정수 ID 리스트 → 문자열로 디코딩.
        vocab 정보 없이 ID를 직접 문자열로 변환 (decode는 train_wav2vec.py에서 processor 사용).
        여기서는 단순 int→chr 매핑 대신 ID 리스트만 반환하고 문자열은 나중에 처리.
        → 실제로는 model.vocab을 써야 하므로 processor를 외부에서 주입받아 사용.
        """
        # blank_id와 repeated token을 제거한 후 빈 문자열 리스트 반환
        # (processor 없이 동작: id → " " 치환 방식)
        results = []
        blank = self._blank_id
        for row in ids:
            tokens = row.tolist()
            if collapse:
                # CTC collapse: blank 및 반복 제거
                decoded = []
                prev = None
                for t in tokens:
                    if t == blank:
                        prev = None
                        continue
                    if t != prev:
                        decoded.append(t)
                    prev = t
            else:
                decoded = [t for t in tokens if t != blank]
            results.append(decoded)
        return results

    # ============================================================
    # XAI / Visualization (models.py에서 그대로)
    # ============================================================

    def _xai_visualize(self, txt_emb, spk_emb, pros_emb, txt_rep, spk_rep,
                       rec_txt, rec_spk, speaker_ids=None, enc_len=None):
        step = int(getattr(self, "global_step", 0))
        if (step % self.vis_interval) != 0:
            return
        B = txt_emb.size(0)
        num = min(self.vis_max_samples, B)

        if self._last_mel is not None:
            mel = self._last_mel[:num].detach().cpu()
            for b in range(num):
                self._plot_heatmap(mel[b], f"Mel (sample {b}, step {step})",
                                   f"step{step:06d}_s{b}_mel.png", "time", "mel-bin")

        T = txt_emb.size(-1)
        mask = None
        if enc_len is not None:
            mask = self._make_len_mask(enc_len, T, device=txt_emb.device, dtype=torch.float32)

        def masked_mean_BCT(x):
            if mask is None:
                return x.mean(dim=-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (x * mask[:, None, :]).sum(dim=-1) / denom[:, None]

        txt_stat = masked_mean_BCT(txt_emb).detach()
        spk_stat_vis = masked_mean_BCT(spk_emb).detach()
        pros_stat_vis = masked_mean_BCT(pros_emb).detach()

        max_pts = 256
        if B > max_pts:
            idx = torch.randperm(B, device=txt_stat.device)[:max_pts]
            txt_stat = txt_stat[idx]
            spk_stat_vis = spk_stat_vis[idx]
            pros_stat_vis = pros_stat_vis[idx]
            if speaker_ids is not None:
                speaker_ids = speaker_ids[idx]
            B = max_pts

        def proj2d(X):
            X = X.float() - X.float().mean(dim=0, keepdim=True)
            if X.size(0) < 2:
                return X.new_zeros((X.size(0), 2)).cpu().numpy()
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            W = Vh[:2].T
            return (X @ W).detach().cpu().numpy()

        Z_txt = proj2d(txt_stat)
        Z_spk = proj2d(spk_stat_vis)
        Z_pros = proj2d(pros_stat_vis)
        c = speaker_ids.detach().cpu().numpy() if speaker_ids is not None else np.arange(B)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        scat0 = axes[0].scatter(Z_txt[:, 0], Z_txt[:, 1], c=c, s=14)
        axes[0].set_title("txt_stat 2D (speaker)")
        axes[1].scatter(Z_spk[:, 0], Z_spk[:, 1], c=c, s=14)
        axes[1].set_title("spk_stat 2D (speaker)")
        axes[2].scatter(Z_pros[:, 0], Z_pros[:, 1], c=c, s=14)
        axes[2].set_title("pros_stat 2D (speaker)")
        fig.colorbar(scat0, ax=axes, fraction=0.02, pad=0.02)
        fig.suptitle(f"2D projection @ step {step}")
        self._save_fig(fig, f"step{step:06d}_proj_pca_speaker.png")

    def _save_fig(self, fig, name: str):
        path = self.vis_dir / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _plot_heatmap(self, mat, title, fname, xlabel=None, ylabel=None):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(mat, aspect="auto", origin="lower")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        self._save_fig(fig, fname)
