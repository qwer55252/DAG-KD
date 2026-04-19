#!/usr/bin/env python3
"""
GRP-KD baseline training script (based on asr_train_diffm.py)

asr_train_diffm.py의 DistilFlowMatchingCTCModelBPE를 그대로 재현.
데이터 인프라는 DAG-KD의 기존 manifest 방식 재사용.

version 1~8:
  ver1: AE + KD
  ver2: AE + FM
  ver3: AE + NoiseAdapter + Diffusion + KD
  ver4: AE + FM(pre) + NoiseAdapter + Diffusion + KD(post)
  ver5: AE + NoiseAdapter + Diffusion + FM(post)
  ver6: AE + FM(pre) + NoiseAdapter + Diffusion + FM(post)
  ver7: AE + FM(pre) + NoiseAdapter + Diffusion + FM(post) [order 다름]
  ver8: AE + FM(pre) + NoiseAdapter + Diffusion + KD(post)

공통:
  - Teacher AE recon loss 항상 계산
  - Student 1x1 proj: latent 매핑
  - 모든 레이어 피처 합산으로 손실 집계
"""

import os
import gc
import json
import math
import torch
import aiohttp
import argparse
import statistics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from omegaconf import OmegaConf
from nemo.collections import asr as nemo_asr
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from datasets import load_dataset, DownloadConfig, config as hf_config

from models import ClubGaussian, GradientReversalLayer
from utils import (
    scan_speakers,
    build_manifest_from_hf_with_meta,
    str2bool,
    release_nemoAPI,
    compute_speaker_durations,
    compute_sample_wers,
    snapshot_sources,
    load_speaker_table_from_manifest,
    extract_speaker_ids_from_batch,
)


# ============================================================
# Noise schedule helpers (asr_train_diffm.py 원본 그대로)
# ============================================================
def rectified_flow_schedule(t):
    return t, 1 - t

def rectified_flow_schedule_deriv(t):
    return torch.ones_like(t), -torch.ones_like(t)

def vp_ode_schedule(t, a=19.9, b=0.1):
    alpha_t = torch.exp(-0.25 * a * (1 - t) ** 2 - 0.5 * b * (1 - t))
    sigma_t = torch.sqrt(1 - alpha_t ** 2)
    return alpha_t, sigma_t

def vp_ode_schedule_deriv(t, a=19.9, b=0.1):
    alpha_t = torch.exp(-0.25 * a * (1 - t) ** 2 - 0.5 * b * (1 - t))
    dalpha_dt = alpha_t * (0.5 * a * (1 - t) + 0.5 * b)
    sigma_t = torch.sqrt(1 - alpha_t ** 2)
    dsigma_dt = -alpha_t * dalpha_dt / sigma_t
    return dalpha_dt, dsigma_dt


# ============================================================
# FlowMatchingModule (asr_train_diffm.py 원본 그대로)
# ============================================================
class FlowMatchingModule(nn.Module):
    def __init__(self, flow_cfg, router=None, router_weight=0.1):
        super().__init__()
        self.meta_encoder_type  = flow_cfg.get("meta_encoder_type", "mlp")
        time_embed_dim          = flow_cfg.get("time_embed_dim", 32)
        self.hidden_dim         = flow_cfg.get("hidden_dim", 96)
        self.training_sampling  = flow_cfg.get("training_sampling", 8)
        self.weight             = flow_cfg.get("weight", 1.0)
        self.feature_dim        = flow_cfg.get("hidden_dim", 96)   # latent_dim
        self.teacher_dim        = flow_cfg.get("teacher_dim", 176)

        self.time_embed = nn.Linear(1, time_embed_dim)

        assert self.meta_encoder_type == "mlp", \
            f"train_grp_kd.py는 meta_encoder_type='mlp'만 지원합니다. 받은 값: {self.meta_encoder_type}"
        self.meta_encoder = nn.Sequential(
            nn.Linear(self.feature_dim + time_embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )

        shape_transform = flow_cfg.get("shape_transform", "linear")
        self.shape_transform_type = shape_transform
        if shape_transform == "identity":
            self.shape_transformation_function = nn.Identity()
        elif shape_transform == "linear":
            self.shape_transformation_function = nn.Linear(self.feature_dim, self.hidden_dim)
        else:
            raise ValueError(f"Unknown shape_transform: {shape_transform}")

        loss_type = flow_cfg.get("loss", "mse")
        self.metric_based_loss_function = nn.MSELoss() if loss_type == "mse" else nn.L1Loss()

        noise_schedule = flow_cfg.get("noise_schedule", "rectified")
        if noise_schedule == "rectified":
            self.noise_schedule_deriv = rectified_flow_schedule_deriv
        elif noise_schedule == "vp_ode":
            self.noise_schedule_deriv = vp_ode_schedule_deriv
        else:
            raise NotImplementedError(f"noise_schedule={noise_schedule}")

    def forward(self, s_f, t_f=None, target=None, layer_sampling_step=None, layer_id=None):
        x = s_f
        for i in range(layer_sampling_step, 0, -1):
            t = torch.full((s_f.size(0), s_f.size(1), 1), i / layer_sampling_step, device=s_f.device)
            embed_t = self.time_embed(t).permute(0, 2, 1)       # (B, time_embed_dim, T)
            x_perm  = x.permute(0, 2, 1)                        # (B, feature_dim, T)
            embed_x = torch.cat([x_perm, embed_t], dim=1)       # (B, feature_dim+time_embed_dim, T)
            embed_x = embed_x.permute(0, 2, 1)                  # (B, T, feature_dim+time_embed_dim)
            velocity = self.meta_encoder(embed_x)                # (B, T, feature_dim)
            x = x - velocity / layer_sampling_step

        loss = torch.tensor(0.0, device=s_f.device)
        if self.training and t_f is not None:
            dalpha_dt, dsigma_dt = self.noise_schedule_deriv(t)
            noise_scheduled_x = (dalpha_dt * s_f - velocity) / (-dsigma_dt)
            transformed_s_f = self.shape_transformation_function(noise_scheduled_x)
            loss = self.metric_based_loss_function(transformed_s_f, t_f)

        return loss, x


# ============================================================
# 공통 블록 (asr_train_diffm.py 원본 그대로)
# ============================================================
class TeacherAutoEncoder(nn.Module):
    """Teacher feature (B, C_t, T) → latent (B, L, T) → recon (B, C_t, T)"""
    def __init__(self, teacher_dim: int, latent_dim: int):
        super().__init__()
        self.enc = nn.Conv1d(teacher_dim, latent_dim, kernel_size=1)
        self.dec = nn.Conv1d(latent_dim, teacher_dim, kernel_size=1)

    @torch.no_grad()
    def encode_nograd(self, x_ct):
        return self.enc(x_ct)

    def forward(self, x_ct):
        z_t = self.enc(x_ct)
        rec = self.dec(z_t)
        return z_t, rec


class StudentProjector(nn.Module):
    """Student feature (B, C_s, T) → latent (B, L, T)"""
    def __init__(self, student_dim: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(student_dim, latent_dim, kernel_size=1)

    def forward(self, x_cs):
        return self.proj(x_cs)


class NoiseAdapter(nn.Module):
    """γ(x)∈[0,1] 예측: Z_noisy = γ·Z + (1-γ)·ε"""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.gamma_head = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z_latent):
        gamma   = self.gamma_head(z_latent)           # (B, 1, T)
        eps     = torch.randn_like(z_latent)
        z_noisy = gamma * z_latent + (1.0 - gamma) * eps
        return z_noisy, gamma


class SimpleDenoiser(nn.Module):
    """Diffusion용 간단 1D-CNN 디노이저"""
    def __init__(self, latent_dim: int, steps: int = 5):
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
        )

    def forward(self, z_in):
        x = z_in
        for _ in range(self.steps):
            pred_noise = self.net(x)
            x = x - pred_noise / self.steps
        return x


class SpeakerClassifier(nn.Module):
    """z_spk (B, latent_dim, T) → mean pool → fc → logits (B, num_spk)"""
    def __init__(self, latent_dim: int, num_spk: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_spk)

    def forward(self, z_spk_bct):
        pooled = z_spk_bct.mean(dim=2)   # (B, latent_dim)
        return self.fc(pooled)            # (B, num_spk)


class FMLatent(nn.Module):
    """잠재공간 전용 FlowMatching 래퍼 (shape_transform=identity, student_dim=teacher_dim=latent_dim)"""
    def __init__(self, latent_dim: int, flow_cfg: dict):
        super().__init__()
        flow_cfg = dict(flow_cfg or {})
        flow_cfg.setdefault("student_dim", latent_dim)
        flow_cfg.setdefault("teacher_dim", latent_dim)
        flow_cfg.setdefault("hidden_dim",  latent_dim)
        flow_cfg.setdefault("shape_transform", "identity")
        flow_cfg.setdefault("meta_encoder_type", "mlp")
        flow_cfg.setdefault("training_sampling", 8)
        self.fm = FlowMatchingModule(flow_cfg)
        self.default_steps = int(flow_cfg.get("training_sampling", 8))

    def forward(self, s_latent_bct, t_latent_bct, steps=None):
        """
        s_latent_bct / t_latent_bct: (B, L, T)
        returns: fm_loss (scalar), s_out_bct: (B, L, T)
        """
        s_btC = s_latent_bct.transpose(1, 2)   # (B, T, L)
        t_btC = t_latent_bct.transpose(1, 2)
        layer_steps = int(steps or self.default_steps)
        fm_loss, s_out_btC = self.fm(s_btC, t_f=t_btC, layer_sampling_step=layer_steps)
        s_out_bct = s_out_btC.transpose(1, 2)  # (B, L, T)
        if not isinstance(fm_loss, torch.Tensor):
            fm_loss = torch.as_tensor(fm_loss, device=s_latent_bct.device, dtype=s_latent_bct.dtype)
        return fm_loss, s_out_bct


# ============================================================
# DistilFlowMatchingCTCModelBPE (asr_train_diffm.py 원본 그대로)
# ============================================================
class DistilFlowMatchingCTCModelBPE(nemo_asr.models.EncDecCTCModelBPE):
    """
    version 1~8: TeacherAE + StudentProjector + (NoiseAdapter + Denoiser) + (FM / KD) 조합
    공통: Teacher AE recon loss 항상 계산, 레이어별 손실 합산
    """
    def __init__(
        self,
        cfg,
        trainer,
        teacher_model,
        use_ctc=True,
        use_logit_distillation=True,
        kd_alpha=0.1,
        kd_temperature=1.0,
        use_layerwise_distillation=False,
        layer_kd_alpha=1.0,
        version: int = 1,
        kd_loss_type: str = "mse",
        student_dim: int = 88,
        teacher_dim: int = 176,
        latent_dim: int = 96,
        diffusion_steps: int = 9,
        flow_cfg: dict = None,
        disen_mode: int = 0,
        num_spk: int = 1,
        orth_weight: float = 1.0,
        spk_cls_weight: float = 1.0,
        grl_weight: float = 1.0,
        grl_alpha: float = 1.0,
        grl_anneal: bool = False,
        grl_alpha_max: float = 1.0,
        grl_s_weight: float = 1.0,
        layer_disen_decay: float = 0.0,
        crd_weight: float = 0.0,
        crd_temperature: float = 0.07,
        kd_top_k: int = 0,
    ):
        super().__init__(cfg=cfg, trainer=trainer)

        assert version in range(1, 9), "version은 1~8만 허용합니다."
        self.teacher  = teacher_model
        self.version  = version
        self.use_ctc  = use_ctc
        self.use_logit_distillation    = use_logit_distillation
        self.kd_alpha      = kd_alpha
        self.temperature   = kd_temperature
        self.use_layerwise_distillation = use_layerwise_distillation
        self.layer_kd_alpha = layer_kd_alpha
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.latent_dim  = latent_dim

        self.disen_mode     = disen_mode
        self.orth_weight    = orth_weight
        self.spk_cls_weight = spk_cls_weight
        self.grl_weight        = grl_weight
        self.grl_anneal        = grl_anneal
        self.grl_alpha_max     = grl_alpha_max
        self.grl_s_weight      = grl_s_weight
        self.layer_disen_decay = layer_disen_decay
        self.crd_weight        = crd_weight
        self.crd_temperature   = crd_temperature
        self.kd_top_k          = kd_top_k

        self.recon_crit = nn.MSELoss()
        self.kd_crit    = nn.L1Loss() if kd_loss_type == "l1" else nn.MSELoss()

        # 공통 블록 (disen_mode=0: E1 그대로)
        self.tae      = TeacherAutoEncoder(teacher_dim=teacher_dim, latent_dim=latent_dim)
        self.sproj    = StudentProjector(student_dim=student_dim, latent_dim=latent_dim)
        self.adapter  = NoiseAdapter(latent_dim=latent_dim)
        self.denoiser = SimpleDenoiser(latent_dim=latent_dim, steps=diffusion_steps)

        # disen_mode >= 1: 병렬 인코더 + decoder + speaker classifier
        if disen_mode >= 1:
            self.enc_text_t  = nn.Conv1d(teacher_dim, latent_dim, kernel_size=1)
            self.enc_spk_t   = nn.Conv1d(teacher_dim, latent_dim, kernel_size=1)
            self.proj_text_s = nn.Conv1d(student_dim, latent_dim, kernel_size=1)
            # student spk 인코더 없음: KD 타겟이 이미 speaker-clean(z_t_text)이므로
            # FM+Diffusion이 z_s_text를 z_t_text로 당기면 자연히 text-only 표현이 됨
            self.lat_dec = nn.Conv1d(latent_dim, teacher_dim, kernel_size=1)
            self.spk_cls = SpeakerClassifier(latent_dim=latent_dim, num_spk=num_spk)
            # sample_id → speaker_class 룩업 테이블 (register_buffer: 저장/로드 가능, 학습 X)
            self.spk_table: torch.Tensor | None = None
            # disen_mode=2: CLUB MI 추정기 (orth_loss 대체)
            if disen_mode == 2:
                self.club = ClubGaussian(
                    x_dim=latent_dim, y_dim=latent_dim,
                    hidden_size=128, max_samples=2048,
                )
            # disen_mode=3: orth + GRL on z_t_text (E4/E5)
            # disen_mode=4: orth + GRL on z_t_text + GRL on z_s_text (E6)
            if disen_mode in (3, 4):
                self.grl = GradientReversalLayer(alpha=grl_alpha)
                self.spk_cls_text = SpeakerClassifier(latent_dim=latent_dim, num_spk=num_spk)
            if disen_mode == 4:
                self.grl_s = GradientReversalLayer(alpha=grl_alpha)
                self.spk_cls_s = SpeakerClassifier(latent_dim=latent_dim, num_spk=num_spk)

        _flow_cfg = dict(flow_cfg or {})
        self.fm_latent   = FMLatent(latent_dim=latent_dim, flow_cfg=_flow_cfg)
        self.fm_latent_2 = FMLatent(latent_dim=latent_dim, flow_cfg=_flow_cfg)

        # Forward hook
        self.stu_feats, self.tch_feats = [], []
        assert len(teacher_model.encoder.layers) == len(self.encoder.layers), \
            "student/teacher encoder layer 수가 같아야 합니다."
        for layer in self.encoder.layers:
            layer.register_forward_hook(self._capture_stu_feat)
        for layer in teacher_model.encoder.layers:
            layer.register_forward_hook(self._capture_tch_feat)

    def _capture_stu_feat(self, module, inp, out):   # out: (B, Hs, T)
        self.stu_feats.append(out)

    def _capture_tch_feat(self, module, inp, out):   # out: (B, Ht, T)
        self.tch_feats.append(out)

    def forward(self, input_signal=None, input_signal_length=None,
                processed_signal=None, processed_signal_length=None):
        self.stu_feats.clear()
        self.tch_feats.clear()

        has_input     = input_signal is not None and input_signal_length is not None
        has_processed = processed_signal is not None and processed_signal_length is not None
        if not (has_input ^ has_processed):
            raise ValueError("input_signal 또는 processed_signal 중 하나만 제공해야 합니다.")
        if not has_processed:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_length,
            )

        enc_out_s, enc_len = self.encoder(
            audio_signal=processed_signal, length=processed_signal_length,
        )

        with torch.no_grad():
            proc_t, len_t = self.teacher.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )
            _ = self.teacher.encoder(audio_signal=proc_t, length=len_t)

        log_probs = self.decoder(encoder_output=enc_out_s)
        greedy    = log_probs.argmax(dim=-1)

        if self.training:
            dummy = torch.tensor(0.0, device=enc_out_s.device)
            return log_probs, enc_len, greedy, dummy, enc_out_s
        else:
            return log_probs, enc_len, greedy

    def _compute_v_losses_one_layer(self, s_bht, t_bht, spk_id=None,
                                    layer_idx: int = 0, num_layers: int = 16):
        """단일 레이어 (B,Hs,T), (B,Ht,T) → version별 loss dict"""
        # (B,H,T) → (B,T,H) 로 transpose (원본과 동일)
        s_bct = s_bht.transpose(1, 2)
        t_bct = t_bht.transpose(1, 2)

        out = {
            "recon_loss":    torch.zeros((), device=s_bct.device),
            "kd_loss_pre":   torch.zeros((), device=s_bct.device),
            "fm_loss_pre":   torch.zeros((), device=s_bct.device),
            "kd_loss_post":  torch.zeros((), device=s_bct.device),
            "fm_loss_post":  torch.zeros((), device=s_bct.device),
            "orth_loss":     torch.zeros((), device=s_bct.device),
            "spk_cls_loss":  torch.zeros((), device=s_bct.device),
            "club_mi_loss":  torch.zeros((), device=s_bct.device),
            "club_lll_loss": torch.zeros((), device=s_bct.device),
            "grl_loss":      torch.zeros((), device=s_bct.device),
            "grl_s_loss":    torch.zeros((), device=s_bct.device),
            "crd_loss":      torch.zeros((), device=s_bct.device),
        }

        # ── disen_mode >= 1: 병렬 인코더 + 직교 제약 ──────────────
        if self.disen_mode >= 1:
            # E7: lower layers → stronger disentanglement, upper layers → weaker
            # layer_weight = 1.0 at layer 0, decays linearly to (1 - layer_disen_decay) at last layer
            rel = layer_idx / max(1, num_layers - 1)   # 0.0 (bottom) → 1.0 (top)
            layer_disen_w = 1.0 - self.layer_disen_decay * rel

            # Teacher 병렬 인코더
            z_t_text = self.enc_text_t(t_bct)          # (B, latent_dim, T)
            z_t_spk  = self.enc_spk_t(t_bct)           # (B, latent_dim, T)
            z_t_text_d = z_t_text.detach()

            # Recon: (z_t_text + z_t_spk) → teacher feature
            out["recon_loss"] = self.recon_crit(self.lat_dec(z_t_text + z_t_spk), t_bct)

            # Student 투영 (text only)
            z_s_text = self.proj_text_s(s_bct)          # (B, latent_dim, T)

            # 분리 제약 (layer_disen_w 가중치 적용)
            if self.disen_mode == 1:
                out["orth_loss"] = layer_disen_w * (z_t_text * z_t_spk).sum(dim=1).pow(2).mean()
            elif self.disen_mode == 2:
                out["club_mi_loss"]  = layer_disen_w * self.club.mi_upper(z_t_text, z_t_spk, K=8)
                out["club_lll_loss"] = layer_disen_w * self.club.ll_loss(z_t_text, z_t_spk)
            elif self.disen_mode in (3, 4):
                out["orth_loss"] = layer_disen_w * (z_t_text * z_t_spk).sum(dim=1).pow(2).mean()

            # Speaker classifier (teacher spk latent, all disen modes)
            if spk_id is not None:
                spk_logits = self.spk_cls(z_t_spk)
                out["spk_cls_loss"] = F.cross_entropy(spk_logits, spk_id)
                # disen_mode=3,4: teacher GRL on z_t_text
                if self.disen_mode in (3, 4):
                    grl_logits = self.spk_cls_text(self.grl(z_t_text))
                    out["grl_loss"] = layer_disen_w * F.cross_entropy(grl_logits, spk_id)
                # disen_mode=4: student GRL on z_s_text
                if self.disen_mode == 4:
                    grl_s_logits = self.spk_cls_s(self.grl_s(z_s_text))
                    out["grl_s_loss"] = layer_disen_w * F.cross_entropy(grl_s_logits, spk_id)

            # CRD: InfoNCE on mean-pooled text latents (z_s vs z_t, same utterance = positive)
            if self.crd_weight > 0:
                z_s_pool = F.normalize(z_s_text.mean(dim=2), dim=1)   # (B, D)
                z_t_pool = F.normalize(z_t_text_d.mean(dim=2), dim=1) # (B, D)
                sim = z_s_pool @ z_t_pool.T / self.crd_temperature     # (B, B)
                labels = torch.arange(sim.size(0), device=sim.device)
                out["crd_loss"] = F.cross_entropy(sim, labels)

            # FM(pre) + Diffusion on text subspace only
            fm_loss_pre, _ = self.fm_latent(z_s_text, z_t_text_d)
            out["fm_loss_pre"] = fm_loss_pre
            z_noisy, _      = self.adapter(z_s_text)
            z_deno           = self.denoiser(z_noisy)
            out["kd_loss_post"] = self.kd_crit(z_deno, z_t_text_d)
            return out

        # ── disen_mode == 0: E1 원본 로직 ────────────────────────
        # Teacher AE
        z_t, t_rec    = self.tae(t_bct)
        z_t           = z_t.detach()
        out["recon_loss"] = self.recon_crit(t_rec, t_bct)

        # Student projection
        z_s = self.sproj(s_bct)

        if self.version == 1:
            out["kd_loss_pre"] = self.kd_crit(z_s, z_t)

        elif self.version == 2:
            fm_loss, _ = self.fm_latent(z_s, z_t)
            out["fm_loss_pre"] = fm_loss

        elif self.version == 3:
            z_noisy, _ = self.adapter(z_s)
            z_deno      = self.denoiser(z_noisy)
            out["kd_loss_post"] = self.kd_crit(z_deno, z_t)

        elif self.version == 4:
            fm_loss_pre, _ = self.fm_latent(z_s, z_t)
            z_noisy, _     = self.adapter(z_s)
            z_deno          = self.denoiser(z_noisy)
            out["fm_loss_pre"]  = fm_loss_pre
            out["kd_loss_post"] = self.kd_crit(z_deno, z_t)

        elif self.version == 5:
            z_noisy, _ = self.adapter(z_s)
            z_deno      = self.denoiser(z_noisy)
            fm_loss, _  = self.fm_latent(z_deno, z_t)
            out["fm_loss_post"] = fm_loss

        elif self.version == 6:
            fm_loss_pre, z_s_aligned = self.fm_latent(z_s, z_t)
            z_noisy, _               = self.adapter(z_s_aligned)
            z_deno                   = self.denoiser(z_noisy)
            fm_loss_post, _          = self.fm_latent_2(z_deno, z_t)
            out["fm_loss_pre"]  = fm_loss_pre
            out["fm_loss_post"] = fm_loss_post

        elif self.version == 7:
            fm_loss_pre, _  = self.fm_latent(z_s, z_t)
            z_noisy, _      = self.adapter(z_s)
            z_deno           = self.denoiser(z_noisy)
            fm_loss_post, _ = self.fm_latent_2(z_deno, z_t)
            out["fm_loss_pre"]  = fm_loss_pre
            out["fm_loss_post"] = fm_loss_post

        elif self.version == 8:
            fm_loss_pre, z_s_aligned = self.fm_latent(z_s, z_t)
            z_noisy, _               = self.adapter(z_s_aligned)
            z_deno                   = self.denoiser(z_noisy)
            out["fm_loss_pre"]  = fm_loss_pre
            out["kd_loss_post"] = self.kd_crit(z_deno, z_t)

        return out

    def training_step(self, batch, batch_idx):
        if len(batch) == 5:
            signal, sig_len, transcript, transcript_len, sample_id = batch
            # sample_id(dataset index) → speaker class label
            if self.disen_mode >= 1 and self.spk_table is not None:
                spk_id = self.spk_table.to(signal.device)[sample_id.long()]
            else:
                spk_id = None
        else:
            signal, sig_len, transcript, transcript_len = batch
            spk_id = None

        log_probs, enc_len, _, _dummy, enc_out = self.forward(
            input_signal=signal, input_signal_length=sig_len,
        )

        # 1) CTC
        ctc_loss = torch.tensor(0.0, device=log_probs.device)
        if self.use_ctc:
            ctc_loss = self.loss(
                log_probs=log_probs,
                targets=transcript,
                input_lengths=enc_len,
                target_lengths=transcript_len,
            )

        # 2) Logit KD
        logit_kd_loss = torch.tensor(0.0, device=log_probs.device)
        if self.use_logit_distillation and len(self.tch_feats) > 0:
            with torch.no_grad():
                tch_logp = self.teacher.decoder(
                    encoder_output=self.tch_feats[-1].permute(0, 2, 1),
                )
                tch_p = F.softmax(tch_logp / self.temperature, dim=-1)
            stu_logp = F.log_softmax(log_probs / self.temperature, dim=-1)
            logit_kd_loss = (
                F.kl_div(stu_logp, tch_p, reduction="batchmean")
                * (self.temperature ** 2)
            )

        # 3) Layerwise MSE KD (옵션)
        layer_kd_loss = torch.tensor(0.0, device=log_probs.device)
        if self.use_layerwise_distillation:
            layer_proj = nn.Linear(self.student_dim, self.teacher_dim).to(log_probs.device)
            for s, t in zip(self.stu_feats, self.tch_feats):
                B, Hs, T = s.shape
                Ht = t.size(1)
                s_flat = s.transpose(1, 2).reshape(-1, Hs)
                p_flat = layer_proj(s_flat)
                s_proj = p_flat.reshape(B, T, Ht).transpose(1, 2)
                layer_kd_loss = layer_kd_loss + F.mse_loss(s_proj, t)
            layer_kd_loss = layer_kd_loss / max(1, len(self.stu_feats))

        # 4) Version별 latent space 손실 (레이어 합산, 원본과 동일)
        # E5: DANN-style alpha annealing — training_step마다 grl.alpha 업데이트
        if self.disen_mode in (3, 4) and self.grl_anneal:
            total_steps = max(1, self.trainer.estimated_stepping_batches)
            p     = self.global_step / total_steps          # 0 → 1
            alpha = self.grl_alpha_max * (2 / (1 + math.exp(-10 * p)) - 1)
            self.grl.alpha = alpha
            if self.disen_mode == 4:
                self.grl_s.alpha = alpha
            self.log("v/grl_alpha", alpha, on_step=True, on_epoch=False)

        recon_sum    = torch.zeros((), device=log_probs.device)
        kd_pre_sum   = torch.zeros((), device=log_probs.device)
        fm_pre_sum   = torch.zeros((), device=log_probs.device)
        kd_post_sum  = torch.zeros((), device=log_probs.device)
        fm_post_sum  = torch.zeros((), device=log_probs.device)
        orth_sum     = torch.zeros((), device=log_probs.device)
        spk_cls_sum  = torch.zeros((), device=log_probs.device)
        club_mi_sum  = torch.zeros((), device=log_probs.device)
        club_lll_sum = torch.zeros((), device=log_probs.device)
        grl_sum      = torch.zeros((), device=log_probs.device)
        grl_s_sum    = torch.zeros((), device=log_probs.device)
        crd_sum      = torch.zeros((), device=log_probs.device)
        num_layers = len(self.stu_feats)
        k = self.kd_top_k if self.kd_top_k > 0 else num_layers
        stu_feats_sel = self.stu_feats[-k:]
        tch_feats_sel = self.tch_feats[-k:]
        # layer_idx는 전체 레이어 기준 유지 (E7 layer_disen_w 계산에 사용)
        offset = num_layers - k
        for layer_idx, (s, t) in enumerate(zip(stu_feats_sel, tch_feats_sel), start=offset):
            losses       = self._compute_v_losses_one_layer(s, t, spk_id=spk_id,
                                                            layer_idx=layer_idx,
                                                            num_layers=num_layers)
            recon_sum    += losses["recon_loss"]
            kd_pre_sum   += losses["kd_loss_pre"]
            fm_pre_sum   += losses["fm_loss_pre"]
            kd_post_sum  += losses["kd_loss_post"]
            fm_post_sum  += losses["fm_loss_post"]
            orth_sum     += losses["orth_loss"]
            spk_cls_sum  += losses["spk_cls_loss"]
            club_mi_sum  += losses["club_mi_loss"]
            club_lll_sum += losses["club_lll_loss"]
            grl_sum      += losses["grl_loss"]
            grl_s_sum    += losses["grl_s_loss"]
            crd_sum      += losses["crd_loss"]

        # 5) 총 loss
        total_loss = (
            ctc_loss
            + self.kd_alpha * logit_kd_loss
            + self.layer_kd_alpha * layer_kd_loss
            + recon_sum
            + kd_pre_sum + kd_post_sum
            + fm_pre_sum + fm_post_sum
            + self.orth_weight * orth_sum
            + self.spk_cls_weight * spk_cls_sum
            + self.orth_weight * club_mi_sum   # club_mi도 orth_weight 재사용
            + club_lll_sum                     # variational net 학습, 별도 weight 없음
            + self.grl_weight * grl_sum
            + self.grl_s_weight * grl_s_sum
            + self.crd_weight * crd_sum
        )

        # 6) 로깅
        self.log("loss/ctc",      ctc_loss,      on_step=True, on_epoch=True)
        self.log("loss/logit_kd", logit_kd_loss, on_step=True, on_epoch=True)
        self.log("loss/layer_kd", layer_kd_loss, on_step=True, on_epoch=True)
        self.log("v/recon",      recon_sum,    on_step=True, on_epoch=True)
        self.log("v/kd_pre",     kd_pre_sum,   on_step=True, on_epoch=True)
        self.log("v/fm_pre",     fm_pre_sum,   on_step=True, on_epoch=True)
        self.log("v/kd_post",    kd_post_sum,  on_step=True, on_epoch=True)
        self.log("v/fm_post",    fm_post_sum,  on_step=True, on_epoch=True)
        self.log("v/orth",       orth_sum,     on_step=True, on_epoch=True)
        self.log("v/spk_cls",    spk_cls_sum,  on_step=True, on_epoch=True)
        self.log("v/club_mi",    club_mi_sum,  on_step=True, on_epoch=True)
        self.log("v/club_lll",   club_lll_sum, on_step=True, on_epoch=True)
        self.log("v/grl",        grl_sum,      on_step=True, on_epoch=True)
        self.log("v/grl_s",      grl_s_sum,    on_step=True, on_epoch=True)
        self.log("v/crd",        crd_sum,      on_step=True, on_epoch=True)
        self.log("train_loss",   total_loss,   on_step=True, on_epoch=True, prog_bar=True)
        return total_loss


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser("GRP-KD (asr_train_diffm) baseline")

    # Data (DAG-KD 동일)
    p.add_argument("--data_dir",      type=str,      default="data")
    p.add_argument("--data_script",   type=str,      default="./librispeech_asr.py")
    p.add_argument("--data_cfg",      type=str,      default="train_100")
    p.add_argument("--train_split",   type=str,      default="train.clean.100")
    p.add_argument("--val_split",     type=str,      default="dev.clean")
    p.add_argument("--test_split",    type=str,      default="test.clean")
    p.add_argument("--sample_rate",   type=int,      default=16000)
    p.add_argument("--batch_size",    type=int,      default=32)

    # Training
    p.add_argument("--epochs",        type=int,      default=100)
    p.add_argument("--gpus",          type=int,      default=1)
    p.add_argument("--out",           type=str,      default="outputs/grp_kd")
    p.add_argument("--teacher_name",  type=str,      default="stt_en_conformer_ctc_small")
    p.add_argument("--resume_ckpt",   type=str,      default=None)
    p.add_argument("--test_mode",     type=str2bool, default=False)

    # W&B
    p.add_argument("--wandb_run",     type=str, default=os.getenv("EXP_NAME", "grp_kd"))
    p.add_argument("--wandb_project", type=str, default=os.getenv("PRJ_NAME", "dag-kd"))

    # Model version (asr_train_diffm.py 동일)
    p.add_argument("--model_version", type=int, default=1,
                   help="1~8: ver1=AE+KD, ver2=AE+FM, ver3=AE+Diff+KD, ...")
    p.add_argument("--latent_dim",    type=int, default=96)
    p.add_argument("--diffusion_steps", type=int, default=9)
    p.add_argument("--kd_loss_type",  type=str, default="mse", choices=["mse", "l1"])

    # KD
    p.add_argument("--use_ctc",               type=str2bool, default=True)
    p.add_argument("--use_logit_distillation", type=str2bool, default=True)
    p.add_argument("--kd_alpha",              type=float,    default=0.1)
    p.add_argument("--kd_temperature",        type=float,    default=1.0)
    p.add_argument("--use_layerwise_distillation", type=str2bool, default=False)
    p.add_argument("--layer_kd_alpha",        type=float,    default=1.0)

    # Flow Matching
    p.add_argument("--flow_steps",  type=int,   default=8)
    p.add_argument("--flow_weight", type=float, default=1.0)

    # Disentanglement (disen_mode=0: E1 그대로, disen_mode=1: orth+spk_cls)
    p.add_argument("--disen_mode",     type=int,   default=0,
                   help="0=E1, 1=orth+spk_cls, 2=CLUB MI, 3=orth+GRL(teacher), 4=orth+GRL(teacher+student)")
    p.add_argument("--orth_weight",    type=float, default=1.0)
    p.add_argument("--spk_cls_weight", type=float, default=1.0)
    p.add_argument("--grl_weight",     type=float, default=1.0)
    p.add_argument("--grl_alpha",      type=float, default=0.1,
                   help="고정 alpha (grl_anneal=False) 또는 annealing 시작값(=0으로 고정)")
    p.add_argument("--grl_anneal",     type=str2bool, default=False,
                   help="True: DANN-style alpha annealing (0→grl_alpha_max)")
    p.add_argument("--grl_alpha_max",  type=float, default=1.0,
                   help="annealing 최대 alpha (grl_anneal=True일 때 사용)")
    p.add_argument("--grl_s_weight",   type=float, default=1.0,
                   help="student GRL loss 가중치 (disen_mode=4)")
    p.add_argument("--layer_disen_decay", type=float, default=0.0,
                   help="layer-selective disentanglement: orth/grl weight = 1 - decay*(layer_idx/(N-1)). "
                        "0.0=uniform(E1~E6), 0.8=E7(lower layers stronger)")

    # CRD (Contrastive Representation Distillation)
    p.add_argument("--crd_weight",      type=float, default=0.0,
                   help="CRD InfoNCE loss 가중치 (default 0.0=비활성, E8: 1.0)")
    p.add_argument("--crd_temperature", type=float, default=0.07,
                   help="CRD InfoNCE temperature (default 0.07)")
    p.add_argument("--kd_top_k",        type=int,   default=0,
                   help="상위 K개 레이어만 KD에 사용 (default 0=전체, E9: 4)")

    args = p.parse_args()

    # ── 출력 & manifest ──────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    snapshot_sources(args.out)

    manifest_dir = os.path.join(args.data_dir, args.data_cfg, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)
    cache_dir = os.path.join(args.data_dir, args.data_cfg, "cache")

    dl_cfg = DownloadConfig(
        cache_dir=cache_dir, force_download=False, resume_download=True,
        max_retries=10, disable_tqdm=False,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=72000)}},
        delete_extracted=False, extract_compressed_file=True, force_extract=True,
    )

    train_ds = load_dataset(args.data_script, args.data_cfg, split=args.train_split,
                            trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
    val_ds   = load_dataset(args.data_script, args.data_cfg, split=args.val_split,
                            trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)
    test_ds  = load_dataset(args.data_script, args.data_cfg, split=args.test_split,
                            trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir)

    # dev_other / test_other (없으면 skip)
    extra_splits = {}
    for split_name in ["dev.other", "test.other"]:
        try:
            extra_splits[split_name] = load_dataset(
                args.data_script, args.data_cfg, split=split_name,
                trust_remote_code=True, download_config=dl_cfg, cache_dir=cache_dir,
            )
        except Exception:
            pass

    spk_map_path        = os.path.join(manifest_dir, "speaker_id_mapping.json")
    train_manifest      = os.path.join(manifest_dir, "train.json")
    dev_clean_manifest  = os.path.join(manifest_dir, "dev_clean.json")
    dev_other_manifest  = os.path.join(manifest_dir, "dev_other.json")
    test_clean_manifest = os.path.join(manifest_dir, "test_clean.json")
    test_other_manifest = os.path.join(manifest_dir, "test_other.json")

    spk2idx, idx2spk = scan_speakers(train_ds)
    num_spk = len(spk2idx)
    phys_cache_root  = Path(manifest_dir) / "phys_cache"

    if not os.path.isfile(train_manifest):
        build_manifest_from_hf_with_meta(train_ds, train_manifest, cache_dir, spk2idx, "train", phys_cache_root)
    if not os.path.isfile(dev_clean_manifest):
        build_manifest_from_hf_with_meta(val_ds, dev_clean_manifest, cache_dir, spk2idx, "dev_clean", phys_cache_root)
    if not os.path.isfile(test_clean_manifest):
        build_manifest_from_hf_with_meta(test_ds, test_clean_manifest, cache_dir, spk2idx, "test_clean", phys_cache_root)
    if "dev.other" in extra_splits and not os.path.isfile(dev_other_manifest):
        build_manifest_from_hf_with_meta(extra_splits["dev.other"], dev_other_manifest, cache_dir, spk2idx, "dev_other", phys_cache_root)
    if "test.other" in extra_splits and not os.path.isfile(test_other_manifest):
        build_manifest_from_hf_with_meta(extra_splits["test.other"], test_other_manifest, cache_dir, spk2idx, "test_other", phys_cache_root)

    spk_dur_path = os.path.join(manifest_dir, "speaker_durations_train.json")
    if not os.path.isfile(spk_dur_path):
        compute_speaker_durations(train_manifest, spk_dur_path)

    if args.test_mode:
        tm_train = os.path.join(manifest_dir, "test_mode_train.json")
        tm_val   = os.path.join(manifest_dir, "test_mode_val.json")
        tm_test  = os.path.join(manifest_dir, "test_mode_test.json")
        build_manifest_from_hf_with_meta(train_ds.select(range(100)), tm_train, cache_dir, spk2idx, "test_mode_train", phys_cache_root)
        build_manifest_from_hf_with_meta(val_ds.select(range(100)),   tm_val,   cache_dir, spk2idx, "test_mode_val",   phys_cache_root)
        build_manifest_from_hf_with_meta(test_ds.select(range(100)),  tm_test,  cache_dir, spk2idx, "test_mode_test",  phys_cache_root)
        train_manifest, dev_clean_manifest, test_clean_manifest = tm_train, tm_val, tm_test
        args.epochs = 5

    # ── Trainer & W&B ────────────────────────────────────────
    wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_run, save_dir=args.out)
    ckpt_dir     = os.path.join(args.out, "checkpoints")
    ckpt_cb      = ModelCheckpoint(dirpath=ckpt_dir, filename="last", save_top_k=0, save_last=True)

    trainer = pl.Trainer(
        devices=args.gpus, accelerator="gpu",
        max_epochs=args.epochs, default_root_dir=args.out,
        logger=wandb_logger, callbacks=[ckpt_cb],
    )

    # ── Teacher ───────────────────────────────────────────────
    teacher = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name=args.teacher_name,
        map_location="cuda:0" if torch.cuda.is_available() else "cpu",
        trainer=trainer,
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    release_nemoAPI(teacher, out_folder=os.path.join(args.out, "nemo_archive"))

    # ── Student config ────────────────────────────────────────
    stu_cfg = deepcopy(teacher.cfg)
    stu_cfg.train_ds.is_tarred              = False
    stu_cfg.train_ds.manifest_filepath      = train_manifest
    stu_cfg.train_ds.sample_rate            = args.sample_rate
    stu_cfg.train_ds.batch_size             = args.batch_size
    stu_cfg.validation_ds.is_tarred         = False
    stu_cfg.validation_ds.manifest_filepath = dev_clean_manifest
    stu_cfg.validation_ds.sample_rate       = args.sample_rate
    stu_cfg.validation_ds.batch_size        = args.batch_size
    stu_cfg.test_ds.is_tarred               = False
    stu_cfg.test_ds.manifest_filepath       = test_clean_manifest
    stu_cfg.test_ds.sample_rate             = args.sample_rate
    stu_cfg.test_ds.batch_size              = args.batch_size
    stu_cfg.encoder.d_model = max(8, teacher.cfg.encoder.d_model // 2)   # 88
    stu_cfg.encoder.n_heads = max(1, teacher.cfg.encoder.n_heads // 2)   # 4
    stu_cfg.decoder.feat_in = max(8, teacher.cfg.decoder.feat_in  // 2)  # 88

    OmegaConf.set_struct(stu_cfg, False)
    for ds_key in ("train_ds", "validation_ds", "test_ds"):
        OmegaConf.set_struct(getattr(stu_cfg, ds_key), False)
    stu_cfg.train_ds.return_sample_id      = True
    stu_cfg.validation_ds.return_sample_id = False
    stu_cfg.test_ds.return_sample_id       = False

    # ── Dimensions ────────────────────────────────────────────
    dim_s = stu_cfg.encoder.d_model   # 88
    dim_t = teacher.cfg.encoder.d_model  # 176

    # ── Flow cfg ─────────────────────────────────────────────
    flow_cfg = {
        "meta_encoder_type": "mlp",
        "hidden_dim":        args.latent_dim,
        "time_embed_dim":    32,
        "training_sampling": args.flow_steps,
        "weight":            args.flow_weight,
        "noise_schedule":    "rectified",
        "shape_transform":   "identity",
        "loss":              "mse",
    }

    # ── Model ─────────────────────────────────────────────────
    model = DistilFlowMatchingCTCModelBPE(
        cfg=stu_cfg,
        trainer=trainer,
        teacher_model=teacher,
        use_ctc=args.use_ctc,
        use_logit_distillation=args.use_logit_distillation,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
        use_layerwise_distillation=args.use_layerwise_distillation,
        layer_kd_alpha=args.layer_kd_alpha,
        version=args.model_version,
        kd_loss_type=args.kd_loss_type,
        student_dim=dim_s,
        teacher_dim=dim_t,
        latent_dim=args.latent_dim,
        diffusion_steps=args.diffusion_steps,
        flow_cfg=flow_cfg,
        disen_mode=args.disen_mode,
        num_spk=num_spk,
        orth_weight=args.orth_weight,
        spk_cls_weight=args.spk_cls_weight,
        grl_weight=args.grl_weight,
        grl_alpha=args.grl_alpha,
        grl_anneal=args.grl_anneal,
        grl_alpha_max=args.grl_alpha_max,
        grl_s_weight=args.grl_s_weight,
        layer_disen_decay=args.layer_disen_decay,
        crd_weight=args.crd_weight,
        crd_temperature=args.crd_temperature,
        kd_top_k=args.kd_top_k,
    )

    # disen_mode >= 1: sample_id → speaker_class 룩업 테이블 주입
    if args.disen_mode >= 1:
        _manifest = train_manifest if not args.test_mode else \
            os.path.join(manifest_dir, "test_mode_train.json")
        model.spk_table = load_speaker_table_from_manifest(_manifest)
        print(f"[INFO] spk_table loaded: {len(model.spk_table)} samples, {num_spk} speakers")

    # ── Train ─────────────────────────────────────────────────
    ckpt_path = args.resume_ckpt
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[INFO] Resuming from: {ckpt_path}")
        trainer.fit(model, ckpt_path=ckpt_path)
    else:
        if ckpt_path:
            print(f"[WARN] Checkpoint not found at {ckpt_path}, training from scratch.")
        trainer.fit(model)

    # ── Evaluation (DAG-KD 동일 방식) ────────────────────────
    eval_targets = [
        ("dev_clean",  dev_clean_manifest),
        ("dev_other",  dev_other_manifest),
        ("test_clean", test_clean_manifest),
        ("test_other", test_other_manifest),
    ]
    for split_name, manifest in eval_targets:
        if not os.path.isfile(manifest):
            print(f"[WARN] manifest not found for {split_name}: {manifest}, skip")
            continue

        print(f"\n===== Evaluating on {split_name} =====")

        test_cfg = deepcopy(model.cfg.test_ds)
        test_cfg.manifest_filepath = manifest
        test_cfg.shuffle = False
        model.setup_test_data(test_cfg)
        dl      = model.test_dataloader()
        results = trainer.test(model=model, dataloaders=[dl], verbose=True)

        if results and isinstance(results, list):
            res  = results[0]
            wer  = res.get("test_wer", res.get("wer", None))
            loss = res.get("test_loss", res.get("loss", None))
            if wer is not None:
                print(f"→ {split_name}: wer={wer:.2%}")
                wandb_logger.log_metrics({f"{split_name}/wer": wer}, step=trainer.current_epoch)

        del dl, results
        gc.collect()
        torch.cuda.empty_cache()

        # per-sample WER mean ± std
        with open(manifest, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f]
        audio_files = [e["audio_filepath"] for e in entries]
        ref_texts   = [e["text"] for e in entries]

        hyps = model.transcribe(
            audio=audio_files, batch_size=args.batch_size,
            return_hypotheses=False, num_workers=0, verbose=False,
        )
        sample_wers     = compute_sample_wers(ref_texts, hyps)
        sample_wers_pct = [w * 100.0 for w in sample_wers]
        wer_mean = float(statistics.mean(sample_wers_pct))
        wer_std  = float(statistics.stdev(sample_wers_pct)) if len(sample_wers_pct) > 1 else 0.0
        print(f"→ {split_name}: per-sample WER = {wer_mean:.2f}% ± {wer_std:.2f}%")

        wandb_logger.log_metrics(
            {f"{split_name}/wer_mean": wer_mean, f"{split_name}/wer_std": wer_std},
            step=trainer.current_epoch,
        )

        # WER 분포 plot
        plot_dir = os.path.join(args.out, "xai/wer_plots")
        os.makedirs(plot_dir, exist_ok=True)
        wers_np = np.array(sample_wers_pct, dtype=float)
        torch.cuda.empty_cache()
        try:
            fig, ax = plt.subplots()
            ax.hist(wers_np, bins=[0, 10, 20, 30, 50, 100, 200], edgecolor="black")
            ax.set_xlabel("Per-sample WER (%)")
            ax.set_ylabel("Count")
            ax.set_title(f"WER Histogram - {split_name}\nmean={wer_mean:.2f}%, std={wer_std:.2f}%")
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"wer_hist_{split_name}.png"))
        finally:
            plt.close("all")


if __name__ == "__main__":
    main()
