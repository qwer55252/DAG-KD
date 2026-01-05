# models/distil_dagkd_ctc_bpe_ver2.py
from difflib import diff_bytes
import math
import json
from numpy import diff
import torch
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from nemo.collections import asr as nemo_asr
from utils import (
    masked_mse,
    extract_speaker_ids_from_batch,
    load_speaker_table_from_manifest,
)


def kl_div_logits(student_logits, teacher_logits, T: float = 1.0) -> torch.Tensor:
    """
    KL( teacher || student ) over vocab dimension, averaged
    logits: (B,T,V)
    """
    T = float(T)
    s_logp = F.log_softmax(student_logits / T, dim=-1)
    t_prob = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(s_logp, t_prob, reduction="batchmean") * (T * T)
    return kl


class FlowMatchingGenKD(nn.Module):
    """
    매우 단순한 Flow-Matching 스타일 GenKD:
    x0=student_z, x1=teacher_z
    xt = (1-t)*x0 + t*x1
    target flow = (x1 - x0)
    predictor(xt) -> flow_hat
    """
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # z_*: (B,T,D)
        B, T, D = z_s.shape
        t = torch.rand((B, 1, 1), device=z_s.device)
        xt = (1 - t) * z_s + t * z_t
        target = (z_t - z_s)
        pred = self.net(xt)
        loss = (pred - target).pow(2).mean()
        return loss

def rectified_flow_schedule(t: torch.Tensor):
    # alpha=t, sigma=1-t (단순 버전)
    alpha = t
    sigma = 1.0 - t
    return alpha, sigma

def rectified_flow_schedule_deriv(t: torch.Tensor):
    dalpha_dt = torch.ones_like(t)
    dsigma_dt = -torch.ones_like(t)
    return dalpha_dt, dsigma_dt


class FlowMatchingModule(nn.Module):
    def __init__(self, flow_cfg: dict):
        super().__init__()
        self.training_sampling  = int(flow_cfg.get("training_sampling", 8))
        self.inference_sampling = int(flow_cfg.get("inference_sampling", 8))

        self.student_dim = int(flow_cfg["student_dim"])
        self.teacher_dim = int(flow_cfg["teacher_dim"])
        self.latent_dim = int(flow_cfg["latent_dim"])

        time_embed_dim = int(flow_cfg.get("time_embed_dim", 32))
        hidden_dim     = int(flow_cfg.get("hidden_dim", 256))

        # student_dim 공간에서 velocity 예측
        self.time_embed = nn.Linear(1, time_embed_dim)
        self.meta_encoder = nn.Sequential(
            nn.Linear(self.latent_dim + time_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.latent_dim),
        )

        
        self.detach_teacher_feat = bool(flow_cfg.get("detach_teacher_feat", True))
        self.metric = nn.MSELoss()

    def forward(self, student_text_emb: torch.Tensor, teacher_text_emb: torch.Tensor | None = None, lengths=None, layer_sampling_step=None):
        """
        student_text_emb: (B,T,latent_dim)
        teacher_text_emb: (B,T,latent_dim) or None
        """
        if teacher_text_emb is not None and self.detach_teacher_feat:
            teacher_text_emb = teacher_text_emb.detach()
        steps = int(layer_sampling_step or (self.training_sampling if self.training else self.inference_sampling))

        x = student_text_emb
        losses = []

        # reverse time: steps -> 1 (모든 step에서 loss 누적)
        for i in range(steps, 0, -1):
            t = torch.full((x.size(0), x.size(1), 1), i / steps, device=x.device)  # (B,T,1)
            t_emb = self.time_embed(t)  # (B,T,Te)
            inp = torch.cat([x, t_emb], dim=-1)  # (B,T,L+Te)
            velocity = self.meta_encoder(inp)    # (B,T,L)

            # update
            x = x - velocity / steps

            # step별 loss 누적 (안정적)
            if self.training and teacher_text_emb is not None:
                losses.append(masked_mse(x, teacher_text_emb, lengths))

        # last_step loss(원래 코드처럼 마지막만 쓰고 싶으면)
        if self.training and teacher_text_emb is not None:
            loss = torch.stack(losses).mean() if len(losses) else torch.zeros((), device=student_text_emb.device)
        else:
            loss = torch.zeros((), device=student_text_emb.device)

        return loss


class DiffusionGenKD(nn.Module):
    """
    매우 단순한 DiffKD(denoise) 스타일:
    z_t를 노이즈해서 z_noisy 만들고, student z_s를 조건으로 noise 예측
    """
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True))
        self.net = nn.Sequential(
            nn.Linear(hidden + dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise = torch.randn_like(z_t)
        alpha = torch.rand((z_t.size(0), 1, 1), device=z_t.device) * 0.5 + 0.25
        z_noisy = alpha * z_t + (1 - alpha) * noise
        h = self.cond(z_s)
        inp = torch.cat([h, z_noisy], dim=-1)
        pred_noise = self.net(inp)
        loss = (pred_noise - noise).pow(2).mean()
        return loss

class DiffKDModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.steps       = int(cfg.get("steps", 5))
        self.teacher_dim = int(cfg["teacher_dim"])
        self.student_dim = int(cfg["student_dim"])
        self.latent_dim  = int(cfg.get("latent_dim", 96))

        # teacher feature는 항상 detach (teacher freeze 전제)
        self.detach_teacher_feat = bool(cfg.get("detach_teacher_feat", True))

        hidden = int(cfg.get("denoiser_hidden", self.latent_dim))
        self.denoiser = nn.Sequential(
            nn.Conv1d(self.latent_dim, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, self.latent_dim, 3, padding=1),
        )

        self.recon_weight   = float(cfg.get("recon_weight", 1.0))
        self.distill_weight = float(cfg.get("distill_weight", 1.0))

    def forward(self, stu_text_emb, tch_text_emb, lengths=None, sampling_steps=None):
        """
        stu_text_emb, tch_text_emb: (B,T,D)
        """
        if self.detach_teacher_feat:
            tch_text_emb = tch_text_emb.detach()

        steps = int(self.steps if sampling_steps is None else sampling_steps)

        # Conv1d용 (B,D,T)
        stu = stu_text_emb.transpose(1, 2)
        tch = tch_text_emb.transpose(1, 2)

        # ---------- Teacher latent ----------
        z_t = tch                                # (B,L,T) where L==teacher_dim==latent_dim
        ae_loss = torch.zeros((), device=stu_text_emb.device)

        # ---------- Student latent ----------
        z_s = stu                                # (B,L,T)

        # ---------- Iterative denoise ----------
        x = z_s
        for _ in range(steps):
            pred_noise = self.denoiser(x)
            x = x - pred_noise / steps
        denoised = x                                  # (B,L,T)

        # ---------- KD loss ----------
        distill_loss = masked_mse(
            denoised.transpose(1, 2),                 # (B,T,L)
            z_t.transpose(1, 2),                      # (B,T,L)
            lengths
        )

        return self.recon_weight * ae_loss + self.distill_weight * distill_loss


def _to_BTC(x, feat_dim: int):
    # (B,C,T) -> (B,T,C)
    if x.dim() == 3:
        if x.shape[1] == feat_dim and x.shape[2] != feat_dim:
            return x.transpose(1, 2).contiguous()
    return x

def _to_BCT(x):
    # (B,T,C) -> (B,C,T)
    return x.transpose(1, 2).contiguous()

class BottleneckEncDecCTCModelBPE(nemo_asr.models.EncDecCTCModelBPE):
    """
    EncDecCTCModelBPE를 직접 상속해서 forward 경로에 bottleneck을 강제.
    transcribe()도 결국 self.forward()를 타므로 inference에서도 bottleneck이 항상 적용됨.
    """
    def __init__(
        self,
        cfg,
        trainer=None,
        disen_mi_weight: float = 1e-3,
        disen_lll_weight: float = 1.0,
        freeze_pretrained_encoder: bool = False,
        freeze_pretrained_decoder: bool = False,
    ):
        super().__init__(cfg=cfg, trainer=trainer)

        in_dim = int(self.encoder._feat_out) if hasattr(self.encoder, "_feat_out") else int(cfg.encoder.d_model)
        latent_dim = int(getattr(cfg, "latent_dim", 96))
        num_spk = int(getattr(cfg, "num_spk", 0))
        mi_pairs = str(getattr(cfg, "disen_mi_pairs", "ts,tp,ps"))

        self.disent = DisentanglementBottleneck(
            in_dim=in_dim,
            latent_dim=latent_dim,
            num_spk=num_spk,
            mi_pairs=mi_pairs,
            use_txt_spk_probe=bool(getattr(cfg, "use_txt_spk_probe", True)),
            mi_clamp_min0=bool(getattr(cfg, "mi_clamp_min0", True)),
        )

        self.disen_mi_weight = float(disen_mi_weight)
        self.disen_lll_weight = float(disen_lll_weight)

        if freeze_pretrained_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if freeze_pretrained_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False
        train_manifest = self.cfg.train_ds.manifest_filepath
        self.manifest_speakers = self._load_manifest_speakers(train_manifest)

    @torch.no_grad()
    def load_from_pretrained_nemo(self, nemo_model, strict=False):
        # nemo_model도 EncDecCTCModelBPE라 가정
        missing, unexpected = self.load_state_dict(nemo_model.state_dict(), strict=strict)
        return missing, unexpected

    def _extract_speaker_ids(self, batch: Any) -> Optional[torch.Tensor]:
        if isinstance(batch, dict):
            for k in ["speaker", "speaker_id", "speaker_ids", "spk_id", "spk_ids"]:
                if k in batch:
                    return batch[k]
        return None

    def forward_with_z(self, input_signal, input_signal_length,
                speaker_ids=None, compute_loss=False, grl_lambda=1.0):

        # 1) waveform -> features
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal, length=input_signal_length
        )

        # 2) spec augment (train only)
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        # 3) encoder
        enc_out, enc_len = self.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )
        enc_out = _to_BTC(enc_out, self.disent.in_dim) # (B,C,T) -> (B,T,C)
        
        # 4) bottleneck
        disen_out = self.disent(
            enc_out=enc_out, enc_len=enc_len,
            speaker_ids=speaker_ids,
            compute_loss=compute_loss,
            grl_lambda=grl_lambda,
        )

        # 5) decoder
        dec_in = _to_BCT(disen_out.z_for_decoder) # (B,T,C) -> (B,C,T)

        log_probs = self.decoder(
            encoder_output=dec_in
        )

        return log_probs, enc_len, disen_out.z_text_latent, disen_out.aux


    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
        **kwargs,
    ):
        # speaker_ids는 validation/test에서는 보통 안 들어옴 → None이면 disent loss만 안 계산됨
        speaker_ids = kwargs.get("speaker_ids", None)
        
        log_probs, enc_len, _, _ = self.forward_with_z(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            speaker_ids=speaker_ids,
            compute_loss=False,
        )
        greedy_predictions = log_probs.argmax(dim=-1)
        return log_probs, enc_len, greedy_predictions
    
    def _load_manifest_speakers(self, manifest_path: str):
        spk = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                spk.append(int(obj.get("spk_idx", -1)))
        # (N,) 텐서로 들고 있기
        return torch.tensor(spk, dtype=torch.long)  # CPU에 둬도 OK


    
def _to_long_tensor(x, device):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device=device).long()
    # list/tuple/int/np 등
    return torch.tensor(x, device=device).long()

# def extract_speaker_ids_from_batch(batch, device):
#     """
#     utils.py의 build_manifest_from_hf_with_meta()가 넣는 'speaker' (index)를 꺼낸다.
#     가능한 경우들:
#       1) batch가 dict이고 'speaker' 키가 있음
#       2) batch가 tuple/list이고 마지막에 misc가 붙으며 misc 안에 speaker가 있음 (add_misc=True)
#     """
#     # 1) dict batch
#     if isinstance(batch, dict):
#         if "speaker" in batch:
#             return _to_long_tensor(batch["speaker"], device)
#         if "speaker_id" in batch:
#             return _to_long_tensor(batch["speaker_id"], device)
#         return None

#     # 2) tuple/list batch: (signal, signal_len, tokens, tokens_len, misc?) 형태
#     if isinstance(batch, (tuple, list)) and len(batch) >= 5:
#         misc = batch[4]

#         # misc가 dict(list) 타입인 경우들 처리
#         # 케이스 A: misc = list[dict] (B개)
#         if isinstance(misc, (list, tuple)) and len(misc) > 0 and isinstance(misc[0], dict):
#             sids = []
#             for d in misc:
#                 v = d.get("speaker", d.get("speaker_id", -1))
#                 try:
#                     sids.append(int(v))
#                 except Exception:
#                     sids.append(-1)
#             return torch.tensor(sids, device=device, dtype=torch.long)

#         # 케이스 B: misc = dict of lists
#         if isinstance(misc, dict):
#             if "speaker" in misc:
#                 return _to_long_tensor(misc["speaker"], device)
#             if "speaker_id" in misc:
#                 return _to_long_tensor(misc["speaker_id"], device)

#     return None


class TeacherASRWithDisent(BottleneckEncDecCTCModelBPE):
    def __init__(self, cfg, trainer=None, **kwargs):
        super().__init__(cfg=cfg, trainer=trainer, **kwargs)
        self.num_spk = int(getattr(cfg, "num_spk", 0))

    def training_step(self, batch, batch_idx):
        if len(batch) == 5:
            audio, audio_len, tokens, tokens_len, sample_ids = batch
            sample_ids = sample_ids.long()
            
            # sample_id -> speaker_id lookup
            speaker_ids = self.manifest_speakers.to(sample_ids.device)[sample_ids]
        elif len(batch) == 4:
            audio, audio_len, tokens, tokens_len = batch
            speaker_ids = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        if speaker_ids is not None:
            assert speaker_ids.min().item() >= -1
            assert speaker_ids.max().item() < self.num_spk

        log_probs, enc_len, z_lat, aux = self.forward_with_z(
            input_signal=audio,
            input_signal_length=audio_len,
            speaker_ids=speaker_ids,
            compute_loss=True,
            grl_lambda=1.0,
        )

        loss_ctc = self.loss(
            log_probs=log_probs,
            targets=tokens,
            input_lengths=enc_len,
            target_lengths=tokens_len,
        )

        loss_disen = (
            self.disen_mi_weight * aux.get("mi_upper", torch.zeros([], device=log_probs.device))
            + self.disen_lll_weight * aux.get("lll", torch.zeros([], device=log_probs.device))
        )

        loss = loss_ctc + loss_disen
        self.log_dict(
            {"train/loss_ctc": loss_ctc.detach(), "train/loss_disen": loss_disen.detach(), "train/loss_total": loss.detach()},
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        lr = 1e-4
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=1e-2)


class StudentASRWithDisentKD(BottleneckEncDecCTCModelBPE):
    def __init__(
        self,
        cfg,
        trainer=None,
        teacher: Optional[TeacherASRWithDisent] = None,
        use_logit_kd=True,
        use_layer_kd=False,
        kd_alpha=0.5,
        kd_temperature=1.0,
        use_gen_kd=True,
        gen_kd_type="flow",
        gen_kd_weight=1.0,
        **kwargs
    ):
        super().__init__(cfg=cfg, trainer=trainer, **kwargs)

        self.teacher = teacher
        self.use_logit_kd = bool(use_logit_kd)
        self.use_layer_kd = bool(use_layer_kd)
        self.kd_alpha = float(kd_alpha)
        self.kd_temperature = float(kd_temperature)

        self.use_gen_kd = bool(use_gen_kd)
        self.gen_kd_type = str(gen_kd_type)
        self.gen_kd_weight = float(gen_kd_weight)
        self.num_spk = int(getattr(cfg, "num_spk", 0))

        latent_dim = int(getattr(cfg, "latent_dim", 96))
        if self.use_gen_kd: # TODO: flow와 diff dual path로 사용할 수 있도록 수정
            if self.gen_kd_type == "mse":
                self.genkd = None
            else:
                diff_cfg = dict(
                    steps=5,
                    teacher_dim=176,
                    student_dim=88,
                    latent_dim=96,
                    stopgrad_teacher_latent=True,
                    recon_weight=1.0,
                    distill_weight=1.0,
                )

                flow_cfg = dict(
                    time_embed_dim=32,
                    hidden_dim=128,
                    training_sampling=8,
                    inference_sampling=8,
                    student_dim=88,
                    teacher_dim=176,
                    latent_dim=96,
                    shape_transform="linear",
                    loss="mse",
                    noise_schedule="rectified",
                    detach_teacher_feat=True,
                    loss_mode="sum_steps",
                )

                self.genkd_diff = DiffKDModule(diff_cfg)
                self.genkd_flow   = FlowMatchingModule(flow_cfg)

        else:
            self.genkd = None

    def training_step(self, batch, batch_idx):
        if self.teacher is None:
            raise RuntimeError("Student인데 teacher가 주입되지 않았습니다.")
        
        if len(batch) == 5:
            audio, audio_len, tokens, tokens_len, sample_ids = batch
            sample_ids = sample_ids.long()
            
            # sample_id -> speaker_id lookup
            speaker_ids = self.manifest_speakers.to(sample_ids.device)[sample_ids]
        elif len(batch) == 4:
            audio, audio_len, tokens, tokens_len = batch
            speaker_ids = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        if speaker_ids is not None:
            assert speaker_ids.min().item() >= -1
            assert speaker_ids.max().item() < self.num_spk


        # student forward
        s_logp, s_enc_len, student_text_emb, s_aux = self.forward_with_z(
            audio, audio_len, speaker_ids=speaker_ids, compute_loss=True
        )

        loss_ctc = self.loss(
            log_probs=s_logp,
            targets=tokens,
            input_lengths=s_enc_len,
            target_lengths=tokens_len,
        )

        loss_disen = (
            self.disen_mi_weight * s_aux.get("mi_upper", torch.zeros([], device=s_logp.device))
            + self.disen_lll_weight * s_aux.get("lll", torch.zeros([], device=s_logp.device))
        )

        loss = loss_ctc + loss_disen

        # teacher forward (frozen)
        self.teacher.eval()
        with torch.no_grad():
            t_logp, t_enc_len, teacher_text_emb, _ = self.teacher.forward_with_z(
                audio, audio_len, speaker_ids=speaker_ids, compute_loss=False
            )

        # genKD
        loss_genkd = torch.zeros([], device=s_logp.device)
        if self.use_gen_kd:
            # student_text_emb, teacher_text_emb: (B,T,D=latent_dim)
            flow_loss = self.genkd_flow(student_text_emb, teacher_text_emb, lengths=s_enc_len)
            diff_loss = self.genkd_diff(student_text_emb, teacher_text_emb, lengths=s_enc_len)
            loss_genkd = flow_loss + diff_loss
            
            loss = loss + self.gen_kd_weight * loss_genkd

        # logitKD
        loss_logitkd = torch.zeros([], device=s_logp.device)
        if self.use_logit_kd:
            loss_logitkd = kl_div_logits(s_logp, t_logp, T=self.kd_temperature)
            loss = loss + self.kd_alpha * loss_logitkd
        
        # layerKD
        loss_layerkd = torch.zeros([], device=s_logp.device)
        if self.use_layer_kd:
            # student_text_emb, teacher_text_emb: (B,T,D=latent_dim)
            loss_layerkd = F.mse_loss(student_text_emb, teacher_text_emb)
            loss = loss + self.kd_alpha * loss_layerkd

        self.log_dict(
            {
                "train/loss_ctc": loss_ctc.detach(),
                "train/loss_disen": loss_disen.detach(),
                "train/loss_genkd": loss_genkd.detach(),
                "train/loss_logitkd": loss_logitkd.detach(),
                "train/loss_layerkd": loss_layerkd.detach(),
                "train/loss_total": loss.detach(),
            },
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        lr = 1e-4
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=1e-2)


class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_: float):
    return GradReverseFn.apply(x, lambda_)


class SimpleClsHead(nn.Module):
    """time-avg pooling 후 MLP 분류"""
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        # x: (B,T,C)
        if lengths is None:
            pooled = x.mean(dim=1)
        else:
            B, T, C = x.shape
            mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]
            mask = mask.float().unsqueeze(-1)  # (B,T,1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (x * mask).sum(dim=1) / denom
        return self.net(pooled)


class ARClubGaussian(nn.Module):
    """
    Autoregressive vCLUB for I(U_{1:T}; V_static)
    q(u_{1:T}|v) = Π_t N(u_t | mu_t, diag(sigma^2_t))
    mu_t, sigma_t from causal GRU that sees u_{<t} with h0 from v.
    """
    def __init__(self, u_dim: int, v_dim: int, hidden: int = 256):
        super().__init__()
        self.u_dim = u_dim
        self.v_dim = v_dim
        self.hidden = hidden

        self.v_to_h0 = nn.Sequential(
            nn.Linear(v_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.gru = nn.GRU(input_size=u_dim, hidden_size=hidden, batch_first=True)
        self.mu = nn.Linear(hidden, u_dim)
        self.logvar = nn.Linear(hidden, u_dim)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        u: (B,T,Du), v: (B,Dv)
        return mu, logvar : (B,T,Du)
        """
        h0 = self.v_to_h0(v).unsqueeze(0)  # (1,B,H)
        out, _ = self.gru(u, h0)           # (B,T,H)
        mu = self.mu(out)
        logvar = self.logvar(out).clamp(min=-12.0, max=6.0)
        return mu, logvar

    def log_prob(self, u: torch.Tensor, v: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        log q(u|v) summed over time+dim, then batch-mean
        """
        mu, logvar = self(u, v)
        # log N(u|mu, diag(var))
        # = -0.5 * [ (u-mu)^2/var + log var + log(2pi) ]
        var = logvar.exp()
        lp = -0.5 * (((u - mu) ** 2) / var + logvar + math.log(2.0 * math.pi))
        # sum over dim
        lp = lp.sum(dim=-1)  # (B,T)

        if lengths is not None:
            B, T = lp.shape
            mask = (torch.arange(T, device=lp.device)[None, :] < lengths[:, None]).float()
            lp = (lp * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            lp = lp.mean(dim=1)
        return lp.mean()  # scalar


@dataclass
class DisenLossOut:
    z_for_decoder: torch.Tensor         # (B,T,in_dim)
    z_text_latent: torch.Tensor         # (B,T,latent_dim)  <-- KD는 보통 이걸로
    aux: Dict[str, torch.Tensor]        # losses/embeddings


class DisentanglementBottleneck(nn.Module):
    """
    Encoder output (B,T,in_dim) -> latent -> disent -> latent(z_text) -> out_proj -> (B,T,in_dim)

    - train/infer 공용: forward가 항상 지나감
    - loss는 학습 시에만 speaker_ids 등 들어오면 계산
    """
    def __init__(
        self,
        in_dim: int,
        latent_dim: int = 96,
        num_spk: int = 0,
        mi_pairs: str = "ts,tp,ps",
        club_hidden: int = 256,
        use_txt_spk_probe: bool = True,
        txt_probe_hidden: int = 256,
        txt_probe_dropout: float = 0.2,
        mi_clamp_min0: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.num_spk = int(num_spk)
        self.mi_pairs = [p.strip() for p in mi_pairs.split(",") if p.strip()]
        self.mi_clamp_min0 = bool(mi_clamp_min0)

        # in/out mapping (teacher/student 각각 dim이 다르니 인스턴스를 분리)
        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, in_dim)

        # latent processing
        self.txt_ln = nn.LayerNorm(latent_dim)
        self.txt_ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

        # static embeddings (speaker/prosody) by pooling + MLP
        self.spk_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )
        self.pros_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

        # CLUB estimators (dynamic u = z_text, static v = spk/pros)
        self.club_ts = ARClubGaussian(u_dim=latent_dim, v_dim=latent_dim, hidden=club_hidden)
        self.club_tp = ARClubGaussian(u_dim=latent_dim, v_dim=latent_dim, hidden=club_hidden)
        self.club_ps = ARClubGaussian(u_dim=latent_dim, v_dim=latent_dim, hidden=club_hidden)

        # optional adversarial speaker probe on z_text (GRL)
        self.use_txt_spk_probe = bool(use_txt_spk_probe) and (self.num_spk > 0)
        if self.use_txt_spk_probe:
            self.txt_spk_probe = SimpleClsHead(
                in_dim=latent_dim,
                num_classes=self.num_spk,
                hidden=txt_probe_hidden,
                dropout=txt_probe_dropout,
            )
        else:
            self.txt_spk_probe = None

    def _masked_mean(self, x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B,T,C) -> (B,C)
        if lengths is None:
            return x.mean(dim=1)
        B, T, C = x.shape
        mask = (torch.arange(T, device=x.device)[None, :] < lengths[:, None]).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (x * mask).sum(dim=1) / denom

    def _vclub_mi_upper_and_lll(
        self,
        club: ARClubGaussian,
        u: torch.Tensor,
        v: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MI upper:  E_{p(u,v)} log q(u|v) - E_{p(u)p(v)} log q(u|v)
        LLL:      -E_{p(u,v)} log q(u|v)
        """
        joint = club.log_prob(u, v, lengths=lengths)
        # shuffle v across batch for product-of-marginals
        perm = torch.randperm(v.size(0), device=v.device)
        v_shuf = v[perm]
        marg = club.log_prob(u, v_shuf, lengths=lengths)
        mi_upper = joint - marg
        lll = -joint
        if self.mi_clamp_min0:
            mi_upper = mi_upper.clamp_min(0.0)
        return mi_upper, lll

    def forward(
        self,
        enc_out: torch.Tensor,                 # (B,T,in_dim)
        enc_len: Optional[torch.Tensor] = None, # (B,)
        speaker_ids: Optional[torch.Tensor] = None,
        compute_loss: bool = False,
        grl_lambda: float = 1.0,
    ) -> DisenLossOut:
        # map to latent
        z = self.in_proj(enc_out)  # (B,T,latent)
        z = self.txt_ln(z)
        z = z + self.txt_ff(z)

        # build static spk/pros embeddings from pooled latent
        pooled = self._masked_mean(z, enc_len)   # (B,latent)
        spk_emb = self.spk_mlp(pooled)           # (B,latent)
        pros_emb = self.pros_mlp(pooled)         # (B,latent)

        # for now: z_text_latent = z (you can insert routing/subtraction if 기존 구현이 그랬다면 여기)
        z_text_latent = z

        aux: Dict[str, torch.Tensor] = {
            "spk_emb": spk_emb,
            "pros_emb": pros_emb,
        }

        if compute_loss:
            mi_total = torch.zeros([], device=z.device)
            lll_total = torch.zeros([], device=z.device)

            # pairs: ts,tp,ps where t=dynamic(z_text), s=spk_emb, p=pros_emb
            if "ts" in self.mi_pairs:
                mi_ts, lll_ts = self._vclub_mi_upper_and_lll(self.club_ts, z_text_latent, spk_emb, enc_len)
                mi_total = mi_total + mi_ts
                lll_total = lll_total + lll_ts
                aux["mi_ts"] = mi_ts
                aux["lll_ts"] = lll_ts

            if "tp" in self.mi_pairs:
                mi_tp, lll_tp = self._vclub_mi_upper_and_lll(self.club_tp, z_text_latent, pros_emb, enc_len)
                mi_total = mi_total + mi_tp
                lll_total = lll_total + lll_tp
                aux["mi_tp"] = mi_tp
                aux["lll_tp"] = lll_tp

            if "ps" in self.mi_pairs:
                # ps는 static-static이라 ARClubGaussian이랑 구조가 100% 맞진 않음
                # 그래도 "ps"를 유지하고 싶다면, p를 time 축으로 broadcast해서 억지로 맞추는 방식(간단)로 처리
                # (기존 구현이 다르면 여기만 네 구현으로 교체)
                B, T, _ = z_text_latent.shape
                p_dyn = pros_emb[:, None, :].expand(B, T, self.latent_dim)
                mi_ps, lll_ps = self._vclub_mi_upper_and_lll(self.club_ps, p_dyn, spk_emb, enc_len)
                mi_total = mi_total + mi_ps
                lll_total = lll_total + lll_ps
                aux["mi_ps"] = mi_ps
                aux["lll_ps"] = lll_ps

            aux["mi_upper"] = mi_total
            aux["lll"] = lll_total

            # optional adversarial speaker probe on z_text
            if self.use_txt_spk_probe and (speaker_ids is not None):
                speaker_ids = speaker_ids.long()
                valid = (speaker_ids >= 0) & (speaker_ids < self.num_spk)
                if valid.any():
                    spk_logits = self.txt_spk_probe(z_text_latent[valid].detach(), enc_len[valid] if enc_len is not None else None)
                    aux["txt_spk_probe_ce"] = F.cross_entropy(spk_logits, speaker_ids[valid]) # TODO: 내가 logging하고 싶은거는 ce가 아니라 실제 acc임.
                else:
                    aux["txt_spk_probe_ce"] = torch.zeros([], device=z.device)
            else:
                aux["txt_spk_probe_ce"] = torch.zeros([], device=z.device)

            

        # map back to encoder dim for decoder
        z_for_decoder = self.out_proj(z_text_latent)  # (B,T,in_dim)

        return DisenLossOut(
            z_for_decoder=z_for_decoder,
            z_text_latent=z_text_latent,
            aux=aux,
        )
