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
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt_main, opt_club = self.optimizers()  # configure_optimizers에서 반환한 순서

        # ----- batch unpack -----
        if len(batch) == 5:
            audio, audio_len, tokens, tokens_len, sample_ids = batch
            sample_ids = sample_ids.long()
            speaker_ids = self.manifest_speakers.to(sample_ids.device)[sample_ids]
        elif len(batch) == 4:
            audio, audio_len, tokens, tokens_len = batch
            speaker_ids = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        # =========================================================
        # (1) CLUB 업데이트: lll_club만 최소화 (u,v는 detach되어 club만 학습)
        # =========================================================
        toggle_club_grad(self.disent, True)   # club 파라미터 학습 ON

        _, _, _, aux_c = self.forward_with_z(
            input_signal=audio,
            input_signal_length=audio_len,
            speaker_ids=speaker_ids,
            compute_loss=True,
            grl_lambda=1.0,
        )
        club_loss = self.disen_lll_weight * aux_c.get("lll_club", torch.zeros([], device=audio.device))

        opt_club.zero_grad(set_to_none=True)
        self.manual_backward(club_loss)
        opt_club.step()

        # =========================================================
        # (2) MAIN 업데이트: CTC + MI(encoder용 raw) (club은 freeze)
        # =========================================================
        toggle_club_grad(self.disent, False)  # club 파라미터 학습 OFF (하지만 u로의 grad는 흐름)

        log_probs, enc_len, _, aux_m = self.forward_with_z(
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

        mi_loss = aux_m.get("mi_upper_enc_raw", torch.zeros([], device=log_probs.device))
        loss_disen = self.disen_mi_weight * mi_loss  # <= main step에는 MI만

        loss = loss_ctc + loss_disen

        opt_main.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        opt_main.step()

        # ----- logging -----
        self.log_dict(
            {
                "train/loss_ctc": loss_ctc.detach(),
                "train/club_loss": club_loss.detach(),
                "train/mi_loss": (self.disen_mi_weight * mi_loss).detach(),
                "train/loss_total": loss.detach(),
                "train/mi_upper_log": aux_m.get("mi_upper_log", torch.zeros([], device=log_probs.device)).detach(),
            },
            prog_bar=True,
        )

        return loss


    # def configure_optimizers(self):
    #     lr = 1e-4
    #     params = [p for p in self.parameters() if p.requires_grad]
    #     return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98), weight_decay=1e-2)
    def configure_optimizers(self):
        lr_main = 1e-4
        lr_club = 1e-4

        club_params = []
        for m in [self.disent.club_ts, self.disent.club_tp, self.disent.club_ps]:
            club_params += list(m.parameters())

        # club 파라미터를 제외한 나머지
        club_param_ids = set(id(p) for p in club_params)
        main_params = [p for p in self.parameters() if p.requires_grad and id(p) not in club_param_ids]

        opt_main = torch.optim.AdamW(main_params, lr=lr_main, betas=(0.9, 0.98), weight_decay=1e-2)
        opt_club = torch.optim.AdamW(club_params, lr=lr_club, betas=(0.9, 0.98), weight_decay=0.0)

        return [opt_main, opt_club]


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
    def __init__(self, u_dim: int, v_dim: int, hidden: int = 256, shift_right: bool = True):
        super().__init__()
        self.u_dim = u_dim
        self.v_dim = v_dim
        self.hidden = hidden
        self.shift_right = shift_right

        self.v_to_h0 = nn.Sequential(
            nn.Linear(v_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.gru = nn.GRU(input_size=u_dim, hidden_size=hidden, batch_first=True)
        self.mu = nn.Linear(hidden, u_dim)
        self.logvar = nn.Linear(hidden, u_dim)

    def forward(self, u: torch.Tensor, v: torch.Tensor):
        # u: (B,T,Du), v: (B,Dv)
        if self.shift_right:
            u_in = torch.zeros_like(u)
            u_in[:, 1:, :] = u[:, :-1, :]   # u_<t만 보게 함
        else:
            u_in = u

        h0 = self.v_to_h0(v).unsqueeze(0)   # (1,B,H)
        out, _ = self.gru(u_in, h0)         # (B,T,H)
        mu = self.mu(out)
        logvar = self.logvar(out).clamp(min=-12.0, max=6.0)
        return mu, logvar

    def log_prob(self, u: torch.Tensor, v: torch.Tensor, lengths=None, time_reduce: str = "mean"):
        mu, logvar = self(u, v)
        var = logvar.exp()
        lp = -0.5 * (((u - mu) ** 2) / var + logvar + math.log(2.0 * math.pi))
        lp = lp.sum(dim=-1)  # (B,T)

        if lengths is not None:
            B, T = lp.shape
            mask = (torch.arange(T, device=lp.device)[None, :] < lengths[:, None]).float()
            if time_reduce == "sum":
                lp = (lp * mask).sum(dim=1)
            else:  # mean
                lp = (lp * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            lp = lp.sum(dim=1) if time_reduce == "sum" else lp.mean(dim=1)

        return lp.mean()  # batch mean


@dataclass
class DisenLossOut:
    z_for_decoder: torch.Tensor         # (B,T,in_dim)
    z_text_latent: torch.Tensor         # (B,T,latent_dim)  <-- KD는 보통 이걸로
    aux: Dict[str, torch.Tensor]        # losses/embeddings


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

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
        num_neg: int = 4,              # <= 추가: negative K개
        time_reduce: str = "mean",     # "sum"으로 바꾸면 논문과 더 정합
        detach_for_lll: bool = True,   # CLUB 업데이트용일 때 True 권장
    ):
        # joint: E_{p(u,v)} log q(u|v)
        u_joint = u.detach() if detach_for_lll else u
        v_joint = v.detach() if detach_for_lll else v
        joint = club.log_prob(u_joint, v_joint, lengths=lengths, time_reduce=time_reduce)

        # marg: E_{p(u)p(v)} log q(u|v)  (K-shuffle 평균)
        B = v.size(0)
        marg_acc = 0.0
        for _ in range(num_neg):
            perm = torch.randperm(B, device=v.device)
            v_shuf = v[perm]
            u_m = u.detach() if detach_for_lll else u
            v_m = v_shuf.detach() if detach_for_lll else v_shuf
            marg_acc = marg_acc + club.log_prob(u_m, v_m, lengths=lengths, time_reduce=time_reduce)
        marg = marg_acc / float(num_neg)

        mi_upper_raw = joint - marg
        lll = -joint

        # clamp는 로깅용으로만 두는 게 안전
        mi_upper_log = mi_upper_raw.clamp_min(0.0) if self.mi_clamp_min0 else mi_upper_raw

        return mi_upper_raw, mi_upper_log, lll

    def forward(
        self,
        enc_out: torch.Tensor,                 # (B,T,in_dim)
        enc_len: Optional[torch.Tensor] = None, # (B,)
        speaker_ids: Optional[torch.Tensor] = None,
        compute_loss=False, grl_lambda=1.0, vclub_num_neg: int = 4, vclub_time_reduce: str = "mean"
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
            mi_enc_total = torch.zeros([], device=z.device)   # encoder 업데이트용 (raw)
            mi_log_total = torch.zeros([], device=z.device)   # 로깅용 (clamp)
            lll_club_total = torch.zeros([], device=z.device) # club 업데이트용

            def add_pair(name, club, u_dyn, v_stat):
                nonlocal mi_enc_total, mi_log_total, lll_club_total

                # (A) encoder용 MI: detach_for_lll=False (즉 u,v에 gradient 허용)
                mi_raw, mi_log, _ = self._vclub_mi_upper_and_lll(
                    club, u_dyn, v_stat, enc_len,
                    num_neg=vclub_num_neg,
                    time_reduce=vclub_time_reduce,
                    detach_for_lll=False,
                )
                mi_enc_total = mi_enc_total + mi_raw
                mi_log_total = mi_log_total + mi_log
                aux[f"mi_{name}_raw"] = mi_raw
                aux[f"mi_{name}_log"] = mi_log

                # (B) club용 LLL: u,v detach해서 club만 학습하게
                _, _, lll = self._vclub_mi_upper_and_lll(
                    club, u_dyn, v_stat, enc_len,
                    num_neg=1,  # LLL은 negative 필요 없음
                    time_reduce=vclub_time_reduce,
                    detach_for_lll=True,
                )
                lll_club_total = lll_club_total + lll
                aux[f"lll_{name}"] = lll

            if "ts" in self.mi_pairs:
                add_pair("ts", self.club_ts, z_text_latent, spk_emb)
            if "tp" in self.mi_pairs:
                add_pair("tp", self.club_tp, z_text_latent, pros_emb)
            if "ps" in self.mi_pairs:
                B, T, _ = z_text_latent.shape
                p_dyn = pros_emb[:, None, :].expand(B, T, self.latent_dim)
                add_pair("ps", self.club_ps, p_dyn, spk_emb)

            aux["mi_upper_enc_raw"] = mi_enc_total     # 학습용 (clamp X)
            aux["mi_upper_log"] = mi_log_total         # 로깅용 (clamp O 가능)
            aux["lll_club"] = lll_club_total           # club 학습용

            # speaker probe: 지금 grl_lambda 인자가 죽어있음. 두 가지 모드 중 선택.
            if self.use_txt_spk_probe and (speaker_ids is not None):
                speaker_ids = speaker_ids.long()
                valid = (speaker_ids >= 0) & (speaker_ids < self.num_spk)
                if valid.any():
                    # 1) "순수 probe(측정)"만: detach 유지
                    probe_logits = self.txt_spk_probe(z_text_latent[valid].detach(),
                                                    enc_len[valid] if enc_len is not None else None)
                    aux["txt_spk_probe_ce"] = F.cross_entropy(probe_logits, speaker_ids[valid])
                    aux["txt_spk_probe_acc"] = (probe_logits.argmax(-1) == speaker_ids[valid]).float().mean()

                    # 2) "adversarial로 speaker 제거"까지 하고 싶으면: GRL 사용 (detach 제거)
                    # adv_feat = GradReverseFn.apply(z_text_latent[valid], grl_lambda)
                    # adv_logits = self.txt_spk_probe(adv_feat, enc_len[valid] if enc_len is not None else None)
                    # aux["txt_spk_adv_ce"] = F.cross_entropy(adv_logits, speaker_ids[valid])
                    # aux["txt_spk_adv_acc"] = (adv_logits.argmax(-1) == speaker_ids[valid]).float().mean()
                else:
                    aux["txt_spk_probe_ce"] = torch.zeros([], device=z.device)
                    aux["txt_spk_probe_acc"] = torch.zeros([], device=z.device)

            

        # map back to encoder dim for decoder
        z_for_decoder = self.out_proj(z_text_latent)  # (B,T,in_dim)

        return DisenLossOut(
            z_for_decoder=z_for_decoder,
            z_text_latent=z_text_latent,
            aux=aux,
        )

def toggle_club_grad(disen: DisentanglementBottleneck, flag: bool):
    set_requires_grad(disen.club_ts, flag)
    set_requires_grad(disen.club_tp, flag)
    set_requires_grad(disen.club_ps, flag)