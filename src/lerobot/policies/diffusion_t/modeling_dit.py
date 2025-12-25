# Copyright 2025 Nur Muhammad Mahi Shafiullah,
# and The HuggingFace Inc. team. All rights reserved.
#
# Adapted for Standard Diffusion (DDPM) using the clean DiT architecture.

import copy
import math
from collections import deque

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.utils.constants import (
    OBS_ENV_STATE,
    OBS_STATE,
    ACTION,
    OBS_IMAGES,
)
from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot.policies.diffusion_t.configuration_dit import DitConfig 
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)


def _get_activation_fn(activation: str):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


# --- Architecture Classes (Kept from the "Clean" DiT implementation) ---

class _TimeNetwork(nn.Module):
    def __init__(self, frequency_embedding_dim, hidden_dim, learnable_w=False, max_period=10000):
        # NOTE: Standard diffusion uses max_period=10000 typically
        assert frequency_embedding_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(frequency_embedding_dim // 2)
        super().__init__()

        w = np.log(max_period) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        # t: [B] or [B, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.float() * self.w[None]
        t = torch.cat((torch.cos(t), torch.sin(t)), dim=1)
        return self.out_net(t)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * (1 + self.scale(c)[None]) + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(
        self, d_model=256, nhead=6, dim_feedforward=2048, dropout=0.0, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        self.mlp = nn.Sequential(
            self.linear1,
            self.activation,
            self.dropout2,
            self.linear2,
            self.dropout3,
        )

        # AdaLN Modulation
        self.attn_modulate = _ShiftScaleMod(d_model)
        self.attn_gate = _ZeroScaleMod(d_model)
        self.mlp_modulate = _ShiftScaleMod(d_model)
        self.mlp_gate = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond, need_weights=False):
        # cond + t injection
        cond = cond + t

        x2 = self.attn_modulate(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=need_weights)
        x = x + self.attn_gate(self.dropout1(x2), cond)

        x3 = self.mlp_modulate(self.norm2(x), cond)
        x3 = self.mlp(x3)
        x3 = self.mlp_gate(x3, cond)
        return x + x3

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for s in (self.attn_modulate, self.attn_gate, self.mlp_modulate, self.mlp_gate):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, cond):
        cond = cond + t
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerDecoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )
        self.reset_parameters()

    def forward(self, src, t, cond, **kwargs):
        x = src
        for layer in self.layers:
            x = layer(x, t, cond, **kwargs)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


class _DiTNet(nn.Module):
    """
    Standard DiT Network.
    Accepts noisy_actions, timestep, and global_cond.
    Predicts Noise (epsilon).
    """
    def __init__(
        self,
        ac_dim,
        ac_chunk,
        cond_dim,
        time_dim=256,
        hidden_dim=256,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
    ):
        super().__init__()
        self.ac_dim, self.ac_chunk = ac_dim, ac_chunk

        # Positional embedding for the action sequence
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # Time and Input embeddings
        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # Backbone
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # Output Head
        self.final_layer = _FinalLayer(hidden_dim, ac_dim)

        print(
            "DiT Param Count: {:.2f}M".format(
                sum(p.numel() for p in self.parameters()) / 1e6
            )
        )

    def forward(self, noisy_actions, time, global_cond):
        """
        noisy_actions: (B, T, action_dim)
        time: (B,) tensor of integers or floats
        global_cond: (B, cond_dim)
        """
        c = self.cond_proj(global_cond)
        time_enc = self.time_net(time)

        ac_tokens = self.ac_proj(noisy_actions)  # [B, T, hidden]
        ac_tokens = ac_tokens.transpose(0, 1)    # [T, B, hidden]

        # Add positional embedding
        dec_in = ac_tokens + self.dec_pos[: ac_tokens.size(0)]

        # Run Transformer
        dec_out = self.decoder(dec_in, time_enc, c)

        # Final prediction
        output = self.final_layer(dec_out, time_enc, c) # [T, B, dim]
        return output.transpose(0, 1)  # [B, T, dim]


# --- Scheduler (Self-Contained DDPM) ---

class DDPMScheduler(nn.Module):
    """
    A lightweight DDPM Scheduler to avoid heavy dependencies (like diffusers).
    """
    def __init__(self, num_train_timesteps=100, beta_start=0.0001, beta_end=0.02, clip_sample=True, clip_sample_range=1.0):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # Beta Schedule (Linear)
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Helper function to register buffer (automatically moves to device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def add_noise(self, original_samples, noise, timesteps):
        """
        Forward process: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
        """
        # Make sure shapes match for broadcasting: (B, 1, 1)
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample, generator=None):
        """
        Reverse process: Predict x_{t-1} given x_t and predicted noise (model_output).
        Using DDPM formulation.
        """
        t = timestep
        # 1. Compute predicted x_0 (optional but good for clipping)
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_prod_t = self.alphas_cumprod[t]
        
        # 2. Compute Mean
        # coeff1 = 1 / sqrt(alpha_t)
        # coeff2 = beta_t / sqrt(1 - alpha_prod_t)
        # pred_prev_sample = coeff1 * (sample - coeff2 * model_output)
        
        # Alternative stable formulation (predict x0 then reconstruct):
        # x_0 = (x_t - sqrt(1-alpha_bar)*eps) / sqrt(alpha_bar)
        pred_original_sample = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -self.clip_sample_range, self.clip_sample_range)

        # Re-compute mean using posterior formula
        # mean = (sqrt(alpha_bar_prev) * beta / (1-alpha_bar)) * x0 + (sqrt(alpha) * (1-alpha_bar_prev) / (1-alpha_bar)) * xt
        # Simplified for DDPM:
        pred_prev_sample = (1 / torch.sqrt(alpha_t)) * (sample - (beta_t / torch.sqrt(1 - alpha_prod_t)) * model_output)

        # 3. Add Variance (if t > 0)
        variance = 0
        if t > 0:
            noise = torch.randn(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
            # variance = beta_t (standard) or beta_tilde (posterior)
            # using fixed large variance (beta_t)
            variance = torch.sqrt(beta_t) * noise
        
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample


class DitPolicy(PreTrainedPolicy):
    """
    Standard Diffusion Policy with DiT Architecture.
    """
    config_class = DitConfig
    name = "DiT"

    def __init__(self, config: DitConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        
        # Initialize Model
        self.model = DiTModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        """Clear observation and action queues."""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        必须实现的方法：基于当前队列中的观测历史，预测一段动作序列。
        """
        # 将队列中的数据堆叠起来 (B, T, ...)
        batch_data = {
            k: torch.stack(list(self._queues[k]), dim=1) 
            for k in batch 
            if k in self._queues
        }
        # 调用模型生成动作
        return self.model.generate_actions(batch_data)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        执行单个动作。如果动作队列为空，则调用 predict_action_chunk 补充新动作。
        """
        if ACTION in batch: 
            batch.pop(ACTION)
        
        # 处理图像特征堆叠
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        
        # 更新观测队列
        self._queues = populate_queues(self._queues, batch)

        # 如果动作队列用完了，预测新的一块
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            # 转置并存入队列: (B, T, D) -> (T, B, D) 以便逐个弹出
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.model.compute_loss(batch)
        return loss, None


class DiTModel(nn.Module):
    def __init__(self, config: DitConfig):
        super().__init__()
        self.config = config

        # --- 1. Encoder Setup (Same as before) ---
        global_cond_dim = (
            self.config.robot_state_feature.shape[0] if self.config.use_proprioceptive else 0
        )
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]
        
        self.global_cond_dim = global_cond_dim

        # --- 2. Diffusion Network ---
        self.net = _DiTNet(
            ac_dim=config.action_feature.shape[0],
            ac_chunk=config.horizon,
            cond_dim=self.global_cond_dim * config.n_obs_steps,
            time_dim=config.frequency_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            dim_feedforward=config.dim_feedforward,
            nhead=config.num_heads,
            activation=config.activation,
        )

        # --- 3. Noise Scheduler (Standard DDPM) ---
        # NOTE: Using num_inference_steps as train steps for simplicity in config mapping, 
        # or use a default 100 for training. 
        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.num_inference_steps or 100,
            beta_start=1e-4, 
            beta_end=0.02,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
        )

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Same encoding logic
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [encoder(images) for encoder, images in zip(self.rgb_encoder, images_per_camera)]
                )
                img_features = einops.rearrange(img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps)
            else:
                img_features = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
                img_features = einops.rearrange(img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps)
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Standard DDPM Loss: MSE(epsilon_pred, epsilon_target)
        """
        # 1. Validation and Encoding
        n_obs_steps = batch["observation.state"].shape[1]
        trajectory = batch["action"] # (B, T, D)
        batch_size = trajectory.shape[0]
        global_cond = self._prepare_global_conditioning(batch)

        # 2. Sample Noise
        noise = torch.randn_like(trajectory)
        
        # 3. Sample Timesteps (Uniform discrete)
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (batch_size,), device=trajectory.device
        ).long()

        # 4. Add Noise (Forward Process)
        noisy_trajectory = self.scheduler.add_noise(trajectory, noise, timesteps)

        # 5. Predict Noise
        noise_pred = self.net(noisy_trajectory, timesteps, global_cond)

        # 6. MSE Loss
        loss = F.mse_loss(noise_pred, noise, reduction="none")

        # 7. Masking (for padding)
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError("action_is_pad is required.")
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()

    @torch.no_grad()
    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        DDPM Sampling Loop
        """
        batch_size = batch["observation.state"].shape[0]
        device = batch["observation.state"].device
        
        # 1. Encode
        global_cond = self._prepare_global_conditioning(batch)

        # 2. Start from pure noise
        current_sample = torch.randn(
            batch_size, self.config.horizon, self.config.action_feature.shape[0],
            device=device
        )

        # 3. Denoising Loop (T -> 0)
        for t in reversed(range(self.scheduler.num_train_timesteps)):
            # Create batch of timesteps
            ts = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            model_output = self.net(current_sample, ts, global_cond)
            
            # Step inverse dynamics
            current_sample = self.scheduler.step(model_output, t, current_sample)

        # 4. Extract action chunk
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return current_sample[:, start:end]