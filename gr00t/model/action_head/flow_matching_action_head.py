# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer
from .gate_transformer import GateHorizonTransformer, UncertaintyHorizonTransformer

def linear_ramp(step, start, end):
    if step <= start: return 0.0
    if step >= end:   return 1.0
    return (step - start) / float(end - start)

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


def masked_mse_per_timestep(pred, target, mask, eps=1e-6):
    """
    pred, target: (B, H, D)
    mask:         (B, H, D)  (0/1)
    return:       (B, H, 1)  per-timestep masked MSE
    """
    se = (pred - target).pow(2) * mask
    denom = mask.sum(dim=-1, keepdim=True).clamp(min=eps)
    return se.sum(dim=-1, keepdim=True) / denom


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    phase_weighted_loss: bool = field(default=False, metadata={"help": "Whether to use phase weighted loss."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # assert NotImplementedError("Flow matching action head loss needs to be checked. !!Negative Phase Actions!!")
        if self.config.phase_weighted_loss:
            action_mask = action_input.action_mask
            gt_phase_t = (actions[:,:, 31:32] + 1)/2
            upper_loss = F.mse_loss(pred_actions[:,:, :28], velocity[:,:, :28], reduction="none")
            loco_loss = F.mse_loss(pred_actions[:,:, 28:31], velocity[:,:, 28:31], reduction="none")
            phase_loss = F.mse_loss(pred_actions[:,:, 31:32], velocity[:,:, 31:32], reduction="none")
            loss = (gt_phase_t * upper_loss * action_mask[:,:,:28]).sum() \
                     + ((1 - gt_phase_t) * loco_loss * action_mask[:,:,28:31]).sum() \
                     + (phase_loss * action_mask[:,:,31:32]).sum()
            loss = loss / action_mask.sum()
            # print('-'*30)
            # print('upper:', upper_loss.mean().item(), 'loco:', loco_loss.mean().item(), 'phase:', phase_loss.mean().item())
            # print('phase:', gt_phase_t.mean().item())
            # loss = (gt_phase_t * upper_loss + (1 - gt_phase_t) * loco_loss + phase_loss) * action_mask
            # loss = loss.sum() / action_mask.sum()
        else:
            # Slice out only the action portion of pred and target.
            action_mask = action_input.action_mask
            loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
            loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class GateFlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
        loss_type='energy-expert'
    ):
        super().__init__()
        self.loss_type=loss_type #'energy-expert' # 'guide' #'energy-expert'
        
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.phase_model = GateHorizonTransformer(gate_dim=512, vl_dim=2048, state_dim=1536, horizon=config.action_horizon, num_layers=2, num_heads=4)
        #self.phase_model = UncertaintyHorizonTransformer(gate_dim=512, max_horizon=config.action_horizon, vl_dim=2048, state_dim=1536, num_layers=2, num_heads=4)
        
        # self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.manip_action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=28, #self.action_dim,
        )
        self.loco_action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments, 
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=3, #self.action_dim,
        )
        self.phase_action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=1, #self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )
        
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

        self.alpha_manip = 500 #0.5
        self.alpha_loco = 2500 #3.0
        self.tau_manip = 0.02 #1.5
        self.tau_loco = 0.001 #0.5
        self.offset_loco = torch.tensor([-1, 0.6828, -1], device='cuda:0')
        self.offset_manip = torch.tensor([ 0.2492, -0.1749, -0.0321, -0.7335, -0.0948,  0.2868,  0.1586,  0.4287,
                                0.2091,  0.2716, -0.6546, -0.4897,  0.8802, -0.4716,  0.9989,  0.9987,
                                0.9989,  0.9987, -0.9988, -0.1189,  0.0000, -0.9979, -0.9978, -0.9988,
                                -0.9985,  0.9996,  0.1189,  0.0000], device='cuda:0')
        if self.loss_type=='guide':
            self.phase_weight = 5.0
        else:
            self.phase_weight = 2.0

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.manip_action_decoder.requires_grad_(False)
            self.loco_action_decoder.requires_grad_(False)
            self.phase_action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
            self.phase_model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.manip_action_decoder.eval()
                self.loco_action_decoder.eval()
                self.phase_action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
                self.phase_model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred_manip = self.manip_action_decoder(model_output, embodiment_id)[:, -actions.shape[1] :]
        pred_loco = self.loco_action_decoder(model_output, embodiment_id)[:, -actions.shape[1] :]
        
        # sa_gate = torch.cat((state_features, future_tokens), dim=1)
        logit_phase = self.phase_model(vl_embs, vl_attn_mask, state_features)
        pred_phase = torch.sigmoid(logit_phase) * 2 - 1
        # pred_phase: -1 ~ 1

        # assert NotImplementedError("Flow matching action head loss needs to be checked. !!Negative Phase Actions!!")
        action_mask = action_input.action_mask
        # gt_phase_t = (actions[:,:, 31:32] + 1)/2
        manip_loss = F.mse_loss(pred_manip, velocity[:,:, :28], reduction="none")
        loco_loss = F.mse_loss(pred_loco, velocity[:,:, 28:31], reduction="none")

        gt_g = (actions[:,:, 31:32] + 1)/2
        phase_loss = F.binary_cross_entropy_with_logits(logit_phase, gt_g)
        # phase_loss = F.mse_loss(pred_phase, actions[:,:, 31:32], reduction="none")
        
        pred_phase_denorm = (pred_phase + 1.25) / 2.5
        # pred_phase_denorm = (pred_phase + 1) / 2
        if self.loss_type=='gt':
            loss = (pred_phase_denorm * manip_loss * action_mask[:,:,:28]).sum() / action_mask[:,:,:28].sum() \
                    + ((1 - pred_phase_denorm) * loco_loss * action_mask[:,:,28:31]).sum() / action_mask[:,:,28:31].sum() \
                    + (phase_loss * action_mask[:,:,31:32]).sum() / action_mask[:,:,31:32].sum()
        elif self.loss_type=='simple':
            loss = (pred_phase_denorm * manip_loss * action_mask[:,:,:28]).sum() / action_mask[:,:,:28].sum() \
                    + ((1 - pred_phase_denorm) * loco_loss * action_mask[:,:,28:31]).sum() / action_mask[:,:,28:31].sum() 
        elif self.loss_type=='guide':
            # pred_phase_denorm = self.phase_weight/5.0 * 0.5 + (1 - self.phase_weight/5.0) * pred_phase_denorm
            loss = (pred_phase_denorm * manip_loss * action_mask[:,:,:28]).sum() / action_mask[:,:,:28].sum() \
                    + ((1 - pred_phase_denorm) * loco_loss * action_mask[:,:,28:31]).sum() / action_mask[:,:,28:31].sum() \
                    + self.phase_weight * (phase_loss * action_mask[:,:,31:32]).sum() / action_mask[:,:,31:32].sum()
            self.phase_weight *= 0.9999
            self.phase_weight = max(self.phase_weight, 0.3)
        elif self.loss_type=='energy':
            energy_manip = ((actions[:,:,:28] - self.offset_manip) * action_mask[:,:,:28]).pow(2).sum(2, keepdim=True)
            mask_manip = torch.sigmoid(self.alpha_manip * (energy_manip - self.tau_manip)).detach()
            energy_loco = ((actions[:,:,28:31] - self.offset_loco) * action_mask[:,:,28:31]).pow(2).sum(2, keepdim=True)
            mask_loco = torch.sigmoid(self.alpha_loco * (energy_loco - self.tau_loco)).detach()
            loss = (pred_phase_denorm * mask_manip * manip_loss * action_mask[:,:,:28]).sum() / action_mask[:,:,:28].sum() \
                    + ((1 - pred_phase_denorm) * mask_loco * loco_loss * action_mask[:,:,28:31]).sum() / action_mask[:,:,28:31].sum()
        elif self.loss_type=='energy-guide':
            energy_manip = ((actions[:,:,:28] - self.offset_manip) * action_mask[:,:,:28]).pow(2).sum(2, keepdim=True)
            mask_manip = torch.sigmoid(self.alpha_manip * (energy_manip - self.tau_manip)).detach()
            energy_loco = ((actions[:,:,28:31] - self.offset_loco) * action_mask[:,:,28:31]).pow(2).sum(2, keepdim=True)
            mask_loco = torch.sigmoid(self.alpha_loco * (energy_loco - self.tau_loco)).detach()
            loss = (pred_phase_denorm * mask_manip * manip_loss * action_mask[:,:,:28]).sum() / action_mask[:,:,:28].sum() \
                    + ((1 - pred_phase_denorm) * mask_loco * loco_loss * action_mask[:,:,28:31]).sum() / action_mask[:,:,28:31].sum() \
                    + self.phase_weight * (phase_loss * action_mask[:,:,31:32]).sum() / action_mask[:,:,31:32].sum()
            self.phase_weight *= 0.9999
            self.phase_weight = max(self.phase_weight, 0.3)
        elif self.loss_type=='energy-expert':
            u_slice = slice(0, 28)
            l_slice = slice(28, 31)
            v_u = velocity[:, :, u_slice]          # (B,H,28)
            v_l = velocity[:, :, l_slice]          # (B,H,3)
            # Masks per part
            m_u = action_mask[:, :, u_slice]        # (B,H,28)
            m_l = action_mask[:, :, l_slice]        # (B,H,3)
            # ---------------------------------------------------------
            # Baseline velocity under "inactive clean action x0=0"
            # v = x0 - eps  ->  v_base = 0 - eps = -eps
            # 여기서 eps == noise
            # ---------------------------------------------------------
            vbase_u = -noise[:, :, u_slice]         # (B,H,28)
            vbase_l = -noise[:, :, l_slice]         # (B,H,3)
            # ---------------------------------------------------------
            # Per-timestep losses
            #   L_u_exp: expert가 upper를 설명
            #   L_l_exp: expert가 lower를 설명
            #   L_u_base: upper가 inactive(0)라고 가정했을 때 baseline 설명
            #   L_l_base: lower가 inactive(0)라고 가정했을 때 baseline 설명
            # 모두 (B,H,1)로 만들어 비교 가능하게 만듦
            # ---------------------------------------------------------
            L_u_exp  = masked_mse_per_timestep(pred_manip, v_u, m_u)         # (B,H,1)
            L_l_exp  = masked_mse_per_timestep(pred_loco, v_l, m_l)         # (B,H,1)
            L_u_base = masked_mse_per_timestep(vbase_u, v_u, m_u)     # (B,H,1)
            L_l_base = masked_mse_per_timestep(vbase_l, v_l, m_l)     # (B,H,1)
            # ---------------------------------------------------------
            # Mode energies:
            # z=0 (loco):  lower=expert, upper=baseline
            # z=1 (manip): upper=expert, lower=baseline
            # ---------------------------------------------------------
            E0 = L_l_exp + L_u_base   # (B,H,1)
            E1 = L_u_exp + L_l_base   # (B,H,1)
            logit_phase = logit_phase.float()
            E0 = E0.float()
            E1 = E1.float()
            # ---------------------------------------------------------
            # Gate probability g in (0,1)
            # pred_phase: (-1~1)라고 했으니 보통 (pred_phase+1)/2가 자연스러움.
            # 기존 denorm(1.25/2.5) 쓰고 싶으면 그걸로 교체해도 됨.
            # ---------------------------------------------------------
            # g = (pred_phase + 1.0) / 2.0
            # g = g.clamp(1e-4, 1.0 - 1e-4)  # log 안정성
            # ---------------------------------------------------------
            # Mixture likelihood loss (log-sum-exp)
            # L = -log( (1-g) exp(-E0) + g exp(-E1) )
            # ---------------------------------------------------------
            log_q1 = F.logsigmoid(logit_phase)      # log(sigmoid(logit))
            log_q0 = F.logsigmoid(-logit_phase)     # log(1-sigmoid(logit)) 안정형
            # log_q1 = torch.log(g)            # log p(z=1|obs)
            # log_q0 = torch.log(1.0 - g)      # log p(z=0|obs)
            logp0 = log_q0 - E0              # (B,H,1)
            logp1 = log_q1 - E1              # (B,H,1)
            # logsumexp over z in {0,1}
            log_mix = torch.logsumexp(torch.cat([logp0, logp1], dim=-1), dim=-1)  # (B,H)
            # timestep mask: upper/lower 둘 다 마스크가 0인 (패딩) timestep 제외
            timestep_valid = ((m_u.sum(dim=-1) + m_l.sum(dim=-1)) > 0).to(log_mix.dtype)  # (B,H)
            timestep_valid = timestep_valid.float() 
            loss = -(log_mix * timestep_valid).sum() / timestep_valid.sum().clamp(min=1.0) \
                    + self.phase_weight * (phase_loss * action_mask[:,:,31:32]).sum() / action_mask[:,:,31:32].sum().clamp(min=1e-6)
            self.phase_weight *= 0.9999
            self.phase_weight = max(self.phase_weight, 0.3)
            # (선택) phase smoothness regularizer (horizon gate가 너무 출렁이지 않게)
            # if g.shape[1] > 1:
            #     loss = loss + lambda_smooth * (g[:, 1:] - g[:, :-1]).abs().mean()
        
        # print('-'*30)
        # print('upper:', manip_loss.mean().item(), 'loco:', loco_loss.mean().item(), 'phase:', phase_loss.mean().item())
        
        output_dict = {
            "loss": loss,
            "manip-loss": manip_loss.mean().item(),
            "loco-loss": loco_loss.mean().item(),
            "phase-loss": phase_loss.mean().item(),
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim), #31
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            # pred = self.action_decoder(model_output, embodiment_id)
            pred_manip = self.manip_action_decoder(model_output, embodiment_id)
            pred_loco = self.loco_action_decoder(model_output, embodiment_id)
            empty_phase = torch.zeros([pred_loco.shape[0], pred_loco.shape[1], 1]).to(pred_loco.device)
            
            pred_velocity = torch.cat([
                pred_manip[:, -self.action_horizon :], 
                pred_loco[:, -self.action_horizon :],
                empty_phase[:, -self.action_horizon :]
                ], -1)

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        
        # sa_gate = torch.cat((state_features, future_tokens), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask
        # pred_phase = self.phase_model(vl_embs, vl_attn_mask, state_features)
        logit_phase = self.phase_model(vl_embs, vl_attn_mask, state_features)
        pred_phase = torch.sigmoid(logit_phase) * 2 - 1
        actions = torch.cat([actions[:,:,:-1], pred_phase], -1)
        
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class MixtureFlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig
    ):
        super().__init__()
        self.version = 3
        self.step = 0

        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.phase_model = GateHorizonTransformer(gate_dim=512, vl_dim=2048, state_dim=1536, horizon=config.action_horizon, num_layers=2, num_heads=4)
        #self.phase_model = UncertaintyHorizonTransformer(gate_dim=512, max_horizon=config.action_horizon, vl_dim=2048, state_dim=1536, num_layers=2, num_heads=4)
        
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.manip_action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim-1,
        )
        self.loco_action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments, 
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim-1,
        )
        self.phase_action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=1,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )
        
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

        # tau scheduling: 0.02 -> 0.001 in 5k steps
        self.tau = 0.02 #2.0
        self.tau_end = 0.001
        self.tau_decay = 0.999 #0.99995
        
        # tau scheduling: 2.0 -> 0.3 in 50k steps
        # self.tau = 2.0
        # self.tau_end = 0.3
        # self.tau_decay = 0.9999 #0.99995

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.manip_action_decoder.requires_grad_(False)
            self.loco_action_decoder.requires_grad_(False)
            self.phase_action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
            self.phase_model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.manip_action_decoder.eval()
                self.loco_action_decoder.eval()
                self.phase_action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
                self.phase_model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        noisy_trajectory[:,:,31:32] = 0
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred_manip = self.manip_action_decoder(model_output, embodiment_id)[:, -actions.shape[1] :]
        pred_loco = self.loco_action_decoder(model_output, embodiment_id)[:, -actions.shape[1] :]
        
        # sa_gate = torch.cat((state_features, future_tokens), dim=1)
        logit_phase = self.phase_model(vl_embs, vl_attn_mask, state_features)
        pred_phase = torch.sigmoid(logit_phase)

        # assert NotImplementedError("Flow matching action head loss needs to be checked. !!Negative Phase Actions!!")
        action_mask = action_input.action_mask
        # gt_phase_t = (actions[:,:, 31:32] + 1)/2
        manip_loss = F.mse_loss(pred_manip, velocity[:,:, :31], reduction="none")
        loco_loss = F.mse_loss(pred_loco, velocity[:,:, :31], reduction="none")

        # mask and norm
        mask_a = action_mask[:, :, :31].float()  # [B,T,31]
        denom_a = mask_a.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B,T,1]
        timestep_valid = (denom_a > 1e-6).float()

        # gt phase loss
        gt_g = (actions[:,:, 31:32] + 1)/2
        gt_phase_loss_t = F.binary_cross_entropy_with_logits(logit_phase, gt_g)
        gt_phase_loss = (gt_phase_loss_t * timestep_valid).sum() / timestep_valid.sum().clamp(min=1.0)

        # mixed noise
        pred_mix = pred_phase * pred_manip + (1.0 - pred_phase) * pred_loco
        mix_loss = F.mse_loss(pred_mix, velocity[:,:, :31], reduction="none")
        l_mix_t = (mix_loss * mask_a).sum(dim=-1, keepdim=True) / denom_a
        l_mix = (l_mix_t * timestep_valid).sum() / timestep_valid.sum().clamp(min=1.0)

        ## responsibility ##
        l_man_t  = (manip_loss * mask_a).sum(dim=-1, keepdim=True) / denom_a
        l_loco_t = (loco_loss  * mask_a).sum(dim=-1, keepdim=True) / denom_a

        self.tau = max(self.tau * self.tau_decay, self.tau_end)
        logits = torch.cat([-l_man_t / self.tau, -l_loco_t / self.tau], dim=-1)  # [B,T,2]
        r = torch.softmax(logits, dim=-1)                               # [B,T,2]
        r_man = r[..., 0:1]                                             # [B,T,1]
        r_man_detached = r_man.detach()
        
        # responsibility-weighted expert loss
        experts_loss_t = (r_man_detached * l_man_t + (1.0 - r_man_detached) * l_loco_t)  # [B,T,1]
        experts_loss = (experts_loss_t * timestep_valid).sum() / timestep_valid.sum().clamp(min=1.0)
        # energy-based phase-weighted expert loss
        experts_gt_loss_t = (gt_g * l_man_t + (1.0 - gt_g) * l_loco_t)  # [B,T,1]
        experts_gt_loss = (experts_gt_loss_t * timestep_valid).sum() / timestep_valid.sum().clamp(min=1.0)

        # BCE target is r_man_detached (soft target)
        phase_loss_t = F.binary_cross_entropy_with_logits(logit_phase, r_man_detached, reduction="none")
        phase_loss = (phase_loss_t * timestep_valid).sum() / timestep_valid.sum().clamp(min=1.0)

        # regularization
        # binary prior: pushes p to {0,1}
        bin_loss = (pred_phase * (1.0 - pred_phase) * timestep_valid).sum() / timestep_valid.sum().clamp(min=1.0)
        # load balancing: keep average gate near rho (set rho based on expected ratio, or 0.5)
        rho = 0.5
        p_mean = (pred_phase * timestep_valid).sum() / timestep_valid.sum().clamp(min=1.0)
        bal_loss = (p_mean - rho).pow(2)
        # temporal smoothness (TV): fewer switches
        tv_loss = (pred_phase[:, 1:] - pred_phase[:, :-1]).abs()
        tv_loss = (tv_loss * timestep_valid[:, 1:]).sum() / timestep_valid[:, 1:].sum().clamp(min=1.0)

        lam_phase = 1.0
        lam_bin = 0.1
        lam_bal = 0.1
        lam_tv = 0.05
        lam_gt = 1.0
        lam_mix = 1.0
        if self.version==1:
            loss = experts_loss + lam_phase * phase_loss \
                            + lam_bin * bin_loss \
                            + lam_bal * bal_loss \
                            + lam_tv * tv_loss \
                            + lam_gt * gt_phase_loss
        elif self.version==2:
            loss = experts_loss + lam_phase * phase_loss \
                            + lam_bin * bin_loss \
                            + lam_bal * bal_loss \
                            + lam_tv * tv_loss \
                            + lam_gt * gt_phase_loss \
                            + lam_mix * l_mix
        # Curriculum learning for model version 3
        elif self.version==3:
            lam_exp = linear_ramp(self.step, start=3000, end=10000)
            lam_phase = 0.3 * linear_ramp(self.step, start=5000, end=10000)
            # if self.step < 3000: lam_exp = 0.0
            # else: lam_exp = 1.0
            # if self.step < 5000: lam_phase = 0.0
            # else: lam_phase = 1.0
            loss = lam_exp * experts_loss + (1-lam_exp) * experts_gt_loss \
                            + lam_phase * phase_loss \
                            + lam_bin * bin_loss \
                            + lam_bal * bal_loss \
                            + lam_tv * tv_loss \
                            + lam_gt * gt_phase_loss \
                            + lam_mix * l_mix
            self.step += 1
        
        output_dict = {
            "loss": loss,
            "manip-loss": (l_man_t).mean().item(),
            "loco-loss": (l_loco_t).mean().item(),
            "res-manip-loss": (r_man_detached * l_man_t).mean().item(),
            "res-loco-loss": ((1.0 - r_man_detached) * l_loco_t).mean().item(),
            "phase-loss": phase_loss.mean().item(),
            "gt-phase-loss": gt_phase_loss.mean().item(),
            "bin-loss": bin_loss.mean().item(),
            "bal-loss": bal_loss.mean().item(),
            "tv-loss": tv_loss.mean().item(),
            "mix-loss": l_mix.mean().item(),
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        vl_attn_mask = backbone_output.backbone_attention_mask
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        actions[:,:,31:32] = 0

        # phase: compute once (since your phase_model doesn't use actions)
        with torch.no_grad():
            logit_phase = self.phase_model(vl_embs, vl_attn_mask, state_features)
            pred_phase = torch.sigmoid(logit_phase)  # shape: [B,1] or [B,T,1]
        
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            # pred = self.action_decoder(model_output, embodiment_id)
            pred_manip = self.manip_action_decoder(model_output, embodiment_id)
            pred_loco = self.loco_action_decoder(model_output, embodiment_id)

            # take last horizon chunk to match actions
            v_man = pred_manip[:, -self.action_horizon:]  # [B,H,31]
            v_loco = pred_loco[:, -self.action_horizon:]  # [B,H,31]

            # mixture velocity
            p = pred_phase[:, -self.action_horizon:]  # [B,H,1]
            pred_velocity = p * v_man + (1.0 - p) * v_loco  # [B,H,31]

            # Update actions using euler integration.
            actions[:,:,:31] = actions[:,:,:31] + dt * pred_velocity
        
        if True:
            min_val = 0.2
            max_val = 0.8
            pred_phase = (pred_phase - min_val) / (max_val - min_val)  # 0 ~ 1
        
        pred_phase = (pred_phase * 2 - 1).clamp(-1, 1) # scale to -1 ~ 1
        actions = torch.cat([actions[:,:,:-1], pred_phase], -1)
        
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
