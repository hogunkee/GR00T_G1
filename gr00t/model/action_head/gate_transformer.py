
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x:    (B, T, C)
    mask: (B, T) where 1=valid, 0=pad
    """
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom    # (B,C)


class GateHorizonTransformer(nn.Module):
    """
    Predict horizon-wise gates g_{1:H} using only observation+language+state tokens.
    Handles mismatched input dims via projections.

    Inputs:
      - vl_embs:        (B, S, D_vl)   e.g., 2048
      - vl_attn_mask:   (B, S)
      - state_features: (B, T_s, D_s)  e.g., 1536

    Output:
      - g: (B, H, 1)
    """
    def __init__(
        self,
        gate_dim: int,
        horizon: int,
        vl_dim: int,
        state_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_vl_token: bool = True,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        self.gate_dim = gate_dim
        self.horizon = horizon
        self.use_vl_token = use_vl_token

        # Project inputs to gate_dim
        self.vl_proj = nn.Linear(vl_dim, gate_dim)
        self.state_proj = nn.Linear(state_dim, gate_dim)

        # H learned query tokens
        self.gate_queries = nn.Parameter(torch.zeros(1, horizon, gate_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, gate_dim)) if use_pos_embed else None

        ff_dim = int(gate_dim * mlp_ratio)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=gate_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(gate_dim, 1)

        nn.init.normal_(self.gate_queries, mean=0.0, std=0.02)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, vl_embs, vl_attn_mask, state_features):
        B = state_features.shape[0]

        # Project to common dim
        state = self.state_proj(state_features)  # (B, T_s, gate_dim)

        ctx = [state]
        if self.use_vl_token:
            vl_pool = masked_mean(vl_embs, vl_attn_mask)          # (B, D_vl)
            vl_pool = self.vl_proj(vl_pool).unsqueeze(1)          # (B, 1, gate_dim)
            ctx.append(vl_pool)

        ctx = torch.cat(ctx, dim=1)  # (B, T_s(+1), gate_dim)

        q = self.gate_queries.expand(B, -1, -1)  # (B, H, gate_dim)
        if self.pos_embed is not None:
            q = q + self.pos_embed.expand(B, -1, -1)

        x = torch.cat([q, ctx], dim=1)  # (B, H + T_s(+1), gate_dim)
        x = self.encoder(x)

        q_out = x[:, : self.horizon, :]          # (B, H, gate_dim)
        g = self.head(q_out)      # (B, H, 1)
        # g = torch.tanh(self.head(q_out))      # (B, H, 1)
        return g


class UncertaintyHorizonTransformer(nn.Module):
    """
    Predict per-timestep log-variance for (upper, lower) actions using only (VL + state).
    Output logvars are used for heteroscedastic weighting: exp(-s)*L + s.
    """
    def __init__(
        self,
        gate_dim: int,
        max_horizon: int,
        vl_dim: int = 2048,
        state_dim: int = 1536,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_vl_token: bool = True,
        use_pos_embed: bool = True,
        out_channels: int = 2,   # (logvar_u, logvar_l)
    ):
        super().__init__()
        self.gate_dim = gate_dim
        self.max_horizon = max_horizon
        self.use_vl_token = use_vl_token

        # project to common dim
        self.vl_proj = nn.Linear(vl_dim, gate_dim)
        self.state_proj = nn.Linear(state_dim, gate_dim)

        # learned queries: one per timestep up to max_horizon
        self.queries = nn.Parameter(torch.zeros(1, max_horizon, gate_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_horizon, gate_dim)) if use_pos_embed else None

        ff_dim = int(gate_dim * mlp_ratio)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=gate_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Linear(gate_dim, out_channels)

        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, vl_embs, vl_attn_mask, state_features):
        """
        horizon: H (<= max_horizon)
        returns:
          logvar_u: (B,H,1)
          logvar_l: (B,H,1)
        """
        B = state_features.shape[0]

        # context tokens
        state = self.state_proj(state_features)  # (B, Ts, D)
        ctx = [state]
        if self.use_vl_token:
            vl_pool = masked_mean(vl_embs, vl_attn_mask)         # (B, vl_dim)
            vl_tok = self.vl_proj(vl_pool).unsqueeze(1)          # (B,1,D)
            ctx.append(vl_tok)
        ctx = torch.cat(ctx, dim=1)  # (B, Ts(+1), D)

        # queries
        q = self.queries.expand(B, -1, -1)  # (B,H,D)
        if self.pos_embed is not None:
            q = q + self.pos_embed.expand(B, -1, -1)

        x = torch.cat([q, ctx], dim=1)  # (B, H + ctx_len, D)
        x = self.encoder(x)
        q_out = x[:, :self.horizon, :]             # (B,H,D)

        out = self.head(q_out)          # (B,H,2)
        logvar_u = out[..., 0:1]
        logvar_l = out[..., 1:2]
        return logvar_u, logvar_l