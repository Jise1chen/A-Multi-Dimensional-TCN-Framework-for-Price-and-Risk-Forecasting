"""Causal residual TCN, independent attention heads, and learned task weights."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalConv(nn.Conv1d):
    def forward(self, x):
        return super().forward(F.pad(x, ((self.kernel_size[0] - 1) * self.dilation[0], 0)))


class ResidualBlock(nn.Module):
    def __init__(self, inputs, hidden, kernel, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv(inputs, hidden, kernel, dilation=dilation)
        self.conv2 = CausalConv(hidden, hidden, kernel, dilation=dilation)
        self.skip = nn.Conv1d(inputs, hidden, 1) if inputs != hidden else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        mask = valid[:, None, :].to(x.dtype)
        h = self.dropout(F.relu(self.conv1(x))) * mask
        h = self.dropout(F.relu(self.conv2(h))) * mask
        return F.relu(h + self.skip(x)) * mask


class CausalAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)

    def forward(self, h, valid, return_weights=False):
        q, k, v = self.q(h), self.k(h), self.v(h)
        length = h.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=h.device).tril()
        allowed = causal[None] & valid[:, None, :]
        # Padded queries can otherwise have all -inf logits. Give only those
        # queries a temporary self-key; zero them immediately after attention.
        eye = torch.eye(length, dtype=torch.bool, device=h.device)[None]
        allowed = allowed | (~valid[:, :, None] & eye)
        weights = None
        if return_weights:
            scores = (q @ k.transpose(-1, -2)) / h.shape[-1] ** 0.5
            weights = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
            weights = weights * valid[:, :, None]
            output = weights @ v
        else:
            # SDPA may use a fused kernel; quadratic arithmetic remains.
            output = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None],
                                                     attn_mask=allowed[:, None], dropout_p=0.0)[:, 0]
            output = output * valid[:, :, None]
        pooled = output.sum(dim=1) / valid.sum(dim=1, keepdim=True)
        return pooled, weights


class FINMIND(nn.Module):
    def __init__(self, inputs, hidden=32, levels=6, kernel=3, dropout=0.1, variant="decoupled"):
        super().__init__()
        if variant not in {"decoupled", "shared", "none", "price_only"}:
            raise ValueError("Unknown model variant")
        self.variant = variant
        self.receptive_field = 1 + 2 * (kernel - 1) * (2 ** levels - 1)
        self.blocks = nn.ModuleList([ResidualBlock(inputs if i == 0 else hidden, hidden,
                                                 kernel, 2 ** i, dropout) for i in range(levels)])
        self.price_attention = CausalAttention(hidden) if variant != "none" else None
        self.risk_attention = CausalAttention(hidden) if variant == "decoupled" else None
        def head():
            if variant == "none":
                return nn.Linear(hidden, 2)
            return nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.price_head = head()
        self.risk_head = head() if variant != "price_only" else None

    def forward(self, x, valid=None, return_attention=False):
        if valid is None:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if valid.shape != x.shape[:2] or not valid.any(dim=1).all():
            raise ValueError("Every sequence must contain an observed timestep")
        h = (x * valid[:, :, None]).transpose(1, 2)
        for block in self.blocks:
            h = block(h, valid)
        h = h.transpose(1, 2)
        maps = {}
        if self.price_attention is None:
            price = risk = (h * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True)
        else:
            price, maps["price"] = self.price_attention(h, valid, return_attention)
            risk = price
            if self.risk_attention is not None:
                risk, maps["risk"] = self.risk_attention(h, valid, return_attention)
        pred = self.price_head(price)
        if self.risk_head is not None:
            pred = torch.cat([pred, self.risk_head(risk)], dim=-1)
        return (pred, maps) if return_attention else pred


class UncertaintyLoss(nn.Module):
    """s_m = log(sigma_m^2): sum_m 0.5 * (exp(-s_m)*MSE_m + s_m)."""
    def __init__(self, outputs=4):
        super().__init__()
        self.log_variance = nn.Parameter(torch.zeros(outputs))

    def forward(self, pred, target):
        mse = (pred - target[:, :pred.shape[1]]).square().mean(0)
        return (0.5 * (torch.exp(-self.log_variance) * mse + self.log_variance)).sum()
