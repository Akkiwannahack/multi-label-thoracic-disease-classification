"""
vision_mamba_block.py — Phase 3b: Vision Mamba (SSM) Block

Implements bidirectional selective SSM blocks for global context modeling
on the 49-token sequence produced from the ResNet50 7×7 feature map.

Two backends:
  1. mamba-ssm (compiled CUDA kernel) — fast but requires CUDA build.
  2. Pure-PyTorch fallback — numerically equivalent, works everywhere.

Both expose the same `VisionMambaBlock` interface.

Residual fusion revisions:
  - legacy_sum: original behaviour — y = fwd + bwd  (each includes its own
    residual, so the skip connection is applied twice: y ≈ 2x + updates).
  - mean_v2: corrected — y = x + 0.5 * (fwd_update + bwd_update).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------------------------
# Try importing the compiled mamba-ssm package; availability checked later
# ---------------------------------------------------------------------------
MAMBA_SSM_AVAILABLE = False
try:
    from mamba_ssm import Mamba as MambaSSM
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    pass

# Valid residual fusion modes
VALID_FUSION_MODES = ("legacy_sum", "mean_v2")

# Valid backend identifiers
VALID_BACKENDS = ("custom_pytorch", "mamba_ssm")


# ========================== Pure-PyTorch SSM ================================

class SelectiveSSM(nn.Module):
    """Pure-PyTorch selective state-space model (simplified Mamba core).

    Implements the discretized recurrence:
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t

    where A_bar, B_bar are input-dependent (selective) via small linear
    projections. This is slower than the fused CUDA kernel but numerically
    equivalent in spirit and perfectly fine at sequence length 49.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input-dependent parameter projections
        self.proj_delta = nn.Linear(d_model, d_model, bias=True)
        self.proj_B = nn.Linear(d_model, d_state, bias=False)
        self.proj_C = nn.Linear(d_model, d_state, bias=False)

        # Learnable A — parameterized as log for numerical stability
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1))
        )

        # Output projection
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        B_size, L, D = x.shape

        # Input-dependent discretization step Δ (softplus to keep positive)
        delta = F.softplus(self.proj_delta(x))  # (B, L, D)

        # Input-dependent B and C
        B_input = self.proj_B(x)   # (B, L, N)   N = d_state
        C_input = self.proj_C(x)   # (B, L, N)

        # Continuous A → discretized A_bar
        A = -torch.exp(self.A_log)  # (D, N) — negative for stability

        # Run the selective scan (sequential recurrence)
        y = torch.zeros_like(x)  # (B, L, D)
        h = torch.zeros(B_size, D, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(L):
            dt = delta[:, t, :]               # (B, D)
            b_t = B_input[:, t, :]             # (B, N)
            c_t = C_input[:, t, :]             # (B, N)
            x_t = x[:, t, :]                   # (B, D)

            # Discretize: A_bar = exp(Δ * A)
            A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (B, D, N)

            # B_bar = Δ * B  (simplified ZOH discretization)
            B_bar = dt.unsqueeze(-1) * b_t.unsqueeze(1)           # (B, D, N)

            # State update
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)             # (B, D, N)

            # Output
            y_t = (h * c_t.unsqueeze(1)).sum(dim=-1)              # (B, D)

            # Skip connection (D parameter)
            y[:, t, :] = y_t + self.D * x_t

        return y


class PurePyTorchMambaBlock(nn.Module):
    """Single Mamba block with pre-norm, SiLU gating, and selective SSM."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D) — residual connection included
        """
        residual = x
        x = self.norm(x)

        # Split into main path and gate
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 1D conv (operates along sequence dim)
        x_main = rearrange(x_main, "b l d -> b d l")
        x_main = self.conv1d(x_main)
        x_main = rearrange(x_main, "b d l -> b l d")
        x_main = F.silu(x_main)

        # Selective SSM
        x_main = self.ssm(x_main)

        # Gating
        x_main = x_main * F.silu(z)

        # Project back
        out = self.out_proj(x_main)

        return out + residual


# ========================== CUDA-kernel Mamba block ==========================

class CUDAMambaBlock(nn.Module):
    """Mamba block using the compiled mamba-ssm package (faster, requires CUDA build)."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaSSM(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + residual


# ========================= Bidirectional wrapper =============================

class BidirectionalMambaBlock(nn.Module):
    """Bidirectional Mamba: forward scan + backward scan.

    Images have no natural left-to-right causality, so bidirectional
    scanning captures context from all spatial directions.

    Residual fusion modes:
        legacy_sum — y = fwd + bwd  (original, residual applied twice)
        mean_v2    — y = x + 0.5 * ((fwd - x) + (bwd - x))
                       = x + 0.5 * (fwd + bwd - 2x)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        use_cuda: bool | None = None,
        residual_fusion: str = "mean_v2",
    ):
        super().__init__()

        if residual_fusion not in VALID_FUSION_MODES:
            raise ValueError(
                f"Unknown residual_fusion '{residual_fusion}'. "
                f"Valid: {VALID_FUSION_MODES}"
            )
        self.residual_fusion = residual_fusion

        if use_cuda is None:
            use_cuda = MAMBA_SSM_AVAILABLE

        BlockCls = CUDAMambaBlock if use_cuda else PurePyTorchMambaBlock

        self.forward_block = BlockCls(d_model, d_state, expand)
        self.backward_block = BlockCls(d_model, d_state, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        fwd = self.forward_block(x)
        bwd = self.backward_block(x.flip(dims=[1])).flip(dims=[1])

        if self.residual_fusion == "legacy_sum":
            # Original behaviour: y = fwd + bwd (residual applied twice)
            return fwd + bwd
        else:
            # mean_v2: y = x + 0.5 * (fwd_update + bwd_update)
            # where fwd_update = fwd - x, bwd_update = bwd - x
            return x + 0.5 * (fwd + bwd - 2 * x)


# ======================== Vision Mamba Encoder ===============================

class VisionMambaEncoder(nn.Module):
    """Stack of bidirectional Mamba blocks with input projection and
    learnable 2D positional embeddings.

    Converts a (B, C, H, W) feature map into a global context vector (B, d_model).
    """

    def __init__(
        self,
        in_channels: int = 2048,
        spatial_size: int = 7,
        d_model: int = 256,
        d_state: int = 16,
        num_blocks: int = 4,
        use_cuda: bool | None = None,
        residual_fusion: str = "mean_v2",
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.num_tokens = spatial_size * spatial_size  # 49 for 7×7
        self.d_model = d_model
        self.residual_fusion = residual_fusion

        # Project from ResNet channel dim to SSM working dim
        self.input_proj = nn.Linear(in_channels, d_model)

        # Learnable 2D positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, d_model) * 0.02
        )

        # Stack of bidirectional Mamba blocks
        self.blocks = nn.ModuleList([
            BidirectionalMambaBlock(
                d_model, d_state,
                use_cuda=use_cuda,
                residual_fusion=residual_fusion,
            )
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: (B, C, H, W) from ResNet50 layer4

        Returns:
            Global context vector (B, d_model) — mean-pooled over tokens
        """
        B, C, H, W = feature_map.shape

        # --- Fix I: Shape guard ---
        num_tokens = H * W
        if num_tokens != self.num_tokens:
            raise ValueError(
                f"Spatial size mismatch: positional embedding expects "
                f"{self.spatial_size}×{self.spatial_size} = {self.num_tokens} tokens, "
                f"but feature map has {H}×{W} = {num_tokens} tokens. "
                f"This model only supports image_size=224 (producing 7×7 features)."
            )

        # Flatten spatial dims → token sequence
        tokens = rearrange(feature_map, "b c h w -> b (h w) c")  # (B, 49, 2048)

        # Project to d_model
        tokens = self.input_proj(tokens)  # (B, 49, 256)

        # Add positional embedding
        tokens = tokens + self.pos_embed

        # Pass through Mamba blocks
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Mean-pool over tokens → global context
        context = tokens.mean(dim=1)  # (B, d_model)

        return context


def resolve_backend(requested: str) -> bool:
    """Resolve a backend string to a use_cuda boolean.

    Args:
        requested: one of VALID_BACKENDS

    Returns:
        True for mamba_ssm, False for custom_pytorch

    Raises:
        ValueError for unknown or unavailable backends.
    """
    if requested not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown mamba_backend '{requested}'. Valid: {VALID_BACKENDS}"
        )
    if requested == "mamba_ssm":
        if not MAMBA_SSM_AVAILABLE:
            raise RuntimeError(
                "mamba_backend='mamba_ssm' requested but mamba_ssm package "
                "is not installed. Install it or use 'custom_pytorch'."
            )
        return True
    return False
