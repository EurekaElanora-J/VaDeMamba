import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearSSMFunction(torch.autograd.Function):
    """
    Custom autograd Function for selective SSM linear recurrence:
      h_t = A_t * h_{t-1} + B_u_t
      y_t = sum_n (h_t * C_t)
    
    Eliminates the autograd graph entirely during recurrence:
    - Forward pass: executes sequential recurrence in-place without retaining intermediate operator graph.
    - Backward pass: executes exact analytical adjoint recurrence in reverse.
    Memory usage: strictly O(L*D*N) (~200 MB), zero graph bloat, zero OOM.
    """
    @staticmethod
    def forward(ctx, A, B_u, C):
        """
        A:   (Batch, L, D, N)  -- discretized transition matrix deltaA
        B_u: (Batch, L, D, N)  -- input injection deltaB * u
        C:   (Batch, L, N)     -- output projection C
        Returns:
        y:   (Batch, L, D)
        """
        batch, L, D, N = A.shape
        h = torch.empty(batch, L, D, N, device=A.device, dtype=A.dtype)
        
        curr_h = torch.zeros(batch, D, N, device=A.device, dtype=A.dtype)
        for t in range(L):
            curr_h = A[:, t] * curr_h + B_u[:, t]
            h[:, t] = curr_h
            
        y = torch.sum(h * C.unsqueeze(2), dim=-1) # (batch, L, D)
        ctx.save_for_backward(A, B_u, C, h)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        """
        grad_y: (batch, L, D)
        Exact adjoint analytical backward pass:
        dL/dh_t = grad_y_t * C_t + dL/dh_{t+1} * A_{t+1}
        """
        A, B_u, C, h = ctx.saved_tensors
        batch, L, D, N = A.shape
        
        grad_A = torch.empty_like(A)
        grad_B_u = torch.empty_like(B_u)
        grad_C = torch.zeros_like(C)
        
        curr_grad_h = torch.zeros(batch, D, N, device=A.device, dtype=A.dtype)
        C_unsqueezed = C.unsqueeze(2)
        grad_y_unsqueezed = grad_y.unsqueeze(-1)
        
        for t in range(L - 1, -1, -1):
            curr_grad_h = grad_y_unsqueezed[:, t] * C_unsqueezed[:, t] + curr_grad_h
            grad_B_u[:, t] = curr_grad_h
            grad_C[:, t] = torch.sum(grad_y_unsqueezed[:, t] * h[:, t], dim=1)
            
            if t > 0:
                grad_A[:, t] = curr_grad_h * h[:, t - 1]
                curr_grad_h = curr_grad_h * A[:, t]
            else:
                grad_A[:, t] = torch.zeros(batch, D, N, device=A.device, dtype=A.dtype)
                
        return grad_A, grad_B_u, grad_C


def chunked_associative_scan(a, b, chunk_size=64):
    """
    Two-level work-efficient parallel associative scan:
    h_t = a_t * h_{t-1} + b_t
    Executes parallel log2(chunk_size) + log2(num_chunks) tensor operations.
    Fully eliminates Python-level recurrence loops and per-step kernel launch overhead.
    """
    B, L, D, N = a.shape
    if L % chunk_size != 0:
        chunk_size = 64 if L % 64 == 0 else (32 if L % 32 == 0 else 16)
    num_chunks = L // chunk_size
    
    a_chunks = a.view(B * num_chunks, chunk_size, D, N)
    b_chunks = b.view(B * num_chunks, chunk_size, D, N)
    
    # 1. Intra-chunk parallel scan
    step = 1
    curr_a = a_chunks
    curr_b = b_chunks
    while step < chunk_size:
        a_left = curr_a[:, :-step]
        b_left = curr_b[:, :-step]
        a_right = curr_a[:, step:]
        b_right = curr_b[:, step:]
        
        b_right_new = a_right * b_left + b_right
        a_right_new = a_right * a_left
        
        curr_a = torch.cat([curr_a[:, :step], a_right_new], dim=1)
        curr_b = torch.cat([curr_b[:, :step], b_right_new], dim=1)
        step *= 2
        
    chunk_totals_a = curr_a[:, -1].view(B, num_chunks, D, N)
    chunk_totals_b = curr_b[:, -1].view(B, num_chunks, D, N)
    
    # 2. Inter-chunk parallel scan across the num_chunks
    step = 1
    inter_a = chunk_totals_a
    inter_b = chunk_totals_b
    while step < num_chunks:
        a_left = inter_a[:, :-step]
        b_left = inter_b[:, :-step]
        a_right = inter_a[:, step:]
        b_right = inter_b[:, step:]
        
        b_right_new = a_right * b_left + b_right
        a_right_new = a_right * a_left
        
        inter_a = torch.cat([inter_a[:, :step], a_right_new], dim=1)
        inter_b = torch.cat([inter_b[:, :step], b_right_new], dim=1)
        step *= 2
        
    prefix_from_prev_chunk = torch.cat(
        [torch.zeros(B, 1, D, N, device=b.device, dtype=b.dtype), inter_b[:, :-1]], dim=1
    ).view(B * num_chunks, 1, D, N)
    
    out = curr_b + curr_a * prefix_from_prev_chunk
    return out.view(B, L, D, N)


class SelectiveSSM1D(nn.Module):
    """
    Selective State Space Model (SSM) using LinearSSMFunction for training
    and work-efficient chunked_associative_scan for inference.
    Discretizes continuous parameters via ZOH:
      deltaA = exp(delta * A)
      deltaB = delta * B
      h_t = deltaA_t * h_{t-1} + deltaB_t * u_t
      y_t = C_t * h_t + D * u_t
    """
    def __init__(self, d_model, d_state=16, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A)) # (D, N)
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

    def forward(self, u):
        B, L, D = u.shape
        x_proj = self.x_proj(u)
        delta_rank, B_proj, C_proj = torch.split(
            x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        delta = F.softplus(self.dt_proj(delta_rank)) # (B, L, D)
        A = -torch.exp(self.A_log.float())          # (D, N)
        
        deltaA = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, D, self.d_state)) # (B, L, D, N)
        deltaB = delta.unsqueeze(-1) * B_proj.unsqueeze(2)                      # (B, L, D, N)
        dB_u = deltaB * u.unsqueeze(-1)                                         # (B, L, D, N)
        
        if self.training and torch.is_grad_enabled():
            # Exact analytical autograd function for training
            y = LinearSSMFunction.apply(deltaA, dB_u, C_proj)
        else:
            # Parallel chunked associative scan for fast inference
            h = chunked_associative_scan(deltaA, dB_u, chunk_size=64)
            y = torch.sum(h * C_proj.unsqueeze(2), dim=-1)
            
        return y + u * self.D.view(1, 1, D)


class BidirectionalMambaBlock3D(nn.Module):
    """
    3D Bidirectional Mamba Block:
      - InstanceNorm3d
      - Input projection with gating
      - 3D Depthwise Conv (3x3x3)
      - Forward & Backward 3D raster sequence scan
      - Output projection + Residual
    """
    def __init__(self, dim=96, d_state=16, expand=1):
        super().__init__()
        self.dim = dim
        self.d_inner = dim * expand
        
        self.norm = nn.InstanceNorm3d(dim)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        
        self.conv3d = nn.Conv3d(
            self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner, bias=True
        )
        self.act = nn.SiLU()
        
        self.ssm_fwd = SelectiveSSM1D(self.d_inner, d_state=d_state)
        self.ssm_bwd = SelectiveSSM1D(self.d_inner, d_state=d_state)
        
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        residual = x
        x_norm = self.norm(x)
        
        L = D * H * W
        x_flat = x_norm.permute(0, 2, 3, 4, 1).reshape(B, L, C)
        
        xz = self.in_proj(x_flat)
        x_proj, z = torch.chunk(xz, 2, dim=-1)
        
        # 3D Depthwise Conv
        x_conv = x_proj.reshape(B, D, H, W, self.d_inner).permute(0, 4, 1, 2, 3)
        x_conv = self.act(self.conv3d(x_conv))
        x_conv_flat = x_conv.permute(0, 2, 3, 4, 1).reshape(B, L, self.d_inner)
        
        # Bidirectional SSM scan
        y_fwd = self.ssm_fwd(x_conv_flat)
        y_bwd = self.ssm_bwd(torch.flip(x_conv_flat, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])
        
        y = (y_fwd + y_bwd) * self.act(z)
        out = self.out_proj(y)
        out = out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        
        return out + residual


class SharedReadoutHead(nn.Module):
    """
    Standardized weight-shared readout head:
      Applies shared 1x1x1 Conv3d (96 -> 4 classes) followed by trilinear upsampling (x4).
      Mathematically equivalent to Upsample(Conv1x1x1(F)) == Conv1x1x1(Upsample(F))
      due to linearity of both operators, reducing intermediate activation memory 24x.
    """
    def __init__(self, in_channels=96, num_classes=4):
        super().__init__()
        self.classifier = nn.Conv3d(in_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, f, target_shape=(128, 128, 128)):
        # f: (B, 96, 32, 32, 32)
        logits_low = self.classifier(f) # (B, 4, 32, 32, 32)
        logits = F.interpolate(logits_low, size=target_shape, mode='trilinear', align_corners=False) # (B, 4, 128, 128, 128)
        probs = F.softmax(logits, dim=1)
        return logits, probs


class ModularMambaBackbone(nn.Module):
    """
    Experiment 0 Modular 3D Mamba Backbone:
      Input (B, 4, 128, 128, 128)
        ↓
      Patch Stem (stride 4 conv) -> F_0: (B, 96, 32, 32, 32)
        ↓
      Block 1 (K=1) -> F_1 -> Readout Exit 1: P_1
        ↓
      Block 2 (K=2) -> F_2 -> Readout Exit 2: P_2
        ↓
      Block 3 (K=3) -> F_3 -> Readout Exit 3: P_3
    """
    def __init__(self, in_channels=4, hidden_dim=96, num_classes=4, d_state=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 3 Sequential Mamba Blocks (K=1, 2, 3)
        self.block1 = BidirectionalMambaBlock3D(dim=hidden_dim, d_state=d_state, expand=1)
        self.block2 = BidirectionalMambaBlock3D(dim=hidden_dim, d_state=d_state, expand=1)
        self.block3 = BidirectionalMambaBlock3D(dim=hidden_dim, d_state=d_state, expand=1)
        
        # Strictly weight-shared readout
        self.shared_head = SharedReadoutHead(in_channels=hidden_dim, num_classes=num_classes)

    def forward(self, x):
        """
        x: (B, 4, 128, 128, 128)
        Returns:
          logits: {"exit1": logits1, "exit2": logits2, "exit3": logits3}
          probs:  {"exit1": probs1,  "exit2": probs2,  "exit3": probs3}
        """
        target_size = (x.shape[2], x.shape[3], x.shape[4])
        
        # Stem: 128^3 -> 32^3
        f0 = self.stem(x) # (B, 96, 32, 32, 32)
        
        # Exit 1 (K=1)
        f1 = self.block1(f0)
        logits1, probs1 = self.shared_head(f1, target_shape=target_size)
        
        # Exit 2 (K=2)
        f2 = self.block2(f1)
        logits2, probs2 = self.shared_head(f2, target_shape=target_size)
        
        # Exit 3 (K=3)
        f3 = self.block3(f2)
        logits3, probs3 = self.shared_head(f3, target_shape=target_size)
        
        return {
            "logits": {"exit1": logits1, "exit2": logits2, "exit3": logits3},
            "probs":  {"exit1": probs1,  "exit2": probs2,  "exit3": probs3},
            "features": {"f0": f0, "f1": f1, "f2": f2, "f3": f3}
        }

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing updated ModularMambaBackbone on {device}...")
    model = ModularMambaBackbone().to(device)
    x = torch.randn(1, 4, 128, 128, 128, device=device)
    out = model(x)
    print("Forward pass successful! Exits:")
    for k in ["exit1", "exit2", "exit3"]:
        print(f"  {k}: logits {out['logits'][k].shape}, probs {out['probs'][k].shape}")
    print("ModularMambaBackbone verification passed!")
