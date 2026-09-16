"""
Benefit Predictor Models: 2-Layer MLP and Linear Ridge Baseline
--------------------------------------------------------------
The benefit predictor is intentionally lightweight, and subject-grouped
cross-validation is used to evaluate generalization while preventing patient-level leakage.
"""

import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

class MarginalBenefitMLP(nn.Module):
    """
    Lightweight 2-layer MLP continuous regressor:
      Linear(in_dim -> hidden_dim) -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim -> 1)
    Outputs unconstrained continuous scalar in (-inf, +inf).
    """
    def __init__(self, in_dim, hidden_dim=32, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def get_ridge_baseline(alpha=1.0):
    """Linear baseline for comparison with non-linear MLP."""
    return Ridge(alpha=alpha)
