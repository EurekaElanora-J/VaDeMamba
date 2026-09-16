"""Synthetic, data-free checks for VaDeMamba's active inference path.

Run directly with the project Python environment; no BraTS data or final-campaign
execution is required.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_final_test_campaign import (
    M1,
    M2,
    apply_block_to_selected_tiles,
    calculate_gmac,
    partition_latent_tiles,
    reassemble_latent_tiles,
)
from experiments.train_hardened_deployment_predictors import LinearRankNet, PredictorMLP
from src.models.modular_backbone import ModularMambaBackbone, SharedReadoutHead


class CountingBlock(nn.Module):
    """Records exactly which batched tile count entered a synthetic Mamba block."""

    def __init__(self, increment):
        super().__init__()
        self.increment = increment
        self.batch_sizes = []

    def forward(self, x):
        self.batch_sizes.append(x.shape[0])
        return x + self.increment


def test_model_and_shared_head_shapes():
    """Exercise the active model's three exits without expensive 3D SSM scans."""
    model = ModularMambaBackbone(in_channels=4, hidden_dim=96, num_classes=4, d_state=16)
    model.stem = nn.Conv3d(4, 96, kernel_size=1, stride=4)
    model.block1 = nn.Identity()
    model.block2 = nn.Identity()
    model.block3 = nn.Identity()
    with torch.no_grad():
        output = model(torch.zeros(1, 4, 128, 128, 128))
    for exit_name in ("exit1", "exit2", "exit3"):
        assert output["logits"][exit_name].shape == (1, 4, 128, 128, 128)
        assert output["probs"][exit_name].shape == (1, 4, 128, 128, 128)
    assert output["features"]["f1"].shape == (1, 96, 32, 32, 32)
    assert model.shared_head.classifier.out_channels == 4

    head = SharedReadoutHead(in_channels=96, num_classes=4)
    logits, probs = head(torch.zeros(1, 96, 32, 32, 32))
    assert logits.shape == probs.shape == (1, 4, 128, 128, 128)


def test_partition_reassembly_and_4_2_routing():
    # Encode spatial tile IDs so exact spatial reassembly can be verified.
    features = torch.zeros(1, 96, 32, 32, 32)
    for tile_id, (d, h, w) in enumerate(
        [(d, h, w) for d in range(2) for h in range(2) for w in range(2)]
    ):
        features[:, :, d * 16:(d + 1) * 16, h * 16:(h + 1) * 16, w * 16:(w + 1) * 16] = tile_id

    f1_tiles = partition_latent_tiles(features)
    assert f1_tiles.shape == (8, 96, 16, 16, 16)
    assert torch.equal(reassemble_latent_tiles(f1_tiles), features)

    s1 = [0, 2, 5, 7]
    s2 = [2, 7]
    assert len(s1) == M1 == 4
    assert len(s2) == M2 == 2
    assert set(s2).issubset(s1)

    block2 = CountingBlock(increment=10)
    f2_tiles = apply_block_to_selected_tiles(f1_tiles, s1, block2)
    assert block2.batch_sizes == [4]
    for tile_id in range(8):
        expected_increment = 10 if tile_id in s1 else 0
        assert torch.equal(f2_tiles[tile_id], f1_tiles[tile_id] + expected_increment)

    block3 = CountingBlock(increment=100)
    f3_tiles = apply_block_to_selected_tiles(f2_tiles, s2, block3)
    assert block3.batch_sizes == [2]
    for tile_id in range(8):
        expected_increment = 10 * (tile_id in s1) + 100 * (tile_id in s2)
        assert torch.equal(f3_tiles[tile_id], f1_tiles[tile_id] + expected_increment)
    assert reassemble_latent_tiles(f3_tiles).shape == (1, 96, 32, 32, 32)


def test_locked_gmac_values():
    assert round(calculate_gmac(0, 0), 3) == 6.893
    assert round(calculate_gmac(8, 0), 3) == 8.333
    assert round(calculate_gmac(8, 8), 3) == 9.773
    assert round(calculate_gmac(4, 2), 3) == 7.973


def test_deployed_predictor_checkpoint_compatibility():
    g1 = torch.load(PROJECT_ROOT / "checkpoints" / "gate1_deployment.pt", map_location="cpu", weights_only=False)
    assert g1["in_dim"] == 7
    assert g1["feature_columns"] == [
        "U1_entropy", "U1_margin", "P1_BG", "P1_NCR", "P1_ED", "P1_ET", "P1_max",
    ]
    g1_model = PredictorMLP(in_dim=g1["in_dim"], hidden_dim=g1["hidden_dim"])
    g1_model.load_state_dict(g1["model_state_dict"])
    assert g1_model(torch.zeros(4, 7)).shape == (4,)

    g2 = torch.load(PROJECT_ROOT / "checkpoints" / "gate2_deployment.pt", map_location="cpu", weights_only=False)
    assert g2["in_dim"] == 16
    assert len(g2["feature_columns"]) == 16
    g2_model = LinearRankNet(in_dim=g2["in_dim"])
    g2_model.load_state_dict(g2["model_state_dict"])
    assert g2_model(torch.zeros(4, 16)).shape == (4,)


if __name__ == "__main__":
    test_model_and_shared_head_shapes()
    test_partition_reassembly_and_4_2_routing()
    test_locked_gmac_values()
    test_deployed_predictor_checkpoint_compatibility()
    print("Synthetic VaDeMamba active-path tests passed.")
