"""Guards on the installed LeWM stack (skipped when it is not installed).

These exist because of real failures found in Phase 0: with transformers 5.x the
HF ViT parameters are renamed and the released LeWM weights no longer load.
"""

import pytest

swm = pytest.importorskip("stable_worldmodel")
pytest.importorskip("stable_pretraining")

from jmem.utils import count_params  # noqa: E402

# Predictor block of the released config.json (identical for all four checkpoints).
RELEASED_PREDICTOR = dict(
    num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
    depth=6, heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0,
)


def test_vit_parameter_names_match_released_weights():
    from stable_pretraining.backbone.utils import vit_hf

    vit = vit_hf("tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
    keys = set(vit.state_dict())
    # naming used by quentinll/lewm-*/weights.pt (transformers 4.x layout)
    assert "encoder.layer.0.attention.attention.query.weight" in keys, (
        "HF ViT parameter names differ from the released checkpoints; "
        "install transformers>=4.50,<5"
    )
    assert "embeddings.cls_token" in keys
    assert count_params(vit) == 5_501_376  # measured on quentinll/lewm-tworooms


def test_released_predictor_size_and_history_attribute():
    from stable_worldmodel.wm.lewm.module import Predictor

    pred = Predictor(**RELEASED_PREDICTOR)
    # reference size for the "matched parameters" rule (brief 6.1)
    assert count_params(pred) == 10_791_360
    assert tuple(pred.pos_embedding.shape) == (1, 3, 192)


def test_lewm_rollout_from_a_single_frame():
    """Planner-shaped call: 1 real frame in, horizon+1 embeddings out."""
    import torch
    from torch import nn
    from stable_worldmodel.wm.lewm import LeWM
    from stable_worldmodel.wm.lewm.module import Embedder, Predictor

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3 * 8 * 8, 192)

        def forward(self, x, interpolate_pos_encoding=True):
            class Out:
                pass

            out = Out()
            out.last_hidden_state = self.lin(x.flatten(1)).unsqueeze(1)
            return out

    model = LeWM(
        encoder=TinyEncoder(),
        predictor=Predictor(**RELEASED_PREDICTOR),
        action_encoder=Embedder(input_dim=10, emb_dim=192),
    ).eval()
    # B must be 1: LeWM.criterion broadcasts goal_emb (B, 1, D) against (B, S, T, D), which
    # only works for B == 1 -- the reason the official CEM config uses batch_size: 1.
    B, S, T, A = 1, 4, 5, 10
    info = {
        "pixels": torch.randn(B, S, 1, 3, 8, 8),
        "goal": torch.randn(B, S, 1, 3, 8, 8),
        "action": torch.zeros(B, S, 1, A),
    }
    with torch.inference_mode():
        cost = model.get_cost(info, torch.randn(B, S, T, A))
    assert cost.shape == (B, S)
    assert info["predicted_emb"].shape == (B, S, 1 + T, 192)
    assert torch.isfinite(cost).all()
