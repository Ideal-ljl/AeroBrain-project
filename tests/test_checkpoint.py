"""Check model loading and action inference with a small synthetic checkpoint."""

import torch
from transformers import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

from load_model import load_model
from openfly_his.configuration_openfly import OpenFLYModelConfig
from openfly_his.modeling_openfly import OpenFLYModel


def make_model():
    text = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=32,
        num_position_embeddings=16,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
    )
    vlm = Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=254,
        video_token_id=255,
        vision_start_token_id=252,
        vision_end_token_id=253,
    )
    prior = dict(
        horizon=3,
        state_dim=4,
        head_settings=dict(
            hidden_size=32,
            state_dim=4,
            history_steps=12,
            action_dim=4,
            arm="token",
            history_dropout=0.0,
        ),
        environments=["synthetic"],
        length_scale=[6.0],
        action_min=[[-1.0, -2.0, -3.0]],
        action_max=[[5.0, 4.0, 3.0]],
        template_distance=[[1 / 3] * 3 for _ in range(4)],
        template_yaw=[[k * 0.01] * 3 for k in range(4)],
        template_z=[[0.0] * 3 for _ in range(4)],
        noise_cholesky=(torch.eye(9)[None].repeat(4, 1, 1) * 0.2).tolist(),
        logit_shift=[0.1, -0.1, 0.2, -0.2],
    )
    config = OpenFLYModelConfig(
        vlm_config=vlm,
        action_dim=4,
        state_dim=4,
        chunk_size=3,
        expert_hidden_size=32,
        share_linear_attention_state=True,
        history_token_id=10,
        motion_prior_config=prior,
        mam_m_max=2,
        mam_tokens_per_event=2,
        mam_num_heads=4,
        mam_num_layers=1,
        runtime_vlm_prefix_cache_size=2,
    )
    return OpenFLYModel(config).eval()


def make_inputs():
    return dict(
        input_ids=torch.tensor([[1, 8, 10, 2]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        states=torch.randn(1, 1, 4),
        motion_environment=torch.tensor([0]),
        denoise_steps=2,
        mam_history_actions=torch.randn(1, 4, 4),
        mam_history_mask=torch.ones(1, 4, dtype=torch.bool),
        mam_anchor_pixel_values=torch.randn(8, 12),
        mam_anchor_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
        mam_anchor_n=torch.tensor([2]),
        mam_event_indices=torch.tensor([[1, 3]]),
        mam_event_mask=torch.ones(1, 2, dtype=torch.bool),
        mam_seg_start=torch.tensor([[0, 2]]),
        mam_seg_end=torch.tensor([[1, 3]]),
        mam_event_anchor_start=torch.tensor([[0, 0]]),
        mam_event_anchor_end=torch.tensor([[1, 1]]),
        mam_event_cache_keys=torch.tensor([[[0, 0, 1, 1], [0, 2, 3, 3]]]),
        mam_anchor_cache_keys=torch.tensor([[0, 1]]),
    )


def test_checkpoint_roundtrip_and_memory_cache(tmp_path):
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = make_model()
            inputs = make_inputs()
            model.save_pretrained(tmp_path)
            loaded, processor = load_model(
                tmp_path, device="cpu", dtype=torch.float32, load_processor=False
            )
            assert processor is None
            assert loaded.mam_memory_encoder.chunk_size == loaded.config.chunk_size == 3
            assert model.state_dict().keys() == loaded.state_dict().keys()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, loaded.state_dict()[name], rtol=0, atol=0
                )
            for case in (
                inputs,
                {k: v for k, v in inputs.items() if not k.startswith("mam_")},
            ):
                expected = model.sample_actions(
                    **case, gmm_generator=torch.Generator().manual_seed(33)
                )
                for _ in range(2):
                    actual = loaded.sample_actions(
                        **case, gmm_generator=torch.Generator().manual_seed(33)
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    actions, source_noise = actual
                    assert actions.shape == source_noise.shape == (1, 3, 4)
                    assert torch.isfinite(actions).all()
                    assert torch.isfinite(source_noise).all()
            assert loaded._mam_event_content_cache
            assert loaded._runtime_vlm_prefix_cache
    finally:
        torch.set_num_threads(threads)
