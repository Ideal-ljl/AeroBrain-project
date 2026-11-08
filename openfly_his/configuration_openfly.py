import copy
from typing import Optional, Union
from pathlib import Path
from transformers import PretrainedConfig, AutoConfig
from transformers import Qwen3_5Config, Qwen3_5TextConfig


class Qwen3_5WithExpertConfig(PretrainedConfig):
    """
    中间层配置：专门用于传递给 Model 层的初始化。
    它不需要独立注册为 AutoConfig，因为它是由 OpenFLYModelConfig 动态生成的。
    """

    model_type = "qwen3_5_with_expert"

    def __init__(
        self,
        vlm_config=None,
        expert_config=None,
        action_dim=4,
        state_dim=4,
        num_gmm_clusters=17,
        share_linear_attention_state=False,
        **kwargs,
    ):
        if num_gmm_clusters is None:
            num_gmm_clusters = 17
        self.vlm_config = vlm_config
        self.expert_config = expert_config
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_gmm_clusters = num_gmm_clusters
        self.share_linear_attention_state = bool(share_linear_attention_state)

        if expert_config and hasattr(expert_config, "hidden_size"):
            self.width = expert_config.hidden_size
        else:
            self.width = None

        super().__init__(**kwargs)


class OpenFLYModelConfig(PretrainedConfig):
    r"""
    OpenFLY 主配置类：作为容器，管理 VLM (Vision-Language) 和 Action Expert 的超参。
    基于 Qwen3.5 架构。
    """

    has_no_defaults_at_init = True
    model_type = "openfly"
    is_composition = True

    def __init__(
        self,
        vlm_config: Optional[Union[dict, "Qwen3_5Config"]] = None,
        expert_config: Optional[Union[dict, "Qwen3_5TextConfig"]] = None,
        action_dim=4,
        state_dim=4,
        max_action_dim=14,
        chunk_size=10,
        history_size=0,
        use_history=None,
        expert_hidden_size=640,
        token_merge=False,
        share_linear_attention_state=False,
        history_token_id=151669,
        num_gmm_clusters=17,
        use_gmm=False,
        gmm_prior_type="motion_conditional",
        mode_prior_spec=None,
        motion_prior_config=None,
        conditional_source_config=None,
        action_normalization="legacy_env",
        conditioning_normalization=None,
        flow_time_distribution="beta",
        use_mam_memory=True,
        mam_history_steps=48,
        mam_tau_percentile=80.0,
        mam_lambda_visual=1.0,
        mam_nms_gap=-1,
        mam_m_max=16,
        mam_tokens_per_event=2,
        mam_num_heads=4,
        mam_num_layers=1,
        mam_dropout=0.0,
        mam_detach_features=True,
        mam_prompt_text="Recent flight memory:",
        mam_prompt_token_ids=None,
        mam_chunk_boundary_mode="first_only",
        mam_event_mode="window",
        mam_tau_action=0.25,
        mam_max_seg_steps=40,
        mam_min_gap=4,
        mam_live_tail=True,
        use_action_prefix_conditioning=False,
        action_prefix_max_len=0,
        action_prefix_p_zero=0.5,
        action_prefix_noise_std=0.0,
        **kwargs,
    ):
        if num_gmm_clusters is None:
            num_gmm_clusters = 17
        self.token_merge = token_merge
        super().__init__(**kwargs)
        if (
            gmm_prior_type != "motion_conditional"
            or use_gmm
            or not use_mam_memory
            or conditional_source_config is not None
            or mode_prior_spec is not None
        ):
            raise ValueError("This model requires MSP and MAM")
        from .motion_prior_config import resolve_motion_prior_config

        if (
            action_dim != 4
            or motion_prior_config is None
            or action_normalization != "legacy_env"
        ):
            raise ValueError(
                "MSP requires embedded template parameters and per-environment MIN_MAX action coordinates"
            )
        motion_prior_config = resolve_motion_prior_config(
            motion_prior_config, horizon=chunk_size, state_dim=state_dim
        )
        self.motion_prior_config = copy.deepcopy(motion_prior_config)
        if flow_time_distribution not in ("beta", "uniform"):
            raise ValueError("Unknown flow time distribution")
        self.gmm_prior_type = gmm_prior_type
        self.action_mode = "motion_gmm"
        self.action_normalization = action_normalization
        if conditioning_normalization is None:
            conditioning_normalization = "legacy_dataset"
        if conditioning_normalization not in ("runtime_env", "legacy_dataset"):
            raise ValueError("Unknown conditioning normalization")
        self.conditioning_normalization = conditioning_normalization
        self.flow_time_distribution = flow_time_distribution

        if isinstance(vlm_config, dict):
            self.vlm_config = Qwen3_5Config(**vlm_config)
        elif isinstance(vlm_config, Qwen3_5Config):
            self.vlm_config = copy.deepcopy(vlm_config)
        elif isinstance(vlm_config, (str, Path)):
            self.vlm_config = Qwen3_5Config.from_pretrained(vlm_config)
        else:
            self.vlm_config = Qwen3_5Config()

        if isinstance(expert_config, dict):
            self.expert_config = Qwen3_5TextConfig(**expert_config)
        elif isinstance(expert_config, Qwen3_5TextConfig):
            self.expert_config = copy.deepcopy(expert_config)
        elif isinstance(expert_config, (str, Path)):
            self.expert_config = Qwen3_5TextConfig.from_pretrained(expert_config)
        else:
            self.expert_config = copy.deepcopy(self.vlm_config.text_config)
            if expert_hidden_size != self.expert_config.hidden_size:
                self.expert_config.hidden_size = expert_hidden_size

                self.expert_config.num_attention_heads = (
                    self.expert_config.num_key_value_heads
                )

                vlm_h = self.vlm_config.text_config.hidden_size
                vlm_inter = self.vlm_config.text_config.intermediate_size
                self.expert_config.intermediate_size = (
                    round(vlm_inter * expert_hidden_size / vlm_h / 64) * 64
                )

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_action_dim = max_action_dim
        self.chunk_size = chunk_size
        self.history_size = 0 if history_size is None else int(history_size)
        self.use_history = (
            (self.history_size > 0) if use_history is None else bool(use_history)
        )
        self.expert_hidden_size = expert_hidden_size
        self.share_linear_attention_state = bool(share_linear_attention_state)

        self.num_hidden_layers = self.vlm_config.text_config.num_hidden_layers
        self.hidden_size = self.vlm_config.text_config.hidden_size
        self.num_attention_heads = self.vlm_config.text_config.num_attention_heads

        self.num_key_value_heads = getattr(
            self.vlm_config.text_config, "num_key_value_heads", self.num_attention_heads
        )

        self.vocab_size = self.vlm_config.text_config.vocab_size
        self.use_gmm = use_gmm
        self.num_gmm_clusters = num_gmm_clusters
        self.history_token_id = history_token_id

        self.use_mam_memory = bool(use_mam_memory)
        self.mam_history_steps = int(mam_history_steps)
        self.mam_tau_percentile = float(mam_tau_percentile)
        self.mam_lambda_visual = float(mam_lambda_visual)
        self.mam_nms_gap = int(mam_nms_gap)
        self.mam_m_max = int(mam_m_max)
        self.mam_tokens_per_event = int(mam_tokens_per_event)
        self.mam_num_heads = int(mam_num_heads)
        self.mam_num_layers = int(mam_num_layers)
        self.mam_dropout = float(mam_dropout)
        self.mam_detach_features = bool(mam_detach_features)
        self.mam_prompt_text = str(mam_prompt_text or "")

        self.mam_prompt_token_ids = (
            list(mam_prompt_token_ids) if mam_prompt_token_ids else []
        )
        if mam_chunk_boundary_mode not in ("all", "first_only", "none"):
            raise ValueError(
                f"mam_chunk_boundary_mode must be 'all'/'first_only'/'none', "
                f"got {mam_chunk_boundary_mode!r}"
            )
        self.mam_chunk_boundary_mode = str(mam_chunk_boundary_mode)

        if mam_event_mode not in ("window", "causal"):
            raise ValueError(
                f"mam_event_mode must be 'window'/'causal', got {mam_event_mode!r}"
            )
        self.mam_event_mode = str(mam_event_mode)
        self.mam_tau_action = float(mam_tau_action)
        self.mam_max_seg_steps = int(mam_max_seg_steps)
        self.mam_min_gap = int(mam_min_gap)
        self.mam_live_tail = bool(mam_live_tail)

        self.use_action_prefix_conditioning = bool(use_action_prefix_conditioning)
        self.action_prefix_max_len = int(action_prefix_max_len)
        self.action_prefix_p_zero = float(action_prefix_p_zero)
        self.action_prefix_noise_std = float(action_prefix_noise_std)

        self._qwen3_5_with_expert_config = Qwen3_5WithExpertConfig(
            vlm_config=self.vlm_config,
            expert_config=self.expert_config,
            action_dim=action_dim,
            state_dim=state_dim,
            num_gmm_clusters=num_gmm_clusters,
            share_linear_attention_state=self.share_linear_attention_state,
        )

    @property
    def qwen3_5_with_expert_config(self):
        return self._qwen3_5_with_expert_config

    def to_dict(self):
        output = super().to_dict()
        output["vlm_config"] = self.vlm_config.to_dict()
        output["expert_config"] = self.expert_config.to_dict()
        return output


AutoConfig.register("openfly", OpenFLYModelConfig)
