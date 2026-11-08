"""Qwen vision-language prefix and action Expert sharing attention state."""

from typing import Optional, Tuple
import torch
from transformers import PreTrainedModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from openfly_his.configuration_openfly import Qwen3_5WithExpertConfig
from openfly_his.modeling_expert import Qwen3_5ExpertModel


class Qwen3_5WithExpertModel(PreTrainedModel):
    config_class = Qwen3_5WithExpertConfig
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: Qwen3_5WithExpertConfig):
        super().__init__(config)
        self.config = config
        self.vlm_backbone = Qwen3_5ForConditionalGeneration(
            config=self.config.vlm_config
        )
        router_num_clusters = getattr(config, "num_gmm_clusters", 17)
        if router_num_clusters is None:
            router_num_clusters = 17
        self.expert = Qwen3_5ExpertModel(
            config.expert_config,
            config.action_dim,
            config.state_dim,
            config.vlm_config.text_config.hidden_size,
            int(router_num_clusters),
            share_linear_attention_state=getattr(
                config, "share_linear_attention_state", False
            ),
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        suffix_embeds: Optional[torch.FloatTensor] = None,
        full_suffix_attention: bool = False,
        past_key_values_seq_length: Optional[int] = None,
    ) -> Tuple:
        """
        Returns: ([prefix_hidden_states, suffix_hidden_states], past_key_values, rope_deltas)

        Prefix mode  (suffix_embeds is None):
            Runs the full Qwen3.5 VLM pipeline (image embedding + MRoPE + language model).
        Suffix mode  (suffix_embeds is not None, input_ids is None):
            Runs only the action Expert.
        """
        if suffix_embeds is None:
            vlm_output = self.vlm_backbone.model(
                input_ids=input_ids if inputs_embeds is None else None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                cache_position=cache_position,
                use_cache=use_cache,
            )
            prefix_hidden = vlm_output.last_hidden_state
            prefix_pkv = vlm_output.past_key_values
            rope_deltas = vlm_output.rope_deltas
            return ([prefix_hidden, None], prefix_pkv, rope_deltas)
        else:
            expert_out = self.expert(
                inputs_embeds=suffix_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                full_suffix_attention=full_suffix_attention,
                past_key_values_seq_length=past_key_values_seq_length,
            )
            return ([None, expert_out.last_hidden_state], None, None)
