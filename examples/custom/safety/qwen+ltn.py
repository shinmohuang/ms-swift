from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
import torch
import torch.nn as nn
import ltn
from ltn.fuzzy_ops import *
from ltn.core import Constant, Variable


class QwenLTN(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        last_hidden_state = outputs.hidden_states[-1]

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        return

    class RelationLTN(nn.Module):
        def __init__(self, embed_size=768, output_dim=4):
            super().__init__()
            self.embed_size = embed_size
            self.output_dim = output_dim

            self.fc1 = nn.Linear(embed_size, embed_size)
            self.fc2 = nn.Linear(embed_size, embed_size)

            self.sim_shreshold = nn.Parameter(torch.tensor(0.5))

        def forward(self, p_img, p_text):
