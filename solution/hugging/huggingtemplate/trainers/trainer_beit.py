from typing import Dict, Union, Any

import torch
import torch.nn as nn

from transformers import Trainer


class BeitForMIMTrainer(Trainer):

    def __init__(self, dall_e_tokenizer: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.dall_e_tokenizer = dall_e_tokenizer

    def train(self, **kwargs):
        self.dall_e_tokenizer = self.dall_e_tokenizer.to(self.args.device)
        super().train(**kwargs)

    def _prepare_inputs(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        inputs = super()._prepare_inputs(inputs)
        visual_tokens = inputs.pop("visual_tokens")
        bool_masked_pos = inputs.pop("bool_masked_pos")
        with torch.no_grad():
            z_logits = self.dall_e_tokenizer(visual_tokens)
            input_ids = torch.argmax(z_logits, dim=1).flatten(1)
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
            labels = input_ids[bool_masked_pos]
        inputs.update({
            "bool_masked_pos": bool_masked_pos,
            "labels": labels,
        })
        return inputs
