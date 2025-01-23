from trl import SFTTrainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import importlib.metadata
from packaging import version
from utils import attack_loss
import torch

def first_difference(str1, str2):
    for a, b in zip(str1, str2):
        if a != b:
            return a+b

from transformers.utils import (
    is_peft_available,
)

if is_peft_available():
    from peft import PeftModel

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

class DualTrainer(SFTTrainer):
    def __init__(self, prompt, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = prompt
        self.device = device
    
    def compute_loss(self, model, inputs):
        embeddings = model.get_input_embeddings().weight
        vocab_size = embeddings.shape[0]
        one_hot = torch.zeros(
            self.prompt.seq_len,
            vocab_size,
            device=self.device,
            dtype=embeddings.dtype
            ).to(self.device)
        one_hot.scatter_(
            1,
            self.prompt.tokens.squeeze(0).unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=self.device, dtype=embeddings.dtype)
            )
        one_hot_embeddings = (one_hot @ embeddings).unsqueeze(0)
        logits = model(inputs_embeds=one_hot_embeddings).logits
        loss = attack_loss(logits[0], self.prompt)
        return loss