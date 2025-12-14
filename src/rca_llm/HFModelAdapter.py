"""
Encapsulation class for HuggingFace models to be trained with my Trainer class
"""

from transformers import AutoTokenizer
from transformers.trainer import get_parameter_names
import torch, torch.nn as nn

class HFModelAdapter(nn.Module):
    def __init__(self, hf_model, model_type):
        super().__init__()
        self.hf_model = hf_model
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type, token=True)
        # setup for pad_token id, making sure it is an int, not a list, 
        # as some tokenizers return a list of ids for the pad token
        pad_id = self.tokenizer.pad_token_id

        if not isinstance(pad_id, int):
            # Try to reuse EOS if it is a single int
            eos_id = self.tokenizer.eos_token_id
            if isinstance(eos_id, int) and self.tokenizer.eos_token is not None:
                # Reuse EOS as PAD in the tokenizer
                self.tokenizer.pad_token = self.tokenizer.eos_token
                pad_id = self.tokenizer.pad_token_id
            else:
                # Add a real PAD token
                self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
                pad_id = self.tokenizer.pad_token_id
                # Resize model embeddings to accommodate new token
                try:
                    self.hf_model.resize_token_embeddings(len(self.tokenizer))
                except AttributeError:
                    pass  # some wrappers may not expose this <- understand this

        # Now pad_id is guaranteed to be an int
        pad_id = int(pad_id)

        # Set on both config and generation_config (HF uses the latter in generate)
        self.hf_model.config.pad_token_id = pad_id
        if hasattr(self.hf_model, "generation_config"):
            self.hf_model.generation_config.pad_token_id = pad_id

    # here, idx and target are the x and y returned by the dataloader in trainer.py
    def forward(self, idx, targets=None):
        # Add a mask so pads aren’t attended at inference
        pad_id = self.hf_model.config.pad_token_id
        attn = (idx != pad_id).long() if pad_id is not None else None
        out = self.hf_model(input_ids=idx, labels=targets, attention_mask=attn)
        out = self.hf_model(input_ids=idx, labels=targets)
        # HF returns a ModelOutput with .logits and .loss
        # getattr(obj, name, default) tries to read attribute name from obj. 
        # If it exists, you get its value; if it doesn’t exist, you get default instead of raising an AttributeError.
        # HF models have a default parameter: return_dict=True, if it is set to False by mistake, then the getattr will avoid a crash by AttributeError
        return out.logits, getattr(out, "loss", None)
    
    def generate_from_prompt(
        self,
        prompt,
        device,
        max_new_tokens=300,
        do_sample=True,
        temperature=1.0,
        top_k=None,
        return_new_text_only=True,
        skip_special_tokens=True,
    ):
        if not prompt:
            return "no prompt given, please provide a non-empty prompt"
    
        
        encoded_input = self.tokenizer(prompt, return_tensors='pt').to(device)
        input_ids = encoded_input['input_ids'].to(device)

        attention_mask = encoded_input.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # pull the single int pad id set in __init__
        pad_id = int(self.hf_model.generation_config.pad_token_id
                    if hasattr(self.hf_model, "generation_config") and self.hf_model.generation_config.pad_token_id is not None
                    else self.hf_model.config.pad_token_id)

        y = self.hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=pad_id,  # ensure it's an int
            eos_token_id=self.hf_model.config.eos_token_id
        )

        if return_new_text_only:
            new_ids = y[0, input_ids.shape[1]:].tolist()
            return self.tokenizer.decode(new_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self.tokenizer.decode(y[0].tolist(), skip_special_tokens=skip_special_tokens)
    

    def configure_optimizers(self, train_config):
        decay_names = get_parameter_names(self, [torch.nn.LayerNorm])
        decay_names = [n for n in decay_names if not n.endswith(".bias")]

        decay_params, nodecay_params = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay_params if n in decay_names else nodecay_params).append(p)

        groups = [
            {"params": decay_params,   "weight_decay": train_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=train_config.learning_rate, betas=train_config.betas)