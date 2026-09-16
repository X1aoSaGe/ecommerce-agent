"""Load a base model and optionally a LoRA adapter.

Shared by the agent (Phase 5), evaluation (Phase 6) and DPO (Phase 4), so the
"which model" logic lives in one place. For inference the adapter is attached
frozen (eval mode, no gradients); for DPO training pass `trainable=True` so the
LoRA adapter parameters carry gradients again.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, adapter_path: str | None = None, trainable: bool = False):
    """Return (model, tokenizer).

    `adapter_path` is a directory saved by PeftModel.save_pretrained (e.g.
    models/sft_lora or models/dpo_lora). When given, the LoRA adapter is
    attached on top of the frozen base model.

    `trainable=True` re-enables gradients on the LoRA parameters only (the base
    stays frozen) and puts the model in train mode — used by train/dpo.py, whose
    policy model starts from the SFT adapter and gets fine-tuned further.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto"
    )
    if adapter_path:
        from peft import PeftModel

        # `is_trainable=True` marks the adapter params requires_grad=True and
        # freezes the base; the default (False) leaves everything frozen.
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    if trainable:
        for n, p in model.named_parameters():
            p.requires_grad = "lora_" in n
        model.train()
    else:
        model.eval()
    return model, tokenizer
