"""Validate and construct parameter-efficient fine-tuning settings."""

import re


DTYPES = {"float16", "bfloat16", "float32"}


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def validate_adaptation(config):
    """Fail on ambiguous LoRA/QLoRA settings before loading data or weights."""
    lora = config.get("lora")
    quantization = config.get("quantization")
    if not lora and not quantization:
        return config
    revision = config.get("model_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("LoRA and QLoRA experiments require an immutable 40-character model revision.")
    if not isinstance(lora, dict):
        raise ValueError("Parameter-efficient training requires a lora object.")
    targets = lora.get("target_modules")
    if targets != "all-linear" and (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(value, str) and value for value in targets)
    ):
        raise ValueError("LoRA target_modules must be 'all-linear' or a nonempty list of names.")
    _positive_int(lora.get("r", 8), "LoRA r")
    _positive_int(lora.get("alpha", 16), "LoRA alpha")
    dropout = lora.get("dropout", 0.05)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be from 0 (inclusive) to 1 (exclusive).")
    modules_to_save = lora.get("modules_to_save", [])
    if not isinstance(modules_to_save, list) or not all(
        isinstance(value, str) and value for value in modules_to_save
    ):
        raise ValueError("LoRA modules_to_save must be a list of nonempty names.")
    if quantization is None:
        return config
    if not isinstance(quantization, dict) or quantization.get("method") != "bitsandbytes_4bit":
        raise ValueError("QLoRA currently supports method=bitsandbytes_4bit only.")
    if config.get("use_cpu"):
        raise ValueError("The verified QLoRA path requires an NVIDIA CUDA device.")
    if quantization.get("quant_type", "nf4") not in {"nf4", "fp4"}:
        raise ValueError("4-bit quant_type must be nf4 or fp4.")
    if quantization.get("compute_dtype", "float16") not in DTYPES:
        raise ValueError("4-bit compute_dtype must be float16, bfloat16 or float32.")
    if type(quantization.get("double_quant", True)) is not bool:
        raise ValueError("4-bit double_quant must be a boolean.")
    return config


def validate_module_names(model, lora):
    """Reject misspelled adapter and saved-module names for the selected architecture."""
    names = [name for name, _ in model.named_modules()]
    targets = lora["target_modules"]
    if targets != "all-linear" and any(
        not any(name == target or name.endswith("." + target) for name in names)
        for target in targets
    ):
        raise ValueError("At least one LoRA target module does not exist in the selected model.")
    if any(
        not any(name == target or name.endswith("." + target) for name in names)
        for target in lora.get("modules_to_save", [])
    ):
        raise ValueError("At least one LoRA modules_to_save entry does not exist in the selected model.")


def make_quantization_config(config):
    """Build the optional bitsandbytes configuration only in a training environment."""
    settings = config.get("quantization")
    if not settings:
        return None
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires an available NVIDIA CUDA device.")
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401 - verifies the optional runtime is installed.
    except ImportError as exc:
        raise RuntimeError("QLoRA dependencies are missing. Install the project with its qlora extra.") from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=settings.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=settings.get("double_quant", True),
        bnb_4bit_compute_dtype=getattr(torch, settings.get("compute_dtype", "float16")),
    )
