"""Bounded pretrained pilot settings, independent of optional ML dependencies."""
import re

PILOT_LIMITS = {'train_rows': 128, 'validation_rows': 32, 'max_steps': 20,
                'max_length': 256, 'batch_size': 2, 'eval_batch_size': 2,
                'gradient_accumulation_steps': 4}


def validate_pilot(config, require_budget=True):
    if config.get('pilot') is not True or config.get('mode') != 'pretrained':
        raise ValueError('Pilot requires pilot=true and mode=pretrained.')
    # Explicit first supported pilot keeps a config typo from loading a much larger model.
    if config.get('model_name') != 'EleutherAI/gpt-neo-125m':
        raise ValueError('The bounded pilot currently supports EleutherAI/gpt-neo-125m only.')
    if config.get('lora') or config.get('bf16') or config.get('fp16'):
        raise ValueError('First pretrained pilot uses full fine-tuning in float32.')
    for field, maximum in PILOT_LIMITS.items():
        value = config.get(field)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f'{field} must be an integer from 1 through {maximum}.')
    if type(config.get('seed')) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('An explicit nonnegative 32-bit seed is required.')
    if not isinstance(config.get('model_revision'), str) or not config['model_revision'].strip():
        raise ValueError('An explicit model revision is required.')
    if require_budget:
        budget = config.get('compute_budget_seconds')
        if type(budget) is not int or not 30 <= budget <= 1800:
            raise ValueError('Explicit compute_budget_seconds must be between 30 and 1800.')
    return config


def immutable_revision(revision):
    return isinstance(revision, str) and re.fullmatch(r'[0-9a-fA-F]{40}', revision) is not None
