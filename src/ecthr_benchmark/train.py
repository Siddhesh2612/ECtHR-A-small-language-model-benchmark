"""Configuration-driven classification training; never reads test data to select models."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import uuid

from .labels import LABEL_NAMES as ARTICLE_NAMES, multi_hot
from .metrics import trainer_metrics
from .pilot import immutable_revision, validate_pilot
from .adaptation import (make_quantization_config, validate_adaptation,
                         validate_module_names)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError(f'{path}:{line_number}: text must be nonempty.')
        if not isinstance(row.get('labels'), list):
            raise ValueError(f'{path}:{line_number}: labels must be a list.')
        multi_hot(row['labels'])
        rows.append(row)
    if not rows:
        raise ValueError(f'{path} has no cases.')
    return rows


def encode_rows(rows, tokenizer, max_length):
    # Float labels make the classifier use binary cross entropy, not single-label CE.
    import torch
    encoded = []
    for row in rows:
        item = tokenizer(row['text'], truncation=True, max_length=max_length,
                         padding='max_length', return_token_type_ids=False)
        item['labels'] = multi_hot(row['labels'])
        encoded.append({key: torch.tensor(value, dtype=torch.float32 if key == 'labels' else torch.long)
                        for key, value in item.items()})
    return encoded


def smoke_model(max_length):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2ForSequenceClassification, PreTrainedTokenizerFast
    words = ['[PAD]', '[UNK]', '[EOS]', 'case', 'facts', 'court', 'hearing', 'claim', 'dismissed',
             'detention', 'review', 'fair', 'family', 'speech', 'property', 'synthetic']
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                       unk_token='[UNK]', eos_token='[EOS]', model_max_length=max_length)
    config = GPT2Config(vocab_size=len(words), n_positions=max_length, n_embd=16, n_layer=1,
                        n_head=2, num_labels=10, pad_token_id=0, eos_token_id=2,
                        problem_type='multi_label_classification')
    return tokenizer, GPT2ForSequenceClassification(config)


def smoke_data(directory):
    directory.mkdir()
    for split, count in [('train', 12), ('validation', 4), ('test', 4)]:
        rows = [{'text': f'synthetic {split} case {i} court hearing detention review',
                 'labels': [] if i % 3 == 0 else [i % 10]} for i in range(count)]
        (directory / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


def train(config_path):
    try:
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  Trainer, TrainingArguments, default_data_collator, set_seed)
    except ImportError as exc:
        raise SystemExit('Training dependencies missing. Install this project with its train extra.') from exc
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    validate_adaptation(config)
    mode = config.get('mode', 'pretrained')
    if mode not in ('smoke', 'pretrained'):
        raise ValueError('mode must be smoke or pretrained')
    if config.get('pilot'):
        validate_pilot(config)
    max_length = int(config.get('max_length', 512))
    if max_length < 2:
        raise ValueError('max_length must be at least 2')
    seed = int(config.get('seed', 42))
    set_seed(seed)  # Before random model / classification-head initialization.
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    run_dir = Path(config.get('output_root', 'runs')).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / 'status.json', {'status': 'started', 'mode': mode})
    try:
        if mode == 'smoke':
            torch.set_num_threads(1)
            smoke_data(run_dir / 'synthetic_data')
            data_dir = run_dir / 'synthetic_data'
            tokenizer, model = smoke_model(max_length)
        else:
            data_dir = Path(config['data_dir']).resolve()
            manifest = json.loads((data_dir / 'manifest.json').read_text())
            for split in ('train', 'validation'):
                if file_hash(data_dir / f'{split}.jsonl') != manifest['splits'][split]['prepared_sha256']:
                    raise ValueError(f'{split} hash does not match prepared manifest.')
            # Fail on missing/invalid data before downloading model weights.
            checked_train = read_rows(data_dir / 'train.jsonl')
            checked_validation = read_rows(data_dir / 'validation.jsonl')
            if {row['text'] for row in checked_train} & {row['text'] for row in checked_validation}:
                raise ValueError('Exact text overlap between training and validation.')
            expected_mapping = {str(i): name for i, name in enumerate(ARTICLE_NAMES)}
            if manifest.get('label_mapping') != expected_mapping:
                raise ValueError('Prepared label mapping does not match ECtHR-A.')
            if config.get('pilot'):
                if manifest.get('pilot') is not True or manifest.get('mode') != 'pretrained_pilot_subset':
                    raise ValueError('Pilot runs require a pretrained_pilot_subset manifest.')
                if len(checked_train) != config['train_rows'] or len(checked_validation) != config['validation_rows']:
                    raise ValueError('Pilot prepared row counts differ from configured bounded subset.')
            model_id = config['model_name']
            revision = config.get('model_revision', 'main')
            if config.get('pilot') and not immutable_revision(revision):
                from huggingface_hub import HfApi
                revision = HfApi().model_info(model_id, revision=revision).sha
                if not immutable_revision(revision):
                    raise ValueError('Could not resolve an immutable model revision.')
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
            quantization_config = make_quantization_config(config)
            model_kwargs = {}
            if quantization_config is not None:
                model_kwargs['quantization_config'] = quantization_config
                model_kwargs['dtype'] = getattr(torch, config['quantization'].get('compute_dtype', 'float16'))
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, revision=revision, num_labels=10,
                problem_type='multi_label_classification', trust_remote_code=False,
                **model_kwargs)
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    model.resize_token_embeddings(len(tokenizer))
        context_limit = getattr(model.config, 'max_position_embeddings', None) or getattr(model.config, 'n_positions', None)
        if context_limit and max_length > context_limit:
            raise ValueError(f'max_length {max_length} exceeds model context limit {context_limit}.')
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.id2label = dict(enumerate(ARTICLE_NAMES))
        model.config.label2id = {name: i for i, name in enumerate(ARTICLE_NAMES)}
        model.config.use_cache = False
        lora = config.get('lora')
        if lora:
            if mode == 'smoke':
                raise ValueError('Smoke mode verifies full fine-tuning only.')
            validate_module_names(model, lora)
            from peft import (LoraConfig, TaskType, get_peft_model,
                              prepare_model_for_kbit_training)
            if config.get('quantization'):
                model = prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=bool(config.get('gradient_checkpointing', False)))
            model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS,
                r=int(lora.get('r', 8)), lora_alpha=int(lora.get('alpha', 16)),
                lora_dropout=float(lora.get('dropout', 0.05)),
                target_modules=lora['target_modules'],
                modules_to_save=lora.get('modules_to_save'),
                bias=lora.get('bias', 'none'), use_rslora=bool(lora.get('use_rslora', False))))
        train_rows = read_rows(data_dir / 'train.jsonl')
        validation_rows = read_rows(data_dir / 'validation.jsonl')
        if {row['text'] for row in train_rows} & {row['text'] for row in validation_rows}:
            raise ValueError('Exact text overlap between training and validation.')
        resolved = {**config, 'mode': mode, 'seed': seed, 'max_length': max_length, 'run_id': run_id,
                    'run_dir': str(run_dir), 'data_dir': str(data_dir),
                    'config_source': str(config_path), 'config_sha256': file_hash(config_path),
                    'resolved_model_revision': getattr(model.config, '_commit_hash', None) or (revision if mode != 'smoke' else None),
                    'label_names': ARTICLE_NAMES, 'threshold': 'logit > 0',
                    'evaluation_policy': '10 article outputs + derived no-violation; test is separate',
                    'input_sha256': {split: file_hash(data_dir / f'{split}.jsonl')
                                     for split in ('train', 'validation')}}
        if mode != 'smoke':
            resolved['prepared_manifest_sha256'] = file_hash(data_dir / 'manifest.json')
            resolved['expected_test_sha256'] = manifest['splits']['test']['prepared_sha256']
            write_json(run_dir / 'prepared_manifest.json', manifest)
        write_json(run_dir / 'resolved_config.json', resolved)
        train_data = encode_rows(train_rows, tokenizer, max_length)
        val_data = encode_rows(validation_rows, tokenizer, max_length)
        args = TrainingArguments(output_dir=str(run_dir / 'checkpoints'),
            num_train_epochs=float(config.get('epochs', 1)), max_steps=int(config.get('max_steps', -1)),
            learning_rate=float(config.get('learning_rate', 3e-5)),
            per_device_train_batch_size=int(config.get('batch_size', 2)),
            per_device_eval_batch_size=int(config.get('eval_batch_size', 2)),
            gradient_accumulation_steps=int(config.get('gradient_accumulation_steps', 1)),
            weight_decay=float(config.get('weight_decay', 0)), seed=seed, data_seed=seed,
            eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
            metric_for_best_model='micro_f1', greater_is_better=True, save_total_limit=1,
            report_to='none', use_cpu=mode == 'smoke' or config.get('use_cpu', False),
            bf16=config.get('bf16', False), fp16=config.get('fp16', False),
            dataloader_pin_memory=False, optim='adamw_torch', logging_steps=int(config.get('logging_steps', 1 if mode == 'smoke' else 50)),
            gradient_checkpointing=bool(config.get('gradient_checkpointing', False)),
            label_names=['labels'])
        trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=val_data,
                          data_collator=default_data_collator, compute_metrics=trainer_metrics,
                          processing_class=tokenizer)
        if torch.cuda.is_available() and args.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        training_result = trainer.train()
        elapsed = time.perf_counter() - started
        trainer.save_model(str(run_dir / 'model'))
        tokenizer.save_pretrained(run_dir / 'model')
        trainer.save_state()
        write_json(run_dir / 'validation_metrics.json', trainer.evaluate())
        write_json(run_dir / 'runtime.json', {
            'training_wall_seconds': elapsed, 'training_metrics': training_result.metrics,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'device': str(args.device), 'hardware': platform.platform(),
            'cuda_device': torch.cuda.get_device_name() if args.device.type == 'cuda' else None,
            'peak_cuda_memory_allocated_bytes': torch.cuda.max_memory_allocated() if args.device.type == 'cuda' else None,
            'model_memory_footprint_bytes': model.get_memory_footprint() if hasattr(model, 'get_memory_footprint') else None,
            'versions': {p: importlib.metadata.version(p) for p in
                         (('torch', 'transformers', 'tokenizers', 'accelerate')
                          + (('peft',) if lora else ())
                          + (('bitsandbytes',) if config.get('quantization') else ()))},
            'best_checkpoint': trainer.state.best_model_checkpoint,
            'synthetic': mode == 'smoke', 'pilot': bool(config.get('pilot')), 'test_evaluated': False})
        write_json(run_dir / 'status.json', {'status': 'completed', 'mode': mode})
    except Exception as exc:
        write_json(run_dir / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__})
        raise
    print(f'Run saved: {run_dir}')
    if config.get('pilot'):
        print('Pilot complete. Final test evaluation is disabled; plan a separate full experiment next.')
    else:
        print(f'Evaluate separately: python -m ecthr_benchmark.evaluate --run-dir "{run_dir}"')
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == '__main__':
    main()
