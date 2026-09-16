"""Evaluate a saved checkpoint on the held-out test split, without model selection."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from .metrics import trainer_metrics
from .train import encode_rows, file_hash, read_rows, write_json


def publish_evaluation(run_dir, result, prediction_rows):
    """Publish metrics last, so an incomplete prediction write can be retried."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / 'test_metrics.json'
    if metrics_path.exists():
        raise FileExistsError('This run already has completed test metrics.')
    staged = []
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=run_dir, suffix='.tmp', delete=False) as handle:
            prediction_temp = Path(handle.name)
            staged.append(prediction_temp)
            for row in prediction_rows:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
        with tempfile.NamedTemporaryFile(mode='w', dir=run_dir, suffix='.tmp', delete=False) as handle:
            metrics_temp = Path(handle.name)
            staged.append(metrics_temp)
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(prediction_temp, run_dir / 'test_predictions.jsonl')
        os.replace(metrics_temp, metrics_path)
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def evaluate(run_dir):
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / 'resolved_config.json').read_text())
    if config.get('pilot'):
        raise ValueError('Final test evaluation is disabled for pilot runs; create a separate full experiment.')
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
    status = json.loads((run_dir / 'status.json').read_text())
    if status['status'] != 'completed':
        raise ValueError('Training must have completed before final evaluation.')
    # Prevent silent replacement or repeated score-shopping with the same run.
    output = run_dir / 'test_metrics.json'
    if output.exists():
        raise FileExistsError('This run already has test metrics; inspect the existing result.')
    model_dir = run_dir / 'model'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if config.get('lora'):
        from peft import PeftModel
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'], revision=config.get('resolved_model_revision') or config.get('model_revision', 'main'),
            num_labels=10, problem_type='multi_label_classification', trust_remote_code=False)
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    data_dir = Path(config['data_dir'])
    test_path = data_dir / 'test.jsonl'
    if config.get('expected_test_sha256') and file_hash(test_path) != config['expected_test_sha256']:
        raise ValueError('Test data differs from the prepared dataset registered at training.')
    test_rows = read_rows(test_path)
    used_texts = set()
    for split in ('train', 'validation'):
        split_path = data_dir / f'{split}.jsonl'
        if file_hash(split_path) != config['input_sha256'][split]:
            raise ValueError(f'{split} data changed since training.')
        used_texts.update(row['text'] for row in read_rows(split_path))
    if used_texts & {row['text'] for row in test_rows}:
        raise ValueError('Exact text overlap between test and development splits.')
    dataset = encode_rows(test_rows, tokenizer, config['max_length'])
    args = TrainingArguments(output_dir=str(run_dir / 'evaluation'), report_to='none',
        per_device_eval_batch_size=int(config.get('eval_batch_size', 2)),
        use_cpu=config['mode'] == 'smoke' or config.get('use_cpu', False),
        dataloader_pin_memory=False, label_names=['labels'])
    trainer = Trainer(model=model, args=args, data_collator=default_data_collator,
                      compute_metrics=trainer_metrics, processing_class=tokenizer)
    if config['mode'] == 'smoke':
        torch.set_num_threads(1)
    started = time.perf_counter()
    predictions = trainer.predict(dataset, metric_key_prefix='test')
    elapsed = time.perf_counter() - started
    result = {'metrics': predictions.metrics, 'test_sha256': file_hash(test_path),
              'test_rows': len(test_rows), 'synthetic': config['mode'] == 'smoke',
              'wall_seconds': elapsed, 'device': str(args.device),
              'note': 'End-to-end Trainer timing includes batching and metric computation; not isolated model latency.'}
    # Case indices and numeric decisions avoid redistributing case text.
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    prediction_rows = ({'row_index': index, 'logits': [float(x) for x in scores],
                        'predicted_labels': [i for i, score in enumerate(scores) if score > 0],
                        'labels': row['labels']} for index, (scores, row) in enumerate(zip(logits, test_rows)))
    publish_evaluation(run_dir, result, prediction_rows)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    evaluate(args.run_dir)


if __name__ == '__main__':
    main()
