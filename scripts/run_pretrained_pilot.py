"""Inspect or execute a bounded pretrained pilot; never performs final test evaluation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from ecthr_benchmark.pilot import validate_pilot
from ecthr_benchmark.pilot_data import create_pilot_subset
from ecthr_benchmark.train import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pilot_gpt_neo_125m.json')
    parser.add_argument('--budget-seconds', type=int, required=True,
                        help='Whole subprocess budget, including revision lookup, downloads, training and saves (30..1800).')
    parser.add_argument('--execute', action='store_true', help='Actually permit model download and pilot training.')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    config['compute_budget_seconds'] = args.budget_seconds
    validate_pilot(config)
    if not args.execute:
        print(json.dumps({'status': 'plan_only', 'config': config,
            'note': 'No model download, training or data preparation performed. Add --execute to run. Test evaluation is excluded.'}, indent=2))
        return
    attempt = Path(config['output_root']).resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    write_json(attempt / 'status.json', {'status': 'preparing', 'compute_budget_seconds': args.budget_seconds})
    try:
        create_pilot_subset(config['data_dir'], attempt / 'data', config['train_rows'],
                            config['validation_rows'], config['seed'])
        effective = {**config, 'data_dir': str(attempt / 'data'), 'output_root': str(attempt / 'training'),
                     'pilot_source_config_sha256': file_hash(config_path)}
        effective_path = attempt / 'pilot_config.json'
        write_json(effective_path, effective)
        command = [sys.executable, '-m', 'ecthr_benchmark.train', '--config', str(effective_path)]
        write_json(attempt / 'status.json', {'status': 'running', 'compute_budget_seconds': args.budget_seconds})
        started = time.perf_counter()
        with (attempt / 'training.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=os.name == 'posix')
            try:
                returncode = process.wait(timeout=args.budget_seconds)
                status = 'completed' if returncode == 0 else 'failed'
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                status = 'stopped_budget' if isinstance(exc, subprocess.TimeoutExpired) else 'interrupted'
                returncode = process.returncode
                for path in (attempt / 'training').glob('*/status.json'):
                    write_json(path, {'status': status, 'pilot': True})
        result = {'status': status, 'returncode': returncode, 'wall_seconds': time.perf_counter() - started,
                  'compute_budget_seconds': args.budget_seconds, 'test_evaluated': False,
                  'log': str(attempt / 'training.log'), 'attempt_dir': str(attempt)}
        write_json(attempt / 'status.json', result)
        print(json.dumps(result, indent=2))
        if returncode != 0:
            raise SystemExit(1)
    except Exception as exc:
        write_json(attempt / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()
