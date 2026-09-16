#!/usr/bin/env python3
"""Curate publication copies from the private audit; never modify source files.

Run from the repository root: python scripts/curate_archive.py
The ignored project-audit directory is a local prerequisite, not a download.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

TOKEN = re.compile(r"hf_[A-Za-z0-9]{10,}")
PRIVATE_PATH = re.compile(r"(?:/Users/|/home/|/root/|/content/drive/)[^\n\"']+")


def clean_source(text: str, code: bool) -> str:
    if code:
        # A whole credential literal becomes an expression, not a dummy credential.
        text = re.sub(r"(['\"])hf_[A-Za-z0-9]{10,}\1", "__import__('os').environ['HF_TOKEN']", text)
    text = TOKEN.sub('[credential removed; use HF_TOKEN]', text)
    # Retain the data filename, remove the personal directory hierarchy.
    text = PRIVATE_PATH.sub(lambda m: 'data/historical/' + m.group().rstrip('/').rsplit('/', 1)[-1], text)
    return text


def slug(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')


def family(row: dict) -> str:
    p = row['path'].lower()
    if 'preprocessing' in p:
        return 'preprocessing'
    if 'results' in p:
        return 'analysis'
    if 'gpt2' in p or 'dialogrpt' in p:
        return 'dialogrpt'
    return {'gpt-neo': 'gpt_neo', 'phi2': 'phi2', 't5': 't5', 'dissertation_code': 'gemma', 'phi1': 'phi1'}[p.split('/')[0]]


def limitations(group: str) -> str:
    common = 'Historical code, not independently reproduced. Saved settings can differ from report labels; notebook filenames are not authoritative configurations.'
    notes = {
        'preprocessing': 'Article descriptions map IDs to Articles 1–10 incorrectly. Historic preprocessing is retained for provenance; use the new preparation pipeline for corrected labels.',
        'analysis': 'Historical plotting code contains unavailable baseline weighted-F1 values represented as zero and mixed score scales. Do not reuse these figures as corrected comparisons.',
        'dialogrpt': 'The source named GPT2.ipynb loads DialogRPT, so it does not substantiate a GPT-2 Medium run. Principal training code evaluates on validation; a final held-out test run is not established.',
        'gpt_neo': 'Principal training code evaluates on validation; a final held-out test run is not established. A report metric/configuration transcription discrepancy remains unresolved.',
        'phi2': 'Test data is used during checkpoint selection in the recovered training notebooks. These results are not independent held-out test estimates.',
        't5': 'Checkpoint IDs identify google-t5/t5-base, not FLAN-T5. Test data is used during checkpoint selection; filenames may differ from epoch settings.',
        'phi1': 'Exploratory work; source records include incomplete results and training budgets that require reconciliation.',
        'gemma': 'Exploratory classification and prompting variants. The report describes NaN failures; no verified comparable performance claim is made.',
    }
    return common + ' ' + notes[group]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=Path('project-audit'))
    args = parser.parse_args()
    audit = args.audit_dir
    index = json.loads((audit / 'notebook-index.json').read_text())
    seen = set()
    manifest = []
    catalogue = [
        '# Historical experiment catalogue', '',
        'These publication copies preserve the distinct notebook code versions recovered from the MSc archive. They are research records, not the supported training entry point. No historical notebook has been rerun during curation.', '',
        'Identical code-cell hashes are grouped even when original outputs differ. Outputs, attachments, execution counts and runtime metadata are removed. Credential literals read `HF_TOKEN` from the environment; remaining credential mentions are redacted. Personal absolute paths become `data/historical/<original filename>`. Local relative Parquet paths remain historical dependencies and are not supplied. Original source files remain unchanged in the private audit.', '',
        'The manifest records source-relative paths, original file and code hashes, publication hashes and duplicate sources. Original saved outputs are intentionally private; reported metrics are preserved separately in `results/historical/reported_results.csv`.', '',
        '## Important interpretation limits', '',
        '- Most historical classification experiments use an explicit eleventh no-violation output. New benchmark-compatible experiments use ten article outputs with no violation derived during evaluation; scores should not be silently pooled.',
        '- Historic article display names are incorrect; numeric IDs were preserved in the audited local preprocessing round-trip. Prompt-based experiments can be affected by descriptions.',
        '- GPT-Neo 125M and RoBERTa training notebooks were not located. A file named GPT2.ipynb loads DialogRPT. T5 checkpoint IDs indicate T5-base rather than FLAN-T5.',
        '- Phi-2 and T5 use test data during checkpoint selection. Their historic scores are not independent test estimates.',
        '- Historical analysis code is preserved, including figure defects; unavailable baseline metrics must not be interpreted as zero.',
        '- Metric rows are source transcriptions. Missing values stay blank; unusual token/budget strings remain as written. The GPT-Neo 2,048-token report-only row is unreconciled and must not be silently corrected.',
        '- Token removal does not revoke credentials. Any still-active credentials from the original archive need rotation through their issuing service.', '',
        '## Distinct code versions', '',
        '| Publication notebook | Checkpoint identity / purpose | Source |', '| --- | --- | --- |',
    ]
    for row in index:
        code_hash = row['code_sha256']
        if code_hash in seen:
            continue
        seen.add(code_hash)
        src = audit / 'sources' / row['path'].replace(':', '_')
        raw = src.read_bytes()
        notebook = json.loads(raw)
        group = family(row)
        target = Path('notebooks/historical') / group / (slug(Path(row['path']).stem) + '_' + code_hash[:8] + '.ipynb')
        target.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = ', '.join(row['model_ids']) or ('microsoft/DialogRPT-updown (loaded by GPT2.ipynb)' if group == 'dialogrpt' else 'Analysis / preprocessing')
        header = f'# Historical archive: {Path(row["path"]).stem}\n\n**Status: not independently reproduced.**\n\nCheckpoint / purpose: `{checkpoint}`.\n\n{limitations(group)}\n\nPublication edits only: credentials use `HF_TOKEN`; personal absolute paths were relocated; saved outputs, attachments and runtime metadata were removed. Historical installation cells and local data dependencies are retained. Review code before running; it can download models, install packages and start expensive training. See `docs/experiment_catalogue.md` and the new pipeline for supported experiments.\n'
        cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': header.splitlines(keepends=True)}]
        for cell in notebook['cells']:
            code = cell['cell_type'] == 'code'
            source = cell.get('source', '')
            source = source if isinstance(source, str) else ''.join(source)
            entry = {'cell_type': cell['cell_type'], 'metadata': {}, 'source': clean_source(source, code).splitlines(keepends=True)}
            if code:
                entry.update(outputs=[], execution_count=None)
            cells.append(entry)
        clean = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 4}
        payload = json.dumps(clean, ensure_ascii=False, indent=1) + '\n'
        if TOKEN.search(payload) or PRIVATE_PATH.search(payload):
            raise ValueError(f'Publication scan failed for {target}')
        target.write_text(payload)
        manifest.append({'notebook': str(target), 'source_relative_path': row['path'], 'source_file_sha256': hashlib.sha256(raw).hexdigest(), 'source_code_sha256': code_hash, 'publication_sha256': hashlib.sha256(payload.encode()).hexdigest(), 'source_code_lines': row['code_lines'], 'checkpoint_or_purpose': checkpoint, 'duplicate_code_sources': [r['path'] for r in index if r['code_sha256'] == code_hash and r['path'] != row['path']], 'transformations': ['remove outputs, attachments, execution counts and runtime metadata', 'replace credential literals with HF_TOKEN environment lookup and redact remaining mentions', 'relocate personal absolute paths to data/historical', 'prepend historical status and limitations; preserve other source behavior'], 'limitations': limitations(group)})
        catalogue.append(f'| [{target.stem}](../{target}) | {checkpoint} | `{row["path"]}` |')
    Path('notebooks/historical/manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    Path('docs').mkdir(exist_ok=True)
    Path('docs/experiment_catalogue.md').write_text('\n'.join(catalogue) + '\n')
    out = Path('results/historical')
    out.mkdir(parents=True, exist_ok=True)
    # Preserve every source value while normalizing record separators for clean Git diffs.
    source_results = audit / 'reported-results.csv'
    with source_results.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    with (out / 'reported_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Curated {len(manifest)} distinct notebooks ({sum(r["source_code_lines"] for r in manifest)} original code lines), preserved {len(rows)} reported result rows. Credential/path scans passed.')


if __name__ == '__main__':
    main()
