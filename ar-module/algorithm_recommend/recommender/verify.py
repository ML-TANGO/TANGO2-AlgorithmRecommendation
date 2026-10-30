"""Verify prepared data, upstream preservation and checkpoint integrity without training."""
import argparse
import json
from pathlib import Path

from .common import SCHEMA_VERSION, read_json, read_jsonl, file_hash, target_errors, write_json, checkpoint_errors
from ..io_utils import sha
from ..paths import DATA_DIR, DEFAULT_CHECKPOINT, REV4, ARTIFACTS, resolve, rel


def verify(data_dir=DATA_DIR, checkpoint=DEFAULT_CHECKPOINT, stage4=REV4, output=None):
    data_dir = Path(data_dir); stage4 = Path(stage4)
    prep = read_json(data_dir / 'preparation_report.json'); catalog = read_json(data_dir / 'catalog.json')
    if prep['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Schema mismatch')
    absent = []
    for name, digest in prep['source_sha256'].items():
        path = resolve(name)
        if not path.exists():
            absent.append(name); continue
        if file_hash(path) != digest:
            raise ValueError('Prepared source changed: ' + name)
    if file_hash(data_dir / 'catalog.json') != prep['catalog_sha256']:
        raise ValueError('Catalog changed')
    source = {r['case_id']: r for r in read_json(stage4 / 'stage4_cases.json')} if (stage4 / 'stage4_cases.json').exists() else None
    ids = set(); splits = {}; groups = {}
    for split in ['train', 'validation', 'test']:
        path = data_dir / f'{split}.jsonl'
        if file_hash(path) != prep['data_sha256'][path.name]:
            raise ValueError('Export changed: ' + path.name)
        rows = read_jsonl(path); splits[split] = len(rows)
        for row in rows:
            cid = row['case_id']; m = row['messages']; answer = json.loads(m[2]['content'])
            if cid in ids:
                raise ValueError('Duplicate case: ' + cid)
            ids.add(cid)
            if m[0] != {'role': 'system', 'content': prep['system_prompt']} or m[2]['role'] != 'assistant' or target_errors(answer, catalog):
                raise ValueError('Invalid record: ' + cid)
            if source is not None:
                s = source[cid]
                if s['split'] != split or m[1] != {'role': 'user', 'content': s['input_2']} or answer != s['label_2']:
                    raise ValueError('Input/target changed: ' + cid)
                for name in ['semantic_group', 'source_semantic_group', 'split_group']:
                    key = (name, s[name])
                    if key in groups and groups[key] != split:
                        raise ValueError('Group leakage')
                    groups[key] = split
    if source is not None and ids != set(source):
        raise ValueError('Missing cases')
    ckpt = Path(checkpoint); ckpt_problems = checkpoint_errors(ckpt) if ckpt.exists() else ['checkpoint not found: ' + str(ckpt)]
    manifest = read_json(ckpt / 'run_config.json') if not ckpt_problems else {}
    data_match = manifest.get('data_sha256') == prep['data_sha256'] if manifest else None
    report = dict(valid=True, schema_version=SCHEMA_VERSION, total_cases=len(ids), split_counts=splits,
        upstream_stage4_present=source is not None, upstream_sources_absent=absent, semantic_split_leaks=0,
        thinking_positive_examples=prep['thinking_positive_examples'],
        checkpoint=str(ckpt), checkpoint_problems=ckpt_problems, checkpoint_trained_on_this_data=data_match,
        checkpoint_base_model=manifest.get('base_model'), checkpoint_finetuning=manifest.get('finetuning'),
        data_sha256=prep['data_sha256'], code_sha256={rel(p): sha(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))},
        limitations=['Rule and memory-estimate feasibility, not measured training success.', 'Thinking-positive examples: 0.', 'TTS contains repeated environments with distinct requests grouped in the same split.'])
    write_json(output or ARTIFACTS / 'verification_report.json', report)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('sha256')}, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=DATA_DIR); p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--stage4-dir', type=Path, default=REV4); p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    return verify(args.data_dir, args.checkpoint, args.stage4_dir, args.output)


if __name__ == '__main__':
    main()
