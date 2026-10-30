"""Export verified revised Stage 4 recommendation pairs as supervised chat data."""
import argparse
import json
import shutil
from pathlib import Path

from .common import SCHEMA_VERSION, SYSTEM_PROMPT, read_json, read_jsonl, write_json, json_text, file_hash, split_input, target_errors
from ..io_utils import now_tag, write_jsonl
from ..paths import DATASET_DIR, DATA_DIR, REV4, ARTIFACTS, rel, resolve


def build_catalog(dataset_dir=DATASET_DIR):
    return {'models': [r['model_id'] for r in read_json(dataset_dir / 'model_list.json')['models']],
            'algorithms': read_json(dataset_dir / 'training_algorithm_list.json'),
            'metrics': sorted({v['name'] for task in read_json(dataset_dir / 'metric_list.json').values() for v in task.values()})}


def prepare(stage4=REV4, output=DATA_DIR, dataset_dir=DATASET_DIR):
    stage4 = Path(stage4).resolve(); output = Path(output).resolve(); dataset_dir = Path(dataset_dir)
    report = read_json(stage4 / 'validation_report.json'); summary = read_json(stage4 / 'generation_summary.json')
    if report.get('valid') is not True:
        raise ValueError('Stage 4 verification did not pass')
    for mapping in [report['artifact_sha256'], summary['artifact_sha256'], summary['training_data_sha256']]:
        for name, digest in mapping.items():
            if file_hash(stage4 / name) != digest:
                raise ValueError('Stage 4 artifact changed: ' + name)
    for name, item in read_json(stage4 / 'source_manifest.json').items():
        if file_hash(resolve(name)) != item['sha256']:
            raise ValueError('Stage 4 source changed: ' + name)
    catalog = build_catalog(dataset_dir)
    master = {r['case_id']: r for r in read_json(stage4 / 'stage4_cases.json')}; datasets = {}; seen = set(); groups = {}
    for split in ['train', 'validation', 'test']:
        records = []
        for row in read_jsonl(stage4 / 'training_data' / f'{split}.jsonl'):
            cid = row['case_id']; source = master[cid]
            if cid in seen:
                raise ValueError('Duplicate/cross-split case: ' + cid)
            seen.add(cid); group = source['semantic_group']
            if group in groups and groups[group] != split:
                raise ValueError('Semantic group leakage')
            groups[group] = split
            if not source['supervised_recommendation_eligible'] or source['split'] != split:
                raise ValueError('Ineligible or wrong-split case: ' + cid)
            if row['input'] != source['input_2'] or row['output'] != source['label_2']:
                raise ValueError('Source pair mismatch: ' + cid)
            split_input(row['input']); errors = target_errors(row['output'], catalog)
            if errors:
                raise ValueError((cid, errors))
            records.append({'case_id': cid, 'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': row['input']}, {'role': 'assistant', 'content': json_text(row['output'])}]})
        datasets[split] = records
    expected = {cid for cid, r in master.items() if r['supervised_recommendation_eligible']}
    if seen != expected:
        raise ValueError('Eligible records missing from Stage 4 training export')
    if output.exists() and any(output.iterdir()):
        backup = ARTIFACTS / 'backup' / ('data_before_' + now_tag())
        backup.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(output, backup)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in datasets.items():
        write_jsonl(output / f'{split}.jsonl', rows)
    write_json(output / 'catalog.json', catalog)
    sources = [stage4 / n for n in ['stage4_cases.json', 'generation_summary.json', 'validation_report.json', 'source_manifest.json']]
    sources += [stage4 / 'training_data' / f'{s}.jsonl' for s in datasets]
    sources += [dataset_dir / n for n in ['model_list.json', 'training_algorithm_list.json', 'metric_list.json']]
    metadata = {'schema_version': SCHEMA_VERSION, 'system_prompt': SYSTEM_PROMPT, 'counts': {s: len(r) for s, r in datasets.items()},
                'input_2_and_label_2_preserved': True, 'excluded_stage4_cases': len(master) - len(seen), 'semantic_groups': len(groups),
                'source_sha256': {rel(p): file_hash(p) for p in sources}, 'data_sha256': {f'{s}.jsonl': file_hash(output / f'{s}.jsonl') for s in datasets},
                'catalog_sha256': file_hash(output / 'catalog.json'), 'thinking_positive_examples': sum(json.loads(r['messages'][2]['content'])['thinking']['recommended'] for rows in datasets.values() for r in rows)}
    write_json(output / 'preparation_report.json', metadata)
    print(json.dumps({k: metadata[k] for k in ['counts', 'excluded_stage4_cases', 'thinking_positive_examples', 'input_2_and_label_2_preserved']}, ensure_ascii=False, indent=2))
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage4-dir', type=Path, default=REV4); p.add_argument('--output-dir', type=Path, default=DATA_DIR)
    args = p.parse_args(argv)
    return prepare(args.stage4_dir, args.output_dir)


if __name__ == '__main__':
    main()
