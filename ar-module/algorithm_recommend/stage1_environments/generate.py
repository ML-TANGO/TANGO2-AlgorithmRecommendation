"""Uniform random sampling of four-field cases, without LLM or feasibility filters."""
import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from ..io_utils import read, write, sha, canon, backup_existing
from ..paths import DATASET_DIR, STAGE1, PACKAGE_DIR, rel

VERSION = '6.0'
SOURCE_NAMES = {'task': 'task_list.json', 'gpu': 'gpu_list.json',
                'language': 'language_list.json', 'context_length': 'context_length_list.json'}
OUTPUT_NAMES = ('candidate_lists.json', 'environments.json', 'generation_summary.json',
                'language_counts.json', 'validation_report.json', 'reproducibility_report.json')
IMPLEMENTATION = [Path(__file__).resolve(), Path(__file__).resolve().parent / 'validate.py']


def load_candidates(dataset_dir=DATASET_DIR):
    data = {key: read(Path(dataset_dir) / name) for key, name in SOURCE_NAMES.items()}
    data['gpu'] = data['gpu']['gpus']
    lengths = []
    for value in data['context_length']:
        if isinstance(value, bool):
            raise ValueError('Boolean is not a context length')
        match = re.fullmatch(r'([0-9][0-9,]*)(?:\([0-9]+[Kk]\))?', str(value))
        if not match:
            raise ValueError(f'Unsupported context length: {value!r}')
        n = int(match.group(1).replace(',', ''))
        if n <= 0:
            raise ValueError('Context length must be positive')
        lengths.append(n)
    data['context_length'] = lengths
    for key, values in data.items():
        if not isinstance(values, list) or not values or len({canon(v) for v in values}) != len(values):
            raise ValueError(f'Empty or duplicate candidate list: {key}')
    if not all(isinstance(v, str) and v for key in ('task', 'language') for v in data[key]):
        raise ValueError('Task and language candidates must be strings')
    if len({g['gpu_name'] for g in data['gpu']}) != len(data['gpu']):
        raise ValueError('GPU names must be unique')
    for g in data['gpu']:
        if not {'gpu_name', 'full_name', 'vram_gb'} <= set(g):
            raise ValueError('Incomplete GPU record')
    return data


def generate(output_dir=STAGE1, count=5000, seed=20261024, dataset_dir=DATASET_DIR):
    output_dir = Path(output_dir)
    dataset_dir = Path(dataset_dir)
    source_paths = {rel(dataset_dir / name): dataset_dir / name for name in SOURCE_NAMES.values()}
    sources_before = {name: sha(path) for name, path in source_paths.items()}
    candidates = load_candidates(dataset_dir)
    fields = tuple(SOURCE_NAMES)
    total = math.prod(len(candidates[key]) for key in fields)
    if not 1 <= count <= total:
        raise ValueError(f'count must be in 1..{total}')
    # Sampling distinct product indices is uniform over the complete Cartesian product.
    # No model, label, reasoning, memory policy or task compatibility is read here.
    indices = random.Random(seed).sample(range(total), count)
    cases = []
    for number, index in enumerate(indices, 1):
        selected = {}
        for key in reversed(fields):
            index, offset = divmod(index, len(candidates[key]))
            selected[key] = candidates[key][offset]
        cases.append({'case_id': f'case_v6_{number:05d}', **{key: selected[key] for key in fields}})
    if sources_before != {name: sha(path) for name, path in source_paths.items()}:
        raise RuntimeError('Source lists changed during sampling')
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = read(output_dir / 'generation_summary.json').get('version', 'unknown') if (output_dir / 'generation_summary.json').exists() else 'unknown'
    backup_existing(output_dir, OUTPUT_NAMES, f'v{previous}_before_regeneration')
    for name in ('validation_report.json', 'reproducibility_report.json'):
        (output_dir / name).unlink(missing_ok=True)
    distributions = {
        key: dict(sorted(Counter(str(c[key]) if key == 'context_length' else
                                 c[key]['gpu_name'] if key == 'gpu' else c[key] for c in cases).items()))
        for key in fields}
    write(output_dir / 'candidate_lists.json', candidates)
    write(output_dir / 'environments.json', cases)
    write(output_dir / 'language_counts.json', distributions['language'])
    summary = {
        'version': VERSION, 'case_id_prefix': 'case_v6_', 'seed': seed, 'total_cases': count,
        'case_fields': list(fields), 'candidate_counts': {k: len(v) for k, v in candidates.items()},
        'total_possible_combinations': total, 'unique_combinations': count,
        'sampling_method': 'uniform random sampling without replacement from the full Cartesian product',
        'llm_used': False, 'feasibility_filter_applied': False, 'reasoning_included': False,
        'language_weighting_applied': False, 'fixed_task_or_language_quotas': False,
        'task_counts': distributions['task'], 'gpu_counts': distributions['gpu'],
        'language_counts': distributions['language'], 'context_length_counts': distributions['context_length'],
        'source_sha256': sources_before,
        'implementation_sha256': {rel(p): sha(p) for p in IMPLEMENTATION},
        'artifact_sha256': {name: sha(output_dir / name) for name in
                            ['candidate_lists.json', 'environments.json', 'language_counts.json']},
        'feasibility_status': 'not_assessed; unsuitable combinations are intentionally retained for later evaluation',
        'context_semantics': 'Candidate value only. Do not translate it into image resolution or audio duration.',
        'downstream_status': 'Stage 2/3/4/5 artifacts must be regenerated from these cases; never join by row position.',
    }
    write(output_dir / 'generation_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['version', 'seed', 'total_cases', 'candidate_counts',
                     'total_possible_combinations', 'task_counts', 'language_counts']}, ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=STAGE1)
    parser.add_argument('--count', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261024)
    args = parser.parse_args(argv)
    return generate(args.output_dir, args.count, args.seed)


if __name__ == '__main__':
    main()
