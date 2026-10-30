"""Independent source, schema, uniqueness and uniform sampling reproduction audit."""
import argparse
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

from ..io_utils import read, sha, check
from ..paths import DATASET_DIR, STAGE1, resolve


def validate(output_dir=STAGE1, dataset_dir=DATASET_DIR):
    output_dir = Path(output_dir)
    dataset_dir = Path(dataset_dir)
    rows = read(output_dir / 'environments.json')
    snapshot = read(output_dir / 'candidate_lists.json')
    summary = read(output_dir / 'generation_summary.json')
    fields = ['task', 'gpu', 'language', 'context_length']
    check(summary['version'] == '6.0' and summary['case_fields'] == fields, 'Schema version mismatch')
    expected = {
        'task': read(dataset_dir / 'task_list.json'),
        'gpu': read(dataset_dir / 'gpu_list.json')['gpus'],
        'language': read(dataset_dir / 'language_list.json'),
        'context_length': [int(re.match(r'[0-9,]+', str(x)).group().replace(',', ''))
                           for x in read(dataset_dir / 'context_length_list.json')],
    }
    check(snapshot == expected, 'Candidate snapshot differs from source lists')
    for field in ('source_sha256', 'implementation_sha256'):
        for name, digest in summary[field].items():
            check(sha(resolve(name)) == digest, f'Stale source or implementation: {name}')
    for name, digest in summary['artifact_sha256'].items():
        check(sha(output_dir / name) == digest, f'Changed artifact: {name}')
    check(len(rows) == summary['total_cases'], 'Wrong case count')
    identities = set()
    for index, row in enumerate(rows, 1):
        check(set(row) == {'case_id', *fields}, f'Unexpected case fields at row {index}')
        check(row['case_id'] == f'case_v6_{index:05d}', 'Case ID mismatch')
        for key in fields:
            check(row[key] in expected[key], f'Outside catalog: {key}')
        check(type(row['context_length']) is int, 'Context length is not an integer')
        identities.add((row['task'], row['gpu']['gpu_name'], row['language'], row['context_length']))
    check(len(identities) == len(rows), 'Duplicate four-field combinations')
    # Build the product directly, independently of the generator's mixed-radix decoding.
    product = list(itertools.product(*(expected[key] for key in fields)))
    check(len(product) == summary['total_possible_combinations'], 'Product size mismatch')
    sampled = random.Random(summary['seed']).sample(product, len(rows))
    check(sampled == [tuple(row[key] for key in fields) for row in rows], 'Not reproducible uniform sampling')
    for key, out in [('task', 'task_counts'), ('gpu', 'gpu_counts'), ('language', 'language_counts'), ('context_length', 'context_length_counts')]:
        counts = Counter(row[key]['gpu_name'] if key == 'gpu' else str(row[key]) for row in rows)
        check(dict(counts) == summary[out], f'Wrong distribution: {key}')
    check(read(output_dir / 'language_counts.json') == summary['language_counts'], 'Language counts mismatch')
    for key in ('llm_used', 'feasibility_filter_applied', 'reasoning_included', 'language_weighting_applied', 'fixed_task_or_language_quotas'):
        check(summary[key] is False, f'Unexpected policy: {key}')
    report = {
        'valid': True, 'version': '6.0', 'cases_verified': len(rows),
        'unique_four_field_combinations': len(identities), 'reasoning_fields': 0,
        'catalog_violations': 0, 'full_product_sampling_reproduced': True,
        'feasibility_evaluated': False,
        'observed_candidate_counts': {key: len({row[key]['gpu_name'] if key == 'gpu' else row[key] for row in rows}) for key in fields},
        'artifact_sha256': {name: sha(output_dir / name) for name in
                            ['candidate_lists.json', 'environments.json', 'generation_summary.json', 'language_counts.json']},
    }
    (output_dir / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=STAGE1)
    return validate(parser.parse_args(argv).output_dir)


if __name__ == '__main__':
    main()
