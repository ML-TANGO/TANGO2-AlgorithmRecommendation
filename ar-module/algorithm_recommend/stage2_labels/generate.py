"""Stage 2 v7: rule-based labels for all unchanged four-field Stage 1 v6 cases."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .feasibility_policy import Evaluator, POLICY
from ..io_utils import read, write, sha, backup_existing
from ..paths import DATASET_DIR, SOURCE_CONFIGS, STAGE1, STAGE2, PACKAGE_DIR, rel, resolve

FILES = ('labeled_cases.json', 'labels.json', 'model_profiles.json', 'labeling_policy.json',
         'training_assumptions.json', 'labeling_summary.json', 'source_manifest.json',
         'validation_report.json', 'reproducibility_report.json')


def source_paths(stage1=STAGE1, dataset_dir=DATASET_DIR):
    paths = [Path(stage1) / n for n in ['environments.json', 'candidate_lists.json', 'generation_summary.json', 'validation_report.json']]
    paths += [Path(dataset_dir) / n for n in ['model_list.json', 'training_algorithm_list.json', 'metric_list.json', 'model_task_capabilities.json']]
    paths += sorted(SOURCE_CONFIGS.glob('*.json'))
    here = PACKAGE_DIR / 'stage2_labels'
    paths += [here / n for n in ['labeling_core.py', 'feasibility_policy.py', 'generate.py', 'validate.py']]
    return paths


def generate(output_dir=STAGE2, stage1=STAGE1, dataset_dir=DATASET_DIR):
    output_dir = Path(output_dir); stage1 = Path(stage1)
    paths = source_paths(stage1, dataset_dir); before = {rel(p): sha(p) for p in paths}
    cases = read(stage1 / 'environments.json')
    report = read(stage1 / 'validation_report.json')
    if report.get('valid') is not True or report['artifact_sha256']['environments.json'] != sha(stage1 / 'environments.json'):
        raise ValueError('Stage 1 validation missing or stale')
    for key in ('source_sha256', 'implementation_sha256'):
        for name, digest in read(stage1 / 'generation_summary.json')[key].items():
            if sha(resolve(name)) != digest:
                raise ValueError('Stage 1 provenance changed: ' + name)
    if len(cases) != 5000 or len({r['case_id'] for r in cases}) != 5000:
        raise ValueError('Expected all 5000 current Stage 1 cases')
    allowed_fields = {'case_id', 'task', 'gpu', 'language', 'context_length'}
    if any(set(r) != allowed_fields or not r['case_id'].startswith('case_v6_') for r in cases):
        raise ValueError('Expected four-field Stage 1 v6 inputs, without reasoning')
    evaluator = Evaluator(dataset_dir); rows = []
    for item in cases:
        assessed = evaluator.evaluate(item)
        rows.append(dict(case={k: v for k, v in item.items() if k != 'case_id'}, **assessed))
    if before != {rel(p): sha(p) for p in paths}:
        raise RuntimeError('Sources changed while generating')
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = read(output_dir / 'labeling_summary.json').get('version', 'unknown') if (output_dir / 'labeling_summary.json').exists() else 'unknown'
    backup_existing(output_dir, FILES, f'v{previous}_before_regeneration')
    for n in ('validation_report.json', 'reproducibility_report.json'):
        (output_dir / n).unlink(missing_ok=True)
    assumptions = dict(version='1.0', origin='system_calculation_policy', user_requested=False,
        scope='All Stage 2 v7 resource estimates', workload=POLICY['workload'],
        memory_policy={k: POLICY[k] for k in ('positive_memory_fraction', 'memory', 'negative_memory_rule')},
        downstream_instruction='Keep these assumptions separate from case fields and natural-language user requirements. Do not claim empirical training success.')
    data = {'labeled_cases.json': rows,
            'labels.json': [{k: r[k] for k in ('case_id', 'label', 'training_status', 'training_possible', 'impossibility_reason', 'reason_codes', 'review_notes', 'rationale', 'assumptions_ref')} for r in rows],
            'model_profiles.json': evaluator.profiles, 'labeling_policy.json': POLICY,
            'training_assumptions.json': assumptions,
            'source_manifest.json': {name: {'path': name, 'sha256': digest} for name, digest in before.items()}}
    for name, value in data.items():
        write(output_dir / name, value)
    bytask = defaultdict(Counter)
    for row in rows:
        bytask[row['case']['task']][row['training_status']] += 1
    possible = [r for r in rows if r['training_status'] == 'possible']
    summary = dict(version='7.0', source_stage1_version='6.0', total_cases=len(rows), source_cases_preserved=True,
        case_fields=['task', 'gpu', 'language', 'context_length'], label_fields=['model', 'algorithm', 'evaluation_metric'],
        status_counts=dict(Counter(r['training_status'] for r in rows)), task_status_counts={k: dict(v) for k, v in bytask.items()},
        model_counts=dict(Counter(r['label']['model'] for r in possible)),
        algorithm_counts=dict(Counter(r['label']['algorithm'] for r in possible)),
        metric_counts=dict(Counter(r['label']['evaluation_metric'] for r in rows)),
        label_method='codex_authored_rules_no_per_case_llm_call', per_case_gpt_api_called=False,
        reasoning_input_or_decision_used=False, empirically_validated_cases=0,
        unassigned_model_algorithm_cases=len(rows) - len(possible),
        null_semantics='Unverified or unsupported combinations retain case and suggested metric but have null model/algorithm; never fabricate a feasible label.',
        source_sha256=before,
        artifact_sha256={name: sha(output_dir / name) for name in data},
        downstream_regeneration_required=['stage_3', 'stage_4', 'stage_5'])
    write(output_dir / 'labeling_summary.json', summary)
    print(json.dumps({k: summary[k] for k in ['version', 'total_cases', 'status_counts', 'algorithm_counts', 'task_status_counts', 'label_method']}, ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=STAGE2)
    p.add_argument('--stage1-dir', type=Path, default=STAGE1)
    args = p.parse_args(argv)
    return generate(args.output_dir, args.stage1_dir)


if __name__ == '__main__':
    main()
