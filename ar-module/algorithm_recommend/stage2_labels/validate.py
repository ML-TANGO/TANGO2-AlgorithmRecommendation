"""Audit label catalogs, unchanged cases, explicit assumptions and candidate decisions."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .feasibility_policy import Evaluator, POLICY
from ..io_utils import read, sha, check
from ..paths import DATASET_DIR, STAGE1, STAGE2, resolve


def validate(output_dir=STAGE2, stage1=STAGE1, dataset_dir=DATASET_DIR):
    output_dir = Path(output_dir); stage1 = Path(stage1)
    rows = read(output_dir / 'labeled_cases.json'); labels = read(output_dir / 'labels.json')
    cases = read(stage1 / 'environments.json'); summary = read(output_dir / 'labeling_summary.json')
    check(len(rows) == len(labels) == len(cases) == 5000, 'Case count mismatch')
    check(len({r['case_id'] for r in rows}) == 5000, 'Duplicate IDs')
    for name, digest in summary['source_sha256'].items():
        check(sha(resolve(name)) == digest, 'Changed source: ' + name)
    for name, item in read(output_dir / 'source_manifest.json').items():
        check(item == {'path': name, 'sha256': summary['source_sha256'][name]}, 'Manifest mismatch')
    for name, digest in summary['artifact_sha256'].items():
        check(sha(output_dir / name) == digest, 'Changed artifact: ' + name)
    check(read(output_dir / 'labeling_policy.json') == POLICY, 'Policy snapshot mismatch')
    assumptions = read(output_dir / 'training_assumptions.json')
    check(assumptions['origin'] == 'system_calculation_policy' and assumptions['user_requested'] is False, 'Assumption origin')
    check(assumptions['workload'] == POLICY['workload'], 'Shared workload mismatch')
    model_ids = {m['model_id'] for m in read(dataset_dir / 'model_list.json')['models']}
    algorithms = set(read(dataset_dir / 'training_algorithm_list.json'))
    metric_catalog = read(dataset_dir / 'metric_list.json')
    e = Evaluator(dataset_dir); profiles = {p['model_id']: p for p in read(output_dir / 'model_profiles.json')}
    check(set(profiles) == model_ids, 'Model profile/catalog mismatch')
    check(list(profiles.values()) == e.profiles, 'Model profile drift')
    counts = Counter(); task_counts = defaultdict(Counter)
    for row, case, exported in zip(rows, cases, labels):
        cid = case['case_id']; label = row['label']; task = case['task']; status = row['training_status']
        check(row['case_id'] == exported['case_id'] == cid, f'{cid}: ID mismatch')
        check(row['case'] == {k: v for k, v in case.items() if k != 'case_id'}, f'{cid}: case changed')
        check(set(row['case']) == {'task', 'gpu', 'language', 'context_length'}, f'{cid}: hidden case field')
        check(set(label) == {'model', 'algorithm', 'evaluation_metric'}, f'{cid}: label schema')
        check('thinking' not in row and 'reasoning' not in row, f'{cid}: unexpected reasoning decision')
        check(exported == {k: row[k] for k in exported}, f'{cid}: labels export drift')
        check(label['evaluation_metric'] in {m['name'] for m in metric_catalog[task].values()}, f'{cid}: metric outside task catalog')
        check(row['assumptions_ref'] == 'training_assumptions.json', f'{cid}: shared policy reference')
        check(row['label_method'] == 'codex_authored_rules_no_per_case_llm_call' and row['empirically_validated'] is False, f'{cid}: unsupported provenance claim')
        fresh = e.evaluate(case)
        check(all(row.get(k) == v for k, v in fresh.items()) and set(row) == set(fresh) | {'case'}, f'{cid}: rule reproduction mismatch')
        checks = row['candidate_checks']; candidate_map = {c['model']: c for c in checks}
        check(set(candidate_map) == {p['model_id'] for p in e.profiles if task in p['target_tasks']}, f'{cid}: missing candidate checks')
        if status == 'possible':
            check(row['training_possible'] is True and row['impossibility_reason'] is None, f'{cid}: false status')
            check(label['model'] in model_ids and label['algorithm'] in algorithms, f'{cid}: outside model/algorithm lists')
            p = profiles[label['model']]; algo = label['algorithm']; check(algo in p['algorithms'] and task in p['target_tasks'], f'{cid}: unsupported pairing')
            selected = candidate_map[p['model_id']]
            check(selected['status'] == 'possible' and not selected['reason_codes'] and not selected['review_codes'], f'{cid}: rejected candidate selected')
            check(algo in selected['fitting_algorithms'], f'{cid}: unsupported memory selection')
            check(row['resource_check']['estimated_memory_gb'] <= row['resource_check']['positive_budget_gb'] + 1e-6, f'{cid}: over budget')
            check(row['resource_check']['gpu_vram_gb'] == case['gpu']['vram_gb'], f'{cid}: GPU changed')
            if p['languages'] is not None:
                check(case['language'] in p['languages'], f'{cid}: language outside profile')
            if p['context_applicable']:
                seq = case['context_length']
                check(p['native_context'] is None or seq + p['extra_tokens'] <= p['native_context'], f'{cid}: native length exceeded')
                check(p['max_text_context'] is None or seq <= p['max_text_context'], f'{cid}: reviewed workload exceeded')
            check(row['selected_workload_assumptions'] == {'origin': 'system_calculation_policy', 'user_requested': False, 'values': p['workload']}, f'{cid}: wrong workload')
        else:
            check(label['model'] is None and label['algorithm'] is None, f'{cid}: fabricated recommendation')
            check(not any(c['status'] == 'possible' for c in checks), f'{cid}: viable candidate omitted')
            if status == 'not_possible':
                check(row['training_possible'] is False and bool(row['impossibility_reason']), f'{cid}: missing failure reason')
                check(checks and all(c['status'] == 'not_possible' and c['reason_codes'] for c in checks), f'{cid}: unjustified impossibility')
            else:
                check(status == 'needs_review' and row['training_possible'] is None and row['impossibility_reason'] is None, f'{cid}: uncertain case falsely impossible')
                check(bool(row.get('review_reason')), f'{cid}: missing review explanation')
        counts[status] += 1; task_counts[task][status] += 1
    check(dict(counts) == summary['status_counts'], 'Status counts mismatch')
    check({k: dict(v) for k, v in task_counts.items()} == summary['task_status_counts'], 'Task counts mismatch')
    report = dict(valid=True, version='7.0', cases_verified=len(rows), source_case_changes=0, catalog_violations=0,
        status_counts=dict(counts), case_label_pairs_verified=len(rows), reasoning_inputs_or_decisions=0,
        possible_labels_checked_for_support_and_budget=counts['possible'],
        empirical_training_performed=False, per_case_llm_calls=0,
        artifact_sha256={name: sha(output_dir / name) for name in ['labeled_cases.json', 'labels.json', 'model_profiles.json', 'training_assumptions.json', 'labeling_summary.json']})
    (output_dir / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2)); return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=STAGE2)
    p.add_argument('--stage1-dir', type=Path, default=STAGE1)
    args = p.parse_args(argv)
    return validate(args.output_dir, args.stage1_dir)


if __name__ == '__main__':
    main()
