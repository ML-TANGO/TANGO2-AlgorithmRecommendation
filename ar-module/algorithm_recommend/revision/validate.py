"""Re-evaluate revised support, memory, provenance and training splits (Stage 2/3/4 revised)."""
import argparse
import collections
import copy
import json
from pathlib import Path

from .revision_core import JointPolicy, key, gpu_key
from .sources import load_audited_requests
from ..stage3_hyperparameters.generate import propose
from ..stage4_requests.request_checks import inspect
from ..io_utils import read, write, sha, digest, check
from ..paths import DATASET_DIR, STAGE2, REV2, REV3, REV4, resolve


def validate(rev2=REV2, rev3=REV3, rev4=REV4, stage2=STAGE2, dataset_dir=DATASET_DIR):
    rev2 = Path(rev2); rev3 = Path(rev3); rev4 = Path(rev4); stage2 = Path(stage2)
    policy = JointPolicy(dataset_dir); rows = read(rev2 / 'labeled_cases.json'); outputs = read(rev3 / 'output_1.json'); combined = read(rev3 / 'hyperparameter_cases.json'); changes = read(rev2 / 'case_changes.json')
    source = {r['case_id']: r for r in read(stage2 / 'labeled_cases.json')}; tasks = read(dataset_dir / 'task_list.json')
    seen = set(); original_inputs = read(rev3 / 'input_1.json'); a100 = 0
    for row, out, joined, change, inp in zip(rows, outputs, combined, changes, original_inputs):
        cid = row['case_id']; case = row['case']; label = row['label']; origin = source[row['source_case_id']]
        check(cid not in seen, 'Duplicate case ID'); seen.add(cid)
        check(case['task'] in tasks and case['task'] == origin['case']['task'], cid + ': task changed')
        check(case['gpu'] in policy.gpus and case['language'] in policy.languages and case['context_length'] in policy.contexts, cid + ': outside input catalog')
        check(set(case) == {'task', 'gpu', 'language', 'context_length'}, cid + ': unexpected case fields')
        check(label['model'] in policy.profiles and label['algorithm'] in policy.evaluator.allowed, cid + ': outside label catalog')
        metrics = {v['name'] for v in policy.evaluator.metrics[case['task']].values()}; check(label['evaluation_metric'] in metrics, cid + ': metric/task mismatch')
        effective = policy.effective_case(case); evaluated = policy.evaluator.evaluate(effective)
        chosen = next(c for c in evaluated['candidate_checks'] if c['model'] == label['model'])
        check(chosen['status'] == 'possible' and label['algorithm'] in chosen['fitting_algorithms'], cid + ': support/budget gate failed')
        estimate = chosen['algorithm_estimates_gb'][label['algorithm']]
        check(row['resource_check']['estimated_memory_gb'] == estimate, cid + ': Stage 2 estimate mismatch')
        check(row['resource_check']['positive_budget_gb'] == .75 * effective['gpu']['vram_gb'], cid + ': resource budget loosened')
        fresh = propose(row, policy.hp, policy.profiles, policy.contexts)
        check(out == fresh, cid + ': Stage 3 recomputation mismatch')
        check(out['output_1'] is not None and out['stage3_status'] == 'recommended_draft' and out['resource_check']['within_budget'], cid + ': unresolved Stage 3 output')
        check(out['resource_check']['estimated_memory_gb'] <= out['resource_check']['positive_budget_gb'], cid + ': scoped memory too large')
        check(out['empirically_validated'] is False and out['ready_to_execute'] is False, cid + ': unsupported empirical claim')
        check(inp == dict(case_id=cid, input_1=dict(case=case, label=label)), cid + ': input_1 mismatch')
        check(joined == dict(**out, input_1=inp['input_1']), cid + ': combined mismatch')
        check(change['original_case'] == origin['case'] and change['original_label'] == origin['label'], cid + ': lineage mismatch')
        check(change['revised_case'] == case and change['revised_label'] == label, cid + ': revised lineage mismatch')
        actual = [f for f in ['task', 'gpu', 'language', 'context_length'] if origin['case'][f] != case[f]]
        check(actual == change['changed_case_fields'], cid + ': undisclosed case change')
        if gpu_key(case['gpu']) == 'A100':
            a100 += 1
            for capacity in [40, 80]:
                other = copy.deepcopy(row); other['case']['gpu'] = copy.deepcopy(policy.gpu_map[f'A100 {capacity}GB'])
                other['resource_check']['positive_budget_gb'] = .75 * capacity
                result = propose(other, policy.hp, policy.profiles, policy.contexts)
                check(result['stage3_status'] == 'recommended_draft' and result['output_1'] == out['output_1'], cid + ': configuration does not cover both A100 capacities')
    check(len(rows) == len(outputs) == len(combined) == len(changes) == len(seen) == 5000, 'Count mismatch')
    for path, h in read(rev2 / 'original_preservation.json').items():
        check(sha(resolve(path)) == h, 'Upstream file changed since revision: ' + path)
    report = dict(valid=True, version='revised-1.0', cases_verified=5000, stage2_possible=5000, stage3_recommended_draft=5000,
        all_catalog_and_language_context_support_checks_passed=True, stage3_scope_memory_recomputed=True, positive_memory_fraction=.75,
        a100_common_configuration_cases_verified=a100, upstream_files_unchanged=True, empirical_training_success_claimed=False,
        unique_case_tuples=len({key(r['case']) for r in rows}), repeated_case_records=5000 - len({key(r['case']) for r in rows}),
        artifact_sha256={'labeled_cases.json': sha(rev2 / 'labeled_cases.json')})
    write(rev2 / 'validation_report.json', report)
    report3 = copy.deepcopy(report); report3['artifact_sha256'] = {n: sha(rev3 / n) for n in ['input_1.json', 'output_1.json', 'hyperparameter_cases.json']}; write(rev3 / 'validation_report.json', report3)
    r4 = read(rev4 / 'stage4_cases.json'); summary4 = read(rev4 / 'generation_summary.json')
    orig4 = load_audited_requests()
    newgroups = collections.defaultdict(set); oldgroups = collections.defaultdict(set)
    for row, s2, s3 in zip(r4, rows, combined):
        cid = row['case_id']; check(cid == s2['case_id'], cid + ': Stage4 order mismatch')
        check(row['input_1'] == s3['input_1'] and row['source_output_1'] == s3['output_1'], cid + ': Stage4 source mismatch')
        check(not inspect(row['input_2'], s2['case'], row['expression_metadata']), cid + ': invalid natural request')
        answer = row['label_2']; check(answer['status'] == 'conditional_recommendation_draft' and row['supervised_recommendation_eligible'], cid + ': not eligible')
        rec = answer['recommendation']; check(rec is not None, cid + ': null recommendation')
        for field in ['model', 'algorithm', 'evaluation_metric']:
            check(rec[field] == s2['label'][field], cid + ': label mismatch ' + field)
        for field in ['hyperparameters', 'training_plan', 'backend_requirements', 'parameter_rationale']:
            check(rec[field] == s3['output_1'][field], cid + ': output mismatch ' + field)
        capacity = answer['recommendation_assumptions']['gpu_capacity']
        check(capacity['value_gb'] == s2['gpu_resource_basis']['budget_capacity_gb'] and capacity['requires_confirmation'] is False, cid + ': capacity policy mismatch')
        check(answer['empirically_validated'] is False and answer['ready_to_execute'] is False, cid + ': fake validation')
        original = orig4[row['source_case_id']]
        check(row['transformation']['source_input_sha256'] == digest(original['input_2']), cid + ': original request mismatch')
        newgroups[row['semantic_group']].add(row['split']); oldgroups[row['source_semantic_group']].add(row['split'])
    check(len(r4) == len({r['input_2'] for r in r4}) == 5000, 'Duplicate/missing revised request')
    check(all(len(s) == 1 for s in list(newgroups.values()) + list(oldgroups.values())), 'Source or revised semantic group leakage')
    for split in ['train', 'validation', 'test']:
        got = [json.loads(x) for x in (rev4 / 'training_data' / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        expected = [dict(case_id=r['case_id'], input=r['input_2'], output=r['label_2']) for r in r4 if r['split'] == split]
        check(got == expected, 'Training export mismatch: ' + split)
    for path, item in read(rev4 / 'source_manifest.json').items():
        check(sha(resolve(path)) == item['sha256'], 'Stage4 source changed: ' + path)
    for mapping in [summary4['artifact_sha256'], summary4['training_data_sha256']]:
        for path, h in mapping.items():
            check(sha(rev4 / path) == h, 'Stage4 artifact changed: ' + path)
    report4 = dict(valid=True, version='revised-1.0', cases_verified=5000, supervised_recommendation_records=5000,
                   upstream_files_unchanged=True, exact_duplicate_requests=0, source_and_revised_group_split_leaks=0,
                   validation_method='Original semantic audits reused; controlled field edits, source comparisons, full rule checks and revised sample inspection. No new per-case LLM audit.',
                   artifact_sha256={n: sha(rev4 / n) for n in ['input_2.json', 'label_2.json', 'stage4_cases.json', 'generation_summary.json']})
    write(rev4 / 'validation_report.json', report4)
    print(json.dumps(dict(stage2=report, stage4=report4), ensure_ascii=False, indent=2))
    return report, report4


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rev2-dir', type=Path, default=REV2); p.add_argument('--rev3-dir', type=Path, default=REV3); p.add_argument('--rev4-dir', type=Path, default=REV4)
    args = p.parse_args(argv)
    return validate(args.rev2_dir, args.rev3_dir, args.rev4_dir)


if __name__ == '__main__':
    main()
