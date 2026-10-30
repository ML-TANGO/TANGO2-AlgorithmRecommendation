"""Build 5,000 revised Stage 2/3 pairs that jointly pass the unchanged support/resource policy.

Cases that cannot be labelled as-is keep their task and are moved, by distance,
to the nearest catalog combination that passes both Stage 2 and Stage 3.
"""
import argparse
import collections
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

from .revision_core import JointPolicy, key, module_spec
from .sources import audited_requests_path, load_audited_requests
from ..io_utils import read, write, sha
from ..paths import DATASET_DIR, SOURCE_CONFIGS, STAGE1, STAGE2, STAGE3, REV2, REV3, TTS_FEASIBLE_SPACE, PACKAGE_DIR, rel

VERSION = 'revised-1.0'


def distance(old, new):
    fields = ['gpu', 'language', 'context_length']; changes = sum(old[k] != new[k] for k in fields)
    return (changes, old['language'] != new['language'], old['context_length'] != new['context_length'],
            abs(math.log2(new['context_length'] / old['context_length'])), abs(new['gpu']['vram_gb'] - old['gpu']['vram_gb']), new['gpu']['vram_gb'])


def upstream_paths(stage1=STAGE1, stage2=STAGE2, stage3=STAGE3, dataset_dir=DATASET_DIR):
    files = sorted(p for p in Path(dataset_dir).glob('*.json'))
    files += [Path(stage1) / n for n in ['environments.json', 'candidate_lists.json', 'generation_summary.json', 'validation_report.json']]
    files += [Path(stage2) / n for n in ['labeled_cases.json', 'model_profiles.json', 'labeling_policy.json', 'training_assumptions.json', 'labeling_summary.json', 'validation_report.json']]
    files += [Path(stage3) / n for n in ['hyperparameter_cases.json', 'module_specs.json', 'training_assumptions.json', 'generation_summary.json', 'validation_report.json']]
    files += sorted(SOURCE_CONFIGS.glob('*.json')) + [TTS_FEASIBLE_SPACE, audited_requests_path()]
    return sorted(set(p for p in files if p.exists()))


def generate(rev2=REV2, rev3=REV3, stage1=STAGE1, stage2=STAGE2, stage3=STAGE3, dataset_dir=DATASET_DIR):
    rev2 = Path(rev2); rev3 = Path(rev3); stage2 = Path(stage2); stage3 = Path(stage3)
    upstream = upstream_paths(stage1, stage2, stage3, dataset_dir)
    preservation = {rel(p): sha(p) for p in upstream}
    if read(stage2 / 'validation_report.json').get('valid') is not True or read(stage3 / 'validation_report.json').get('valid') is not True:
        raise ValueError('Stage 2/3 must be validated before revision')
    p = JointPolicy(dataset_dir); original = read(stage2 / 'labeled_cases.json'); old4 = load_audited_requests()
    direct = {}; used = collections.Counter()
    for row in original:
        choice = p.choose(row['case'], row['label'])
        if choice:
            direct[row['case_id']] = choice; used[key(row['case'])] += 1
    print('Same-case feasible after joint selection:', len(direct), flush=True)
    tts_space = read(TTS_FEASIBLE_SPACE); results = []; outputs = []; lineage = []
    lang_order = p.languages
    for i, src in enumerate(original, 1):
        old = src['case']; cid = src['case_id']; newid = f'case_r1_{i:05d}'; selection = direct.get(cid); duplicate = False
        if selection is None:
            if old['task'] == 'TTS':
                pool = tts_space
            else:
                # Preserve task and never increase the requested context to make a
                # case pass. Search every candidate field combination by distance.
                pool = [dict(task=old['task'], gpu=g, language=lang, context_length=n)
                        for g, lang, n in itertools.product(p.gpus, lang_order, [x for x in p.contexts if x <= old['context_length']])]
            ranked = sorted(pool, key=lambda c: (distance(old, c), hashlib.sha256((cid + repr(key(c))).encode()).hexdigest()))
            for candidate in ranked:
                if used[key(candidate)]:
                    continue
                selection = p.choose(candidate)
                if selection:
                    break
            else:
                # Limited specialist space: retain task counts and disclose repeated
                # environments as language paraphrases, never hide duplication.
                for candidate in sorted(ranked, key=lambda c: (used[key(c)], distance(old, c))):
                    selection = p.choose(candidate)
                    if selection:
                        duplicate = used[key(candidate)] > 0; break
                else:
                    raise ValueError('No feasible catalog combination for task ' + old['task'])
            used[key(selection[0]['case'])] += 1
        row, out = copy.deepcopy(selection); row['case_id'] = newid; row['source_case_id'] = cid; out['case_id'] = newid
        changed = [f for f in ['task', 'gpu', 'language', 'context_length'] if old[f] != row['case'][f]]
        if 'task' in changed:
            raise AssertionError('Task cannot change')
        row['revision'] = VERSION; row['joint_stage3_check'] = copy.deepcopy(out['resource_check'])
        row['feasibility_basis'] = 'documented_support_and_stage3_scoped_budget_passed_not_empirically_measured'
        record = dict(case_id=newid, source_case_id=cid, source_stage4_status=old4[cid]['label_2_status'],
            original_case=old, original_label=src['label'], revised_case=row['case'], revised_label=row['label'],
            changed_case_fields=changed, label_changed=src['label'] != row['label'],
            resolution='case_fields_revised_within_catalog' if changed else 'same_case_jointly_valid_configuration',
            repeated_environment_due_to_finite_task_catalog=duplicate,
            source_reason=old4[cid]['label_2_reason'], gpu_resource_basis=row['gpu_resource_basis'])
        row['revision_trace'] = dict(source_case_id=cid, changed_case_fields=changed, lineage_ref='case_changes.json')
        results.append(row); outputs.append(out); lineage.append(record)
        if i % 1000 == 0:
            print('Revised pairs', i, flush=True)
    if len(results) != 5000 or any(o['stage3_status'] != 'recommended_draft' for o in outputs):
        raise ValueError('Not all pairs passed')
    write(rev2 / 'labeled_cases.json', results); write(rev2 / 'labels.json', [dict(case_id=r['case_id'], label=r['label'], training_possible=True) for r in results])
    write(rev2 / 'environments.json', [dict(case_id=r['case_id'], **r['case']) for r in results]); write(rev2 / 'case_changes.json', lineage)
    write(rev2 / 'model_profiles.json', list(p.profiles.values()))
    assumptions = read(stage2 / 'training_assumptions.json'); assumptions['revision'] = VERSION; assumptions['scope'] = 'Revised Stage 2/3 joint evaluation; unchanged support/resource constraints'
    assumptions['gpu_capacity_policy'] = 'For capacity-omitted A100 aliases, one configuration must fit both 40GB and 80GB; use 40GB for selection and budgeting.'
    write(rev2 / 'training_assumptions.json', assumptions)
    policy = read(stage2 / 'labeling_policy.json'); policy['revision'] = VERSION; policy['joint_stage3_gate'] = True; policy['case_revision_authorized'] = True; policy['runtime_training_performed'] = False; write(rev2 / 'labeling_policy.json', policy)
    inputs = [dict(case_id=r['case_id'], input_1=dict(case=r['case'], label=r['label'])) for r in results]
    combined = [dict(**o, input_1=inp['input_1']) for o, inp in zip(outputs, inputs)]
    for name, data in {'input_1.json': inputs, 'output_1.json': outputs, 'hyperparameter_cases.json': combined, 'review_cases.json': [],
                       'not_trainable_cases.json': [], 'hyper_parameter_candidates.json': p.hp,
                       'module_specs.json': {r['label']['model']: module_spec(r['label']['model']) for r in results if r['label']['algorithm'] in {'LoRA', 'QLoRA'}}}.items():
        write(rev3 / name, data)
    s3assumptions = read(stage3 / 'training_assumptions.json'); s3assumptions.update(revision=VERSION, source=rel(rev2 / 'training_assumptions.json'), source_sha256=sha(rev2 / 'training_assumptions.json'), gpu_capacity_policy=assumptions['gpu_capacity_policy']); write(rev3 / 'training_assumptions.json', s3assumptions)
    here = PACKAGE_DIR / 'revision'
    manifest = {rel(path): dict(sha256=sha(path)) for path in upstream + [here / 'revision_core.py', here / 'generate.py']}
    write(rev2 / 'source_manifest.json', manifest); write(rev3 / 'source_manifest.json', manifest)
    summary = dict(version=VERSION, total_cases=5000, training_status_counts={'possible': 5000}, stage3_status_counts={'recommended_draft': 5000},
        unchanged_case_fields=sum(not r['changed_case_fields'] for r in lineage), changed_cases=sum(bool(r['changed_case_fields']) for r in lineage),
        changed_field_counts=dict(collections.Counter(f for r in lineage for f in r['changed_case_fields'])),
        label_changes=sum(r['label_changed'] for r in lineage), task_counts=dict(collections.Counter(r['case']['task'] for r in results)),
        language_counts=dict(collections.Counter(r['case']['language'] for r in results)), algorithm_counts=dict(collections.Counter(r['label']['algorithm'] for r in results)),
        unique_case_tuples=len({key(r['case']) for r in results}), duplicate_case_records=5000 - len({key(r['case']) for r in results}),
        source_status_counts=dict(collections.Counter(r['source_stage4_status'] for r in lineage)),
        all_tasks_preserved=True, all_candidates_from_original_dataset=True, empirically_validated_cases=0,
        method='Codex-authored joint Stage 2/3 rules; no per-case GPT calls; no relaxation of language/context/memory gates')
    write(rev2 / 'labeling_summary.json', summary); write(rev3 / 'generation_summary.json', summary)
    for path, h in preservation.items():
        if sha(Path(rel(path)) if Path(path).is_absolute() else PACKAGE_DIR.parent / path) != h:
            raise ValueError('Upstream changed during generation: ' + path)
    write(rev2 / 'original_preservation.json', preservation)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rev2-dir', type=Path, default=REV2); p.add_argument('--rev3-dir', type=Path, default=REV3)
    args = p.parse_args(argv)
    return generate(args.rev2_dir, args.rev3_dir)


if __name__ == '__main__':
    main()
