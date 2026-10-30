"""Stage 4 full audit: source preservation, no leakage, supervised eligibility."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .assemble import read_requests, read_audits, answer_for, semantic_group
from .request_checks import inspect, TAGS
from ..io_utils import read, write, sha, digest, check
from ..paths import STAGE2, STAGE3, STAGE4, resolve


def validate(output_dir=STAGE4, stage3=STAGE3, stage2=STAGE2):
    output_dir = Path(output_dir); stage3 = Path(stage3); stage2 = Path(stage2)
    rows = read(output_dir / 'stage4_cases.json'); source = read(stage3 / 'hyperparameter_cases.json')
    source2 = {r['case_id']: r for r in read(stage2 / 'labeled_cases.json')}
    profiles = {p['model_id']: p for p in read(stage2 / 'model_profiles.json')}
    inputs = read(output_dir / 'input_2.json'); labels = read(output_dir / 'label_2.json'); shared = read(output_dir / 'training_assumptions.json'); summary = read(output_dir / 'generation_summary.json')
    requests = read_requests(output_dir); audits = read_audits(output_dir)
    check(len(rows) == len(inputs) == len(labels) == len(source) == 5000, 'Counts mismatch')
    check(len({r['case_id'] for r in rows}) == 5000, 'Duplicate IDs')
    for n, m in read(output_dir / 'source_manifest.json').items():
        check(sha(resolve(n)) == m['sha256'], 'Changed source: ' + n)
    for n, h in summary['artifact_sha256'].items():
        check(sha(output_dir / n) == h, 'Changed artifact: ' + n)
    for n, h in summary['training_data_sha256'].items():
        check(sha(output_dir / n) == h, 'Changed training file: ' + n)
    check(shared == read(stage3 / 'training_assumptions.json'), 'Assumption policy changed')
    group_splits = defaultdict(set); counts = Counter(); lengths = []; texts = set()
    for r, s, inp, lab in zip(rows, source, inputs, labels):
        cid = r['case_id']; text = r['input_2']; answer = r['label_2']; req = requests[cid]; audit = audits[(cid, digest(text))]
        check(cid == s['case_id'] == inp['case_id'] == lab['case_id'], cid + ': ID mismatch')
        check(r['input_1'] == s['input_1'] and r['source_output_1'] == s['output_1'], cid + ': source case/label/hyperparameters changed')
        check(text == inp['input_2'] == req['text'], cid + ': input mismatch')
        check(answer == lab['label_2'], cid + ': answer mismatch')
        check(not inspect(text, s['input_1']['case'], r['expression_metadata']), cid + ': input constraints failed')
        check(digest(text) not in texts, cid + ': exact duplicate request'); texts.add(digest(text))
        check(audit['assessment']['faithful'] is True and r['request_audit'] == audit['assessment'], cid + ': missing faithful audit')
        check(answer == answer_for(s, req, audit, source2[cid], profiles, shared), cid + ': answer reproduction mismatch')
        check(answer['empirically_validated'] is False and answer['ready_to_execute'] is False, cid + ': false empirical claim')
        if s['output_1'] is None:
            check(answer['recommendation'] is None, cid + ': invented unresolved recommendation')
        else:
            rec = answer['recommendation']
            check(all(rec[k] == s['input_1']['label'][k] for k in ['model', 'algorithm', 'evaluation_metric']), cid + ': original label changed')
            check(rec['hyperparameters'] == s['output_1']['hyperparameters'] and rec['training_plan'] == s['output_1']['training_plan'], cid + ': training settings changed')
        thinking = answer['thinking']
        check(type(thinking['recommended']) is bool and thinking['explicitly_requested'] is False, cid + ': thinking schema')
        check(thinking['recommended'] == audit['assessment']['thinking_recommended'], cid + ': thinking decision mismatch')
        if thinking['recommended']:
            check(thinking['request_evidence'] in text, cid + ': fabricated reasoning evidence')
        check(r['semantic_group'] == semantic_group(r['input_1']['case']), cid + ': grouping mismatch')
        group_splits[r['semantic_group']].add(r['split'])
        check(r['supervised_recommendation_eligible'] == (answer['status'] == 'conditional_recommendation_draft'), cid + ': unsafe training eligibility')
        ctx = r['expression_metadata']['context']
        if ctx['style'] == 'pages':
            ass = answer['recommendation_assumptions']['context_interpretation']
            check(not ctx['exact_recoverable'] and ass['assumed_tokens_per_page'] == 512 and ass['user_requested'] is False, cid + ': false exact page conversion')
        counts[answer['status']] += 1; lengths.append(len(TAGS.fullmatch(text).group(1)))
    check(all(len(v) == 1 for v in group_splits.values()), 'Semantic groups leak across splits')
    for split in ['train', 'validation', 'test']:
        got = [json.loads(l) for l in (output_dir / 'training_data' / (split + '.jsonl')).read_text(encoding='utf-8').splitlines()]
        expected = [dict(case_id=r['case_id'], input=r['input_2'], output=r['label_2']) for r in rows if r['split'] == split and r['supervised_recommendation_eligible']]
        check(got == expected, 'Training export mismatch ' + split)
    check(summary['recommendation_status_counts'] == dict(counts), 'Summary mismatch')
    report = dict(valid=True, version='4.0', cases_verified=len(rows), case_label_and_hyperparameters_preserved=True,
        exact_duplicate_requests=0, source_answer_leaks_detected=0, gpu_capacity_disclosures_detected=0,
        all_requests_independently_audited=True, description_characters=dict(min=min(lengths), max=max(lengths), mean=round(sum(lengths) / len(lengths), 1)),
        semantic_groups=len(group_splits), group_split_leaks=0, status_counts=dict(counts),
        validation_limitations='Automated lexical and local-model semantic audits plus human-style sample inspection; not a proof of perfect semantic equivalence or human authorship',
        artifact_sha256={n: sha(output_dir / n) for n in ['input_2.json', 'label_2.json', 'stage4_cases.json', 'generation_summary.json']})
    write(output_dir / 'validation_report.json', report); print(json.dumps(report, ensure_ascii=False, indent=2)); return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=STAGE4)
    p.add_argument('--stage3-dir', type=Path, default=STAGE3)
    p.add_argument('--stage2-dir', type=Path, default=STAGE2)
    args = p.parse_args(argv)
    return validate(args.output_dir, args.stage3_dir, args.stage2_dir)


if __name__ == '__main__':
    main()
