"""Revised Stage 4: retain audited natural wording; revise only recorded case facts and derived budgets."""
import argparse
import collections
import copy
import json
import re
from pathlib import Path

from .thinking_policy import reason_for
from .sources import audited_requests_path, load_audited_requests
from ..stage4_requests.expression_utils import gpu_aliases, gpu_identity, korean_number
from ..stage4_requests.request_design import LANGUAGES
from ..stage4_requests.request_checks import inspect
from ..stage4_requests.assemble import answer_for, semantic_group
from ..io_utils import read, write, sha, digest, write_jsonl
from ..paths import REV2, REV3, REV4, MANUAL_CORRECTIONS, PACKAGE_DIR, rel


def number_pattern(s):
    return r'[\s,]*'.join(re.escape(c) for c in s.replace(',', '').replace(' ', ''))


def alias_pattern(s):
    return r'[\s\-]*'.join(re.escape(c) for c in re.sub(r'[\s\-]', '', s))


def revise_request(source, case, cid):
    """source: audited record {case, input_2, expression_metadata, ...}; case: revised case."""
    old = source['case']; text = source['input_2']; meta = copy.deepcopy(source['expression_metadata']); edits = []
    if gpu_identity(old['gpu']) != gpu_identity(case['gpu']):
        aliases = gpu_aliases(case['gpu']); alias = aliases[int(digest(cid + '/gpu')[:8], 16) % len(aliases)]['text']
        old_aliases = {x['text'] for x in gpu_aliases(old['gpu'])} | {meta['gpu_alias']}
        pattern = r'(?<![A-Za-z0-9])(?:' + '|'.join(alias_pattern(x) for x in sorted(old_aliases, key=len, reverse=True)) + r')(?![A-Za-z0-9])'
        text, n = re.subn(pattern, lambda m: alias, text, flags=re.I)
        if not n:
            raise ValueError((cid, 'GPU alias replacement did not match'))
        edits.append(dict(field='gpu', old=meta['gpu_alias'], new=alias, matches=n)); meta.update(gpu_alias=alias, gpu_identity=gpu_identity(case['gpu']))
        last = alias[-1]; jong = (ord(last) - 0xAC00) % 28 != 0 if '가' <= last <= '힣' else last in '013678lLmMnNrR'
        for a, b in ([('는', '은'), ('를', '을'), ('와', '과')] if jong else [('은', '는'), ('을', '를'), ('과', '와')]):
            text = re.sub(re.escape(alias) + a + r'(?=\s|[,.])', alias + b, text)
    if old['language'] != case['language']:
        a, b = LANGUAGES[old['language']], LANGUAGES[case['language']]; n = text.count(a); text = text.replace(a, b)
        if not n:
            raise ValueError((cid, 'language replacement did not match'))
        edits.append(dict(field='language', old=a, new=b, matches=n)); meta['language_name'] = b
    ctx = meta['context']; before = copy.deepcopy(ctx); n0 = old['context_length']; n = case['context_length']
    ctx['source_context_length'] = n
    if n0 != n and ctx['applicable']:
        oldstyle = ctx['style']; style = oldstyle
        if oldstyle in {'k', 'pages'} and n < 1024:
            style = 'tokens'
        surface = f'{n:,}토큰' if style == 'tokens' else korean_number(n) + ' 토큰' if style == 'written' else f'{n // 1024}K' if style == 'k' else None
        if oldstyle == 'pages':
            pages = n0 // 512; native = {2: ['두', '둘'], 4: ['네', '넷'], 8: ['여덟'], 16: ['열여섯'], 32: ['서른두'], 64: ['예순네'], 128: ['백스물여덟'], 256: ['이백쉰여섯']}
            forms = [str(pages), korean_number(pages)] + native.get(pages, [])
            pattern = r'(?:(보고서|문서|책)\s*)?(?<!\d)(?:' + '|'.join(number_pattern(v) for v in sorted(forms, key=len, reverse=True)) + r')\s*(페이지|쪽)'

            def page_sub(m):
                return ((m[1] + ' ') if m[1] else '') + str(n // 512) + m[2] if style == 'pages' else surface
            text, count = re.subn(pattern, page_sub, text)
            if style == 'pages':
                ctx.update(pages=n // 512, assumed_tokens_per_page=512, exact_recoverable=False)
                surface = re.sub(str(pages), str(n // 512), before['surface'], count=1)
            else:
                for k in ['pages', 'assumed_tokens_per_page', 'conversion_origin']:
                    ctx.pop(k, None)
                ctx['exact_recoverable'] = True
        else:
            forms = [str(n0), f'{n0:,}', korean_number(n0)]
            tokens = r'(?<!\d)(?:' + '|'.join(number_pattern(v) for v in sorted(set(forms), key=len, reverse=True)) + r')\s*(?:토큰|tokens?)'
            patterns = [tokens]
            if n0 >= 1024:
                patterns.append(r'(?<!\d)' + str(n0 // 1024) + r'\s*[kK](?:\s*토큰)?(?![A-Za-z0-9])')
            text, count = re.subn('|'.join(patterns), lambda m: surface, text, flags=re.I)
            ctx['exact_recoverable'] = True
        if count == 0:
            raise ValueError((cid, 'context replacement did not match', before['surface']))
        ctx.update(style=style, surface=surface, budget_tokens=n)
        # Remove stale qualitative volume adjectives after a large reduction;
        # preserve the model/training questions and all other request wording.
        if n < n0:
            for a, b in [('방대한 양의 ', '전체 '), ('방대한 ', '전체 '), ('대량의 ', '주어진 '), ('대규모 ', '해당 '), ('상당히 많은 편', '예상하는 양'), ('많은 양의 ', '해당 '), ('긴 분량의 ', '해당 '), ('긴 문서', '문서'), ('긴 텍스트', '텍스트'), ('긴 글', '글')]:
                text = text.replace(a, b)
        edits.append(dict(field='context_length', old=n0, new=n, matches=count, old_style=oldstyle, new_style=style))
    errors = inspect(text, case, meta)
    return text, meta, edits, errors


def build(output_dir=REV4, rev2=REV2, rev3=REV3):
    output_dir = Path(output_dir); rev2 = Path(rev2); rev3 = Path(rev3)
    source_path = audited_requests_path(); sources = load_audited_requests(source_path)
    s2 = read(rev2 / 'labeled_cases.json'); s3 = read(rev3 / 'hyperparameter_cases.json')
    profiles = {p['model_id']: p for p in read(rev2 / 'model_profiles.json')}; shared = read(rev3 / 'training_assumptions.json')
    rows = []; issues = []; transforms = []
    corrections = read(MANUAL_CORRECTIONS) if MANUAL_CORRECTIONS.exists() else {}
    for r2, r3 in zip(s2, s3):
        cid = r2['case_id']; source = sources[r2['source_case_id']]; text, meta, edits, errors = revise_request(source, r2['case'], cid)
        if cid in corrections:
            for before, after in corrections[cid]:
                if before not in text:
                    raise ValueError((cid, 'stale manual correction', before))
                text = text.replace(before, after)
            errors = inspect(text, r2['case'], meta)
        if errors:
            issues.append(dict(case_id=cid, source_case_id=r2['source_case_id'], errors=errors, text=text, case=r2['case'], metadata=meta))
        source_audit = copy.deepcopy(source['request_audit'])
        # Task semantics and reasoning demands are unchanged by fact substitution.
        source_audit['source_thinking_reason'] = source_audit['thinking_reason']
        source_audit['thinking_reason'] = reason_for(r2['case']['task'])
        if source_audit['thinking_recommended']:
            raise ValueError('Positive thinking requires a new semantic review')
        audit = dict(assessment=source_audit, method='source_negative_judgment_preserved_with_task_grounded_Codex_policy_reason')
        request = dict(text=text, metadata=meta)
        answer = answer_for(r3, request, audit, r2, profiles, shared)
        basis = r2['gpu_resource_basis']
        capacity = answer['recommendation_assumptions']['gpu_capacity']
        capacity.update(value_gb=basis['budget_capacity_gb'], origin='minimum_catalog_capacity_common_configuration',
                        requires_confirmation=False, validated_catalog_capacities_gb=basis['validated_vram_gb'],
                        note='입력에서 용량을 생략하므로 해당 GPU 이름의 모든 후보 용량에 같은 설정이 맞는지 검증했습니다. value_gb는 최소 용량으로 계산한 시스템 예산입니다.')
        answer['status'] = 'conditional_recommendation_draft'; answer['review_reasons'] = []
        answer['thinking']['capability_source'] = 'stage_2/revised/model_profiles.json; not a new benchmark'
        answer['source_stage3_status'] = 'recommended_draft'
        row = dict(case_id=cid, source_case_id=r2['source_case_id'], input_1=r3['input_1'], input_2=text, label_2=answer,
                   source_output_1=r3['output_1'], source_stage3_status='recommended_draft', expression_metadata=meta,
                   request_audit=source_audit, semantic_group=semantic_group(r3['input_1']['case']),
                   source_semantic_group=source['semantic_group'], supervised_recommendation_eligible=True,
                   generation_provenance=dict(method='reuse_audited_natural_request_with_recorded_fact_replacements', source_file=rel(source_path), source_case_id=r2['source_case_id'], per_case_GPT_API_called=False),
                   transformation=dict(edits=edits, manual_corrections=corrections.get(cid, []), source_input_sha256=digest(source['input_2']), revised_input_sha256=digest(text), new_conditions_added=False))
        rows.append(row); transforms.append(dict(case_id=cid, **row['transformation']))
    write(output_dir / 'request_review_queue.json', issues)
    if issues:
        raise ValueError(f'{len(issues)} requests need phrase review; see request_review_queue.json')
    # Connected components keep both original paraphrase groups and newly equal
    # revised environments together. This prevents duplicate-case split leakage.
    parent = {r['case_id']: r['case_id'] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    for field in ['semantic_group', 'source_semantic_group']:
        first = {}
        for r in rows:
            g = r[field]
            if g in first:
                union(r['case_id'], first[g])
            else:
                first[g] = r['case_id']
    for r in rows:
        r['split_group'] = find(r['case_id']); bucket = int(digest(r['split_group'])[:8], 16) % 100
        r['split'] = 'test' if bucket < 5 else 'validation' if bucket < 10 else 'train'
    if len({r['input_2'] for r in rows}) != len(rows):
        raise ValueError('Duplicate request text after substitution')
    data = {'input_2.json': [dict(case_id=r['case_id'], input_2=r['input_2']) for r in rows],
            'label_2.json': [dict(case_id=r['case_id'], label_2=r['label_2']) for r in rows],
            'stage4_cases.json': rows, 'transformation_log.json': transforms, 'training_assumptions.json': shared,
            'split_manifest.json': [dict(case_id=r['case_id'], semantic_group=r['semantic_group'], split_group=r['split_group'], split=r['split'], eligible=True) for r in rows]}
    for name, value in data.items():
        write(output_dir / name, value)
    td = output_dir / 'training_data'; td.mkdir(exist_ok=True)
    for split in ['train', 'validation', 'test']:
        write_jsonl(td / f'{split}.jsonl', [dict(case_id=r['case_id'], input=r['input_2'], output=r['label_2']) for r in rows if r['split'] == split])
    here = PACKAGE_DIR / 'revision'; s4 = PACKAGE_DIR / 'stage4_requests'
    files = [source_path, rev2 / 'labeled_cases.json', rev2 / 'model_profiles.json', rev3 / 'hyperparameter_cases.json', rev3 / 'training_assumptions.json', here / 'requests.py', here / 'thinking_policy.py']
    files += [s4 / n for n in ['request_checks.py', 'expression_utils.py', 'request_design.py', 'assemble.py']]
    if MANUAL_CORRECTIONS.exists():
        files.append(MANUAL_CORRECTIONS)
    write(output_dir / 'source_manifest.json', {rel(p): dict(sha256=sha(p)) for p in files})
    summary = dict(version='revised-1.0', total_cases=5000, supervised_recommendation_records=5000,
        recommendation_status_counts={'conditional_recommendation_draft': 5000}, thinking_recommended_counts={'false': 5000},
        training_split_counts=dict(collections.Counter(r['split'] for r in rows)),
        exact_duplicate_inputs=0, unchanged_input_texts=sum(r['input_2'] == sources[r['source_case_id']]['input_2'] for r in rows),
        expression_counts=dict(collections.Counter(r['expression_metadata']['context']['style'] for r in rows)),
        generation_method='audited source natural wording plus explicit case-fact substitutions; no fresh LLM generation',
        no_vram_in_input=True, empirically_validated=False,
        artifact_sha256={name: sha(output_dir / name) for name in data}, training_data_sha256={str(p.relative_to(output_dir)): sha(p) for p in sorted(td.glob('*.jsonl'))})
    write(output_dir / 'generation_summary.json', summary); print(json.dumps({k: v for k, v in summary.items() if 'sha256' not in k}, ensure_ascii=False, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=REV4); p.add_argument('--rev2-dir', type=Path, default=REV2); p.add_argument('--rev3-dir', type=Path, default=REV3)
    args = p.parse_args(argv)
    return build(args.output_dir, args.rev2_dir, args.rev3_dir)


if __name__ == '__main__':
    main()
