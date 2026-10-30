"""Offline per-case request generation with the original (untrained) Gemma text model.

Requires one CUDA GPU. Labels, hyperparameters and GPU capacity are never shown
to the generator (see request_design.brief_for)."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from .request_design import SYSTEM, brief_for
from .request_checks import inspect
from ..io_utils import read
from ..paths import STAGE3, STAGE4, DEFAULT_BASE_MODEL, DEFAULT_BASE_REVISION


def parser():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--model', default=DEFAULT_BASE_MODEL, help='Hugging Face ID or local snapshot of the ORIGINAL instruction model')
    a.add_argument('--revision', default=DEFAULT_BASE_REVISION)
    a.add_argument('--stage3-dir', type=Path, default=STAGE3)
    a.add_argument('--limit', type=int, default=5000); a.add_argument('--batch-size', type=int, default=8)
    a.add_argument('--output', type=Path, help='JSONL attempts file (default: <stage_4>/requests_shard<N>.jsonl)')
    a.add_argument('--max-new-tokens', type=int, default=650); a.add_argument('--shard', type=int, default=0); a.add_argument('--shards', type=int, default=1)
    a.add_argument('--rounds', type=int, default=4); a.add_argument('--repair-file', type=Path)
    a.add_argument('--min-free-gib', type=float, default=23.0, help='Refuse to start on a GPU with less free memory')
    a.add_argument('--local-files-only', action='store_true')
    return a


def main(argv=None):
    args = parser().parse_args(argv)
    import torch
    from ..recommender.modeling import resolve_spec, load_tokenizer, load_text_model, render_chat
    if not torch.cuda.is_available():
        raise RuntimeError('The request writer needs a CUDA GPU')
    free, total = torch.cuda.mem_get_info()
    if free < args.min_free_gib * 1024 ** 3:
        raise RuntimeError('Selected GPU does not have sufficient free memory')
    rows = read(args.stage3_dir / 'input_1.json')[:args.limit]
    rows = [r for i, r in enumerate(rows) if i % args.shards == args.shard]
    output = args.output or STAGE4 / f'requests_shard{args.shard}.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = set(); latest = {}
    if output.exists():
        latest = {x['case_id']: x for x in (json.loads(l) for l in output.read_text(encoding='utf-8').splitlines() if l.strip())}
        case_lookup = {r['case_id']: r['input_1']['case'] for r in rows}
        existing = {cid for cid, x in latest.items() if x.get('accepted') and cid in case_lookup and not inspect(x['text'], case_lookup[cid], x['metadata'])}
    if args.repair_file:
        repairs = {x['case_id']: x for x in read(args.repair_file)}
        rows = [r for r in rows if r['case_id'] in repairs]
        for cid, x in repairs.items():
            if cid not in existing:
                latest[cid] = dict(errors=x['issues'], text=x.get('previous_text', ''), semantic_repair=True)
    rows = [r for r in rows if r['case_id'] not in existing]
    expected_total = len(rows) + len(existing)
    if not rows:
        print('Already complete'); return
    spec = resolve_spec(args.model, args.revision, local_files_only=args.local_files_only)
    tok = load_tokenizer(spec, training=False, local_files_only=args.local_files_only)
    model, load_info = load_text_model(spec, dtype=torch.bfloat16, device='cuda:0', attention='sdpa', local_files_only=args.local_files_only, quiet_expected_modalities=True)
    model.eval(); model.config.use_cache = True
    print(json.dumps({'loaded': load_info, 'remaining': len(rows)}, ensure_ascii=False), flush=True)
    start = time.monotonic(); done = 0
    with output.open('a', encoding='utf-8') as f:
        for round_index in range(args.rounds):
            pending = [r for r in rows if r['case_id'] not in existing]
            if not pending:
                break
            for i in range(0, len(pending), args.batch_size):
                batch = pending[i:i + args.batch_size]; prompts = []; metas = []
                for r in batch:
                    brief, meta = brief_for(r['input_1']['case'], r['case_id']); metas.append(meta)
                    previous = latest.get(r['case_id'])
                    if previous:
                        brief += '\n이전 시도는 다음 검사에서 탈락했습니다: ' + ', '.join(previous.get('errors', [])) + '. 조건을 정확히 보존하되 표현을 새롭게 작성하세요.'
                        if previous.get('semantic_repair'):
                            brief += '\n수정할 이전 글(문제를 고치되 새로운 사실을 추가하지 않기):\n' + previous.get('text', '')
                    prompts.append(render_chat(tok, [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': brief}], generation_prompt=True))
                encoded = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda:0')
                torch.manual_seed(20261024 + i + round_index * 10000 + args.shard * 100000)
                with torch.inference_mode():
                    out = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=1.05, top_p=.95, top_k=60, repetition_penalty=1.08, use_cache=True, pad_token_id=tok.pad_token_id)
                for r, ids, meta, prompt in zip(batch, out[:, encoded['input_ids'].shape[1]:], metas, prompts):
                    text = tok.decode(ids, skip_special_tokens=True).strip()
                    errors = inspect(text, r['input_1']['case'], meta)
                    record = dict(case_id=r['case_id'], text=text, metadata=meta, generator=spec.name, revision=spec.revision, method='local_model_per_case_generation',
                                  generated_tokens=len(ids), accepted=not errors, errors=errors, round=round_index, prompt_revision='v3_task_specific_text_scope',
                                  generation_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                    latest[r['case_id']] = record
                    if not errors:
                        existing.add(r['case_id'])
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush(); done += len(batch)
                print(json.dumps({'round': round_index, 'attempts': done, 'accepted': len(existing), 'shard_total': expected_total, 'elapsed_seconds': round(time.monotonic() - start, 1), 'last_case': batch[-1]['case_id']}, ensure_ascii=False), flush=True)
    print(json.dumps({'finished': True, 'accepted': len(existing), 'unresolved': expected_total - len(existing)}), flush=True)


if __name__ == '__main__':
    main()
