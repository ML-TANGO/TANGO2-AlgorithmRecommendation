"""Second, label-blind pass: compare each written request with case facts (one CUDA GPU)."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from .audit_contract import parse
from .request_design import TASK_MEANING
from .request_checks import inspect
from ..io_utils import read
from ..paths import STAGE3, STAGE4, DEFAULT_BASE_MODEL, DEFAULT_BASE_REVISION

SYSTEM = '''사용자 요청문과 주어진 사실을 비교하는 독립 검수자입니다. 모델 추천 정답은 제공되지 않습니다.
모든 요청의 공통 목적은 적합한 모델과 학습 방법을 추천받는 것입니다. 모델 선정·학습 방법·훈련 전략·튜닝 조언 요청은 사용자가 명시적으로 허용한 내용이며 추가 task 오류가 아닙니다. 1K는 1024토큰이며 약/정도 표현은 허용합니다. 한국어 글의 표현만 다르면 허용합니다. 단, task·GPU 종류·언어·처리 분량의 의미는 유지해야 합니다. NLP 총 텍스트 예산은 입력과 출력 합계이며 원문만의 길이로 바꾸면 오류입니다. 음성 task의 분량은 facts.workload.scope를 따릅니다. ASR은 지시문과 전사문, TTS는 지시문과 음성으로 읽을 입력 텍스트만 합친 양입니다. TTS의 음성 출력은 텍스트 예산에 더하지 않습니다. 페이지 표현의 환산은 시스템 가정이므로 요청에 토큰 환산값이 없어도 됩니다. 입력에 없는 GPU 수/용량, 데이터 보유 사실, 번역 방향, 해상도/음성 길이/화자, 속도·비용의 수치 제한, 새로운 task, '특정 기능은 필요 없다'는 제한을 추가하면 오류입니다. 모델을 고르려는 고민, 일반적인 품질 기대, 작업 구상 같은 비제약적인 배경은 허용합니다. 미지정 실행 조건을 전제로 한 가짜 사실인지 구별하세요. 작성 문체 지시가 작업 요구사항으로 섞인 경우(예: 논리적으로 말하는 TTS, 상담처럼 간결한 답변 조건)도 추가 조건이므로 오류입니다. 이미지/오디오 분류에 텍스트 분량 관리 조건을 새로 넣으면 오류입니다. 단순히 자세하지 않다는 이유로 오류로 삼지 마세요. GPU와 언어는 모든 case에 제공된 조건입니다. 이미지 작업에 언어가 제시되어도 작업 환경 언어로 자연스럽게 언급하는 것은 허용하며, 그 언어의 기술적 역할을 새로 설명할 필요는 없습니다. workload.applicable=false이면 비텍스트 작업이므로 분량을 언급하지 않는 것이 정확합니다.
또한 추천 대상 task를 수행할 때 별도 thinking 모델/모드가 권장되는지 요청문만 보고 판단하세요. 당신이 추천을 고민하는 과정의 난이도를 판단하는 것이 아닙니다. 단순 QA/분류/번역/요약, 긴 분량, '정확한 결과/설명/학습법 추천'만으로는 thinking이 필요하다고 하지 마세요. 작업 자체에 여러 단계의 추론, 조건·근거의 논리적 연결, 수학적 도출이 명확히 요구된 경우에만 recommended=true입니다. 이미지·음성 전용 분류/인식/합성은 일반적인 경우 false입니다. true이면 요청에 실제로 존재하는 짧은 근거 구절을 evidence에 그대로 인용하세요. 숨겨진 모델의 지원 여부를 추측하지 마세요.
다음 JSON 객체만 출력하세요. 장황한 추론 과정은 쓰지 않습니다.
{"faithful":true,"issues":[],"thinking_recommended":false,"thinking_reason":"요청의 어떤 내용에 근거했는지 짧은 한국어 한 문장","evidence":null}
faithful=false이면 issues에 실제 문제와 문제 구절을 구체적으로 적으세요. thinking_reason에는 '모델 이름'이나 제공되지 않은 사실을 쓰지 마세요.'''


def sha_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=DEFAULT_BASE_MODEL); p.add_argument('--revision', default=DEFAULT_BASE_REVISION)
    p.add_argument('--stage3-dir', type=Path, default=STAGE3); p.add_argument('--stage4-dir', type=Path, default=STAGE4)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=2); p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--limit', type=int, default=5000); p.add_argument('--requests', nargs='*', type=Path, help='Request JSONL files (default: <stage_4>/requests_shard*.jsonl)')
    p.add_argument('--output', type=Path); p.add_argument('--min-free-gib', type=float, default=23.0); p.add_argument('--local-files-only', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    import torch
    from ..recommender.modeling import resolve_spec, load_tokenizer, load_text_model, render_chat
    if not torch.cuda.is_available():
        raise RuntimeError('The request auditor needs a CUDA GPU')
    free, _ = torch.cuda.mem_get_info()
    if free < args.min_free_gib * 1024 ** 3:
        raise RuntimeError('GPU busy')
    cases = {r['case_id']: r['input_1']['case'] for r in read(args.stage3_dir / 'input_1.json')}
    latest = {}
    for path in args.requests or sorted(args.stage4_dir.glob('requests_shard*.jsonl')):
        for l in Path(path).read_text(encoding='utf-8').splitlines():
            r = json.loads(l)
            if r.get('accepted'):
                latest[r['case_id']] = r
    records = [r for cid, r in sorted(latest.items()) if (int(cid[-5:]) - 1) % args.shards == args.shard and not inspect(r['text'], cases[cid], r['metadata'])][:args.limit]
    output = args.output or args.stage4_dir / f'audits_shard{args.shard}.jsonl'; existing = set()
    if output.exists():
        for line in output.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if r.get('assessment') is not None:
                existing.add((r['case_id'], r['request_sha256']))
    records = [r for r in records if (r['case_id'], sha_text(r['text'])) not in existing]
    if not records:
        print('Already complete'); return
    spec = resolve_spec(args.model, args.revision, local_files_only=args.local_files_only); tok = load_tokenizer(spec, training=False, local_files_only=args.local_files_only)
    model, _ = load_text_model(spec, dtype=torch.bfloat16, device='cuda:0', attention='sdpa', local_files_only=args.local_files_only, quiet_expected_modalities=True)
    model.eval(); model.config.use_cache = True; start = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('a', encoding='utf-8') as f:
        for i in range(0, len(records), args.batch_size):
            batch = records[i:i + args.batch_size]; prompts = []
            for r in batch:
                case = cases[r['case_id']]; meta = r['metadata']; facts = dict(task=TASK_MEANING[case['task']], gpu=meta['gpu_alias'], language=meta['language_name'], workload=meta['context'] if meta['context']['applicable'] else {'applicable': False})
                # Only page quantity and scope are visible to generator; audit may know
                # the disclosed conversion assumption without mistaking it for user text.
                content = '事実 / 사실:\n' + json.dumps(facts, ensure_ascii=False) + '\n검수할 요청문:\n' + r['text']
                prompts.append(render_chat(tok, [dict(role='system', content=SYSTEM), dict(role='user', content=content)], generation_prompt=True))
            encoded = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda:0')
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=450, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
            for r, ids in zip(batch, generated[:, encoded['input_ids'].shape[1]:]):
                raw = tok.decode(ids, skip_special_tokens=True).strip(); assessment = parse(raw, r['text'])
                f.write(json.dumps(dict(case_id=r['case_id'], request_sha256=sha_text(r['text']), assessment=assessment, raw_assessment=raw,
                    method='independent_prompt_local_gemma4_request_audit', model_revision=spec.revision, answer_fields_provided=False, audit_prompt_revision='v4_task_specific_text_scope', audit_system_sha256=sha_text(SYSTEM)), ensure_ascii=False) + '\n')
            f.flush(); print(json.dumps(dict(audited=min(i + len(batch), len(records)), total=len(records), elapsed_seconds=round(time.monotonic() - start, 1))), flush=True)


if __name__ == '__main__':
    main()
