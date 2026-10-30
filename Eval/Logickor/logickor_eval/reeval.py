import argparse
import glob
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .templates import JUDGE_TEMPLATE
except ImportError:
    from templates import JUDGE_TEMPLATE


@dataclass(frozen=True)
class LanguageRule:
    rule_id: str
    question_id: int
    turn: str
    question: str
    judge_override: str


LANGUAGE_RULES = (
    LanguageRule(
        rule_id="english_three_sentence_summary",
        question_id=1,
        turn="multi",
        question=(
            "위 답변을 영어 문장 3개로 요약한 후. "
            "해당 문장에 있는 단어 3개를 선택하여 단어의 의미를 설명해라."
        ),
        judge_override=(
            "후속 질문은 영어 문장 작성을 명시적으로 요구한다. "
            "따라서 영어 사용은 허용되며, 영어 사용 자체를 언어 위반으로 "
            "판정하거나 감점해서는 안 된다. 이 언어 판정을 다시 판단하지 말고 "
            "답변의 내용, 정확성, 형식 준수 여부만 1점부터 10점 사이로 평가한다."
        ),
    ),
    LanguageRule(
        rule_id="korean_response_misclassified_as_english_reasoning",
        question_id=2,
        turn="multi",
        question=(
            "지금까지 등장한 모든 숫자를 더한 후, "
            "더한 값의 각 자릿수 숫자들의 합을 구해보세요."
        ),
        judge_override=(
            "모델의 후속 답변은 한국어로 작성되어 있으므로 영어 답변이라는 이유로 "
            "언어 위반 판정을 내려서는 안 된다. 이 언어 판정을 다시 판단하지 말고, "
            "이전 대화에 실제로 등장한 숫자를 정확히 식별했는지, 불필요한 맥락을 "
            "혼입하지 않았는지, 계산과 최종 답이 정확한지를 기준으로 1점부터 "
            "10점 사이에서 평가한다."
        ),
    ),
    LanguageRule(
        rule_id="korean_response_misclassified_as_english_coding",
        question_id=28,
        turn="multi",
        question="주어진 문제의 난이도를 높일 방법을 제시해라.",
        judge_override=(
            "모델의 후속 답변은 한국어로 작성되어 있다. 코드, 식별자, 수식, "
            "알고리즘 용어에 포함된 영어는 언어 위반이 아니므로 영어 답변이라는 "
            "이유로 감점해서는 안 된다. 이 언어 판정을 다시 판단하지 말고, 제안한 "
            "난이도 상승 방법의 타당성, 기술적 정확성, 구체성을 기준으로 1점부터 "
            "10점 사이에서 평가한다."
        ),
    ),
    LanguageRule(
        rule_id="korean_code_response_misclassified_as_language_violation",
        question_id=28,
        turn="single",
        question=(
            "코딩 문제\n"
            "주어진 리스트에서 중복되지 않는 첫 번째 문제를 반환하는 함수를 작성해라.\n"
            "함수명: find_unique_character\n"
            "매개변수: characters (list)\n"
            "반환값: 중복되지 않는 첫 번째 문자\n"
            "예시:\n"
            "입력: ['a', 'b', 'c', 'a', 'd']\n"
            "출력: 'b'\n"
            "입력: ['a', 'b', 'a', 'b', 'c']\n"
            "출력: 'c'\n"
            "입력: ['a', 'b', 'c', 'd', 'e']\n"
            "출력: 'a'\n"
            "언어는 자유롭게 사용 할수 있다."
        ),
        judge_override=(
            "모델의 답변은 한국어 설명을 사용하고 있으며, 프로그래밍 언어의 문법, "
            "식별자, 타입 표기, API 이름과 코드 주석에 포함된 영어는 언어 위반이 "
            "아니다. 이를 한국어 답변 요구 위반으로 판정하거나 감점해서는 안 된다. "
            "함수 명세 준수, 알고리즘 정확성, 예시 결과, 복잡도 설명을 기준으로 "
            "1점부터 10점 사이에서 평가한다."
        ),
    ),
    LanguageRule(
        rule_id="korean_grammar_response_misclassified_as_language_violation",
        question_id=40,
        turn="single",
        question=(
            "제26항 한자어에서, 'ㄹ' 받침 뒤에 연결되는 'ㄷ, ㅅ, ㅈ'은 된소리로 발음한다.\n"
            "제28항 표기상으로는 사이시옷이 없더라도, 관형격 기능을 지니는 사이시옷이 "
            "있어야 할(휴지가 성립되는) 합성어의 경우에는, 뒤 단어의 첫소리 "
            "'ㄱ, ㄷ, ㅂ, ㅅ, ㅈ'을 된소리로 발음한다.\n"
            "위을 참고할 때 다음 문장 중 넷과 다른 하나는?\n"
            "- [길가]에 개나리가 만개했다.에서 '길가'\n"
            "- 너희들이 그 모양이니 [발전]이 없는 거야. 에서 '발전'\n"
            "- [발바닥]에 땀이 나도록 뛰었다. 에서 '발바닥'\n"
            "- [초승달]이 뜬 저녁, 매화가 흐트러졌다.  에서 '초승달'\n"
            "- 민수는 [손재주]가 좋아 무엇이든 잘 만든다. 에서 '손재주'"
        ),
        judge_override=(
            "모델의 답변은 한국어로 작성되어 있으므로 언어 요구사항 위반으로 0점 "
            "처리해서는 안 된다. 언어 판정을 다시 판단하지 말고, 제26항과 제28항의 "
            "적용, 각 단어의 실제 발음, Additional Reference와의 일치 여부, 최종 정답의 "
            "정확성만을 기준으로 1점부터 10점 사이에서 평가한다. 내용이 틀렸다면 그 "
            "정도에 맞게 낮은 점수를 부여한다."
        ),
    ),
)

LANGUAGE_CONTRADICTIONS = (
    "영어로 답변할 것을 요구하지",
    "영어 답변을 요구하지",
    "한국어로 답변해야 했",
    "언어 요구사항을 명백히 위반",
    "답변이 영어로 작성",
    "전체 답변이 영어",
    "영어로 답변하였",
    "언어 요구사항(한국어)도 준수하지",
    "한국어 언어 요구사항을 준수하지",
    "한국어 답변 요건을 충족하지",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate false zero scores caused by language-rule misclassification."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Evaluated JSONL file(s) or glob pattern(s).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "-k",
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help=(
            "OpenAI API key. Defaults to the OPENAI_API_KEY environment "
            "variable and is required unless --dry-run is used."
        ),
    )
    parser.add_argument("--judge-model", default="gpt-4.1")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_question(text: str) -> str:
    return " ".join(text.split())


def find_language_rule(row: dict[str, Any], turn: str) -> LanguageRule | None:
    question_index = 0 if turn == "single" else 1
    questions = row.get("questions")
    if not isinstance(questions, list) or len(questions) <= question_index:
        return None

    for rule in LANGUAGE_RULES:
        if (
            rule.turn == turn
            and row.get("id") == rule.question_id
            and normalize_question(questions[question_index])
            == normalize_question(rule.question)
        ):
            return rule
    return None


def find_candidates(row: dict[str, Any]) -> list[tuple[str, LanguageRule]]:
    candidates = []
    for turn, judge_key in (("single", "query_single"), ("multi", "query_multi")):
        rule = find_language_rule(row, turn)
        judge_result = row.get(judge_key)
        if (
            rule is not None
            and isinstance(judge_result, dict)
            and judge_result.get("judge_score") == 0
        ):
            candidates.append((turn, rule))
    return candidates


def build_judge_messages(
    row: dict[str, Any], turn: str, rule: LanguageRule
) -> list[dict[str, str]]:
    questions = row["questions"]
    outputs = row["outputs"]
    references = row.get("references") or []
    is_multi_turn = turn == "multi"

    prompt = (
        "아래의 내용을 주어진 평가 기준들을 충실히 반영하여 평가해라. "
        "특히 모델 답변이 언어 요구사항을 준수하는지 반드시 확인해야 한다.\n\n"
        f"**Question**\n{questions[0]}"
    )
    if len(references) > 0 and references[0]:
        prompt += f"\n\n**Additional Reference**\n{references[0]}"
    prompt += f"\n\n**Model's Response**\n{outputs[0]}"

    if is_multi_turn:
        prompt += f"\n\n**Follow-up Question.**\n{questions[1]}"
        if len(references) > 1 and references[1]:
            prompt += f"\n\n**Additional Reference**\n{references[1]}"
        prompt += f"\n\n**Model's Response**\n{outputs[1]}"

    prompt += (
        "\n\n**Rule-based Language Requirement (authoritative)**\n"
        f"{rule.judge_override}\n\n"
        "[[대화 종료. 평가 시작.]]"
    )
    system_prompt = (
        JUDGE_TEMPLATE["multi_turn" if is_multi_turn else "single_turn"]
        + "\n\n# 후처리에서 확정된 언어 규칙\n"
        + rule.judge_override
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def parse_judge_response(content: str) -> dict[str, Any]:
    plain_content = content.replace("*", "")
    message_match = re.search(r"평가:(.*?)점수:", plain_content, re.DOTALL)
    score_match = re.search(r"점수:\s*(\d+(?:\.\d+)?)", plain_content)
    if message_match is None or score_match is None:
        raise ValueError("Judge response does not contain the required 평가/점수 fields.")

    judge_message = message_match.group(1).strip()
    judge_score = float(score_match.group(1))
    if not 1 <= judge_score <= 10:
        raise ValueError(f"Re-evaluation score must be between 1 and 10: {judge_score}")
    if any(phrase in judge_message for phrase in LANGUAGE_CONTRADICTIONS):
        raise ValueError("Judge contradicted the authoritative language rule.")
    return {"judge_message": judge_message, "judge_score": judge_score}


def rejudge(
    client: Any,
    row: dict[str, Any],
    turn: str,
    rule: LanguageRule,
    judge_model: str,
    max_attempts: int,
    retry_delay: float,
) -> tuple[dict[str, Any], int]:
    messages = build_judge_messages(row, turn, rule)
    errors = []
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                temperature=0.0,
                n=1,
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            return parse_judge_response(content), attempt
        except Exception as exc:
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < max_attempts:
                time.sleep(retry_delay)
    raise RuntimeError("; ".join(errors))


def is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def is_reeval_output(path: Path) -> bool:
    return any(part.endswith("_reeval") for part in path.parts)


def expand_inputs(patterns: list[str]) -> list[tuple[Path, Path]]:
    files: dict[Path, Path] = {}
    for pattern in patterns:
        literal_path = Path(pattern)
        matches = [literal_path] if literal_path.exists() else [
            Path(match) for match in glob.glob(pattern, recursive=True)
        ]
        for match in matches:
            if match.is_file() and match.suffix == ".jsonl":
                resolved = match.resolve()
                files.setdefault(resolved, Path(match.name))
            elif match.is_dir():
                for file_path in match.rglob("*.jsonl"):
                    relative_path = file_path.relative_to(match)
                    if not is_hidden(relative_path) and not is_reeval_output(relative_path):
                        files.setdefault(file_path.resolve(), relative_path)

    if not files:
        raise ValueError(f"No JSONL files matched: {patterns}")
    return sorted(files.items())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")

    input_files = expand_inputs(args.input)
    output_paths = [relative_path for _, relative_path in input_files]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("Input files map to duplicate output paths; use separate output runs.")

    files_and_rows = [
        (path, relative_path, read_jsonl(path))
        for path, relative_path in input_files
    ]
    candidate_count = 0
    for path, _, rows in files_and_rows:
        for row in rows:
            for turn, rule in find_candidates(row):
                candidate_count += 1
                print(
                    f"CANDIDATE file={path} id={row.get('id')} "
                    f"turn={turn} rule={rule.rule_id}"
                )

    print(f"Matched {candidate_count} zero-score language candidate(s).")
    if args.dry_run:
        return
    if candidate_count == 0:
        print("No re-evaluation candidates found; exiting without API calls or output files.")
        return

    if not args.openai_api_key:
        raise ValueError(
            "--openai-api-key or OPENAI_API_KEY is required unless --dry-run is used."
        )

    from openai import OpenAI

    client = OpenAI(api_key=args.openai_api_key)
    corrected_count = 0
    unresolved_count = 0

    for input_path, relative_path, rows in files_and_rows:
        for row in rows:
            for turn, rule in find_candidates(row):
                judge_key = "query_single" if turn == "single" else "query_multi"
                original_result = dict(row[judge_key])
                try:
                    new_result, attempts = rejudge(
                        client=client,
                        row=row,
                        turn=turn,
                        rule=rule,
                        judge_model=args.judge_model,
                        max_attempts=args.max_attempts,
                        retry_delay=args.retry_delay,
                    )
                    row[f"{judge_key}_original"] = original_result
                    row[judge_key] = new_result
                    row.setdefault("language_rejudge", {})[turn] = {
                        "status": "corrected",
                        "rule_id": rule.rule_id,
                        "original_score": original_result["judge_score"],
                        "final_score": new_result["judge_score"],
                        "attempts": attempts,
                        "judge_model": args.judge_model,
                    }
                    corrected_count += 1
                except Exception as exc:
                    row.setdefault("language_rejudge", {})[turn] = {
                        "status": "manual_review",
                        "rule_id": rule.rule_id,
                        "original_score": original_result["judge_score"],
                        "error": str(exc),
                        "judge_model": args.judge_model,
                    }
                    unresolved_count += 1

        write_jsonl(args.output_dir / relative_path, rows, args.overwrite)

    print(
        f"Wrote {len(input_files)} file(s) to {args.output_dir}; "
        f"corrected={corrected_count}, manual_review={unresolved_count}."
    )
    if unresolved_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
