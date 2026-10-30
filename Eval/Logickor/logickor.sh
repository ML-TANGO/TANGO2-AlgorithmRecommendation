#!/usr/bin/env bash
set -euo pipefail

# API 키는 실행 전에 OPENAI_API_KEY 환경변수로 설정하세요. 파일에 저장하지 마세요.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")}"
CONDA_ENV="etri-infer"
CONDA_ENV_PATH="$CONDA_BASE/envs/$CONDA_ENV"
SINGLE_GPU="0"
PARALLEL_GPUS="0,3"

PARALLEL="n"
MODEL_SIZE=""
RL=""
usage() {
  echo "사용법: bash logickor.sh [--model e4b|12b --rl dpo|ipo] [--parallel y|n]"
}

while (($#)); do
  case "$1" in
    --model)
      [[ $# -ge 2 && ( "$2" == "e4b" || "$2" == "12b" ) ]] || {
        usage >&2
        exit 2
      }
      MODEL_SIZE="$2"
      shift 2
      ;;
    --rl)
      [[ $# -ge 2 && ( "$2" == "dpo" || "$2" == "ipo" ) ]] || {
        usage >&2
        exit 2
      }
      RL="$2"
      shift 2
      ;;
    --parallel)
      [[ $# -ge 2 && ( "$2" == "y" || "$2" == "n" ) ]] || {
        usage >&2
        exit 2
      }
      PARALLEL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      echo "--model과 --rl은 함께 지정하세요."
      echo "IPO: $PROJECT_ROOT/adapters/gemma4-e4b-ipo 또는 gemma4-12b-ipo"
      echo "DPO: $PROJECT_ROOT/runs/gemma4_<model>_dpo_v1/adapter (기본 경로)"
      echo "Conda: $CONDA_ENV_PATH"
      echo "둘 다 생략하면 모델 경로/Hugging Face ID를 직접 입력합니다."
      echo "${CONDA_ENV} 자동 활성화 → 5회 생성 → 평가 → 재평가 → 전략별 평균"
      echo "단일: GPU ${SINGLE_GPU}, model_len=4096, max_tokens=4096"
      echo "병렬: GPU ${PARALLEL_GPUS}, model_len=8192, max_tokens=3072"
      exit 0
      ;;
    *) echo "알 수 없는 인자: $1" >&2; exit 2 ;;
  esac
done

ROOT="$PROJECT_ROOT"
[[ -d "$ROOT" ]] || {
  echo "프로젝트 디렉터리가 없습니다: $ROOT" >&2
  exit 1
}
cd -- "$ROOT"
if [[ -n "$MODEL_SIZE" || -n "$RL" ]]; then
  [[ -n "$MODEL_SIZE" && -n "$RL" ]] || {
    echo "--model과 --rl을 함께 지정하세요." >&2
    usage >&2
    exit 2
  }
  if [[ "$RL" == "ipo" ]]; then
    MODEL="$ROOT/adapters/gemma4-${MODEL_SIZE}-ipo"
  else
    MODEL="$ROOT/runs/gemma4_${MODEL_SIZE}_${RL}_v1/adapter"
  fi
  [[ -f "$MODEL/adapter_config.json" && -s "$MODEL/adapter_model.safetensors" ]] || {
    echo "선택한 체크포인트가 없습니다: $MODEL" >&2
    exit 1
  }
else
  read -r -p "평가할 모델 경로 또는 Hugging Face ID: " MODEL
  [[ -n "$MODEL" ]] || { echo "모델을 입력해야 합니다." >&2; exit 2; }
fi

[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || {
  echo "Conda 초기화 파일이 없습니다: $CONDA_BASE/etc/profile.d/conda.sh" >&2
  exit 1
}
[[ -d "$CONDA_ENV_PATH" && -x "$CONDA_ENV_PATH/bin/python" ]] || {
  echo "추론 가상환경이 없습니다: $CONDA_ENV_PATH" >&2
  exit 1
}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

# pip로 설치된 CUDA 13 툴킷이 있으면 vLLM이 그것을 쓰도록 지정합니다.
if [[ -d "$CONDA_ENV_PATH/lib/python3.12/site-packages/nvidia/cu13" ]]; then
  export CUDA_HOME="$CONDA_ENV_PATH/lib/python3.12/site-packages/nvidia/cu13"
  export PATH="$CUDA_HOME/bin:$PATH"
  hash -r
fi

PYTHON_BIN="$CONDA_ENV_PATH/bin/python"
echo "가상환경: $CONDA_ENV ($PYTHON_BIN)"
export OPENAI_API_KEY PYTHONUNBUFFERED=1

"$PYTHON_BIN" - "$ROOT" "$MODEL" "$PARALLEL" "$SINGLE_GPU" "$PARALLEL_GPUS" <<'PY'
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile

root, model_arg, parallel, single_gpu, parallel_gpus = sys.argv[1:]
root = Path(root)
strategies = ("default", "1-shot", "cot-1-shot")
runs = tuple(f"v{i}" for i in range(1, 6))
gpu_devices = parallel_gpus if parallel == "y" else single_gpu
model_len, max_tokens = (8192, 3072) if parallel == "y" else (4096, 4096)


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    missing = [name for name in ("vllm", "pandas", "openai", "rich")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"현재 Python에 필요한 패키지가 없습니다: {', '.join(missing)}. etri-infer 환경을 확인하세요.")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "PASTE_YOUR_OPENAI_API_KEY":
        raise RuntimeError("실행 전에 OPENAI_API_KEY 환경변수를 설정하세요.")
    # evaluator.py는 Azure 환경변수가 있으면 Azure를 선택하므로 OpenAI 실행을 명시합니다.
    for name in ("AZURE_ENDPOINT", "AZURE_DEPLOYMENT_NAME", "AZURE_API_VERSION"):
        os.environ.pop(name, None)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    model_path = Path(model_arg).expanduser()
    if model_path.is_dir():
        model = str(model_path.resolve())
        model_name = model_path.resolve().parent.name if model_path.name in ("adapter", "merged") else model_path.resolve().name
    else:
        if model_path.is_absolute() or model_arg.startswith(("./", "../", "~/", "runs/", "models/")):
            raise FileNotFoundError(f"모델 디렉토리가 없습니다: {model_arg}")
        model = model_arg
        model_name = model_arg.rstrip("/").split("/")[-1]
    if not model_name or model_name in (".", ".."):
        raise ValueError("모델 이름을 확인하세요.")
    if shutil.which("nvidia-smi"):
        available = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        ).split()
        if not set(gpu_devices.split(",")).issubset(available):
            raise RuntimeError(f"요청 GPU={gpu_devices}, 사용 가능한 GPU={','.join(available)}. 스크립트 상단 GPU 설정을 확인하세요.")

    questions = read_jsonl(root / "logickor_eval/questions.jsonl")
    expected = {row["id"]: row for row in questions}
    if len(questions) != 42 or set(expected) != set(range(1, 43)):
        raise ValueError("LogicKor 원본 42문항(ID 1~42)이 필요합니다.")
    generated_root = root / "generated" / model_name
    evaluated_root = root / "evaluated" / model_name
    numbers = [int(path.name[5:]) for base in (generated_root, evaluated_root)
               if base.exists() for path in base.iterdir()
               if path.name.startswith("cycle") and path.name[5:].isdigit()]
    cycle = f"cycle{max(numbers, default=0) + 1}"
    generated = generated_root / cycle
    evaluated = evaluated_root / cycle
    generated.mkdir(parents=True, exist_ok=False)
    evaluated.mkdir(parents=True, exist_ok=False)
    logs = evaluated / "logs"
    logs.mkdir()
    config = dict(model=model, parallel=parallel, gpu_devices=gpu_devices,
                  model_len=model_len, max_tokens=max_tokens, judge_model="gpt-4.1",
                  threads=30, runs=5, scope="42 questions; all IDs included")
    (evaluated / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    console.print(f"모델: {model}\n실험: {model_name}/{cycle}\nGPU: {gpu_devices}, model_len={model_len}, max_tokens={max_tokens}", markup=False)

    def command(script, args, log_name, cwd=root, allow_manual_review=False):
        console.print(f"실행: {script} → {log_name}", markup=False)
        log_path = logs / log_name
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                [sys.executable, str(root / "logickor_eval" / script), *map(str, args)],
                cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                status = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if status and not (allow_manual_review and status == 2):
            raise RuntimeError(f"{script} 실패(exit={status}). 로그: {log_path}")
        return status

    def validate(directory, require_scores=False, source=None):
        files = {}
        for strategy in strategies:
            path = directory / f"{strategy}.jsonl"
            rows = read_jsonl(path)
            if len(rows) != 42 or {row.get("id") for row in rows} != set(expected):
                raise ValueError(f"{path}: 중복·누락 없이 42문항이어야 합니다.")
            for row in rows:
                question = expected[row["id"]]
                if any(row.get(key) != question.get(key) for key in ("category", "questions", "references")):
                    raise ValueError(f"{path}: id={row['id']} 원본 문항 불일치")
                outputs = row.get("outputs")
                if not isinstance(outputs, list) or len(outputs) != 2 or not all(isinstance(x, str) and x.strip() for x in outputs):
                    raise ValueError(f"{path}: id={row['id']} 두 턴의 답변이 필요합니다.")
                if source is not None and outputs != source[strategy][row["id"]]["outputs"]:
                    raise ValueError(f"{path}: id={row['id']} 생성 답변과 평가 답변 불일치")
                if require_scores:
                    for key in ("query_single", "query_multi"):
                        score = row.get(key, {}).get("judge_score")
                        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 10:
                            raise ValueError(f"{path}: id={row['id']} {key} 점수가 유효하지 않습니다.")
            files[strategy] = {row["id"]: row for row in rows}
        return files

    generated_rows = {}
    for run in runs:
        console.rule(f"답변 생성 {run}/v5")
        # 생성기는 cwd/generated/<model>에 쓰므로 작업 폴더를 회차별로 격리합니다.
        with tempfile.TemporaryDirectory(prefix=f".generate-{run}-", dir=generated) as work:
            command("generator.py", ["--model", model, "--gpu_devices", gpu_devices,
                                     "--model_len", model_len, "--max_tokens", max_tokens],
                    f"generate_{run}.log", cwd=work)
            source = Path(work) / "generated" / model.lstrip("/")
            generated_rows[run] = validate(source)
            shutil.move(str(source), str(generated / run))

    for run in runs:
        console.rule(f"초기 평가 {run}/v5")
        command("evaluator.py", ["-o", generated / run, "--output-dir", evaluated / run,
                                 "-j", "gpt-4.1", "-t", 30], f"evaluate_{run}.log")
        validate(evaluated / run, require_scores=True, source=generated_rows[run])

    console.rule("재평가 대상 dry-run")
    command("reeval.py", ["--input", evaluated, "--output-dir", evaluated, "--dry-run"], "reeval_dry_run.log")
    console.rule("언어 규칙에 따른 0점 재평가")
    reeval_status = command(
        "reeval.py", ["--input", evaluated, "--output-dir", evaluated, "--overwrite",
                      "--judge-model", "gpt-4.1"], "reeval.log", allow_manual_review=True,
    )

    summary = dict(config, cycle=cycle, strategy_mean_across_runs={}, runs=[], zero_scores=[], manual_review=[])
    details = []
    for run in runs:
        files = validate(evaluated / run, require_scores=True, source=generated_rows[run])
        run_result = {"run": run, "strategies": {}}
        for strategy, indexed in files.items():
            path = evaluated / run / f"{strategy}.jsonl"
            command("score.py", ["-p", path], f"score_{run}_{strategy}.log")
            details.append(f"## {run}/{strategy}.jsonl\n\n" + (logs / f"score_{run}_{strategy}.log").read_text(encoding="utf-8"))
            single = [row["query_single"]["judge_score"] for row in indexed.values()]
            multi = [row["query_multi"]["judge_score"] for row in indexed.values()]
            run_result["strategies"][strategy] = {
                "single": statistics.mean(single), "multi": statistics.mean(multi),
                "overall": statistics.mean(single + multi),
            }
            for row in indexed.values():
                for turn, key in (("single", "query_single"), ("multi", "query_multi")):
                    if row[key]["judge_score"] == 0:
                        summary["zero_scores"].append(dict(run=run, strategy=strategy, id=row["id"], turn=turn))
                for turn, review in row.get("language_rejudge", {}).items():
                    if review.get("status") == "manual_review":
                        summary["manual_review"].append(dict(run=run, strategy=strategy, id=row["id"], turn=turn))
        summary["runs"].append(run_result)
    for strategy in strategies:
        summary["strategy_mean_across_runs"][strategy] = {
            key: statistics.mean(run["strategies"][strategy][key] for run in summary["runs"])
            for key in ("single", "multi", "overall")
        }
    summary["zero_score_count_after_reeval"] = len(summary["zero_scores"])
    summary["reeval_exit_code"] = reeval_status
    (evaluated / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (evaluated / "score_details.md").write_text("\n\n".join(details) + "\n", encoding="utf-8")

    report = Console(record=True)
    table = Table(title=f"{model_name}/{cycle} — 회차별 Overall (42문항 전체)")
    for label in ("회차", *strategies):
        table.add_column(label, justify="left" if label == "회차" else "right")
    for run in summary["runs"]:
        table.add_row(run["run"], *(f"{run['strategies'][s]['overall']:.4f}" for s in strategies))
    table.add_section()
    table.add_row("5회 평균", *(f"{summary['strategy_mean_across_runs'][s]['overall']:.4f}" for s in strategies), style="bold cyan")
    report.print(table)
    table = Table(title="전략별 5회 평균 — 재평가 반영")
    for label in ("전략", "Single", "Multi", "Overall"):
        table.add_column(label, justify="left" if label == "전략" else "right")
    for strategy, scores in summary["strategy_mean_across_runs"].items():
        table.add_row(strategy, *(f"{scores[key]:.4f}" for key in ("single", "multi", "overall")))
    report.print(table)
    report.print(f"남은 0점: {len(summary['zero_scores'])}건 / 재평가 수동 검토: {len(summary['manual_review'])}건")
    for label in ("zero_scores", "manual_review"):
        for item in summary[label]:
            report.print(f"{label}: {item}", markup=False)
    report.print(f"생성: {generated}\n평가·점수·로그: {evaluated}", markup=False)
    (evaluated / "result.txt").write_text(report.export_text(), encoding="utf-8")
    return reeval_status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\n중단: {exc or '사용자 인터럽트'}\n이미 저장된 회차 결과는 해당 cycle에 남아 있습니다.", file=sys.stderr)
        raise SystemExit(1)
PY
