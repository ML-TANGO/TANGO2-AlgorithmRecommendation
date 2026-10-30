# LogicKor 평가

Gemma 4 E4B·12B IPO LoRA 어댑터로 LogicKor 답변을 생성하고 GPT-4.1로 평가하는 방법입니다.
명령은 모두 이 디렉터리(`Eval/Logickor`)에서 실행합니다.

```text
logickor.sh              5회 생성 → 평가 → 언어 규칙 0점 재평가 → 전략별 집계를 한 번에 실행
logickor_eval/
  questions.jsonl        LogicKor 원본 42문항
  generator.py           vLLM으로 답변 생성 (default, 1-shot, cot-1-shot 세 전략)
  evaluator.py           OpenAI judge 평가
  reeval.py              언어 규칙 관련 0점 재평가
  score.py               전략별 점수 집계
  templates.py           생성 프롬프트 템플릿
requirements/etri-infer.txt
```

## 1. 환경 준비

```bash
git clone https://github.com/ML-TANGO/TANGO2-AlgorithmRecommendation.git
cd TANGO2-AlgorithmRecommendation/Eval/Logickor

conda create -n etri-infer python=3.12 -y
conda activate etri-infer
python -m pip install -r requirements/etri-infer.txt
```

새 GPU 서버에서는 드라이버와 설치 패키지(PyTorch CUDA 13.0, vLLM)의 호환성을 확인하세요.
기본 모델(`google/gemma-4-E4B-it`, `google/gemma-4-12B-it`)을 받을 수 있도록 Hugging Face 로그인도 준비합니다.

```bash
hf auth login
```

## 2. 어댑터 다운로드

어댑터는 용량이 커서 저장소에 포함하지 않습니다. Hugging Face에서 `adapters/` 아래로 받습니다.
`logickor.sh --rl ipo`는 이 경로를 그대로 사용하므로 디렉터리 이름을 바꾸지 마세요.

```bash
hf download yanghoon/gemma4-e4b-ipo-v1-lora --local-dir adapters/gemma4-e4b-ipo
hf download yanghoon/gemma4-12b-ipo-v1-lora --local-dir adapters/gemma4-12b-ipo
```

각 어댑터의 `adapter_config.json`에 기본 모델 ID가 들어 있으며, 생성 시 기본 모델은 Hugging Face 캐시에서 불러옵니다.

## 3. 5회 자동 생성·평가

답변 생성 5회, GPT-4.1 평가, 언어 규칙 관련 0점 재평가 대상 확인·재평가, 전략별 집계를 한 번에 실행합니다.
API 평가와 재평가에는 비용이 발생합니다. 모든 0점이 자동 재평가 대상인 것은 아닙니다.

실행 전에 API 키를 환경변수로 설정하세요. 키를 파일에 저장하거나 공유하지 마세요.

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

```bash
bash logickor.sh --model e4b --rl ipo     # adapters/gemma4-e4b-ipo
bash logickor.sh --model 12b --rl ipo     # adapters/gemma4-12b-ipo
```

`--model`은 `e4b` 또는 `12b`, `--rl`은 `dpo` 또는 `ipo`를 받습니다.
`--rl dpo`는 직접 학습한 어댑터를 `runs/gemma4_<model>_dpo_v1/adapter`에 둔 경우에만 사용합니다.
다른 경로의 모델이나 기본 IT 모델을 평가하려면 인자 없이 실행한 뒤 경로 또는 Hugging Face 모델 ID를 입력하세요.

```bash
bash logickor.sh
```

### Conda, GPU, 생성 길이 설정

`logickor.sh`는 `conda info --base`로 Conda 위치를 찾고 `etri-infer` 환경을 자동으로 활성화합니다.
Conda가 PATH에 없으면 `CONDA_BASE=/path/to/miniconda3 bash logickor.sh ...`처럼 지정하세요.
실행 전에 스크립트 상단의 `CONDA_ENV`, `SINGLE_GPU`, `PARALLEL_GPUS`를 서버에 맞게 확인합니다.

| 실행 옵션 | 사용할 GPU 설정 | model_len | max_tokens |
|---|---|---:|---:|
| 기본 또는 `--parallel n` | `SINGLE_GPU` (현재 `"0"`) | 4096 | 4096 |
| `--parallel y` | `PARALLEL_GPUS` (현재 `"0,3"`) | 8192 | 3072 |

```bash
bash logickor.sh --model e4b --rl ipo --parallel y
```

### 저장 위치

같은 모델로 다시 실행하면 기존 cycle을 덮어쓰지 않고 다음 번호를 사용합니다.

```text
generated/gemma4-e4b-ipo/cycle1/v1/ ... v5/
evaluated/gemma4-e4b-ipo/cycle1/v1/ ... v5/
evaluated/gemma4-e4b-ipo/cycle1/summary.json
evaluated/gemma4-e4b-ipo/cycle1/result.txt
evaluated/gemma4-e4b-ipo/cycle1/logs/
```

완료 후 `result.txt`에서 회차별·전략별 결과와 남은 0점을 확인하세요.
중단된 cycle은 보존됩니다. 같은 명령을 다시 실행하면 중단 지점 재개가 아니라 새 cycle을 만듭니다.

## 4. 단계별로 실행하는 경우

5회 자동 실행 대신 생성·평가를 각각 실행할 때 사용합니다.

### 답변 생성

```bash
conda activate etri-infer

python logickor_eval/generator.py \
  --model adapters/gemma4-e4b-ipo \
  --gpu_devices 0 \
  --model_len 8192 \
  --max_tokens 3072
```

GPU 두 장은 `--gpu_devices 0,1`로 지정합니다. 쉼표 사이에 공백을 넣지 마세요.
기본 IT 모델을 평가하려면 `--model google/gemma-4-E4B-it` 또는 `google/gemma-4-12B-it`을 지정합니다.
`model_len`은 입력과 답변을 합친 최대 길이, `max_tokens`는 새 답변의 최대 길이입니다.
모델을 비교할 때는 생성 길이와 GPU 등 평가 조건을 동일하게 유지하세요.

생성 결과는 `generated/<모델 경로>/`(위 예시는 `generated/adapters/gemma4-e4b-ipo/`)에 저장됩니다.
같은 명령을 다시 실행하면 이 위치의 생성 파일을 덮어씁니다.

### API 평가

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

python logickor_eval/evaluator.py \
  -o generated/adapters/gemma4-e4b-ipo \
  --output-dir evaluated/gemma4-e4b-ipo/manual/v1 \
  -j gpt-4.1 \
  -t 30
```

새 실험에서는 `manual/v2`처럼 출력 경로를 바꾸세요. 같은 출력 파일이 이미 있으면 평가는 해당 파일을 건너뜁니다.

### 언어 규칙 관련 0점 재평가

먼저 API 호출 없이 대상 건수를 확인합니다.

```bash
python logickor_eval/reeval.py \
  --input evaluated/gemma4-e4b-ipo/manual/v1 \
  --output-dir evaluated/gemma4-e4b-ipo/manual/v1 \
  --dry-run
```

대상을 확인한 뒤 재평가합니다. `--overwrite`는 지정한 평가 결과를 수정하므로 필요하면 먼저 백업하세요.

```bash
python logickor_eval/reeval.py \
  --input evaluated/gemma4-e4b-ipo/manual/v1 \
  --output-dir evaluated/gemma4-e4b-ipo/manual/v1 \
  --overwrite \
  --judge-model gpt-4.1
```

### 전략별 집계

세 전략을 합치지 않고 각각 확인합니다.

```bash
python logickor_eval/score.py -p evaluated/gemma4-e4b-ipo/manual/v1/default.jsonl
python logickor_eval/score.py -p evaluated/gemma4-e4b-ipo/manual/v1/1-shot.jsonl
python logickor_eval/score.py -p evaluated/gemma4-e4b-ipo/manual/v1/cot-1-shot.jsonl
```
