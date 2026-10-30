# Qwen3-8B × MixATIS (UGEN QA 패러다임)

[Young1993/UGEN](https://github.com/Young1993/UGEN)(COLING 2022, T5 기반)의
**Unified Generative QA** 방식을 Qwen3-8B(decoder-only)에 맞게 재구성한
학습·평가 코드입니다.

## 구성
- `ugen_data.py` — MixATIS 로드 + UGEN 프롬프트/정답 형식 생성 + 예측 파싱(공용)
- `train_qwen3_mixatis.py` — Qwen3-8B LoRA SFT (completion-only loss)
- `evaluate_qwen3_mixatis.py` — test 셋 추론 + UGEN 메트릭
- `requirements.txt`

## UGEN 패러다임 요약
한 발화를 두 개의 QA 로 변환합니다(원본 Q1·Q5).
- **Intent QA**: `"...question: what are the intents...? options: ..."` → `airport,city,quantity`
- **Slot QA**: `"...List them with their slot names. options: ..."` → `la is one city name,...`

학습은 causal LM SFT 이며 프롬프트 토큰은 라벨 마스킹(-100)되어
assistant 정답 부분에만 loss 가 걸립니다.
Qwen3 thinking 충돌은 `enable_thinking=False` 로 방지합니다.

## 1) 데이터 준비
UGEN 저장소의 `data/MixATIS_clean`(seq.in / seq.out / label + intent.json / slot.json)을 사용합니다.
```bash
git clone https://github.com/Young1993/UGEN.git
```

## 2) 설치
```bash
pip install -r requirements.txt
```

## 3) 학습 (LoRA)
```bash
python train_qwen3_mixatis.py \
    --model_name Qwen/Qwen3-8B \
    --data_dir ./UGEN/data/MixATIS_clean \
    --output_dir ./out/qwen3-8b-mixatis-lora \
    --epochs 3 --batch_size 2 --grad_accum 8 --use_4bit
```
- `--use_4bit`: QLoRA(4bit) 로딩. 80GB 미만 GPU 권장.
- OOM 시: `--optim paged_adamw_8bit`(기본) 유지, `--batch_size 1`, `--max_len 640`.
- 빠른 점검: `--debug`(앞 200개 예제).

## 4) 평가
```bash
python evaluate_qwen3_mixatis.py \
    --model_name Qwen/Qwen3-8B \
    --adapter_dir ./out/qwen3-8b-mixatis-lora \
    --data_dir ./UGEN/data/MixATIS_clean \
    --out_csv ./out/mixatis_eval.csv --use_4bit
```
출력 메트릭: **Intent Acc / Intent F1 / Slot F1 / Overall(Joint) Acc**
(UGEN `utils/metric.py` 의 `Evaluator` 정의와 동일). 발화별 예측은 CSV 로 저장됩니다.
`--adapter_dir` 를 생략하면 베이스 모델 zero-shot 평가입니다.

## 메모
- MixSNIPS 도 동일하게 `--data_dir ./UGEN/data/MixSNIPS_clean` 로 사용 가능.
- 학습/추론 패딩 방향이 다릅니다(학습 right, 추론 left) — 코드에 반영됨.
- 슬롯 옵션 텍스트가 길어 `--max_len` 기본 768. 길이 초과 경고가 잦으면 늘리세요.
