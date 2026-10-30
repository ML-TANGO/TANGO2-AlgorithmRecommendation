import argparse
import json
import os
from pathlib import Path

import pandas as pd

from templates import PROMPT_STRATEGY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_devices", help=" : CUDA_VISIBLE_DEVICES", default="0")
    parser.add_argument(
        "-m",
        "--model",
        help=" : Model to evaluate",
        default="yanolja/EEVE-Korean-Instruct-2.8B-v1.0",
    )
    parser.add_argument("-ml", "--model_len", help=" : Maximum Model Length", default=4096, type=int)
    parser.add_argument(
        "--max_tokens",
        help=" : Maximum number of generated tokens per response. Defaults to --model_len when omitted.",
        type=int,
    )
    parser.add_argument(
        "-a",
        "--adapter",
        help=" : Optional local LoRA adapter. If --model itself is an adapter directory, it is detected automatically.",
    )
    args = parser.parse_args()

    if args.max_tokens is None:
        args.max_tokens = args.model_len
    if args.max_tokens < 1:
        parser.error("--max_tokens must be a positive integer")

    print(f"Args - {args}")

    requested_model = args.model
    runtime_model = args.model
    adapter_path = args.adapter
    model_is_adapter = False
    model_adapter_config = Path(args.model) / "adapter_config.json"
    if adapter_path is None and model_adapter_config.is_file():
        adapter_path = args.model
        model_is_adapter = True

    lora_rank = None
    if adapter_path is not None:
        adapter_config_path = Path(adapter_path) / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise FileNotFoundError(f"LoRA adapter config not found: {adapter_config_path}")
        with adapter_config_path.open(encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        lora_rank = int(adapter_config["r"])
        if model_is_adapter:
            runtime_model = adapter_config["base_model_name_or_path"]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    gpu_counts = len(args.gpu_devices.split(","))

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print("- Using vLLM")
    if adapter_path is not None:
        print(f"- Applying LoRA online: {adapter_path} -> {runtime_model}")

    llm_kwargs = {}
    if adapter_path is not None:
        llm_kwargs.update(enable_lora=True, max_lora_rank=lora_rank)

    llm = LLM(
        model=runtime_model,
        tensor_parallel_size=gpu_counts,
        max_model_len=args.model_len,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        language_model_only=True,
        **llm_kwargs,
    )

    lora_request = (
        LoRARequest("evaluation_adapter", 1, str(adapter_path), base_model_name=runtime_model)
        if adapter_path is not None
        else None
    )

    sampling_params = SamplingParams(
        temperature=0,
        skip_special_tokens=True,
        max_tokens=args.max_tokens,
        stop=["<|endoftext|>", "[INST]", "[/INST]", "<|im_end|>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<eos>"],
    )

    questions_path = Path(__file__).with_name("questions.jsonl")
    df_questions = pd.read_json(questions_path, orient="records", encoding="utf-8-sig", lines=True)

    output_name = adapter_path if adapter_path is not None else requested_model
    output_dir = Path("generated") / str(output_name).lstrip("/")
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_tokenizer = llm.get_tokenizer()

    for strategy_name, prompts in PROMPT_STRATEGY.items():

        def format_single_turn_question(question):
            return chat_tokenizer.apply_chat_template(
                prompts + [{"role": "user", "content": question[0]}],
                tokenize=False,
                add_generation_prompt=True,
            )

        single_turn_questions = df_questions["questions"].map(format_single_turn_question)
        print(single_turn_questions.iloc[0])
        single_turn_outputs = [
            output.outputs[0].text.strip() for output in llm.generate(single_turn_questions, sampling_params, lora_request=lora_request)
        ]

        def format_double_turn_question(question, single_turn_output):
            return chat_tokenizer.apply_chat_template(
                prompts
                + [
                    {"role": "user", "content": question[0]},
                    {"role": "assistant", "content": single_turn_output},
                    {"role": "user", "content": question[1]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        multi_turn_questions = df_questions[["questions", "id"]].apply(
            lambda x: format_double_turn_question(x["questions"], single_turn_outputs[x["id"] - 1]),
            axis=1,
        )
        multi_turn_outputs = [
            output.outputs[0].text.strip() for output in llm.generate(multi_turn_questions, sampling_params, lora_request=lora_request)
        ]

        df_output = pd.DataFrame(
            {
                "id": df_questions["id"],
                "category": df_questions["category"],
                "questions": df_questions["questions"],
                "outputs": list(zip(single_turn_outputs, multi_turn_outputs)),
                "references": df_questions["references"],
            }
        )
        df_output.to_json(
            output_dir / f"{strategy_name}.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )


if __name__ == "__main__":
    main()
