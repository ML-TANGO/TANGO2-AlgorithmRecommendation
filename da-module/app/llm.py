import json
import threading

from .config import Settings


class MockBackend:
    def __init__(self, settings: Settings):
        self.name = f"mock:{settings.model}"

    def generate(self, messages: list[dict[str, str]]) -> str:
        if messages[0]["content"].startswith("You prepare a labeling plan"):
            return json.dumps({
                "instruction": "입력 한 건에 테스트 라벨과 한국어 근거를 부여하세요.",
                "output_format": "json",
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "minLength": 1},
                        "reasoning": {"type": "string", "minLength": 1},
                    },
                    "required": ["label", "reasoning"],
                    "additionalProperties": False,
                },
            }, ensure_ascii=False)
        return json.dumps({"label": "mock_label", "reasoning": "모델 없이 연결을 확인하는 테스트 응답입니다."})


class HFBackend:
    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.settings = settings
        self.name = f"hf:{settings.model}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model, trust_remote_code=settings.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model,
            dtype=getattr(torch, settings.torch_dtype),
            device_map=settings.device_map,
            trust_remote_code=settings.trust_remote_code,
        ).eval()
        self.lock = threading.Lock()

    def generate(self, messages: list[dict[str, str]]) -> str:
        import torch
        from jinja2 import TemplateError

        with self.lock, torch.inference_mode():
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
                )
            except (ValueError, TemplateError):
                # system 역할을 지원하지 않는 모델은 지시를 user 메시지에 합칩니다.
                merged = "\n\n".join(message["content"] for message in messages)
                encoded = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": merged}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            encoded = encoded["input_ids"].to(self.model.device)
            output = self.model.generate(
                input_ids=encoded,
                attention_mask=torch.ones_like(encoded),
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            return self.tokenizer.decode(output[0, encoded.shape[1]:], skip_special_tokens=True).strip()


def build_backend(settings: Settings) -> HFBackend | MockBackend:
    return MockBackend(settings) if settings.backend == "mock" else HFBackend(settings)
