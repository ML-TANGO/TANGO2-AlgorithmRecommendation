from __future__ import annotations

import contextlib
import logging
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import Generation, LLMBackend, Message

logger = logging.getLogger(__name__)
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class HFModelHandle:
    """모델/토크나이저 1벌. generator와 optimizer가 같은 베이스를 쓰면 공유된다."""

    def __init__(
        self,
        model_name: str,
        *,
        adapter: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        enable_thinking: bool = False,
    ):
        logger.info("Loading HF model %s (adapter=%s)", model_name, adapter)
        self.model_name = model_name
        self.adapter = adapter
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {"dtype": _DTYPES[dtype], "device_map": device_map, "trust_remote_code": trust_remote_code}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=_DTYPES[dtype], bnb_4bit_quant_type="nf4"
            )
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self.model = model
        self.lock = threading.Lock()  # GPU 동시 접근 직렬화

        eos = model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

        self.enable_thinking = enable_thinking

    def encode(self, messages: list[Message]) -> torch.Tensor:
        tok = self.tokenizer
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True,
                  enable_thinking=self.enable_thinking)
        try:
            enc = tok.apply_chat_template(messages, **kw)
        except Exception:
            # system role 미지원 템플릿 대응: system을 첫 user 메시지에 병합
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            rest = [dict(m) for m in messages if m["role"] != "system"]
            if system and rest:
                rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
            enc = tok.apply_chat_template(rest, **kw)
        return enc["input_ids"].to(self.model.device)

class HFBackend(LLMBackend):
    def __init__(self, handle: HFModelHandle, *, use_adapter: bool = True):
        self.handle = handle
        self.use_adapter = use_adapter and handle.adapter is not None
        self.name = f"hf:{handle.model_name}" + (f"+{handle.adapter}" if self.use_adapter else "")

    @torch.inference_mode()
    def generate(
        self,
        messages: list[Message],
        *,
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 1024,
        return_logprobs: bool = False,
    ) -> list[Generation]:
        h = self.handle
        tok = h.tokenizer
        input_ids = h.encode(messages)
        do_sample = temperature > 0
        if not do_sample and n > 1:
            raise ValueError("n > 1 requires temperature > 0 (greedy decoding yields identical candidates)")

        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=n,
            return_dict_in_generate=True,
            output_logits=return_logprobs,
            pad_token_id=tok.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)

        # 같은 베이스 모델을 공유할 때 generator는 LoRA를 끄고 사용
        adapter_ctx = h.model.disable_adapter() if (h.adapter and not self.use_adapter) else contextlib.nullcontext()
        with h.lock, adapter_ctx:
            out = h.model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **gen_kwargs
            )

        gen_ids = out.sequences[:, input_ids.shape[1]:]
        token_lp = None
        if return_logprobs:
            # 원본 logits 기준(온도/top-p 적용 전) 선택 토큰의 log-prob
            token_lp = h.model.compute_transition_scores(out.sequences, out.logits, normalize_logits=True)

        results: list[Generation] = []
        for i in range(gen_ids.shape[0]):
            ids = gen_ids[i].tolist()
            length = next((j for j, t in enumerate(ids) if t in h.eos_ids), len(ids))
            text = tok.decode(ids[:length], skip_special_tokens=True).strip()
            mean_lp = None
            if token_lp is not None and length > 0:
                lp = token_lp[i, :length].float()
                lp = lp[torch.isfinite(lp)]
                mean_lp = lp.mean().item() if lp.numel() else None
            results.append(Generation(text=text, mean_logprob=mean_lp, num_tokens=length))
        return results
