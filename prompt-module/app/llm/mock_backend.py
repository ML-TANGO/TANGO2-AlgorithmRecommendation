from __future__ import annotations

import hashlib
import random

from .base import Generation, LLMBackend, Message


class MockBackend(LLMBackend):
    """GPU/모델 없이 API·파이프라인 흐름을 검증하기 위한 결정적(deterministic) 가짜 모델."""

    def __init__(self, model: str = "mock"):
        self.name = f"mock:{model}"

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
        last = messages[-1]["content"]
        seed = int(hashlib.sha256(last.encode()).hexdigest()[:8], 16)
        out = []
        for i in range(n):
            rng = random.Random(seed + i)
            if "Score:" in last:  # verbalized confidence 요청
                text = f"Score: {rng.randint(30, 95)}"
            else:
                text = f"<prompt>\n[{self.name} | sample {i}]\n{last[:300]}\n{{{{input_data}}}}\n</prompt>"
            out.append(Generation(
                text=text,
                mean_logprob=-rng.uniform(0.05, 1.5) if return_logprobs else None,
                num_tokens=len(text.split()),
            ))
        return out
