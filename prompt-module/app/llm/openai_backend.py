from __future__ import annotations

import math

from openai import OpenAI

from .base import Generation, LLMBackend, Message


class OpenAICompatBackend(LLMBackend):
    """vLLM / TGI / SGLang 등 OpenAI 호환 서버용 백엔드."""

    def __init__(self, model: str, *, base_url: str | None, api_key: str, timeout: float):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.name = f"openai:{model}"

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
        kwargs: dict = dict(
            model=self.model, messages=messages, n=n,
            temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
        )
        if return_logprobs:
            kwargs["logprobs"] = True
        resp = self.client.chat.completions.create(**kwargs)

        results: list[Generation] = []
        for ch in resp.choices:
            mean_lp, ntok = None, None
            if return_logprobs and ch.logprobs and ch.logprobs.content:
                lps = [c.logprob for c in ch.logprobs.content if c.logprob is not None and math.isfinite(c.logprob)]
                ntok = len(lps)
                mean_lp = sum(lps) / ntok if ntok else None
            results.append(Generation(text=(ch.message.content or "").strip(), mean_logprob=mean_lp, num_tokens=ntok))
        return results
