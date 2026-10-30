from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

Message = dict[str, str]


@dataclass
class Generation:
    text: str
    mean_logprob: float | None = None   # 생성 토큰 평균 log-prob (self-confidence 계산용)
    num_tokens: int | None = None


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
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
        ...
