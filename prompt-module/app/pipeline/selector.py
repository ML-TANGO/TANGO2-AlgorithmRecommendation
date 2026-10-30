from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from ..config import Settings
from ..llm import Generation, LLMBackend
from ..templates import PromptTemplates

logger = logging.getLogger(__name__)
_SCORE = re.compile(r"score\s*[:=]?\s*(\d+(?:\.\d+)?)", re.I)
_NUM = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\b")


@dataclass
class ScoreResult:
    final: list[float]
    logprob: list[float | None]
    verbalized: list[float | None]
    method: str
    warnings: list[str] = field(default_factory=list)


class ConfidenceSelector:
    """
    Self-confidence 기반 후보 선택 (generator 자신의 확신도를 사용).
      - logprob    : exp(평균 토큰 log-prob) = 토큰 확률의 기하평균 (0~1). 추가 호출 없음.
      - verbalized : generator에게 후보별 0~100 점수를 직접 묻고 /100.
      - hybrid     : w * logprob + (1-w) * verbalized
    """

    def __init__(self, backend: LLMBackend, templates: PromptTemplates, settings: Settings):
        self.backend = backend
        self.templates = templates
        self.s = settings

    @staticmethod
    def _logprob_conf(g: Generation) -> float | None:
        return math.exp(g.mean_logprob) if g.mean_logprob is not None else None

    def _verbalized_conf(self, prompt: str, ctx: dict) -> float | None:
        if not prompt:
            return 0.0
        msg = self.templates.render(self.s.confidence_template, candidate_prompt=prompt, **ctx)
        text = self.backend.generate(
            [{"role": "user", "content": msg}], n=1, temperature=0.0, max_new_tokens=64
        )[0].text
        m = _SCORE.search(text) or _NUM.search(text)
        if not m:
            logger.warning("Could not parse verbalized score: %r", text)
            return None
        return max(0.0, min(100.0, float(m.group(1)))) / 100.0

    def score(self, prompts: list[str], gens: list[Generation], method: str, ctx: dict) -> ScoreResult:
        warnings: list[str] = []
        k = len(gens)

        lp: list[float | None] = [None] * k
        if method in ("logprob", "hybrid"):
            lp = [self._logprob_conf(g) for g in gens]
            if any(v is None for v in lp):
                warnings.append(f"logprobs unavailable from {self.backend.name}; fell back to 'verbalized'")
                method, lp = "verbalized", [None] * k

        vb: list[float | None] = [None] * k
        if method in ("verbalized", "hybrid"):
            vb = [self._verbalized_conf(p, ctx) for p in prompts]
            if any(v is None for v in vb):
                warnings.append("some verbalized scores could not be parsed (treated as 0)")

        w = self.s.hybrid_weight
        final = []
        for a, b in zip(lp, vb):
            b = b if b is not None else 0.0
            if method == "logprob":
                final.append(a)
            elif method == "verbalized":
                final.append(b)
            else:
                final.append(w * a + (1 - w) * b)
        return ScoreResult(final=final, logprob=lp, verbalized=vb, method=method, warnings=warnings)
