from __future__ import annotations

import re
import time
from contextlib import contextmanager

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_PROMPT = re.compile(r"<prompt>\s*(.*?)\s*</prompt>", re.S | re.I)
_FENCE = re.compile(r"^```[\w-]*\n(.*)\n```$", re.S)


class PipelineError(RuntimeError):
    pass


def extract_prompt(text: str) -> tuple[str, bool]:
    """모델 출력에서 <prompt>...</prompt> 내용을 추출. (prompt, 태그 발견 여부) 반환."""
    text = _THINK.sub("", text).strip()
    matches = [m.strip() for m in _PROMPT.findall(text) if m.strip()]
    if matches:
        return max(matches, key=len), True
    lower = text.lower()
    if "<prompt>" in lower:  # 닫는 태그 없이 잘린 경우
        return text[lower.index("<prompt>") + len("<prompt>"):].strip(), False
    fence = _FENCE.match(text)
    if fence:
        return fence.group(1).strip(), False
    return text, False


@contextmanager
def timed(timings: dict[str, float], key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = round((time.perf_counter() - start) * 1000, 1)
