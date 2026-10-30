from __future__ import annotations

import logging

from ..config import Settings
from .base import Generation, LLMBackend, Message

logger = logging.getLogger(__name__)

__all__ = ["Generation", "LLMBackend", "Message", "build_backends"]


def _build_remote(kind: str, model: str, base_url: str | None, api_key: str, s: Settings) -> LLMBackend:
    if kind == "mock":
        from .mock_backend import MockBackend

        return MockBackend(model)
    if kind == "openai":
        from .openai_backend import OpenAICompatBackend

        return OpenAICompatBackend(model, base_url=base_url, api_key=api_key, timeout=s.request_timeout)
    raise ValueError(f"unknown backend: {kind}")


def build_backends(s: Settings) -> tuple[LLMBackend, LLMBackend]:
    """(generator, optimizer) 백엔드 생성. HF 백엔드는 동일 베이스 모델이면 가중치를 공유한다."""
    handles: dict = {}

    def hf(model: str, adapter: str | None):
        from .hf_backend import HFModelHandle

        key = (model, adapter)
        if key not in handles:
            handles[key] = HFModelHandle(
                model, adapter=adapter, dtype=s.torch_dtype, device_map=s.device_map,
                load_in_4bit=s.load_in_4bit, trust_remote_code=s.trust_remote_code,
                enable_thinking=s.enable_thinking,
            )
        return handles[key]

    # optimizer
    if s.optimizer_backend == "hf":
        from .hf_backend import HFBackend

        optimizer = HFBackend(hf(s.optimizer_model, s.optimizer_adapter), use_adapter=True)
    else:
        optimizer = _build_remote(s.optimizer_backend, s.optimizer_model, s.optimizer_base_url, s.optimizer_api_key, s)

    # generator
    if s.generator_backend == "hf":
        from .hf_backend import HFBackend

        if s.optimizer_backend == "hf" and s.generator_model == s.optimizer_model:
            logger.info("Generator shares base weights with optimizer (adapter disabled for generator)")
            generator = HFBackend(hf(s.optimizer_model, s.optimizer_adapter), use_adapter=False)
        else:
            generator = HFBackend(hf(s.generator_model, None), use_adapter=False)
    else:
        generator = _build_remote(s.generator_backend, s.generator_model, s.generator_base_url, s.generator_api_key, s)

    return generator, optimizer
