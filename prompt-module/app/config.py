from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BackendType = Literal["hf", "openai", "mock"]
ConfidenceMethod = Literal["logprob", "verbalized", "hybrid"]


class Settings(BaseSettings):
    """모든 설정은 PM_ 접두사의 환경변수로 덮어쓸 수 있다."""

    model_config = SettingsConfigDict(env_prefix="PM_", env_file=".env", extra="ignore")

    # ---------- Prompt generator (pretrained LLM) ----------
    generator_backend: BackendType = "hf"
    generator_model: str = "Qwen/Qwen2.5-7B-Instruct"
    generator_base_url: str | None = None
    generator_api_key: str = "EMPTY"
    generator_temperature: float = 0.8          # 후보 여러 개 생성 시 샘플링 온도
    generator_top_p: float = 0.95
    generator_max_new_tokens: int = 1024
    single_candidate_temperature: float = 0.0   # 선택 off(후보 1개)일 때: 0이면 greedy → 재현성
    generator_system_prompt: str = "You are a world-class prompt engineer."

    # ---------- Prompt optimizer ----------
    optimizer_backend: BackendType = "hf"
    optimizer_model: str = "Qwen/Qwen2.5-7B-Instruct"
    optimizer_adapter: str | None = None        # LoRA 어댑터 경로 (선택)
    optimizer_base_url: str | None = None
    optimizer_api_key: str = "EMPTY"
    optimizer_temperature: float = 0.0
    optimizer_top_p: float = 1.0
    optimizer_max_new_tokens: int = 1536
    optimizer_input_mode: Literal["template", "raw"] = "template"
    optimizer_system_prompt: str = (
        "You are a prompt optimizer. You rewrite prompts so that LLMs follow them more accurately."
    )  # 빈 문자열이면 system 메시지 생략 (학습 포맷에 맞출 때)

    # ---------- Candidate selection ----------
    enable_candidate_selection: bool = True
    num_candidates: int = 4
    max_candidates: int = 16
    confidence_method: ConfidenceMethod = "logprob"
    hybrid_weight: float = 0.5                  # hybrid에서 logprob 점수의 가중치

    # ---------- Templates ----------
    template_dir: str = "templates"
    meta_prompt_template: str = "meta_prompt.j2"
    optimizer_template: str = "optimizer_prompt.j2"
    confidence_template: str = "confidence_prompt.j2"

    # ---------- HF runtime ----------
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device_map: str = "auto"
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    request_timeout: float = 300.0
    enable_thinking: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
