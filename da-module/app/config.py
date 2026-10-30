from typing import Literal

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DA_", env_file=".env", extra="ignore")

    backend: Literal["hf", "mock"] = "hf"
    model: str = "google/gemma-4-E4B-it"
    device_map: str = "auto"
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    trust_remote_code: bool = False
    max_new_tokens: int = Field(512, ge=1)
    downstream_url: HttpUrl = "http://127.0.0.1:8003/v1/labels"
    downstream_timeout: float = Field(30.0, gt=0)
