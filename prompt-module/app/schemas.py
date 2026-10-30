from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TEXT_MAX = 20000  # AR 모듈과 동일한 텍스트 길이 상한


class UserInput(BaseModel):
    """
    공통 사용자 입력 블록. PO와 AR이 같은 필드명·의미로 사용한다.
      - PO: 전체 필드 사용
      - AR: task_description, requirements만 사용 (나머지는 무시)
    모르는 필드는 무시하므로, 같은 요청 본문을 두 모듈에 그대로 보낼 수 있다.
    """

    model_config = ConfigDict(extra="ignore")

    task_description: str = Field(..., min_length=1, max_length=TEXT_MAX, description="사용자 태스크 설명 (text)")
    requirements: str = Field(..., min_length=1, max_length=TEXT_MAX, description="사용자 요구사항 (text)")
    data_format: str = Field(..., min_length=1, description="데이터 형식", examples=["csv", "json", "image + csv"])
    data_schema: str | dict[str, Any] | list[Any] = Field(..., description="데이터 스키마 (문자열 또는 JSON)")
    data_label: str | list[Any] | dict[str, Any] | None = Field(None, description="데이터 라벨 (선택)")

    @field_validator("task_description", "requirements")
    @classmethod
    def _text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("내용을 입력해 주세요.")
        return v

    @field_validator("data_format")
    @classmethod
    def _format(cls, v: str) -> str:
        v = v.strip().lstrip(".").lower()
        if not v:
            raise ValueError("데이터 형식을 입력해 주세요.")
        return v

    @field_validator("data_schema")
    @classmethod
    def _schema(cls, v):
        if isinstance(v, str):
            v = v.strip()
        if not v:
            raise ValueError("데이터 스키마를 입력해 주세요.")
        return v

    @field_validator("data_label")
    @classmethod
    def _label(cls, v):
        if isinstance(v, str):
            v = v.strip()
        return v or None  # 빈 문자열/목록/객체는 '라벨 없음'으로 처리


class PipelineOptions(BaseModel):
    use_candidate_selection: bool | None = Field(None, description="후보 생성+self-confidence 선택 on/off (미지정 시 서버 기본값)")
    num_candidates: int | None = Field(None, ge=1, description="생성할 후보 수")
    confidence_method: Literal["logprob", "verbalized", "hybrid"] | None = None
    skip_optimization: bool = Field(False, description="optimizer 단계를 건너뜀 (테스트용)")
    return_intermediate: bool = Field(True, description="meta-prompt, 후보 목록 등 중간 산출물 반환")


class GenerateRequest(UserInput):
    """PO 요청 = 공통 사용자 입력 + 시스템 입력(prompt_objective) + 실행 옵션."""

    prompt_objective: str = Field(
        ..., min_length=1,
        description="[시스템 입력] 생성할 프롬프트의 역할. 사용자가 아닌 시스템(호출자)이 고정값으로 지정",
    )
    options: PipelineOptions = Field(default_factory=PipelineOptions)

    @field_validator("prompt_objective")
    @classmethod
    def _objective(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("prompt_objective가 비어 있습니다.")
        return v

    def user_input(self) -> UserInput:
        return UserInput.model_validate(self.model_dump(include=set(UserInput.model_fields)))


class OptimizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="최적화할 프롬프트 초안")
    user_input: UserInput | None = Field(None, description="태스크 컨텍스트 (선택, optimizer_input_mode=template일 때만 사용)")
    prompt_objective: str | None = Field(None, description="[시스템 입력] 프롬프트 역할 (선택)")


class Candidate(BaseModel):
    index: int
    prompt: str
    confidence: float | None = None
    logprob_confidence: float | None = None
    verbalized_confidence: float | None = None
    mean_logprob: float | None = None
    num_tokens: int | None = None
    selected: bool = False


class GenerateResponse(BaseModel):
    request_id: str
    optimized_prompt: str
    initial_prompt: str
    candidate_selection: bool
    confidence_method: str | None = None
    meta_prompt: str | None = None
    candidates: list[Candidate] | None = None
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float]
    models: dict[str, str]


class OptimizeResponse(BaseModel):
    request_id: str
    optimized_prompt: str
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float]
    model: str
