from typing import Any, Literal

from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


def check_output_schema(schema: dict[str, Any]) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ValueError(f"Invalid output_schema: {error.message}") from error


class LabelRequest(BaseModel):
    # Unknown fields (such as the removed user_input) are rejected so old callers fail loudly.
    model_config = ConfigDict(extra="forbid")

    po_request_id: str | None = None
    optimized_prompt: str = Field(min_length=1, description="Data augmentation prompt optimized by PO")
    raw_data: dict[str, Any] | str
    output_format: Literal["json", "text"] | None = Field(
        default=None, description="Optional DA output format override; omit to let DA choose."
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional DA output schema override. DA builds its own schema when omitted.",
    )

    @field_validator("output_schema")
    @classmethod
    def validate_output_schema(cls, value):
        if value is not None:
            check_output_schema(value)
        return value

    @field_validator("optimized_prompt")
    @classmethod
    def validate_prompt(cls, value):
        if not value.strip():
            raise ValueError("optimized_prompt must not be blank")
        return value

    @field_validator("raw_data")
    @classmethod
    def validate_record(cls, value):
        if not value or isinstance(value, str) and not value.strip():
            raise ValueError("raw_data must contain one non-empty record")
        return value


class LabelingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1)
    output_format: Literal["json", "text"]
    output_schema: dict[str, Any]

    @field_validator("instruction")
    @classmethod
    def validate_instruction(cls, value):
        if not value.strip():
            raise ValueError("instruction must not be blank")
        return value

    @field_validator("output_schema")
    @classmethod
    def validate_output_schema(cls, value):
        check_output_schema(value)
        return value


class LabelResult(BaseModel):
    request_id: str
    po_request_id: str | None = None
    raw_data: dict[str, Any] | str
    annotation: JsonValue
    model: str


class Delivery(BaseModel):
    status: Literal["delivered", "failed"]
    status_code: int | None = None
    error: str | None = None


class LabelResponse(BaseModel):
    labeling_plan: LabelingPlan
    result: LabelResult
    delivery: Delivery
