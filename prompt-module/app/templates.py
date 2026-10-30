from __future__ import annotations

import json
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .schemas import UserInput


class PromptTemplates:
    """Jinja2 템플릿 렌더러. 파일 변경 시 자동 리로드되므로 볼륨 마운트로 템플릿 교체 가능."""

    def __init__(self, template_dir: str):
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, name: str, **ctx: Any) -> str:
        return self.env.get_template(name).render(**ctx).strip()


def _fmt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return json.dumps(value, ensure_ascii=False, indent=2)


def user_input_context(ui: UserInput | None, prompt_objective: str | None = None) -> dict[str, str | None]:
    """템플릿 변수: 공통 사용자 입력(정규화는 스키마 검증에서 완료) + 시스템 입력."""
    ctx: dict[str, str | None] = {"prompt_objective": prompt_objective}
    if ui is None:
        ctx.update(dict.fromkeys(["task_description", "requirements", "data_format", "data_schema", "data_label"]))
        return ctx
    ctx.update(
        task_description=ui.task_description,
        requirements=ui.requirements,
        data_format=ui.data_format,
        data_schema=_fmt(ui.data_schema),
        data_label=_fmt(ui.data_label),
    )
    return ctx
