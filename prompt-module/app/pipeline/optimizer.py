from __future__ import annotations

from ..config import Settings
from ..llm import LLMBackend
from ..schemas import UserInput
from ..templates import PromptTemplates, user_input_context
from .utils import extract_prompt


class PromptOptimizer:
    """selected initial prompt → optimizer model → optimized prompt"""

    def __init__(self, backend: LLMBackend, templates: PromptTemplates, settings: Settings):
        self.backend = backend
        self.templates = templates
        self.s = settings

    def optimize(
        self, initial_prompt: str, ui: UserInput | None = None, prompt_objective: str | None = None
    ) -> tuple[str, list[str]]:
        s = self.s
        if s.optimizer_input_mode == "raw":
            content = initial_prompt  # optimizer가 프롬프트 단독 입력으로 학습된 경우
        else:
            content = self.templates.render(
                s.optimizer_template, initial_prompt=initial_prompt, **user_input_context(ui, prompt_objective)
            )
        messages = []
        if s.optimizer_system_prompt:
            messages.append({"role": "system", "content": s.optimizer_system_prompt})
        messages.append({"role": "user", "content": content})

        out = self.backend.generate(
            messages, n=1, temperature=s.optimizer_temperature,
            top_p=s.optimizer_top_p, max_new_tokens=s.optimizer_max_new_tokens,
        )[0]
        prompt, tagged = extract_prompt(out.text)

        warnings: list[str] = []
        if not prompt:
            warnings.append("optimizer returned empty output; returning the initial prompt")
            prompt = initial_prompt
        elif not tagged and s.optimizer_input_mode == "template":
            warnings.append("optimizer output had no <prompt> tags; returned raw output")
        return prompt, warnings
