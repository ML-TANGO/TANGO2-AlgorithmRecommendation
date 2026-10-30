from __future__ import annotations

from ..config import Settings
from ..llm import Generation, LLMBackend
from ..schemas import UserInput
from ..templates import PromptTemplates, user_input_context


class PromptGenerator:
    """user input + meta-prompt template → meta-prompt → pretrained LLM → prompt candidates"""

    def __init__(self, backend: LLMBackend, templates: PromptTemplates, settings: Settings):
        self.backend = backend
        self.templates = templates
        self.s = settings

    def build_meta_prompt(self, ui: UserInput, prompt_objective: str) -> str:
        return self.templates.render(self.s.meta_prompt_template, **user_input_context(ui, prompt_objective))

    def generate(self, meta_prompt: str, *, n: int, temperature: float, return_logprobs: bool) -> list[Generation]:
        messages = []
        if self.s.generator_system_prompt:
            messages.append({"role": "system", "content": self.s.generator_system_prompt})
        messages.append({"role": "user", "content": meta_prompt})
        return self.backend.generate(
            messages,
            n=n,
            temperature=temperature,
            top_p=self.s.generator_top_p,
            max_new_tokens=self.s.generator_max_new_tokens,
            return_logprobs=return_logprobs,
        )
