from __future__ import annotations

import logging
import time
import uuid

from ..config import Settings
from ..schemas import Candidate, GenerateRequest, GenerateResponse, OptimizeRequest, OptimizeResponse
from ..templates import user_input_context
from .generator import PromptGenerator
from .optimizer import PromptOptimizer
from .selector import ConfidenceSelector
from .utils import PipelineError, extract_prompt, timed

logger = logging.getLogger(__name__)


class PromptPipeline:
    def __init__(self, *, settings: Settings, generator: PromptGenerator,
                 selector: ConfidenceSelector, optimizer: PromptOptimizer):
        self.s = settings
        self.generator = generator
        self.selector = selector
        self.optimizer = optimizer

    def run(self, req: GenerateRequest) -> GenerateResponse:
        s, o = self.s, req.options
        rid = uuid.uuid4().hex
        start = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []

        use_sel = s.enable_candidate_selection if o.use_candidate_selection is None else o.use_candidate_selection
        n = min(o.num_candidates or s.num_candidates, s.max_candidates) if use_sel else 1
        method = o.confidence_method or s.confidence_method
        ui = req.user_input()
        ctx = user_input_context(ui, req.prompt_objective)

        # 1) user input + meta-prompt template → meta-prompt
        with timed(timings, "meta_prompt"):
            meta_prompt = self.generator.build_meta_prompt(ui, req.prompt_objective)

        # 2) meta-prompt → pretrained LLM → prompt candidates
        temperature = s.generator_temperature if n > 1 else s.single_candidate_temperature
        if n > 1 and temperature <= 0:
            temperature = 0.7
            warnings.append("generator_temperature <= 0 with multiple candidates; using 0.7")
        need_lp = use_sel and method in ("logprob", "hybrid")
        with timed(timings, "generation"):
            gens = self.generator.generate(meta_prompt, n=n, temperature=temperature, return_logprobs=need_lp)

        candidates = [
            Candidate(index=i, prompt=extract_prompt(g.text)[0], mean_logprob=g.mean_logprob, num_tokens=g.num_tokens)
            for i, g in enumerate(gens)
        ]

        # 3) (옵션) self-confidence 기반 선택
        method_used = None
        best = 0
        if use_sel:
            with timed(timings, "selection"):
                sr = self.selector.score([c.prompt for c in candidates], gens, method, ctx)
            warnings += sr.warnings
            method_used = sr.method
            for c, f, a, b in zip(candidates, sr.final, sr.logprob, sr.verbalized):
                c.confidence = f if c.prompt else -1.0
                c.logprob_confidence, c.verbalized_confidence = a, b
            best = max(range(len(candidates)), key=lambda i: candidates[i].confidence)
        candidates[best].selected = True
        initial_prompt = candidates[best].prompt
        if not initial_prompt:
            raise PipelineError("prompt generator produced an empty prompt")

        # 4) selected initial prompt → optimizer → optimized prompt
        if o.skip_optimization:
            optimized = initial_prompt
        else:
            with timed(timings, "optimization"):
                optimized, w = self.optimizer.optimize(initial_prompt, ui, req.prompt_objective)
            warnings += w

        timings["total"] = round((time.perf_counter() - start) * 1000, 1)
        logger.info("[%s] n=%d selection=%s method=%s best=%d timings=%s", rid, n, use_sel, method_used, best, timings)

        return GenerateResponse(
            request_id=rid,
            optimized_prompt=optimized,
            initial_prompt=initial_prompt,
            candidate_selection=use_sel,
            confidence_method=method_used,
            meta_prompt=meta_prompt if o.return_intermediate else None,
            candidates=candidates if o.return_intermediate else None,
            warnings=warnings,
            timings_ms=timings,
            models={
                "generator": self.generator.backend.name,
                "optimizer": "skipped" if o.skip_optimization else self.optimizer.backend.name,
            },
        )

    def optimize_only(self, req: OptimizeRequest) -> OptimizeResponse:
        timings: dict[str, float] = {}
        with timed(timings, "optimization"):
            optimized, warnings = self.optimizer.optimize(req.prompt, req.user_input, req.prompt_objective)
        return OptimizeResponse(
            request_id=uuid.uuid4().hex, optimized_prompt=optimized, warnings=warnings,
            timings_ms=timings, model=self.optimizer.backend.name,
        )
