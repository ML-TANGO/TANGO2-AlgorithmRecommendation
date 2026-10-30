from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .config import get_settings
from .llm import build_backends
from .pipeline import ConfidenceSelector, PipelineError, PromptGenerator, PromptOptimizer, PromptPipeline
from .schemas import GenerateRequest, GenerateResponse, OptimizeRequest, OptimizeResponse
from .templates import PromptTemplates

logging.basicConfig(
    level=os.getenv("PM_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("prompt_module")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Loading models... generator=%s/%s optimizer=%s/%s",
                settings.generator_backend, settings.generator_model,
                settings.optimizer_backend, settings.optimizer_model)
    gen_backend, opt_backend = build_backends(settings)
    templates = PromptTemplates(settings.template_dir)
    app.state.settings = settings
    app.state.pipeline = PromptPipeline(
        settings=settings,
        generator=PromptGenerator(gen_backend, templates, settings),
        selector=ConfidenceSelector(gen_backend, templates, settings),  # self-confidence: generator 자신이 평가
        optimizer=PromptOptimizer(opt_backend, templates, settings),
    )
    logger.info("Ready.")
    yield


app = FastAPI(title="Prompt Generation & Optimization Module", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    p: PromptPipeline = request.app.state.pipeline
    return {"status": "ok", "generator": p.generator.backend.name, "optimizer": p.optimizer.backend.name}


@app.get("/v1/config")
def config(request: Request):
    s = request.app.state.settings
    return {
        "enable_candidate_selection": s.enable_candidate_selection,
        "num_candidates": s.num_candidates,
        "max_candidates": s.max_candidates,
        "confidence_method": s.confidence_method,
        "optimizer_input_mode": s.optimizer_input_mode,
        "templates": {"meta_prompt": s.meta_prompt_template, "optimizer": s.optimizer_template,
                      "confidence": s.confidence_template},
    }


# sync 핸들러 → FastAPI threadpool에서 실행되어 이벤트 루프를 막지 않음
@app.post("/v1/prompts/generate", response_model=GenerateResponse)
def generate_prompt(req: GenerateRequest, request: Request):
    try:
        return request.app.state.pipeline.run(req)
    except PipelineError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("pipeline failed")
        raise HTTPException(status_code=500, detail=f"pipeline failed: {e}")


@app.post("/v1/prompts/optimize", response_model=OptimizeResponse)
def optimize_prompt(req: OptimizeRequest, request: Request):
    """프롬프트 초안을 직접 넣어 optimizer 단계만 실행 (optimizer 단독 테스트용)."""
    try:
        return request.app.state.pipeline.optimize_only(req)
    except Exception as e:
        logger.exception("optimization failed")
        raise HTTPException(status_code=500, detail=f"optimization failed: {e}")
