import logging
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .labeling import build_explicit_plan, build_messages, build_plan_messages, parse_annotation, parse_plan
from .llm import build_backend
from .schemas import Delivery, LabelRequest, LabelResponse, LabelResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("da_module")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.backend = build_backend(settings)
    app.state.settings = settings
    with httpx.Client(timeout=settings.downstream_timeout) as downstream:
        app.state.downstream = downstream
        logger.info("Ready. model=%s", app.state.backend.name)
        yield


app = FastAPI(title="Data Augmentation / Labeling Module", version="2.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model": request.app.state.backend.name}


@app.post("/v1/labels/generate", response_model=LabelResponse)
def generate_label(payload: LabelRequest, request: Request):
    backend = request.app.state.backend
    settings = request.app.state.settings
    if payload.output_schema is not None:
        plan = build_explicit_plan(payload)
    else:
        try:
            plan_text = backend.generate(build_plan_messages(payload))
        except Exception:
            logger.exception("Labeling plan generation failed")
            raise HTTPException(status_code=500, detail="Labeling plan generation failed; see server logs")
        try:
            plan = parse_plan(plan_text, payload)
        except ValueError as error:
            raise HTTPException(status_code=502, detail=f"Invalid DA labeling plan: {error}") from error

    try:
        text = backend.generate(build_messages(payload, plan))
    except Exception:
        logger.exception("Label generation failed")
        raise HTTPException(status_code=500, detail="Label generation failed; see server logs")
    try:
        annotation = parse_annotation(text, plan)
    except ValueError as error:
        raise HTTPException(status_code=502, detail=f"Invalid model output: {error}") from error

    result = LabelResult(
        request_id=uuid.uuid4().hex,
        po_request_id=payload.po_request_id,
        raw_data=payload.raw_data,
        annotation=annotation,
        model=backend.name,
    )
    try:
        remote = request.app.state.downstream.post(
            str(settings.downstream_url), json=result.model_dump(mode="json")
        )
        remote.raise_for_status()
        delivery = Delivery(status="delivered", status_code=remote.status_code)
    except httpx.HTTPStatusError as error:
        delivery = Delivery(
            status="failed", status_code=error.response.status_code,
            error=f"Downstream returned HTTP {error.response.status_code}",
        )
    except httpx.RequestError as error:
        delivery = Delivery(status="failed", error=f"Downstream request failed: {error}")

    response = LabelResponse(labeling_plan=plan, result=result, delivery=delivery)
    if delivery.status == "failed":
        return JSONResponse(status_code=502, content=response.model_dump(mode="json"))
    return response
