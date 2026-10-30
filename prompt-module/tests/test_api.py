import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update({
    "PM_GENERATOR_BACKEND": "mock",
    "PM_OPTIMIZER_BACKEND": "mock",
    "PM_TEMPLATE_DIR": str(ROOT / "templates"),
})

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.pipeline import extract_prompt  # noqa: E402

# 공통 사용자 입력 (AR 모듈과 동일한 필드명)
USER_INPUT = {
    "task_description": "Classify the sentiment of product reviews.",
    "requirements": "Output one label per review with a one-sentence reason, as JSON.",
    "data_format": ".CSV",
    "data_schema": {"review_id": "string", "review_text": "string"},
    "data_label": ["positive", "neutral", "negative"],
}
# 시스템 입력
OBJECTIVE = "Write a data augmentation prompt that paraphrases one review into three new reviews with the same label."
PAYLOAD = {**USER_INPUT, "prompt_objective": OBJECTIVE}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _post(client, **options):
    r = client.post("/v1/prompts/generate", json={**PAYLOAD, "options": options})
    assert r.status_code == 200, r.text
    return r.json()


def _post_payload(client, payload):
    r = client.post("/v1/prompts/generate", json={**payload, "options": {"use_candidate_selection": False}})
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_meta_prompt_rendering(client):
    res = _post(client, use_candidate_selection=False)
    meta = res["meta_prompt"]
    assert "Data format: csv" in meta and "negative" in meta
    assert OBJECTIVE in meta and USER_INPUT["requirements"] in meta


@pytest.mark.parametrize("method", ["logprob", "verbalized", "hybrid"])
def test_selection_on(client, method):
    res = _post(client, use_candidate_selection=True, num_candidates=3, confidence_method=method)
    cands = res["candidates"]
    assert res["candidate_selection"] is True and res["confidence_method"] == method
    assert len(cands) == 3 and sum(c["selected"] for c in cands) == 1
    best = next(c for c in cands if c["selected"])
    assert best["confidence"] == max(c["confidence"] for c in cands)
    assert res["initial_prompt"] == best["prompt"]
    assert res["optimized_prompt"]


def test_selection_off(client):
    res = _post(client, use_candidate_selection=False)
    assert res["candidate_selection"] is False and len(res["candidates"]) == 1
    assert res["candidates"][0]["confidence"] is None and "selection" not in res["timings_ms"]


def test_skip_optimization(client):
    res = _post(client, use_candidate_selection=False, skip_optimization=True)
    assert res["optimized_prompt"] == res["initial_prompt"] and res["models"]["optimizer"] == "skipped"


def test_no_intermediate(client):
    res = _post(client, return_intermediate=False)
    assert res["meta_prompt"] is None and res["candidates"] is None


def test_label_optional(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "data_label"}
    assert client.post("/v1/prompts/generate", json=payload).status_code == 200


def test_validation_error(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "task_description"}
    assert client.post("/v1/prompts/generate", json=payload).status_code == 422


def test_objective_required(client):
    payload = {k: v for k, v in PAYLOAD.items() if k != "prompt_objective"}
    assert client.post("/v1/prompts/generate", json=payload).status_code == 422


def test_legacy_requirement_field_rejected(client):
    payload = {**{k: v for k, v in PAYLOAD.items() if k != "requirements"}, "requirement": USER_INPUT["requirements"]}
    r = client.post("/v1/prompts/generate", json=payload)
    assert r.status_code == 422 and r.json()["detail"][0]["loc"][-1] == "requirements"


@pytest.mark.parametrize("field,value", [
    ("task_description", "   "), ("requirements", "x" * 20001), ("data_schema", {}), ("data_schema", " "),
])
def test_user_input_validation(client, field, value):
    assert client.post("/v1/prompts/generate", json={**PAYLOAD, field: value}).status_code == 422


def test_blank_objective_rejected(client):
    assert client.post("/v1/prompts/generate", json={**PAYLOAD, "prompt_objective": "  "}).status_code == 422


def test_empty_label_treated_as_absent(client):
    res = _post_payload(client, {**PAYLOAD, "data_label": []})
    assert "Labels:" not in res["meta_prompt"]


def test_unknown_fields_ignored(client):
    # 다른 모듈용 필드나 향후 추가 필드가 섞여 있어도 PO는 자기 필드만 사용
    res = _post_payload(client, {**PAYLOAD, "some_other_module_field": 1})
    assert res["optimized_prompt"]


def test_optimize_only(client):
    r = client.post("/v1/prompts/optimize", json={"prompt": "Classify: {{input_data}}"})
    assert r.status_code == 200 and r.json()["optimized_prompt"]


def test_optimize_only_with_context(client):
    r = client.post("/v1/prompts/optimize", json={"prompt": "Classify the review.", "user_input": USER_INPUT,
                                                  "prompt_objective": OBJECTIVE})
    assert r.status_code == 200 and r.json()["optimized_prompt"]


def test_extract_prompt():
    assert extract_prompt("<think>x</think>blah <prompt>\nHello\n</prompt>") == ("Hello", True)
    assert extract_prompt("```\nHi\n```") == ("Hi", False)
    assert extract_prompt("<prompt> cut off") == ("cut off", False)
