"""후속 모듈이 준비되기 전 REST 전달을 확인하는 데모 수신기."""
from fastapi import FastAPI, HTTPException

app = FastAPI(title="DA Demo Receiver")
latest_result: dict | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/labels")
def receive(result: dict):
    global latest_result
    latest_result = result
    return {"status": "received", "request_id": result.get("request_id")}


@app.get("/last")
def last():
    if latest_result is None:
        raise HTTPException(status_code=404, detail="No result received yet")
    return latest_result
