"""Web server: serves the page and a /classify endpoint.

Run from this folder:  .\\.venv\\Scripts\\python -m uvicorn app:app --port 8001
Then open:             http://localhost:8001
"""
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import requests

from classifier import classify, get_available_embedding_models
from jev_classifier import JEV_SHORTLIST, classify_jev
from llm_classifier import DEFAULT_MODEL, classify_openai, get_available_llm_models

HERE = Path(__file__).resolve().parent
app = FastAPI(title="Intent Classifier")

class Request(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    threshold: float = Field(default=0.35, ge=-1, le=1)
    margin: float = Field(default=0.03, ge=0, le=2)
    classifier: str = Field(default="default")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    llm_model: str | None = None
    jev_shortlist: bool = False


@app.get("/")
def page():
    return FileResponse(HERE / "index.html")


@app.get("/models")
def models():
    embedding_models = get_available_embedding_models()
    try:
        llm_models = get_available_llm_models()
        llm_error = None if llm_models else "No OpenAI text models are available for this key."
    except requests.RequestException as error:
        llm_models = []
        llm_error = f"Could not load OpenAI models ({type(error).__name__})."

    return {
        "embeddingModels": embedding_models,
        "llmModels": llm_models,
        "llmError": llm_error,
        "jevShortlist": JEV_SHORTLIST,
    }


def _attach_usage(result):
    """Copy OpenAI-style token usage (Responses or Chat Completions naming) onto the result."""
    usage = result.get("usage") or result.get("jevDiagnostics", {}).get("response", {}).get("usage")
    if not isinstance(usage, dict):
        return
    output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    result["inputTokens"] = usage.get("input_tokens", usage.get("prompt_tokens"))
    result["outputTokens"] = usage.get("output_tokens", usage.get("completion_tokens"))
    result["reasoningTokens"] = output_details.get("reasoning_tokens") or 0


@app.post("/classify")
def run(request: Request):
    start = time.perf_counter()
    classifier_name = request.classifier.lower()

    if classifier_name == "jev":
        result = classify_jev(
            request.message,
            request.threshold,
            request.margin,
            shortlist=JEV_SHORTLIST if request.jev_shortlist else 0,
        )
        if not result.get("mock"):
            _attach_usage(result)
        # Loading the shortlist embedding model is one-time setup, excluded like the local classifier's.
        result["latencyMs"] = (time.perf_counter() - start) * 1000 - result.get("setupMs", 0)
    elif classifier_name == "openai":
        llm_model = request.llm_model or DEFAULT_MODEL
        result = classify_openai(
            request.message,
            llm_model,
            request.threshold,
            request.margin,
        )
        _attach_usage(result)
    else:
        try:
            result = classify(
                request.message,
                request.threshold,
                request.margin,
                model_name=request.embedding_model,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

    # The local embedding classifier reports its own inference-only latency.
    result.setdefault("latencyMs", (time.perf_counter() - start) * 1000)
    result["classifier"] = classifier_name
    return result

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
