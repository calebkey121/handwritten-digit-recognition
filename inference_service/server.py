from __future__ import annotations

import os
import time
import threading
from typing import List, Optional
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference_service.inference import InferenceEngine


class PredictRequest(BaseModel):
    pixels: List[float] = Field(..., description="Length-784 list of 0..255 pixel intensities")
    invert: Optional[bool] = Field(None, description="Override inversion for this request")
    model_select: str = Field(
        "all",
        description="Which model family to run: all | scratch | keras | torch",
    )
    model_subset: Optional[List[str]] = Field(
        None,
        description="(Legacy) Optional list of model_id strings to run; omit to run all models",
    )


app = FastAPI(title="Digit Inference API", version="0.1")

# CORS
# Use explicit origins (required if the browser request includes credentials).
# You can override with ALLOWED_ORIGINS as a comma-separated list.
_default_origins = [
    "https://calebkey121.github.io",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
]
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
allowed_origins = (
    [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
    if _allowed_origins_env
    else _default_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)

ENGINE = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> InferenceEngine:
    """Lazily initialize the inference engine once per process."""
    global ENGINE
    if ENGINE is not None:
        return ENGINE

    with _ENGINE_LOCK:
        if ENGINE is not None:
            return ENGINE
        ENGINE = InferenceEngine(
            models_dir=os.environ.get("MODELS_DIR", "models"),
            scratch_pickle=os.environ.get("SCRATCH_MODEL_PATH", "models/my_model.pkl"),
            preprocess_config_path=os.environ.get("PREPROCESS_CONFIG"),
        )
        return ENGINE


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    engine = get_engine()
    return engine.describe_models()


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        engine = get_engine()
        t0 = time.perf_counter()
        try:
            out = engine.predict_all(
                pixels=req.pixels,
                invert_override=req.invert,
                model_select=req.model_select,
                model_subset=req.model_subset,
            )
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {type(e).__name__}")

        out["server_overhead_ms"] = (time.perf_counter() - t0) * 1000.0
        return out
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Inference failed")  # prints full traceback to logs
        raise HTTPException(status_code=500, detail=f"Inference failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("inference_service.server:app", host="0.0.0.0", port=port, reload=True)
