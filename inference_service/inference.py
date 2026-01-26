from __future__ import annotations

import json
import os
import time
import uuid
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# In the container this file lives at /app/inference_service/inference.py.
# Treat /app as the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_repo_path(p: Path) -> Path:
    """If `p` is relative, resolve it relative to the repo root."""
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _as_numpy_1d(x: Any) -> np.ndarray:
    """Convert model output (torch/tf/numpy/list) to 1D numpy array."""
    # Torch tensor
    try:
        import torch

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    # TensorFlow tensor
    try:
        import tensorflow as tf

        if isinstance(x, (tf.Tensor,)):
            x = x.numpy()
    except Exception:
        pass

    arr = np.asarray(x)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32)


def _to_probabilities(output_vec: Any) -> np.ndarray:
    """Return a length-10 probability vector.

    Heuristic:
    - If vector looks like probabilities (all in [0,1] and sums ~1), return as-is.
    - Otherwise apply softmax.
    """
    v = _as_numpy_1d(output_vec)
    if v.shape[0] != 10:
        raise ValueError(f"Expected 10 outputs, got shape={v.shape}")

    if np.all(v >= -1e-6) and np.all(v <= 1 + 1e-6):
        s = float(np.sum(v))
        if abs(s - 1.0) < 1e-2:
            # Already probs
            v = np.clip(v, 0.0, 1.0)
            return v / np.sum(v)

    return _softmax(v)


def _list_files(directory: Path, ext: str) -> List[Path]:
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and p.name.endswith(ext)])


@dataclass(frozen=True)
class PreprocessConfig:
    """Controls how we turn 28x28 (784) 0..255 pixels into model inputs."""

    # If true: pixels /= 255.0
    divide_by_255: bool = True

    # If true: pixels = (pixels - mean) / std
    standardize: bool = True

    # If divide_by_255=True, typical MNIST-ish values are ~0.1307 / 0.3081.
    # If divide_by_255=False (raw 0..255), mean/std will be much larger.
    mean: float = 0.1307
    std: float = 0.3081

    # If true, invert (black/white) by doing 255 - pixel (before scaling)
    invert: bool = False

    @staticmethod
    def load_from_file(path: Path) -> "PreprocessConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return PreprocessConfig(
            divide_by_255=bool(data.get("divide_by_255", True)),
            standardize=bool(data.get("standardize", True)),
            mean=float(data.get("mean", 0.1307)),
            std=float(data.get("std", 0.3081)),
            invert=bool(data.get("invert", False)),
        )


@dataclass
class ModelResult:
    model_id: str
    framework: str
    predicted: int
    probabilities: List[float]
    confidence: float
    latency_ms: float


class InferenceEngine:
    """Loads all available models once, and serves predictions for requests."""

    def __init__(
        self,
        models_dir: str = "models",
        scratch_pickle: Optional[str] = "models/my_model.pkl",
        preprocess_config_path: Optional[str] = None,
    ) -> None:
        self.models_dir = _resolve_repo_path(Path(models_dir))
        self.scratch_pickle = _resolve_repo_path(Path(scratch_pickle)) if scratch_pickle else None

        self.preprocess = self._load_preprocess(preprocess_config_path)

        # Each entry: (model_id, framework, predict_fn)
        self._models: List[Tuple[str, str, Callable[[np.ndarray], Any]]] = []

        self._load_models()

    def _load_preprocess(self, preprocess_config_path: Optional[str]) -> PreprocessConfig:
        # 1) explicit path
        if preprocess_config_path:
            p = Path(preprocess_config_path)
            if p.exists():
                logger.info("Loading preprocess config from %s", p)
                return PreprocessConfig.load_from_file(p)

        # 2) env var
        env_path = os.environ.get("PREPROCESS_CONFIG")
        if env_path and Path(env_path).exists():
            p = Path(env_path)
            logger.info("Loading preprocess config from PREPROCESS_CONFIG=%s", p)
            return PreprocessConfig.load_from_file(p)

        # 3) default file next to models
        default = self.models_dir / "preprocess.json"
        if default.exists():
            logger.info("Loading preprocess config from %s", default)
            return PreprocessConfig.load_from_file(default)

        # 4) hardcoded fallback
        logger.warning("No preprocess config found; using placeholder mean/std.")
        return PreprocessConfig()

    def _load_models(self) -> None:
        scratch_state_path = self.models_dir / "scratch_model_state.json"
        if scratch_state_path.exists():
            try:
                from neural_network import NeuralNetwork

                logger.info("Loading scratch weights-only state from %s", scratch_state_path)
                state = json.loads(scratch_state_path.read_text(encoding="utf-8"))

                # Bypass __init__ (which expects training/test data)
                scratch_nn = NeuralNetwork.__new__(NeuralNetwork)
                scratch_nn.load_state_dict(state)

                def scratch_state_predict(x_1d: np.ndarray) -> Any:
                    scratch_nn.forward_pass(x_1d)
                    return scratch_nn.output_layer

                self._models.append(("scratch:state_dict", "scratch", scratch_state_predict))
            except Exception as e:
                logger.exception("Failed to load scratch_model_state.json: %s", e)

        # Keras (.keras)
        keras_files = _list_files(self.models_dir, ".keras")
        if keras_files:
            try:
                import tensorflow as tf
                import keras

                for path in keras_files:
                    model_id = f"keras:{path.stem}"
                    logger.info("Loading Keras model %s from %s", model_id, path)
                    model = keras.saving.load_model(str(path))

                    def make_predict_fn(m):
                        def _predict(x_1d: np.ndarray) -> Any:
                            x = x_1d.reshape(28, 28)
                            x = np.expand_dims(x, axis=0)
                            t = tf.convert_to_tensor(x, dtype=tf.float32)
                            out = m(t, training=False)
                            return out[0]

                        return _predict

                    self._models.append((model_id, "keras", make_predict_fn(model)))
            except Exception as e:
                logger.exception("Keras models present but could not be loaded: %s", e)

        # PyTorch (.pt/.pth)
        torch_files = _list_files(self.models_dir, ".pt") + _list_files(self.models_dir, ".pth")
        if torch_files:
            try:
                import torch
                import torch.nn as torch_nn

                class MLP(torch_nn.Module):
                    def __init__(self, hidden: int = 128):
                        super().__init__()
                        self.net = torch_nn.Sequential(
                            torch_nn.Linear(784, hidden),
                            torch_nn.ReLU(),
                            torch_nn.Linear(hidden, 10),
                        )

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return self.net(x)

                # If a meta file exists, we can infer hidden size for the MLP.
                meta_path = self.models_dir / "torch_mlp_meta.json"
                hidden = 128
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        hidden = int(meta.get("hidden", hidden))
                    except Exception:
                        pass

                for path in torch_files:
                    model_id = f"torch:{path.stem}"
                    logger.info("Loading Torch model %s from %s", model_id, path)

                    # Prefer treating this as a state_dict for our MLP.
                    sd = None
                    try:
                        sd = torch.load(str(path), map_location="cpu", weights_only=True)
                    except TypeError:
                        # Older torch may not support weights_only
                        sd = torch.load(str(path), map_location="cpu")

                    model = MLP(hidden=hidden)
                    if isinstance(sd, dict) and all(isinstance(k, str) for k in sd.keys()):
                        model.load_state_dict(sd)
                    else:
                        # If someone saved a full model object, allow explicit opt-in.
                        if os.environ.get("TORCH_LOAD_FULL_OBJECT", "0") == "1":
                            model = torch.load(str(path), map_location="cpu")
                        else:
                            raise RuntimeError(
                                f"Unexpected torch artifact in {path.name}. Expected a state_dict; "
                                "set TORCH_LOAD_FULL_OBJECT=1 to allow full-object loading."
                            )

                    model.eval()

                    def make_predict_fn(m):
                        def _predict(x_1d: np.ndarray) -> Any:
                            with torch.no_grad():
                                x1 = np.expand_dims(x_1d, axis=0).astype(np.float32)
                                t1 = torch.tensor(x1, dtype=torch.float32)
                                out = m(t1)
                                return out[0]

                        return _predict

                    self._models.append((model_id, "torch", make_predict_fn(model)))
            except Exception as e:
                logger.exception("Torch models present but could not be loaded: %s", e)

        if not self._models:
            logger.error(
                "No models were loaded. Check your models/ directory (%s) and scratch pickle path (%s).",
                self.models_dir,
                self.scratch_pickle,
            )
            return

        logger.info("Loaded %d models: %s", len(self._models), [m[0] for m in self._models])

    def describe_models(self) -> Dict[str, Any]:
        return {
            "models": [
                {"model_id": model_id, "framework": fw}
                for (model_id, fw, _) in self._models
            ],
            "preprocess": {
                "divide_by_255": self.preprocess.divide_by_255,
                "standardize": self.preprocess.standardize,
                "mean": self.preprocess.mean,
                "std": self.preprocess.std,
                "invert": self.preprocess.invert,
            },
        }

    def _preprocess_pixels(self, pixels: List[float], invert_override: Optional[bool] = None) -> np.ndarray:
        arr = np.asarray(pixels, dtype=np.float32)
        if arr.size != 28 * 28:
            raise ValueError(f"Expected 784 pixels, got {arr.size}")

        # Clamp for safety
        arr = np.clip(arr, 0.0, 255.0)

        invert = self.preprocess.invert if invert_override is None else bool(invert_override)
        if invert:
            arr = 255.0 - arr

        if self.preprocess.divide_by_255:
            arr = arr / 255.0

        if self.preprocess.standardize:
            std = float(self.preprocess.std) if float(self.preprocess.std) != 0.0 else 1.0
            arr = (arr - float(self.preprocess.mean)) / std

        return arr.astype(np.float32)

    def predict_all(
        self,
        pixels: List[float],
        invert_override: Optional[bool] = None,
        model_select: str = "all",
        model_subset: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run inference for all loaded models (or a selection/subset) and return per-model + ensemble.

        Selection precedence:
        1) If `model_subset` is provided: run exactly those model_ids.
        2) Else use `model_select` in {all, scratch, keras, torch}.
        """
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        if not self._models:
            raise RuntimeError("No models are loaded on this server instance.")

        x_1d = self._preprocess_pixels(pixels, invert_override=invert_override)

        selected = self._models

        # 1) Legacy / explicit override: exact model ids
        if model_subset:
            wanted = set(model_subset)
            selected = [m for m in self._models if m[0] in wanted]
            if not selected:
                raise ValueError("model_subset specified but no matching model_ids were found")

        # 2) New selector: all | scratch | keras | torch
        else:
            sel = (model_select or "all").strip().lower()
            if sel in ("", "all"):
                selected = self._models
            elif sel in ("scratch", "keras", "torch"):
                # Prefer framework match, but also allow prefix match as a fallback.
                prefix_map = {"scratch": "scratch:", "keras": "keras:", "torch": "torch:"}
                prefix = prefix_map[sel]
                selected = [m for m in self._models if (m[1] == sel) or str(m[0]).startswith(prefix)]
                if not selected:
                    raise ValueError(f"model_select={model_select} but no matching models were found")
            else:
                raise ValueError(f"Unknown model_select: {model_select}")

        results: List[ModelResult] = []
        probs_for_ensemble: List[np.ndarray] = []

        for model_id, fw, predict_fn in selected:
            t_model0 = time.perf_counter()
            out = predict_fn(x_1d)
            probs = _to_probabilities(out)
            pred = int(np.argmax(probs))
            conf = float(np.max(probs))
            latency_ms = (time.perf_counter() - t_model0) * 1000.0

            results.append(
                ModelResult(
                    model_id=model_id,
                    framework=fw,
                    predicted=pred,
                    probabilities=[float(p) for p in probs.tolist()],
                    confidence=conf,
                    latency_ms=latency_ms,
                )
            )
            probs_for_ensemble.append(probs)

        total_ms = (time.perf_counter() - t0) * 1000.0

        # Simple equal-weight ensemble
        ensemble_probs = np.mean(np.stack(probs_for_ensemble, axis=0), axis=0)
        ensemble_pred = int(np.argmax(ensemble_probs))
        ensemble_conf = float(np.max(ensemble_probs))

        # "Best guess" model = highest single-model confidence
        best = max(results, key=lambda r: r.confidence)

        return {
            "request_id": request_id,
            "latency_ms_total": total_ms,
            "preprocess": {
                "divide_by_255": self.preprocess.divide_by_255,
                "standardize": self.preprocess.standardize,
                "mean": self.preprocess.mean,
                "std": self.preprocess.std,
                "invert_effective": self.preprocess.invert if invert_override is None else bool(invert_override),
            },
            "ensemble": {
                "predicted": ensemble_pred,
                "confidence": ensemble_conf,
                "probabilities": [float(p) for p in ensemble_probs.tolist()],
            },
            "best_single_model": {
                "model_id": best.model_id,
                "predicted": best.predicted,
                "confidence": best.confidence,
            },
            "models": [
                {
                    "model_id": r.model_id,
                    "framework": r.framework,
                    "predicted": r.predicted,
                    "confidence": r.confidence,
                    "probabilities": r.probabilities,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ],
        }
