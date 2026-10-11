"""Zero-shot intent classification with sentence embeddings.

The model never sees training examples. It only measures how similar the
customer's message is to a short sentence describing each intent.
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache" / "models"

AVAILABLE_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "multi-qa-mpnet-base-dot-v1": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
}
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# The 77 BANKING77 intent names. The model does not know these; we supply them.
LABELS = json.loads((HERE / "labels.json").read_text(encoding="utf-8"))

# Step 1: turn each label into an English sentence the model can understand.
SENTENCES = ["The customer asks for help with " + label.replace("_", " ") + "." for label in LABELS]

_model = None
_label_vectors = None
_current_model_name = DEFAULT_EMBEDDING_MODEL


def get_available_embedding_models():
    return list(AVAILABLE_EMBEDDING_MODELS.keys())


def _resolve_model_name(model_name):
    requested = (model_name or DEFAULT_EMBEDDING_MODEL).strip()
    if requested in AVAILABLE_EMBEDDING_MODELS:
        return requested
    for short_name, repo_name in AVAILABLE_EMBEDDING_MODELS.items():
        if repo_name == requested:
            return short_name
    raise ValueError(
        f"Unsupported embedding model '{model_name}'. Choose one of: {list(AVAILABLE_EMBEDDING_MODELS.keys())}"
    )


def load(model_name=None):
    """Load the chosen embedding model and precompute the intent vectors."""
    global _model, _label_vectors, _current_model_name

    resolved_name = _resolve_model_name(model_name)
    repo_name = AVAILABLE_EMBEDDING_MODELS[resolved_name]

    if _model is None or _current_model_name != resolved_name:
        from sentence_transformers import SentenceTransformer

        try:
            _model = SentenceTransformer(repo_name, cache_folder=str(CACHE_DIR), local_files_only=True)
        except OSError:
            _model = SentenceTransformer(repo_name, cache_folder=str(CACHE_DIR))

        _label_vectors = _model.encode(SENTENCES, normalize_embeddings=True, convert_to_numpy=True)
        _current_model_name = resolved_name

    return _model, _label_vectors, resolved_name


def classify(message, threshold=0.35, margin=0.03, model_name=None):
    # Loading the model and embedding the 77 intent sentences is one-time setup,
    # so it is reported separately and kept out of the latency figure.
    setup_start = time.perf_counter()
    model, label_vectors, resolved_name = load(model_name)
    setup_ms = (time.perf_counter() - setup_start) * 1000

    start = time.perf_counter()
    message_vector = model.encode(message, normalize_embeddings=True, convert_to_numpy=True)
    scores = label_vectors @ message_vector
    ranked = np.argsort(-scores)
    latency_ms = (time.perf_counter() - start) * 1000

    input_tokens = len(model.tokenizer(message, truncation=True, max_length=model.max_seq_length)["input_ids"])
    top, second = float(scores[ranked[0]]), float(scores[ranked[1]])
    abstained = top < threshold or (top - second) < margin

    return {
        "predictedIntent": LABELS[ranked[0]],
        "abstained": bool(abstained),
        "threshold": threshold,
        "margin": margin,
        "allScores": [{"label": LABELS[i], "sentence": SENTENCES[i], "score": float(scores[i])} for i in ranked],
        "model": resolved_name,
        "latencyMs": latency_ms,
        "setupMs": setup_ms,
        "inputTokens": input_tokens,
        "outputTokens": None,
    }
