"""TypeSafe JEV intent classifier adapter.

This module calls the JEV API when a valid key is present, and falls back to a
mock result when the key is missing or the API is unavailable. The fallback keeps
app behavior deterministic without breaking the UI.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

JEV_KEY = (os.getenv("JEV_KEY") or "").strip()
JEV_MODEL = (os.getenv("JEV_MODEL") or "jev-latest").strip()
JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
# JEV scores every option it is given, so tokens grow by ~10 input and ~10 output
# per label. The shortlist option sends only this many labels, chosen by the
# local embedding model.
DEFAULT_LABELS = json.loads((HERE / "labels.json").read_text(encoding="utf-8"))
JEV_SHORTLIST = int(os.getenv("JEV_SHORTLIST") or 15)
if not 0 < JEV_SHORTLIST < len(DEFAULT_LABELS):
    JEV_SHORTLIST = 15


def _mock_result(threshold=0.35, margin=0.03, model_name=None, reason="JEV did not return a usable response.", attempts=None):
    label = "mock_intent"
    return {
        "predictedIntent": label,
        "abstained": True,
        "threshold": threshold,
        "margin": margin,
        "allScores": [{"label": label, "score": 0.0}],
        "model": model_name or JEV_MODEL,
        "source": "MOCK – not a result",
        "classifier": "jev",
        "mock": True,
        "jevDiagnostics": {
            "mode": "mock",
            "reason": reason,
            "attempts": attempts or [],
        },
    }


def _normalize_response(payload, threshold, margin, model_name, status_code):
    answers = payload.get("answers") if isinstance(payload, dict) else None
    answer = answers.get("intent") if isinstance(answers, dict) else None
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None

    if not isinstance(answer, dict) or not isinstance(probabilities, dict) or not probabilities:
        return _mock_result(
            threshold,
            margin,
            model_name,
            reason="The TypeSafe response did not contain an intent Choice answer with probabilities.",
            attempts=[{"httpStatus": status_code, "response": payload}],
        )

    scores = [{"label": label, "score": float(probability)} for label, probability in probabilities.items()]
    scores.sort(key=lambda item: item["score"], reverse=True)
    predicted = answer.get("choice") or scores[0]["label"]
    top_score = scores[0]["score"]
    second_score = scores[1]["score"] if len(scores) > 1 else 0.0
    confidence = float(answer.get("confidence", 0.0))

    return {
        "predictedIntent": predicted,
        "abstained": confidence < threshold or (top_score - second_score) < margin,
        "threshold": threshold,
        "margin": margin,
        "confidence": confidence,
        "allScores": scores,
        "model": payload.get("model") or model_name,
        "classifier": "jev",
        "source": "TypeSafe System One API",
        "mock": False,
        "jevDiagnostics": {
            "mode": "api",
            "endpoint": JEV_API_URL,
            "httpStatus": status_code,
            "response": payload,
        },
    }


def _shortlist_labels(message, size):
    """Return (labels, one-time model setup ms). Setup is reported so it can be excluded from latency."""
    if size <= 0 or size >= len(DEFAULT_LABELS):
        return DEFAULT_LABELS, 0.0
    from classifier import classify

    ranked = classify(message)
    return [item["label"] for item in ranked["allScores"][:size]], ranked["setupMs"]


def _call_jev_api(message, threshold, margin, model_name, shortlist):
    headers = {
        "Authorization": f"Bearer {JEV_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # Options are BANKING77 intent names only. A null description sends the
    # name alone, as documented for Choice criteria.
    shortlist_start = time.perf_counter()
    labels, setup_ms = _shortlist_labels(message, shortlist)
    shortlist_ms = (time.perf_counter() - shortlist_start) * 1000 - setup_ms
    criteria = {label: None for label in labels}
    body = {
        "state": message,
        "model": model_name,
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "Which banking intent best describes the customer's message? Select the single best match.",
                "criteria": criteria,
            }
        },
    }

    try:
        response = requests.post(JEV_API_URL, json=body, headers=headers, timeout=30)
        try:
            response_body = response.json()
        except ValueError:
            response_body = {"body": response.text[:4000]}

        if response.status_code >= 400:
            return _mock_result(
                threshold,
                margin,
                model_name,
                reason=f"TypeSafe API returned HTTP {response.status_code}.",
                attempts=[{
                    "endpoint": JEV_API_URL,
                    "httpStatus": response.status_code,
                    "response": response_body,
                }],
            )

        if not isinstance(response_body, dict):
            return _mock_result(
                threshold,
                margin,
                model_name,
                reason="TypeSafe API returned JSON that was not an object.",
                attempts=[{"endpoint": JEV_API_URL, "httpStatus": response.status_code, "response": response_body}],
            )

        result = _normalize_response(response_body, threshold, margin, model_name, response.status_code)
        result["jevDiagnostics"]["optionsSent"] = labels
        result["setupMs"] = setup_ms
        if len(labels) < len(DEFAULT_LABELS):
            # Local pre-processing time, included in the reported latency.
            result["shortlistMs"] = shortlist_ms
        return result
    except requests.RequestException as error:
        return _mock_result(
            threshold,
            margin,
            model_name,
            reason=f"Could not connect to the TypeSafe API ({type(error).__name__}).",
            attempts=[{"endpoint": JEV_API_URL, "error": str(error)[:400]}],
        )


def classify_jev(message, threshold=0.35, margin=0.03, model_name=None, shortlist=0):
    if os.getenv("FORCE_MOCK") == "1":
        return _mock_result(
            threshold,
            margin,
            model_name or JEV_MODEL,
            reason="FORCE_MOCK=1 is enabled.",
        )

    if not JEV_KEY:
        return _mock_result(
            threshold,
            margin,
            model_name or JEV_MODEL,
            reason="JEV_KEY is not configured in .env.",
        )

    return _call_jev_api(message, threshold, margin, model_name or JEV_MODEL, shortlist)
