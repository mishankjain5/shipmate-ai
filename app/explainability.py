"""
Explainable AI layer for the delay-risk model.

- Exact Shapley values: the model has only 4 features, so we enumerate all
  2^4 = 16 feature coalitions instead of approximating (no `shap` dependency).
  Absent features are marginalised over a background sample of training data.
- Counterfactuals: "what would have to change for the risk tier to change?"
- Vocabulary warnings: flags categorical values the model never saw in training
  (OneHotEncoder(handle_unknown="ignore") silently encodes them as all-zeros).
"""
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from app.ml_delay_model import ml_pipeline, generate_synthetic_training_data, risk_tier

FEATURES = ["carrier", "last_hub", "dwell_time_hours", "is_cross_border"]
FEATURE_LABELS = {
    "carrier": "Carrier",
    "last_hub": "Last hub",
    "dwell_time_hours": "Dwell time at hub",
    "is_cross_border": "Cross-border shipment",
}
BACKGROUND_SIZE = 150

_encoder = ml_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
TRAINING_VOCAB = {
    "carrier": set(_encoder.categories_[0]),
    "last_hub": set(_encoder.categories_[1]),
}

_background = generate_synthetic_training_data(n_samples=BACKGROUND_SIZE)[FEATURES]


def _predict(df: pd.DataFrame) -> np.ndarray:
    return ml_pipeline.predict_proba(df[FEATURES])[:, 1]


def _instance(carrier: str, last_hub: str, dwell_time_hours: float, is_cross_border: int) -> dict:
    return {
        "carrier": carrier,
        "last_hub": last_hub,
        "dwell_time_hours": float(dwell_time_hours),
        "is_cross_border": int(is_cross_border),
    }


def shapley_values(instance: dict) -> tuple[float, dict]:
    """
    Returns (base_value, {feature: contribution}) in probability units (0-1).
    Efficiency property holds exactly: base_value + sum(contributions) == prediction.
    """
    n = len(FEATURES)
    coalitions = [frozenset(c) for size in range(n + 1) for c in combinations(FEATURES, size)]

    # One batched predict call for all 16 coalitions x background rows
    frames = []
    for coalition in coalitions:
        X = _background.copy()
        for feature in coalition:
            X[feature] = instance[feature]
        frames.append(X)
    preds = _predict(pd.concat(frames, ignore_index=True)).reshape(len(coalitions), len(_background))
    value = {c: float(p.mean()) for c, p in zip(coalitions, preds)}

    contributions = {}
    for feature in FEATURES:
        phi = 0.0
        others = [f for f in FEATURES if f != feature]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in combinations(others, size):
                s = frozenset(subset)
                phi += weight * (value[s | {feature}] - value[s])
        contributions[feature] = phi

    return value[frozenset()], contributions


def _counterfactuals(instance: dict, prob: float) -> list[str]:
    current_tier = risk_tier(prob)
    statements = []

    # 1. Dwell time sweep: where does the tier flip?
    dwell_grid = np.arange(0, 97, 2, dtype=float)
    sweep = pd.DataFrame([{**instance, "dwell_time_hours": d} for d in dwell_grid])
    sweep_probs = _predict(sweep)
    dwell = instance["dwell_time_hours"]
    if current_tier != "LOW_RISK":
        lower = [(d, p) for d, p in zip(dwell_grid, sweep_probs) if d < dwell and risk_tier(p) == "LOW_RISK"]
        if lower:
            d, p = lower[-1]
            statements.append(
                f"If dwell time were {d:.0f}h instead of {dwell:.0f}h, risk would drop to {p * 100:.1f}% (LOW_RISK)."
            )
        else:
            statements.append("Reducing dwell time alone would not bring this shipment to LOW_RISK.")
    else:
        higher = [(d, p) for d, p in zip(dwell_grid, sweep_probs) if d > dwell and risk_tier(p) == "CRITICAL_RISK"]
        if higher:
            d, p = higher[0]
            statements.append(
                f"Risk would become CRITICAL ({p * 100:.1f}%) if the parcel waits {d:.0f}h or more at its hub - worth monitoring."
            )

    # 2. Cross-border flip
    flipped = {**instance, "is_cross_border": 1 - instance["is_cross_border"]}
    p_flip = float(_predict(pd.DataFrame([flipped]))[0])
    label = "a domestic" if instance["is_cross_border"] else "a cross-border"
    statements.append(f"As {label} shipment, risk would be {p_flip * 100:.1f}% ({risk_tier(p_flip)}).")

    # 3. Hub effect: average risk at the other known hubs
    other_hubs = sorted(TRAINING_VOCAB["last_hub"] - {instance["last_hub"]})
    hub_df = pd.DataFrame([{**instance, "last_hub": h} for h in other_hubs])
    p_hub = float(_predict(hub_df).mean())
    statements.append(f"At a typical other hub, risk would be about {p_hub * 100:.1f}%.")

    return statements


def _vocabulary_warnings(instance: dict) -> list[str]:
    warnings = []
    for feature in ("carrier", "last_hub"):
        if instance[feature] not in TRAINING_VOCAB[feature]:
            warnings.append(
                f"{FEATURE_LABELS[feature]} '{instance[feature]}' was never seen in training. "
                f"The one-hot encoder maps it to an all-zero vector the model never trained on, "
                f"so its contribution is an artifact, not a real signal."
            )
    return warnings


def explain_prediction(carrier: str, last_hub: str, dwell_time_hours: float, is_cross_border: int = 1) -> dict:
    """Full explanation bundle for a single delay-risk prediction."""
    instance = _instance(carrier, last_hub, dwell_time_hours, is_cross_border)
    prob = float(_predict(pd.DataFrame([instance]))[0])
    base, contributions = shapley_values(instance)

    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {
        "method": "Exact Shapley values (16 coalitions, 150-row background)",
        "prediction": round(prob * 100, 1),
        "base_value": round(base * 100, 1),
        "contributions": [
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "value": instance[feature],
                "impact_pct_points": round(phi * 100, 1),
            }
            for feature, phi in ranked
        ],
        "counterfactuals": _counterfactuals(instance, prob),
        "warnings": _vocabulary_warnings(instance),
    }
