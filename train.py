from __future__ import annotations

import json
from pathlib import Path

import joblib

from biomedical_ai import candidate_models, compare_models, evaluate, feature_ranges, split_data


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def main() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    split = split_data()
    comparison = compare_models(split.X_train, split.y_train)

    # Keep an interpretable model for the demo. The comparison table remains visible
    # so the choice is transparent rather than silently optimizing one score.
    model = candidate_models()["Logistic Regression"]
    model.fit(split.X_train, split.y_train)
    metrics = evaluate(model, split.X_test, split.y_test)

    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    joblib.dump(split, ARTIFACT_DIR / "split.joblib")
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    feature_ranges(split.X_train).to_csv(ARTIFACT_DIR / "feature_ranges.csv")
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(comparison.to_string(index=False))
    print("\nHeld-out test metrics (Logistic Regression):")
    print(json.dumps(metrics, indent=2))
    print(f"\nArtifacts written to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()

