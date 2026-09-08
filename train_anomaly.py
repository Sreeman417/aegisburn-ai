from src.preprocessing import (
    load_and_validate_data,
    preprocess_data,
)

from src.features import create_early_features

from src.anomaly_detection import DynamicAnomalyDetector


def main():
    print("Loading dataset...")

    df = load_and_validate_data(
        "data/raw/component_data.csv"
    )

    df = preprocess_data(df)

    print("Creating early-stage features...")

    df = create_early_features(df)

    print("Training anomaly detector...")

    model = DynamicAnomalyDetector(
        contamination=0.20,
        n_estimators=300,
        random_state=42,
    )

    model.fit(df)

    metrics = model.evaluate(df)

    print("\nMODULE A — DYNAMIC ANOMALY DETECTION")
    print("=" * 60)

    print(
        f"Precision:           {metrics['precision']:.4f}"
    )

    print(
        f"Recall:              {metrics['recall']:.4f}"
    )

    print(
        f"F1:                  {metrics['f1']:.4f}"
    )

    print(
        f"False Positive Rate: {metrics['false_positive_rate']:.4f}"
    )

    print(
        f"False Negative Rate: {metrics['false_negative_rate']:.4f}"
    )

    print("\nSaving model...")

    model.save(
        "models/anomaly_model.joblib"
    )

    print(
        "Saved: models/anomaly_model.joblib"
    )


if __name__ == "__main__":
    main()
