import numpy as np

from biomedical_ai import candidate_models, evaluate, load_data, logistic_contributions, split_data


def test_target_encoding_and_shape():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}
    assert int(y.sum()) == 212  # UCI malignant count


def test_pipeline_trains_and_predicts_probabilities():
    split = split_data()
    model = candidate_models()["Logistic Regression"]
    model.fit(split.X_train, split.y_train)
    probabilities = model.predict_proba(split.X_test)[:, 1]
    assert probabilities.shape == (len(split.X_test),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_evaluation_and_explanation_are_consistent():
    split = split_data()
    model = candidate_models()["Logistic Regression"]
    model.fit(split.X_train, split.y_train)
    metrics = evaluate(model, split.X_test, split.y_test)
    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == len(split.X_test)
    explanation = logistic_contributions(model, split.X_test.iloc[[0]])
    assert len(explanation) == 30
    assert explanation["feature"].is_unique

