from predictor.models import ResidualRidgeModel
from predictor.models.uncertainty import PlaceholderUncertaintyModel
from predictor.types import FeatureVector


def test_residual_ridge_model_predicts_and_persists(tmp_path) -> None:
    model = ResidualRidgeModel()
    features = [
        FeatureVector(values={"m": 512.0, "n": 512.0, "k": 1024.0, "estimated_waves": 1.0}),
        FeatureVector(values={"m": 1024.0, "n": 512.0, "k": 1024.0, "estimated_waves": 2.0}),
        FeatureVector(values={"m": 1024.0, "n": 1024.0, "k": 2048.0, "estimated_waves": 3.0}),
    ]
    targets = [0.02, 0.05, 0.11]

    fitted_model = model.fit(features, targets)
    prediction = fitted_model.predict(features[1])
    batch_predictions = fitted_model.predict_batch(features)
    model_path = tmp_path / "residual_model.joblib"

    fitted_model.save(model_path)
    restored_model = ResidualRidgeModel.load(model_path)

    assert isinstance(prediction, float)
    assert len(batch_predictions) == 3
    assert restored_model.predict(features[1]) == prediction


def test_uncertainty_model_stays_placeholder() -> None:
    uncertainty_model = PlaceholderUncertaintyModel()
    features = FeatureVector(values={"estimated_waves": 1.0})

    p90 = uncertainty_model.predict_p90(features, 1.5)

    assert p90 == 1.65
