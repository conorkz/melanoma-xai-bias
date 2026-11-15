import numpy as np
import pytest

from melanoma.evaluation.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    prior_shift_correction,
)


def test_prior_shift_is_identity_when_priors_match():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(prior_shift_correction(p, 0.3, 0.3), p, atol=1e-6)


def test_prior_shift_lowers_scores_for_rarer_deployment():
    p = np.array([0.5])
    corrected = prior_shift_correction(p, 0.5, 0.1)
    assert corrected[0] == pytest.approx(0.1, abs=1e-6)


def test_prior_shift_rejects_invalid_prior():
    with pytest.raises(ValueError):
        prior_shift_correction([0.5], 0.0, 0.1)


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 1.5, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-true_logit))).astype(int)
    overconfident = 1 / (1 + np.exp(-3.0 * true_logit))
    t = fit_temperature(y, overconfident)
    assert t == pytest.approx(3.0, rel=0.1)
    fixed = apply_temperature(overconfident, t)
    assert expected_calibration_error(y, fixed) < expected_calibration_error(
        y, overconfident
    )


def test_apply_temperature_rejects_nonpositive():
    with pytest.raises(ValueError):
        apply_temperature([0.5], 0.0)
