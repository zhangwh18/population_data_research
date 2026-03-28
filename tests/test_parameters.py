import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from src import parameters

def test_generate_std_asfr():
    asfr = parameters.generate_std_asfr()
    assert len(asfr) == 35
    assert np.isclose(asfr.sum(), 1.0)

def test_estimate_tfr_from_cbr():
    cbr = 0.00852
    total_pop = 1.41e9
    fertile_women = np.ones(35) * 1e7
    std_asfr = np.ones(35) / 35
    tfr = parameters.estimate_tfr_from_cbr(cbr, total_pop, fertile_women, std_asfr)
    births = cbr * total_pop
    expected = births / (fertile_women.sum() * (1/35))
    assert np.isclose(tfr, expected)

def test_approx_survival_from_e0():
    surv = parameters.approx_survival_from_e0(75, max_age=100)
    assert len(surv) == 101
    assert surv[0] == pytest.approx(1.0, rel=1e-3)
    assert np.all(np.diff(surv) <= 0)  # 单调递减
    assert np.all((surv >= 0) & (surv <= 1))

def test_compute_dynamic_historical_tfr(sample_census_data, sample_macro_rates):
    std_asfr = parameters.generate_std_asfr()
    hist_tfr = parameters.compute_dynamic_historical_tfr(sample_census_data, sample_macro_rates, std_asfr)
    assert set(hist_tfr.keys()) == {2020, 2021, 2022, 2023, 2024}
    for tfr in hist_tfr.values():
        assert 0.5 < tfr < 2.5

def test_get_tfr_forecaster(sample_census_data, sample_macro_rates):
    std_asfr = parameters.generate_std_asfr()
    hist_tfr = parameters.compute_dynamic_historical_tfr(sample_census_data, sample_macro_rates, std_asfr)
    forecaster = parameters.get_tfr_forecaster(hist_tfr, method='linear')
    tfr_2025 = forecaster(2025)
    assert isinstance(tfr_2025, float)
    assert 0.5 < tfr_2025 < 2.5

    # 测试指数衰减（如果 scipy 可用）
    try:
        from scipy.optimize import curve_fit
        forecaster_exp = parameters.get_tfr_forecaster(hist_tfr, method='exp')
        tfr_2025_exp = forecaster_exp(2025)
        assert 0.5 < tfr_2025_exp < 2.5
    except ImportError:
        pass