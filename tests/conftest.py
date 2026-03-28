import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_census_data():
    ages = np.arange(0, 101)
    male = np.exp(-ages / 50) * 1000000
    female = male * 0.95
    return pd.DataFrame({'age': ages, 'male': male.astype(int), 'female': female.astype(int)})

@pytest.fixture
def sample_macro_rates():
    years = [2020, 2021, 2022, 2023, 2024]
    cbr = [8.52, 7.52, 6.77, 6.39, 6.77]
    cdr = [7.07, 7.18, 7.37, 7.87, 7.76]
    natural_growth = [cbr[i] - cdr[i] for i in range(len(cbr))]
    return pd.DataFrame({
        'cbr': [x/1000 for x in cbr],
        'cdr': [x/1000 for x in cdr],
        'natural_growth': [x/1000 for x in natural_growth]
    }, index=years)