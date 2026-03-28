import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from src.model import CohortComponentModel

def test_model_initialization(sample_census_data):
    base_pop = sample_census_data.set_index('age')[['male', 'female']]
    model = CohortComponentModel(base_pop)
    assert model.max_age == 100

def test_model_step(sample_census_data):
    base_pop = sample_census_data.set_index('age')[['male', 'female']]
    pop_male = base_pop['male'].values
    pop_female = base_pop['female'].values
    asfr = np.ones(35) * 0.05   # 总和生育率 1.75
    surv = np.ones(101) * 0.99
    surv[-1] = 0.0
    model = CohortComponentModel(base_pop)
    new_male, new_female = model.step(pop_male, pop_female, asfr, surv, surv)
    assert len(new_male) == 101
    assert len(new_female) == 101
    # 检查人口正增长（本例中出生大于死亡）
    total_initial = pop_male.sum() + pop_female.sum()
    total_new = new_male.sum() + new_female.sum()
    assert total_new > total_initial * 0.99   # 允许轻微下降

def test_model_age_migration():
    """测试年龄组推移：10岁人口推移到11岁，且无出生死亡"""
    ages = np.arange(0, 101)
    male = np.zeros(101)
    female = np.zeros(101)
    male[10] = 1000
    female[10] = 1000
    base_pop = pd.DataFrame({'male': male, 'female': female}, index=ages)
    asfr = np.zeros(35)          # 无出生
    surv = np.ones(101) * 1.0    # 无死亡
    surv[-1] = 0.0
    model = CohortComponentModel(base_pop)
    new_male, new_female = model.step(male, female, asfr, surv, surv)
    # 10岁人口应全部推移到11岁
    assert new_male[11] == 1000
    assert new_female[11] == 1000
    # 其他年龄应为0
    for age in range(101):
        if age != 11:
            assert new_male[age] == 0
            assert new_female[age] == 0