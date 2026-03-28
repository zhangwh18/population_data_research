import numpy as np
from scipy.stats import gamma, linregress
import config

def generate_std_asfr():
    """生成标准生育模式（归一化）"""
    ages = np.arange(15, 50)
    shape, loc, scale = 3.5, 15, 2.5
    pdf = gamma.pdf(ages, shape, loc, scale)
    return pdf / pdf.sum()

def estimate_tfr_from_cbr(cbr, total_pop, fertile_women, std_asfr):
    """根据粗出生率、总人口、育龄妇女数、标准模式估算TFR"""
    births = cbr * total_pop
    births_per_tfr1 = np.sum(std_asfr * fertile_women)
    return births / births_per_tfr1

def compute_historical_tfr(census_df, macro_df, std_asfr):
    """
    计算历史年份（2020-2024）的 TFR
    返回字典 {year: tfr}
    """
    base_pop = census_df.set_index('age')[['male', 'female']].sort_index()
    total_pop_2020 = base_pop['male'].sum() + base_pop['female'].sum()
    fertile_women_2020 = base_pop.loc[config.FERTILE_AGE_START:config.FERTILE_AGE_END, 'female'].values

    hist_years = [2020, 2021, 2022, 2023, 2024]
    hist_tfr = {}
    for year in hist_years:
        if year in macro_df.index:
            cbr = macro_df.loc[year, 'cbr']
            births = cbr * total_pop_2020
            births_per_tfr1 = np.sum(std_asfr * fertile_women_2020)
            tfr = births / births_per_tfr1
            hist_tfr[year] = tfr
    return hist_tfr

def get_tfr_forecaster(hist_tfr, method='linear'):
    """
    返回一个可调用对象，输入年份返回该年的预测 TFR
    method: 'linear' 或 'exp'（指数衰减）
    """
    years = np.array(list(hist_tfr.keys()))
    values = np.array(list(hist_tfr.values()))
    if method == 'linear':
        coeffs = np.polyfit(years, values, 1)
        slope, intercept = coeffs[0], coeffs[1]
        def predict(year):
            tfr = intercept + slope * year
            return max(tfr, 0.5)   # 限制最低0.5
        return predict
    elif method == 'exp':
        from scipy.optimize import curve_fit
        def exp_func(x, a, b, c):
            return a * np.exp(b * x) + c
        popt, _ = curve_fit(exp_func, years, values, p0=[1, -0.05, 0.5])
        def predict(year):
            return exp_func(year, *popt)
        return predict
    else:
        raise ValueError(f"未知外推方法: {method}")

def approx_survival_from_e0(e0, max_age=100):
    """根据预期寿命生成年龄别存活率（Weibull模型）"""
    from scipy.stats import weibull_min
    from math import gamma
    shape = 2.0
    scale = e0 / gamma(1 + 1/shape)
    ages = np.arange(max_age + 1)
    lx = np.exp(-(ages / scale) ** shape)
    lx = lx / lx[0]
    px = np.ones(max_age + 1)
    px[:-1] = lx[1:] / lx[:-1]
    px[-1] = 0.0
    return px

def compute_dynamic_historical_tfr(census_df, macro_df, std_asfr):
    """
    使用逐年人口结构计算历史 TFR（2020-2024年）
    返回字典 {year: tfr}
    """
    base_pop = census_df.set_index('age')[['male', 'female']].sort_index()
    pop_male = base_pop['male'].values.copy()
    pop_female = base_pop['female'].values.copy()
    hist_tfr = {}

    # 基年（2020）数据
    total_pop_2020 = pop_male.sum() + pop_female.sum()
    fertile_women_2020 = pop_female[config.FERTILE_AGE_START:config.FERTILE_AGE_END+1]
    cbr_2020 = macro_df.loc[2020, 'cbr']
    tfr_2020 = estimate_tfr_from_cbr(cbr_2020, total_pop_2020, fertile_women_2020, std_asfr)
    hist_tfr[2020] = tfr_2020

    # 临时模型
    from src.model import CohortComponentModel
    temp_model = CohortComponentModel(base_pop)
    e0_male_base = 75.37
    e0_female_base = 80.88
    delta_e0 = getattr(config, 'LIFE_EXPECTANCY_INCREASE', 0.1)

    for year in range(2021, 2025):   # 2021-2024
        total_pop = pop_male.sum() + pop_female.sum()
        fertile_women = pop_female[config.FERTILE_AGE_START:config.FERTILE_AGE_END+1]
        cbr = macro_df.loc[year, 'cbr']
        tfr = estimate_tfr_from_cbr(cbr, total_pop, fertile_women, std_asfr)
        hist_tfr[year] = tfr

        # 用该 TFR 预测下一年人口（为下一年的结构做准备）
        e0_male = e0_male_base + delta_e0 * (year - 2020)
        e0_female = e0_female_base + delta_e0 * (year - 2020)
        surv_male = approx_survival_from_e0(e0_male, max_age=config.MAX_AGE)
        surv_female = approx_survival_from_e0(e0_female, max_age=config.MAX_AGE)
        asfr = std_asfr * tfr
        pop_male, pop_female = temp_model.step(pop_male, pop_female, asfr, surv_male, surv_female)

    return hist_tfr


def predict_future_cbr(years, macro_rates, method='linear'):
    """
    预测未来年份的粗出生率
    years: 需要预测的年份列表（如 [2030, 2040]）
    macro_rates: 包含历史CBR的DataFrame，索引为年份
    method: 外推方法
        'linear': 使用线性回归
        'last': 使用最近一年值
        'mean': 使用最近5年均值
    """
    hist = macro_rates['cbr'].dropna().sort_index()
    if method == 'last':
        last_cbr = hist.iloc[-1]
        return {year: last_cbr for year in years}
    elif method == 'mean':
        recent = hist.iloc[-5:]
        mean_cbr = recent.mean()
        return {year: mean_cbr for year in years}
    elif method == 'linear':
        from scipy.stats import linregress
        x = hist.index.values
        y = hist.values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        predictions = {}
        for year in years:
            pred = intercept + slope * year
            predictions[year] = max(pred, 0.0)
        return predictions
    else:
        raise ValueError(f"未知的外推方法: {method}")