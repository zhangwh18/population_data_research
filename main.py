import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import config
from src.data_loader import load_census_data, load_macro_data
from src.parameters import (
    generate_std_asfr,
    estimate_tfr_from_cbr,
    approx_survival_from_e0,
    get_tfr_forecaster,
    compute_dynamic_historical_tfr   # 新增
)

from src.model import CohortComponentModel
from src.analysis import (
    plot_population_pyramid_enhanced,
    print_detailed_stats,
    calculate_dependency_ratio_custom,
    print_year_summary
)

# 解决控制台乱码
sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("加载数据...")
    census_df = load_census_data()
    macro_df = load_macro_data()
    base_pop = census_df.set_index('age')[['male', 'female']].sort_index()
    std_asfr = generate_std_asfr()


    # 基年人口
    base_pop = census_df.set_index('age')[['male', 'female']].sort_index()
    pop_male = base_pop['male'].values
    pop_female = base_pop['female'].values

    # 固定生育模式
    std_asfr = generate_std_asfr()

   # 计算动态历史 TFR
    print("计算历史 TFR（动态人口结构）:")
    hist_tfr = compute_dynamic_historical_tfr(census_df, macro_df, std_asfr)
    for y, t in hist_tfr.items():
        print(f"  {y}年: {t:.3f}")

    # 获取 TFR 预测函数（线性外推）
    tfr_forecaster = get_tfr_forecaster(hist_tfr, method='linear')

    # 预期寿命参数
    e0_male_base = 75.37
    e0_female_base = 80.88
    delta_e0 = getattr(config, 'LIFE_EXPECTANCY_INCREASE', 0.1)

    # 模型实例
    model = CohortComponentModel(base_pop)

    # 逐年预测
    years = list(range(2021, config.MAX_PREDICTION_YEAR + 1))
    pop_history = {2020: base_pop.copy()}
    yearly_dir = os.path.join(config.PREDICTIONS_DIR, 'yearly')
    os.makedirs(yearly_dir, exist_ok=True)

    for year in years:
        # 1. 当年预期寿命
        e0_male = e0_male_base + delta_e0 * (year - 2020)
        e0_female = e0_female_base + delta_e0 * (year - 2020)
        surv_male = approx_survival_from_e0(e0_male, max_age=config.MAX_AGE)
        surv_female = approx_survival_from_e0(e0_female, max_age=config.MAX_AGE)

        # 2. 当年 TFR（外推）
        tfr = tfr_forecaster(year)
        asfr = std_asfr * tfr

        # 3. 单步预测
        pop_male, pop_female = model.step(pop_male, pop_female, asfr, surv_male, surv_female)
        pop_df = pd.DataFrame({'age': range(config.MAX_AGE+1), 'male': pop_male, 'female': pop_female})
        pop_history[year] = pop_df

        # 4. 保存当年详细数据
        pop_df.to_csv(os.path.join(yearly_dir, f'pop_{year}.csv'), index=False)

        # 5. 输出摘要（每5年或关键年份）
        if year % 5 == 0 or year in [2030, 2040]:
            dep_trad = calculate_dependency_ratio_custom(pop_df, work_age_max_male=59, work_age_max_female=59)
            print_year_summary(pop_df, year, dep_trad)

    # 保存关键年份数据
    pop_2030 = pop_history.get(2030, pop_history[2020])
    pop_2040 = pop_history.get(2040, pop_history[2020])
    pop_2030.to_csv(f'{config.PREDICTIONS_DIR}/pop_2030.csv', index=False)
    pop_2040.to_csv(f'{config.PREDICTIONS_DIR}/pop_2040.csv', index=False)
    print(f"预测结果已保存至 {config.PREDICTIONS_DIR} 和 {yearly_dir}")

    # 绘图与详细统计
    plot_population_pyramid_enhanced(pop_2030, 2030)
    plot_population_pyramid_enhanced(pop_2040, 2040)
    print_detailed_stats(pop_2030, "2030")
    print_detailed_stats(pop_2040, "2040")

    # 抚养比对比
    dep_trad_2030 = calculate_dependency_ratio_custom(pop_2030, work_age_max_male=59, work_age_max_female=59)
    dep_trad_2040 = calculate_dependency_ratio_custom(pop_2040, work_age_max_male=59, work_age_max_female=59)
    dep_new_2030 = calculate_dependency_ratio_custom(pop_2030, work_age_max_male=62, work_age_max_female=58)
    dep_new_2040 = calculate_dependency_ratio_custom(pop_2040, work_age_max_male=62, work_age_max_female=58)

    print("\n【传统劳动力定义（16-59岁）】")
    print(f"2030年 总抚养比: {dep_trad_2030['total_dependency']:.2f}% (少儿: {dep_trad_2030['young_dependency']:.2f}%, 老年: {dep_trad_2030['old_dependency']:.2f}%)")
    print(f"2040年 总抚养比: {dep_trad_2040['total_dependency']:.2f}% (少儿: {dep_trad_2040['young_dependency']:.2f}%, 老年: {dep_trad_2040['old_dependency']:.2f}%)")
    print("\n【延迟退休政策假设（男16-62，女16-58）】")
    print(f"2030年 总抚养比: {dep_new_2030['total_dependency']:.2f}% (少儿: {dep_new_2030['young_dependency']:.2f}%, 老年: {dep_new_2030['old_dependency']:.2f}%)")
    print(f"2040年 总抚养比: {dep_new_2040['total_dependency']:.2f}% (少儿: {dep_new_2040['young_dependency']:.2f}%, 老年: {dep_new_2040['old_dependency']:.2f}%)")

if __name__ == "__main__":
    main()