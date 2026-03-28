import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config

def calculate_age_group_stats(pop_df):
    """计算每10岁组和三大年龄组统计（中国标准：0-15,16-59,60+）"""
    total = pop_df['male'] + pop_df['female']
    total_pop = total.sum()
    age_groups = []
    for start in range(0, 101, 10):
        if start == 100:
            label = '100+'
            mask = pop_df['age'] >= 100
        else:
            label = f'{start}-{start+9}'
            mask = (pop_df['age'] >= start) & (pop_df['age'] <= start+9)
        group_pop = total[mask].sum()
        age_groups.append({'group': label, 'population': group_pop,
                           'percentage': group_pop/total_pop*100})

    young = total[pop_df['age'] <= 15].sum()
    work = total[(pop_df['age'] >= 16) & (pop_df['age'] <= 59)].sum()
    old = total[pop_df['age'] >= 60].sum()
    three_groups = {
        '少儿 (0-15)': {'population': young, 'percentage': young/total_pop*100},
        '劳动力 (16-59)': {'population': work, 'percentage': work/total_pop*100},
        '老年 (60+)': {'population': old, 'percentage': old/total_pop*100}
    }
    return age_groups, three_groups, total_pop

def plot_population_pyramid_enhanced(pop_df, year):
    """绘制增强版人口金字塔，右侧带年龄组表格"""
    male = pop_df['male'].values
    female = -pop_df['female'].values
    ages = pop_df['age'].values
    age_groups, three_groups, total = calculate_age_group_stats(pop_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                    gridspec_kw={'width_ratios': [2, 1]})
    ax1.barh(ages, male, color='steelblue', label='Male')
    ax1.barh(ages, female, color='lightcoral', label='Female')
    ax1.set_xlabel('Population')
    ax1.set_ylabel('Age')
    ax1.set_title(f'Population Pyramid of China ({year})')
    ax1.legend()
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    from matplotlib.ticker import FuncFormatter

    def format_func(x, p):
        return f'{abs(int(x)):,}'

    ax1.xaxis.set_major_formatter(FuncFormatter(format_func))

    ax2.axis('off')
    table_data = [[g['group'], f"{g['population']:,}", f"{g['percentage']:.1f}%"] for g in age_groups]
    table_data.append(['---', '---', '---'])
    for name, stats in three_groups.items():
        table_data.append([name, f"{stats['population']:,}", f"{stats['percentage']:.1f}%"])
    table = ax2.table(cellText=table_data, colLabels=['年龄组', '人口数', '占比'],
                      cellLoc='left', loc='center', colWidths=[0.3, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig(f'{config.FIGURES_DIR}/pyramid_{year}_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_dependency_ratio_custom(pop_df, work_age_min=16,
                                      work_age_max_male=62, work_age_max_female=54):
    """自定义抚养比计算（分性别劳动力年龄）"""
    male = pop_df['male'].values
    female = pop_df['female'].values
    ages = pop_df['age'].values

    young_mask = ages <= work_age_min - 1
    young_pop = (male[young_mask] + female[young_mask]).sum()

    work_male_mask = (ages >= work_age_min) & (ages <= work_age_max_male)
    work_female_mask = (ages >= work_age_min) & (ages <= work_age_max_female)
    work_pop = male[work_male_mask].sum() + female[work_female_mask].sum()

    old_male_mask = ages >= work_age_max_male + 1
    old_female_mask = ages >= work_age_max_female + 1
    old_pop = male[old_male_mask].sum() + female[old_female_mask].sum()

    young_dep = young_pop / work_pop * 100
    old_dep = old_pop / work_pop * 100
    total_dep = (young_pop + old_pop) / work_pop * 100

    return {'young_dependency': young_dep, 'old_dependency': old_dep,
            'total_dependency': total_dep, 'work_pop': work_pop,
            'young_pop': young_pop, 'old_pop': old_pop}

def print_detailed_stats(pop_df, label):
    """打印详细年龄组统计"""
    age_groups, three_groups, total = calculate_age_group_stats(pop_df)
    print(f"\n===== {label} 人口结构（中国标准：0-15, 16-59, 60+） =====")
    print(f"总人口: {total:,.0f}")
    print("\n每10岁年龄组:")
    for g in age_groups:
        print(f"  {g['group']}: {g['population']:,.0f} ({g['percentage']:.2f}%)")
    print("\n三大年龄组:")
    for name, stats in three_groups.items():
        print(f"  {name}: {stats['population']:,.0f} ({stats['percentage']:.2f}%)")

def print_year_summary(pop_df, year, dep_custom=None):
    """打印单年摘要"""
    total = pop_df['male'].sum() + pop_df['female'].sum()
    print(f"\n===== {year}年人口概况 =====")
    print(f"总人口: {total:,.0f}")
    if dep_custom:
        print(f"总抚养比(传统): {dep_custom['total_dependency']:.2f}%")