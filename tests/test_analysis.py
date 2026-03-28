import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import analysis

def test_calculate_age_group_stats(sample_census_data):
    age_groups, three_groups, total = analysis.calculate_age_group_stats(sample_census_data)
    # 检查每10岁组数量（0-9,10-19,...,90-99,100+）共11组
    assert len(age_groups) == 11
    # 检查总人口一致
    total_calc = sum(g['population'] for g in age_groups)
    assert total_calc == total
    # 检查三大组
    assert set(three_groups.keys()) == {'少儿 (0-15)', '劳动力 (16-59)', '老年 (60+)'}
    young = three_groups['少儿 (0-15)']['population']
    work = three_groups['劳动力 (16-59)']['population']
    old = three_groups['老年 (60+)']['population']
    assert young + work + old == total

def test_calculate_dependency_ratio_custom(sample_census_data):
    # 测试默认情况
    dep = analysis.calculate_dependency_ratio_custom(sample_census_data)
    assert 'young_dependency' in dep
    assert 'old_dependency' in dep
    assert 'total_dependency' in dep
    # 测试自定义性别上限
    dep_custom = analysis.calculate_dependency_ratio_custom(
        sample_census_data,
        work_age_max_male=62, work_age_max_female=57,
    )
    assert dep_custom['total_dependency'] > 0
    # 不同定义应有不同结果
    assert dep_custom['old_dependency'] <= dep['old_dependency']

def test_print_detailed_stats(capsys, sample_census_data):
    analysis.print_detailed_stats(sample_census_data, "Test")
    captured = capsys.readouterr()
    assert "总人口" in captured.out
    assert "每10岁年龄组" in captured.out
    assert "三大年龄组" in captured.out