import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from src import data_loader

def test_load_census_data(monkeypatch):
    """测试加载普查数据（模拟文件读取）"""
    # 构造0-99岁的年龄和人口数据
    ages = list(range(0, 100))  # 0到99
    age_strs = [str(a) for a in ages] + ['100岁及以上']
    # 构造模拟人口数（简单线性增长，便于验证）
    male_vals = [1000000 + i*1000 for i in range(100)] + [118866]
    female_vals = [950000 + i*1000 for i in range(100)] + [83737]
    total_vals = [male+fem for male,fem in zip(male_vals[:-1], female_vals[:-1])] + [118866+83737]
    # 关键：将所有人口数字列也转换为字符串，模拟真实读取的 dtype=str
    mock_raw_data = pd.DataFrame({
        'Unnamed: 0': age_strs,
        'Unnamed: 1': [str(x) for x in total_vals],
        'Unnamed: 2': [str(x) for x in male_vals],
        'Unnamed: 3': [str(x) for x in female_vals]
    })
    def mock_read_excel(*args, **kwargs):
        return mock_raw_data
    monkeypatch.setattr(pd, 'read_excel', mock_read_excel)

    # 运行函数
    df = data_loader.load_census_data()

    # 验证结果
    assert len(df) == 101  # 0-100岁
    assert df['age'].min() == 0
    assert df['age'].max() == 100
    # 检查100岁及以上数据是否合并
    hundred_row = df[df['age'] == 100]
    assert len(hundred_row) == 1
    assert hundred_row['male'].iloc[0] == 118866
    assert hundred_row['female'].iloc[0] == 83737
    # 检查年龄0的值
    zero_row = df[df['age'] == 0]
    assert len(zero_row) == 1
    assert zero_row['male'].iloc[0] == 1000000
    assert zero_row['female'].iloc[0] == 950000

def test_load_macro_data(monkeypatch, tmp_path):
    """测试加载宏观数据"""
    # 创建临时CSV文件
    content = """数据库：年度数据
时间：最近20年
指标,2024年,2023年,2022年,2021年,2020年
人口出生率(‰),6.77,6.39,6.77,7.52,8.52
人口死亡率(‰),7.76,7.87,7.37,7.18,7.07
人口自然增长率(‰),-0.99,-1.48,-0.60,0.34,1.45
注：注释"""
    temp_file = tmp_path / "test_macro.csv"
    temp_file.write_text(content, encoding='utf-8')
    # 临时修改 config.MACRO_RAW
    original_path = config.MACRO_RAW
    config.MACRO_RAW = str(temp_file)
    try:
        df = data_loader.load_macro_data()
        # 验证
        assert df.index.tolist() == [2020, 2021, 2022, 2023, 2024]
        assert 'cbr' in df.columns
# 使用 pytest.approx 进行近似比较
        assert df.loc[2020, 'cbr'] == pytest.approx(0.00852)
        assert df.loc[2024, 'cbr'] == pytest.approx(0.00677)
        assert df.loc[2024, 'cdr'] == pytest.approx(0.00776)
        # 检查保存的文件存在
        assert os.path.exists(config.MACRO_CLEAN)
    finally:
        config.MACRO_RAW = original_path