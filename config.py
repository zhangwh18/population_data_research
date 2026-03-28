import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data/raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data/processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'predictions')

# 创建目录
for d in [PROCESSED_DATA_DIR, FIGURES_DIR, PREDICTIONS_DIR]:
    os.makedirs(d, exist_ok=True)

# 数据文件路径
CENSUS_RAW = os.path.join(RAW_DATA_DIR, 'A0301.xls')
MACRO_RAW = os.path.join(RAW_DATA_DIR, 'population_change_rate.csv')

# 处理后文件
CENSUS_CLEAN = os.path.join(PROCESSED_DATA_DIR, 'census_2020_clean.csv')
MACRO_CLEAN = os.path.join(PROCESSED_DATA_DIR, 'macro_rates_clean.csv')

# 模型参数
MAX_AGE = 100                     # 最高年龄（0~100岁）
FERTILE_AGE_START = 15            # 育龄起始
FERTILE_AGE_END = 49              # 育龄结束
SEX_RATIO_BIRTH = 105             # 出生性别比（男/女，以女性100计）

# 模型假设
USE_CBR_TREND_FOR_TFR = True      # 是否根据CBR趋势调整TFR（若为False，则固定使用2020年TFR）
CBR_EXTRAPOLATION_METHOD = 'linear'  # CBR外推方法：'linear', 'last', 'mean'
MAX_PREDICTION_YEAR = 2040          # 预测最远年份
LIFE_EXPECTANCY_INCREASE = 0.1        # 预期寿命年增长率（岁/年）
OFFICIAL_TFR_2020 = 1.3              # 官方TFR（可手动覆盖）
USE_OFFICIAL_TFR = False             # 若True则使用官方TFR，否则估算