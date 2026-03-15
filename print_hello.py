print("Hello, World!")
import pandas as pd

pd.DataFrame()

# 读取csv文件
df = pd.read_csv('data.csv')
# 显示前5行数据
print(df.head())
# 显示数据的基本信息
print(df.info())
# 显示数据的统计信息
print(df.describe())
# 显示数据的列名
print(df.columns)
# 显示数据的行数和列数
print(df.shape)
# 显示数据的缺失值情况
print(df.isnull().sum())
# 显示数据的数据类型
print(df.dtypes)