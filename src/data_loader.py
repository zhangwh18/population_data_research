import pandas as pd
import config

def load_census_data():
    """加载并清洗普查分年龄人口数据 (A0301.xls)"""
    # 跳过前4行说明，第5行作为列名
    df = pd.read_excel(config.CENSUS_RAW, header=4, dtype=str)
    # 重命名前三列
    df = df.rename(columns={df.columns[0]: 'age', df.columns[1]: 'total',
                             df.columns[2]: 'male', df.columns[3]: 'female'})
    # 只保留需要的列
    df = df[['age', 'male', 'female']].dropna(how='all')
    # 清理年龄列，去除空格
    df['age'] = df['age'].astype(str).str.strip()

    # 判断是否为数字（单岁年龄）
    def is_number(s):
        try:
            float(s)
            return True
        except:
            return False
    df['is_num'] = df['age'].apply(is_number)
    df_num = df[df['is_num']].copy()
    df_num['age'] = df_num['age'].astype(int)

    # 转换人口数为整数（去除千位分隔符）
    for col in ['male', 'female']:
        df_num[col] = df_num[col].str.replace(',', '').str.strip()
        df_num[col] = pd.to_numeric(df_num[col], errors='coerce').fillna(0).astype(int)

    # 按年龄排序
    df_num = df_num.sort_values('age').reset_index(drop=True)

    # 处理“100岁及以上”行
    hundred_plus = df[df['age'].str.contains('100岁及以上', na=False)]
    if not hundred_plus.empty:
        male_plus = int(str(hundred_plus['male'].iloc[0]).replace(',', ''))
        female_plus = int(str(hundred_plus['female'].iloc[0]).replace(',', ''))
        # 移除可能存在的100岁记录（普查中通常没有单岁100，只有100+）
        df_num = df_num[df_num['age'] != 100]
        new_row = pd.DataFrame({'age': [100], 'male': [male_plus], 'female': [female_plus]})
        df_num = pd.concat([df_num, new_row], ignore_index=True)
        df_num = df_num.sort_values('age').reset_index(drop=True)

    # 保存清洗后数据
    df_num.to_csv(config.CENSUS_CLEAN, index=False, encoding='utf-8-sig')
    print(f"普查数据已清洗并保存至 {config.CENSUS_CLEAN}")
    return df_num


def load_macro_data():
    """加载并清洗宏观人口变动率数据 (population_change_rate.csv)"""
    # 尝试多种常见编码
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'latin1']
    df = None
    for enc in encodings:
        try:
            # 跳过前2行说明，跳过最后1行注释
            df = pd.read_csv(config.MACRO_RAW, skiprows=2, skipfooter=1,
                             engine='python', encoding=enc)
            print(f"成功使用编码 {enc} 读取文件")
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError("无法读取文件，请检查文件编码。")

    # 第一列是 '指标'，设为索引
    df = df.set_index('指标').T
    # 将年份列名（如 '2024年'）转换为整数（去除“年”字）
    df.index = df.index.str.replace('年', '').astype(int)
    # 按年份升序排序
    df = df.sort_index()
    # 重命名列为英文
    df = df.rename(columns={
        '人口出生率(‰)': 'cbr',
        '人口死亡率(‰)': 'cdr',
        '人口自然增长率(‰)': 'natural_growth'
    })
    # 将千分率转换为小数（除以1000）
    for col in df.columns:
        df[col] = df[col] / 1000.0
    # 保存清洗后数据
    df.to_csv(config.MACRO_CLEAN)
    print(f"宏观数据已清洗并保存至 {config.MACRO_CLEAN}")
    return df
if __name__ == "__main__":
    # 测试加载
    pop = load_census_data()
    rates = load_macro_data()
    print(pop.head())
    print(rates.head())