import os
import pandas as pd
import config
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 实际出生人口（万人）
actual_births = {
    2021: 1062,
    2022: 956,
    2023: 902,
    2024: 954
}

def main():
    # 预测结果目录
    yearly_dir = os.path.join(config.PREDICTIONS_DIR, 'yearly')
    
    print("年份\t预测出生(万人)\t实际出生(万人)\t绝对误差(万)\t相对误差(%)")
    print("-" * 70)
    
    for year in [2021, 2022, 2023, 2024]:
        # 读取预测文件
        file_path = os.path.join(yearly_dir, f'pop_{year}.csv')
        if not os.path.exists(file_path):
            print(f"警告：文件 {file_path} 不存在，跳过 {year} 年。")
            continue
        
        df = pd.read_csv(file_path)
        # 筛选0岁人口
        zero_age = df[df['age'] == 0]
        if zero_age.empty:
            print(f"警告：{year}年文件中无0岁数据，跳过。")
            continue
        
        pred_births = zero_age['male'].iloc[0] + zero_age['female'].iloc[0]
        pred_births_wan = pred_births / 10000  # 转换为万人
        
        actual = actual_births[year]
        abs_error = pred_births_wan - actual
        rel_error = (abs_error / actual) * 100
        
        print(f"{year}\t{pred_births_wan:.1f}\t\t{actual}\t\t{abs_error:.1f}\t\t{rel_error:.1f}%")

if __name__ == "__main__":
    main()