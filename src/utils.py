import os
import pandas as pd
import config

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def save_result(df, filename, format='csv'):
    """保存结果到 results/ 目录"""
    if format == 'csv':
        df.to_csv(os.path.join(config.RESULTS_DIR, filename), index=False)
    elif format == 'excel':
        df.to_excel(os.path.join(config.RESULTS_DIR, filename), index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")