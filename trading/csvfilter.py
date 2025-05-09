

import pandas as pd
import os
# 这是一个csv处理工具，需要按照字段筛选，删除原文中的行

# 删除 年化收益率(%) <= 0、 总盈亏(元) <=0


def filter_csv(file_path: str):
    """删除csv文件中的某些行"""
    df = pd.read_csv(file_path)
    df = df[df['年化收益率(%)'] > 0]
    df = df[df['总盈亏(元)'] > 0]
    # df = df[df['最大亏损比例(%)'] <= 10]
    df = df[df['夏普比率'] >= 1.5]
    df = df[df['最大回撤(%)'] <= 10]
    df = df[df['VWR'] >= 5]
    df = df[df['长期均线周期'] - df['短期均线周期'] >= 2]


    
    df = df.sort_values(by='年化收益率(%)', ascending=False)
    

    output_file = f"{os.path.splitext(file_path)[0]}_filter{os.path.splitext(file_path)[1]}"
    df.to_csv(output_file, index=False)


filter_csv("backtest_results.csv")