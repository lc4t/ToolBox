import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import argparse


def get_and_save_etf_data(symbol, start_date, end_date):
    csv_file = f"finance/etf_data_{symbol}.csv"
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # 检查CSV文件是否存在
    if os.path.exists(csv_file):
        existing_data = pd.read_csv(csv_file)
        existing_data["日期"] = pd.to_datetime(existing_data["日期"])

        # 检查是否需要更新数据
        if not existing_data.empty:
            if start_date >= existing_data["日期"].min() and end_date <= existing_data["日期"].max():
                print(f"{symbol} 数据已是最新，无需更新。")
                return
            else:
                # 只获取缺失的数据
                start_date = max(start_date, existing_data["日期"].max() + timedelta(days=1))

    try:
        # 获取新数据
        etf_data = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1))

        if etf_data.empty:
            print(f"警告：未找到 {symbol} 在指定日期范围内的数据。")
            return

        # 重置索引，将日期变为列
        etf_data.reset_index(inplace=True)

        # 重命名列以匹配之前的格式
        etf_data = etf_data.rename(
            columns={"Date": "日期", "Open": "开盘", "High": "最高", "Low": "最低", "Close": "收盘", "Volume": "成交量"}
        )

        # 只保留需要的列
        etf_data = etf_data[["日期", "开盘", "收盘", "最高", "最低", "成交量"]]

        # 保留价格的3位小数
        for col in ["开盘", "收盘", "最高", "最低"]:
            etf_data[col] = etf_data[col].round(3)

        # 如果文件存在，合并新旧数据
        if os.path.exists(csv_file):
            existing_data = pd.read_csv(csv_file)
            existing_data["日期"] = pd.to_datetime(existing_data["日期"])
            etf_data = pd.concat([existing_data, etf_data]).drop_duplicates(subset=["日期"], keep="last")

        # 按日期排序
        etf_data = etf_data.sort_values("日期")

        # 保存到CSV
        etf_data.to_csv(csv_file, index=False)
        print(f"{symbol} 数据已更新并保存到 {csv_file}")

    except Exception as e:
        print(f"获取 {symbol} 数据时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="获取并保存ETF数据")
    parser.add_argument("--symbol", type=str, help="ETF代码")
    parser.add_argument(
        "--start-date", type=str, help="开始日期 (YYYY-MM-DD)", default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    )
    parser.add_argument("--end-date", type=str, help="结束日期 (YYYY-MM-DD)", default=datetime.now().strftime("%Y-%m-%d"))

    args = parser.parse_args()

    get_and_save_etf_data(args.symbol, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
