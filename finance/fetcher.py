import argparse
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def get_and_save_etf_data(symbol, start_date, end_date):
    csv_file = f"finance/etf_data_{symbol}.csv"

    # 确保输入的日期是无时区的
    start_date = pd.to_datetime(start_date).normalize()
    today = pd.to_datetime(datetime.now().date())

    # 如果提供的结束日期超过今天，使用今天作为结束日期
    end_date = pd.to_datetime(end_date).normalize()
    end_date = min(end_date, today)

    # 检查CSV文件是否存在
    if os.path.exists(csv_file):
        existing_data = pd.read_csv(csv_file)
        existing_data["日期"] = pd.to_datetime(existing_data["日期"]).dt.normalize()

        # 显示现有数据的时间范围
        print(
            f"现有数据范围：{existing_data['日期'].min().date()} 到 {existing_data['日期'].max().date()}"
        )

        # 检查是否需要更新数据
        if not existing_data.empty:
            if (
                start_date >= existing_data["日期"].min()
                and end_date <= existing_data["日期"].max()
            ):
                print(f"{symbol} 数据已是最新，无需更新。")
                return
            else:
                # 只获取缺失的数据
                start_date = max(
                    start_date, existing_data["日期"].max() + timedelta(days=1)
                )
                # 再次确保开始日期不会超过结束日期
                if start_date > end_date:
                    print(f"{symbol} 数据已是最新，无需更新。")
                    return

    try:
        # 获取新数据
        etf_data = yf.download(symbol, start=start_date, end=end_date, progress=False)

        if etf_data.empty:
            print(f"警告：未找到 {symbol} 在指定日期范围内的数据。")
            return

        # 重置索引，将日期变为列，并只保留日期部分
        etf_data.reset_index(inplace=True)
        etf_data["Date"] = pd.to_datetime(etf_data["Date"]).dt.normalize()

        # 重命名列以匹配之前的格式
        etf_data = etf_data.rename(
            columns={
                "Date": "日期",
                "Open": "开盘",
                "High": "最高",
                "Low": "最低",
                "Close": "收盘",
                "Volume": "成交量",
            }
        )

        # 只保留需要的列
        etf_data = etf_data[["日期", "开盘", "收盘", "最高", "最低", "成交量"]]

        # 保留价格的3位小数
        for col in ["开盘", "收盘", "最高", "最低"]:
            etf_data[col] = etf_data[col].round(3)

        # 如果文件存在，合并新旧数据
        if os.path.exists(csv_file):
            existing_data = pd.read_csv(csv_file)
            existing_data["日期"] = pd.to_datetime(existing_data["日期"]).dt.normalize()
            etf_data = pd.concat([existing_data, etf_data]).drop_duplicates(
                subset=["日期"], keep="last"
            )

        # 按日期排序
        etf_data = etf_data.sort_values("日期")

        # 保存到CSV时使用日期格式
        etf_data.to_csv(csv_file, index=False, date_format="%Y-%m-%d")
        print(f"{symbol} 数据已更新并保存到 {csv_file}")

    except Exception as e:
        print(f"获取 {symbol} 数据时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description="获取并保存ETF数据")
    parser.add_argument("symbol", type=str, help="ETF代码，需要添加市场后缀.SZ 或 .SS ")
    parser.add_argument(
        "--start-date",
        type=str,
        help="开始日期 (YYYY-MM-DD)",
        default=(datetime.now() - timedelta(days=3650)).strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="结束日期 (YYYY-MM-DD)",
        default=datetime.now().strftime("%Y-%m-%d"),
    )

    args = parser.parse_args()

    get_and_save_etf_data(args.symbol, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
