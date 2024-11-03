import argparse
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from finance.logger import logger


def get_and_save_etf_data(symbol, start_date, end_date, max_retries=3, retry_delay=5):
    """
    获取并保存ETF数据，带有重试机制

    Args:
        symbol: ETF代码
        start_date: 开始日期
        end_date: 结束日期
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    """
    # 确保downloads目录存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    downloads_dir = os.path.join(script_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    # 为中国ETF添加市场后缀
    if not (symbol.endswith(".SS") or symbol.endswith(".SZ")):
        if symbol.startswith("159") or symbol.startswith("150"):  # 深圳ETF
            yf_symbol = f"{symbol}.SZ"
        elif (
            symbol.startswith("510")
            or symbol.startswith("511")
            or symbol.startswith("513")
        ):  # 上海ETF
            yf_symbol = f"{symbol}.SS"
        else:
            yf_symbol = symbol
    else:
        yf_symbol = symbol

    csv_file = os.path.join(downloads_dir, f"etf_data_{symbol}.csv")

    # 确保输入的日期是无时区的
    start_date = pd.to_datetime(start_date).tz_localize(None)
    today = pd.to_datetime(datetime.now().date()).tz_localize(None)

    # 如果提供的结束日期超过今天，使用今天作为结束日期
    end_date = pd.to_datetime(end_date).tz_localize(None)
    end_date = min(end_date, today)

    # 检查CSV文件是否存在
    if os.path.exists(csv_file):
        existing_data = pd.read_csv(csv_file, encoding="utf-8")
        existing_data["日期"] = pd.to_datetime(existing_data["日期"]).dt.tz_localize(
            None
        )

        # 显示现有数据的时间范围
        logger.info(
            f"现有数据范围：{existing_data['日期'].min().date()} 到 {existing_data['日期'].max().date()}"
        )

        # 检查是否需要更新数据
        if not existing_data.empty:
            if (
                start_date >= existing_data["日期"].min()
                and end_date <= existing_data["日期"].max()
            ):
                logger.info(f"{symbol} 数据已是最新，无需更新。")
                return
            else:
                # 只获取缺失的数据
                start_date = max(
                    start_date, existing_data["日期"].max() + timedelta(days=1)
                )
                # 再次确保开始日期不会超过结束日期
                if start_date > end_date:
                    logger.info(f"{symbol} 数据已是最新，无需更新。")
                    return

    try:
        # 获取新数据
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"正在从Yahoo Finance获取 {yf_symbol} 的数据... (尝试 {attempt + 1}/{max_retries})"
                )
                # 添加调试日志，显示请求参数
                logger.debug(
                    f"请求参数:\n"
                    f"- 股票代码: {yf_symbol}\n"
                    f"- 开始日期: {start_date}\n"
                    f"- 结束日期: {end_date}\n"
                    f"- 超时时间: 10秒"
                )

                etf_data = yf.download(
                    yf_symbol,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    timeout=10,
                )

                if etf_data is not None and not etf_data.empty:
                    break  # 成功获取数据，跳出重试循环

                logger.warning(
                    f"获取到的数据为空，将重试... ({attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:  # 如果不是最后一次尝试，等待后重试
                    time.sleep(retry_delay)

            except Exception as e:
                logger.error(f"第 {attempt + 1} 次尝试失败: {e}")
                if attempt < max_retries - 1:  # 如果不是最后一次尝试，等待后重试
                    time.sleep(retry_delay)
                else:
                    raise  # 最后一次尝试也失败，抛出异常

        if etf_data is None or etf_data.empty:
            logger.warning(f"警告：未找到 {symbol} 在指定日期范围内的数据。")
            return

        # 重置索引，将日期变为列，并只保留日期部分
        etf_data.reset_index(inplace=True)
        etf_data["Date"] = pd.to_datetime(etf_data["Date"]).dt.tz_localize(None)

        # 如果有多级列索引，将其压平
        if isinstance(etf_data.columns, pd.MultiIndex):
            etf_data.columns = [col[0] for col in etf_data.columns]

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
            existing_data = pd.read_csv(csv_file, encoding="utf-8")
            existing_data["日期"] = pd.to_datetime(
                existing_data["日期"]
            ).dt.tz_localize(None)
            etf_data = pd.concat([existing_data, etf_data]).drop_duplicates(
                subset=["日期"], keep="last"
            )

        # 按日期排序
        etf_data = etf_data.sort_values("日期")

        # 保存到CSV时指定编码
        etf_data.to_csv(csv_file, index=False, date_format="%Y-%m-%d", encoding="utf-8")
        logger.info(f"{symbol} 数据已更新并保存到 {csv_file}")

    except Exception as e:
        logger.error(f"获取 {symbol} 数据时出错: {e}")
        logger.error(
            f"请检查：\n1. ETF代码 {symbol} 是否正确\n2. 网络连接是否正常\n3. 是否需要使用代理"
        )
        return


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

    get_and_save_etf_data(
        args.symbol,
        args.start_date,
        args.end_date,
    )


if __name__ == "__main__":
    main()
