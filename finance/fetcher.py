import argparse
import os
from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
import yfinance as yf

from finance.logger import logger


def get_and_save_etf_data(
    symbol: str,
    start_date: Union[str, datetime, None] = None,
    end_date: Union[str, datetime, None] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> Optional[pd.DataFrame]:
    """获取并保存ETF数据,带有重试机制"""
    # 创建下载目录
    downloads_dir = os.path.join(os.path.dirname(__file__), "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    # 为中国ETF添加市场后缀
    if not (symbol.endswith(".SS") or symbol.endswith(".SZ")):
        if symbol.startswith(("159", "150")):  # 深圳ETF
            yf_symbol = f"{symbol}.SZ"
        elif symbol.startswith(("510", "511", "513")):  # 上海ETF
            yf_symbol = f"{symbol}.SS"
        else:
            yf_symbol = symbol
    else:
        yf_symbol = symbol

    csv_file = os.path.join(downloads_dir, f"etf_data_{symbol}.csv")

    # 确保输入的日期是无时区的
    if start_date:
        start_date = pd.to_datetime(start_date)
        if start_date.tzinfo is not None:
            start_date = start_date.tz_localize(None)
    if end_date:
        # 确保end_date包含当前时分秒
        current_time = datetime.now()
        end_date = pd.to_datetime(end_date).replace(
            hour=current_time.hour,
            minute=current_time.minute,
            second=current_time.second,
        )
        if end_date.tzinfo is not None:
            end_date = end_date.tz_localize(None)

    force_update = False
    existing_data = None

    # 检查CSV文件是否存在
    if os.path.exists(csv_file):
        existing_data = pd.read_csv(csv_file, encoding="utf-8")
        # 确保日期列是无时区的
        existing_data["日期"] = pd.to_datetime(existing_data["日期"]).apply(
            lambda x: x.tz_localize(None) if x.tzinfo is not None else x
        )

        # 检查是否需要重新获取全部数据
        if not existing_data.empty and start_date:
            data_start = existing_data["日期"].min()
            # 确保比较的时间戳都是无时区的
            if data_start.tzinfo is not None:
                data_start = data_start.tz_localize(None)
            if start_date.tzinfo is not None:
                start_date = start_date.tz_localize(None)

            if data_start > start_date:
                logger.warning(
                    "CSV数据的开始日期晚于请求的开始日期,将重新获取所有历史数据"
                )
                force_update = True

    # 获取数据
    try:
        ticker = yf.Ticker(yf_symbol)
        logger.debug(f"创建Ticker对象: {yf_symbol}")

        # 总是从2010年开始获取数据，确保数据足够早
        fetch_start = pd.to_datetime("2010-01-01").tz_localize(None)
        logger.debug(
            f"API调用参数:\n"
            f"- start: {fetch_start}\n"
            f"- end: {end_date}\n"
            f"- interval: 1d\n"
            f"- auto_adjust: False\n"
            f"- actions: True\n"
            f"- timeout: 60"
        )
        new_data = ticker.history(
            start=fetch_start,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            actions=True,
            timeout=60,
        )

        if new_data is not None and not new_data.empty:
            # 确保索引（日期）是无时区的
            new_data.index = new_data.index.tz_localize(None)
            earliest_date = new_data.index.min()
            latest_date = new_data.index.max()

            logger.debug(
                f"API返回数据:\n"
                f"- 数据行数: {len(new_data)}\n"
                f"- 最早日期: {earliest_date.date()}\n"
                f"- 最新日期: {latest_date.date()}\n"
                f"- 列名: {list(new_data.columns)}"
            )

            # 处理数据时确保日期是无时区的
            processed_data = process_data(new_data, ticker)
            logger.debug(
                f"处理后的数据:\n"
                f"- 数据行数: {len(processed_data)}\n"
                f"- 列名: {list(processed_data.columns)}\n"
                f"- 前5行:\n{processed_data.head()}"
            )

            # 直接覆盖CSV文件
            processed_data.to_csv(csv_file, index=False, encoding="utf-8")
            logger.info(f"成功保存数据到文件: {csv_file}")

            # 验证保存的数据
            saved_data = pd.read_csv(csv_file, encoding="utf-8")
            logger.debug(
                f"验证保存的数据:\n"
                f"- 数据行数: {len(saved_data)}\n"
                f"- 最早日期: {pd.to_datetime(saved_data['日期']).min().date()}\n"
                f"- 最新日期: {pd.to_datetime(saved_data['日期']).max().date()}"
            )

            return processed_data
        else:
            logger.error("获取数据失败：返回的数据为空")
            return None

    except Exception as e:
        logger.error(
            f"获取数据时出错:\n"
            f"- 错误信息: {str(e)}\n"
            f"- 错误类型: {type(e).__name__}\n"
            f"- 符号: {yf_symbol}\n"
            f"- 开始日期: {fetch_start}\n"
            f"- 结束日期: {end_date}"
        )
        return None


def process_data(data: pd.DataFrame, ticker: yf.Ticker) -> pd.DataFrame:
    """处理从API获取的数据"""
    # 重置索引,将日期变为列
    data.reset_index(inplace=True)

    # 获取ETF的基本信息
    info = ticker.info
    shares_outstanding = info.get("sharesOutstanding", 0)

    # 重命名列
    data = data.rename(
        columns={
            "Date": "日期",
            "Open": "开盘",
            "High": "最高",
            "Low": "最低",
            "Close": "收盘",
            "Volume": "成交量",
            "Dividends": "分红",
            "Stock Splits": "拆分比例",
            "Adj Close": "复权收盘",
        }
    )

    # 添加流通股本和换手率
    data["流通股本"] = shares_outstanding
    data["换手率"] = (
        data["成交量"] / shares_outstanding * 100 if shares_outstanding > 0 else 0
    )

    # 格式化数据
    price_columns = ["开盘", "最高", "最低", "收盘", "复权收盘"]
    for col in price_columns:
        data[col] = data[col].round(4)
    data["换手率"] = data["换手率"].round(2)

    return data[
        [
            "日期",
            "开盘",
            "最高",
            "最低",
            "收盘",
            "成交量",
            "分红",
            "拆分比例",
            "复权收盘",
            "流通股本",
            "换手率",
        ]
    ]


def main():
    parser = argparse.ArgumentParser(description="获取并保存ETF数据")
    parser.add_argument("symbol", type=str, help="ETF代码,需要添加市场后缀.SZ或.SS")
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
