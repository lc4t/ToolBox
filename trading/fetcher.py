from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Dict, List, Type, Union

import yfinance as yf
from db import DBClient
from loguru import logger


class DataFetcher(ABC):
    """数据获取基类"""

    @abstractmethod
    def fetch_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> List[Dict[str, Any]]:
        """获取股票数据的抽象方法"""
        pass


class YFinanceFetcher(DataFetcher):
    """YFinance数据获取实现"""

    def fetch_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> List[Dict[str, Any]]:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            return [
                {
                    "symbol": symbol,
                    "date": index.date(),
                    "open_price": float(row["Open"]),
                    "close_price": float(row["Close"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "volume": int(row["Volume"]),
                }
                for index, row in df.iterrows()
            ]
        except Exception as e:
            logger.error(f"Error fetching data from YFinance: {e}")
            return []


class DataFetcherFactory:
    """数据获取工厂类"""

    _fetchers: Dict[str, Type[DataFetcher]] = {
        "yfinance": YFinanceFetcher,
    }

    @classmethod
    def get_fetcher(cls, source: str = "yfinance") -> DataFetcher:
        fetcher_class = cls._fetchers.get(source)
        if not fetcher_class:
            raise ValueError(f"Unsupported data source: {source}")
        return fetcher_class()

    @classmethod
    def register_fetcher(cls, name: str, fetcher_class: Type[DataFetcher]) -> None:
        """注册新的数据源实现"""
        cls._fetchers[name] = fetcher_class


class StockDataManager:
    """股票数据管理类"""

    def __init__(
        self,
        db_client: DBClient,
        fetcher_source: str = "yfinance",
    ):
        self.db_client = db_client
        self.fetcher = DataFetcherFactory.get_fetcher(fetcher_source)

    def _get_valid_start_date(
        self, symbol: str, requested_start: Union[str, datetime]
    ) -> datetime:
        """获取有效的开始日期（不早于上市日期）"""
        symbol_info = self.db_client.get_symbol_info(symbol)
        if not symbol_info:
            logger.warning(
                f"No symbol info found for {symbol}, using requested start date"
            )
            return (
                requested_start
                if isinstance(requested_start, datetime)
                else datetime.strptime(requested_start, "%Y-%m-%d")
            )

        listing_date = symbol_info["listing_date"]
        if not listing_date:
            logger.warning(
                f"No listing date found for {symbol}, using requested start date"
            )
            return (
                requested_start
                if isinstance(requested_start, datetime)
                else datetime.strptime(requested_start, "%Y-%m-%d")
            )

        requested_date = (
            requested_start
            if isinstance(requested_start, datetime)
            else datetime.strptime(requested_start, "%Y-%m-%d")
        )
        return max(listing_date, requested_date)

    def update_stock_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        upsert: bool = False,
    ) -> bool:
        """更新股票数据，已存在的数据会被跳过"""
        try:
            # 获取有效的开始日期
            valid_start_date = self._get_valid_start_date(symbol, start_date)

            # 如果结束日期早于有效开始日期，返回成功（无需获取数据）
            end_dt = (
                end_date
                if isinstance(end_date, datetime)
                else datetime.strptime(end_date, "%Y-%m-%d")
            )
            if end_dt < valid_start_date:
                logger.info(
                    f"End date {end_dt} is earlier than valid start date {valid_start_date}, skipping"
                )
                return True

            # 获取数据
            data_list = self.fetcher.fetch_data(symbol, valid_start_date, end_date)
            if not data_list:
                logger.warning(f"No data fetched for {symbol}")
                return False

            # 根据 upsert 参数选择插入方式
            if upsert:
                result = self.db_client.upsert_trading_data(data_list)
            else:
                result = self.db_client.batch_insert_trading_data(data_list)

            if result:
                logger.info(f"Successfully updated data for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Error updating stock data: {e}")
            return False


def main():
    parser = ArgumentParser(description="股票数据获取工具")
    parser.add_argument("symbol", help="股票代码，例如：AAPL, 000001.SZ")
    parser.add_argument(
        "--start-date",
        default="2000-01-01",
        help="开始日期 (YYYY-MM-DD)，默认为2000-01-01",
    )
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="结束日期 (YYYY-MM-DD)，默认为今天",
    )
    parser.add_argument(
        "--source",
        default="yfinance",
        choices=["yfinance"],
        help="数据源，默认为yfinance",
    )

    args = parser.parse_args()

    # 初始化数据库客户端
    db_client = DBClient()

    # 确保数据库表已创建
    db_client.init_db()

    # 创建股票数据管理器
    stock_manager = StockDataManager(
        db_client=db_client,
        fetcher_source=args.source,
    )

    # 更新股票数据
    success = stock_manager.update_stock_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if success:
        logger.info(f"Successfully updated data for {args.symbol}")
    else:
        logger.error(f"Failed to update data for {args.symbol}")


if __name__ == "__main__":
    main()
