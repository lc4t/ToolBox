from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser
from abc import ABC, abstractmethod
from tqdm import tqdm
import tenacity
from loguru import logger
import yfinance as yf

from db import DBClient
import argparse


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


class StockDataManager:
    def __init__(self, max_workers: int = 5, max_retries: int = 3):
        self.db_client = DBClient()
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.yfinance_fetcher = YFinanceFetcher()

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying {retry_state.attempt_number}/3 after error: {retry_state.outcome.exception()}"
        )
    )
    def _fetch_single_stock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """获取单个股票的据，带重试机制"""
        return self.yfinance_fetcher.fetch_data(symbol, start_date, end_date)

    def fetch_data(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        """获取股票数据，支持单个股票或所有活跃股票"""
        end_date = end_date or datetime.now()
        symbols = [symbol] if symbol else self.db_client.get_active_symbols()
        
        if not symbols:
            logger.warning("No symbols to fetch")
            return False

        # 准备任务列表
        tasks = []
        for sym in symbols:
            sym_start_date = start_date
            if not sym_start_date:
                latest_date = self.db_client.get_latest_date(sym)
                sym_start_date = (latest_date + timedelta(days=1)) if latest_date else (end_date - timedelta(days=30))
            
            tasks.append((sym, sym_start_date, end_date))

        # 使用线程池并发获取数据
        success = True
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_single_stock_data, sym, start, end): sym 
                for sym, start, end in tasks
            }
            
            with tqdm(total=len(tasks), desc="Fetching data") as pbar:
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        data = future.result()
                        if data:
                            # 输出最新一条数据的信息
                            latest_record = max(data, key=lambda x: x['date'])
                            logger.info(
                                f"Symbol: {symbol} | "
                                f"Latest Date: {latest_record['date']} | "
                                f"Open: {latest_record['open_price']:.4f} | "
                                f"Close: {latest_record['close_price']:.4f} | "
                                f"High: {latest_record['high']:.4f} | "
                                f"Low: {latest_record['low']:.4f} | "
                                f"Volume: {latest_record['volume']:,}"
                            )
                            
                            if not self.db_client.upsert_trading_data(data):
                                success = False
                                logger.error(f"Failed to save data for {symbol}")
                        else:
                            success = False
                            logger.warning(f"No data fetched for {symbol}")
                    except Exception as e:
                        success = False
                        logger.error(f"Error processing {symbol}: {e}")
                    finally:
                        pbar.update(1)

        return success


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="股票数据获取工具")
    parser.add_argument("--symbol", help="股票代码，不指定则更新所有股票", default=None)
    parser.add_argument("--start-date", help="开始日期 (YYYY-MM-DD)", default=None)
    parser.add_argument("--end-date", help="结束日期 (YYYY-MM-DD)", default=None)
    parser.add_argument("--workers", help="并发数", type=int, default=5)
    parser.add_argument("--retries", help="重试次数", type=int, default=3)
    parser.add_argument("--yes", "-y", help="跳过所有确认", action="store_true")
    
    args = parser.parse_args()
    
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None
    
    if not args.yes:
        confirm = input("是否开始获取数据？(y/N) ")
        if confirm.lower() != 'y':
            logger.info("取消操作")
            return
    
    manager = StockDataManager(max_workers=args.workers, max_retries=args.retries)
    success = manager.fetch_data(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    if success:
        logger.info("Data fetch completed successfully")
    else:
        logger.warning("Data fetch completed with some errors")
        exit(1)


if __name__ == "__main__":
    main()
