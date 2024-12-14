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
            # 获取历史数据，包括拆分信息
            df = ticker.history(start=start_date, end=end_date, actions=False, 
            # auto_adjust=True, back_adjust=True
            )
            if symbol == '159941.SZ' and (start_date <= datetime(2022, 7, 4).date() <= end_date):
                if any(df.index.date == datetime(2022, 7, 4).date()):
                    df.loc[df.index.date == datetime(2022, 7, 4).date(), ['Open', 'High', 'Low', 'Close']] /= 4

            if 'Dividends' in df.columns:
                dividends = df[df['Dividends'] != 0]['Dividends']
                if not dividends.empty:
                    for div_date, div_amount in dividends.items():
                        logger.warning(f"检测到股票 {symbol} 在 {div_date.date()} 分红 {div_amount:.4f}")
                        # 对分红日期之前的价格进行调整
                        for col in ['Open', 'High', 'Low', 'Close']:
                            df.loc[:div_date, col] = df.loc[:div_date, col] - div_amount

            # 检查是否有拆分事件
            if 'Stock Splits' in df.columns:
                splits = df[df['Stock Splits'] != 0]['Stock Splits']
                if not splits.empty:
                    for split_date, split_ratio in splits.items():
                        logger.warning(f"检测到股票 {symbol} 在 {split_date.date()} 发生 1:{split_ratio} 拆分")
                        # 对拆分日期之前的价格进行调整
                        for col in ['Open', 'High', 'Low', 'Close']:
                            df.loc[:split_date, col] = df.loc[:split_date, col] / split_ratio
                        # 对拆分日期之前的成交量进行调整
                        df.loc[:split_date, 'Volume'] = df.loc[:split_date, 'Volume'] * split_ratio

            if not df.empty:
                logger.info(f"{symbol} 数据统计:\n价格范围: {df['Low'].min():.2f}-{df['High'].max():.2f}\n"
                          f"成交量范围: {df['Volume'].min()}-{df['Volume'].max()}")

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
        """获取单个股票的数据，带重试机制"""
        logger.info(f"正在获取 {symbol} 的数据 ({start_date} 至 {end_date})")
        data = self.yfinance_fetcher.fetch_data(symbol, start_date, end_date)
        
        if data:
            # 获取实际数据的时间范围
            actual_dates = [record['date'] for record in data]
            actual_start = min(actual_dates)
            actual_end = max(actual_dates)
            logger.info(
                f"成功获取 {symbol} 的数据：\n"
                f"  请求范围：{start_date} 至 {end_date}\n"
                f"  实际范围：{actual_start} 至 {actual_end}\n"
                f"  数据条数：{len(data)}"
            )
        else:
            logger.warning(f"未获取到 {symbol} 在 {start_date} 至 {end_date} 期间的数据")
        
        return data

    def fetch_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        db_client: Optional[DBClient] = None,
    ) -> bool:
        """获取股票数据"""
        if db_client is None:
            db_client = DBClient()

        try:
            # 准备任务列表
            tasks = []
            for sym in [symbol]:
                sym_start_date = start_date
                if not sym_start_date:
                    # 优先使用 listing_date
                    symbol_info = db_client.get_symbol_info(sym)
                    if symbol_info and symbol_info.get('listing_date'):
                        sym_start_date = symbol_info['listing_date']
                    else:
                        # 如果没有 listing_date，则使用最新数据日期
                        latest_date = db_client.get_latest_date(sym)
                        sym_start_date = (latest_date + timedelta(days=1)) if latest_date else (end_date - timedelta(days=30))
                        logger.warning(f"未找到 {sym} 的上市日期，使用 {sym_start_date} 作为起始日期")
                
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
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return False


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="股票数据获取工具")
    parser.add_argument("--symbol", help="股票代码，不指定则更新所有股票", default=None)
    parser.add_argument("--start-date", help="开始日期 (YYYY-MM-DD)", default=None)
    parser.add_argument("--end-date", help="结束日期 (YYYY-MM-DD)，默认为今天", default=None)
    parser.add_argument("--workers", help="并发数", type=int, default=5)
    parser.add_argument("--retries", help="重试次数", type=int, default=3)
    
    args = parser.parse_args()
    
    # 处理日期格式
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    else:
        start_date = None
        
    # 结束日期默认使用今天
    end_date = datetime.now().date()
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        if end_date > datetime.now().date():
            logger.warning("结束日期超过今天，将使用今天作为结束日期")
            end_date = datetime.now().date()
    
    # 创建数据库客户端
    db_client = DBClient()
    
    # 获取要更新的股票列表
    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = db_client.get_active_symbols()
        if not symbols:
            logger.error("数据库中没有找到活跃的股票")
            return
        logger.info(f"找到 {len(symbols)} 个活跃股票需要更新")
    

    
    manager = StockDataManager(max_workers=args.workers, max_retries=args.retries)
    
    # 记录总体成功状态
    overall_success = True
    
    # 更新每个股票的数据
    for symbol in symbols:
        logger.info(f"\n开始更新 {symbol} 的数据...")
        success = manager.fetch_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            db_client=db_client
        )
        if not success:
            overall_success = False
            logger.error(f"{symbol} 数据更新失败")
    
    # 关闭数据库连接
    db_client.close()
    
    if overall_success:
        logger.info("所有数据更新完成")
    else:
        logger.warning("部分数据更新失败")
        exit(1)


if __name__ == "__main__":
    main()
