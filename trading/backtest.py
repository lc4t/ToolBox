import json
import multiprocessing
import os
import re
import sys
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from itertools import product
from typing import Dict, List, Optional, Tuple

import backtrader as bt
import pandas as pd
from loguru import logger
from tabulate import tabulate

from analyzers import PerformanceAnalyzer
from db import DBClient, SymbolInfo
from notify import NotifyManager, create_backtest_result
from utils import (
    print_metrics,
    print_combination_result,
    print_next_signal,
    print_trades,
    print_parameter_summary,
    print_best_results,
    get_stock_name,
    format_params_string,
    format_for_json,
)

logger.remove()  # 移除默认的处理器
log_level = "INFO"
logger.add(sys.stderr, level=log_level)
logger.add(
    "logs/backtest.log",
    level="INFO",
    rotation="1000MB",
    compression="zip",
)

# 在文件开头添加常量定义
SEPARATOR_WIDTH = 60
SECTION_SEPARATOR = "=" * SEPARATOR_WIDTH
SUBSECTION_SEPARATOR = "-" * SEPARATOR_WIDTH


@dataclass
class TradeRecord:
    """详细交易记录"""

    date: datetime
    action: str  # "BUY" or "SELL"
    price: float
    size: int
    value: float
    commission: float
    pnl: float  # 每笔交易的盈亏
    total_value: float  # 交易后的总资产
    signal_reason: str  # 交易信号原因
    cash: float  # 交易后的现金


class SignalCalculator:
    """信号计算器，独立处理各种策略的信号计算"""

    @staticmethod
    def check_ma_signal(
        short_ma: float,
        long_ma: float,
        prev_short_ma: float,
        prev_long_ma: float,
    ) -> bool:
        """计算均线交叉信号"""
        # 金叉买入
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            return True
        return False

    @staticmethod
    def check_ma_exit(
        short_ma: float,
        long_ma: float,
        prev_short_ma: float,
        prev_long_ma: float,
    ) -> bool:
        """计算均线死叉信号"""
        # 死叉卖出
        if prev_short_ma >= prev_long_ma and short_ma < long_ma:
            return True
        return False

    @staticmethod
    def check_chandelier_exit(
        close: float,
        highest_high: float,
        atr: float,
        multiplier: float,
    ) -> bool:
        """计算吊灯止损信号"""
        stop_price = highest_high - (multiplier * atr)
        return close < stop_price

    @staticmethod
    def check_adr_exit(
        close: float,
        entry_price: float,
        adr: float,
        multiplier: float,
    ) -> bool:
        """计算ADR止损信号"""
        stop_price = entry_price - (multiplier * adr)
        return close < stop_price


class DualMAStrategy(bt.Strategy):
    """双均线策略"""

    params = (
        ("short_period", 5),
        ("long_period", 20),
        ("chandelier_period", 22),
        ("chandelier_multiplier", 3.0),
        ("adr_period", 20),
        ("adr_multiplier", 1.0),
        ("use_ma", True),
        ("use_chandelier", False),
        ("use_adr", False),
        ("trade_start_time", None),  # 交易开始时间
        ("trade_end_time", None),  # 交易结束时间
        ("position_size", 0.95),  # 添加资金使用比例参数
    )

    def __init__(self):
        # 双均线指标
        self.short_ma = bt.indicators.SMA(
            self.data.close, period=self.params.short_period
        )
        self.long_ma = bt.indicators.SMA(
            self.data.close, period=self.params.long_period
        )

        # 吊灯止损指标
        if self.params.use_chandelier:
            self.atr = bt.indicators.ATR(
                self.data, period=self.params.chandelier_period
            )
            self.highest = bt.indicators.Highest(
                self.data.high, period=self.params.chandelier_period
            )

        # ADR止损指标
        if self.params.use_adr:
            self.daily_range = self.data.high - self.data.low
            self.adr = bt.indicators.SMA(
                self.daily_range, period=self.params.adr_period
            )

        self.order = None
        self.entry_price = None
        self.buy_price = None
        self.buy_comm = None
        self.signal_calculator = SignalCalculator()
        self.trade_records: List[TradeRecord] = []  # 存储交易记录

        # 转换交易时间
        self.trade_start = None
        self.trade_end = None
        if self.params.trade_start_time:
            self.trade_start = pd.Timestamp(self.params.trade_start_time).time()
        if self.params.trade_end_time:
            self.trade_end = pd.Timestamp(self.params.trade_end_time).time()

    def _is_trading_allowed(self) -> bool:
        """检查当前是否允许交易"""
        if not (self.trade_start and self.trade_end):
            return True

        current_time = self.data.datetime.time()
        return self.trade_start <= current_time <= self.trade_end

    def notify_trade(self, trade):
        """交易完成时的回调"""
        pass

    def notify_order(self, order):
        """订单状态变化的回调"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                # 买入订单完成
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
                self.entry_price = order.executed.price

                # 计算持仓市值和总资产（使用abs确保数量为正数）
                size = abs(order.executed.size)
                position_value = order.executed.price * size
                cash_after_buy = self.broker.get_cash()
                total_value = cash_after_buy + position_value

                # 记录买入交易
                self.trade_records.append(
                    TradeRecord(
                        date=self.data.datetime.datetime(),
                        action="BUY",
                        price=order.executed.price,
                        size=size,  # 使用正数
                        value=position_value,
                        commission=order.executed.comm,
                        pnl=0.0,
                        total_value=total_value,
                        signal_reason=self._get_detailed_signal_reason("MA Cross Buy"),
                        cash=cash_after_buy,
                    )
                )

                # 打印详细日志
                logger.debug(
                    f"Buy Executed - Price: {order.executed.price:.3f}, "
                    f"Size: {size}, "
                    f"Value: {position_value:.2f}, "
                    f"Commission: {order.executed.comm:.2f}, "
                    f"Cash: {cash_after_buy:.2f}, "
                    f"Total Value: {total_value:.2f}"
                )

            else:
                # 卖出订单完成
                # 详细记录盈亏计算过程，确保使用正数
                sell_price = order.executed.price
                sell_size = abs(order.executed.size)  # 使用abs确保为正数
                sell_value = sell_price * sell_size
                sell_commission = order.executed.comm

                buy_price = self.buy_price
                buy_size = sell_size  # 应该相同
                buy_value = buy_price * buy_size
                buy_commission = self.buy_comm

                total_commission = buy_commission + sell_commission
                pnl = (sell_value - buy_value) - total_commission

                # 打印详细的计算过程
                logger.debug(
                    f"\n盈亏计算过程:"
                    f"\n1. 买入详情:"
                    f"\n   - 买入价格: {buy_price:.4f}"
                    f"\n   - 买入数量: {buy_size}"
                    f"\n   - 买入金额: {buy_value:.4f}"
                    f"\n   - 买入手续费: {buy_commission:.4f}"
                    f"\n2. 卖出详情:"
                    f"\n   - 卖出价格: {sell_price:.4f}"
                    f"\n   - 卖出数: {sell_size}"
                    f"\n   - 卖出金额: {sell_value:.4f}"
                    f"\n   - 卖出手续: {sell_commission:.4f}"
                    f"\n3. 计算:"
                    f"\n   - 交易差价: {sell_value - buy_value:.4f} (卖出金额 - 买入金额)"
                    f"\n   - 总手续费: {total_commission:.4f} (买入手续费 + 卖出手续费)"
                    f"\n   - 最终盈亏: {pnl:.4f} (交易差价 - 总手续费)"
                )

                # 卖出后的现金和总资产
                cash_after_sell = self.broker.get_cash()
                total_value = cash_after_sell  # 卖出后资产等于现金

                # 记录卖出交易
                self.trade_records.append(
                    TradeRecord(
                        date=self.data.datetime.datetime(),
                        action="SELL",
                        price=sell_price,
                        size=sell_size,
                        value=sell_value,
                        commission=sell_commission,
                        pnl=pnl,
                        total_value=total_value,
                        signal_reason=self._get_detailed_signal_reason(
                            self._last_signal_reason
                        ),
                        cash=cash_after_sell,
                    )
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"Order Failed - Status: {order.status}")

        self.order = None

    def _calculate_entry_signal(self) -> bool:
        """计算入场信号"""
        if not self.params.use_ma:
            return False

        return self.signal_calculator.check_ma_signal(
            self.short_ma[0],
            self.long_ma[0],
            self.short_ma[-1],
            self.long_ma[-1],
        )

    def _calculate_exit_signals(self) -> Tuple[bool, str]:
        """计算出场信号，返回(是否退出, 退出原因)"""
        # 检查均线死叉
        if self.params.use_ma:
            if self.signal_calculator.check_ma_exit(
                self.short_ma[0],
                self.long_ma[0],
                self.short_ma[-1],
                self.long_ma[-1],
            ):
                return True, "MA Cross"

        # 检查吊灯止
        if self.params.use_chandelier:
            if self.signal_calculator.check_chandelier_exit(
                self.data.close[0],
                self.highest[0],
                self.atr[0],
                self.params.chandelier_multiplier,
            ):
                return True, "Chandelier Exit"

        # 检查ADR止损
        if self.params.use_adr and self.entry_price is not None:
            if self.signal_calculator.check_adr_exit(
                self.data.close[0],
                self.entry_price,
                self.adr[0],
                self.params.adr_multiplier,
            ):
                return True, "ADR Stop"

        return False, ""

    def next(self):
        # 如果有未完成的订单，等待
        if self.order:
            return

        # 检查是否在允许交易的时间段内
        if not self._is_trading_allowed():
            return

        if not self.position:
            if self._calculate_entry_signal():
                # 使用指定的资金使用比例
                cash = self.broker.get_cash() * self.params.position_size
                size = int(cash / self.data.close[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    self._last_signal_reason = "MA Cross Buy"
                    logger.debug(
                        f"Buy Order Created - Price: {self.data.close[0]:.2f}, "
                        f"Size: {size}"
                    )
        else:
            should_exit, exit_reason = self._calculate_exit_signals()
            if should_exit:
                self.order = self.sell(size=self.position.size)
                self._last_signal_reason = exit_reason
                logger.debug(
                    f"Sell Order Created - Price: {self.data.close[0]:.2f}, "
                    f"Size: {self.position.size}, "
                    f"Reason: {exit_reason}"
                )

    def _get_detailed_signal_reason(self, signal_type: str) -> str:
        """获取详细的信号说明"""
        if signal_type == "MA Cross Buy":
            return (
                f"MA交叉买入 "
                f"[MA{self.params.short_period}={self.short_ma[0]:.3f} > "
                f"MA{self.params.long_period}={self.long_ma[0]:.3f}]"
            )
        elif signal_type == "MA Cross":
            return (
                f"MA交叉卖出 "
                f"[MA{self.params.short_period}={self.short_ma[0]:.3f} < "
                f"MA{self.params.long_period}={self.long_ma[0]:.3f}]"
            )
        elif signal_type == "Chandelier Exit":
            stop_price = self.highest[0] - (
                self.params.chandelier_multiplier * self.atr[0]
            )
            return (
                f"吊灯止损 "
                f"[ATR={self.atr[0]:.3f}, "
                f"最高价={self.highest[0]:.3f}, "
                f"止损价={stop_price:.3f}]"
            )
        elif signal_type == "ADR Stop":
            stop_price = self.entry_price - (self.params.adr_multiplier * self.adr[0])
            return (
                f"ADR止损 "
                f"[ADR={self.adr[0]:.3f}, "
                f"入场价={self.entry_price:.3f}, "
                f"止损价={stop_price:.3f}]"
            )
        return signal_type

    def _predict_next_signal(self) -> Dict:
        """预测下一个交易信号"""
        current_position = bool(self.position)

        # 获取最新价格信息（从数据中获取最后一个数据点）
        latest_date = self.data.datetime.datetime()
        latest_prices = {
            "open": self.data.open[-1],  # 使用-1获取最后一个数据点
            "close": self.data.close[-1],
            "high": self.data.high[-1],
            "low": self.data.low[-1]
        }
        
        signal = {
            "action": "观察",  # 默认动作
            "conditions": [],
            "stop_loss": None,
            "position_info": None,
            "prices": latest_prices,
            "timestamp": latest_date.strftime("%Y-%m-%d")  # 添加时间戳
        }

        # 获取最新的技术指标数据
        short_ma = self.short_ma[0]
        long_ma = self.long_ma[0]
        prev_short_ma = self.short_ma[-1]
        prev_long_ma = self.long_ma[-1]

        if current_position:
            # 如果有持仓，获取最近的买入记录
            last_buy = next(
                (t for t in reversed(self.trade_records) if t.action == "BUY"), None
            )
            if last_buy:
                current_price = self.data.close[0]
                unrealized_pnl = (current_price - last_buy.price) * last_buy.size
                unrealized_pnl_pct = ((current_price / last_buy.price) - 1) * 100

                signal["action"] = "持有"
                signal["position_info"] = {
                    "entry_date": last_buy.date.strftime("%Y-%m-%d %H:%M:%S"),
                    "entry_price": last_buy.price,
                    "position_size": last_buy.size,
                    "position_value": last_buy.value,  # 买入金额
                    "current_price": current_price,
                    "current_value": current_price * last_buy.size,  # 当前市值
                    "cost": last_buy.value,  # 买入成本
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                }

            # 检查止损条件
            if self.params.use_chandelier:
                stop_price = self.highest[0] - (
                    self.params.chandelier_multiplier * self.atr[0]
                )
                signal["stop_loss"] = {
                    "type": "吊灯止损",
                    "price": stop_price,
                    "distance_pct": (
                        (self.data.close[0] - stop_price) / self.data.close[0]
                    )
                    * 100,
                }

            if self.params.use_adr:
                stop_price = self.entry_price - (
                    self.params.adr_multiplier * self.adr[0]
                )
                signal["stop_loss"] = {
                    "type": "ADR止损",
                    "price": stop_price,
                    "distance_pct": (
                        (self.data.close[0] - stop_price) / self.data.close[0]
                    )
                    * 100,
                }

            # 检查是否有卖出信号
            should_exit, exit_reason = self._calculate_exit_signals()
            if should_exit:
                signal["action"] = "卖出"
                signal["conditions"].append(exit_reason)

        else:
            # 检查买入条件
            if self.signal_calculator.check_ma_signal(
                short_ma, long_ma, prev_short_ma, prev_long_ma
            ):
                signal["action"] = "买入"
                signal["conditions"].append(
                    f"MA{self.params.short_period}({short_ma:.2f}) > "
                    f"MA{self.params.long_period}({long_ma:.2f})"
                )

        return signal

    def _generate_report(self, strat, initial_capital, final_value, trade_records):
        """生成回测报告"""
        last_trade_date = strat.data.datetime.datetime()

        # 收集分析器结果
        analyzers_results = {
            "trades": strat.analyzers.trades.get_analysis(),
            "sharpe": strat.analyzers.sharpe.get_analysis(),
            "drawdown": strat.analyzers.drawdown.get_analysis(),
            "vwr": strat.analyzers.vwr.get_analysis(),
            "last_date": last_trade_date,
        }

        # 使用性能分析器计算指标
        metrics = PerformanceAnalyzer.calculate_metrics(
            initial_capital, final_value, trade_records, analyzers_results
        )

        # 计算总收益率
        total_return = ((final_value / initial_capital) - 1) * 100

        # 预测下一个交易信号 - 这里修改调用方式
        next_signal = strat._predict_next_signal()  # 直接调用strat的方法

        # 转换所有交易记录
        all_trades = [
            {
                "date": t.date.strftime("%Y-%m-%d %H:%M:%S"),
                "action": t.action,
                "price": t.price,
                "size": t.size,
                "value": t.value,
                "commission": t.commission,
                "pnl": t.pnl,
                "total_value": t.total_value,
                "signal_reason": t.signal_reason,
                "cash": t.cash,
            }
            for t in trade_records
        ]

        # 只取最近20条用于邮件通知
        recent_trades = all_trades[-20:] if all_trades else []

        return {
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "metrics": metrics,
            "all_trades": all_trades,
            "trades": recent_trades,
            "next_signal": next_signal,
            "last_trade_date": last_trade_date,
        }


class BacktestRunner:
    """回测运行器"""

    def __init__(self, db_client: DBClient):
        self.db_client = db_client

    def run(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0001,  # 默认万分之一
        **strategy_params,
    ) -> Dict:
        """运行回测"""
        # 获取数据
        data = self._prepare_data(symbol, start_date, end_date)
        if data.empty:
            raise ValueError("No data available for backtesting")

        # 创建回测引擎
        cerebro = bt.Cerebro()

        # 添加数据
        data_feed = self._create_data_feed(data)
        cerebro.adddata(data_feed)

        # 设置初始资金
        cerebro.broker.setcash(initial_capital)

        # 设置手续费
        cerebro.broker.setcommission(commission=commission_rate)

        # 添加策略
        cerebro.addstrategy(DualMAStrategy, **strategy_params)

        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.VWR, _name="vwr")  # 添加 VWR 分析器

        # 运行回测
        results = cerebro.run()
        strat = results[0]

        # 成回测报告，添加易记录
        return self._generate_report(
            strat, initial_capital, cerebro.broker.getvalue(), strat.trade_records
        )

    def _prepare_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """准备回测数据"""
        data = self.db_client.query_by_symbol_and_date_range(
            symbol, start_date, end_date
        )
        df = pd.DataFrame(data)

        # 将 date 转换为 datetime
        df["date"] = pd.to_datetime(df["date"])

        # 确保有价格列都是 float 类型
        price_columns = ["open_price", "close_price", "high", "low"]
        for col in price_columns:
            df[col] = df[col].astype(float)

        df["volume"] = df["volume"].astype(int)

        return df

    def _create_data_feed(self, data: pd.DataFrame) -> bt.feeds.PandasData:
        """创建数据源"""
        # 确保索引是 datetime
        data.set_index("date", inplace=True)

        # 创建数据源
        data_feed = bt.feeds.PandasData(
            dataname=data,
            datetime=None,  # 使用索引作为日期时间
            open="open_price",
            high="high",
            low="low",
            close="close_price",
            volume="volume",
            openinterest=-1,
        )

        return data_feed

    def _generate_report(self, strat, initial_capital, final_value, trade_records):
        """生成回测报告"""
        last_trade_date = strat.data.datetime.datetime()

        # 收集分析器结果
        analyzers_results = {
            "trades": strat.analyzers.trades.get_analysis(),
            "sharpe": strat.analyzers.sharpe.get_analysis(),
            "drawdown": strat.analyzers.drawdown.get_analysis(),
            "vwr": strat.analyzers.vwr.get_analysis(),
            "last_date": last_trade_date,
        }

        # 使用性能分析器计算指标
        metrics = PerformanceAnalyzer.calculate_metrics(
            initial_capital, final_value, trade_records, analyzers_results
        )

        # 计算总收益率
        total_return = ((final_value / initial_capital) - 1) * 100

        # 预测下一个交易信号 - 这里修改调用方式
        next_signal = strat._predict_next_signal()  # 直接调用strat的方法

        # 转换所有交易记录
        all_trades = [
            {
                "date": t.date.strftime("%Y-%m-%d %H:%M:%S"),
                "action": t.action,
                "price": t.price,
                "size": t.size,
                "value": t.value,
                "commission": t.commission,
                "pnl": t.pnl,
                "total_value": t.total_value,
                "signal_reason": t.signal_reason,
                "cash": t.cash,
            }
            for t in trade_records
        ]

        # 只取最近20条用于邮件通知
        recent_trades = all_trades[-20:] if all_trades else []

        return {
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "metrics": metrics,
            "all_trades": all_trades,
            "trades": recent_trades,
            "next_signal": next_signal,
            "last_trade_date": last_trade_date,
        }

    def _predict_next_signal(self, strat) -> Dict:
        """预测下一个交易信号"""
        current_position = bool(strat.position)

        # 获取最新价格信息（从数据中获取最后一个数据点）
        latest_date = strat.data.datetime.datetime()
        latest_prices = {
            "open": strat.data.open[-1],  # 使用-1获取最后一个数据点
            "close": strat.data.close[-1],
            "high": strat.data.high[-1],
            "low": strat.data.low[-1]
        }
        
        signal = {
            "action": "观察",  # 默认动作
            "conditions": [],
            "stop_loss": None,
            "position_info": None,
            "prices": latest_prices,
            "timestamp": latest_date.strftime("%Y-%m-%d")  # 添加时间戳
        }

        # 获取最新的技术指标数据
        short_ma = strat.short_ma[0]
        long_ma = strat.long_ma[0]
        prev_short_ma = strat.short_ma[-1]
        prev_long_ma = strat.long_ma[-1]

        if current_position:
            # 如果有持仓，获取最近的买入记录
            last_buy = next(
                (t for t in reversed(strat.trade_records) if t.action == "BUY"), None
            )
            if last_buy:
                current_price = strat.data.close[0]
                unrealized_pnl = (current_price - last_buy.price) * last_buy.size
                unrealized_pnl_pct = ((current_price / last_buy.price) - 1) * 100

                signal["action"] = "持有"
                signal["position_info"] = {
                    "entry_date": last_buy.date.strftime("%Y-%m-%d %H:%M:%S"),
                    "entry_price": last_buy.price,
                    "position_size": last_buy.size,
                    "position_value": last_buy.value,  # 买入金额
                    "current_price": current_price,
                    "current_value": current_price * last_buy.size,  # 当前市值
                    "cost": last_buy.value,  # 买入成本
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                }

            # 检查止损条件
            if strat.params.use_chandelier:
                stop_price = strat.highest[0] - (
                    strat.params.chandelier_multiplier * strat.atr[0]
                )
                signal["stop_loss"] = {
                    "type": "吊灯止损",
                    "price": stop_price,
                    "distance_pct": (
                        (strat.data.close[0] - stop_price) / strat.data.close[0]
                    )
                    * 100,
                }

            if strat.params.use_adr:
                stop_price = strat.entry_price - (
                    strat.params.adr_multiplier * strat.adr[0]
                )
                signal["stop_loss"] = {
                    "type": "ADR止损",
                    "price": stop_price,
                    "distance_pct": (
                        (strat.data.close[0] - stop_price) / strat.data.close[0]
                    )
                    * 100,
                }

            # 检查是否有卖出信号
            should_exit, exit_reason = strat._calculate_exit_signals()
            if should_exit:
                signal["action"] = "卖出"
                signal["conditions"].append(exit_reason)

        else:
            # 检查买入条件
            if strat.signal_calculator.check_ma_signal(
                short_ma, long_ma, prev_short_ma, prev_long_ma
            ):
                signal["action"] = "买入"
                signal["conditions"].append(
                    f"MA{strat.params.short_period}({short_ma:.2f}) > "
                    f"MA{strat.params.long_period}({long_ma:.2f})"
                )

        return signal

    @staticmethod
    def _run_single_combination(params: Dict, base_config: Dict) -> Dict:
        """运行单个参数组合的回测"""
        try:
            # 创建回测引擎
            cerebro = bt.Cerebro()

            # 添加数据
            data_feed = bt.feeds.PandasData(
                dataname=base_config["df"],
                datetime=None,
                open="open_price",
                high="high",
                low="low",
                close="close_price",
                volume="volume",
                openinterest=-1,
            )
            cerebro.adddata(data_feed)

            # 设置初始资金和手续费
            cerebro.broker.setcash(base_config["initial_capital"])
            cerebro.broker.setcommission(commission=base_config["commission_rate"])

            # 添加策略
            cerebro.addstrategy(DualMAStrategy, **params)

            # 添加分析器
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            cerebro.addanalyzer(bt.analyzers.VWR, _name="vwr")

            # 运行回测
            results = cerebro.run()
            strat = results[0]

            # 生成回测报告
            final_value = cerebro.broker.getvalue()
            result = {
                "initial_capital": base_config["initial_capital"],
                "final_value": final_value,
                "total_return": ((final_value / base_config["initial_capital"]) - 1)
                * 100,
                "metrics": PerformanceAnalyzer.calculate_metrics(
                    base_config["initial_capital"],
                    final_value,
                    strat.trade_records,
                    {
                        "trades": strat.analyzers.trades.get_analysis(),
                        "sharpe": strat.analyzers.sharpe.get_analysis(),
                        "drawdown": strat.analyzers.drawdown.get_analysis(),
                        "vwr": strat.analyzers.vwr.get_analysis(),
                    },
                ),
                "all_trades": [
                    {
                        "date": t.date.strftime("%Y-%m-%d %H:%M:%S"),
                        "action": t.action,
                        "price": t.price,
                        "size": t.size,
                        "value": t.value,
                        "commission": t.commission,
                        "pnl": t.pnl,
                        "total_value": t.total_value,
                        "signal_reason": t.signal_reason,
                        "cash": t.cash,
                    }
                    for t in strat.trade_records
                ],
                "next_signal": strat._predict_next_signal(),
                "last_trade_date": strat.data.datetime.datetime(),
            }

            return {
                "params": params,
                "annual_return": result["metrics"]["annual_return"],
                "max_drawdown": result["metrics"]["max_drawdown"],
                "total_trades": result["metrics"]["total_trades"],
                "first_trade": (
                    result["all_trades"][0]["date"] if result["all_trades"] else None
                ),
                "last_trade": (
                    result["all_trades"][-1]["date"] if result["all_trades"] else None
                ),
                "full_result": result,
            }
        except Exception as e:
            logger.error(f"回测失败，参数: {params}, 错误: {str(e)}")
            return None

    def run_parameter_combinations(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0001,
        base_params: Dict = None,
        param_ranges: Dict = None,
        max_workers: int = None,
    ) -> Dict:
        """运行参数组合的回测"""
        # 在主进程中获取数据
        data = self.db_client.query_by_symbol_and_date_range(
            symbol, start_date, end_date
        )

        if not data:
            raise ValueError(f"No data available for {symbol}")

        # 准备DataFrame
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        # 参数名称映射
        param_name_map = {
            "ma_short": "short_period",
            "ma_long": "long_period",
            "chandelier_period": "chandelier_period",
            "chandelier_multiplier": "chandelier_multiplier",
            "adr_period": "adr_period",
            "adr_multiplier": "adr_multiplier",
        }

        # 生成参数组合
        param_lists = {}
        for param, value_range in param_ranges.items():
            if param in ["ma_short", "ma_long", "adr_period"]:
                values = parse_range(value_range, 1.0)
                param_lists[param] = [int(v) for v in values]
            elif param == "chandelier_period":
                values = parse_range(value_range, 5.0)
                param_lists[param] = [int(v) for v in values]
            elif param in ["chandelier_multiplier", "adr_multiplier"]:
                param_lists[param] = parse_range(value_range, 0.1)

        # 生成所有可能的参数组合
        param_names = list(param_lists.keys())
        param_values = [param_lists[name] for name in param_names]
        combinations = list(product(*param_values))

        # 过滤无效的参数组合
        valid_combinations = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            # 检查均线参数
            if "ma_short" in params and "ma_long" in params:
                if params["ma_short"] >= params["ma_long"]:
                    continue
            valid_combinations.append(combo)

        combinations = valid_combinations

        # 准备基础配置
        base_config = {
            "df": df,  # 直接传递DataFrame
            "initial_capital": initial_capital,
            "commission_rate": commission_rate,
        }

        # 设置并行进程数
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        # 存储所有回测结果
        all_results = []
        total_combinations = len(combinations)

        logger.info(
            f"开始并行回测，共 {total_combinations} 个参数组合，使用 {max_workers} 个进程"
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_params = {}
            for combo in combinations:
                params = base_params.copy()
                # 使用参数名称映射转换参数名
                for name, value in zip(param_names, combo):
                    strategy_param_name = param_name_map.get(name, name)
                    params[strategy_param_name] = value

                future = executor.submit(
                    self._run_single_combination, params, base_config
                )
                future_to_params[future] = params

            # 处理完成的任务
            completed = 0
            for future in as_completed(future_to_params):
                completed += 1
                params = future_to_params[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                    # 打印进度
                    logger.info(
                        f"进度: {completed}/{total_combinations} "
                        f"({(completed/total_combinations*100):.1f}%)"
                    )
                except Exception as e:
                    logger.error(f"参数组合运行失败: {params}")
                    logger.error(f"错误信息: {str(e)}")

        if not all_results:
            raise ValueError("所有参数组合都运行失败")

        return {
            "combinations": all_results,
            "best_annual_return": max(all_results, key=lambda x: x["annual_return"]),
            "min_drawdown": min(all_results, key=lambda x: x["max_drawdown"]),
            "param_ranges": param_ranges,
        }


def parse_range(value: str, step: float = 1.0) -> List[float]:
    """解范围参数

    Examples:
        "5" -> [5]
        "5-10" -> [5, 6, 7, 8, 9, 10]
    """
    if "-" not in value:
        return [float(value)]

    start, end = map(float, value.split("-"))
    return [round(x * step, 1) for x in range(int(start / step), int(end / step) + 1)]


def main():
    parser = ArgumentParser(description="股票回测工具")
    parser.add_argument("symbol", help="股票代码")
    parser.add_argument("--start-date", required=True, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument(
        "--end-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="结束日期 (YYYY-MM-DD)，默认为今天",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100000, help="初始资金"
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.0001,  # 默认万分之一
        help="手续费率，例如0.0001表示万分之一",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.95,  # 默认95%
        help="资金使用比例，例如0.95表95%",
    )

    # 策略参数
    parser.add_argument("--use-ma", action="store_true", help="使用双均线策")
    parser.add_argument(
        "--ma-short", type=str, default="5", help="短期均线周期，支持范围，如5-10"
    )
    parser.add_argument(
        "--ma-long", type=str, default="20", help="长期均线周期，支持范围，如15-30"
    )
    parser.add_argument("--use-chandelier", action="store_true", help="使用吊灯止损")
    parser.add_argument(
        "--chandelier-period", type=str, default="22", help="ATR周期，支持范围，如20-25"
    )
    parser.add_argument(
        "--chandelier-multiplier",
        type=str,
        default="3.0",
        help="ATR乘数，支持范围，如2.5-4.0",
    )
    parser.add_argument("--use-adr", action="store_true", help="使用ADR止损")
    parser.add_argument(
        "--adr-period", type=str, default="20", help="ADR周期，支持范围，如15-25"
    )
    parser.add_argument(
        "--adr-multiplier", type=str, default="1.0", help="ADR乘数，支持范围，如0.5-2.0"
    )

    # 添加交易时间控制参数
    parser.add_argument(
        "--trade-start-time",
        help="每日交易开始时间 (HH:MM:SS)，例如 09:30:00",
    )
    parser.add_argument(
        "--trade-end-time",
        help="每日交易结束时间 (HH:MM:SS)，例如 15:00:00",
    )

    # 添加通知相关参数
    parser.add_argument(
        "--notify",
        choices=["email", "wecom", "all"],
        help="通知方式：email=邮件, wecom=企业微信, all=所有方式",
    )
    parser.add_argument(
        "--email-to",
        help="邮件接收者，多个接收者用逗号分隔，例如：user1@example.com,user2@example.com",
    )

    # 添加 --yes 参数
    parser.add_argument("--yes", "-y", action="store_true", help="跳过所有确认")

    # 添加并行进程数参数
    parser.add_argument(
        "--workers",
        type=int,
        help="并行进程数，默认使用CPU核心数",
    )

    # 添加调试开关
    # parser.add_argument("--debug", action="store_true", help="启用调试日志")

    # 添加输出JSON参数
    parser.add_argument("--output-json", type=str, help="输出结果到指定的JSON文件")

    args = parser.parse_args()

    # 如果没有指定结束日期，使用当前日期
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")

    # 从数据库获取最新的交易日期
    db_client = DBClient()
    latest_data = db_client.query_latest_by_symbol(args.symbol)  # 使用现有的方法获取最新数据
    
    if latest_data:
        args.end_date = latest_data["date"].strftime("%Y-%m-%d")
        logger.info(f"使用数据库中的最新日期: {args.end_date}")


    # 检查是否有范围参数
    has_ranges = any(
        "-" in str(getattr(args, param))
        for param in [
            "ma_short",
            "ma_long",
            "chandelier_period",
            "chandelier_multiplier",
            "adr_period",
            "adr_multiplier",
        ]
    )

    runner = BacktestRunner(DBClient())

    if has_ranges:
        # 准备基础参数
        base_params = {
            "use_ma": args.use_ma,
            "use_chandelier": args.use_chandelier,
            "use_adr": args.use_adr,
            "trade_start_time": args.trade_start_time,
            "trade_end_time": args.trade_end_time,
        }

        # 准备范围参数
        param_ranges = {}
        if args.use_ma:
            param_ranges.update(
                {
                    "ma_short": args.ma_short,
                    "ma_long": args.ma_long,
                }
            )
        if args.use_chandelier:
            param_ranges.update(
                {
                    "chandelier_period": args.chandelier_period,
                    "chandelier_multiplier": args.chandelier_multiplier,
                }
            )
        if args.use_adr:
            param_ranges.update(
                {
                    "adr_period": args.adr_period,
                    "adr_multiplier": args.adr_multiplier,
                }
            )

        # 运行参数组合回测
        results = runner.run_parameter_combinations(
            symbol=args.symbol,
            start_date=datetime.strptime(args.start_date, "%Y-%m-%d"),
            end_date=datetime.strptime(args.end_date, "%Y-%m-%d"),
            initial_capital=args.initial_capital,
            commission_rate=args.commission_rate,
            base_params=base_params,
            param_ranges=param_ranges,
            max_workers=args.workers,  # 添加这个参数
        )

        # 为每个参数组合打印详细结果
        for idx, result in enumerate(results["combinations"], 1):
            logger.info(f"\n\n{'='*20} 参数组合 {idx} {'='*20}")
            logger.info(f"参数: {format_params_string(result['params'])}")

            full_result = result["full_result"]
            metrics = full_result["metrics"]

            print_metrics(metrics)
            print_combination_result(idx, result)
            print_next_signal(full_result["next_signal"])
            print_trades(full_result["all_trades"])

            logger.info("\n" + "=" * 50)  # 分隔线

        # 打印汇总信息
        print_parameter_summary(results["combinations"])
        print_best_results(results)

        # 如果需要通知，使用最佳年化收益率的结果
        if args.notify:
            results = results["best_annual_return"]["full_result"]
    else:
        # 原有的单次回测逻辑
        strategy_params = {
            "use_ma": args.use_ma,
            "short_period": int(float(args.ma_short)),
            "long_period": int(float(args.ma_long)),
            "use_chandelier": args.use_chandelier,
            "chandelier_period": int(float(args.chandelier_period)),
            "chandelier_multiplier": float(args.chandelier_multiplier),
            "use_adr": args.use_adr,
            "adr_period": int(float(args.adr_period)),
            "adr_multiplier": float(args.adr_multiplier),
            "trade_start_time": args.trade_start_time,
            "trade_end_time": args.trade_end_time,
            "position_size": args.position_size,
        }

        results = runner.run(
            symbol=args.symbol,
            start_date=datetime.strptime(args.start_date, "%Y-%m-%d"),
            end_date=datetime.strptime(args.end_date, "%Y-%m-%d"),
            initial_capital=args.initial_capital,
            commission_rate=args.commission_rate,
            **strategy_params,
        )

        # 将策略参数添加到结果中
        results.update(strategy_params)

        # 打印结果
        logger.info("\n=== 回测结果 ===")
        logger.info(f"初始资金: {results['initial_capital']:.2f}")
        logger.info(f"最终权益: {results['final_value']:.2f}")
        logger.info(f"总收益率: {results['total_return']:.2f}%")

        # 打印性能指标
        metrics = results["metrics"]
        print_metrics(metrics)
        print_combination_result(1, results)
        print_next_signal(results["next_signal"])
        print_trades(results["all_trades"])

        # 如果需要发送通知
        if args.notify:
            notify_methods = []
            if args.notify in ["email", "all"] and args.email_to:
                os.environ["EMAIL_RECIPIENTS"] = args.email_to
                notify_methods.append("email")
            if args.notify in ["wecom", "all"]:
                notify_methods.append("wecom")

            if notify_methods:
                notify_manager = NotifyManager(notify_methods)
                backtest_result = create_backtest_result(
                    results,
                    {
                        "symbol": args.symbol,
                        "start_date": datetime.strptime(args.start_date, "%Y-%m-%d"),
                        "end_date": datetime.strptime(args.end_date, "%Y-%m-%d"),
                        "use_ma": args.use_ma,
                        "ma_short": args.ma_short,
                        "ma_long": args.ma_long,
                        "use_chandelier": args.use_chandelier,
                        "chandelier_period": args.chandelier_period,
                        "chandelier_multiplier": args.chandelier_multiplier,
                        "use_adr": args.use_adr,
                        "adr_period": args.adr_period,
                        "adr_multiplier": args.adr_multiplier,
                    },
                )

                # 打印邮件内容预览
                print("\n=== 邮件内容预览 ===")
                print(notify_manager.get_message_preview(backtest_result))

                # 如果没有使用 -y 参数，询问确认
                if not args.yes:
                    confirm = input("是否发送邮件？(y/N) ")
                    if confirm.lower() != "y":
                        logger.info("取消发送邮件")
                        return

                if notify_manager.send_report(backtest_result):
                    logger.info("Report sent successfully")
                else:
                    logger.error("Failed to send report")

    # 如果需要输出JSON
    if args.output_json:
        if has_ranges:
            best_result = results["best_annual_return"]["full_result"]
        else:
            best_result = results

        # 获取股票名称
        db_client = DBClient()
        stock_name = get_stock_name(db_client, args.symbol)

        # 构建参数字典
        params_dict = {
            "symbol": args.symbol,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "initial_capital": args.initial_capital,
            "use_ma": args.use_ma,
            "ma_short": args.ma_short,
            "ma_long": args.ma_long,
            "use_chandelier": args.use_chandelier,
            "chandelier_period": args.chandelier_period,
            "chandelier_multiplier": args.chandelier_multiplier,
            "use_adr": args.use_adr,
            "adr_period": args.adr_period,
            "adr_multiplier": args.adr_multiplier,
        }

        json_data = format_for_json(
            best_result["metrics"],
            best_result["all_trades"],
            best_result["next_signal"],
            params_dict,
            args.symbol,
            stock_name,
        )

        # 输出到JSON文件
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
            logger.info(f"结果已输出到JSON文件: {args.output_json}")


if __name__ == "__main__":
    main()
