import argparse
import itertools
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Optional

import backtrader as bt
import matplotlib
import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finance.config import get_etf_name, get_notifiers
from finance.fetcher import get_and_save_etf_data
from finance.logger import logger
from finance.message_builder import build_message
from finance.notifiers.bark import BarkNotifier
from finance.notifiers.mail import MailNotifier
from finance.notifiers.wecom import WecomBotNotifier

matplotlib.use("Agg")

# 在文件开头创建logger实例
logger = logger.bind(name="etf_backtest")


# 定义策略
class ETFStrategy(bt.Strategy):
    params = (
        ("short_period", 10),  # 默认短期均线为10日
        ("long_period", 20),  # 默认长期均线为20日
        ("stop_loss", 0.05),  # 默认止损为5%
        ("atr_period", None),  # 修改默认值为 None
        ("atr_multiplier", 2.5),  # ATR倍数
        ("use_atr_stop", True),  # 是否使用ATR止损
        ("cash_ratio", 0.95),  # 使用95%的可用资金
        ("enable_crossover_sell", True),  # 是否启用均线下穿卖出
        ("debug", False),  # debug模式开关
        ("symbol", None),  # ETF代码
        ("notifiers", {}),  # 通知器字典
        ("signal_date", None),  # 信号日期
        ("notify_date", None),  # 修改参数名
        ("adx_period", 0),  # ADX周期，0表示不使用
        ("adx_threshold", 0),  # ADX阈值，0表示不使用
        ("bbands_period", 0),  # 布林带周期，0表示不使用
        ("bbands_width_threshold", 0.0),  # 布林带宽度阈值，0表示不使用
        ("trend_filter_period", 0),  # 趋势过滤器周期，0表示不使用
        ("signal_delay", 0),  # 信号滞后天数，0表示不滞后
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.buy_date = None  # 添加买入日期变量
        self.trades = []
        self.cash = self.broker.getvalue()
        self.highest_price = 0

        # 设置 ATR 周期
        atr_period = self.params.atr_period
        if atr_period is None:
            atr_period = self.params.long_period  # 如果未指定，使用长期均线周期
            if self.params.debug:
                logger.debug(f"ATR周期未指定，使用长期均线周期: {atr_period}")

        # 添加移动平均线指标
        self.sma_short = bt.indicators.ExponentialMovingAverage(
            self.datas[0], period=self.params.short_period
        )
        self.sma_long = bt.indicators.ExponentialMovingAverage(
            self.datas[0], period=self.params.long_period
        )

        # 添加ATR指标，使用处理后的 atr_period
        self.atr = bt.indicators.ATR(self.datas[0], period=atr_period)

        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

        # 确保添加了所有必要的分析器
        self.analyzers.returns = bt.analyzers.Returns()
        self.analyzers.sharpe = bt.analyzers.SharpeRatio()
        self.analyzers.drawdown = bt.analyzers.DrawDown()

        # 添加趋势过滤器指标
        if self.params.trend_filter_period > 0:
            self.trend_sma = bt.indicators.ExponentialMovingAverage(
                self.datas[0], period=self.params.trend_filter_period
            )

        # 用于信号滞后的变量
        self.pending_buy_signal = 0  # 用于记录买入信号等待天数
        self.pending_sell_signal = 0  # 用于记录卖出信号等待天数

        # 添加ADX指标
        if self.params.adx_period > 0:
            self.adx = bt.indicators.AverageDirectionalMovementIndex(
                self.datas[0], period=self.params.adx_period
            )

        # 添加布林带指标
        if self.params.bbands_period > 0:
            self.bbands = bt.indicators.BollingerBands(
                self.datas[0], period=self.params.bbands_period
            )
            # 计算布林带宽度百分比
            self.bbands_width = (
                100 * (self.bbands.top - self.bbands.bot) / self.bbands.mid
            )

    def next(self):
        # 获取当前日期和下一个交易日
        current_date = self.data.datetime.date(0)
        try:
            next_date = self.data.datetime.date(1)
        except IndexError:
            next_date = current_date

        # 准备止损策略说明
        stop_loss_strategy = (
            f"止损策略:\n"
            f"- {'使用' if self.params.use_atr_stop else '不使用'}ATR动态止损\n"
            f"- ATR周期: {self.params.atr_period}日\n"
            f"- ATR倍数: {self.params.atr_multiplier}\n"
            f"- 固定止损比例: {self.params.stop_loss:.1%}\n"
            f"- 最终止损价格: 取ATR止损价和比例止损价的较高者\n"
            f"- 均线下穿卖出: {'启用' if self.params.enable_crossover_sell else '禁用'}\n"
        )

        # 准备基础信号数据
        signal_data = {
            "symbol": self.params.symbol,
            "name": get_etf_name(self.params.symbol),
            "date": current_date,
            "start_date": self.data.datetime.date(0),
            "price": self.dataclose[0],
            "sma_short": self.sma_short[0],
            "sma_long": self.sma_long[0],
            "atr": self.atr[0],
            "short_period": self.params.short_period,
            "long_period": self.params.long_period,
            "atr_period": self.params.atr_period,
            "atr_multiplier": self.params.atr_multiplier,
            "stop_loss": self.params.stop_loss,
            "signal_type": "持有" if self.position else "观察",
            "position_details": "当前持仓" if self.position else "当前无持仓",
            "signal_description": "策略运行中",
            "signal_details": (
                f"当前状态指标:\n"
                f"短期均线: MA{self.params.short_period} = {self.sma_short[0]:.3f}\n"
                f"长期均线: MA{self.params.long_period} = {self.sma_long[0]:.3f}\n"
                f"均线差值: {(self.sma_short[0] - self.sma_long[0]):.3f}\n"
                f"ATR指标: {self.atr[0]:.3f}\n"
                f"当前价格: {self.dataclose[0]:.3f}\n\n"
                f"{stop_loss_strategy}"
            ),
        }

        # 准备持仓数据
        holding_data = None
        if self.position:
            current_return = (self.dataclose[0] - self.buyprice) / self.buyprice
            holding_data = {
                "买入时间": self.buy_date,
                "买入价格": self.buyprice,
                "当前收益": current_return,
                "最高价格": self.highest_price,
                "当前价格": self.dataclose[0],
            }

            # 添加次日止损提示
            current_price = self.dataclose[0]
            atr_stop = current_price - (self.atr[0] * self.params.atr_multiplier)
            pct_stop = current_price * (1 - self.params.stop_loss)
            final_stop = max(atr_stop, pct_stop)

            signal_data["signal_details"] += (
                f"\n次日止损提示:\n"
                f"当前价格: {current_price:.3f}\n"
                f"ATR止损价: {atr_stop:.3f} (当前价格 - {self.params.atr_multiplier}倍ATR)\n"
                f"比例止损价: {pct_stop:.3f} (当前价格 × (1 - {self.params.stop_loss:.1%}))\n"
                f"最终止损价: {final_stop:.3f} (取两者较高者)\n\n"
                f"注意：如果次日价格在不超过当前价格的情况下回落到 {final_stop:.3f}，"
                f"将触发止损卖出。如果价格创新高，止损价格会相应提高。"
            )

        # 保存最后一天的信号数据，供回测结束后使用
        if next_date == current_date:
            self.last_signal_data = signal_data
            self.last_holding_data = holding_data

        # 执行交易逻辑
        if self.order:
            return

        # 检查趋势过滤器条件
        trend_ok = True
        trend_filter_msg = []

        # ADX过滤器
        if self.params.adx_period > 0 and self.params.adx_threshold > 0:
            adx_ok = self.adx[0] > self.params.adx_threshold
            trend_ok = trend_ok and adx_ok
            trend_filter_msg.append(
                f"ADX({self.params.adx_period}): {self.adx[0]:.2f} "
                f"{'>' if adx_ok else '<='} {self.params.adx_threshold}"
            )

        # 布林带宽度过滤器
        if self.params.bbands_period > 0 and self.params.bbands_width_threshold > 0:
            bbands_ok = self.bbands_width[0] > self.params.bbands_width_threshold
            trend_ok = trend_ok and bbands_ok
            trend_filter_msg.append(
                f"布林带宽度({self.params.bbands_period}): {self.bbands_width[0]:.2f}% "
                f"{'>' if bbands_ok else '<='} {self.params.bbands_width_threshold}%"
            )

        if not self.position:  # 没有持仓
            if self.crossover > 0:  # 买入信号
                if not trend_ok:
                    logger.debug(
                        f"趋势过滤器阻止买入 - 日期: {current_date}\n"
                        f"过滤条件: {', '.join(trend_filter_msg)}"
                    )
                    return

                # 处理信号滞后
                if self.params.signal_delay > 0:
                    self.pending_buy_signal += 1
                    if self.pending_buy_signal < self.params.signal_delay:
                        logger.debug(
                            f"买入信号等待中 - 日期: {current_date}, "
                            f"已等待: {self.pending_buy_signal}天"
                        )
                        return
                    logger.debug(f"买入信号确认 - 日期: {current_date}")

                # 先计算买入数量
                available_cash = self.broker.getcash() * self.params.cash_ratio
                size = int(available_cash / self.dataclose[0])

                # 更新信号数据
                signal_data.update(
                    {
                        "signal_type": "买入",
                        "signal_details": (
                            f"买入信号指标:\n"
                            f"短期均线: MA{self.params.short_period} = {self.sma_short[0]:.3f}\n"
                            f"长期均线: MA{self.params.long_period} = {self.sma_long[0]:.3f}\n"
                            f"均线差值: {(self.sma_short[0] - self.sma_long[0]):.3f}\n"
                            f"ATR指标: {self.atr[0]:.3f}\n"
                            f"买入价格: {self.dataclose[0]:.3f}\n"
                            f"买入数量: {size}股\n"
                            f"使用资金: {size * self.dataclose[0]:.2f}\n"
                            f"可用资金: {available_cash:.2f}"
                        ),
                        "position_details": "当前无持仓",
                        "signal_description": "短期均线上穿长期均线，产生买入信号",
                    }
                )

                if size > 0:  # 可以买入
                    # 执行买入操作
                    self.order = self.buy(size=size)
                    logger.info(
                        f"买入信号 - 日期: {current_date}, "
                        f"价格: {self.dataclose[0]:.2f}, "
                        f"数量: {size}, "
                        f"金额: {size * self.dataclose[0]:.2f}"
                    )
                    self.last_signal_date = current_date
                else:
                    logger.debug(
                        f"跳过买入通知 - 日期: {current_date}, "
                        f"notifiers: {list(self.params.notifiers.keys()) if self.params.notifiers else '无'}"
                    )

                # 重置信号等待计数
                self.pending_buy_signal = 0

        else:  # 有持仓
            # 更新最高价格
            if self.dataclose[0] > self.highest_price:
                self.highest_price = self.dataclose[0]

            # 检查是否触发止损或均线下穿卖出
            sell_signal = False
            sell_reason = ""

            # 计算止损价格
            if self.params.use_atr_stop:
                stop_price = self.highest_price - (
                    self.atr[0] * self.params.atr_multiplier
                )
                min_stop_price = self.highest_price * (1 - self.params.stop_loss)
                stop_price = max(stop_price, min_stop_price)
            else:
                stop_price = self.highest_price * (1 - self.params.stop_loss)

            # 检查是否触发止损（止损优先级最高，不受趋势过滤器影响）
            if self.dataclose[0] <= stop_price:
                sell_signal = True
                sell_reason = "触发止损"
            # 检查是否触发均线下穿卖出（受趋势过滤器影响）
            elif self.params.enable_crossover_sell and self.crossover < 0:
                if trend_ok:  # 趋势过滤器允许卖出
                    sell_signal = True
                    sell_reason = "均线下穿"
                else:
                    logger.debug(
                        f"趋势过滤器阻止均线下穿卖出 - 日期: {current_date}\n"
                        f"过滤条件: {', '.join(trend_filter_msg)}"
                    )

            # 处理卖出信号
            if sell_signal:
                # 处理信号滞后
                if self.params.signal_delay > 0:
                    self.pending_sell_signal += 1
                    if self.pending_sell_signal < self.params.signal_delay:
                        logger.debug(
                            f"卖出信号等待中 - 日期: {current_date}, "
                            f"已等待: {self.pending_sell_signal}天"
                        )
                        return
                    logger.debug(f"卖出信号确认 - 日期: {current_date}")

                # 执行卖出操作
                self.order = self.sell(size=self.position.size)
                # 设置卖出原因，供 notify_order 使用
                self.order.sell_reason = sell_reason

                logger.info(
                    f"卖出信号 - 日期: {current_date}, "
                    f"价格: {self.dataclose[0]:.2f}, "
                    f"数量: {self.position.size}, "
                    f"原因: {sell_reason}"
                )

                # 重置信号等待计数
                self.pending_sell_signal = 0

    def notify_order(self, order):
        if self.params.debug:
            if order.status in [order.Submitted, order.Accepted]:
                logger.debug(f"Order {order.ref} {order.Status[order.status]}")
                return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.buy_date = self.data.datetime.date(0)  # 记录买入日期
                self.highest_price = self.buyprice  # 初始化最高价格

                # 记录买入交易
                self.trades.append(
                    {
                        "类型": "买入",
                        "日期": self.data.datetime.date(0),
                        "价格": order.executed.price,
                        "数量": order.executed.size,
                        "消耗资金": order.executed.value,
                        "手续费": order.executed.comm,
                        "收益": 0.0,  # 买入时收益为0
                        "剩余资金": self.broker.getcash(),
                        "指标条件": f"MA{self.params.short_period}({self.sma_short[0]:.3f}) > MA{self.params.long_period}({self.sma_long[0]:.3f})",
                    }
                )

                logger.info(
                    f"Buy executed - Date: {self.data.datetime.date(0)}, "
                    f"Price: {order.executed.price:.2f}, "
                    f"Size: {order.executed.size}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Commission: {order.executed.comm:.2f}"
                )
            else:  # 卖出时
                # 计算卖出收益（需要考虑买入和卖出的总手续费）
                sell_value = order.executed.price * order.executed.size
                buy_value = self.buyprice * order.executed.size
                total_commission = (
                    order.executed.comm + self.buycomm
                )  # 买入和卖出的总手续费
                profit = round(
                    buy_value - sell_value - total_commission, 4
                )  # 收益 = 卖出价值 - 买入价值 - 总手续费

                # 获取实际的止损价格和指标条件
                if getattr(order, "sell_reason", "") == "触发止损":
                    if self.params.use_atr_stop:
                        stop_price = self.highest_price - (
                            self.atr[0] * self.params.atr_multiplier
                        )
                        min_stop_price = self.highest_price * (
                            1 - self.params.stop_loss
                        )
                        stop_price = max(stop_price, min_stop_price)
                    else:
                        stop_price = self.highest_price * (1 - self.params.stop_loss)

                    indicator_condition = (
                        f"价格({order.executed.price:.3f}) <= 止损价({stop_price:.3f})"
                    )
                elif getattr(order, "sell_reason", "") == "均线下穿":
                    indicator_condition = (
                        f"MA{self.params.short_period}({self.sma_short[0]:.3f}) < "
                        f"MA{self.params.long_period}({self.sma_long[0]:.3f})"
                    )
                else:
                    indicator_condition = "未知条件"

                # 记录卖出交易（只记录卖出时的手续费）
                self.trades.append(
                    {
                        "类型": "卖出",
                        "日期": self.data.datetime.date(0),
                        "价格": order.executed.price,
                        "数量": -order.executed.size,
                        "消耗资金": sell_value,
                        "手续费": order.executed.comm,  # 只记录卖出时的手续费
                        "收益": profit,
                        "剩余资金": self.broker.getcash(),
                        "指标条件": indicator_condition,
                    }
                )

                logger.info(
                    f"卖出执行完成 - 日期: {self.data.datetime.date(0)}, "
                    f"价格: {order.executed.price:.2f}, "
                    f"数量: {order.executed.size}, "
                    f"收益: {profit:.2f}, "  # 修改这里：取负号
                    f"原因: {getattr(order, 'sell_reason', '未知原因')}"
                )

                # 重置相关变量
                self.highest_price = 0
                self.buy_date = None
                self.buyprice = None
                self.buycomm = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(
                f"订单 {order.ref} {order.Status[order.status]} - 日期: {self.data.datetime.date(0)}, "
                f"价格: {self.dataclose[0]:.2f}, 可用资金: {self.broker.getcash():.2f}, "
                f"原因: {order.info.get('reject_reason', 'Unknown')}, "
                f"数量: {order.created.size}, 价格: {order.created.price:.2f}"
            )

        self.order = None

    def stop(self):
        """策略结束时调用"""
        # 移除所有通知发送代码，因为已经在 next 方法中发送了最后一天的完整通知
        pass

    # 添加一个新方法来生成交易记录表格
    def get_trade_table(self):
        """生成交易记录表格"""
        if not self.trades:
            return "没有发生交易"

        headers = [
            "类型",
            "日期",
            "价格",
            "数量",
            "消耗资金",
            "手续费",
            "收益",
            "剩余资金",
            "资产总值",  # 新增列
            "指标条件",
        ]
        table_data = []
        for trade in self.trades:
            if trade["类型"] == "买入":
                # 计算资产总值 = 剩余资金 + 持仓市值
                total_value = trade["剩余资金"] + trade["数量"] * trade["价格"]
                table_data.append(
                    [
                        trade["类型"],
                        str(trade["日期"]),
                        f"{trade['价格']:.3f}",
                        f"{trade['数量']:.0f}",
                        f"{trade['消耗资金']:.2f}",
                        f"{trade['手续费']:.2f}",
                        "-",
                        f"{trade['剩余资金']:.2f}",
                        f"{total_value:.2f}",  # 新增资产总值
                        trade["指标条件"],  # 买入时的指标条件
                    ]
                )
            else:  # 卖出
                # 卖出时资产总值就是剩余资金
                total_value = trade["剩余资金"]
                table_data.append(
                    [
                        trade["类型"],
                        str(trade["日期"]),
                        f"{trade['价格']:.3f}",
                        f"{trade['数量']:.0f}",
                        f"{trade['消耗资金']:.2f}",
                        f"{trade['手续费']:.2f}",
                        f"{trade['收益']:.2f}",
                        f"{trade['剩余资金']:.2f}",
                        f"{total_value:.2f}",  # 新增资产总值
                        trade["指标条件"],  # 卖出时的指标条件
                    ]
                )

        return "\n交易记录:\n" + tabulate(
            table_data,
            headers=headers,
            tablefmt="grid",
            colalign=(
                "left",  # 类型
                "left",  # 日期
                "right",  # 价格
                "right",  # 数量
                "right",  # 消耗资金
                "right",  # 手续费
                "right",  # 收益
                "right",  # 剩余资金
                "right",  # 资产总值
                "left",  # 指标条件
            ),
        )


# 获取ETF数据
def get_etf_data(symbol, start_date=None):
    """
    获取ETF数据，总是获取全部历史数据，start_date 只用回测过滤
    """
    # 添加市场后缀处理
    if not symbol.endswith((".SZ", ".SS", ".BJ")):
        if symbol.startswith(("15", "16", "18")):  # 深交所ETF
            symbol = f"{symbol}.SZ"
        elif symbol.startswith(("51", "56", "58")):  # 上交所ETF
            symbol = f"{symbol}.SS"
        else:
            logger.warning(f"无法确定 {symbol} 的市场，将尝试作为深市ETF处理")
            symbol = f"{symbol}.SZ"

    logger.debug(f"处理ETF代码: {symbol}")

    # 获取ETF名称
    etf_name = get_etf_name(symbol)
    logger.debug(f"获取到ETF名称: {etf_name}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    downloads_dir = os.path.join(script_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    # 使用完整的代码（包含后缀）作为文件名
    csv_file = os.path.join(downloads_dir, f"etf_data_{symbol}.csv")

    try:
        need_update = False
        if os.path.exists(csv_file):
            # 检查现有数据是否是最新的
            existing_data = pd.read_csv(csv_file, encoding="utf-8")
            existing_data["日期"] = pd.to_datetime(existing_data["日期"]).dt.normalize()
            latest_date = existing_data["日期"].max()

            # 如果最新数不是昨天或更新，则需更新
            yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
            if latest_date.date() < yesterday.date():
                need_update = True
                logger.info(
                    f"数据不是最新的（最新日期：{latest_date.date()}），需要更新"
                )
        else:
            need_update = True
            logger.info("数据文件不存在，需要获取数据")

        if need_update:
            # 使用带市场后缀的代码获取数据
            get_and_save_etf_data(
                symbol,  # 使用带市场后缀的代码
                "2000-01-01",
                datetime.now().strftime("%Y-%m-%d"),
            )

        # 读取数据文件
        df = pd.read_csv(csv_file, encoding="utf-8")
        if df.empty:
            logger.error(f"错误：{symbol} 的数据文件为空。")
            return None

        # 将日期列转换为datetime索引，并移除区信息
        df["日期"] = pd.to_datetime(df["日期"]).dt.normalize()
        df = df.set_index("日期")

        # 重命名列
        df = df.rename(
            columns={
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
        )
        df = df.sort_index()  # 确保数据按日期排序
        logger.info(f"数据范围：{df.index.min().date()} 到 {df.index.max().date()}")

        # 确保所有数据列都是数值类型
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 添加ETF名称
        df.name = etf_name

        # 如果指定了回测开始日期，则过滤数据
        if start_date:
            start_date = pd.to_datetime(start_date).normalize()
            df = df[df.index >= start_date]
            logger.info(
                f"回测数据范围：{df.index.min().date()} 到 {df.index.max().date()}"
            )

        return df[numeric_columns]  # 只返回数值列

    except Exception as e:
        logger.error(f"处理 {symbol} 数据出错: {e}")
        return None


# 主函数
def run_backtest(
    symbol,
    start_date,
    end_date,
    initial_cash,
    short_period,
    long_period,
    stop_loss,
    atr_period,
    atr_multiplier,
    use_atr_stop,
    cash_ratio,
    enable_crossover_sell,
    debug,
    notifiers=None,
    trend_filter_period=0,  # 添加趋势过滤器参数
    signal_delay=0,  # 添加信号滞后参数
    adx_period=0,  # 添加ADX参数
    adx_threshold=0,  # 添加ADX阈值参数
    bbands_period=0,  # 添加布林带参数
    bbands_width_threshold=0.0,  # 添加布林带宽度阈值参数
):
    cerebro = bt.Cerebro()

    # 传入开始日期和结束日期，确保数据是最新
    data = get_etf_data(symbol, start_date)
    if data is None:
        return None

    # 日期过滤，确保无时区
    start_date = pd.to_datetime(start_date).tz_localize(None)
    end_date = pd.to_datetime(end_date).tz_localize(None)
    data = data[(data.index >= start_date) & (data.index <= end_date)]

    # 创建PandasData feed
    feed = bt.feeds.PandasData(
        dataname=data,
        datetime=None,  # 使用引作日期
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,  # 不使用持仓量
    )
    cerebro.adddata(feed)

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0001)

    cerebro.addstrategy(
        ETFStrategy,
        symbol=symbol,  # 只传入代码，名称由策略内部获取
        short_period=short_period,
        long_period=long_period,
        stop_loss=stop_loss,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        use_atr_stop=use_atr_stop,
        cash_ratio=cash_ratio,
        enable_crossover_sell=enable_crossover_sell,
        debug=debug,
        notifiers=notifiers or {},
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        bbands_period=bbands_period,
        bbands_width_threshold=bbands_width_threshold,
        trend_filter_period=trend_filter_period,  # 传递趋势过滤器参数
        signal_delay=signal_delay,  # 传递信号滞后参数
    )

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    results = cerebro.run()
    strat = results[0]

    # 获取分析器数据
    analysis = strat.analyzers.returns.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()

    # 处理数据为None的情况，用0.0填充
    sharpe = {key: 0.0 if value is None else value for key, value in sharpe.items()}
    analysis = {key: 0.0 if value is None else value for key, value in analysis.items()}
    drawdown = {
        key: {"drawdown": 0.0} if value is None else value
        for key, value in drawdown.items()
    }

    # 生成交易记录表格
    trade_table = strat.get_trade_table()
    if trade_table != "没有发生交易":
        logger.info("\n交易记录:")
        logger.info(f"\n{trade_table}")

    # 如果有通知器，发送最后一天的信号数据和性能数据
    if notifiers and hasattr(strat, "last_signal_data"):
        # 添加性能数据到信号详情
        performance_text = "\n\n策略表现:\n"
        performance_text += f"- 夏普比率: {sharpe['sharperatio']:.4f}\n"
        performance_text += f"- 年化收益率: {analysis['rnorm100']:.2f}%\n"
        performance_text += f"- 最大回撤: {drawdown['max']['drawdown']:.2f}%"

        strat.last_signal_data["signal_details"] += performance_text
        strat.last_signal_data["trade_table"] = trade_table

        # 发送通知
        for notifier_name, notifier in notifiers.items():
            message_type = (
                "markdown" if isinstance(notifier, WecomBotNotifier) else "html"
            )
            message = build_message(
                strat.last_signal_data,
                strat.last_holding_data,
                message_type=message_type,
            )
            response = notifier.send(message)
            if debug:
                logger.debug(
                    f"通知发送结果 - 通知器: {notifier_name}, " f"返回值: {response}"
                )

    return {
        "短期均线": strat.params.short_period,
        "长期均线": strat.params.long_period,
        "止损比例": stop_loss,  # 使用传入的参数值
        "ATR周期": strat.params.atr_period,
        "ATR倍数": strat.params.atr_multiplier,
        "使用ATR止损": strat.params.use_atr_stop,
        "资金使用比例": strat.params.cash_ratio,
        "最终资金": cerebro.broker.getvalue(),
        "夏普比率": sharpe["sharperatio"],  # 直接使用原始数据
        "年化收益率": analysis["rnorm100"],  # 直接使用原始数据
        "最大回撤": drawdown["max"]["drawdown"],  # 直接使用原始数据
    }


def optimize_parameters(
    symbol,
    start_date,
    end_date,
    initial_cash,
    short_period_range,
    long_period_range,
    stop_loss_range,
    atr_period,
    atr_multiplier_range,
    use_atr_stop,
    cash_ratio_range,
    enable_crossover_sell,
    debug,
    notifiers=None,
    trend_filter_period=0,  # 添加趋势过滤器参数
    signal_delay=0,  # 添加信号滞后参数
    adx_period=0,  # 添加ADX参数
    adx_threshold=0,  # 添加ADX阈值参数
    bbands_period=0,  # 添加布林带参数
    bbands_width_threshold=0.0,  # 添加布林带宽度阈值参数
):
    results = []

    for (
        short_period,
        long_period,
        stop_loss,
        atr_multiplier,
        cash_ratio,
    ) in itertools.product(
        short_period_range,
        long_period_range,
        stop_loss_range,
        atr_multiplier_range,
        cash_ratio_range,
    ):
        if short_period >= long_period:
            continue

        current_atr_period = atr_period if atr_period is not None else long_period
        if debug:
            logger.debug(
                f"当前参数组合 - 长期均线: {long_period}, ATR周期: {current_atr_period}"
            )

        result = run_backtest(
            symbol,
            start_date,
            end_date,
            initial_cash,
            short_period,
            long_period,
            stop_loss,
            current_atr_period,
            atr_multiplier,
            use_atr_stop,
            cash_ratio,
            enable_crossover_sell,
            debug,
            notifiers=notifiers,
            trend_filter_period=trend_filter_period,  # 传递趋势过滤器参数
            signal_delay=signal_delay,  # 传递信号滞后参数
            adx_period=adx_period,  # 传递ADX参数
            adx_threshold=adx_threshold,  # 传递ADX阈值参数
            bbands_period=bbands_period,  # 传递布林带参数
            bbands_width_threshold=bbands_width_threshold,  # 传递布林带宽度阈值参数
        )
        if result:
            results.append(result)

    return results


# 在文件开头添加这个新函数
def parse_range(range_str):
    values = []
    for part in range_str.split(","):
        if "-" in part:
            start, end = map(float, part.split("-"))
            if start.is_integer() and end.is_integer():
                values.extend(range(int(start), int(end) + 1))
            else:
                values.extend(np.arange(start, end + 0.01, 0.01).tolist())
        else:
            values.append(float(part))
    return sorted(set(values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF策略回测程序")
    parser.add_argument(
        "symbol",
        type=str,
        help="ETF代码，如159915（无需添加.SZ/.SS后缀，程序会自动处理）",
    )
    parser.add_argument(
        "--start-date", type=str, default="2018-01-01", help="回测开始日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="回测结束日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial-cash", type=float, default=10000.0, help="初资金，默认10000元"
    )
    parser.add_argument(
        "--short-period-range",
        type=str,
        default="20",
        help="短期均线范围，用逗号分隔或短横线表示范围，如：10,20,30 或 10-30",
    )
    parser.add_argument(
        "--long-period-range",
        type=str,
        default="26",
        help="长期均线范围，逗号分隔或短横线表示范围",
    )
    parser.add_argument(
        "--stop-loss-range",
        type=str,
        default="0.10",
        help="止损比例范围，用逗号分隔或短横线表示范围",
    )
    parser.add_argument(
        "--cash-ratio-range",
        type=str,
        default="0.95",
        help="使用资金比例范围，用逗号分隔",
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        help="ATR计算周期，默认使用长期均线周期",
        default=None,
    )
    parser.add_argument(
        "--atr-multiplier-range",
        type=str,
        default="2.5",
        help="ATR倍数范围，用逗号分隔或短横线表示范围",
    )
    parser.add_argument(
        "--use-atr-stop",
        action="store_true",
        help="使用ATR动态止损",
    )
    parser.add_argument(
        "--disable-crossover-sell",
        action="store_true",
        help="禁用均线下穿出策略",
    )
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    parser.add_argument(
        "--email-recipients",
        type=str,
        nargs="+",
        help="邮件接收者列表，多个接收者用空格分隔。指定此参数将启用邮件通知。",
    )
    parser.add_argument(
        "--wecom-webhook",
        type=str,
        help="企业微信机器人webhook地址。指定此参数将启用企业微信通知。",
    )
    parser.add_argument(
        "--bark-url",
        type=str,
        help="Bark推送URL指定此参数将启用Bark通知。",
    )

    parser.add_argument(
        "--trend-filter-period",
        type=int,
        default=0,
        help="趋势过滤器周期，0表示不使用。使用时，只在价格高于该周期均线时允许买入",
    )
    parser.add_argument(
        "--signal-delay",
        type=int,
        default=0,
        help="信号滞后天数，0表示不滞后。使用时，需要连续N天保持信号才执行交易",
    )

    parser.add_argument(
        "--adx-period",
        type=int,
        default=14,
        help="ADX指标周期，默认14。使用时，只在ADX大于阈值时允许交易",
    )
    parser.add_argument(
        "--adx-threshold",
        type=float,
        default=0,
        help="ADX指标阈值，0表示不使用。使用时，只在ADX大于此值时允许交易",
    )
    parser.add_argument(
        "--bbands-period",
        type=int,
        default=20,
        help="布林带周期，默认20。使用时，只在布林带宽度大于阈值时允许交易",
    )
    parser.add_argument(
        "--bbands-width-threshold",
        type=float,
        default=0.0,
        help="布林带宽度阈值（百分比），0表示不使用。使用时，只在布林带宽度大于此值时允许交易",
    )

    args = parser.parse_args()

    # 移除可能的市场后缀
    symbol = args.symbol.split(".")[0]

    # 验证初始资金
    if args.initial_cash <= 0:
        logger.error("初始资金必须大于0")
        sys.exit(1)

    short_period_range = [int(x) for x in parse_range(args.short_period_range)]
    long_period_range = [int(x) for x in parse_range(args.long_period_range)]
    stop_loss_range = [round(x, 2) for x in parse_range(args.stop_loss_range)]
    cash_ratio_range = [round(x, 2) for x in parse_range(args.cash_ratio_range)]
    atr_multiplier_range = [round(x, 1) for x in parse_range(args.atr_multiplier_range)]

    # 自动别需要启用的通知器
    notifier_types = []
    if args.wecom_webhook:
        notifier_types.append("wecom")
        logger.debug(f"检测到企业微信webhook: {args.wecom_webhook}")
    if args.bark_url:
        notifier_types.append("bark")
        logger.debug(f"检测到Bark推送URL: {args.bark_url}")
    if args.email_recipients:
        notifier_types.append("mail")
        logger.debug(f"检测到邮件接收者: {args.email_recipients}")

    # 初始化通知器
    notifiers = {}
    if notifier_types:
        notifiers = get_notifiers(
            notifier_types,
            args.email_recipients,
            args.wecom_webhook,
            args.bark_url,
        )
        logger.debug(f"初始化的通知器: {notifiers}")

    # 如果指定了 atr_multiplier_range，则自动启用 ATR 止损
    if args.atr_multiplier_range:
        args.use_atr_stop = True

    # 在运行回测前添加 ATR 周期的处理
    if args.atr_period is None:
        # 如果用户没有指定 ATR 周期，使用长期均线的值
        args.atr_period = long_period_range[0]  # 使用第一个长期均线值
        logger.debug(f"ATR周期未指定，使用长期均线周期: {args.atr_period}")

    # 运行回测
    results = optimize_parameters(
        symbol,
        args.start_date,
        args.end_date,
        args.initial_cash,
        short_period_range,
        long_period_range,
        stop_loss_range,
        args.atr_period,
        atr_multiplier_range,
        args.use_atr_stop,
        cash_ratio_range,
        not args.disable_crossover_sell,
        args.debug,
        notifiers=notifiers,
        trend_filter_period=args.trend_filter_period,  # 添加趋势过滤器参数
        signal_delay=args.signal_delay,  # 添加信号滞后参数
        adx_period=args.adx_period,
        adx_threshold=args.adx_threshold,
        bbands_period=args.bbands_period,
        bbands_width_threshold=args.bbands_width_threshold,
    )

    # 将排序逻辑修改为按年化收益率降序排列
    def sort_key(x):
        return (
            x["年化收益率"] is not None,
            x["年化收益率"] if x["年化收益率"] is not None else float("-inf"),
        )

    sorted_results = sorted(results, key=sort_key, reverse=True)

    # 打印所有参数组合的表现结果
    headers = [
        "短期均线",
        "长期均线",
        "止损比例",
        "ATR周期",
        "ATR倍数",
        "使用ATR止损",
        "资金使用比例",
        "最终资金",
        "年化收益率",
        "夏普比率",
        "最大回撤",
    ]
    table_data = []
    for r in sorted_results:
        row = [
            r["短期均线"],
            r["长期均线"],
            f"{r['止损比例']:.2f}",
            f"{r['ATR周期']}",
            f"{r['ATR倍数']}",
            f"{r['使用ATR止损']}",
            f"{r['资金使用比例']:.2f}",
            f"{r['最终资金']:.2f}",
            (
                r["年化收益率"]
                if r["年化收益率"] == "N/A"
                else f"{float(r['年化收益率']):.2f}"
            ),
            r["夏普比率"] if r["夏普比率"] == "N/A" else f"{float(r['夏普比率']):.4f}",
            r["最大回撤"] if r["最大回撤"] == "N/A" else f"{float(r['最大回撤']):.2f}",
        ]
        table_data.append(row)

    # 打印结果
    logger.info("\n所有参数组合的表现结果（按年化收益率降序排列）:")
    logger.info(
        "\n"
        + tabulate(table_data, headers=headers, tablefmt="pretty", numalign="right")
    )

    # 打印最佳参数组合（如果有结果的话）
    if sorted_results:
        best_params = sorted_results[0]
        best_row = [
            best_params["短期均线"],
            best_params["长期均线"],
            f"{best_params['止损比例']:.2f}",
            f"{best_params['ATR周期']}",
            f"{best_params['ATR倍数']}",
            f"{best_params['使用ATR止损']}",
            f"{best_params['资金使用比例']:.2f}",
            f"{best_params['最终资金']:.2f}",
            (
                best_params["年化收益率"]
                if best_params["年化收益率"] == "N/A"
                else f"{float(best_params['年化收益率']):.2f}"
            ),
            (
                best_params["夏普比率"]
                if best_params["夏普比率"] == "N/A"
                else f"{float(best_params['夏普比率']):.4f}"
            ),
            (
                best_params["最大回撤"]
                if best_params["最大回撤"] == "N/A"
                else f"{float(best_params['最大回撤']):.2f}"
            ),
        ]
        logger.info("\n最佳参数组合（基于最高年化收益率）:")
        logger.info(
            "\n"
            + tabulate([best_row], headers=headers, tablefmt="pretty", numalign="right")
        )
        logger.info(
            f"\n最佳参数组合的回测结果图表已保存为 {args.symbol}_backtest_result.png"
        )
    else:
        logger.warning("\n没有找到有效的参数组合。请尝试调整参数范围或增加回测时间。")
