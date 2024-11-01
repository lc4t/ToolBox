import argparse
import itertools
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional

import backtrader as bt
import matplotlib
import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finance.config import get_notifiers
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
        ("atr_period", 14),  # ATR周期
        ("atr_multiplier", 2.5),  # ATR倍数
        ("use_atr_stop", True),  # 是否使用ATR止损
        ("cash_ratio", 0.95),  # 使用95%的可用资金
        ("enable_crossover_sell", True),  # 是否启用均线下穿卖出
        ("debug", False),  # debug模式开关
        ("symbol", None),  # ETF代码
        ("notifiers", {}),  # 通知器字典
        ("signal_date", None),  # 信号日期
        ("notify_date", None),  # 修改参数名
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

        # 添加移动平均线指标
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.short_period
        )
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.long_period
        )

        # 添加ATR指标
        self.atr = bt.indicators.ATR(self.datas[0], period=self.params.atr_period)

        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

    def next(self):
        # 获取当前日期和下一个交易日
        current_date = self.data.datetime.date(0)
        try:
            next_date = self.data.datetime.date(1)  # 尝试获取下一个交易日
        except IndexError:
            next_date = current_date  # 如果是最后一个交易日，使用当前日期

        # 检查是否需要发送通知
        should_send_notify = False
        today = datetime.now().date()

        # 添加更详细的调试日志
        if self.params.debug:
            logger.debug(
                f"通知条件检查:\n"
                f"- 当前日期: {current_date}\n"
                f"- 下一个交易日: {next_date}\n"
                f"- 今天: {today}\n"
                f"- 最后一个交易日: {self.data.datetime.date(-1)}\n"
                f"- 最后信号日期: {getattr(self, 'last_signal_date', None)}\n"
                f"- 通知器: {list(self.params.notifiers.keys()) if self.params.notifiers else '无'}"
            )

        # 添加日期过滤，确保每个日期只发送一次通知
        if hasattr(self, "last_signal_date") and current_date == self.last_signal_date:
            logger.debug(f"跳过重复通知 - 日期: {current_date}")
            return  # 如果今天已经发过信号了，就不再发送

        if self.order:
            return

        # 准备基础信号数据
        signal_data = {
            "symbol": self.params.symbol,
            "date": current_date,
            "price": self.dataclose[0],
            "sma_short": self.sma_short[0],
            "sma_long": self.sma_long[0],
            "atr": self.atr[0],
            "short_period": self.params.short_period,
            "long_period": self.params.long_period,
            "atr_period": self.params.atr_period,
            "atr_multiplier": self.params.atr_multiplier,
            "stop_loss": self.params.stop_loss,
            "signal_type": "持有" if self.position else "观察",  # 默认状态
        }

        # 构建持仓数据（如果有）
        holding_data = None
        if self.position:
            holding_data = {
                "买入时间": self.buy_date,
                "买入价格": self.buyprice,
                "当前收益": (
                    (self.dataclose[0] - self.buyprice) / self.buyprice
                    if self.buyprice
                    else 0
                ),
                "最高价格": self.highest_price,
                "当前价格": self.dataclose[0],
            }

        # 修改判断逻辑：当next_date == current_date时，说明这是最后一个交易日
        if next_date == current_date:
            should_send_notify = True
            logger.debug(
                f"将发送通知 - 当前日期: {current_date}, "
                f"原因: 最后一个交易日, "
                f"should_send_notify: {should_send_notify}, "
                f"has_notifiers: {bool(self.params.notifiers)}"
            )

            # 在最后一个交易日，直接发送当前状态通知
            if self.params.notifiers:
                # 更新 signal_data 添加必要的字段
                signal_data.update(
                    {
                        "signal_details": (
                            f"当前状态指标:\n"
                            f"- 短期均线(MA{self.params.short_period}): {self.sma_short[0]:.3f}\n"
                            f"- 长期均线(MA{self.params.long_period}): {self.sma_long[0]:.3f}\n"
                            f"- 均线差值: {(self.sma_short[0] - self.sma_long[0]):.3f}\n"
                            f"- ATR({self.params.atr_period}): {self.atr[0]:.3f}\n"
                            f"- 当前价格: {self.dataclose[0]:.3f}"
                        ),
                        "signal_description": "最后交易日状态汇报",
                        "position_details": "持有" if self.position else "观察",
                    }
                )

                message = build_message(signal_data, holding_data)
                if self.params.debug:
                    logger.debug(f"准备发送的消息内容: {message}")

                for notifier_name, notifier in self.params.notifiers.items():
                    try:
                        response = notifier.send(message)
                        if self.params.debug:
                            logger.debug(
                                f"通知发送结果 - 通知器: {notifier_name}, "
                                f"返回值: {response}"
                            )
                    except Exception as e:
                        logger.error(
                            f"发送通知失败 - 通知器: {notifier_name}, "
                            f"错误: {str(e)}"
                        )
                self.last_signal_date = current_date

        # 添加日期过滤，确保每个日期只发送一次通知
        if hasattr(self, "last_signal_date") and current_date == self.last_signal_date:
            logger.debug(f"跳过重复通知 - 日期: {current_date}")
            return  # 如果今天已经发过信号了，就不再发送

        if self.order:
            return

        # 准备基础信号数据
        signal_data = {
            "symbol": self.params.symbol,
            "date": current_date,
            "price": self.dataclose[0],
            "sma_short": self.sma_short[0],
            "sma_long": self.sma_long[0],
            "atr": self.atr[0],
            "short_period": self.params.short_period,
            "long_period": self.params.long_period,
            "atr_period": self.params.atr_period,
            "atr_multiplier": self.params.atr_multiplier,
            "stop_loss": self.params.stop_loss,
            "signal_type": "持有" if self.position else "观察",  # 默认状态
        }

        # 构建持仓数据（如果有）
        holding_data = None
        if self.position:
            holding_data = {
                "买入时间": self.buy_date,
                "买入价格": self.buyprice,
                "当前收益": (
                    (self.dataclose[0] - self.buyprice) / self.buyprice
                    if self.buyprice
                    else 0
                ),
                "最高价格": self.highest_price,
                "当前价格": self.dataclose[0],
            }

        # 执行交易逻辑，不受通知日期限制
        if not self.position:  # 没有持仓
            if self.crossover > 0:  # 买入信号
                # 先计算买入数量
                available_cash = self.broker.getcash() * self.params.cash_ratio
                size = int(available_cash / self.dataclose[0])

                # 更新信号数据
                signal_data.update(
                    {
                        "signal_type": "买入",
                        "signal_details": (
                            f"买入信号指标:\n"
                            f"- 短期均线(MA{self.params.short_period}): {self.sma_short[0]:.3f}\n"
                            f"- 长期均线(MA{self.params.long_period}): {self.sma_long[0]:.3f}\n"
                            f"- 均线差值: {(self.sma_short[0] - self.sma_long[0]):.3f}\n"
                            f"- ATR({self.params.atr_period}): {self.atr[0]:.3f}\n\n"
                            f"建议买入价格: {self.dataclose[0]:.3f}\n"
                            f"建议买入数量: {size}股\n"
                            f"预计使用资金: {size * self.dataclose[0]:.2f}\n"
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
                    if should_send_notify and self.params.notifiers:
                        logger.info(
                            f"发送买入通知 - 日期: {current_date}, "
                            f"价格: {self.dataclose[0]:.2f}, "
                            f"数量: {size}"
                        )
                        message = build_message(signal_data, holding_data)
                        if self.params.debug:
                            logger.debug(f"准备发送的消息内容: {message}")

                        for notifier_name, notifier in self.params.notifiers.items():
                            try:
                                response = notifier.send(message)
                                if self.params.debug:
                                    logger.debug(
                                        f"通知发送结果 - 通知器: {notifier_name}, "
                                        f"返回值: {response}"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"发送通知失败 - 通知器: {notifier_name}, "
                                    f"错误: {str(e)}"
                                )
                        self.last_signal_date = current_date
                    else:
                        logger.debug(
                            f"跳过买入通知 - 日期: {current_date}, "
                            f"should_send_notify: {should_send_notify}, "
                            f"has_notifiers: {bool(self.params.notifiers)}, "
                            f"notifiers: {list(self.params.notifiers.keys()) if self.params.notifiers else '无'}"
                        )

        else:  # 有持仓
            # 更新最高价格
            if self.dataclose[0] > self.highest_price:
                self.highest_price = self.dataclose[0]

            # 计算止损价格
            if self.params.use_atr_stop:
                stop_price = self.highest_price - (
                    self.atr[0] * self.params.atr_multiplier
                )
                min_stop_price = self.highest_price * (1 - self.params.stop_loss)
                stop_price = max(stop_price, min_stop_price)
            else:
                stop_price = self.highest_price * (1 - self.params.stop_loss)

            # 检查是否触发止损
            if self.dataclose[0] <= stop_price:
                signal_data.update(
                    {
                        "signal_type": "卖出",
                        "signal_details": (
                            f"止损卖出指标:\n"
                            f"- 当前价格: {self.dataclose[0]:.3f}\n"
                            f"- 最高价: {self.highest_price:.3f}\n"
                            f"- ATR止损价: {self.highest_price - (self.atr[0] * self.params.atr_multiplier):.3f}\n"
                            f"- 比例止损价: {self.highest_price * (1 - self.params.stop_loss):.3f}\n"
                            f"- 最终止损价: {stop_price:.3f}\n"
                            f"- 跌幅: {((self.dataclose[0] - self.highest_price) / self.highest_price * 100):.2f}%\n"
                            f"- ATR({self.params.atr_period}): {self.atr[0]:.3f}"
                        ),
                        "position_details": "即将清空持仓",
                        "signal_description": "触发止损条件，建议卖出",
                    }
                )
                order = self.sell(size=self.position.size)
                order.sell_reason = "触发止损"  # 设置卖出原因
                self.order = order

                # 只在需要时发送通知
                if should_send_notify and self.params.notifiers:
                    logger.info(
                        f"发送止损卖出通知 - 日期: {current_date}, "
                        f"价格: {self.dataclose[0]:.2f}, "
                        f"止损价: {stop_price:.2f}"
                    )
                    message = build_message(signal_data, holding_data)
                    if self.params.debug:
                        logger.debug(f"准备发送的消息内容: {message}")

                    for notifier_name, notifier in self.params.notifiers.items():
                        try:
                            response = notifier.send(message)
                            if self.params.debug:
                                logger.debug(
                                    f"通知发送结果 - 通知器: {notifier_name}, "
                                    f"返回值: {response}"
                                )
                        except Exception as e:
                            logger.error(
                                f"发送通知失败 - 通知器: {notifier_name}, "
                                f"错误: {str(e)}"
                            )
                    self.last_signal_date = current_date
                else:
                    logger.debug(
                        f"跳过止损卖出通知 - 日期: {current_date}, "
                        f"should_send_notify: {should_send_notify}, "
                        f"has_notifiers: {bool(self.params.notifiers)}, "
                        f"notifiers: {list(self.params.notifiers.keys()) if self.params.notifiers else '无'}"
                    )

            # 检查是否触发均线下穿卖出
            elif self.params.enable_crossover_sell and self.crossover < 0:
                signal_data.update(
                    {
                        "signal_type": "卖出",
                        "signal_details": (
                            f"均线下穿卖出指标:\n"
                            f"- 短期均线(MA{self.params.short_period}): {self.sma_short[0]:.3f}\n"
                            f"- 长期均线(MA{self.params.long_period}): {self.sma_long[0]:.3f}\n"
                            f"- 均线差值: {(self.sma_short[0] - self.sma_long[0]):.3f}\n"
                            f"- ATR({self.params.atr_period}): {self.atr[0]:.3f}\n"
                            f"- 当前价格: {self.dataclose[0]:.3f}\n"
                            f"- 持仓收益: {((self.dataclose[0] - self.buyprice) / self.buyprice * 100):.2f}%"
                        ),
                        "position_details": "即将清空持仓",
                        "signal_description": "短期均线下穿长期均线，建议卖出",
                    }
                )
                order = self.sell(size=self.position.size)
                order.sell_reason = "均线下穿"  # 设置卖出原因
                self.order = order

                # 只在需要时发送通知
                if should_send_notify and self.params.notifiers:
                    logger.info(
                        f"发送均线下穿卖出通知 - 日期: {current_date}, "
                        f"价格: {self.dataclose[0]:.2f}"
                    )
                    message = build_message(signal_data, holding_data)
                    if self.params.debug:
                        logger.debug(f"准备发送的消息内容: {message}")

                    for notifier_name, notifier in self.params.notifiers.items():
                        try:
                            response = notifier.send(message)
                            if self.params.debug:
                                logger.debug(
                                    f"通知发送结果 - 通知器: {notifier_name}, "
                                    f"返回值: {response}"
                                )
                        except Exception as e:
                            logger.error(
                                f"发送通知失败 - 通知器: {notifier_name}, "
                                f"错误: {str(e)}"
                            )
                    self.last_signal_date = current_date
                else:
                    logger.debug(
                        f"跳过均线下穿卖出通知 - 日期: {current_date}, "
                        f"should_send_notify: {should_send_notify}, "
                        f"has_notifiers: {bool(self.params.notifiers)}, "
                        f"notifiers: {list(self.params.notifiers.keys()) if self.params.notifiers else '无'}"
                    )

            else:  # 持有信号
                signal_data.update(
                    {
                        "signal_type": "持有",
                        "signal_details": (
                            f"持有状态指标:\n"
                            f"- 短期均线(MA{self.params.short_period}): {self.sma_short[0]:.3f}\n"
                            f"- 长期均线(MA{self.params.long_period}): {self.sma_long[0]:.3f}\n"
                            f"- 均线差值: {(self.sma_short[0] - self.sma_long[0]):.3f}\n"
                            f"- ATR({self.params.atr_period}): {self.atr[0]:.3f}\n"
                            f"- 当前价格: {self.dataclose[0]:.3f}\n"
                            f"- 最高价格: {self.highest_price:.3f}\n"
                            f"- 止损价格: {stop_price:.3f}\n"
                            f"- 止损距离: {((self.dataclose[0] - stop_price) / self.dataclose[0] * 100):.2f}%\n"
                            f"- 持仓收益: {((self.dataclose[0] - self.buyprice) / self.buyprice * 100):.2f}%"
                        ),
                        "position_details": "继续持有",
                        "signal_description": "当前位置安全，继续持有",
                    }
                )

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
                        "剩余资金": self.broker.getcash(),
                    }
                )

                logger.info(
                    f"Buy executed - Date: {self.data.datetime.date(0)}, "
                    f"Price: {order.executed.price:.2f}, "
                    f"Size: {order.executed.size}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Commission: {order.executed.comm:.2f}"
                )
            else:
                sell_value = order.executed.price * order.executed.size
                commission = order.executed.comm
                profit = round(
                    -(
                        sell_value
                        - self.buyprice * order.executed.size
                        - commission
                        - self.buycomm
                    ),
                    4,
                )

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

                # 记录卖出交易
                self.trades.append(
                    {
                        "类型": "卖出",
                        "日期": self.data.datetime.date(0),
                        "价格": order.executed.price,
                        "数量": -order.executed.size,
                        "获得资金": sell_value,
                        "手续费": commission,
                        "收益": profit,
                        "剩余资金": self.broker.getcash(),
                        "指标条件": indicator_condition,  # 保存实际的指标条件
                    }
                )

                logger.info(
                    f"Sell executed - Date: {self.data.datetime.date(0)}, "
                    f"Price: {order.executed.price:.2f}, "
                    f"Size: {order.executed.size}, "
                    f"Profit: {profit:.2f}, "
                    f"Reason: {getattr(order, 'sell_reason', '未知原因')}"  # 添加卖出原因到日志
                )
                self.highest_price = 0  # 重置最高价格
                self.buy_date = None  # 重置买入日期
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(
                f"Order {order.ref} {order.Status[order.status]} on {self.data.datetime.date(0)}, "
                f"Price: {self.dataclose[0]:.2f}, Cash: {self.broker.getcash():.2f}, "
                f"Reason: {order.info.get('reject_reason', 'Unknown')}, "
                f"Size: {order.created.size}, Price: {order.created.price:.2f}"
            )

        self.order = None

    def stop(self):
        print("\n交易记录:")
        if not self.trades:
            print("没有发生交易")
        else:
            headers = [
                "类型",
                "日期",
                "价格",
                "数量",
                "消耗/获得资金",
                "手续费",
                "收益",
                "剩余资金",
                "指标条件",
            ]
            table_data = []
            for trade in self.trades:
                if trade["类型"] == "买入":
                    table_data.append(
                        [
                            trade["类型"],
                            trade["日期"],
                            f"{trade['价格']:.3f}",
                            f"{trade['数量']:.0f}",
                            f"{trade['消耗资金']:.4f}",
                            f"{trade['手续费']:.4f}",
                            "-",
                            f"{trade['剩余资金']:.4f}",
                            f"MA{self.params.short_period}({self.sma_short[0]:.3f}) > MA{self.params.long_period}({self.sma_long[0]:.3f})",
                        ]
                    )
                else:
                    table_data.append(
                        [
                            trade["类型"],
                            trade["日期"],
                            f"{trade['价格']:.3f}",
                            f"{trade['数量']:.0f}",
                            f"{trade['获得资金']:.4f}",
                            f"{trade['手续费']:.4f}",
                            f"{trade['收益']:.4f}",
                            f"{trade['剩余资金']:.4f}",
                            trade["指标条件"],  # 使用保存的指标条件
                        ]
                    )

            # 将打印操作移到循环外
            print(
                tabulate(
                    table_data,
                    headers=headers,
                    tablefmt="grid",
                    colalign=(
                        "left",  # 类型
                        "left",  # 日期
                        "right",  # 价格
                        "right",  # 数量
                        "right",  # 消耗/获得资金
                        "right",  # 手续费
                        "right",  # 收益
                        "right",  # 剩余资金
                        "left",  # 指标条件
                    ),
                )
            )


# 获取ETF数据
def get_etf_data(symbol, start_date=None):
    """
    获取ETF数据，总是获取全部历史数据，start_date 只用回测过滤
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    downloads_dir = os.path.join(script_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    csv_file = os.path.join(downloads_dir, f"etf_data_{symbol}.csv")

    try:
        need_update = False
        if os.path.exists(csv_file):
            # 检查现有数据是否是最新的
            existing_data = pd.read_csv(csv_file, encoding="utf-8")
            existing_data["日期"] = pd.to_datetime(existing_data["日期"]).dt.normalize()
            latest_date = existing_data["日期"].max()

            # 如果最新数据不是昨天或更新，则需更新
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
            # 使用 fetcher 获取数据
            get_and_save_etf_data(
                symbol,
                "2000-01-01",  # 使用较早的日期以获取全部历史数据
                datetime.now().strftime("%Y-%m-%d"),
            )

        # 读取数据文件
        df = pd.read_csv(csv_file, encoding="utf-8")
        if df.empty:
            logger.error(f"错误：{symbol} 的数据文件为空。")
            return None

        # 将日期列转换为datetime索引，并移除时区信息
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

        # 如果指定了回测开始日期，则过滤数据
        if start_date:
            start_date = pd.to_datetime(start_date).normalize()
            df = df[df.index >= start_date]
            logger.info(
                f"回测数据范围：{df.index.min().date()} 到 {df.index.max().date()}"
            )

        return df[numeric_columns]  # 只返回数值列

    except Exception as e:
        logger.error(f"处理 {symbol} 数据时出错: {e}")
        return None


# 主函数
def run_backtest(
    symbol,
    start_date,
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
    notify_date=None,
):
    cerebro = bt.Cerebro()

    # 传入开始日期，确保数据是最新
    data = get_etf_data(symbol, start_date)
    if data is None:
        return None

    # 日期过滤，确保无时区
    start_date = pd.to_datetime(start_date).tz_localize(None)
    data = data[(data.index >= start_date)]

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
        symbol=symbol,
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
        notify_date=notify_date,
    )

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    results = cerebro.run()
    strat = results[0]

    return {
        "短期均线": short_period,
        "长期均线": long_period,
        "止损比例": stop_loss,
        "ATR周期": atr_period,
        "ATR倍数": atr_multiplier,
        "使用ATR止损": use_atr_stop,
        "资金使用比例": cash_ratio,
        "最终资金": cerebro.broker.getvalue(),
        "夏普比率": strat.analyzers.sharpe.get_analysis()["sharperatio"],
        "年化收益率": strat.analyzers.returns.get_analysis()["rnorm100"],
        "最大回撤": strat.analyzers.drawdown.get_analysis()["max"]["drawdown"],
    }


def optimize_parameters(
    symbol,
    start_date,
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
    notify_date=None,
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
            continue  # 跳过短期均线大于或等于长期均线的情

        result = run_backtest(
            symbol,
            start_date,
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
            notifiers=notifiers,
            notify_date=notify_date,
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
    parser = argparse.ArgumentParser(description="ETF回测数优化")
    parser.add_argument(
        "symbol",
        type=str,
        help="ETF代码，如159915（无需添加.SZ/.SS后缀，程序会自动处理）",
    )
    parser.add_argument(
        "--start_date", type=str, default="2018-01-01", help="回测开始日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--initial_cash", type=float, default=10000.0, help="初资金，默认10000元"
    )
    parser.add_argument(
        "--short_period_range",
        type=str,
        default="20",
        help="短期均线范围，用逗号分隔或短横线表示范围，如：10,20,30 或 10-30",
    )
    parser.add_argument(
        "--long_period_range",
        type=str,
        default="26",
        help="长期均线范围，逗号分隔或短横线表示范围",
    )
    parser.add_argument(
        "--stop_loss_range",
        type=str,
        default="0.10",
        help="止损比例范围，用逗号分隔或短横线表示范围",
    )
    parser.add_argument(
        "--cash_ratio_range",
        type=str,
        default="0.95",
        help="使用资金比例范围，用逗号分隔",
    )
    parser.add_argument(
        "--atr_period",
        type=int,
        default=14,
        help="ATR计算周期",
    )
    parser.add_argument(
        "--atr_multiplier_range",
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
    # parser.add_argument(
    #     "--notifiers",
    #     type=str,
    #     nargs="+",
    #     choices=["mail", "wecom", "bark"],
    #     help="通知方式，可选多个：mail=邮件, wecom=企业微信机器人, bark=Bark推送",
    # )
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
        help="Bark推送URL。指定此参数将启用Bark通知。",
    )
    parser.add_argument(
        "--notify-date",
        type=str,
        help="指定要发送通知的日期 (YYYY-MM-DD)，默认为今天",
        default=datetime.now().strftime("%Y-%m-%d"),
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

    # 自动识别需要启用的通知器
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

    # 处理通知日期
    notify_date = None
    if args.notify_date:
        try:
            notify_date = pd.to_datetime(args.notify_date).date()
            today = datetime.now().date()
            if notify_date > today:
                logger.warning(f"指定的日期 {notify_date} 是未来日期，将使用今天的日期")
                notify_date = today
            logger.debug(f"设置通知日期为: {notify_date}")
        except Exception as e:
            logger.error(f"通知日期格式错误: {e}")
            sys.exit(1)

    # 如果指定了 atr_multiplier_range，则自动启用 ATR 止损
    if args.atr_multiplier_range:
        args.use_atr_stop = True

    # 运行回测
    results = optimize_parameters(
        symbol,
        args.start_date,
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
        notify_date=notify_date,
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
            f"{r['年化收益率']:.2f}",
            f"{r['夏普比率']:.4f}" if r["夏普比率"] is not None else "N/A",
            f"{r['最大回撤']:.2f}",
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
            f"{best_params['年化收益率']:.2f}",
            (
                f"{best_params['夏普比率']:.4f}"
                if best_params["夏普比率"] is not None
                else "N/A"
            ),
            f"{best_params['最大回撤']:.2f}",
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
