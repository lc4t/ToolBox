import argparse
import itertools
import os
import sys
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional


class FilterMode(Enum):
    """过滤器模式枚举类"""

    BOTH = "both"  # 买入和卖出都过滤
    BUY_ONLY = "buy_only"  # 只过滤买入
    NONE = "none"  # 不过滤


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
        ("bbands_width_dynamic", False),  # 是否使用动态布林带宽度阈值（中位数倍数）
        ("trend_filter_period", 0),  # 趋势过滤器周期，0表示不使用
        ("signal_delay", 0),  # 信号滞后天数，0表示不滞后
        ("adx_filter_mode", FilterMode.BOTH),  # ADX过滤器模式
        ("bbands_filter_mode", FilterMode.BOTH),  # 布林带过滤器模式
        ("trend_filter_mode", FilterMode.BOTH),  # 趋势过滤器模式
        ("signal_delay_mode", FilterMode.BOTH),  # 信号滞后模式
        ("start_date", None),  # 添加回测开始日期参数
        ("macd_fast_period", 0),  # MACD快线周期，0表示不使用
        ("macd_slow_period", 0),  # MACD慢线周期
        ("macd_signal_period", 0),  # MACD信号线周期
        ("macd_mode", FilterMode.BOTH),  # MACD信号模式
        ("ma_mode", FilterMode.BOTH),  # 均线信号模式
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

        # 设置 ATR 周期，确保至少为1
        atr_period = self.params.atr_period
        if atr_period is None or atr_period <= 0:
            # 如果使用MACD系统，使用MACD慢线周期
            if self.params.macd_slow_period > 0:
                atr_period = self.params.macd_slow_period
            # 如果使用均线系统，使用长期均线周期
            elif self.params.long_period > 0:
                atr_period = self.params.long_period
            # 如果都没有，使用默认值20
            else:
                atr_period = 20
            if self.params.debug:
                logger.debug(f"ATR周期未指定或无效，使用: {atr_period}")

        # 添加移动平均线指标（只在使用均线系统时添加）
        if self.params.short_period > 0 and self.params.long_period > 0:
            self.sma_short = bt.indicators.ExponentialMovingAverage(
                self.datas[0], period=self.params.short_period
            )
            self.sma_long = bt.indicators.ExponentialMovingAverage(
                self.datas[0], period=self.params.long_period
            )
            # 交叉信号
            self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)
        else:
            # 如果不使用均线系统，创建空指标
            self.sma_short = bt.indicators.ExponentialMovingAverage(
                self.datas[0], period=1
            )
            self.sma_long = self.sma_short
            self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

        # 添加ATR指标，使用处理后的 atr_period
        self.atr = bt.indicators.ATR(self.datas[0], period=atr_period)

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
            # 添加布林带宽度的移动中位数
            if self.params.bbands_width_dynamic:
                self.bbands_width_median = bt.indicators.MovingAverageSimple(
                    self.bbands_width, period=self.params.bbands_period
                )

        # 记录是否是第一次运行next方法
        self.first_next = True

        # 添加MACD指标（只在使用MACD系统时添加）
        if self.params.macd_fast_period > 0 and self.params.macd_slow_period > 0:
            self.macd = bt.indicators.MACD(
                self.datas[0],
                period_me1=self.params.macd_fast_period,
                period_me2=self.params.macd_slow_period,
                period_signal=self.params.macd_signal_period,
            )
            # MACD交叉信号
            self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        else:
            # 如果不使用MACD系统，创建空指标
            self.macd = bt.indicators.MACD(
                self.datas[0],
                period_me1=1,
                period_me2=2,
                period_signal=1,
            )
            self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        # 获取当前日期和下一个交易日
        current_date = self.data.datetime.date(0)
        try:
            next_date = self.data.datetime.date(1)
            has_next_bar = True
        except IndexError:
            has_next_bar = False
            next_date = current_date

        # 首先初始化 signal_data
        signal_data = {
            "symbol": self.params.symbol,
            "name": get_etf_name(self.params.symbol),
            "date": current_date,
            "start_date": self.params.start_date,
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
            "signal_details": "",  # 先设置为空字符串，后面再更新
        }

        # 检查信号生成条件
        ma_buy_signal = False
        ma_sell_signal = False
        macd_buy_signal = False
        macd_sell_signal = False
        trend_ok = True  # 初始化 trend_ok
        trend_filter_msg = []  # 初始化过滤器消息列表

        # 添加调试日志
        if self.params.debug:
            logger.debug(
                f"[{current_date}] 价格={self.dataclose[0]:.3f}, "
                f"MA{self.params.short_period}={self.sma_short[0]:.3f}, "
                f"MA{self.params.long_period}={self.sma_long[0]:.3f}, "
                f"差值={self.sma_short[0] - self.sma_long[0]:.3f}, "
                f"ATR={self.atr[0]:.3f}"
            )

        # 检查均线系统信号
        if self.params.short_period > 0 and self.params.long_period > 0:
            if self.crossover > 0:
                ma_buy_signal = True
                logger.debug(f"[{current_date}] MA系统生成买入信号")
            elif self.crossover < 0:
                ma_sell_signal = True
                logger.debug(f"[{current_date}] MA系统生成卖出信号")

        # 检查MACD系统信号
        if self.params.macd_fast_period > 0 and self.params.macd_slow_period > 0:
            if self.macd_cross > 0:
                macd_buy_signal = True
                logger.debug(
                    f"[{current_date}] MACD系统生成买入信号: "
                    f"MACD={self.macd.macd[0]:.3f}, "
                    f"Signal={self.macd.signal[0]:.3f}, "
                    f"Hist={self.macd.macd[0] - self.macd.signal[0]:.3f}"
                )
            elif self.macd_cross < 0:
                macd_sell_signal = True
                logger.debug(
                    f"[{current_date}] MACD系统生成卖出信号: "
                    f"MACD={self.macd.macd[0]:.3f}, "
                    f"Signal={self.macd.signal[0]:.3f}, "
                    f"Hist={self.macd.macd[0] - self.macd.signal[0]:.3f}"
                )

        # 计算止损价格
        if self.position:  # 有持仓时才计算止损价格
            if self.params.use_atr_stop:
                stop_price = self.highest_price - (
                    self.atr[0] * self.params.atr_multiplier
                )
                min_stop_price = self.highest_price * (1 - self.params.stop_loss)
                stop_price = max(stop_price, min_stop_price)
                if self.params.debug:
                    logger.debug(
                        f"[{current_date}] 止损价格计算: "
                        f"ATR止损={self.highest_price - (self.atr[0] * self.params.atr_multiplier):.3f}, "
                        f"比例止损={min_stop_price:.3f}, "
                        f"最终止损={stop_price:.3f}"
                    )
            else:
                stop_price = self.highest_price * (1 - self.params.stop_loss)
                if self.params.debug:
                    logger.debug(
                        f"[{current_date}] 止损价格计算: 比例止损={stop_price:.3f}"
                    )
        else:
            stop_price = 0

        # 根据模式设置最终信号
        buy_signal = False
        sell_signal = False
        sell_reason = ""

        # 买入信号处理
        if not self.position:  # 没有持仓
            # 检查MA系统买入信号
            ma_signal_valid = ma_buy_signal and self.params.ma_mode in [
                FilterMode.BOTH,
                FilterMode.BUY_ONLY,
            ]
            # 检查MACD系统买入信号
            macd_signal_valid = macd_buy_signal and self.params.macd_mode in [
                FilterMode.BOTH,
                FilterMode.BUY_ONLY,
            ]

            if ma_signal_valid or macd_signal_valid:  # 只有在有信号时才检查过滤器
                # ADX过滤器
                if self.params.adx_period > 0 and self.params.adx_threshold > 0:
                    adx_ok = self.adx[0] > self.params.adx_threshold
                    # 根据过滤器模式决定是否应用
                    if self.params.adx_filter_mode in [
                        FilterMode.BOTH,
                        FilterMode.BUY_ONLY,
                    ]:
                        if not adx_ok:
                            trend_ok = False
                    trend_filter_msg.append(
                        f"ADX({self.params.adx_period}): {self.adx[0]:.2f} "
                        f"{'>' if adx_ok else '<='} {self.params.adx_threshold} "
                        f"{'[买入过滤]' if self.params.adx_filter_mode == FilterMode.BUY_ONLY else ''}"
                    )

                # 布林带宽度过滤器
                if (
                    self.params.bbands_period > 0
                    and self.params.bbands_width_threshold > 0
                ):
                    if self.params.bbands_width_dynamic:
                        threshold = (
                            self.bbands_width_median[0]
                            * self.params.bbands_width_threshold
                        )
                        bbands_ok = self.bbands_width[0] > threshold
                    else:
                        bbands_ok = (
                            self.bbands_width[0] > self.params.bbands_width_threshold
                        )
                    # 根据过滤器模式决定是否应用
                    if self.params.bbands_filter_mode in [
                        FilterMode.BOTH,
                        FilterMode.BUY_ONLY,
                    ]:
                        if not bbands_ok:
                            trend_ok = False
                    trend_filter_msg.append(
                        f"布林带宽度({self.params.bbands_period}): {self.bbands_width[0]:.2f}% "
                        f"{'>' if bbands_ok else '<='} "
                        f"{threshold:.2f}% ({self.params.bbands_width_threshold}倍中位数) "
                        f"{'[买入过滤]' if self.params.bbands_filter_mode == FilterMode.BUY_ONLY else ''}"
                    )

                # 趋势过滤器
                if self.params.trend_filter_period > 0:
                    price_above_ma = self.dataclose[0] > self.trend_sma[0]
                    # 根据过滤器模式决定是否应用
                    if self.params.trend_filter_mode in [
                        FilterMode.BOTH,
                        FilterMode.BUY_ONLY,
                    ]:
                        trend_ok = trend_ok and price_above_ma
                    trend_filter_msg.append(
                        f"趋势过滤器({self.params.trend_filter_period}): 价格({self.dataclose[0]:.3f}) "
                        f"{'>' if price_above_ma else '<='} "
                        f"MA{self.params.trend_filter_period}({self.trend_sma[0]:.3f}) "
                        f"{'[买入过滤]' if self.params.trend_filter_mode == FilterMode.BUY_ONLY else ''}"
                    )

                if trend_ok:  # 趋势过滤器允许买入
                    buy_signal = True
                    if self.params.debug:
                        logger.debug(
                            f"[{current_date}] 生成买入信号: "
                            f"MA信号={ma_buy_signal}(模式={self.params.ma_mode.value}), "
                            f"MACD信号={macd_buy_signal}(模式={self.params.macd_mode.value})"
                        )
                else:  # 被过滤器阻止
                    if self.params.debug:
                        logger.debug(
                            f"[{current_date}] 买入信号被过滤器阻止:\n"
                            f"- MA信号={ma_buy_signal}(模式={self.params.ma_mode.value})\n"
                            f"- MACD信号={macd_buy_signal}(模式={self.params.macd_mode.value})\n"
                            f"- 过滤条件: {', '.join(trend_filter_msg)}"
                        )
                    signal_data["signal_details"] += (
                        f"\n过滤器阻止买入:\n"
                        f"- 信号类型: {'MA' if ma_signal_valid else ''}{'和' if ma_signal_valid and macd_signal_valid else ''}{'MACD' if macd_signal_valid else ''}\n"
                        f"- 过滤条件: {', '.join(trend_filter_msg)}"
                    )

        # 卖出信号处理
        else:  # 有持仓
            # 检查止损条件（止损优先级最高，不受趋势过滤器影响）
            if self.dataclose[0] <= stop_price:
                sell_signal = True
                sell_reason = "触发止损"
            else:
                # 检查MA系统卖出信号
                ma_signal_valid = (
                    ma_sell_signal and self.params.ma_mode == FilterMode.BOTH
                )
                # 检查MACD系统卖出信号
                macd_signal_valid = (
                    macd_sell_signal and self.params.macd_mode == FilterMode.BOTH
                )

                if ma_signal_valid or macd_signal_valid:
                    if trend_ok:  # 趋势过滤器允许卖出
                        sell_signal = True
                        sell_reason = "MACD下穿" if macd_signal_valid else "均线下穿"
                        if self.params.debug:
                            logger.debug(
                                f"[{current_date}] 生成卖出信号: "
                                f"MA信号={ma_sell_signal}(模式={self.params.ma_mode.value}), "
                                f"MACD信号={macd_sell_signal}(模式={self.params.macd_mode.value}), "
                                f"原因={sell_reason}"
                            )
                    else:  # 被过滤器阻止
                        if self.params.debug:
                            logger.debug(
                                f"[{current_date}] 卖出信号被过滤器阻止:\n"
                                f"- MA信号={ma_sell_signal}(模式={self.params.ma_mode.value})\n"
                                f"- MACD信号={macd_sell_signal}(模式={self.params.macd_mode.value})\n"
                                f"- 过滤条件: {', '.join(trend_filter_msg)}"
                            )
                        signal_data["signal_details"] += (
                            f"\n过滤器阻止卖出:\n"
                            f"- 信号类型: {'MA' if ma_signal_valid else ''}{'和' if ma_signal_valid and macd_signal_valid else ''}{'MACD' if macd_signal_valid else ''}\n"
                            f"- 过滤条件: {', '.join(trend_filter_msg)}"
                        )

        # 如果有买入或卖出信号，并且有下一个交易日，则在下一个交易日执行
        if has_next_bar:
            if buy_signal:
                # 计算买入数量
                available_cash = self.broker.getcash() * self.params.cash_ratio
                size = int(available_cash / self.dataclose[0])
                if size > 0:
                    self.order = self.buy(
                        size=size,
                        exectype=bt.Order.Market,
                        valid=next_date,  # 使用下一个交易日
                    )
                    logger.info(
                        f"买入信号 - 信号日期: {current_date}, "
                        f"执行日期: {next_date}, "
                        f"价格: {self.dataclose[0]:.2f}, "
                        f"数量: {size}"
                    )
            elif sell_signal:
                self.order = self.sell(
                    size=self.position.size,
                    exectype=bt.Order.Market,
                    valid=next_date,  # 使用下一个交易日
                )
                self.order.sell_reason = sell_reason
                logger.info(
                    f"卖出信号 - 信号日期: {current_date}, "
                    f"执行日期: {next_date}, "
                    f"价格: {self.dataclose[0]:.2f}, "
                    f"数量: {self.position.size}, "
                    f"原因: {sell_reason}"
                )

        # 更新信号数据中添加系统信息
        signal_details = "当前系统状态:\n"

        # 添加双均线系统状态
        if self.params.short_period > 0 and self.params.long_period > 0:
            signal_details += (
                f"双均线系统(已启用):\n"
                f"- 短期均线: MA{self.params.short_period} = {self.sma_short[0]:.3f}\n"
                f"- 长期均线: MA{self.params.long_period} = {self.sma_long[0]:.3f}\n"
                f"- 均线差值: {(self.sma_short[0] - self.sma_long[0]):.3f}\n"
                f"- 信号模式: {self.params.ma_mode.value}\n"
            )
        else:
            signal_details += "双均线系统(未启用)\n"

        # 添加MACD系统状态
        if self.params.macd_fast_period > 0 and self.params.macd_slow_period > 0:
            signal_details += (
                f"MACD系统(已启用):\n"
                f"- MACD线: {self.macd.macd[0]:.3f}\n"
                f"- 信号线: {self.macd.signal[0]:.3f}\n"
                f"- MACD柱: {(self.macd.macd[0] - self.macd.signal[0]):.3f}\n"
                f"- 快线周期: {self.params.macd_fast_period}\n"
                f"- 慢线周期: {self.params.macd_slow_period}\n"
                f"- 信号线周期: {self.params.macd_signal_period}\n"
                f"- 信号模式: {self.params.macd_mode.value}\n"
            )
        else:
            signal_details += "MACD系统(未启用)\n"

        # 添加止损策略状态
        signal_details += (
            f"\n止损策略:\n"
            f"- ATR动态止损: {'启用' if self.params.use_atr_stop else '禁用'}\n"
            f"- ATR周期: {self.params.atr_period}日\n"
            f"- ATR倍数: {self.params.atr_multiplier}\n"
            f"- 固定止损比例: {self.params.stop_loss:.1%}\n"
            f"- 最终止损价格: 取ATR止损价和比例止损价的较高者\n"
            f"- 均线下穿卖出: {'启用' if self.params.enable_crossover_sell else '禁用'}\n"
        )

        # 添加技术指标状态
        signal_details += (
            f"\n技术指标:\n"
            f"- ATR: {self.atr[0]:.3f}\n"
            f"- 当前价格: {self.dataclose[0]:.3f}\n"
        )

        signal_data["signal_details"] = signal_details

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
        signal_data["signal_description"] = "策略运行中"
        signal_data["signal_details"] += stop_loss_strategy

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

        # 检查趋势过滤器条件
        trend_ok = True
        trend_filter_msg = []

        # ADX过滤器
        if self.params.adx_period > 0 and self.params.adx_threshold > 0:
            adx_ok = self.adx[0] > self.params.adx_threshold
            if self.params.debug:
                logger.debug(
                    f"ADX过滤器检查:\n"
                    f"- ADX值: {self.adx[0]:.2f}\n"
                    f"- 阈值: {self.params.adx_threshold}\n"
                    f"- 结果: {'通过' if adx_ok else '未通过'}\n"
                    f"- 模式: {'买卖双向' if self.params.adx_filter_mode == FilterMode.BOTH else '仅买入' if self.params.adx_filter_mode == FilterMode.BUY_ONLY else '不过滤'}"
                )
            # 根据过滤模式决定是否应用过滤
            if not self.position:  # 买入时
                if self.params.adx_filter_mode in [
                    FilterMode.BOTH,
                    FilterMode.BUY_ONLY,
                ]:
                    trend_ok = trend_ok and adx_ok
            else:  # 卖出时
                if self.params.adx_filter_mode == FilterMode.BOTH:
                    trend_ok = trend_ok and adx_ok
            trend_filter_msg.append(
                f"ADX({self.params.adx_period}): {self.adx[0]:.2f} "
                f"{'>' if adx_ok else '<='} {self.params.adx_threshold}"
                f"{'[买入过滤]' if self.params.adx_filter_mode == FilterMode.BUY_ONLY else ''}"
            )

        # 布林带宽度过滤器
        if self.params.bbands_period > 0 and self.params.bbands_width_threshold > 0:
            if self.params.bbands_width_dynamic:
                threshold = (
                    self.bbands_width_median[0] * self.params.bbands_width_threshold
                )
                bbands_ok = self.bbands_width[0] > threshold
                if self.params.debug:
                    logger.debug(
                        f"布林带宽度过滤器检查(动态):\n"
                        f"- 当前宽度: {self.bbands_width[0]:.2f}%\n"
                        f"- 中位数: {self.bbands_width_median[0]:.2f}%\n"
                        f"- 阈值倍数: {self.params.bbands_width_threshold}\n"
                        f"- 最终阈值: {threshold:.2f}%\n"
                        f"- 结果: {'通过' if bbands_ok else '未通过'}\n"
                        f"- 模式: {'买卖双向' if self.params.bbands_filter_mode == FilterMode.BOTH else '仅买入' if self.params.bbands_filter_mode == FilterMode.BUY_ONLY else '不过滤'}"
                    )
                # 根据过滤模式决定是否应用过滤
                if not self.position:  # 买入时
                    if self.params.bbands_filter_mode in [
                        FilterMode.BOTH,
                        FilterMode.BUY_ONLY,
                    ]:
                        trend_ok = trend_ok and bbands_ok
                else:  # 卖出时
                    if self.params.bbands_filter_mode == FilterMode.BOTH:
                        trend_ok = trend_ok and bbands_ok
                trend_filter_msg.append(
                    f"布林带宽度({self.params.bbands_period}): {self.bbands_width[0]:.2f}% "
                    f"{'>' if bbands_ok else '<='} "
                    f"{threshold:.2f}% ({self.params.bbands_width_threshold}倍中位数)"
                    f"{'[买入过滤]' if self.params.bbands_filter_mode == FilterMode.BUY_ONLY else ''}"
                )
            else:
                # 使用固定阈值
                bbands_ok = self.bbands_width[0] > self.params.bbands_width_threshold
                trend_filter_msg.append(
                    f"布林带宽度({self.params.bbands_period}): {self.bbands_width[0]:.2f}% "
                    f"{'>' if bbands_ok else '<='} {self.params.bbands_width_threshold}%"
                )
                trend_ok = trend_ok and bbands_ok

        # 趋势过滤器
        if self.params.trend_filter_period > 0:
            trend_ok = self.dataclose[0] > self.trend_sma[0]
            if self.params.debug:
                logger.debug(
                    f"趋势过滤器检查:\n"
                    f"- 当前价格: {self.dataclose[0]:.3f}\n"
                    f"- MA{self.params.trend_filter_period}: {self.trend_sma[0]:.3f}\n"
                    f"- 结果: {'通过' if trend_ok else '未通过'}\n"
                    f"- 模式: {'买卖双向' if self.params.trend_filter_mode == FilterMode.BOTH else '仅买入' if self.params.trend_filter_mode == FilterMode.BUY_ONLY else '不过滤'}"
                )
            # 根据过滤模式决定是否应用过滤
            if not self.position:  # 买入时
                if self.params.trend_filter_mode in [
                    FilterMode.BOTH,
                    FilterMode.BUY_ONLY,
                ]:
                    trend_ok = trend_ok and trend_ok
            else:  # 卖出时
                if self.params.trend_filter_mode == FilterMode.BOTH:
                    trend_ok = trend_ok and trend_ok
            trend_filter_msg.append(
                f"趋势过滤器({self.params.trend_filter_period}): 价格({self.dataclose[0]:.3f}) "
                f"{'>' if trend_ok else '<='} MA{self.params.trend_filter_period}({self.trend_sma[0]:.3f})"
                f"{'[买入过滤]' if self.params.trend_filter_mode == FilterMode.BUY_ONLY else ''}"
            )

        # 处理信号滞后
        if self.params.signal_delay > 0:
            if not self.position:  # 买入时
                if self.params.signal_delay_mode in [
                    FilterMode.BOTH,
                    FilterMode.BUY_ONLY,
                ]:
                    if self.crossover > 0 or (  # 首次触发买入信号
                        self.pending_buy_signal > 0  # 已经在等待中
                        and self.sma_short[0] > self.sma_long[0]  # 维持买入条件
                    ):
                        self.pending_buy_signal += 1
                        if self.params.debug:
                            logger.debug(
                                f"买入信号等待中:\n"
                                f"- 已等待天数: {self.pending_buy_signal}\n"
                                f"- 需要等待天数: {self.params.signal_delay}\n"
                                f"- 触发条件: {'首次交叉' if self.crossover > 0 else '维持条件'}\n"
                                f"- 短期均线 {self.sma_short[0]:.3f} > 长期均线 {self.sma_long[0]:.3f}"
                            )
                        if self.pending_buy_signal >= self.params.signal_delay:
                            if self.params.debug:
                                logger.debug("买入信号确认，将执行买入操作")
                            # 先计算买入数量
                            available_cash = (
                                self.broker.getcash() * self.params.cash_ratio
                            )
                            size = int(available_cash / self.dataclose[0])

                            # 执行买入操作
                            if size > 0:  # 可以买入
                                if has_next_bar:  # 只有在有下一个交易日时才执行买入
                                    self.order = self.buy(
                                        size=size,
                                        exectype=bt.Order.Market,  # 使用市价单
                                        valid=next_date,  # 使用已经检查过的next_date
                                    )
                                    logger.info(
                                        f"买入信号 - 日期: {current_date}, "
                                        f"价格: {self.dataclose[0]:.2f}, "
                                        f"数量: {size}, "
                                        f"金额: {size * self.dataclose[0]:.2f}"
                                    )
                                else:
                                    logger.info(
                                        f"跳过买入信号 - 日期: {current_date} (已到最后交易日)"
                                    )
                            # 重置信号等待计数
                            self.pending_buy_signal = 0
                            return
                        else:
                            if self.params.debug:
                                logger.debug("继续等待买入信号")
                            return
                    else:  # 信号消失
                        if self.pending_buy_signal > 0:  # 之前有等待的信号
                            if self.params.debug:
                                logger.debug(
                                    f"买入信号在等待期间消失:\n"
                                    f"- 已等待天数: {self.pending_buy_signal}\n"
                                    f"- 当前条件: 短期均线 {self.sma_short[0]:.3f} {'>' if self.sma_short[0] > self.sma_long[0] else '<='} 长期均线 {self.sma_long[0]:.3f}"
                                )
                            self.pending_buy_signal = 0
                            return

            else:  # 卖出时
                if self.params.signal_delay_mode == FilterMode.BOTH:
                    if self.crossover < 0 or (  # 首次触发卖出信号
                        self.pending_sell_signal > 0  # 已经在等待中
                        and (
                            self.sma_short[0] < self.sma_long[0]  # 维持均线下穿条件
                            or self.dataclose[0] <= stop_price  # 或维持止损条件
                        )
                    ):
                        self.pending_sell_signal += 1
                        if self.params.debug:
                            logger.debug(
                                f"卖出信号等待中:\n"
                                f"- 已等待天数: {self.pending_sell_signal}\n"
                                f"- 需要等待天数: {self.params.signal_delay}\n"
                                f"- 触发条件: {'首次交叉' if self.crossover < 0 else '维持条件'}\n"
                                f"- 短期均线 {self.sma_short[0]:.3f} < 长期均线 {self.sma_long[0]:.3f}\n"
                                f"- 当前价格 {self.dataclose[0]:.3f} {'<=' if self.dataclose[0] <= stop_price else '>'} 止损价 {stop_price:.3f}"
                            )
                        if self.pending_sell_signal >= self.params.signal_delay:
                            if self.params.debug:
                                logger.debug("卖出信号确认，将执行卖出操作")
                            # 执行卖出操作
                            if has_next_bar:  # 只有在有下一个交易日时才执行卖出
                                self.order = self.sell(
                                    size=self.position.size,
                                    exectype=bt.Order.Market,  # 使用市价单
                                    valid=next_date,  # 使用已经检查过的next_date
                                )
                                self.order.sell_reason = sell_reason
                                logger.info(
                                    f"卖出信号 - 日期: {current_date}, "
                                    f"价格: {self.dataclose[0]:.2f}, "
                                    f"数量: {self.position.size}, "
                                    f"原因: {sell_reason}"
                                )
                            else:
                                logger.info(
                                    f"跳过卖出信号 - 日期: {current_date} (已到最后交易日)"
                                )
                            # 重置信号等待计数
                            self.pending_sell_signal = 0
                            return
                        else:
                            if self.params.debug:
                                logger.debug("继续等待卖出信号")
                            return
                    else:  # 信号消失
                        if self.pending_sell_signal > 0:  # 之前有等待的信号
                            if self.params.debug:
                                logger.debug(
                                    f"卖出信号在等待期间消失:\n"
                                    f"- 已等待天数: {self.pending_sell_signal}\n"
                                    f"- 当前条件: 短期均线 {self.sma_short[0]:.3f} {'<' if self.sma_short[0] < self.sma_long[0] else '>='} 长期均线 {self.sma_long[0]:.3f}\n"
                                    f"- 当前价格 {self.dataclose[0]:.3f} {'<=' if self.dataclose[0] <= stop_price else '>'} 止损价 {stop_price:.3f}"
                                )
                            self.pending_sell_signal = 0
                            return

        # 保存最后一天的信号数据，供回测结束后使用
        if next_date == current_date:
            # 检查是否有信号但被过滤
            has_signal = False
            if not self.position and self.crossover > 0:  # 有买入信号
                has_signal = True
                if not trend_ok:
                    signal_data["signal_details"] += (
                        f"\n过滤器阻止买入:\n" f"- {', '.join(trend_filter_msg)}"
                    )
            elif self.position and self.crossover < 0:  # 有卖出信号
                has_signal = True
                if not trend_ok:
                    signal_data["signal_details"] += (
                        f"\n过滤器阻止均线下穿卖出:\n"
                        f"- {', '.join(trend_filter_msg)}"
                    )

            # 如果有过滤器启用但没有信号，也显示过滤器状态
            if (not has_signal) and (
                (self.params.adx_period > 0 and self.params.adx_threshold > 0)
                or (
                    self.params.bbands_period > 0
                    and self.params.bbands_width_threshold > 0
                )
            ):
                signal_data["signal_details"] += (
                    f"\n当前过滤器状态:\n" f"- {', '.join(trend_filter_msg)}"
                )

            self.last_signal_data = signal_data
            self.last_holding_data = holding_data

        # 执行交易逻辑
        if self.order:
            return

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
                    if self.params.signal_delay_mode in [
                        FilterMode.BOTH,
                        FilterMode.BUY_ONLY,
                    ]:
                        if self.pending_buy_signal < self.params.signal_delay:
                            return  # 继续等待
                        # 达到等待天数后，重置计数并继续执行
                        self.pending_buy_signal = 0

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
                    if has_next_bar:  # 只有在有下一个交易日时才执行买入
                        self.order = self.buy(
                            size=size,
                            exectype=bt.Order.Market,  # 使用市价单
                            valid=next_date,  # 使用已经检查过的next_date
                        )
                        logger.info(
                            f"买入信号 - 日期: {current_date}, "
                            f"价格: {self.dataclose[0]:.2f}, "
                            f"数量: {size}, "
                            f"金额: {size * self.dataclose[0]:.2f}"
                        )
                    else:
                        logger.info(
                            f"跳过买入信号 - 日期: {current_date} (已到最后交易日)"
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
                # 更新信号数据
                signal_data.update(
                    {
                        "signal_type": "卖出",
                        "signal_details": (
                            f"卖出信号指标:\n"
                            f"- 短期均线: MA{self.params.short_period} = {self.sma_short[0]:.3f}\n"
                            f"- 长期均线: MA{self.params.long_period} = {self.sma_long[0]:.3f}\n"
                            f"- 均线差值: {(self.sma_short[0] - self.sma_long[0]):.3f}\n"
                            f"- ATR指标: {self.atr[0]:.3f}\n"
                            f"- 卖出价格: {self.dataclose[0]:.3f}\n"
                            f"- 卖出原因: {sell_reason}\n"
                            f"{'(已到最后交易日，无法执行)' if not has_next_bar else ''}"
                        ),
                        # 移除 position_details，因为持仓信息会在后面添加
                        "signal_description": f"{sell_reason}，产生卖出信号",
                    }
                )
                self.last_signal_data = signal_data
                self.last_holding_data = holding_data

                # 处理信号滞后
                if self.params.signal_delay > 0:
                    if self.params.signal_delay_mode == FilterMode.BOTH:
                        if self.pending_sell_signal < self.params.signal_delay:
                            return  # 继续等待
                        # 达到等待天数后，重置计数并继续执行
                        self.pending_sell_signal = 0

                # 执行卖出操作
                if has_next_bar:  # 只有在有下一个交易日时才执行卖出
                    self.order = self.sell(
                        size=self.position.size,
                        exectype=bt.Order.Market,  # 使用市价单
                        valid=next_date,  # 使用已经检查过的next_date
                    )
                    self.order.sell_reason = sell_reason
                    logger.info(
                        f"卖出信号 - 日期: {current_date}, "
                        f"价格: {self.dataclose[0]:.2f}, "
                        f"数量: {self.position.size}, "
                        f"原因: {sell_reason}"
                    )
                else:
                    logger.info(f"跳过卖出信号 - 日期: {current_date} (已到最后交易日)")

                # 重置信号等待计数
                self.pending_sell_signal = 0

        # 如果有持仓，添加持仓信息
        if self.position and self.position.size > 0:
            current_price = self.dataclose[0]
            position_size = self.position.size
            buy_value = self.buyprice * position_size  # 买入总金额
            current_value = position_size * current_price  # 当前市值
            profit_rate = (current_value / buy_value - 1) * 100  # 收益率(%)

            # 更新 signal_data 中的持仓状态
            signal_data["position_details"] = (
                f"当前持仓\n"
                f"- 买入时间：{self.buy_date.strftime('%Y-%m-%d')}\n"
                f"- 买入价格：{self.buyprice:.3f}\n"
                f"- 买入数量：{position_size:,d}\n"
                f"- 买入金额：{buy_value:,.2f}\n"
                f"- 当前价格：{current_price:.3f}\n"
                f"- 当前市值：{current_value:,.2f}\n"
                f"- 当前收益：{profit_rate:+.2f}%\n"  # 添加+号显示正负
                f"- 最高价格：{self.highest_price:.3f}"
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

                # 根据实际触发的信号系统记录指标条件
                indicator_conditions = []
                if self.params.short_period > 0 and self.params.long_period > 0:
                    indicator_conditions.append(
                        f"MA{self.params.short_period}({self.sma_short[-1]:.3f}) > "
                        f"MA{self.params.long_period}({self.sma_long[-1]:.3f})"
                    )
                if (
                    self.params.macd_fast_period > 0
                    and self.params.macd_slow_period > 0
                ):
                    indicator_conditions.append(
                        f"MACD({self.macd.macd[-1]:.3f}) > "
                        f"Signal({self.macd.signal[-1]:.3f})"
                    )

                # 记录买入交易
                self.trades.append(
                    {
                        "类型": "买入",
                        "信号日期": self.data.datetime.date(-1),
                        "执行日期": self.data.datetime.date(0),
                        "价格": order.executed.price,
                        "数量": order.executed.size,
                        "消耗资金": order.executed.value,
                        "手续费": order.executed.comm,
                        "收益": 0.0,
                        "剩余资金": self.broker.getcash(),
                        "指标条件": (
                            " 且 ".join(indicator_conditions)
                            if indicator_conditions
                            else "未知条件"
                        ),
                    }
                )

                logger.info(
                    f"买入执行完成 - 信号日期: {self.data.datetime.date(-1)}, "
                    f"执行日期: {self.data.datetime.date(0)}, "
                    f"开盘价: {order.executed.price:.2f}, "
                    f"数量: {order.executed.size}, "
                    f"总金额: {order.executed.value:.2f}, "
                    f"手续费: {order.executed.comm:.2f}"
                )
            else:  # 卖出时
                # 计算卖出收益（需要考虑买入和卖出的总手续费）
                sell_value = abs(order.executed.price * order.executed.size)
                buy_value = abs(self.buyprice * order.executed.size)
                total_commission = abs(
                    order.executed.comm + self.buycomm
                )  # 买入和卖出的总手续费
                profit = round(
                    sell_value - buy_value - total_commission, 4
                )  # 修正收益计算公式：卖出价值 - 买入价值 - 总手续费

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
                elif getattr(order, "sell_reason", "") == "MACD下穿":
                    indicator_condition = (
                        f"MACD({self.macd.macd[0]:.3f}) < "
                        f"Signal({self.macd.signal[0]:.3f})"
                    )
                else:
                    indicator_condition = "未知条件"

                # 记录卖出交易（统一使用信号日期和执行日期）
                self.trades.append(
                    {
                        "类型": "卖出",
                        "信号日期": self.data.datetime.date(-1),
                        "执行日期": self.data.datetime.date(0),
                        "价格": order.executed.price,
                        "数量": -order.executed.size,  # 使用负数表示卖出
                        "消耗资金": sell_value,
                        "手续费": order.executed.comm,  # 只记录卖出时的手续费
                        "收益": profit,
                        "剩余资金": self.broker.getcash(),
                        "指标条件": indicator_condition,
                    }
                )

                logger.info(
                    f"卖出执行完成 - 信号日期: {self.data.datetime.date(-1)}, "
                    f"执行日期: {self.data.datetime.date(0)}, "
                    f"价格: {order.executed.price:.2f}, "
                    f"数量: {order.executed.size}, "
                    f"收益: {profit:.2f}, "
                    f"原因: {getattr(order, 'sell_reason', '未知原因')}"
                )

                # 重置相关变量
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
            "信号日期",  # 修改为使用信号日期
            "执行日期",  # 添加执行日期
            "价格",
            "数量",
            "消耗资金",
            "手续费",
            "收益",
            "剩余资金",
            "总资产",  # 添加总资产列
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
                        str(trade["信号日期"]),  # 使用信号日期
                        str(trade["执行日期"]),  # 使用执行日期
                        f"{trade['价格']:.3f}",
                        f"{trade['数量']:.0f}",
                        f"{trade['消耗资金']:.2f}",
                        f"{trade['手续费']:.2f}",
                        f"{trade['收益']:.2f}",
                        f"{trade['剩余资金']:.2f}",
                        f"{total_value:.2f}",  # 添加总资产
                        trade["指标条件"],
                    ]
                )
            else:  # 卖出
                # 卖出时资产总值就是剩余资金
                total_value = trade["剩余资金"]
                table_data.append(
                    [
                        trade["类型"],
                        str(trade["信号日期"]),  # 使用信号日期
                        str(trade["执行日期"]),  # 使用执行日期
                        f"{trade['价格']:.3f}",
                        f"{trade['数量']:.0f}",
                        f"{trade['消耗资金']:.2f}",
                        f"{trade['手续费']:.2f}",
                        f"{trade['收益']:.2f}",
                        f"{trade['剩余资金']:.2f}",
                        f"{total_value:.2f}",  # 添加总资产
                        trade["指标条件"],
                    ]
                )

        return "\n交易记录:\n" + tabulate(
            table_data,
            headers=headers,
            tablefmt="grid",
            colalign=(
                "left",  # 类型
                "center",  # 信号日期
                "center",  # 执行日期
                "right",  # 价格
                "right",  # 数量
                "right",  # 消耗资金
                "right",  # 手续费
                "right",  # 收益
                "right",  # 剩余资金
                "right",  # 总资产
                "left",  # 指标条件
            ),
        )


def get_etf_data(symbol, start_date=None, end_date=None):
    """获取ETF数据，根据需要更新数据"""
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

    csv_file = os.path.join(downloads_dir, f"etf_data_{symbol}.csv")

    try:
        # 标准化日期格式
        if start_date:
            start_date = pd.to_datetime(start_date).normalize()
        if end_date:
            # 确保end_date是当天的23:59:59
            end_date = (
                pd.to_datetime(end_date)
                .normalize()
                .replace(hour=23, minute=59, second=59)
            )

        # 检查是否需要更新数据
        need_update = True
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            if not df.empty:
                latest_date = pd.to_datetime(df["日期"]).max()
                today = pd.Timestamp.now().normalize()

                # 如果最新数据不是今天或昨天的，就需要更新
                if latest_date.normalize() < (today - pd.Timedelta(days=1)):
                    logger.info(
                        f"数据不是最新的 (最新: {latest_date.date()}, 当前: {today.date()}), 需要更新"
                    )
                else:
                    need_update = False
                    logger.debug(
                        f"数据是最新的 (最新: {latest_date.date()}, 当前: {today.date()})"
                    )

        if need_update:
            # 获取新数据
            logger.info("开始获取新数据...")
            new_data = get_and_save_etf_data(
                symbol,
                start_date=start_date,
                end_date=end_date,
            )
            if new_data is not None:
                # 重新读取并检查数据
                df = _get_and_check_data(symbol, start_date, end_date)
                if df is None:
                    logger.error("更新数据后读取失败")
                    return None
            else:
                logger.error("获取新数据失败")
                return None
        else:
            # 使用现有数据
            df = _get_and_check_data(symbol, start_date, end_date)

        return df

    except Exception as e:
        logger.error(f"处理 {symbol} 数据出错: {e}")
        return None


def _get_and_check_data(symbol, start_date=None, end_date=None):
    """内部函数：获取并检查数据"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    downloads_dir = os.path.join(script_dir, "downloads")
    csv_file = os.path.join(downloads_dir, f"etf_data_{symbol}.csv")

    if not os.path.exists(csv_file):
        return None

    # 读取数据文件
    df = pd.read_csv(csv_file, encoding="utf-8")
    if df.empty:
        return None

    # 转换日期并排序
    df["日期"] = pd.to_datetime(df["日期"]).dt.normalize()
    df = df.sort_values("日期")  # 确保数据按日期排序
    df = df.set_index("日期")

    # 准备回测所需的数据列
    df["价格"] = df["收盘"]
    df["数量"] = df["成交量"]
    df["消耗资金"] = 0
    df["手续费"] = 0
    df["收益"] = 0

    # 返回所有需要的列
    return df[
        [
            "开盘",
            "最高",
            "最低",
            "收盘",
            "成交量",
            "价格",
            "数量",
            "消耗资金",
            "手续费",
            "收益",
        ]
    ]


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
    bbands_width_dynamic=False,  # 新增参数
    adx_filter_mode=FilterMode.BOTH,  # 新增过滤器模式参数
    bbands_filter_mode=FilterMode.BOTH,  # 新增过滤器模式参数
    trend_filter_mode=FilterMode.BOTH,  # 新增过滤器模式参数
    signal_delay_mode=FilterMode.BOTH,  # 新增过滤器模式参数
    macd_fast_period=0,  # 添加MACD快线周期参数
    macd_slow_period=0,  # 添加MACD慢线周期参数
    macd_signal_period=9,  # 添加MACD信号线周期参数
    macd_mode=FilterMode.BOTH,  # 添加MACD信号模式参数
    ma_mode=FilterMode.BOTH,  # 添加均线信号模式参数
):
    cerebro = bt.Cerebro()

    # 日期过滤，确保无时区
    start_date = pd.to_datetime(start_date).tz_localize(None)
    end_date = pd.to_datetime(end_date).tz_localize(None)

    # 获取完整的历史数据，传入start_date以确保获取足够的历史数据
    data = get_etf_data(symbol, start_date=start_date, end_date=end_date)
    if data is None:
        return None

    # 确保数据包含回测开始日期
    if data.index[0] > start_date:
        logger.error(
            f"数据的开始日期({data.index[0].date()})晚于回测开始日期({start_date.date()})，无法进行回测"
        )
        return None

    # 创建 PandasData feed，使用完整数据集，通过 fromdate 和 todate 控制回测区间
    feed = bt.feeds.PandasData(
        dataname=data,
        datetime=None,  # 使用索引作为日期
        open="开盘",  # 使用列名而不是Series
        high="最高",
        low="最低",
        close="价格",  # 使用我们设置的"价格"列
        volume="数量",  # 使用我们设置的"数量"列
        openinterest=-1,
        fromdate=start_date,
        todate=end_date,
        plot=False,
    )

    # 添加数据到cerebro
    cerebro.adddata(feed)

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0001)

    cerebro.addstrategy(
        ETFStrategy,
        short_period=short_period,
        long_period=long_period,
        stop_loss=stop_loss,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        use_atr_stop=use_atr_stop,
        cash_ratio=cash_ratio,
        enable_crossover_sell=enable_crossover_sell,
        debug=debug,
        symbol=symbol,
        notifiers=notifiers or {},
        start_date=start_date,  # 添加开始日期参数
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        bbands_period=bbands_period,
        bbands_width_threshold=bbands_width_threshold,
        bbands_width_dynamic=bbands_width_dynamic,  # 新增参数
        trend_filter_period=trend_filter_period,  # 传递趋势过滤器参数
        signal_delay=signal_delay,  # 传递信号滞后参数
        adx_filter_mode=adx_filter_mode,  # 传递过滤器模式
        bbands_filter_mode=bbands_filter_mode,  # 传递过滤器模式
        trend_filter_mode=trend_filter_mode,  # 传递过滤器模式
        signal_delay_mode=signal_delay_mode,  # 传递过滤器模式
        macd_fast_period=macd_fast_period,  # 传递MACD快线周期参数
        macd_slow_period=macd_slow_period,  # 传递MACD慢线周期参数
        macd_signal_period=macd_signal_period,  # 传递MACD信号线周期参数
        macd_mode=macd_mode,  # 传递MACD信号模式参数
        ma_mode=ma_mode,  # 传递均线信号模式参数
    )

    # 添加更多分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")  # 添加交易分析器
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="time_return"
    )  # 添加时间收益分析器
    cerebro.addanalyzer(bt.analyzers.VWR, _name="vwr")  # 添加VWR分析器
    cerebro.addanalyzer(bt.analyzers.SQN, _name="sqn")  # 添加SQN分析器
    cerebro.addanalyzer(
        bt.analyzers.TimeDrawDown, _name="time_drawdown"
    )  # 添加时间回撤分析器

    results = cerebro.run()
    strat = results[0]

    # 获取分析器数据
    analysis = strat.analyzers.returns.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    vwr = strat.analyzers.vwr.get_analysis()
    sqn = strat.analyzers.sqn.get_analysis()
    time_drawdown = strat.analyzers.time_drawdown.get_analysis()

    # 计算各项指标
    total_trades = trades.total.closed if hasattr(trades, "total") else 0
    won_trades = trades.won.total if hasattr(trades, "won") else 0
    lost_trades = trades.lost.total if hasattr(trades, "lost") else 0
    win_rate = won_trades / total_trades * 100 if total_trades > 0 else 0

    # 计算盈亏比
    avg_won = trades.won.pnl.average if hasattr(trades, "won") else 0
    avg_lost = abs(trades.lost.pnl.average) if hasattr(trades, "lost") else 0
    profit_loss_ratio = avg_won / avg_lost if avg_lost != 0 else 0

    # 计算夏普比率
    sharpe_ratio = sharpe.get("sharperatio", 0.0)

    # 计算最大回撤
    max_drawdown = drawdown.get("max", {}).get("drawdown", 0.0) * 100

    # 计算当前回撤
    current_drawdown = time_drawdown.get("drawdown", 0.0) * 100

    # 计算年化收益率
    total_days = (end_date - start_date).days
    total_return = strat.broker.getvalue() / initial_cash - 1
    annual_return = (
        ((1 + total_return) ** (365 / total_days) - 1) * 100 if total_days > 0 else 0
    )

    # 计算Calmar比率
    calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0

    # 获取SQN值
    sqn = sqn.get("sqn", 0.0)

    # 获取VWR值
    vwr = vwr.get("vwr", 0.0)

    # 计算最新净值
    nav = strat.broker.getvalue() / initial_cash

    # 生成交易记录表格
    trade_table = strat.get_trade_table()
    if trade_table != "没有发生交易":
        logger.info("\n交易记录:")
        logger.info(f"\n{trade_table}")

    # 如果有通知器，发送通知；如果没有，打印到控制台
    if hasattr(strat, "last_signal_data"):
        # 添加性能数据到信号详情
        performance_text = "\n\n策略表现:\n"
        performance_text += f"- 建仓日期: {start_date.date()}\n"  # 添加建仓日期
        performance_text += f"- 夏普比率: {sharpe['sharperatio']:.4f}\n"
        performance_text += f"- 年化收益率: {analysis['rnorm100']:.2f}%\n"
        performance_text += f"- 最大回撤: {drawdown['max']['drawdown']:.2f}%\n"
        # 添加新增指标
        performance_text += f"- 胜率: {win_rate:.2f}%\n"
        performance_text += f"- 最新净值: {nav:.4f}\n"
        performance_text += f"- 盈亏比: {profit_loss_ratio:.2f}\n"
        performance_text += f"- 当前回撤: {current_drawdown:.2f}%\n"
        performance_text += f"- Calmar比率: {calmar_ratio:.4f}\n"
        performance_text += f"- SQN: {sqn:.4f}\n"
        performance_text += f"- VWR: {vwr:.4f}\n"
        performance_text += f"- 总交易次数: {total_trades}\n"
        performance_text += f"- 运行天数: {total_days}"

        strat.last_signal_data["signal_details"] += performance_text
        strat.last_signal_data["trade_table"] = trade_table

        # 如果有通知器，发送通知
        if notifiers:
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
                        f"通知发送结果 - 通知器: {notifier_name}, "
                        f"返回值: {response}"
                    )
        else:
            # 如果没有通知器，打印到控制台
            logger.info("\n策略运行结果:")
            logger.info(f"ETF代码: {symbol}")
            logger.info(f"回测区间: {start_date} 到 {end_date}")

            # 打印当前信号
            logger.info("\n当前信号:")
            logger.info(f"- 信号类型: {strat.last_signal_data['signal_type']}")
            logger.info(f"- 当前价格: {strat.last_signal_data['price']:.3f}")
            logger.info(f"- 持仓状态: {strat.last_signal_data['position_details']}")

            # 打印信号详情（包括指标数据）
            logger.info("\n信号详情:")
            signal_details = strat.last_signal_data["signal_details"].split("\n")
            for line in signal_details:
                if line.strip():
                    logger.info(f"{line}")

            # 打印基础策略参数
            logger.info("\n基础策略参数:")
            logger.info(f"- 短期均线: MA{short_period}")
            logger.info(f"- 长期均线: MA{long_period}")
            logger.info(f"- 止损比例: {stop_loss:.1%}")
            logger.info(f"- ATR周期: {atr_period}")
            logger.info(f"- ATR倍数: {atr_multiplier}")
            logger.info(f"- 使用ATR止损: {'是' if use_atr_stop else '否'}")
            logger.info(f"- 资金使用比例: {cash_ratio:.1%}")
            logger.info(
                f"- 启用均线下穿卖出: {'是' if enable_crossover_sell else '否'}"
            )

            # 打印过滤器参数
            logger.info("\n过滤器设置:")
            if trend_filter_period > 0:
                logger.info("\n1. 趋势过滤器:")
                logger.info(f"- 均线周期: MA{trend_filter_period}")
                logger.info(
                    f"- 过滤模式: {'买卖双向过滤' if trend_filter_mode == FilterMode.BOTH else '仅买入过滤' if trend_filter_mode == FilterMode.BUY_ONLY else '不过滤'}"
                )
                logger.info(f"- 过滤条件: 价格必须高于MA{trend_filter_period}")

            if signal_delay > 0:
                logger.info("\n2. 信号滞后:")
                logger.info(f"- 滞后天数: {signal_delay}天")
                logger.info(
                    f"- 滞后模式: {'买卖双向滞后' if signal_delay_mode == FilterMode.BOTH else '仅买入滞后' if signal_delay_mode == FilterMode.BUY_ONLY else '不滞后'}"
                )
                logger.info(f"- 滞后条件: 信号必须持续{signal_delay}天才执行交易")

            if adx_period > 0:
                logger.info("\n3. ADX过滤器:")
                logger.info(f"- 指标周期: {adx_period}天")
                logger.info(f"- 指标阈值: >{adx_threshold}")
                logger.info(
                    f"- 过滤模式: {'买卖双向过滤' if adx_filter_mode == FilterMode.BOTH else '仅买入过滤' if adx_filter_mode == FilterMode.BUY_ONLY else '不过滤'}"
                )
                logger.info(f"- 过滤条件: ADX必须大于{adx_threshold}")

            if bbands_period > 0:
                logger.info("\n4. 布林带过滤器:")
                logger.info(f"- 指标周期: {bbands_period}天")
                if bbands_width_dynamic:
                    logger.info(f"- 宽度阈值: >{bbands_width_threshold}倍中位数")
                    logger.info("- 阈值类型: 动态（相对中位数）")
                else:
                    logger.info(f"- 宽度阈值: >{bbands_width_threshold}%")
                    logger.info("- 阈值类型: 固定百分比")
                logger.info(
                    f"- 过滤模式: {'买卖双向过滤' if bbands_filter_mode == FilterMode.BOTH else '仅买入过滤' if bbands_filter_mode == FilterMode.BUY_ONLY else '不过滤'}"
                )
                logger.info("- 过滤条件: 布林带宽度必须大于阈值")

            if (
                trend_filter_period == 0
                and signal_delay == 0
                and adx_period == 0
                and bbands_period == 0
            ):
                logger.info("- 未启用任何过滤器")

            # 打印策略表现
            logger.info("\n策略表现:")
            logger.info(f"- 夏普比率: {sharpe['sharperatio']:.4f}")
            logger.info(f"- 年化收益率: {analysis['rnorm100']:.2f}%")
            logger.info(f"- 最大回撤: {drawdown['max']['drawdown']:.2f}%")
            logger.info(f"- 初始资金: {initial_cash:.2f}")
            logger.info(f"- 最终资金: {cerebro.broker.getvalue():.2f}")
            logger.info(
                f"- 总收益率: {(cerebro.broker.getvalue() / initial_cash - 1) * 100:.2f}%"
            )

            # 如果有持仓数据，也打印出来
            if strat.last_holding_data:
                logger.info("\n当前持仓:")
                logger.info(f"- 买入时间: {strat.last_holding_data['买入时间']}")
                logger.info(f"- 买入价格: {strat.last_holding_data['买入价格']:.3f}")
                logger.info(f"- 当前收益: {strat.last_holding_data['当前收益']:.2%}")
                logger.info(f"- 最高价格: {strat.last_holding_data['最高价格']:.3f}")
                logger.info(f"- 当前价格: {strat.last_holding_data['当前价格']:.3f}")

    # 构建结果字典
    result = {
        # 双均线系统参数
        "短期均线": short_period,
        "长期均线": long_period,
        # MACD系统参数
        "MACD快线": macd_fast_period if macd_fast_period > 0 else "未启用",
        "MACD慢线": macd_slow_period if macd_slow_period > 0 else "未启用",
        "MACD信号线": macd_signal_period if macd_signal_period > 0 else "未启用",
        # 止损参数
        "止损比例": stop_loss,
        "ATR周期": atr_period,
        "ATR倍数": atr_multiplier,
        "使用ATR止损": use_atr_stop,
        # 资金管理
        "资金使用比例": cash_ratio,
        # 性能指标
        "最终资金": strat.broker.getvalue(),
        "年化收益率": annual_return,
        "夏普比率": sharpe_ratio,
        "最大回撤": max_drawdown,
        "胜率": win_rate,
        "最新净值": nav,
        "盈亏比": profit_loss_ratio,
        "当前回撤": current_drawdown,
        "Calmar比率": calmar_ratio,
        "SQN": sqn,
        "VWR": vwr,
        "总交易次数": total_trades,
        "运行天数": total_days,
    }

    return result


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
    trend_filter_period=0,
    signal_delay=0,
    adx_period=0,
    adx_threshold=0,
    bbands_period=0,
    bbands_width_threshold=0.0,
    bbands_width_dynamic=False,
    adx_filter_mode=FilterMode.BOTH,
    bbands_filter_mode=FilterMode.BOTH,
    trend_filter_mode=FilterMode.BOTH,
    signal_delay_mode=FilterMode.BOTH,
    macd_fast_period_range=[0],
    macd_slow_period_range=[0],
    macd_signal_period_range=[9],
    macd_mode=FilterMode.BOTH,
    ma_mode=FilterMode.BOTH,
):
    """优化策略参数"""
    results = []

    # 检查信号系统
    has_ma_system = any(p > 0 for p in short_period_range) and any(
        p > 0 for p in long_period_range
    )
    has_macd_system = any(p > 0 for p in macd_fast_period_range)

    if not has_ma_system and not has_macd_system:
        logger.error("必须至少启用一个信号系统(双均线或MACD)")
        return None

    # 根据启用的系统准备参数组合
    param_ranges = []
    param_names = []

    # 均线系统参数（保留原有参数，但如果未启用则使用[0]）
    param_ranges.extend(
        [
            short_period_range if has_ma_system else [0],
            long_period_range if has_ma_system else [0],
        ]
    )
    param_names.extend(["short_period", "long_period"])

    # MACD系统参数（如果未启用则使用[0]）
    param_ranges.extend(
        [
            macd_fast_period_range if has_macd_system else [0],
            macd_slow_period_range if has_macd_system else [0],
            macd_signal_period_range if has_macd_system else [0],
        ]
    )
    param_names.extend(
        [
            "macd_fast_period",
            "macd_slow_period",
            "macd_signal_period",
        ]
    )

    # 其他通用参数
    param_ranges.extend(
        [
            stop_loss_range,
            atr_multiplier_range,
            cash_ratio_range,
        ]
    )
    param_names.extend(
        [
            "stop_loss",
            "atr_multiplier",
            "cash_ratio",
        ]
    )

    # 使用zip将参数名和值组合在一起进行迭代
    for params in itertools.product(*param_ranges):
        # 将参数名和值组合成字典
        param_dict = dict(zip(param_names, params))

        # 参数验证
        if has_ma_system and param_dict["short_period"] >= param_dict["long_period"]:
            continue

        if has_macd_system:
            # 验证MACD参数
            if param_dict["macd_fast_period"] >= param_dict["macd_slow_period"]:
                continue
            if param_dict["macd_signal_period"] <= 0:
                continue

        # 设置ATR周期
        current_atr_period = atr_period
        if current_atr_period is None:
            if has_ma_system:
                current_atr_period = param_dict["long_period"]
            elif has_macd_system:
                current_atr_period = param_dict["macd_slow_period"]

        # 调试日志
        if debug:
            log_msg = []
            if has_ma_system:
                log_msg.append(
                    f"MA系统 - 短期: {param_dict['short_period']}, "
                    f"长期: {param_dict['long_period']}"
                )
            if has_macd_system:
                log_msg.append(
                    f"MACD系统 - 快线: {param_dict['macd_fast_period']}, "
                    f"慢线: {param_dict['macd_slow_period']}, "
                    f"信号线: {param_dict['macd_signal_period']}"
                )
            log_msg.append(f"ATR周期: {current_atr_period}")
            logger.debug(f"当前参数组合 - " + ", ".join(log_msg))

        # 运行回测
        result = run_backtest(
            symbol,
            start_date,
            end_date,
            initial_cash,
            param_dict["short_period"],
            param_dict["long_period"],
            param_dict["stop_loss"],
            current_atr_period,
            param_dict["atr_multiplier"],
            use_atr_stop,
            param_dict["cash_ratio"],
            enable_crossover_sell,
            debug,
            notifiers=notifiers,
            trend_filter_period=trend_filter_period,
            signal_delay=signal_delay,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            bbands_period=bbands_period,
            bbands_width_threshold=bbands_width_threshold,
            bbands_width_dynamic=bbands_width_dynamic,
            adx_filter_mode=adx_filter_mode,
            bbands_filter_mode=bbands_filter_mode,
            trend_filter_mode=trend_filter_mode,
            signal_delay_mode=signal_delay_mode,
            macd_fast_period=param_dict["macd_fast_period"],
            macd_slow_period=param_dict["macd_slow_period"],
            macd_signal_period=param_dict["macd_signal_period"],
            macd_mode=macd_mode,
            ma_mode=(
                ma_mode if has_ma_system else FilterMode.NONE
            ),  # 根据是否启用MA系统设置模式
        )
        if result:
            results.append(result)

    return results


# 在文件开头添加这个新函数
def parse_range(range_str):
    if not range_str:
        return [0]
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
        default=None,
        help="短期均线范围，用逗号分隔或短横线表示范围，如：10,20,30 或 10-30",
    )
    parser.add_argument(
        "--long-period-range",
        type=str,
        default=None,
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
        type=str,
        default="0.0",
        help="布林带宽度阈值。可以是绝对值（如2.5表示2.5%）或中位数倍数（如1.2x表示1.2倍中位数）。0表示不使用。",
    )

    parser.add_argument(
        "--adx-filter-mode",
        type=str,
        choices=["both", "buy_only", "none"],
        default="both",
        help="ADX过滤器模式：both(买卖都过滤)，buy_only(只过滤买入)，none(不过滤)",
    )
    parser.add_argument(
        "--bbands-filter-mode",
        type=str,
        choices=["both", "buy_only", "none"],
        default="both",
        help="布林带过滤器模式：both(买卖都过滤)，buy_only(只过滤买入)，none(不过滤)",
    )

    parser.add_argument(
        "--trend-filter-mode",
        type=str,
        choices=["both", "buy_only", "none"],
        default="both",
        help="趋势过滤器模式：both(买卖都过滤)，buy_only(只过滤买入)，none(不过滤)",
    )
    parser.add_argument(
        "--signal-delay-mode",
        type=str,
        choices=["both", "buy_only", "none"],
        default="both",
        help="信号滞后模式：both(买卖都滞后)，buy_only(只买入滞后)，none(不滞后)",
    )

    parser.add_argument(
        "--macd-fast-period",  # 修改参数名
        type=str,
        default=None,
        help="MACD快线周期，如'12'或'10-14'，0表示不使用MACD",
    )
    parser.add_argument(
        "--macd-slow-period",  # 修改参数名
        type=str,
        default=None,
        help="MACD慢线周期，如'26'或'20-30'，需要大于快线周期",
    )
    parser.add_argument(
        "--macd-signal-period",  # 修改参数名
        type=str,
        default=None,
        help="MACD信号线周期，如'9'或'8-10'，默认9",
    )

    parser.add_argument(
        "--macd-mode",
        type=str,
        choices=["both", "buy_only", "none"],
        default="both",
        help="MACD信号模式：both(买卖都触发)，buy_only(只买入)，none(不使用)",
    )
    parser.add_argument(
        "--ma-mode",
        type=str,
        choices=["both", "buy_only", "none"],
        default="both",
        help="均线信号模式：both(买卖都触发)，buy_only(只买入)，none(不使用)",
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

    # 处理布林带宽度阈值
    bbands_width_dynamic = False
    bbands_width_threshold = 0.0
    if args.bbands_width_threshold.endswith("x"):
        bbands_width_dynamic = True
        try:
            bbands_width_threshold = float(args.bbands_width_threshold[:-1])
        except ValueError:
            logger.error("布林带宽度阈值格式错误，应为数字+x，如1.2x")
            sys.exit(1)
    else:
        try:
            bbands_width_threshold = float(args.bbands_width_threshold)
        except ValueError:
            logger.error("布林带宽度阈值必须是数字")
            sys.exit(1)

    # 转换过滤器模式参数
    adx_filter_mode = FilterMode(args.adx_filter_mode)
    bbands_filter_mode = FilterMode(args.bbands_filter_mode)

    # 转换过滤器模式参数
    trend_filter_mode = FilterMode(args.trend_filter_mode)
    signal_delay_mode = FilterMode(args.signal_delay_mode)

    # 修改参数验证部分
    # 解析MACD参数范围
    macd_fast_period_range = [int(x) for x in parse_range(args.macd_fast_period)]
    macd_slow_period_range = [int(x) for x in parse_range(args.macd_slow_period)]
    macd_signal_period_range = [int(x) for x in parse_range(args.macd_signal_period)]

    # 转换信号模式
    macd_mode = FilterMode[args.macd_mode.upper()]
    ma_mode = FilterMode[args.ma_mode.upper()]

    # 检查信号系统
    has_ma_system = bool(args.short_period_range)  # 修改这里，只检查是否指定了短期均线
    has_macd_system = bool(
        macd_fast_period_range[0] > 0
        and macd_slow_period_range[0] > 0
        and macd_signal_period_range[0] > 0
    )

    if not has_ma_system and not has_macd_system:
        logger.error("必须至少启用一个信号系统(双均线或MACD)")
        sys.exit(1)

    # 如果没有启用相应的系统，将相关参数设置为0
    if not has_ma_system:
        short_period_range = [0]
        long_period_range = [0]
        ma_mode = FilterMode.NONE
    if not has_macd_system:
        macd_fast_period_range = [0]
        macd_slow_period_range = [0]
        macd_signal_period_range = [0]
        macd_mode = FilterMode.NONE
    logger.info(
        f"启用了信号系统: {has_ma_system} {has_macd_system}, 参数是: {short_period_range} {long_period_range} {macd_fast_period_range} {macd_slow_period_range} {macd_signal_period_range}"
    )
    # 运行回测
    results = optimize_parameters(
        symbol,
        args.start_date,
        args.end_date,
        args.initial_cash,
        short_period_range,  # 直接使用修改后的值
        long_period_range,  # 直接使用修改后的值
        stop_loss_range,
        args.atr_period,
        atr_multiplier_range,
        args.use_atr_stop,
        cash_ratio_range,
        not args.disable_crossover_sell,
        args.debug,
        notifiers=notifiers,
        trend_filter_period=args.trend_filter_period,
        signal_delay=args.signal_delay,
        adx_period=args.adx_period,
        adx_threshold=args.adx_threshold,
        bbands_period=args.bbands_period,
        bbands_width_threshold=bbands_width_threshold,
        bbands_width_dynamic=bbands_width_dynamic,
        adx_filter_mode=adx_filter_mode,
        bbands_filter_mode=bbands_filter_mode,
        trend_filter_mode=trend_filter_mode,
        signal_delay_mode=signal_delay_mode,
        macd_fast_period_range=macd_fast_period_range,
        macd_slow_period_range=macd_slow_period_range,
        macd_signal_period_range=macd_signal_period_range,
        macd_mode=macd_mode,
        ma_mode=ma_mode,
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
        # 双均线系统参数
        "短期均线",
        "长期均线",
        # MACD系统参数
        "MACD快线",
        "MACD慢线",
        "MACD信号线",
        # 止损参数
        "止损比例",
        "ATR周期",
        "ATR倍数",
        "使用ATR止损",
        # 资金管理
        "资金使用比例",
        # 系统状态
        "启用系统",
        # 性能指标
        "最终资金",
        "年化收益率",
        "夏普比率",
        "最大回撤",
        "胜率",
        "最新净值",
        "盈亏比",
        "当前回撤",
        "Calmar比率",
        "SQN",
        "VWR",
        "总交易次数",
        "运行天数",
    ]
    table_data = []
    for r in sorted_results:
        row = [
            # 双均线系统参数
            r["短期均线"],
            r["长期均线"],
            # MACD系统参数
            r["MACD快线"] if r["MACD快线"] != "未启用" else "",
            r["MACD慢线"] if r["MACD慢线"] != "未启用" else "",
            r["MACD信号线"] if r["MACD信号线"] != "未启用" else "",
            # 止损参数
            f"{r['止损比例']:.1%}",
            r["ATR周期"],
            r["ATR倍数"],
            "是" if r["使用ATR止损"] else "否",
            f"{r['资金使用比例']:.0%}",
            # 添加系统状态
            f"{'双均线' if r['短期均线'] != 0 and r['长期均线'] != 0 else ''}{'+' if r['短期均线'] != 0 and r['长期均线'] != 0 and r['MACD快线'] != 0 else ''}{'MACD' if r['MACD快线'] != 0 else ''}",
            # 性能指标保持不变
            f"{r['最终资金']:.2f}",
            f"{r['年化收益率']:.2f}%",
            f"{r['夏普比率']:.4f}",
            f"{r['最大回撤']:.2f}%",
            f"{r['胜率']:.2f}%",
            f"{r['最新净值']:.4f}",
            f"{r['盈亏比']:.2f}",
            f"{r['当前回撤']:.2f}%",
            f"{r['Calmar比率']:.4f}",
            f"{r['SQN']:.4f}",
            f"{r['VWR']:.4f}",
            r["总交易次数"],
            r["运行天数"],
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
        best_result = sorted_results[0]
        logger.info("\n最佳参数组合:")

        # 输出启用的系统
        systems = []
        if has_ma_system:
            systems.append(
                f"双均线系统(MA{best_result['短期均线']}/{best_result['长期均线']})"
            )
        if has_macd_system:
            systems.append(
                f"MACD系统({best_result['MACD快线']}/{best_result['MACD慢线']}/{best_result['MACD信号线']})"
            )
        logger.info(f"启用系统: {' + '.join(systems)}")

        # 输出止损策略
        logger.info(
            f"止损策略: "
            f"{'ATR动态止损' if best_result['使用ATR止损'] else '固定比例止损'}, "  # 使用结果中的值
            f"止损比例={best_result['止损比例']}, "
            f"ATR周期={best_result['ATR周期']}, "
            f"ATR倍数={best_result['ATR倍数']}"
        )

        # 输出性能指标
        logger.info(
            f"策略表现:\n"
            f"- 收益指标: 年化收益={best_result['年化收益率']:.2f}%, 最新净值={best_result['最新净值']:.4f}\n"
            f"- 风险指标: 夏普比率={best_result['夏普比率']:.4f}, 最大回撤={best_result['最大回撤']:.2f}%, Calmar比率={best_result['Calmar比率']:.4f}\n"
            f"- 交易指标: 胜率={best_result['胜率']:.2f}%, 盈亏比={best_result['盈亏比']:.2f}, SQN={best_result['SQN']:.4f}, VWR={best_result['VWR']:.4f}\n"
            f"- 交易统计: 总交易次数={best_result['总交易次数']}, 运行天数={best_result['运行天数']}"
        )
    else:
        logger.warning("\n没有找到有效的参数组合。请尝试调整参数范围或增加回测时间。")
