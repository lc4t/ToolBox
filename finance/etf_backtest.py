import backtrader as bt
import pandas as pd
import os
from datetime import datetime
import argparse
import matplotlib
from tabulate import tabulate
import itertools
import numpy as np

matplotlib.use("Agg")


# 定义策略
class ETFStrategy(bt.Strategy):
    params = (
        ("short_period", 20),
        ("long_period", 26),
        ("stop_loss", 0.10),  # 保持止损为10%
        ("cash_ratio", 0.95),  # 新增参数，默认使用95%的可用资金
        ("debug", False),  # 新增 debug 参数
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.trades = []
        self.cash = self.broker.getvalue()  # 初始资金
        self.highest_price = 0  # 用于跟踪最高价格（吊灯止损）

        # 添加移动平均线指标
        self.sma_short = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.short_period)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.long_period)

        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

    def next(self):
        if self.params.debug:  # 只在 debug 模式下打印
            print(
                f"Date: {self.data.datetime.date(0)}, Close: {self.dataclose[0]:.2f}, "
                f"SMA20: {self.sma_short[0]:.2f}, SMA26: {self.sma_long[0]:.2f}, "
                f"Crossover: {self.crossover[0]}, Position: {self.position.size}, "
                f"Cash: {self.broker.getcash():.2f}, Portfolio Value: {self.broker.getvalue():.2f}"
            )

        if self.order:
            return

        if not self.position:
            if self.crossover > 0:  # 20日均线上穿26日均线
                available_cash = self.broker.getcash() * self.params.cash_ratio
                size = int(available_cash / self.dataclose[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    print(
                        f"Buy order placed on {self.data.datetime.date(0)}, "
                        f"SMA20: {self.sma_short[0]:.2f}, SMA26: {self.sma_long[0]:.2f}, "
                        f"Price: {self.dataclose[0]:.2f}, Size: {size}, Amount: {size * self.dataclose[0]:.2f}, "
                        f"Cash: {self.broker.getcash():.2f}, Portfolio Value: {self.broker.getvalue():.2f}"
                    )
                else:
                    print(f"Not enough cash to place buy order on {self.data.datetime.date(0)}")
        else:
            # 更新最高价格
            if self.dataclose[0] > self.highest_price:
                self.highest_price = self.dataclose[0]

            # 检查是否触发吊灯止损
            if self.dataclose[0] <= self.highest_price * (1 - self.params.stop_loss):
                self.order = self.sell(size=self.position.size)
                print(
                    f"Stop loss triggered on {self.data.datetime.date(0)}, Highest: {self.highest_price:.2f}, Current: {self.dataclose[0]:.2f}"
                )
            elif self.crossover < 0:  # 20日均线下穿26日均线
                self.order = self.sell(size=self.position.size)
                print(f"Sell signal on {self.data.datetime.date(0)}, SMA20: {self.sma_short[0]:.2f}, SMA26: {self.sma_long[0]:.2f}")

    def notify_order(self, order):
        if self.params.debug:  # 只在 debug 模式下打印
            if order.status in [order.Submitted, order.Accepted]:
                print(f"Order {order.ref} {order.Status[order.status]}")
                return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self.highest_price = self.buyprice  # 初始化最高价格
                self.trades.append(
                    {
                        "类型": "买入",
                        "日期": self.data.datetime.date(0),
                        "价格": round(order.executed.price, 3),
                        "数量": round(order.executed.size, 3),
                        "消耗资金": round(order.executed.value, 4),
                        "手续费": round(order.executed.comm, 4),
                        "剩余资金": round(self.broker.getcash(), 4),
                    }
                )
                print(f"Buy executed on {self.data.datetime.date(0)}, Price: {order.executed.price:.2f}, Size: {order.executed.size}")
            else:  # Sell
                sell_value = order.executed.price * order.executed.size
                commission = order.executed.comm
                profit = round(sell_value - self.buyprice * order.executed.size - commission - self.buycomm, 4)
                self.trades.append(
                    {
                        "类型": "卖出",
                        "日期": self.data.datetime.date(0),
                        "价格": round(order.executed.price, 3),
                        "数量": round(order.executed.size, 3),
                        "获得资金": round(sell_value, 4),
                        "手续费": round(commission, 4),
                        "收益": profit,
                        "剩余资金": round(self.broker.getcash(), 4),
                    }
                )
                print(
                    f"Sell executed on {self.data.datetime.date(0)}, Price: {order.executed.price:.2f}, Size: {order.executed.size}, Profit: {profit:.2f}"
                )
                self.highest_price = 0  # 重置最高价格
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(
                f"Order {order.ref} {order.Status[order.status]} on {self.data.datetime.date(0)}, "
                f"Price: {self.dataclose[0]:.2f}, Cash: {self.broker.getcash():.2f}, "
                f"Reason: {order.info.get('reject_reason', 'Unknown')}, "
                f"Size: {order.created.size}, Price: {order.created.price:.2f}"
            )

        self.order = None

    def stop(self):
        # 在回测结束时打印交易记录
        print("\n交易记录:")
        if not self.trades:
            print("没有发生交易")
        else:
            headers = ["类型", "日期", "价格", "数量", "消耗/获得资金", "手续费", "收益", "剩余资金"]
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
                        ]
                    )
            print(
                tabulate(
                    table_data,
                    headers=headers,
                    tablefmt="grid",
                    colalign=("left", "left", "right", "right", "right", "right", "right", "right"),
                )
            )


# 获取ETF数据
def get_etf_data(symbol):
    csv_file = f"finance/etf_data_{symbol}.csv"
    if not os.path.exists(csv_file):
        print(f"错误：找不到 {symbol} 的数据文件。请先运行 test.py 获取数据。")
        return None

    df = pd.read_csv(csv_file)
    df["datetime"] = pd.to_datetime(df["日期"])
    df.set_index("datetime", inplace=True)
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
    print(f"数据范围：{df.index.min()} 到 {df.index.max()}")  # 打印数据范围
    return df[["open", "high", "low", "close", "volume"]]  # 移除了多余的右括号


# 主函数
def run_backtest(symbol, start_date, initial_cash, short_period, long_period, stop_loss, cash_ratio, debug):
    cerebro = bt.Cerebro()

    data = get_etf_data(symbol)
    if data is None:
        return None

    start_date = pd.to_datetime(start_date)
    end_date = data.index.max()
    data = data[(data.index >= start_date) & (data.index <= end_date)]

    feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(feed)

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0001)

    cerebro.addstrategy(
        ETFStrategy, short_period=short_period, long_period=long_period, stop_loss=stop_loss, cash_ratio=cash_ratio, debug=debug
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
        "资金使用比例": cash_ratio,
        "最终资金": cerebro.broker.getvalue(),
        "夏普比率": strat.analyzers.sharpe.get_analysis()["sharperatio"],
        "年化收益率": strat.analyzers.returns.get_analysis()["rnorm100"],
        "最大回撤": strat.analyzers.drawdown.get_analysis()["max"]["drawdown"],
    }


def optimize_parameters(symbol, start_date, initial_cash, short_period_range, long_period_range, stop_loss_range, cash_ratio_range, debug):
    results = []

    for short_period, long_period, stop_loss, cash_ratio in itertools.product(
        short_period_range, long_period_range, stop_loss_range, cash_ratio_range
    ):
        if short_period >= long_period:
            continue  # 跳过短期均线大于或等于长期均线的情况

        result = run_backtest(symbol, start_date, initial_cash, short_period, long_period, stop_loss, cash_ratio, debug)
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
    parser = argparse.ArgumentParser(description="ETF回测参数优化")
    parser.add_argument("symbol", type=str, help="ETF代码")
    parser.add_argument("--start_date", type=str, default="2018-01-01", help="回测开始日期 (YYYY-MM-DD)")
    parser.add_argument("--initial_cash", type=float, default=10000.0, help="初始资金")
    parser.add_argument("--short_period_range", type=str, default="20", help="短期均线范围，用逗号分隔或短横线表示范围")
    parser.add_argument("--long_period_range", type=str, default="26", help="长期均线范围，用逗号分隔或短横线表示范围")
    parser.add_argument("--stop_loss_range", type=str, default="0.10", help="止损比例范围，用逗号分隔或短横线表示范围")
    parser.add_argument("--cash_ratio_range", type=str, default="0.95", help="使用资金比例范围，用逗号分隔")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()

    short_period_range = [int(x) for x in parse_range(args.short_period_range)]
    long_period_range = [int(x) for x in parse_range(args.long_period_range)]
    stop_loss_range = [round(x, 2) for x in parse_range(args.stop_loss_range)]
    cash_ratio_range = [round(x, 2) for x in parse_range(args.cash_ratio_range)]

    results = optimize_parameters(
        args.symbol,
        args.start_date,
        args.initial_cash,
        short_period_range,
        long_period_range,
        stop_loss_range,
        cash_ratio_range,
        args.debug,
    )

    # 将排序逻辑修改为按年化收益率降序排列
    def sort_key(x):
        return (x["年化收益率"] is not None, x["年化收益率"] if x["年化收益率"] is not None else float("-inf"))

    sorted_results = sorted(results, key=sort_key, reverse=True)

    # 打印所有参数组合的表现结果
    headers = ["短期均线", "长期均线", "止损比例", "资金使用比例", "最终资金", "年化收益率", "夏普比率", "最大回撤"]
    table_data = []
    for r in sorted_results:
        row = [
            r["短期均线"],
            r["长期均线"],
            f"{r['止损比例']:.2f}",
            f"{r['资金使用比例']:.2f}",
            f"{r['最终资金']:.2f}",
            f"{r['年化收益率']:.2f}",
            f"{r['夏普比率']:.4f}" if r["夏普比率"] is not None else "N/A",
            f"{r['最大回撤']:.2f}",
        ]
        table_data.append(row)

    print("\n所有参数组合的表现结果（按年化收益率降序排列）:")
    print(tabulate(table_data, headers=headers, tablefmt="pretty", numalign="right"))

    # 打印最佳参数组合（如果有结果的话）
    if sorted_results:
        best_params = sorted_results[0]
        best_row = [
            best_params["短期均线"],
            best_params["长期均线"],
            f"{best_params['止损比例']:.2f}",
            f"{best_params['资金使用比例']:.2f}",
            f"{best_params['最终资金']:.2f}",
            f"{best_params['年化收益率']:.2f}",
            f"{best_params['夏普比率']:.4f}" if best_params["夏普比率"] is not None else "N/A",
            f"{best_params['最大回撤']:.2f}",
        ]
        print("\n最佳参数组合（基于最高年化收益率）:")
        print(tabulate([best_row], headers=headers, tablefmt="pretty", numalign="right"))

        # 保存最佳参数组合的回测结果图表
        run_backtest(
            args.symbol,
            args.start_date,
            args.initial_cash,
            best_params["短期均线"],
            best_params["长期均线"],
            best_params["止损比例"],
            best_params["资金使用比例"],
            args.debug,
        )
        print(f"\n最佳参数组合的回测结果图表已保存为 {args.symbol}_backtest_result.png")
    else:
        print("\n没有找到有效的参数组合。请尝试调整参数范围或增加回测时间。")
