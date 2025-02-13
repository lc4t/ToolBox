import json
from datetime import date, datetime
from typing import Dict, List

import pandas as pd
from loguru import logger
from tabulate import tabulate

from db import DBClient, SymbolInfo

# 常量定义
SEPARATOR_WIDTH = 60
SECTION_SEPARATOR = "=" * SEPARATOR_WIDTH
SUBSECTION_SEPARATOR = "-" * SEPARATOR_WIDTH


def format_value(v):
    """处理值，确保JSON可序列化"""
    if isinstance(v, (datetime, date)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, float):
        return round(v, 3)
    return v


def print_metrics(metrics: Dict):
    """打印性能指标"""
    logger.info(f"\n{SECTION_SEPARATOR}")
    logger.info(f"{'性能指标':^{SEPARATOR_WIDTH}}")
    logger.info(SECTION_SEPARATOR)

    # 使用三列布局
    metrics_layout = [
        # 基础指标
        (
            "最新净值",
            f"{metrics['latest_nav']:>12.2f}",
            "年化收益率",
            f"{metrics['annual_return']:>12.2f}%",
            "复合年化收益",
            f"{metrics['cagr']:>12.2f}%",
        ),
        # 交易统计
        (
            "总交易次数",
            f"{metrics['total_trades']:>12d}",
            "胜率",
            f"{metrics['win_rate']:>12.2f}%",
            "盈亏比",
            f"{metrics['profit_factor']:>12.2f}",
        ),
        (
            "盈利交易",
            f"{metrics['won_trades']:>12d}",
            "亏损交易",
            f"{metrics['lost_trades']:>12d}",
            "总盈亏",
            f"{metrics['total_pnl']:>12.2f}",
        ),
        (
            "平均盈利",
            f"{metrics['avg_won']:>12.2f}",
            "平均亏损",
            f"{metrics['avg_lost']:>12.2f}",
            "持仓比例",
            f"{metrics['holding_ratio']:>12.2f}%",
        ),
        # 风险指标
        (
            "年化波动率",
            f"{metrics['volatility']:.2f}%",
            "最大回撤",
            f"{metrics['max_drawdown']:.2f}%",
            "最大亏损金额",
            f"{metrics['max_loss_amount']:.2f}",
        ),
        (
            "最大亏损比例",
            f"{metrics['max_loss_pct']:.2f}%",
            "夏普比率",
            f"{metrics['sharpe_ratio']:.2f}",
            "索提诺比率",
            f"{metrics['sortino_ratio']:.2f}",
        ),
        (
            "卡玛比率",
            f"{metrics['calmar_ratio']:.2f}",
            "VWR",
            f"{metrics['vwr']:.2f}",
            "SQN",
            f"{metrics['sqn']:.2f}",
        ),
        # 连续交易统计
        (
            "最大连胜",
            f"{metrics['max_consecutive_wins']:>12d}",
            "最大连亏",
            f"{metrics['max_consecutive_losses']:>12d}",
            "平均持仓",
            f"{metrics['avg_holding_period']:>12}天",
        ),
        # 时间统计
        (
            "运行天数",
            f"{metrics['running_days']:>12d}",
            "开始日期",
            f"{metrics['start_date'].strftime('%Y-%m-%d'):>12}",
            "结束日期",
            f"{metrics['end_date'].strftime('%Y-%m-%d'):>12}",
        ),
        # 波动率相关指标
        (
            "波动率",
            f"{metrics.get('volatility', 0):>12.2f}%",
            "Beta系数",
            f"{metrics.get('beta', 0):>12.2f}",
            "Alpha",
            f"{metrics.get('alpha', 0):>12.2f}%",
        ),
        (
            "Beta参考",
            f"{str(metrics.get('benchmark_symbol', 'N/A')):>12}",
            "Beta状态",
            f"{metrics.get('beta_status', 'N/A'):>12}",
            "最大亏损比例",
            f"{metrics.get('max_loss_pct', 0):>12.2f}%",
        ),
    ]

    # 打印三列布局
    for row in metrics_layout:
        logger.info(
            f"{row[0]:<12}{row[1]:<16}{row[2]:<12}{row[3]:<16}{row[4]:<12}{row[5]}"
        )

    # 打印年度收益率
    if "yearly_returns" in metrics:
        logger.info(SUBSECTION_SEPARATOR)
        logger.info("年度收益率:")
        for year, return_rate in metrics["yearly_returns"].items():
            logger.info(f"{year}年: {return_rate:>12.2f}%")

    # 打印月度收益率（如果有）
    if "monthly_returns" in metrics:
        logger.info(SUBSECTION_SEPARATOR)
        logger.info("月度收益率:")
        for month, return_rate in metrics["monthly_returns"].items():
            logger.info(f"{month}: {return_rate:>12.2f}%")


def print_combination_result(idx: int, result: dict):
    """格式化打印单个参数组合的结果"""
    logger.info(f"\n{SECTION_SEPARATOR}")
    logger.info(f"{'参数组合 ' + str(idx):^{SEPARATOR_WIDTH}}")
    logger.info(SECTION_SEPARATOR)

    # 构建参数字典
    if "params" not in result:
        # 单参数回测的情况，需要从结果中提取指标
        metrics = result.get("metrics", {})  # 直接从结果中获取指标
    else:
        # 参数组合回测的情况
        metrics = result.get("full_result", {}).get("metrics", {})

    # 格式化参数字符串
    params_str = format_params_string(result.get("params", result))  # 修改这里以适应两种情况
    logger.info(f"参数配置: {params_str}")
    logger.info(SUBSECTION_SEPARATOR)
    
    # 构建表格数据
    table_data = [
        ["年化收益率", f"{metrics.get('annual_return', 0):.2f}%"],
        ["最大回撤", f"{metrics.get('max_drawdown', 0):.2f}%"],
        ["夏普比率", f"{metrics.get('sharpe_ratio', 0):.2f}"],
        ["波动率", f"{metrics.get('volatility', 0):.2f}%"],
        ["Beta系数", f"{metrics.get('beta', 0):.2f}"],
        ["Alpha", f"{metrics.get('alpha', 0):.2f}%"],
        ["最大亏损金额", f"{metrics.get('max_loss_amount', 0):.2f}"],
        ["最大亏损比例", f"{metrics.get('max_loss_pct', 0):.2f}%"],
        ["总交易次数", str(metrics.get('total_trades', 0))],
        ["胜率", f"{metrics.get('win_rate', 0):.2f}%"],
        ["盈亏比", f"{metrics.get('profit_factor', 0):.2f}"],
        ["Calmar比率", f"{metrics.get('calmar_ratio', 0):.2f}"],
        ["VWR", f"{metrics.get('vwr', 0):.2f}"],
        ["SQN", f"{metrics.get('sqn', 0):.2f}"],
    ]

    # 使用 tabulate 打印表格
    logger.info("\n" + tabulate(
        table_data,
        headers=["指标", "值"],
        tablefmt="simple",
        colalign=("left", "right")
    ))

    # 打印交易时间范围
    if metrics.get("start_date") and metrics.get("end_date"):
        logger.info(SUBSECTION_SEPARATOR)
        logger.info(
            f"交易区间: {metrics['start_date'].strftime('%Y-%m-%d')} - "
            f"{metrics['end_date'].strftime('%Y-%m-%d')}"
        )


def print_next_signal(next_signal: dict):
    """格式化打印下一交易日信号"""
    logger.info(f"\n{SECTION_SEPARATOR}")
    logger.info(f"{'下一交易日信号':^{SEPARATOR_WIDTH}}")
    logger.info(SECTION_SEPARATOR)

    logger.info(f"建议动作: {next_signal['action']:>12}")

    if next_signal["position_info"]:
        pos_info = next_signal["position_info"]
        logger.info(SUBSECTION_SEPARATOR)
        logger.info("当前持仓信息:")
        info_layout = [
            ("买入日期", pos_info["entry_date"]),
            ("买入价格", f"{pos_info['entry_price']:.3f}"),
            ("买入金额", f"{pos_info['position_value']:.2f}"),
            ("持仓数量", str(pos_info["position_size"])),
            ("当前价格", f"{pos_info['current_price']:.3f}"),
            ("当前市值", f"{pos_info['current_value']:.2f}"),
            (
                "浮动盈亏",
                f"{pos_info['unrealized_pnl']:.2f} ({pos_info['unrealized_pnl_pct']:.2f}%)",
            ),
        ]
        for label, value in info_layout:
            logger.info(f"{label:>12}: {value}")


def print_trades(trades: List[dict], title: str = "交易记录"):
    """格式化打印交易记录"""
    logger.info(f"\n=== {title} ===")
    if not trades:
        logger.info("没有交易记录")
        return

    headers = [
        "日期",
        "动作",
        "价格",
        "数量",
        "交易金额",
        "手续费",
        "盈亏",
        "总资产",
        "信号原因",
    ]

    table_data = [
        [
            trade["date"],
            trade["action"],
            f"{trade['price']:.3f}",
            trade["size"],
            f"{trade['value']:.2f}",
            f"{trade['commission']:.2f}",
            f"{trade['pnl']:.2f}",
            f"{trade['total_value']:.2f}",
            trade["signal_reason"],
        ]
        for trade in trades
    ]

    logger.info(tabulate(table_data, headers=headers, tablefmt="grid"))


def print_parameter_summary(combinations: List[dict]):
    """格式化打印参数组合汇总"""
    logger.info("\n=== 参数组合汇总（按年化收益率排序）===")
    headers = [
        "参数组合",
        "年化收益率(%)",
        "最大回撤(%)",
        "交易次数",
        "首次交易",
        "最后交易",
    ]

    # 按年化收益率排序（从低到高）
    sorted_combinations = sorted(
        combinations,
        key=lambda x: x["annual_return"],
        reverse=False,
    )

    table_data = []
    for result in sorted_combinations:
        params_str = format_params_string(result["params"])
        table_data.append(
            [
                params_str,
                f"{result['annual_return']:.2f}",
                f"{result['max_drawdown']:.2f}",
                result["total_trades"],
                result["first_trade"],
                result["last_trade"],
            ]
        )

    logger.info(tabulate(table_data, headers=headers, tablefmt="grid"))


def format_params_string(params: dict) -> str:
    """格式化参数字符串，只显示启用的参数"""
    parts = []

    # 双均线参数
    if params.get("use_ma"):
        parts.append(f"MA={params['short_period']}/{params['long_period']}")

    # 吊灯止损参数
    if params.get("use_chandelier"):
        parts.append(
            f"ATR={params['chandelier_multiplier']}x{params['chandelier_period']}"
        )

    # ADR止损参数
    if params.get("use_adr"):
        parts.append(f"ADR={params['adr_multiplier']}x{params['adr_period']}")

    return ", ".join(parts)


def print_best_results(results: Dict):
    """打印最佳结果"""
    if not results.get("combinations"):
        logger.warning("没有有效的回测结果")
        return

    # 打印年化收益率最高的组合
    best_return = results.get("best_annual_return")
    if best_return:
        logger.info("\n=== 最佳年化收益率组合 ===")
        logger.info(f"参数: {format_params_string(best_return['params'])}")
        logger.info(f"年化收益率: {best_return['annual_return']:.2f}%")
        logger.info(f"最大回撤: {best_return['max_drawdown']:.2f}%")
        logger.info(f"交易次数: {best_return['total_trades']}")

    # 打印最小回撤组合
    min_dd = results.get("min_drawdown")
    if min_dd:
        logger.info("\n=== 最小回撤组合 ===")
        logger.info(f"参数: {format_params_string(min_dd['params'])}")
        logger.info(f"年化收益率: {min_dd['annual_return']:.2f}%")
        logger.info(f"最大回撤: {min_dd['max_drawdown']:.2f}%")
        logger.info(f"交易次数: {min_dd['total_trades']}")

    # 打印最佳夏普比率的组合
    best_sharpe = results.get("best_sharpe")
    if best_sharpe:
        logger.info("\n=== 最佳夏普比率组合 ===")
        logger.info(f"参数: {format_params_string(best_sharpe['params'])}")
        logger.info(f"年化收益率: {best_sharpe['annual_return']:.2f}%")
        logger.info(f"最大回撤: {best_sharpe['max_drawdown']:.2f}%")
        logger.info(
            f"夏普比率: {best_sharpe['full_result']['metrics']['sharpe_ratio']:.2f}"
        )
        logger.info(f"交易次数: {best_sharpe['total_trades']}")


def get_stock_name(db_client: DBClient, symbol: str) -> str:
    """从数据库获取股票名称"""
    try:
        with db_client.Session() as session:
            symbol_info = (
                session.query(SymbolInfo).filter(SymbolInfo.symbol == symbol).first()
            )
            if symbol_info:
                return symbol_info.name
    except Exception as e:
        logger.error(f"Error getting stock name: {e}")
    return symbol  # 如果获取失败，返回股票代码作为名称


def format_for_json(
    metrics: dict,
    trades: list,
    next_signal: dict,
    params: dict,
    symbol: str,
    stock_name: str,
    initial_capital: float,
) -> dict:
    """格式化回测结果为JSON格式"""
    # 从数据库获取最新价格
    db_client = DBClient()
    latest_data = db_client.query_latest_by_symbol(symbol)

    if latest_data:
        latest_prices = {
            "open": round(latest_data["open_price"], 3),
            "close": round(latest_data["close_price"], 3),
            "high": round(latest_data["high"], 3),
            "low": round(latest_data["low"], 3),
        }
        latest_date = latest_data["date"].strftime("%Y-%m-%d")
    else:
        # 如果无法获取数据库数据，使用next_signal中的价格
        latest_prices = next_signal.get("prices", {})
        latest_date = next_signal.get("timestamp", datetime.now().strftime("%Y-%m-%d"))

    # 基础信息
    result = {
        "symbol": symbol,
        "name": stock_name,
        "reportDate": datetime.now().strftime("%Y-%m-%d"),
        "dateRange": {
            "start": metrics["start_date"].strftime("%Y-%m-%d"),
            "end": latest_date,  # 使用数据库中的最新日期
        },
        "latestSignal": {
            "action": next_signal["action"],
            "asset": stock_name,
            "timestamp": latest_date,  # 使用数据库中的最新日期
            "prices": latest_prices,  # 使用数据库中的最新价格
        },
        "positionInfo": next_signal.get("position_info"),
        # 年度收益率
        "annualReturns": [
            {"year": int(year), "value": round(value, 3)}  # 保留3位小数
            for year, value in metrics.get("yearly_returns", {}).items()
        ],
        # 1. 收益指标 - 反映策略的盈利能力
        "returnMetrics": [
            {
                "name": "最新净值",
                "value": round(metrics["latest_nav"], 3),
                "description": "当前投资组合价值相对于初始资金的比值，反映总体收益情况",
            },
            {
                "name": "年化收益率",
                "value": round(metrics["annual_return"], 2),
                "description": "将总收益按年度平均计算的收益率，用于评估策略的整体盈利能力",
            },
            {
                "name": "复合年化收益",
                "value": round(metrics["cagr"], 2),
                "description": "考虑收益再投资的年化收益率，更准确地反映长期投资回报",
            },
            {
                "name": "Alpha",
                "value": round(metrics.get("alpha", 0), 2),
                "description": "策略相对于市场基准的超额收益，反映策略的选股择时能力",
            },
            {
                "name": "总盈亏",
                "value": round(metrics["total_pnl"], 2),
                "description": "所有交易产生的盈亏总和，包括已实现和未实现盈亏",
            },
        ],
        # 2. 风险指标 - 反映策略的风险水平
        "riskMetrics": [
            {
                "name": "最大回撤",
                "value": round(metrics["max_drawdown"], 2),
                "description": "任意时间段内净值的最大跌幅，反映策略最大可能损失，越小越好",
            },
            {
                "name": "当前回撤",
                "value": round(metrics["current_drawdown"], 2),
                "description": "当前净值距离历史最高点的跌幅，反映当前的风险状况",
            },
            {
                "name": "波动率",
                "value": round(metrics["volatility"], 2),
                "description": "收益率的标准差，反映策略的波动性和风险大小，越小越稳定",
            },
            {
                "name": "最大亏损金额",
                "value": round(metrics["max_loss_amount"], 2),  # 直接使用金额，不需要转换
                "description": "单笔交易中的最大亏损金额，反映策略的风险控制能力",
            },
            {
                "name": "最大亏损比例",
                "value": round(metrics["max_loss_pct"], 2),
                "description": "单笔交易中的最大亏损比例，反映策略的风险控制能力",
            },
            {
                "name": "Beta系数",
                "value": round(metrics.get("beta", 0), 2),
                "description": "策略收益相对于市场基准的敏感度，=1表示与市场同步，<1表示波动小于市场",
            },
        ],
        # 3. 风险调整收益指标 - 综合考虑收益和风险
        "riskAdjustedMetrics": [
            {
                "name": "夏普比率",
                "value": round(metrics["sharpe_ratio"], 2),
                "description": "超额收益与波动率的比值，大于1表示较好，大于2表示优秀，是衡量风险调整后收益的重要指标",
            },
            {
                "name": "索提诺比率",
                "value": round(metrics["sortino_ratio"], 2),
                "description": "类似夏普比率，但只考虑下行波动率，更关注亏损风险。大于2表示较好",
            },
            {
                "name": "Calmar比率",
                "value": round(metrics["calmar_ratio"], 2),
                "description": "年化收益率除以最大回撤，反映单位风险下的收益能力。大于1表示较好，大于3表示优秀",
            },
            {
                "name": "VWR",
                "value": round(metrics["vwr"], 2),
                "description": "波动率加权收益率，综合考虑收益和波动性，通常大于5表示较好",
            },
            {
                "name": "SQN",
                "value": round(metrics["sqn"], 2),
                "description": "系统质量指数，反映策略的稳定性和可靠性，大于2表示较好，大于4表示优秀",
            },
        ],
        # 4. 交易统计指标 - 反映策略的交易特征
        "tradingMetrics": [
            {
                "name": "总交易次数",
                "value": metrics["total_trades"],
                "description": "策略产生的买入和卖出操作总次数，反映交易频率",
            },
            {
                "name": "胜率",
                "value": round(metrics["win_rate"], 2),
                "description": "盈利交易占总交易的比例，反映策略的准确性和稳定性",
            },
            {
                "name": "盈亏比",
                "value": round(metrics["profit_factor"], 2),
                "description": "总盈利除以总亏损的比值，大于1表示策略整体盈利，越大越好",
            },
            {
                "name": "平均盈利",
                "value": round(metrics["avg_won"], 2),
                "description": "单次盈利交易的平均盈利金额",
            },
            {
                "name": "平均亏损",
                "value": round(metrics["avg_lost"], 2),
                "description": "单次亏损交易的平均亏损金额",
            },
            {
                "name": "盈利交易",
                "value": metrics["won_trades"],
                "description": "产生盈利的交易次数",
            },
            {
                "name": "亏损交易",
                "value": metrics["lost_trades"],
                "description": "产生亏损的交易次数",
            },
            {
                "name": "最大连胜",
                "value": metrics["max_consecutive_wins"],
                "description": "最大连续盈利的交易次数，反映策略的稳定性",
            },
            {
                "name": "最大连亏",
                "value": metrics["max_consecutive_losses"],
                "description": "最大连续亏损的交易次数，反映策略的风险控制能力",
            },
        ],
        # 5. 持仓特征指标 - 反映策略的持仓特点
        "positionMetrics": [
            {
                "name": "持仓比例",
                "value": round(metrics["holding_ratio"], 2),
                "description": "持仓时间占总交易时间的百分比，反映资金利用率",
            },
            {
                "name": "平均持仓",
                "value": f"{metrics['avg_holding_period']}天",
                "description": "每笔交易的平均持有时间，反映策略的交易周期",
            },
        ],
        # 6. 基准对比指标 - 用于市场对比分析
        "benchmarkMetrics": [
            {
                "name": "Beta参考",
                "value": metrics.get("benchmark_symbol", "N/A"),
                "description": "用于计算Beta和Alpha的市场基准指数",
            },
            {
                "name": "Beta状态",
                "value": metrics.get("beta_status", "N/A"),
                "description": "Beta系数的计算状态，说明计算是否成功及原因",
            },
        ],
        # 7. 时间统计指标 - 反映策略的时间维度
        "timeMetrics": [
            {
                "name": "运行天数",
                "value": metrics["running_days"],
                "description": "回测的总天数，反映样本期长度",
            },
            {
                "name": "开始日期",
                "value": metrics["start_date"].strftime("%Y-%m-%d"),
                "description": "回测起始的交易日期",
            },
            {
                "name": "结束日期",
                "value": metrics["end_date"].strftime("%Y-%m-%d"),
                "description": "回测结束的交易日期",
            },
        ],
        "strategyParameters": None,
        "showStrategyParameters": False,
    }

    # 格式化交易记录
    formatted_trades = []
    buy_queue = []  # 用于追踪买入记录的FIFO队列

    for trade in trades:
        # 检查日期格式，如果已经是字符串就直接使用
        trade_date = trade["date"]
        if isinstance(trade_date, (datetime, date)):
            trade_date = trade_date.strftime("%Y-%m-%d")

        # 计算盈亏百分比（仅针对卖出交易）
        pnl_percentage = None
        entry_price = None

        if trade["action"] == "BUY":
            # 记录买入信息
            buy_queue.append(
                {
                    "price": trade["price"],
                    "size": trade["size"],
                    "remaining": trade["size"],  # 用于追踪剩余数量
                }
            )
        elif trade["action"] == "SELL" and buy_queue:
            # 计算加权平均买入价格
            total_size = trade["size"]
            total_cost = 0
            size_left = total_size

            # 从最早的买入记录开始匹配
            for buy in buy_queue:
                if buy["remaining"] > 0:
                    matched_size = min(size_left, buy["remaining"])
                    total_cost += matched_size * buy["price"]
                    size_left -= matched_size
                    buy["remaining"] -= matched_size

                    if size_left == 0:
                        break

            # 如果所有数量都匹配上了
            if size_left == 0:
                entry_price = total_cost / total_size
                pnl_percentage = ((trade["price"] - entry_price) / entry_price) * 100

            # 清理已用完的买入记录
            buy_queue = [buy for buy in buy_queue if buy["remaining"] > 0]

        formatted_trade = {
            "date": trade_date,
            "action": trade["action"],
            "price": round(trade["price"], 3),
            "quantity": trade["size"],
            "value": round(trade["value"], 2),
            "profitLoss": round(trade.get("pnl", 0), 2),
            "profitLossPercentage": (
                round(pnl_percentage, 2) if pnl_percentage is not None else None
            ),
            "totalValue": round(trade["total_value"], 2),
            "reason": trade.get("signal_reason", ""),
            "entryPrice": round(entry_price, 3) if entry_price is not None else None,
        }

        # 调试输出

        formatted_trades.append(formatted_trade)

    result.update(
        {
            "recentTrades": formatted_trades,
        }
    )

    # 格式化持仓信息中的数值
    if next_signal.get("position_info"):
        position_info = next_signal["position_info"]
        result["positionInfo"] = {
            "entry_date": position_info["entry_date"],
            "entry_price": round(position_info["entry_price"], 3),
            "position_size": position_info["position_size"],
            "position_value": round(position_info["position_value"], 2),
            "current_price": round(position_info["current_price"], 3),
            "current_value": round(position_info["current_value"], 2),
            "cost": round(position_info["cost"], 2),
            "unrealized_pnl": round(position_info["unrealized_pnl"], 2),
            "unrealized_pnl_pct": round(position_info["unrealized_pnl_pct"], 2),
        }
    else:
        result["positionInfo"] = None

    return result


def export_results_to_csv(combinations: List[Dict], output_file: str):
    """将回测结果导出为CSV文件"""
    # 准备数据列表
    data = []
    for result in combinations:
        metrics = result.get("full_result", {}).get("metrics", {})
        params = result.get("params", {})
        
        # 构建单行数据
        row = {
            # 策略参数
            "短期均线周期": params.get("short_period"),
            "长期均线周期": params.get("long_period"),
            "吊灯周期": params.get("chandelier_period"),
            "吊灯乘数": params.get("chandelier_multiplier"),
            "ADR周期": params.get("adr_period"),
            "ADR乘数": params.get("adr_multiplier"),
            
            # 收益指标
            "年化收益率(%)": metrics.get("annual_return"),
            "总盈亏(元)": metrics.get("total_pnl"),
            "复合年化收益率(%)": metrics.get("cagr"),
            
            # 风险指标
            "最大回撤(%)": metrics.get("max_drawdown"),
            "波动率(%)": metrics.get("volatility"),
            "Beta系数": metrics.get("beta"),
            "Alpha(%)": metrics.get("alpha"),
            "最大亏损金额(元)": metrics.get("max_loss_amount"),
            "最大亏损比例(%)": metrics.get("max_loss_pct"),
            
            # 风险调整收益
            "夏普比率": metrics.get("sharpe_ratio"),
            "索提诺比率": metrics.get("sortino_ratio"),
            "卡玛比率": metrics.get("calmar_ratio"),
            "VWR": metrics.get("vwr"),
            "SQN": metrics.get("sqn"),
            
            # 交易统计
            "总交易次数": metrics.get("total_trades"),
            "胜率(%)": metrics.get("win_rate"),
            "盈亏比": metrics.get("profit_factor"),
            "平均盈利(元)": metrics.get("avg_won"),
            "平均亏损(元)": metrics.get("avg_lost"),
            "最大连续盈利次数": metrics.get("max_consecutive_wins"),
            "最大连续亏损次数": metrics.get("max_consecutive_losses"),
            
            # 时间统计
            "开始日期": metrics.get("start_date").strftime("%Y-%m-%d") if metrics.get("start_date") else None,
            "结束日期": metrics.get("end_date").strftime("%Y-%m-%d") if metrics.get("end_date") else None,
            "运行天数": metrics.get("running_days"),
        }
        data.append(row)
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data)
    
    # 设置列的显示顺序
    columns_order = [
        # 策略参数
        "短期均线周期", "长期均线周期", "吊灯周期", "吊灯乘数", "ADR周期", "ADR乘数",
        # 收益指标
        "年化收益率(%)", "总盈亏(元)", "复合年化收益率(%)",
        # 风险指标
        "最大回撤(%)", "波动率(%)", "Beta系数", "Alpha(%)", 
        "最大亏损金额(元)", "最大亏损比例(%)",
        # 风险调整收益
        "夏普比率", "索提诺比率", "卡玛比率", "VWR", "SQN",
        # 交易统计
        "总交易次数", "胜率(%)", "盈亏比", 
        "平均盈利(元)", "平均亏损(元)", 
        "最大连续盈利次数", "最大连续亏损次数",
        # 时间统计
        "开始日期", "结束日期", "运行天数"
    ]
    
    # 按指定顺序重排列列
    df = df[columns_order]
    
    # 保存为CSV，使用UTF-8编码
    df.to_csv(output_file, index=False, float_format="%.4f", encoding='utf-8')
    logger.info(f"回测结果已保存到CSV文件: {output_file}")
