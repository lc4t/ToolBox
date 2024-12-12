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

def print_metrics(metrics: dict):
    """格式化打印性能指标"""
    logger.info(f"\n{SECTION_SEPARATOR}")
    logger.info(f"{'性能指标':^{SEPARATOR_WIDTH}}")
    logger.info(SECTION_SEPARATOR)

    # 使用两列布局
    metrics_layout = [
        (
            "最新净值",
            f"{metrics['latest_nav']:>12.2f}",
            "年化收益率",
            f"{metrics['annual_return']:>12.2f}%",
        ),
        (
            "总交易次数",
            f"{metrics['total_trades']:>12d}",
            "胜率",
            f"{metrics['win_rate']:>12.2f}%",
        ),
        (
            "盈利交易",
            f"{metrics['won_trades']:>12d}",
            "亏损交易",
            f"{metrics['lost_trades']:>12d}",
        ),
        (
            "平均盈利",
            f"{metrics['avg_won']:>12.2f}",
            "平均亏损",
            f"{metrics['avg_lost']:>12.2f}",
        ),
        (
            "夏普比率",
            f"{metrics['sharpe_ratio']:>12.2f}",
            "盈亏比",
            f"{metrics['profit_factor']:>12.2f}",
        ),
        (
            "最大回撤",
            f"{metrics['max_drawdown']:>12.2f}%",
            "当前回撤",
            f"{metrics['current_drawdown']:>12.2f}%",
        ),
        (
            "Calmar比率",
            f"{metrics['calmar_ratio']:>12.2f}",
            "VWR",
            f"{metrics['vwr']:>12.2f}",
        ),
        (
            "SQN",
            f"{metrics['sqn']:>12.2f}",
            "运行天数",
            f"{metrics['running_days']:>12d}",
        ),
    ]

    for row in metrics_layout:
        logger.info(f"{row[0]:<12}{row[1]:<16}{row[2]:<12}{row[3]}")

def print_combination_result(idx: int, result: dict):
    """格式化打印单个参数组合的结果"""
    logger.info(f"\n{SECTION_SEPARATOR}")
    logger.info(f"{'参数组合 ' + str(idx):^{SEPARATOR_WIDTH}}")
    logger.info(SECTION_SEPARATOR)

    # 构建参数字典
    if "params" not in result:
        # 单参数回测的情况
        params = {
            "use_ma": result.get("use_ma", False),
            "short_period": result.get("short_period"),
            "long_period": result.get("long_period"),
            "use_chandelier": result.get("use_chandelier", False),
            "chandelier_period": result.get("chandelier_period"),
            "chandelier_multiplier": result.get("chandelier_multiplier"),
            "use_adr": result.get("use_adr", False),
            "adr_period": result.get("adr_period"),
            "adr_multiplier": result.get("adr_multiplier"),
        }
    else:
        # 参数组合回测的情况
        params = result["params"]

    # 格式化参数字符串
    params_str = format_params_string(params)
    logger.info(f"参数配置: {params_str}")
    logger.info(SUBSECTION_SEPARATOR)

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

def print_best_results(results: dict):
    """打印最佳结果"""
    # 打印最佳年化收益率组合
    logger.info("\n=== 最佳年化收益率组合 ===")
    best_return = results["best_annual_return"]
    logger.info(f"参数: {format_params_string(best_return['params'])}")
    logger.info(f"年化收益率: {best_return['annual_return']:.2f}%")

    # 打印最小回撤组合
    logger.info("\n=== 最小回撤组合 ===")
    min_dd = results["min_drawdown"]
    logger.info(f"参数: {format_params_string(min_dd['params'])}")
    logger.info(f"最大回撤: {min_dd['max_drawdown']:.2f}%")

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