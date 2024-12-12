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
            "夏普比率",
            f"{metrics['sharpe_ratio']:>12.2f}",
            "最大回撤",
            f"{metrics['max_drawdown']:>12.2f}%",
            "当前回撤",
            f"{metrics['current_drawdown']:>12.2f}%",
        ),
        (
            "Calmar比率",
            f"{metrics['calmar_ratio']:>12.2f}",
            "索提诺比率",
            f"{metrics['sortino_ratio']:>12.2f}",
            "波动率",
            f"{metrics['volatility']:>12.2f}%",
        ),
        (
            "VWR",
            f"{metrics['vwr']:>12.2f}",
            "SQN",
            f"{metrics['sqn']:>12.2f}",
            "最大亏损",
            f"{metrics['max_loss']:>12.2f}",
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
            "最大亏损",
            f"{metrics.get('max_loss', 0):>12.2f}%",
        ),
    ]

    # 打印三列布局
    for row in metrics_layout:
        logger.info(f"{row[0]:<12}{row[1]:<16}{row[2]:<12}{row[3]:<16}{row[4]:<12}{row[5]}")

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

    # 打印最小回撤的组合
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
        logger.info(f"夏普比率: {best_sharpe['full_result']['metrics']['sharpe_ratio']:.2f}")
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
            "low": round(latest_data["low"], 3)
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
            {
                "year": int(year),
                "value": round(value, 3)  # 保留3位小数
            }
            for year, value in metrics.get("yearly_returns", {}).items()
        ],
        
        # 性能指标
        "performanceMetrics": [
            {
                "name": "夏普比率",
                "value": round(metrics["sharpe_ratio"], 3),
                "description": "衡量投资组合的超额回报与波动性的比率。大于1表示较好，小于0表示不佳。"
            },
            {
                "name": "最大回撤",
                "value": round(metrics["max_drawdown"], 3),
                "description": "历史最大的亏损幅度，反映策略的风险承受能力。越小越好。"
            },
            {
                "name": "总交易次数",
                "value": metrics["total_trades"],  # 整数不需要round
                "description": "策略执行期间的总交易次数。反映策略的交易频率。"
            },
            {
                "name": "胜率",
                "value": round(metrics["win_rate"], 3),
                "description": "盈利交易占总交易的比例。反映策略的准确性。"
            },
            {
                "name": "平均盈利",
                "value": round(metrics["avg_won"], 3),
                "description": "每笔盈利交易的平均收益"
            },
            {
                "name": "平均亏损",
                "value": round(metrics["avg_lost"], 3),
                "description": "每笔亏损交易的平均损失"
            },
            {
                "name": "盈亏比",
                "value": round(metrics["profit_factor"], 3),
                "description": "平均盈利与平均亏损的比值"
            },
            {
                "name": "最大连续盈利次数",
                "value": metrics["max_consecutive_wins"],  # 整数不需要round
                "description": "最多连续盈利的交易次数"
            },
            {
                "name": "最大连续亏损次数",
                "value": metrics["max_consecutive_losses"],  # 整数不需要round
                "description": "最多连续亏损的交易次数"
            }
        ],
        
        # 风险指标
        "riskMetrics": [
            {
                "name": "Calmar比率",
                "value": round(metrics["calmar_ratio"], 3),
                "description": "年化收益率与最大回撤的比值，反映收益与风险的平衡。大于1表示较好，大于3表示优秀。"
            },
            {
                "name": "当前回撤",
                "value": round(metrics["current_drawdown"], 3),
                "description": "当前亏损相对历史最高点的百分比。"
            },
            {
                "name": "VWR指标",
                "value": round(metrics["vwr"], 3),
                "description": "波动率加权收益率，综合考虑收益和波动性。通常大于5表示较好。"
            },
            {
                "name": "SQN指标",
                "value": round(metrics["sqn"], 3),
                "description": "系统质量指数，衡量交易系统的稳定性。大于2表示较好，大于3表示优秀。"
            },
            {
                "name": "波动率",
                "value": round(metrics.get("volatility", 0), 3),
                "description": "收益率的标准差"
            },
            {
                "name": "Beta系数",
                "value": round(metrics.get("beta", 0), 3),
                "description": "相对于大盘的波动程度"
            }
        ],
        
        # 市场指标
        "marketIndicators": [
            {
                "name": "年化收益率",
                "value": round(metrics["annual_return"], 3),
                "description": "将总收益率换算成年化收益率，便于与其他投资品比较。"
            },
            {
                "name": "累计收益",
                "value": round(metrics["total_pnl"], 3),
                "description": "从策略开始到现在的总收益。"
            },
            {
                "name": "累计收益率",
                "value": round((metrics["latest_nav"] - 1) * 100, 3),
                "description": "从策略开始到现在的总收益率"
            },
            {
                "name": "年化夏普比率",
                "value": round(metrics["sharpe_ratio"], 3),
                "description": "年化超额收益与年化波动率之比"
            },
            {
                "name": "最长回撤期",
                "value": round(metrics["max_drawdown"], 3),
                "description": "从最大回撤开始到恢复所需的最长时间（天）"
            },
            {
                "name": "平均持仓时间",
                "value": f"{metrics['avg_holding_period']}天",
                "description": "持仓天数占总天数的比例，反映策略的持仓效率。"
            }
        ],
        
        # 最近交易记录
        "recentTrades": [
            {
                "date": trade["date"],
                "action": trade["action"],
                "price": round(trade["price"], 3),
                "quantity": trade["size"],
                "value": round(trade["value"], 3),
                "profitLoss": round(trade["pnl"], 3),
                "totalValue": round(trade["total_value"], 3),
                "reason": trade["signal_reason"]
            }
            for trade in trades[-30:]  # 保留最近30条记录
        ],
        
        "strategyParameters": None,
        "showStrategyParameters": False
    }
    
    return result