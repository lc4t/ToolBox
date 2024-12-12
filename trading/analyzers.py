from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


class TradeRecord:
    """交易记录数据类"""

    date: datetime
    action: str
    price: float
    size: int
    value: float
    commission: float
    pnl: float
    total_value: float
    signal_reason: str
    cash: float


class PerformanceAnalyzer:
    """性能分析器，用于计算各种性能指标"""

    @staticmethod
    def calculate_metrics(
        initial_capital: float,
        final_value: float,
        trade_records: List[TradeRecord],
        analyzers_results: Dict,
    ) -> Dict:
        """计算所有性能指标"""
        metrics = {}

        # 基础指标
        metrics.update(
            PerformanceAnalyzer._calculate_basic_metrics(
                initial_capital, final_value, trade_records
            )
        )

        # 交易统计
        metrics.update(
            PerformanceAnalyzer._calculate_trade_stats(
                trade_records, analyzers_results.get("trades", {})
            )
        )

        # 风险指标
        metrics.update(
            PerformanceAnalyzer._calculate_risk_metrics(
                initial_capital, final_value, trade_records, analyzers_results
            )
        )

        # 时间相关指标
        metrics.update(PerformanceAnalyzer._calculate_time_metrics(trade_records))

        # 计算最大连续盈亏次数
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        last_pnl = None

        for trade in trade_records:
            if trade.action == "SELL":  # 只在卖出时计算
                if trade.pnl > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                elif trade.pnl < 0:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(
                        max_consecutive_losses, consecutive_losses
                    )

        metrics["max_consecutive_wins"] = max_consecutive_wins
        metrics["max_consecutive_losses"] = max_consecutive_losses

        # 计算平均持仓时间
        holding_periods = []
        buy_date = None

        for trade in trade_records:
            if trade.action == "BUY":
                buy_date = trade.date
            elif trade.action == "SELL" and buy_date:
                holding_period = (trade.date - buy_date).days
                holding_periods.append(holding_period)
                buy_date = None

        metrics["avg_holding_period"] = (
            f"{int(np.mean(holding_periods))}" if holding_periods else "0"
        )

        # 计算持仓/空仓比例
        total_days = (
            (trade_records[-1].date - trade_records[0].date).days
            if trade_records
            else 0
        )
        holding_days = sum(holding_periods)
        holding_ratio = (holding_days / total_days * 100) if total_days > 0 else 0
        metrics["holding_ratio"] = holding_ratio  # 持仓天数占比

        # 更新日期使用回测的最后一个交易日
        metrics["update_date"] = analyzers_results.get(
            "last_date"
        )  # 从analyzers_results中获取最后交易日

        # 添加年度收益率
        metrics["yearly_returns"] = PerformanceAnalyzer._calculate_yearly_returns(
            trade_records
        )

        return metrics

    @staticmethod
    def _calculate_basic_metrics(
        initial_capital: float, final_value: float, trade_records: List[TradeRecord]
    ) -> Dict:
        """计算基础指标"""
        latest_nav = final_value / initial_capital
        total_pnl = sum(t.pnl for t in trade_records) if trade_records else 0

        return {
            "latest_nav": latest_nav,
            "total_pnl": total_pnl,
        }

    @staticmethod
    def _calculate_trade_stats(
        trade_records: List[TradeRecord], trade_analyzer: Dict
    ) -> Dict:
        """计算交易统计指标"""
        # 从trade_analyzer中提取数据
        total_trades = trade_analyzer.get("total", {}).get("total", 0)
        won_trades = trade_analyzer.get("won", {}).get("total", 0)
        lost_trades = trade_analyzer.get("lost", {}).get("total", 0)

        # 计算胜率
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

        # 计算平均盈亏
        winning_trades = [t for t in trade_records if t.pnl > 0]
        losing_trades = [t for t in trade_records if t.pnl < 0]

        avg_won = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_lost = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0

        # 计算盈亏比
        profit_factor = avg_won / avg_lost if avg_lost != 0 else 0

        return {
            "total_trades": total_trades,
            "won_trades": won_trades,
            "lost_trades": lost_trades,
            "win_rate": win_rate,
            "avg_won": avg_won,
            "avg_lost": avg_lost,
            "profit_factor": profit_factor,
        }

    @staticmethod
    def _calculate_risk_metrics(
        initial_capital: float,
        final_value: float,
        trade_records: List[TradeRecord],
        analyzers_results: Dict,
    ) -> Dict:
        """计算风险指标"""
        if trade_records:
            running_days = (trade_records[-1].date - trade_records[0].date).days + 1
            years = running_days / 365

            # 计算年化收益率
            annual_return = (
                (((final_value / initial_capital) ** (1 / years) - 1) * 100)
                if years > 0
                else 0
            )

            # 计算复合年化收益率 (CAGR)
            cagr = (
                ((final_value / initial_capital) ** (1 / years) - 1) * 100
                if years > 0
                else 0
            )
        else:
            annual_return = 0
            cagr = 0

        # 获取回撤数据
        drawdown = analyzers_results.get("drawdown", {})
        max_drawdown = drawdown.get("max", {}).get("drawdown", 0) or 0
        current_drawdown = drawdown.get("drawdown", 0) or 0

        # 计算Calmar比率
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0

        # 计算夏普比率
        sharpe = analyzers_results.get("sharpe", {}).get("sharperatio", 0) or 0

        # 计算VWR
        vwr = analyzers_results.get("vwr", {}).get("vwr", 0) or 0

        # 计算SQN
        if trade_records:
            pnl_array = np.array([t.pnl for t in trade_records])
            sqn = (
                (np.sqrt(len(trade_records)) * (np.mean(pnl_array) / np.std(pnl_array)))
                if len(trade_records) > 1 and np.std(pnl_array) != 0
                else 0
            )
        else:
            sqn = 0

        return {
            "annual_return": annual_return,
            "cagr": cagr,
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "calmar_ratio": calmar_ratio,
            "sharpe_ratio": sharpe,
            "vwr": vwr,
            "sqn": sqn,
        }

    @staticmethod
    def _calculate_time_metrics(trade_records: List[TradeRecord]) -> Dict:
        """计算时间相关指标"""
        if not trade_records:
            return {"running_days": 0, "start_date": None, "end_date": None}

        start_date = trade_records[0].date
        end_date = trade_records[-1].date
        running_days = (end_date - start_date).days + 1

        return {
            "running_days": running_days,
            "start_date": start_date,
            "end_date": end_date,
        }

    @staticmethod
    def _calculate_yearly_returns(trade_records: List[TradeRecord]) -> Dict[str, float]:
        """计算每年的收益率"""
        if not trade_records:
            return {}

        # 按年份分组交易记录
        yearly_trades = {}
        for trade in trade_records:
            year = trade.date.year
            if year not in yearly_trades:
                yearly_trades[year] = []
            yearly_trades[year].append(trade)

        # 计算每年的收益率
        yearly_returns = {}
        for year, trades in yearly_trades.items():
            # 获取该年第一笔交易的起始资金和最后一笔交易的最终资金
            start_value = (
                trades[0].total_value - trades[0].pnl
            )  # 减去第一笔交易的盈亏得到年初资金
            end_value = trades[-1].total_value

            # 计算年度收益率
            yearly_return = ((end_value / start_value) - 1) * 100
            yearly_returns[str(year)] = yearly_return

        return yearly_returns

    @staticmethod
    def _calculate_market_metrics(trade_records: List[TradeRecord], initial_capital: float) -> Dict:
        """计算市场指标"""
        total_buy_value = sum(t.value for t in trade_records if t.action == "BUY")
        total_sell_value = sum(t.value for t in trade_records if t.action == "SELL")
        total_trade_value = total_buy_value + total_sell_value
        
        # 计算交易天数
        trading_days = len(set(t.date.date() for t in trade_records))
        
        return {
            "total_trade_value": total_trade_value,
            "total_buy_value": total_buy_value,
            "total_sell_value": total_sell_value,
            "turnover_rate": (total_trade_value / initial_capital) * 100,
            "avg_trade_value": total_trade_value / len(trade_records) if trade_records else 0,
            "trade_frequency": len(trade_records) / trading_days if trading_days > 0 else 0,
        }

    @staticmethod
    def _calculate_risk_metrics_extended(returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> Dict:
        """计算扩展的风险指标"""
        if len(returns) < 2:
            return {}
        
        # 计算波动率
        volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
        
        # 计算下行波动率（用于索提诺比率）
        downside_returns = returns[returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # 计算无风险利率（假设为3%）
        risk_free_rate = 0.03
        
        # 计算索提诺比率
        excess_return = np.mean(returns) * 252 - risk_free_rate
        sortino_ratio = excess_return / downside_volatility if downside_volatility != 0 else 0
        
        metrics = {
            "volatility": volatility * 100,  # 转换为百分比
            "sortino_ratio": sortino_ratio,
            "max_loss": min(returns) * 100,  # 最大单日亏损
        }
        
        # 如果有基准收益率，计算相对指标
        if benchmark_returns is not None:
            # 计算Beta
            covariance = np.cov(returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # 计算Alpha
            benchmark_return = np.mean(benchmark_returns) * 252
            portfolio_return = np.mean(returns) * 252
            alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
            
            # 计算信息比率
            tracking_error = np.std(returns - benchmark_returns) * np.sqrt(252)
            information_ratio = (portfolio_return - benchmark_return) / tracking_error if tracking_error != 0 else 0
            
            metrics.update({
                "beta": beta,
                "alpha": alpha * 100,  # 转换为百分比
                "information_ratio": information_ratio,
            })
        
        return metrics
