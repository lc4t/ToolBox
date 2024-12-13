from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger
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
        benchmark_data: Optional[Dict] = None,
        benchmark_symbol: Optional[str] = None,
        risk_free_rate: float = 0.03,
    ) -> Dict:
        """计算所有性能指标"""
        metrics = {}

        # 基础指标
        metrics["latest_nav"] = final_value / initial_capital
        
        # 先计算年化收益率
        if trade_records:
            total_days = (trade_records[-1].date - trade_records[0].date).days + 1
            total_years = total_days / 365
            total_return = (final_value / initial_capital) - 1
            metrics["annual_return"] = ((1 + total_return) ** (1 / total_years) - 1) * 100
            metrics["cagr"] = metrics["annual_return"]  # 复合年化增长率
        else:
            metrics["annual_return"] = 0
            metrics["cagr"] = 0

        # 交易统计
        trades_metrics = PerformanceAnalyzer._calculate_trades_metrics(trade_records)
        metrics.update(trades_metrics)

        # 市场指标
        market_metrics = PerformanceAnalyzer._calculate_market_metrics(
            trade_records, initial_capital
        )
        metrics.update(market_metrics)

        # 风险指标（传入年化收益率）
        risk_metrics = PerformanceAnalyzer._calculate_risk_metrics(
            analyzers_results, 
            trade_records,
            metrics["annual_return"],  # 传入年化收益率
            benchmark_data=benchmark_data,
            benchmark_symbol=benchmark_symbol,
            risk_free_rate=risk_free_rate
        )
        metrics.update(risk_metrics)

        # 时间统计（使用analyzers_results中的end_date）
        time_metrics = PerformanceAnalyzer._calculate_time_metrics(
            trade_records, 
            analyzers_results.get("last_date")  # 使用回测的最后一天
        )
        metrics.update(time_metrics)

        # 年度收益率
        metrics["yearly_returns"] = PerformanceAnalyzer._calculate_yearly_returns(
            trade_records
        )

        return metrics

    @staticmethod
    def _calculate_market_metrics(trade_records: List[TradeRecord], initial_capital: float) -> Dict:
        """计算市场相关指标"""
        metrics = {}
        
        # 只计算交易频率
        if trade_records:
            start_date = trade_records[0].date
            end_date = trade_records[-1].date
            trading_days = (end_date - start_date).days + 1
            trade_frequency = len(trade_records) / trading_days * 252  # 年化交易频率
        else:
            trade_frequency = 0
        
        metrics.update({
            "trade_frequency": trade_frequency
        })
        
        return metrics

    @staticmethod
    def _calculate_risk_metrics(
        analyzers_results: Dict, 
        trade_records: List[TradeRecord],
        annual_return: float,
        benchmark_data: Optional[Dict] = None,
        benchmark_symbol: Optional[str] = None,
        risk_free_rate: float = 0.03
    ) -> Dict:
        """计算风险相关指标"""
        metrics = {}
        
        # 计算每日收益率序列
        daily_returns = []
        daily_pnl = []  # 添加每日盈亏金额序列
        dates = []
        for i in range(1, len(trade_records)):
            prev_value = trade_records[i-1].total_value
            curr_value = trade_records[i].total_value
            daily_return = (curr_value / prev_value) - 1
            daily_pnl.append(curr_value - prev_value)  # 计算每日盈亏金额
            daily_returns.append(daily_return)
            dates.append(trade_records[i].date.date())
        
        daily_returns = np.array(daily_returns)
        daily_pnl = np.array(daily_pnl)
        
        # 计算波动率相关指标
        if len(daily_returns) > 0:
            # 年化波动率 (使用对数收益率)
            log_returns = np.log(1 + daily_returns)
            volatility = np.std(log_returns) * np.sqrt(252) * 100
            
            # 计算下行波动率
            downside_returns = log_returns[log_returns < 0]
            downside_vol = np.std(downside_returns) * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
            
            # 计算最大亏损（单日最大跌幅）
            max_loss_pct = np.min(daily_returns) * 100  # 百分比形式
            max_loss_amount = np.min(daily_pnl)  # 金额形式
            
            # 计算夏普比率
            avg_return = np.mean(log_returns) * 252
            sharpe_ratio = (avg_return - risk_free_rate) / (volatility/100) if volatility > 0 else 0
            
            # 计算索提诺比率
            sortino = (avg_return - risk_free_rate) / (downside_vol/100) if downside_vol > 0 else 0
            
            # 计算SQN
            if len(trade_records) > 0:
                trades_pnl = [t.pnl for t in trade_records]
                sqn_value = np.sqrt(len(trades_pnl)) * (np.mean(trades_pnl) / np.std(trades_pnl)) if np.std(trades_pnl) > 0 else 0
            else:
                sqn_value = 0
            
            # 计算Beta和Alpha（如果有基准数据）
            beta = 0
            alpha = 0
            beta_status = "无基准数据"  # 添加状态说明
            
            if benchmark_data and dates:
                benchmark_returns = []
                strategy_returns = []
                
                matched_dates = 0
                for date, ret in zip(dates, daily_returns):
                    if date in benchmark_data:
                        benchmark_returns.append(benchmark_data[date])
                        strategy_returns.append(ret)
                        matched_dates += 1
                
                if matched_dates > 0:
                    logger.info(f"基准数据匹配率: {matched_dates}/{len(dates)} ({matched_dates/len(dates)*100:.2f}%)")
                    
                    if len(benchmark_returns) > 5:  # 降低最小数据点要求
                        benchmark_returns = np.array(benchmark_returns)
                        strategy_returns = np.array(strategy_returns)
                        
                        # 使用对数收益率计算Beta
                        log_strategy_returns = np.log(1 + strategy_returns)
                        log_benchmark_returns = np.log(1 + benchmark_returns)
                        
                        # 计算Beta
                        cov = np.cov(log_strategy_returns, log_benchmark_returns)[0][1]
                        benchmark_var = np.var(log_benchmark_returns)
                        if benchmark_var > 0:
                            beta = cov / benchmark_var
                            beta_status = f"已计算 (匹配率{matched_dates/len(dates)*100:.1f}%)"
                            # logger.info(f"Beta计算成功: {beta:.3f}")
                            
                            # 计算Alpha (使用年化收益率)
                            strategy_mean_return = np.mean(log_strategy_returns) * 252
                            benchmark_mean_return = np.mean(log_benchmark_returns) * 252
                            alpha = (strategy_mean_return - risk_free_rate) - beta * (benchmark_mean_return - risk_free_rate)
                            alpha *= 100  # 转换为百分比
                        else:
                            beta_status = "基准波动率过小"
                            logger.warning(f"基准波动率过小: {benchmark_var:.6f}")
                    else:
                        beta_status = "数据点不足"
                        logger.warning(f"匹配数据点数量不足: {len(benchmark_returns)}")
                else:
                    beta_status = "无匹配交易日"
        else:
            volatility = 0
            downside_vol = 0
            max_loss_pct = 0
            max_loss_amount = 0
            sortino = 0
            beta = 0
            alpha = 0
            sharpe_ratio = 0
            sqn_value = 0
            beta_status = "无交易数据"
        
        # 获取最大回撤
        max_dd = analyzers_results.get("drawdown", {}).get("max", {}).get("drawdown", 0)
        if max_dd > 1:  # 如果已经是百分比形式
            max_drawdown = max_dd
        else:
            max_drawdown = max_dd * 100
        
        # 计算Calmar比率
        if max_drawdown != 0:
            calmar_ratio = annual_return / max_drawdown
            # logger.info(f"Calmar比率计算: 年化收益率({annual_return:.2f}%) / 最大回撤({max_drawdown:.2f}%) = {calmar_ratio:.2f}")
        else:
            calmar_ratio = 0
            logger.warning("无法计算Calmar比率: 最大回撤为0")
        
        metrics.update({
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "current_drawdown": analyzers_results.get("drawdown", {}).get("current", {}).get("drawdown", 0) * 100,
            "calmar_ratio": calmar_ratio,
            "volatility": volatility,
            "downside_vol": downside_vol,
            "max_loss_amount": max_loss_amount,  # 最大亏损金额
            "max_loss_pct": max_loss_pct,        # 最大亏损比例
            "beta": beta,
            "alpha": alpha,
            "beta_status": beta_status,
            "vwr": analyzers_results.get("vwr", {}).get("vwr", 0),
            "sqn": sqn_value,
            "benchmark_symbol": benchmark_symbol or "000300.SS",
            "risk_free_rate": risk_free_rate,
        })
        
        return metrics

    @staticmethod
    def _calculate_trades_metrics(trade_records: List[TradeRecord]) -> Dict:
        """计算交易统计指标"""
        metrics = {}
        
        # 基础交易统计
        total_trades = len(trade_records)
        winning_trades = [t for t in trade_records if t.pnl > 0]
        losing_trades = [t for t in trade_records if t.pnl < 0]
        
        won_trades = len(winning_trades)
        lost_trades = len(losing_trades)
        
        # 计算胜率
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 计算平均盈亏
        avg_won = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_lost = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
        
        # 计算盈亏比
        profit_factor = avg_won / avg_lost if avg_lost != 0 else 0
        
        # 计算总盈亏
        total_pnl = sum(t.pnl for t in trade_records)
        
        # 计算连续交易统计
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for t in trade_records:
            if t.pnl > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_consecutive_wins = max(max_consecutive_wins, current_streak)
            elif t.pnl < 0:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
        
        # 计算持仓时间统计
        if trade_records:
            total_holding_days = sum(
                (t2.date - t1.date).days
                for t1, t2 in zip(
                    [t for t in trade_records if t.action == "BUY"],
                    [t for t in trade_records if t.action == "SELL"]
                )
            )
            avg_holding_period = total_holding_days / (total_trades / 2) if total_trades > 0 else 0
            total_days = (trade_records[-1].date - trade_records[0].date).days + 1
            holding_ratio = (total_holding_days / total_days * 100) if total_days > 0 else 0
        else:
            avg_holding_period = 0
            holding_ratio = 0
        
        metrics.update({
            "total_trades": total_trades,
            "won_trades": won_trades,
            "lost_trades": lost_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_won": avg_won,
            "avg_lost": avg_lost,
            "profit_factor": profit_factor,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "avg_holding_period": round(avg_holding_period),
            "holding_ratio": holding_ratio,
        })
        
        return metrics

    @staticmethod
    def _calculate_time_metrics(trade_records: List[TradeRecord], end_date: datetime) -> Dict:
        """计算时间相关指标"""
        if not trade_records:
            return {"running_days": 0, "start_date": None, "end_date": None}

        start_date = trade_records[0].date
        # 使用传入的end_date而不是最后一笔交易的日期
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
