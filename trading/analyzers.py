from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


@dataclass
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
        analyzers_results: Dict
    ) -> Dict:
        """计算所有性能指标"""
        metrics = {}
        
        # 基础指标
        metrics.update(PerformanceAnalyzer._calculate_basic_metrics(
            initial_capital, final_value, trade_records
        ))
        
        # 交易统计
        metrics.update(PerformanceAnalyzer._calculate_trade_stats(
            trade_records, analyzers_results.get('trades', {})
        ))
        
        # 风险指标
        metrics.update(PerformanceAnalyzer._calculate_risk_metrics(
            initial_capital, final_value, trade_records, analyzers_results
        ))
        
        # 时间相关指标
        metrics.update(PerformanceAnalyzer._calculate_time_metrics(trade_records))
        
        return metrics

    @staticmethod
    def _calculate_basic_metrics(
        initial_capital: float,
        final_value: float,
        trade_records: List[TradeRecord]
    ) -> Dict:
        """计算基础指标"""
        latest_nav = final_value / initial_capital
        total_pnl = sum(t.pnl for t in trade_records) if trade_records else 0
        
        return {
            "latest_nav": latest_nav,
            "total_pnl": total_pnl,
        }

    @staticmethod
    def _calculate_trade_stats(trade_records: List[TradeRecord], trade_analyzer: Dict) -> Dict:
        """计算交易统计指标"""
        # 从trade_analyzer中提取数据
        total_trades = trade_analyzer.get('total', {}).get('total', 0)
        won_trades = trade_analyzer.get('won', {}).get('total', 0)
        lost_trades = trade_analyzer.get('lost', {}).get('total', 0)
        
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
        analyzers_results: Dict
    ) -> Dict:
        """计算风险指标"""
        if trade_records:
            running_days = (trade_records[-1].date - trade_records[0].date).days + 1
            years = running_days / 365
            
            # 计算年化收益率
            annual_return = (((final_value / initial_capital) ** (1 / years) - 1) * 100) if years > 0 else 0
            
            # 计算复合年化收益率 (CAGR)
            cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        else:
            annual_return = 0
            cagr = 0
            
        # 获取回撤数据
        drawdown = analyzers_results.get('drawdown', {})
        max_drawdown = drawdown.get('max', {}).get('drawdown', 0) or 0
        current_drawdown = drawdown.get('drawdown', 0) or 0
        
        # 计算Calmar比率
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
        
        # 计算夏普比率
        sharpe = analyzers_results.get('sharpe', {}).get('sharperatio', 0) or 0
        
        # 计算VWR
        vwr = analyzers_results.get('vwr', {}).get('vwr', 0) or 0
        
        # 计算SQN
        if trade_records:
            pnl_array = np.array([t.pnl for t in trade_records])
            sqn = (np.sqrt(len(trade_records)) * (np.mean(pnl_array) / np.std(pnl_array))) if len(trade_records) > 1 and np.std(pnl_array) != 0 else 0
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
            return {
                "running_days": 0,
                "start_date": None,
                "end_date": None
            }
            
        start_date = trade_records[0].date
        end_date = trade_records[-1].date
        running_days = (end_date - start_date).days + 1
        
        return {
            "running_days": running_days,
            "start_date": start_date,
            "end_date": end_date
        } 