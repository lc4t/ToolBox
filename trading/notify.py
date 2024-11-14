import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

import requests
# 加载环境变量配置
from dotenv import load_dotenv
from jinja2 import (Environment, FileSystemLoader, PackageLoader,
                    select_autoescape)
from loguru import logger

load_dotenv()


@dataclass
class BacktestResult:
    """回测结果数据类"""

    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    won_trades: int
    lost_trades: int
    win_rate: float
    avg_win: Optional[float]
    avg_loss: Optional[float]
    gross_pnl: float
    trades: List[Dict]
    strategy_params: Dict
    metrics: Dict


class NotifyTemplate(ABC):
    """通知模板基类"""

    @abstractmethod
    def format_message(self, result: BacktestResult) -> str:
        """格式化消息"""
        pass

    @abstractmethod
    def send(self, message: str) -> bool:
        """发送消息"""
        pass


class WecomNotifyTemplate(NotifyTemplate):
    """企业微信通知模板"""

    def __init__(self):
        self.webhook_url = os.getenv("WECOM_BOT_WEBHOOK")
        if not self.webhook_url:
            raise ValueError("Missing Wecom webhook URL in environment variables")

    def format_message(self, result: BacktestResult) -> str:
        """格式化为Markdown消息"""
        message = f"""# 回测报告 - {result.symbol}

## 基本信息
- 回测周期：{result.start_date.date()} 至 {result.end_date.date()}
- 初始资金：{result.initial_capital:,.2f}
- 最终权益：{result.final_value:,.2f}
- 总收益率：{result.total_return:.2f}%

## 性能指标
- 夏普比率：{result.sharpe_ratio:.2f}
- 最大回撤：{result.max_drawdown:.2f}%
- 总交易次数：{result.total_trades}
- 胜率：{result.win_rate:.2f}%
- 平均盈利：{result.avg_win:.2f if result.avg_win else 'N/A'}
- 平均亏损：{result.avg_loss:.2f if result.avg_loss else 'N/A'}
- 总盈亏：{result.gross_pnl:.2f}

## 策略参数
"""
        for key, value in result.strategy_params.items():
            message += f"- {key}: {value}\n"

        message += "\n## 最近交易记录（最多显示5条）\n"
        for trade in result.trades[-5:]:
            message += (
                f"- {trade['date']} {trade['action']} "
                f"价格：{trade['price']:.3f} "
                f"数量：{trade['size']} "
                f"盈亏：{trade['pnl']:.2f}\n"
            )

        return message

    def send(self, message: str) -> bool:
        """发送到企业微信"""
        try:
            response = requests.post(
                self.webhook_url,
                json={"msgtype": "markdown", "markdown": {"content": message}},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Wecom notification: {e}")
            return False


class EmailNotifyTemplate(NotifyTemplate):
    """邮件通知模板"""

    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.recipients = os.getenv("EMAIL_RECIPIENTS", "").split(",")

        if not all(
            [self.smtp_server, self.smtp_username, self.smtp_password, self.recipients]
        ):
            raise ValueError("Missing email configuration in environment variables")

        # 获取模板目录的绝对路径
        current_dir = Path(__file__).parent
        template_dir = current_dir / "templates"

        # 初始化Jinja2模板环境
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def format_message(self, result: BacktestResult) -> str:
        """使用HTML模板格式化消息"""
        template = self.env.get_template("backtest_report.html")
        return template.render(result=result)

    def send(self, message: str) -> bool:
        """发送邮件"""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f'回测报告 - {datetime.now().strftime("%Y-%m-%d")}'
            msg["From"] = self.smtp_username
            msg["To"] = ", ".join(self.recipients)

            msg.attach(MIMEText(message, "html"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False


class NotifyManager:
    """通知管理器"""

    def __init__(self, notify_methods: List[str]):
        self.templates: List[NotifyTemplate] = []

        # 初始化通知模板
        for method in notify_methods:
            try:
                if method == "wecom":
                    self.templates.append(WecomNotifyTemplate())
                elif method == "email":
                    self.templates.append(EmailNotifyTemplate())
            except ValueError as e:
                logger.warning(f"Skipping {method} notification: {e}")
                continue

    def send_report(self, result: BacktestResult) -> bool:
        """发送回测报告"""
        if not self.templates:
            logger.warning("No notification templates available")
            return False

        success = True
        for template in self.templates:
            message = template.format_message(result)
            if not template.send(message):
                success = False
        return success


def create_backtest_result(results: Dict, strategy_params: Dict) -> BacktestResult:
    """从回测结果创建BacktestResult对象"""
    trade_analysis = results.get("trade_analysis", {})

    # 提取交易统计信息
    total_trades = getattr(getattr(trade_analysis, "total", None), "total", 0)
    won_trades = getattr(getattr(trade_analysis, "won", None), "total", 0)
    lost_trades = getattr(getattr(trade_analysis, "lost", None), "total", 0)

    # 计算胜率
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    # 获取平均盈亏
    avg_win = None
    if hasattr(trade_analysis, "won") and hasattr(trade_analysis.won, "pnl"):
        avg_win = trade_analysis.won.pnl.average

    avg_loss = None
    if hasattr(trade_analysis, "lost") and hasattr(trade_analysis.lost, "pnl"):
        avg_loss = trade_analysis.lost.pnl.average

    # 获取总盈亏
    gross_pnl = 0.0
    if hasattr(trade_analysis, "pnl"):
        gross_pnl = getattr(trade_analysis.pnl, "gross", {}).get("total", 0.0)

    return BacktestResult(
        symbol=strategy_params.get("symbol", "Unknown"),
        start_date=strategy_params.get("start_date"),
        end_date=strategy_params.get("end_date"),
        initial_capital=results["initial_capital"],
        final_value=results["final_value"],
        total_return=results["total_return"],
        sharpe_ratio=results.get("sharpe_ratio", 0.0),
        max_drawdown=results.get("max_drawdown", 0.0),
        total_trades=total_trades,
        won_trades=won_trades,
        lost_trades=lost_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        gross_pnl=gross_pnl,
        trades=results.get("trades", []),
        strategy_params=strategy_params,
        metrics=results.get("metrics", {}),
    )
