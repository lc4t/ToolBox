import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Any
from db import DBClient
import requests
import time

# 加载环境变量配置
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape
from loguru import logger

load_dotenv()


@dataclass
class BacktestResult:
    """回测结果数据类"""

    symbol: str
    symbol_info: Dict
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
    next_signal: Dict


class NotifyTemplate(ABC):
    """通知模板基类"""

    @abstractmethod
    def format_message(self, result: BacktestResult) -> str:
        """格式化消息"""
        pass

    @abstractmethod
    def send(self, message: str, result: BacktestResult) -> bool:
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
            trade_date = (
                trade["date"].strftime("%Y-%m-%d")
                if isinstance(trade["date"], datetime)
                else trade["date"]
            )
            message += (
                f"- {trade_date} {trade['action']} "
                f"价格：{trade['price']:.3f} "
                f"数量：{trade['size']} "
                f"盈亏：{trade['pnl']:.2f}\n"
            )

        return message

    def send(self, message: str, result: BacktestResult) -> bool:
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


class EmailNotifier:
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.max_retries = 1
        self.retry_delay = 5  # 秒

    def send(self, to_addrs: List[str], subject: str, content: str) -> bool:
        """发送邮件，带重试机制"""
        success = False  # 添加成功标志
        for attempt in range(self.max_retries):
            try:
                # 创建SMTP连接
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                    # 登录
                    server.login(self.username, self.password)

                    # 创建邮件
                    msg = MIMEMultipart("alternative")  # 修改这里，使用 alternative
                    msg["From"] = self.username
                    msg["To"] = ", ".join(to_addrs)
                    msg["Subject"] = subject

                    # 添加 HTML 内容
                    msg.attach(
                        MIMEText(content, "html", "utf-8")
                    )  # 修改这里，指定为 html

                    # 发送邮件
                    server.send_message(msg)

                    success = True  # 标记发送成功
                    logger.info(f"Successfully sent email to {to_addrs}")
                    break  # 发送成功后跳出重试循环

            except smtplib.SMTPServerDisconnected as e:
                logger.warning(
                    f"SMTP Server disconnected (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue

            except smtplib.SMTPException as e:
                logger.error(
                    f"SMTP error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue

            except Exception as e:
                logger.error(f"Unexpected error while sending email: {e}")
                return False

        if not success:
            logger.error(f"Failed to send email after {self.max_retries} attempts")

        return success


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
        # 处理交易记录中的日期格式
        trades_with_formatted_date = []
        for trade in result.trades:
            trade_copy = trade.copy()
            if isinstance(trade["date"], datetime):
                trade_copy["date"] = trade["date"].date()  # 只保留日期部分
            elif (
                isinstance(trade["date"], str) and len(trade["date"]) > 10
            ):  # 如果是带时间的字符串
                trade_copy["date"] = trade["date"][:10]  # 只保留 YYYY-MM-DD 部分
            trades_with_formatted_date.append(trade_copy)

        # 创建一个新的结果对象，避免修改原始数据
        formatted_result = BacktestResult(
            symbol=result.symbol,
            symbol_info=result.symbol_info,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_value=result.final_value,
            total_return=result.total_return,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            total_trades=result.total_trades,
            won_trades=result.won_trades,
            lost_trades=result.lost_trades,
            win_rate=result.win_rate,
            avg_win=result.avg_win,
            avg_loss=result.avg_loss,
            gross_pnl=result.gross_pnl,
            trades=trades_with_formatted_date,  # 使用格式化后的交易记录
            strategy_params=result.strategy_params,
            metrics=result.metrics,
            next_signal=result.next_signal,
        )

        template = self.env.get_template("backtest_report.html")
        return template.render(result=formatted_result)

    def send(self, message: str, result: BacktestResult) -> bool:
        """发送邮件"""
        try:
            msg = MIMEMultipart("alternative")

            # 获取股票名称
            symbol_info = self._get_symbol_info(result.symbol)
            name = symbol_info.get("name", "")

            # 获取最后交易日期
            last_date = result.end_date.strftime("%Y-%m-%d")

            # 构建邮件标题
            signal_text = f"【{result.next_signal['action']}】"
            msg["Subject"] = f"{signal_text}{name}({result.symbol})-{last_date}"

            msg["From"] = self.smtp_username
            msg["To"] = ", ".join(self.recipients)
            msg.attach(MIMEText(message, "html"))

            notifier = EmailNotifier(
                smtp_server=self.smtp_server,
                smtp_port=self.smtp_port,
                username=self.smtp_username,
                password=self.smtp_password,
            )

            return notifier.send(self.recipients, msg["Subject"], message)
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    def _get_symbol_info(self, symbol: str) -> Dict:
        """从数据库获取股票信息"""
        try:
            db_client = DBClient()
            return db_client.get_symbol_info(symbol) or {}
        except Exception as e:
            logger.error(f"Failed to get symbol info: {e}")
            return {}


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
            if not template.send(message, result):
                success = False
        return success

    def get_message_preview(self, result: BacktestResult) -> str:
        """生成消息预览"""
        if not self.templates:
            return "No notification templates available"

        # 优先使用邮件模板，因为它包含更详细的信息
        for template in self.templates:
            if isinstance(template, EmailNotifyTemplate):
                return template.format_message(result)

        # 如果没有邮件模板，使用第一个可用的模板
        return self.templates[0].format_message(result)

    def _get_template(self, template_type: str) -> Optional[NotifyTemplate]:
        """取指定类型的模板"""
        for template in self.templates:
            if (
                template_type == "email" and isinstance(template, EmailNotifyTemplate)
            ) or (
                template_type == "wecom" and isinstance(template, WecomNotifyTemplate)
            ):
                return template
        return None


def create_backtest_result(results: Dict, strategy_params: Dict) -> BacktestResult:
    """从回测结果创建BacktestResult对象"""
    metrics = results.get("metrics", {})

    # 获取股票信息
    try:
        db_client = DBClient()
        symbol_info = db_client.get_symbol_info(strategy_params.get("symbol", "")) or {}
    except Exception as e:
        logger.error(f"Failed to get symbol info: {e}")
        symbol_info = {}

    return BacktestResult(
        symbol=strategy_params.get("symbol", "Unknown"),
        symbol_info=symbol_info,
        start_date=strategy_params.get("start_date"),
        end_date=results.get("last_trade_date"),
        initial_capital=results["initial_capital"],
        final_value=results["final_value"],
        total_return=results["total_return"],
        sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
        max_drawdown=metrics.get("max_drawdown", 0.0),
        total_trades=metrics.get("total_trades", 0),
        won_trades=metrics.get("won_trades", 0),
        lost_trades=metrics.get("lost_trades", 0),
        win_rate=metrics.get("win_rate", 0.0),
        avg_win=metrics.get("avg_won"),
        avg_loss=metrics.get("avg_lost"),
        gross_pnl=metrics.get("total_pnl", 0.0),
        trades=results.get("trades", []),
        strategy_params=strategy_params,
        metrics=metrics,
        next_signal=results.get(
            "next_signal", {"action": "观察", "conditions": [], "stop_loss": None}
        ),
    )
