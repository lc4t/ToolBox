from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from smtplib import SMTP
from typing import Dict, List

from finance.notifiers.base import BaseNotifier


class MailNotifier(BaseNotifier):
    """邮件通知发送器"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        recipients: List[str],
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients

    def send(self, message: Dict) -> None:
        """发送邮件通知"""
        # 创建邮件
        msg = MIMEMultipart()
        msg["Subject"] = message["title"]
        msg["From"] = self.username
        msg["To"] = ", ".join(self.recipients)
        msg.attach(MIMEText(message["content"], "plain", "utf-8"))

        # 发送邮件
        try:
            with SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            print(f"成功发送信号邮件给 {', '.join(self.recipients)}")
        except Exception as e:
            print(f"发送邮件时出错: {e}")
