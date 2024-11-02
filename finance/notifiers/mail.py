from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from smtplib import SMTP
from typing import Any, Dict, List

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

    def send(self, message: Dict[str, Any]) -> bool:
        """发送邮件通知"""
        try:
            title = message["title"]
            content = message["content"]

            # 创建邮件
            msg = MIMEMultipart("alternative")  # 使用 alternative 类型
            msg["Subject"] = title
            msg["From"] = self.username
            msg["To"] = ", ".join(self.recipients)

            # 根据消息内容类型添加正文
            if "<html>" in content:  # HTML 格式
                msg.attach(MIMEText(content, "html", "utf-8"))
            else:  # 纯文本或 markdown 格式
                msg.attach(MIMEText(content, "plain", "utf-8"))

            # 发送邮件
            with SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            print(f"成功发送信号邮件给 {', '.join(self.recipients)}")
            return True
        except Exception as e:
            print(f"发送邮件时出错: {e}")
            return False
