import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from string import Template
from typing import Dict, List, Optional


class MailSender:
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        template_file: str = "finance/mail.template",
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

        # 读取邮件模板
        if not os.path.exists(template_file):
            raise FileNotFoundError(f"找不到邮件模板文件: {template_file}")

        with open(template_file, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    def send_signal(
        self,
        recipients: List[str],
        signal_data: Dict,
        holding_data: Optional[Dict] = None,
    ) -> None:
        """
        发送交易信号邮件

        Args:
            recipients: 收件人列表
            signal_data: 信号数据字典
            holding_data: 持仓数据字典（如果有持仓）
        """
        # 准备持仓详情
        if holding_data:
            holding_details = (
                f"买入时间: {holding_data['买入时间']}\n"
                f"买入价格: {holding_data['买入价格']:.3f}\n"
                f"当前收益: {holding_data['当前收益']:.2%}\n"
                f"最高价格: {holding_data['最高价格']:.3f}\n"
                f"止损价格: {holding_data['止损价格']:.3f}"
            )
        else:
            holding_details = "当前无持仓"

        # 合并数据
        email_data = {**signal_data, "holding_details": holding_details}

        # 渲染邮件内容
        content = self.template.safe_substitute(email_data)

        # 分离主题和正文
        subject, body = content.split("\n", 1)
        subject = subject.replace("Subject: ", "")

        # 创建邮件
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # 发送邮件
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            print(f"成功发送信号邮件给 {', '.join(recipients)}")
        except Exception as e:
            print(f"发送邮件时出错: {e}")
