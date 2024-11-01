import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

from finance.notifiers.bark import BarkNotifier
from finance.notifiers.base import BaseNotifier
from finance.notifiers.mail import MailNotifier
from finance.notifiers.wecom import WecomBotNotifier

# 加载.env文件
load_dotenv()

# SMTP配置
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# ETF代码和名称的映射
ETF_NAMES = {
    "512800": "银行ETF",
    "513380": "恒生科技ETF龙头",
    "159915": "创业板ETF",
    "510300": "300ETF",
    "159929": "医药ETF",
    "159920": "恒生ETF",
    "560510": "中证A500ETF",
    "159941": "纳指ETF",
}


def get_etf_name(code: str) -> str:
    """获取ETF的中文名称"""
    # 移除可能的市场后缀
    code = code.split(".")[0]
    return ETF_NAMES.get(code, code)  # 如果找不到对应的名称，返回代码本身


def get_notifiers(
    notifier_types: List[str],
    email_recipients: List[str] = None,
    wecom_webhook: str = None,
    bark_url: str = None,
) -> Dict[str, BaseNotifier]:
    """
    获取多个通知发送器实例

    Args:
        notifier_types: 通知类型列表，支持 'mail'、'wecom' 和 'bark'
        email_recipients: 邮件接收者列表
        wecom_webhook: 企业微信机器人webhook地址
        bark_url: Bark推送URL

    Returns:
        通知发送器字典，key为通知类型，value为对应的通知器实例
    """
    notifiers = {}

    for notifier_type in notifier_types:
        if notifier_type == "mail":
            if (
                all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD])
                and email_recipients
            ):
                notifiers["mail"] = MailNotifier(
                    smtp_server=SMTP_SERVER,
                    smtp_port=SMTP_PORT,
                    username=SMTP_USERNAME,
                    password=SMTP_PASSWORD,
                    recipients=email_recipients,
                )
            else:
                print("警告：SMTP配置不完整或未指定收件人，邮件功能将被禁用")

        elif notifier_type == "wecom":
            if wecom_webhook:
                notifiers["wecom"] = WecomBotNotifier(webhook_url=wecom_webhook)
            else:
                print("警告：未指定企业微信机器人Webhook，机器人通知功能将被禁用")

        elif notifier_type == "bark":
            if bark_url:
                notifiers["bark"] = BarkNotifier(bark_url=bark_url)
            else:
                print("警告：未指定Bark URL，Bark通知功能将被禁用")

        else:
            print(f"警告：不支持的通知类型 {notifier_type}")

    return notifiers


def get_notifier(
    notifier_type: str = "mail",
    email_recipients: List[str] = None,
    wecom_webhook: str = None,
    bark_url: str = None,
) -> Optional[BaseNotifier]:
    """
    获取单个通知发送器实例（保持向后兼容）

    Args:
        notifier_type: 通知类型，支持 'mail'、'wecom' 和 'bark'
        email_recipients: 邮件接收者列表
        wecom_webhook: 企业微信机器人webhook地址
        bark_url: Bark推送URL

    Returns:
        通知发送器实例或 None
    """
    notifiers = get_notifiers(
        [notifier_type],
        email_recipients,
        wecom_webhook,
        bark_url,
    )
    return notifiers.get(notifier_type)
