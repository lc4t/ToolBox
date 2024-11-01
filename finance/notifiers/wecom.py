from typing import Dict

import requests

from finance.logger import logger
from finance.notifiers.base import BaseNotifier


class WecomBotNotifier(BaseNotifier):
    """企业微信机器人通知发送器"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url.strip()  # 移除可能的空白字符
        logger.debug(f"初始化企业微信机器人，webhook: {self.webhook_url}")

    def send(self, message: Dict) -> None:
        """发送企业微信通知"""
        data = {"msgtype": "markdown", "markdown": {"content": message["content"]}}

        try:
            logger.debug(f"正在发送消息到企业微信: {message['title']}")
            response = requests.post(
                self.webhook_url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            response.raise_for_status()
            logger.info(f"成功发送信号到企业微信机器人: {message['title']}")
        except Exception as e:
            logger.error(f"发送到企业微信机器人时出错: {e}")
            logger.debug(f"请求数据: {data}")
            if hasattr(e, "response") and e.response is not None:
                logger.debug(f"错误响应: {e.response.text}")
