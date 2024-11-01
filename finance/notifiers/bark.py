from typing import Dict

import requests

from finance.notifiers.base import BaseNotifier


class BarkNotifier(BaseNotifier):
    """Bark通知发送器"""

    def __init__(self, bark_url: str):
        self.bark_url = bark_url.rstrip("/")

    def send(self, message: Dict) -> None:
        """发送Bark通知"""
        url = f"{self.bark_url}/{message['title']}/{message['content']}"
        params = {
            "level": message["level"],
            "isArchive": "1",
            "group": "ETF交易信号",
        }

        try:
            response = requests.get(
                url,
                params=params,
                timeout=5,
            )
            response.raise_for_status()
            print("成功发送信号到Bark")
        except Exception as e:
            print(f"发送到Bark时出错: {e}")
