from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseNotifier(ABC):
    """通知发送器基类"""

    @abstractmethod
    def send(self, message: Dict) -> None:
        """
        发送通知消息

        Args:
            message: 消息内容字典，包含 title、content 和 level
        """
        pass
