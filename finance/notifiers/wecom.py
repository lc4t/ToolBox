from typing import Any, Dict, List, Optional

import requests

from finance.logger import logger
from finance.notifiers.base import BaseNotifier


class WecomBotNotifier(BaseNotifier):
    """企业微信机器人通知发送器"""

    MAX_LENGTH = 3800  # 设置一个更保守的最大长度

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url.strip()  # 移除可能的空白字符
        logger.debug(f"初始化企业微信机器人，webhook: {self.webhook_url}")

    def _send_markdown(self, content: str, title: str = "") -> bool:
        """发送单条markdown消息"""
        message = {
            "msgtype": "markdown",
            "markdown": {
                "content": f"# {title}\n\n{content}" if title else content,
            },
        }

        # 检查消息长度
        content_length = len(message["markdown"]["content"])
        if content_length > self.MAX_LENGTH:
            logger.warning(f"消息内容过长({content_length}字符)，需要分段发送")
            parts = self._split_content(message["markdown"]["content"])
            success = True
            for part in parts:
                part_message = {
                    "msgtype": "markdown",
                    "markdown": {"content": part},
                }
                if not self._do_send(part_message):
                    success = False
            return success

        return self._do_send(message)

    def _do_send(self, message: Dict[str, Any]) -> bool:
        """执行实际的发送操作"""
        try:
            response = requests.post(
                self.webhook_url,
                json=message,
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()

            if result.get("errcode") == 0:
                logger.debug("企业微信消息发送成功")
                return True
            else:
                logger.error(f"企业微信消息发送失败: {result}")
                return False

        except Exception as e:
            logger.error(f"企业微信消息发送异常: {e}")
            return False

    def _split_content(self, content: str) -> List[str]:
        """将内容分割成多个部分，每部分不超过最大长度"""
        parts: List[str] = []
        current_part: List[str] = []
        current_length = 0

        for line in content.split("\n"):
            line_length = len(line) + 1  # +1 for newline
            if current_length + line_length > self.MAX_LENGTH:
                # 当前部分已满，保存并开始新部分
                if current_part:  # 只有当有内容时才添加
                    parts.append("\n".join(current_part))
                current_part = [line]
                current_length = line_length
            else:
                current_part.append(line)
                current_length += line_length

        if current_part:  # 添加最后一部分
            parts.append("\n".join(current_part))

        return parts

    def send(self, message: Dict[str, Any]) -> bool:
        """发送企业微信通知"""
        try:
            # 从统一消息格式中提取所需内容
            content = message["content"]
            title = message.get("title", "")  # 标题是可选的

            # 使用 markdown 格式发送
            return self._send_markdown(content, title)

        except Exception as e:
            logger.error(f"企业微信通知发送失败: {e}")
            return False
