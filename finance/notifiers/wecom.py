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
        # 如果有标题，先发送标题
        if title:
            title_message = {
                "msgtype": "markdown",
                "markdown": {
                    "content": f"# {title}",
                },
            }
            if not self._do_send(title_message):
                return False

        # 检查消息长度
        content_length = len(content)
        if content_length > self.MAX_LENGTH:
            logger.warning(f"消息内容过长({content_length}字符)，需要分段发送")
            parts = self._split_content(content)
            success = True
            for i, part in enumerate(parts, 1):
                part_message = {
                    "msgtype": "markdown",
                    "markdown": {
                        "content": f"第{i}/{len(parts)}部分\n\n{part}",
                    },
                }
                if not self._do_send(part_message):
                    success = False
            return success

        # 内容不需要分段时直接发送
        message = {
            "msgtype": "markdown",
            "markdown": {
                "content": content,
            },
        }
        return self._do_send(message)

    def _do_send(self, message: Dict[str, Any]) -> bool:
        """执行实际的发送操作"""
        try:
            # 检查消息长度
            content_length = len(message["markdown"]["content"])
            if content_length > 4000:  # 企业微信的实际限制是4096
                logger.error(
                    f"单条消息内容过长({content_length}字符)，超过企业微信限制"
                )
                return False

            # 添加调试日志，显示完整的请求参数
            logger.debug("企业微信请求参数:")
            logger.debug(f"- URL: {self.webhook_url}")
            logger.debug(f"- 消息类型: {message['msgtype']}")
            logger.debug(f"- 消息内容:\n{message['markdown']['content']}")

            response = requests.post(
                self.webhook_url,
                json=message,
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()

            # 添加调试日志，显示完整的响应内容
            logger.debug("企业微信响应内容:")
            logger.debug(f"- 状态码: {response.status_code}")
            logger.debug(f"- 响应内容: {result}")

            if result.get("errcode") == 0:
                logger.debug("企业微信消息发送成功")
                return True
            else:
                logger.error(f"企业微信消息发送失败: {result}")
                return False

        except Exception as e:
            logger.error(f"企业微信消息发送异常: {e}")
            logger.exception("详细错误信息:")
            return False

    def _split_content(self, content: str) -> List[str]:
        """将内容分割成多个部分，每部分不超过最大长度"""
        parts: List[str] = []
        current_part: List[str] = []
        current_length = 0

        # 按章节分割内容
        sections = content.split("###")

        # 处理第一部分（如果有）
        if sections[0].strip():
            first_part = sections[0].strip()
            if len(first_part) > 3500:  # 如果第一部分就超长
                # 按段落分割
                paragraphs = first_part.split("\n\n")
                for para in paragraphs:
                    if current_length + len(para) + 2 > 3500:  # +2 for \n\n
                        if current_part:
                            parts.append("\n".join(current_part))
                        current_part = [para]
                        current_length = len(para)
                    else:
                        current_part.append(para)
                        current_length += len(para) + 2
            else:
                current_part = [first_part]
                current_length = len(first_part)

        # 处理剩余章节
        for section in sections[1:]:
            section_text = f"### {section.strip()}"
            section_length = len(section_text) + 1  # +1 for newline

            # 如果单个章节就超过限制，需要分割章节
            if section_length > 3500:
                # 先保存当前部分
                if current_part:
                    parts.append("\n".join(current_part))
                    current_part = []
                    current_length = 0

                # 分割章节内容
                section_lines = section_text.split("\n")
                for line in section_lines:
                    if current_length + len(line) + 1 > 3500:
                        if current_part:
                            parts.append("\n".join(current_part))
                        current_part = [line]
                        current_length = len(line)
                    else:
                        current_part.append(line)
                        current_length += len(line) + 1
            # 如果当前部分加上新章节会超出限制
            elif current_length + section_length > 3500:
                # 保存当前部分
                if current_part:
                    parts.append("\n".join(current_part))
                # 开始新部分
                current_part = [section_text]
                current_length = section_length
            else:
                current_part.append(section_text)
                current_length += section_length

        # 添加最后一部分
        if current_part:
            parts.append("\n".join(current_part))

        return parts

    def send(self, message: Dict[str, Any]) -> bool:
        """发送企业微信通知"""
        try:
            # 从统一消息格式中提取所需内容
            content = message["content"]
            title = message.get("title", "")  # 标题是可选的

            # 添加调试日志，显示接收到的消息内容
            logger.debug("收到的消息内容:")
            logger.debug(f"- 标题: {title}")
            logger.debug(f"- 内容:\n{content}")
            logger.debug(f"- 完整消息对象: {message}")

            # 使用 markdown 格式发送
            return self._send_markdown(content, title)

        except Exception as e:
            logger.error(f"企业微信通知发送失败: {e}")
            logger.exception("详细错误信息:")
            return False
