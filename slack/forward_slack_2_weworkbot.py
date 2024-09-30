import os
import time

import requests
from dotenv import load_dotenv
from loguru import logger
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# 加载环境变量
load_dotenv()

# 从环境变量中获取Slack和企业微信的配置
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")
WEWORK_WEBHOOK_URL = os.getenv("WEWORK_WEBHOOK_URL")  # 企业微信 Webhook URL

# 初始化Slack客户端
slack_client = WebClient(token=SLACK_BOT_TOKEN)

# 配置loguru日志输出
logger.add("slack_to_wechat.log", rotation="500 MB")  # 设置日志文件，最大500MB


def fetch_latest_messages(since_ts=None):
    """
    从Slack频道获取最新的消息
    """
    logger.info("正在从Slack频道获取最新消息...")
    try:
        # 获取指定频道的最新5条消息
        response = slack_client.conversations_history(channel=SLACK_CHANNEL_ID, limit=5, oldest=since_ts)
        if response["ok"]:
            logger.info("成功获取消息。")
            return response["messages"]
        else:
            logger.error(f"获取消息时出错: {response['error']}")
            return []
    except SlackApiError as e:
        logger.error(f"Slack API 错误: {e.response['error']}")
        return []
    except Exception as e:
        logger.exception("从Slack获取消息失败。")
        return []


def post_to_wework(message):
    """
    将消息转发到企业微信
    """
    logger.info(f"正在将消息转发到企业微信: {message}")
    headers = {"Content-Type": "application/json"}
    payload = {"msgtype": "text", "text": {"content": message}}
    try:
        # 向企业微信Webhook发送POST请求
        response = requests.post(WEWORK_WEBHOOK_URL, json=payload, headers=headers)
        if response.status_code == 200:
            logger.info("消息成功转发到企业微信。")
        else:
            logger.error(f"转发消息到企业微信失败，状态码: {response.status_code}, 响应: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"网络错误: {e}")
    except Exception as e:
        logger.exception("转发消息到企业微信时发生错误。")


def main():
    """
    主函数：定期拉取Slack消息并转发到企业微信
    """
    logger.info("Slack消息转发到企业微信脚本已启动。")
    last_message_ts = str(time.time())  # 记录脚本启动时的时间戳
    while True:
        logger.debug("正在检查是否有新消息...")
        # 获取最新的消息
        messages = fetch_latest_messages(since_ts=last_message_ts)

        # 只转发新的消息
        for message in reversed(messages):
            message_ts = message.get("ts", "")
            if message_ts > last_message_ts:
                message_text = message.get("text", "")
                logger.info(f"检测到新消息: {message_text}")
                post_to_wework(message_text)
                last_message_ts = message_ts

        # 每隔10秒轮询一次
        # logger.debug("10秒后再次检查新消息。")
        time.sleep(10)


if __name__ == "__main__":
    main()
