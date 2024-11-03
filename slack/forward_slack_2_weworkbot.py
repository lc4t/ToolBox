import argparse
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


def fetch_latest_messages(since_ts=None, message_count=5):
    """
    从Slack频道获取最新的消息
    """
    logger.info(f"获取消息，since_ts: {since_ts}, message_count: {message_count}")
    try:
        # 获取指定频道的最新消息
        response = slack_client.conversations_history(
            channel=SLACK_CHANNEL_ID, limit=message_count, oldest=since_ts, inclusive=False  # 确保不包括 since_ts 对应的消息
        )

        if response["ok"]:
            messages = response["messages"]
            logger.info(f"成功获取 {len(messages)} 条消息。")
            return messages
        else:
            logger.error(f"获取消息时出错: {response['error']}")
            return []
    except SlackApiError as e:
        logger.error(f"Slack API 错误: {e.response['error']}")
        return []
    except Exception:
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


def main(message_count=None):
    """
    主函数：定期拉取Slack消息并转发到企业微信
    """
    logger.info("Slack消息转发到企业微信脚本已启动。")
    last_message_ts = None  # 记录最后一条处理过的消息时间戳

    # 初始化时，如果message_count > 0，则获取最新的message_count条历史消息
    if message_count is not None and message_count > 0:
        logger.info(f"获取最新的 {message_count} 条历史消息...")
        messages = fetch_latest_messages(message_count=message_count)
        if messages:
            last_message_ts = messages[0].get("ts", "")
            for message in messages:  # 不需要reversed，因为messages已经是倒序的
                process_and_forward_message(message)
    else:
        logger.info("不获取历史消息，只监听新消息。")
        # 获取最新的一条消息的时间戳，但不转发
        latest_messages = fetch_latest_messages(message_count=1)
        if latest_messages:
            last_message_ts = latest_messages[0].get("ts", "")
        else:
            last_message_ts = str(time.time())  # 如果没有获取到消息，使用当前时间

    while True:
        logger.debug("正在检查是否有新消息...")
        # 获取最新的消息，限制每次最多获取 100 条
        new_messages = fetch_latest_messages(since_ts=last_message_ts, message_count=100)

        if new_messages:
            # 更新 last_message_ts 为最新消息的时间戳
            last_message_ts = new_messages[0].get("ts", last_message_ts)
            logger.info(f"更新 last_message_ts 为: {last_message_ts}")

            # 处理新消息
            for message in new_messages:  # 不需要reversed，因为new_messages已经是倒序的
                process_and_forward_message(message)
        else:
            # 如果没有新消息，也更新 last_message_ts
            current_time = time.time()
            last_message_ts = str(current_time)
            logger.info(f"没有新消息，更新 last_message_ts 为当前时间: {last_message_ts}")

        # 每隔10秒轮询一次
        logger.info("10秒后再次检查新消息。")
        time.sleep(10)


def process_and_forward_message(message):
    # 输出消息的原始内容
    logger.info(f"原始消息内容: {message}")

    # 获取消息文本
    message_text = message.get("text", "")

    # 初始化 issue_title
    issue_title = ""

    # 处理消息中的块结构（如果存在）
    blocks_to_check = []
    if "attachments" in message:
        for attachment in message["attachments"]:
            if "blocks" in attachment:
                blocks_to_check.extend(attachment["blocks"])
    elif "blocks" in message:
        blocks_to_check = message["blocks"]

    for block in blocks_to_check:
        if block["type"] == "section" and "text" in block:
            block_text = block["text"].get("text", "")
            if "linear.app" in block_text and "|" in block_text:
                # 提取 issue 标题
                issue_title = block_text.split("|")[1].strip(">")
                break

    # 组合最终消息
    final_message = f"{message_text}\n{issue_title}".strip()

    logger.info(f"处理后的消息文本: {final_message}")
    post_to_wework(final_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从Slack获取消息并转发到企业微信")
    parser.add_argument("--message_count", type=int, default=0, help="要获取的历史消息条目数")
    args = parser.parse_args()
    main(message_count=args.message_count)
