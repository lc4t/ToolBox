import argparse
import datetime
import json
import os
import re
import time
import traceback
import uuid
from urllib.parse import urlparse, urlunparse

import feedparser
import requests as R
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中获取数据库连接字符串和其他敏感信息
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
CUBOX_API_URL = os.getenv("CUBOX_API_URL")
OMNIVORE_AUTHORIZATION = os.getenv("OMNIVORE_AUTHORIZATION")
FEED_TOKEN = os.getenv("FEED_TOKEN")  # 新增的可选 token

engine = create_engine(DB_CONNECTION_STRING, max_overflow=500)

Base = declarative_base()


class Fetcher(Base):
    __tablename__ = "fetcher"
    id = Column(Integer, primary_key=True)
    fetch_method = Column(String(32), index=True)
    fetch_url = Column(String(256), index=True)
    webhook_method = Column(String(32), index=True)
    webhook_url = Column(String(256), index=True)
    flag = Column(String(256), index=False, default="")
    last_run = Column(DateTime, index=False, default=datetime.datetime.now)
    white_re = Column(String(256), index=False, default="")
    black_re = Column(String(256), index=False, default="")
    white_tag = Column(String(256), index=False, default="")
    black_tag = Column(String(256), index=False, default="")

    __table_args__ = (
        UniqueConstraint(
            "fetch_method",
            "fetch_url",
            "webhook_url",
            "webhook_method",
            name="fetcher_u",
        ),
    )


class BiliVideo(Base):
    __tablename__ = "bilivideo"
    id = Column(Integer, primary_key=True)
    fetcher_id = Column(Integer, index=True)
    uid = Column(Integer, index=True)
    uname = Column(String(32), index=True)
    title = Column(String(256), index=False)
    publish_time = Column(DateTime, index=False, default=datetime.datetime.fromtimestamp(int(time.time())))
    video_link = Column(String(256), index=True)
    status = Column(String(32), index=True)
    tags = Column(String(256), index=False)  # 新增字段

    __table_args__ = (UniqueConstraint("fetcher_id", "uid", "video_link", name="bilivideo_u"),)


def init_db():
    logger.info("开始初始化数据库")
    try:
        Base.metadata.create_all(engine)
        logger.info("数据库初始化成功")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")


Session = sessionmaker(bind=engine)


def ifttt_api(webhook_url, username, publish_time, title, video_url):
    headers = {
        "Content-Type": "application/json",
    }
    formatted_time = publish_time.strftime("%m%d.%H%M")
    payload = json.dumps({"value1": f"【{username}】({formatted_time})", "value2": title, "value3": video_url})
    try:
        response = R.post(webhook_url, headers=headers, data=payload, timeout=10)
        return "Congratulations" in response.text
    except Exception as e:
        logger.error(f"IFTTT API 调用失败: {e}")
        return False


def cubox_api(webhook_url, content_type, content, title, tags, description="", folder=""):
    logger.info(f"调用 Cubox API: type={content_type}, content={content}, title={title}, tags={tags}")
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "type": content_type,
        "content": content,
        "title": title,
        "tags": tags,
    }
    if description:
        payload["description"] = description
    if folder:
        payload["folder"] = folder
    response = R.post(webhook_url, json=payload).json()
    logger.info(f"Cubox API 返回: {response}")
    return response.get("code")


def get_videos(feed_url, default_author="-", use_time=True, days_limit=7):
    if FEED_TOKEN:
        feed_url += f"?key={FEED_TOKEN}"
    feed_content = R.get(feed_url).text
    feed_data = feedparser.parse(feed_content)

    videos = []
    for entry in feed_data.entries:
        if (
            use_time
            and (datetime.datetime.now() - datetime.datetime.strptime(entry["published"], "%a, %d %b %Y %H:%M:%S %Z")).total_seconds()
            > 60 * 60 * 24 * days_limit
        ):
            logger.debug(f'{days_limit}天限制, {entry["published"]}')
            continue
        logger.info(f"获取视频: feed={feed_url}, default_author={default_author}, use_time={use_time}")
        videos.append(
            {
                "uid": int(feed_url.split("/")[-1].split("?")[0]),  # 修复 int() 转换错误
                "uname": entry.get("author", default_author),
                "url": entry["link"],
                "time": datetime.datetime.strptime(entry["published"], "%a, %d %b %Y %H:%M:%S %Z"),
                "title": entry["title"],
            }
        )
    return videos


def get_bili_tags(video_id):
    url = "https://api.bilibili.com/x/tag/archive/tags"
    params = {}

    if video_id.startswith("BV"):
        params["bvid"] = video_id
    elif video_id.startswith("av"):
        params["aid"] = video_id[2:]
    else:
        raise ValueError("无效的视频ID")

    response = R.get(
        url,
        params=params,
        headers={
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "referer": "https://www.bilibili.com",
        },
    )
    time.sleep(1)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        logger.error(f"获取B站视频标签失败: {response.status_code}, 内容: {response.text}")
        return []


def omnivore_api(webhook_url, content_type, content, title, tags, description="", folder=""):
    headers = {
        "Content-Type": "application/json",
        "authorization": OMNIVORE_AUTHORIZATION,
    }
    payload = {
        "query": (
            "mutation SaveUrl($input: SaveUrlInput!) { saveUrl(input: $input) { "
            "... on SaveSuccess { url clientRequestId } ... on SaveError { errorCodes message } } }"
        ),
        "variables": {
            "input": {
                "clientRequestId": uuid.uuid4().hex,
                "source": "api",
                "url": content,
                "labels": [f"bili/{folder}"],
            }
        },
    }
    response = R.post(webhook_url, json=payload, headers=headers).json()
    logger.info(f"Omnivore API 返回: {response}")


def check_db(fetcher, video, black_tag=None, white_tag=None, download_flag=False, download_filter_status=None, download_wait_status=None):
    session = Session()
    existing_video = (
        session.query(BiliVideo)
        .filter(BiliVideo.uid == video.get("uid"))
        .filter(BiliVideo.fetcher_id == fetcher.id)
        .filter(BiliVideo.video_link == video.get("url"))
        .first()
    )

    video_id = video.get("url").split("/")[-1]

    if existing_video:
        if not existing_video.tags:
            tags = get_bili_tags(video_id=video_id)
            tag_names = [tag["tag_name"] for tag in tags]
            existing_video.tags = ",".join(tag_names)
            session.commit()
            logger.info(f"更新标签: {existing_video.video_link} {existing_video.tags}")
        logger.debug(f"已存在: {existing_video.video_link}")
        return

    tags = get_bili_tags(video_id=video_id)
    tag_names = [tag["tag_name"] for tag in tags]

    re_status = "new"
    tag_status = "new"

    # 检查白名单正则
    if fetcher.white_re and not re.match(fetcher.white_re, video.get("title")):
        logger.info(f"{video.get('title')} 不符合白名单正则 {fetcher.white_re}")
        logger.debug(f"白名单正则未命中: {fetcher.white_re}")
        re_status = "filter"

    # 检查黑名单正则
    if fetcher.black_re and re.match(fetcher.black_re, video.get("title")):
        logger.info(f"{video.get('title')} 符合黑名单正则 {fetcher.black_re}")
        logger.debug(f"黑名单正则命中: {fetcher.black_re}")
        re_status = "filter"

    # 检查白名单标签
    if fetcher.white_tag:
        white_tags = fetcher.white_tag.split(",")
        if not any(tag in tag_names for tag in white_tags):
            logger.info(f"{video.get('title')} 不符合白名单标签 {fetcher.white_tag}")
            logger.debug(f"白名单标签未命中: {fetcher.white_tag}")
            tag_status = "filter"

    # 检查黑名单标签
    if fetcher.black_tag:
        black_tags = fetcher.black_tag.split(",")
        if any(tag in tag_names for tag in black_tags):
            logger.info(f"{video.get('title')} 符合黑名单标签 {fetcher.black_tag}")
            logger.debug(f"黑名单标签命中: {fetcher.black_tag}")
            tag_status = "filter"

    # 检查命令行指定的黑名单标签
    if black_tag:
        black_tags = black_tag.split(",")
        if any(tag in tag_names for tag in black_tags):
            logger.info(f"{video.get('title')} 符合命令行指定的黑名单标签 {black_tag}")
            logger.debug(f"命令行黑名单标签命中: {black_tag}")
            tag_status = "filter"

    # 检查命令行指定的白名单标签
    if white_tag:
        white_tags = white_tag.split(",")
        if any(tag in tag_names for tag in white_tags):
            logger.info(f"{video.get('title')} 符合命令行指定的白名单标签 {white_tag}")
            logger.debug(f"命令行白名单标签命中: {white_tag}")
            tag_status = "new"

    # 如果re和tag检查都不是黑的，设置status为new
    status = "new" if re_status == "new" and tag_status == "new" else "filter"

    # 处理下载标志的特殊情况
    if download_flag:
        if status == "filter":
            status = download_filter_status or "download_filter"
        else:
            status = download_wait_status or "download_wait"

    new_session = Session()
    try:
        logger.info(f"插入新视频: {video}")
        new_video = BiliVideo(
            fetcher_id=fetcher.id,
            uid=video.get("uid"),
            uname=video.get("uname"),
            publish_time=video.get("time"),
            video_link=video.get("url"),
            title=video.get("title"),
            status=status,
            tags=",".join(tag_names),  # 记录视频的tags
        )
        new_session.add(new_video)
        new_session.commit()
        logger.info(f"插入成功: {new_video.video_link} {new_video.tags} {new_video.status} {new_video.publish_time}")
    except Exception as e:
        logger.error(f"插入失败: {e}")
        traceback.print_exc()
        new_session.rollback()
    finally:
        new_session.close()
    session.close()


def push_notify():
    session = Session()
    new_videos = session.query(BiliVideo).filter(BiliVideo.status == "new").all()
    logger.info(f"一共有{len(new_videos)}个待推送")
    for video in new_videos:
        fetcher = session.query(Fetcher).filter(Fetcher.id == video.fetcher_id).first()
        if fetcher.webhook_method == "ifttt":
            logger.info(f"推送视频: {video.video_link}")
            if "bilibili.com" in video.video_link:
                url = video.video_link.replace("https://www.bilibili.com/video/", "bilibili://video/").replace("av", "")
            logger.info(f"推送 URL: {url}")
            status = ifttt_api(
                webhook_url=fetcher.webhook_url,
                username=video.uname,
                publish_time=video.publish_time,
                title=video.title,
                video_url=url,
            )
            if status:
                video.status = "pushed"
                session.commit()
                logger.info("推送成功")
            else:
                logger.error("推送失败")
        elif fetcher.webhook_method == "cubox":
            formatted_time = video.publish_time.strftime("%m%d.%H%M")
            url = video.video_link
            status = cubox_api(
                webhook_url=CUBOX_API_URL,
                content_type="url",
                content=url,
                title=f"[{video.uname}]({formatted_time}){video.title}",
                tags=[fetcher.webhook_url],
                folder=fetcher.webhook_url,
            )
            if status == 200:
                video.status = "pushed"
                session.commit()
                logger.info("推送成功")
            else:
                logger.error(f"cubox -> {status}")
        else:
            logger.info(f"未知的 webhook 方法: {fetcher.webhook_method}")
            video.status = "pass"
            session.commit()
    session.close()


def redirect_url(url, new_host):
    parsed_url = urlparse(url)
    new_parsed_url = urlparse(new_host)
    new_url = parsed_url._replace(
        scheme=new_parsed_url.scheme,
        netloc=new_parsed_url.netloc,
    )
    return urlunparse(new_url)


def main():
    parser = argparse.ArgumentParser(description="Bili2Cubox 脚本")
    parser.add_argument("--filter", type=str, help="按关键词过滤视频")
    parser.add_argument("--fetch-count", type=int, default=3, help="要处理的 fetcher 数量")
    parser.add_argument("--all-videos", action="store_true", help="获取所有视频而不是最新的 7 天")
    parser.add_argument("--days-limit", type=int, default=7, help="获取视频的天数限制")
    parser.add_argument("--webhook-url", type=str, help="更新 fetcher 的 webhook URL")
    parser.add_argument("--exclude-webhook-url", type=str, help="排除指定 webhook URL 的 fetcher")
    parser.add_argument("--init-db", action="store_true", help="初始化或更新数据库模式")
    parser.add_argument("--redirect-host", type=str, help="重定向 webhook URL 的 host")
    parser.add_argument("--black-tag", type=str, help="命中该标签时将状态设置为filter")  # 新增参数
    parser.add_argument("--white-tag", type=str, help="命中该标签时将状态设置为new")  # 新增参数
    parser.add_argument("--download-webhook-url", type=str, help="指定下载 webhook URL")
    parser.add_argument("--download-filter-status", type=str, default="download_filter", help="下载过滤状态")
    parser.add_argument("--download-wait-status", type=str, default="download_wait", help="下载等待状态")
    args = parser.parse_args()

    if args.init_db:
        init_db()
        logger.info("数据库初始化或更新成功")
        return

    session = Session()
    query = session.query(Fetcher).filter(Fetcher.fetch_method == "rsshub")

    if args.webhook_url:
        query = query.filter(Fetcher.webhook_url == args.webhook_url)
    elif args.exclude_webhook_url:
        query = query.filter(Fetcher.webhook_url != args.exclude_webhook_url)
    elif args.download_webhook_url:
        query = query.filter(Fetcher.webhook_url == args.download_webhook_url)

    fetchers = query.order_by(Fetcher.last_run).limit(args.fetch_count).all()

    logger.info(f"一共有{len(fetchers)}个fetcher待检查")
    logger.info([f"{fetcher.flag}-{fetcher.last_run}" for fetcher in fetchers])
    for fetcher in fetchers:
        url = fetcher.fetch_url
        if args.redirect_host:
            url = redirect_url(url, args.redirect_host)
        try:
            videos = get_videos(url, fetcher.flag, not args.all_videos, args.days_limit)
        except Exception as e:
            logger.error(f"获取视频失败: {e}")
            traceback.print_exc()
            continue
        logger.info(f"{url} {fetcher.flag} RSS返回了{len(videos)}个结果")
        for video in videos:
            if args.filter and args.filter not in video["title"]:
                continue
            try:
                check_db(
                    fetcher,
                    video,
                    args.black_tag,
                    args.white_tag,
                    args.download_webhook_url is not None,  # 使用 download_webhook_url 来判断是否为下载模式
                    args.download_filter_status,
                    args.download_wait_status,
                )
            except Exception as e:
                logger.error(f"检查数据库失败: {e}")
                traceback.print_exc()
                continue
        fetcher.last_run = datetime.datetime.now()
        session.commit()
    session.close()
    push_notify()


if __name__ == "__main__":
    main()
