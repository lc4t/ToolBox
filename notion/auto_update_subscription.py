import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from loguru import logger
from notion_client import Client

load_dotenv()

notion = Client(auth=os.getenv("NOTION_TOKEN"))
# 数据库ID
database_id = os.getenv("SUBSCRIPTION_DATABASE_ID")


def update_subscriptions():
    today = datetime.now().date()
    logger.info(f"开始检查订阅更新，当前日期: {today}")

    # 查询所有记录
    results = notion.databases.query(database_id=database_id)
    logger.info(f"数据库中共有 {len(results['results'])} 条记录")

    for page in results["results"]:
        name = (
            page["properties"]["名称"]["title"][0]["plain_text"]
            if page["properties"]["名称"]["title"]
            else "未命名"
        )
        logger.debug(f"正在处理: {name}")

        # 检查状态
        status = (
            page["properties"]["状态"]["status"]["name"]
            if page["properties"]["状态"]["status"]
            else None
        )
        logger.debug(f"  状态: {status}")
        if status != "订阅中":
            logger.debug("  跳过更新: 状态不是'订阅中'")
            continue

        # 检查自动订阅
        auto_subscribe = page["properties"]["自动订阅"]["checkbox"]
        logger.debug(f"  自动订阅: {auto_subscribe}")
        if not auto_subscribe:
            logger.debug("  跳过更新: 未开启自动订阅")
            continue

        # 检查过期时间
        expiry_date_prop = page["properties"]["过期时间"]["date"]
        if not expiry_date_prop or not expiry_date_prop.get("start"):
            logger.debug("  跳过更新: 过期时间未设置")
            continue

        expiry_date = datetime.fromisoformat(expiry_date_prop["start"]).date()
        logger.debug(f"  过期时间: {expiry_date}")
        if expiry_date > today:
            logger.debug("  跳过更新: 过期时间还未到")
            continue

        # 获取周期
        cycle_prop = page["properties"]["周期"]["select"]
        if not cycle_prop:
            logger.debug("  跳过更新: 周期未设置")
            continue

        cycle = cycle_prop["name"]
        logger.debug(f"  周期: {cycle}")

        # 计算新的过期时间
        if cycle == "每月":
            new_expiry = expiry_date + timedelta(days=30)
        elif cycle == "每半年":
            new_expiry = expiry_date + timedelta(days=180)
        elif cycle == "每年":
            new_expiry = expiry_date + timedelta(days=365)
        else:
            logger.warning(f"  未知周期: {cycle}，跳过更新")
            continue

        logger.info(f"  新过期时间: {new_expiry}")

        # 更新Notion页面
        try:
            notion.pages.update(
                page_id=page["id"],
                properties={"过期时间": {"date": {"start": new_expiry.isoformat()}}},
            )
            logger.info(f"  成功更新: {name} - 新过期时间: {new_expiry}")
        except Exception as e:
            logger.error(f"  更新失败: {name} - 错误: {str(e)}")

    logger.info("订阅更新检查完成")


if __name__ == "__main__":
    update_subscriptions()
