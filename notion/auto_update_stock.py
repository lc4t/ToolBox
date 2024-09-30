import os
from datetime import datetime

import requests
import xalpha as xa
from dotenv import load_dotenv
from loguru import logger
from notion_client import Client

# 加载环境变量
load_dotenv()

# 初始化 Notion 客户端
notion = Client(auth=os.getenv("NOTION_TOKEN"))


# 获取状态为指定值的行
def get_rows_with_status(status: str):
    database_id = os.getenv("STOCK_DATABASE_ID")
    response = notion.databases.query(database_id=database_id, filter={"property": "状态", "select": {"equals": status}})
    return response.get("results", [])


# 查询当前净值
def query_current_value(asset_type: str, asset_code: str):
    if asset_type in ["A股"]:
        api_url = "http://web.juhe.cn/finance/stock/hs"
        params = {"gid": asset_code, "key": os.getenv("JUHE_API_KEY")}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.get(api_url, params=params, headers=headers)
        data = response.json()
        if data["resultcode"] == "200":
            current_value = data["result"][0]["data"]["lastestpri"]
        else:
            raise ValueError(f"无法获取股票代码 {asset_code} 的当前价格: {data['reason']}")

    elif asset_type == "港股":
        api_url = "http://web.juhe.cn/finance/stock/hk"
        params = {"num": asset_code, "key": os.getenv("JUHE_API_KEY")}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.get(api_url, params=params, headers=headers)
        data = response.json()
        if data["resultcode"] == "200":
            current_value = data["result"][0]["data"]["lastestpri"]
        else:
            raise ValueError(f"无法获取港股代码 {asset_code} 的当前价格: {data['reason']}")

    elif asset_type in ["基金", "债券"]:
        fund = xa.fundinfo(asset_code)
        current_value = fund.price.iloc[-1]["netvalue"]
    else:
        raise ValueError(f"未知的资产类型: {asset_type}")

    if current_value is None:
        raise ValueError(f"无法获取资产代码 {asset_code} 的当前净值")

    return current_value


# 更新股票或基金的净值
def update_stock_values():
    holding_rows = get_rows_with_status("持有中")

    for row in holding_rows:
        asset_type = row["properties"]["资产类型"]["select"]["name"]
        asset_code = row["properties"]["标的代码"]["rich_text"][0]["text"]["content"]

        current_value = query_current_value(asset_type, asset_code)
        # current_value = 1.2
        logger.info(f"当前净值: {current_value}")

        # 只更新必要的属性
        properties_to_update = {
            "净值": {"type": "number", "number": float(current_value)},
            "净值更新时间": {"type": "date", "date": {"start": datetime.now().isoformat()}},
        }

        # 如果 "统计收益" 属性存在，我们需要保持其原有值
        if "统计收益" in row["properties"]:
            original_stat = row["properties"]["统计收益"]
            if "number" in original_stat:
                properties_to_update["统计收益"] = {"type": "number", "number": original_stat["number"]}

        try:
            notion.pages.update(page_id=row["id"], properties=properties_to_update)
            logger.info(f"已更新行 {row['id']} 的净值为: {current_value}")
        except Exception as e:
            logger.error(f"更新行 {row['id']} 时出错: {str(e)}")
            logger.error(f"properties_to_update: {properties_to_update}")


if __name__ == "__main__":
    update_stock_values()
