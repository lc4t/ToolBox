from pyzotero import zotero
from collections import Counter, defaultdict
from loguru import logger
import os
from dotenv import load_dotenv
from prettytable import PrettyTable
import re
import argparse

load_dotenv()

# 添加翻译字典
ITEM_TYPE_TRANSLATIONS = {
    "journalArticle": "期刊文章",
    "book": "书籍",
    "bookSection": "书籍章节",
    "conferencePaper": "会议论文",
    "report": "报告",
    "thesis": "论文",
    "webpage": "网页",
    "attachment": "附件",
    "note": "笔记",
    # 可以根据需要添加更多类型的翻译
}


class ZoteroManager:
    def __init__(self):
        library_id = os.getenv("ZOTERO_LIBRARY_ID")
        api_key = os.getenv("ZOTERO_API_KEY")
        library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user")

        if not library_id or not api_key:
            raise ValueError("必须在.env文件中设置ZOTERO_LIBRARY_ID和ZOTERO_API_KEY")

        self.zot = zotero.Zotero(library_id, library_type, api_key)

    def get_all_items_and_count_by_type(self):
        logger.info("正在从Zotero获取所有条目...")
        all_items = self.zot.everything(self.zot.top())
        logger.info(f"获取到的总条目数：{len(all_items)}")

        item_types = Counter(item["data"]["itemType"] for item in all_items)

        logger.info("各类型条目数量统计：")

        # 创建表格
        table = PrettyTable()
        table.field_names = ["条目类型", "数量"]
        table.align["条目类型"] = "l"
        table.align["数量"] = "r"

        for item_type, count in item_types.items():
            translated_type = ITEM_TYPE_TRANSLATIONS.get(item_type, item_type)
            table.add_row([translated_type, count])

        logger.info("\n" + str(table))

        return all_items, item_types

    def get_collections(self):
        logger.info("正在获取Zotero文库的分类...")
        collections = self.zot.collections()
        collection_names = [collection["data"]["name"] for collection in collections]
        logger.info(f"获取到的分类数量：{len(collection_names)}")
        return collection_names

    def get_articles_and_papers(self):
        logger.info("正在获取期刊文章和会议论文...")
        items = self.zot.everything(self.zot.top())

        filtered_items = [
            item
            for item in items
            if item["data"]["itemType"] in ["journalArticle", "conferencePaper"]
        ]

        logger.info(f"获取到的文章和论文总数：{len(filtered_items)}")

        # 获取文库中的分类
        collections = self.get_collections()

        # 生成标签
        categories = defaultdict(list)
        for item in filtered_items:
            title = item["data"].get("title", "").lower()
            publication = item["data"].get("publicationTitle", "").lower()
            content = title + " " + publication

            tags = set()
            for collection in collections:
                if collection.lower() in content:
                    tags.add(collection)

            if not tags:
                tags.add("未分类")

            for tag in tags:
                categories[tag].append(item)

        # 创建表格
        table = PrettyTable()
        table.field_names = ["分类", "标题", "期刊/会议名称"]
        table.align = "l"
        table.max_width = 40

        for category, items in categories.items():
            table.add_row([f"--- {category} ---", "", ""])
            for item in items:
                title = item["data"].get("title", "N/A")
                publication = item["data"].get("publicationTitle", "N/A")
                table.add_row(
                    [
                        "",
                        title[:37] + "..." if len(title) > 40 else title,
                        (
                            publication[:37] + "..."
                            if len(publication) > 40
                            else publication
                        ),
                    ]
                )

        logger.info("\n" + str(table))

        return categories

    def classify_items_by_discipline(self, collection_name=None):
        logger.info("正在获取条目...")
        if collection_name:
            collection_id = self.get_collection_id(collection_name)
            if not collection_id:
                logger.error(f"未找到名为'{collection_name}'的分类")
                return
            items = self.zot.everything(self.zot.collection_items(collection_id))
        else:
            items = self.zot.everything(self.zot.items(top=True))

        logger.info(f"获取到的条目总数：{len(items)}")

        # 排除附件、笔记和软件
        excluded_types = {"attachment", "note", "computerProgram"}
        filtered_items = [
            item for item in items if item["data"]["itemType"] not in excluded_types
        ]
        logger.info(f"排除附件、笔记和软件后的条目数量：{len(filtered_items)}")

        # 定义学科关键词
        disciplines = {
            "计算机科学": [
                "computer",
                "computing",
                "软件",
                "程序",
                "算法",
                "数据",
                "网络",
            ],
            "物理学": ["physics", "物理", "量子", "粒子", "力学", "光学"],
            "生物学": ["biology", "生物", "基因", "细胞", "生态", "进化"],
            "化学": ["chemistry", "化学", "分子", "反应", "材料"],
            "数学": ["mathematics", "数学", "代数", "几何", "统计"],
            "工程学": ["engineering", "工程", "机械", "电子", "控制"],
            "医学": ["medicine", "医学", "临床", "疾病", "治疗", "药物"],
            "心理学": ["psychology", "心理", "认知", "行为", "神经"],
            "经济学": ["economics", "经济", "金融", "市场", "投资"],
            "其他": [],  # 用于未匹配到特定学科的条目
        }

        classified_items = defaultdict(list)

        for item in filtered_items:
            title = item["data"].get("title", "").lower()
            publication = item["data"].get("publicationTitle", "").lower()
            content = title + " " + publication

            matched = False
            for discipline, keywords in disciplines.items():
                if any(keyword.lower() in content for keyword in keywords):
                    classified_items[discipline].append(item)
                    matched = True
                    break

            if not matched:
                classified_items["其他"].append(item)

        # 创建表格
        table = PrettyTable()
        table.field_names = ["学科", "数量", "示例标题"]
        table.align = "l"
        table.max_width = 40

        for discipline, items in classified_items.items():
            example_title = (
                items[0]["data"].get("title", "N/A")[:37] + "..." if items else "N/A"
            )
            table.add_row([discipline, len(items), example_title])

        logger.info("\n" + str(table))

        return classified_items

    def classify_unclassified_items(self):
        logger.info("正在获取未分类条目...")
        unclassified_items = self.zot.everything(self.zot.items(top=True))
        logger.info(f"获取到的未分类条目总数：{len(unclassified_items)}")

        # 排除附件、笔记和软件
        excluded_types = {"attachment", "note", "computerProgram"}
        filtered_items = [
            item
            for item in unclassified_items
            if item["data"]["itemType"] not in excluded_types
        ]
        logger.info(f"排除附件、笔记和软件后的未分类条目数量：{len(filtered_items)}")

        # 定义学科关键词
        disciplines = {
            "计算机科学": [
                "computer",
                "computing",
                "软件",
                "程序",
                "算法",
                "数据",
                "网络",
            ],
            "物理学": ["physics", "物理", "量子", "粒子", "力学", "光学"],
            "生物学": ["biology", "生物", "基因", "细胞", "生态", "进化"],
            "化学": ["chemistry", "化学", "分子", "反应", "材料"],
            "数学": ["mathematics", "数学", "代数", "几何", "统计"],
            "工程学": ["engineering", "工程", "机械", "电子", "控制"],
            "医学": ["medicine", "医学", "临床", "疾病", "治疗", "药物"],
            "心理学": ["psychology", "心理", "认知", "行为", "神经"],
            "经济学": ["economics", "经济", "金融", "市场", "投资"],
            "其他": [],  # 用于未匹配到特定学科的条目
        }

        classified_items = defaultdict(list)

        for item in filtered_items:
            title = item["data"].get("title", "").lower()
            publication = item["data"].get("publicationTitle", "").lower()
            content = title + " " + publication

            matched = False
            for discipline, keywords in disciplines.items():
                if any(keyword.lower() in content for keyword in keywords):
                    classified_items[discipline].append(item)
                    matched = True
                    break

            if not matched:
                classified_items["其他"].append(item)

        # 创建表格
        table = PrettyTable()
        table.field_names = ["学科", "数量", "示例标题"]
        table.align = "l"
        table.max_width = 40

        for discipline, items in classified_items.items():
            example_title = (
                items[0]["data"].get("title", "N/A")[:37] + "..." if items else "N/A"
            )
            table.add_row([discipline, len(items), example_title])

        logger.info("\n" + str(table))

        return classified_items


def main():
    parser = argparse.ArgumentParser(description="Zotero 文献管理工具")
    parser.add_argument(
        "action",
        choices=["count", "articles", "classify", "unclassified"],
        help="选择要执行的操作",
    )
    parser.add_argument("--collection", help="指定要处理的分类名称（可选）")

    args = parser.parse_args()

    manager = ZoteroManager()

    if args.action == "count":
        manager.get_all_items_and_count_by_type()
    elif args.action == "articles":
        manager.get_articles_and_papers()
    elif args.action == "classify":
        manager.classify_items_by_discipline(args.collection)
    elif args.action == "unclassified":
        manager.classify_unclassified_items()


if __name__ == "__main__":
    main()
