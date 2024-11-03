from pyzotero import zotero
from loguru import logger
import os
from dotenv import load_dotenv
import argparse
import sys

load_dotenv()

logger.remove()
logger.add(sys.stderr, level="DEBUG")

# 定义要排除的条目类型
EXCLUDED_TYPES = {"attachment", "note"}


class ZoteroTagger:
    def __init__(self):
        library_id = os.getenv("ZOTERO_LIBRARY_ID")
        api_key = os.getenv("ZOTERO_API_KEY")
        library_type = os.getenv("ZOTERO_LIBRARY_TYPE", "user")

        if not library_id or not api_key:
            raise ValueError("必须在.env文件中设置ZOTERO_LIBRARY_ID和ZOTERO_API_KEY")

        self.zot = zotero.Zotero(library_id, library_type, api_key)

    def get_all_collections(self):
        return self.zot.collections()

    def add_source_tags(self, collection_name, tag_name):
        logger.info(f"正在查找'{collection_name}'分类...")
        collection_id = self.get_collection_id(collection_name)
        if not collection_id:
            logger.error(f"未找到名为'{collection_name}'的分类")
            return

        logger.info(f"正在获取'{collection_name}'分类下的所有条目...")
        items = self.zot.everything(self.zot.collection_items(collection_id))
        logger.info(f"获取到的条目总数：{len(items)}")

        # 排除附件和笔记等类型
        filtered_items = [
            item for item in items if item["data"]["itemType"] not in EXCLUDED_TYPES
        ]
        logger.info(f"排除附件和笔记后的条目数量：{len(filtered_items)}")

        tag_to_add = {"tag": tag_name, "type": 1}
        updated_count = 0

        for index, item in enumerate(filtered_items, 1):
            current_tags = item["data"].get("tags", [])
            if tag_to_add not in current_tags:
                current_tags.append(tag_to_add)
                item["data"]["tags"] = current_tags
                self.zot.update_item(item)
                updated_count += 1
                logger.debug(
                    f"进度: {index}/{len(filtered_items)} - "
                    f"已更新: {updated_count} - "
                    f"当前项目: {item['data'].get('title', 'N/A')[:30]}..."
                )
            else:
                logger.debug(
                    f"进度: {index}/{len(filtered_items)} - "
                    f"已更新: {updated_count} - "
                    f"跳过: {item['data'].get('title', 'N/A')[:30]}... (已有标签)"
                )

        logger.info(f"更新了 {updated_count} 个条目，添加了'{tag_name}'标签")

    def batch_modify_tags(self, old_tag, new_tag):
        logger.info("正在获取所有分类...")
        collections = self.get_all_collections()
        logger.info(f"获取到的分类总数：{len(collections)}")

        total_updated_count = 0

        # 处理分类中的条目
        for collection in collections:
            collection_name = collection["data"]["name"]
            collection_key = collection["key"]
            logger.info(f"正在处理分类：{collection_name}")

            items = self.zot.everything(self.zot.collection_items(collection_key))
            logger.info(f"分类'{collection_name}'中的条目总数：{len(items)}")

            updated_count = self._process_items(items, old_tag, new_tag)
            total_updated_count += updated_count

            logger.info(f"在分类'{collection_name}'中更新了 {updated_count} 个条目")

        # 处理未分类的条目
        logger.info("正在处理未分类条目...")
        unclassified_items = self.zot.everything(self.zot.items(top=True))
        logger.info(f"未分类条目总数：{len(unclassified_items)}")

        updated_count = self._process_items(unclassified_items, old_tag, new_tag)
        total_updated_count += updated_count

        logger.info(f"在未分类条目中更新了 {updated_count} 个条目")

        logger.info(
            f"所有条目处理完毕，总共更新了 {total_updated_count} 个条目，"
            f"将标签'{old_tag}'替换为'{new_tag}'"
        )

    def _process_items(self, items, old_tag, new_tag):
        filtered_items = [
            item for item in items if item["data"]["itemType"] not in EXCLUDED_TYPES
        ]
        logger.info(f"排除附件和笔记后的条目数量：{len(filtered_items)}")

        updated_count = 0

        for index, item in enumerate(filtered_items, 1):
            current_tags = item["data"].get("tags", [])
            modified = False
            for tag in current_tags:
                if tag["tag"] == old_tag:
                    tag["tag"] = new_tag
                    modified = True

            if modified:
                self.zot.update_item(item)
                updated_count += 1
                logger.debug(
                    f"进度: {index}/{len(filtered_items)} - "
                    f"已更新: {updated_count} - "
                    f"当前项目: {item['data'].get('title', 'N/A')[:30]}..."
                )
            else:
                logger.debug(
                    f"进度: {index}/{len(filtered_items)} - "
                    f"已更新: {updated_count} - "
                    f"跳过: {item['data'].get('title', 'N/A')[:30]}... (无需修改)"
                )

        return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="为Zotero中的所有分类的学术文献添加或修改标签"
    )
    parser.add_argument(
        "action", choices=["add", "modify"], help="选择操作：添加标签或修改标签"
    )
    parser.add_argument("tag", help="要添加或修改的标签名称")
    parser.add_argument(
        "--new_tag", help="修改标签时的新标签名称（仅在 modify 操作时使用）"
    )
    parser.add_argument("--collection", help="指定要处理的分类名称（可选）")
    args = parser.parse_args()

    tagger = ZoteroTagger()
    if args.action == "add":
        if not args.collection:
            parser.error("add 操作需要提供 --collection 参数")
        tagger.add_source_tags(args.collection, args.tag)
    elif args.action == "modify":
        if not args.new_tag:
            parser.error("modify 操作需要提供 --new_tag 参数")
        tagger.batch_modify_tags(args.tag, args.new_tag)


if __name__ == "__main__":
    main()
