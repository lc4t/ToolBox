import csv
import email
import html
import mailbox
import os
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup


def parse_mbox(mbox_file: str, start_date: datetime) -> Dict[str, List[str]]:
    mbox = mailbox.mbox(mbox_file)
    emails_by_date: Dict[str, List[str]] = {}

    for message in mbox:
        date = parsedate_to_datetime(message["Date"])
        if date < start_date:
            continue

        date_str = date.strftime("%Y%m%d")

        content = ""
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/html":
                    content = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    break
        else:
            content = message.get_payload(decode=True).decode("utf-8", errors="ignore")

        emails_by_date.setdefault(date_str, []).append(content)

    return emails_by_date


def extract_paper_info(html_content: str, email_date: str) -> List[Tuple[str, str, str]]:
    soup = BeautifulSoup(html_content, "html.parser")
    paper_info: List[Tuple[str, str, str]] = []

    paper_divs = soup.find_all("div", id=lambda x: x and x.startswith("stork-paper-"))

    for div in paper_divs:
        # 提取标题
        title_tag = div.find("a", href=lambda x: x and "showPaper.php" in x)
        title = title_tag.text.strip() if title_tag else ""

        # 提取DOI
        doi_span = div.find("span", string=re.compile(r"doi:"))
        doi = ""
        if doi_span:
            doi_match = re.search(r"doi:\s*(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", doi_span.text)
            if doi_match:
                doi = doi_match.group(1)

        # 只有当标题和DOI都存在时才添加到结果中
        if title and doi:
            paper_info.append((email_date, doi, title))

    return paper_info


def process_emails_to_csv(emails_by_date: Dict[str, List[str]], output_file: str):
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["日期", "DOI", "标题"])

        for date, contents in emails_by_date.items():
            for content in contents:
                paper_info = extract_paper_info(content, date)
                for paper_date, doi, title in paper_info:
                    csv_writer.writerow([paper_date, doi, title])


if __name__ == "__main__":
    mbox_file = "Takeout/邮件/前沿-Stork订阅.mbox"
    output_file = "output_papers.csv"
    start_date = datetime(2024, 4, 2, tzinfo=timezone.utc)

    emails_by_date = parse_mbox(mbox_file, start_date)
    process_emails_to_csv(emails_by_date, output_file)
    print(f"已将论文信息保存到 {output_file}")

    total_emails = sum(len(contents) for contents in emails_by_date.values())
    print(f"处理的邮件总数：{total_emails}")
    print(f"包含论文信息的日期数：{len(emails_by_date)}")
