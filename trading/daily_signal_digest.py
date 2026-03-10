import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib import error, request
from zoneinfo import ZoneInfo

from datetime import date, datetime


DEFAULT_PUSHGO_URL = "https://gateway.pushgo.dev/push"
DEFAULT_TIMEZONE = "Asia/Shanghai"
TRADE_ACTIONS = {"买入", "卖出"}
ACTION_PRIORITY = {"买入": 0, "卖出": 1, "持有": 2, "观望": 3}
ACTION_ALIASES = {
    "BUY": "买入",
    "SELL": "卖出",
    "HOLD": "持有",
    "WATCH": "观望",
    "OBSERVE": "观望",
    "观察": "观望",
}


@dataclass(frozen=True)
class ReportRecord:
    source_path: Path
    raw: dict
    action: str
    symbol: str
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总当日交易信号并推送到 PushGo")
    parser.add_argument(
        "--data-dir",
        default="frontend/trading/data",
        help="JSON 数据目录，默认 frontend/trading/data",
    )
    parser.add_argument(
        "--date",
        dest="target_date",
        type=parse_date,
        help="目标日期，格式 YYYY-MM-DD，默认 Asia/Shanghai 当天",
    )
    parser.add_argument(
        "--pushgo-url",
        default=None,
        help="PushGo 推送地址，默认读取 PUSHGO_URL 或使用官方网关",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="PushGo 最大重试次数，默认 3",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="PushGo 重试间隔秒数，默认 5",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出标题和正文，不执行推送",
    )
    return parser.parse_args()


def load_dotenv_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无效日期: {value}") from exc


def today_in_timezone(timezone_name: str = DEFAULT_TIMEZONE) -> date:
    return datetime.now(ZoneInfo(timezone_name)).date()


def normalize_action(action: str | None) -> str:
    if not action:
        return "观望"
    action = str(action).strip()
    return ACTION_ALIASES.get(action.upper(), ACTION_ALIASES.get(action, action))


def load_reports(data_dir: Path, target_date: date) -> tuple[list[ReportRecord], list[str], str | None]:
    if not data_dir.exists():
        return [], [f"数据目录不存在: {data_dir}"], "missing_dir"

    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        return [], [], "no_json"

    warnings: list[str] = []
    reports: list[ReportRecord] = []
    target_date_str = target_date.isoformat()

    for json_file in json_files:
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"跳过损坏 JSON: {json_file} ({exc})")
            continue

        if not isinstance(raw, dict):
            warnings.append(f"跳过非对象 JSON: {json_file}")
            continue

        report_date = raw.get("reportDate")
        if report_date != target_date_str:
            continue

        symbol = str(raw.get("symbol") or "").strip()
        latest_signal = raw.get("latestSignal")

        if not symbol or not isinstance(latest_signal, dict):
            warnings.append(f"跳过缺少关键字段的 JSON: {json_file}")
            continue

        action = normalize_action(latest_signal.get("action"))
        name = str(raw.get("name") or symbol).strip() or symbol

        reports.append(
            ReportRecord(
                source_path=json_file,
                raw=raw,
                action=action,
                symbol=symbol,
                name=name,
            )
        )

    reason = None if reports else "no_matching_reports"
    reports.sort(key=lambda item: (ACTION_PRIORITY.get(item.action, 99), item.symbol))
    return reports, warnings, reason


def extract_latest_price(report: dict, fallback_action: str | None = None) -> tuple[str, str]:
    latest_signal = report.get("latestSignal") or {}
    timestamp = str(latest_signal.get("timestamp") or "暂无")
    prices = latest_signal.get("prices") or {}
    close_price = prices.get("close")
    if close_price is not None:
        return timestamp, format_price(close_price)

    if fallback_action:
        trade = find_latest_trade(report.get("recentTrades"), fallback_action)
        if trade:
            return format_trade_date(trade.get("date")), format_price(trade.get("price"))

    return timestamp, "暂无"


def find_latest_trade(trades: Iterable[dict] | None, action: str) -> dict | None:
    if not trades:
        return None

    normalized_action = action.upper()
    latest_trade = None
    latest_trade_dt = None

    for trade in trades:
        if not isinstance(trade, dict):
            continue
        trade_action = str(trade.get("action") or "").upper()
        if trade_action != normalized_action:
            continue
        trade_date = parse_trade_datetime(trade.get("date"))
        if latest_trade is None or trade_date >= latest_trade_dt:
            latest_trade = trade
            latest_trade_dt = trade_date

    return latest_trade


def find_matching_buy_for_latest_sell(trades: Iterable[dict] | None) -> dict | None:
    if not trades:
        return None

    latest_sell = None
    latest_sell_dt = None

    for trade in trades:
        if not isinstance(trade, dict):
            continue
        if str(trade.get("action") or "").upper() != "SELL":
            continue
        trade_dt = parse_trade_datetime(trade.get("date"))
        if latest_sell is None or trade_dt >= latest_sell_dt:
            latest_sell = trade
            latest_sell_dt = trade_dt

    if latest_sell is None:
        return None

    matching_buy = None
    matching_buy_dt = None
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        if str(trade.get("action") or "").upper() != "BUY":
            continue
        trade_dt = parse_trade_datetime(trade.get("date"))
        if trade_dt <= latest_sell_dt and (matching_buy is None or trade_dt >= matching_buy_dt):
            matching_buy = trade
            matching_buy_dt = trade_dt

    return matching_buy


def build_entry_lines(record: ReportRecord) -> list[str]:
    raw = record.raw
    lines = [f"## [{record.symbol}][{record.name}] - [{record.action}]"]

    if record.action == "买入":
        signal_time, signal_price = extract_latest_price(raw, "BUY")
        lines.append(f"买入信号：{signal_time}，价格：{signal_price}")
        return lines

    if record.action == "卖出":
        trades = raw.get("recentTrades")
        matching_buy = find_matching_buy_for_latest_sell(trades)
        sell_trade = find_latest_trade(trades, "SELL")
        sell_time, sell_price = extract_latest_price(raw, "SELL")
        trigger_reason = "暂无"
        if sell_trade:
            trigger_reason = str(sell_trade.get("reason") or "暂无")
        if matching_buy:
            lines.append(
                f"买入信号：{format_trade_date(matching_buy.get('date'))}，价格：{format_price(matching_buy.get('price'))}"
            )
        else:
            lines.append("买入信号：暂无，价格：暂无")

        if sell_trade and raw.get("latestSignal", {}).get("prices", {}).get("close") is None:
            sell_time = format_trade_date(sell_trade.get("date"))
            sell_price = format_price(sell_trade.get("price"))

        lines.append(f"卖出信号：{sell_time}，价格：{sell_price}")
        lines.append(f"触发条件：{trigger_reason}")
        return lines

    if record.action == "持有":
        position_info = raw.get("positionInfo") or {}
        entry_date = str(position_info.get("entry_date") or "暂无")
        entry_price = format_price(position_info.get("entry_price"))
        pnl_pct = format_percent(position_info.get("unrealized_pnl_pct"))
        lines.append(
            f"当前持仓买入信号：{entry_date}，价格：{entry_price}，盈亏：{pnl_pct}"
        )
        return lines

    latest_buy = find_latest_trade(raw.get("recentTrades"), "BUY")
    latest_sell = find_latest_trade(raw.get("recentTrades"), "SELL")
    if latest_buy:
        lines.append(
            f"上次买入：{format_trade_date(latest_buy.get('date'))}，价格：{format_price(latest_buy.get('price'))}"
        )
    else:
        lines.append("上次买入：暂无，价格：暂无")

    if latest_sell:
        lines.append(
            f"上次卖出：{format_trade_date(latest_sell.get('date'))}，价格：{format_price(latest_sell.get('price'))}"
        )
    else:
        lines.append("上次卖出：暂无，价格：暂无")

    return lines


def build_digest(reports: list[ReportRecord], target_date: date, reason: str | None = None) -> tuple[str, str]:
    date_str = target_date.isoformat()
    if reason in {"missing_dir", "no_json", "no_matching_reports"} or not reports:
        title = f"[{date_str}] 无任何数据可用"
        body = "\n".join(
            [
                "# 交易信号日报",
                f"- 日期：{date_str}",
                "- 状态：无任何数据可用",
                "- 说明：未找到可用于汇总的当日 JSON 数据。",
            ]
        )
        return title, body

    counts = {
        "买入": sum(1 for report in reports if report.action == "买入"),
        "卖出": sum(1 for report in reports if report.action == "卖出"),
        "持有": sum(1 for report in reports if report.action == "持有"),
        "观望": sum(1 for report in reports if report.action == "观望"),
    }
    trade_signal_count = counts["买入"] + counts["卖出"]
    title = (
        f"[{date_str}] {trade_signal_count}个交易信号"
        if trade_signal_count
        else f"[{date_str}] 无交易信号"
    )

    lines = [
        "# 交易信号日报",
        f"- 日期：{date_str}",
        f"- 交易信号数：{trade_signal_count}",
        f"- 买入：{counts['买入']}",
        f"- 卖出：{counts['卖出']}",
        f"- 持有：{counts['持有']}",
        f"- 观望：{counts['观望']}",
        "",
    ]

    for index, report in enumerate(reports):
        if index > 0:
            lines.append("")
        lines.extend(build_entry_lines(report))

    return title, "\n".join(lines)


def format_price(value: object) -> str:
    if value is None or value == "":
        return "暂无"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: object) -> str:
    if value is None or value == "":
        return "暂无"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def format_trade_date(value: object) -> str:
    if not value:
        return "暂无"
    return str(value)


def parse_trade_datetime(value: object) -> datetime:
    if not value:
        return datetime.min
    if isinstance(value, datetime):
        return value
    value_str = str(value)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value_str, fmt)
        except ValueError:
            continue
    return datetime.min


def send_pushgo_notification(
    title: str,
    body: str,
    channel_id: str,
    password: str,
    pushgo_url: str,
    max_retries: int,
    retry_delay: int,
    urlopen: Callable[..., object] = request.urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    payload = json.dumps(
        {
            "channel_id": channel_id,
            "password": password,
            "title": title,
            "body": body,
        }
    ).encode("utf-8")

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        req = request.Request(
            pushgo_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=15) as response:
                response_body = response.read().decode("utf-8")
                status_code = getattr(response, "status", response.getcode())
                if status_code >= 400:
                    raise RuntimeError(f"PushGo 返回状态码 {status_code}: {response_body}")
                parsed_body = json.loads(response_body)
                if not parsed_body.get("success", False):
                    raise RuntimeError(f"PushGo 返回失败: {response_body}")
                return parsed_body
        except (error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            print(f"[WARN] PushGo 推送失败，第 {attempt}/{max_retries} 次: {exc}", file=sys.stderr)
            if attempt < max_retries:
                sleep_fn(retry_delay)

    raise RuntimeError(f"PushGo 推送失败，已重试 {max_retries} 次: {last_error}")


def main() -> int:
    args = parse_args()
    load_dotenv_file(Path(__file__).resolve().parent / ".env")
    target_date = args.target_date or today_in_timezone()
    data_dir = Path(args.data_dir)
    reports, warnings, reason = load_reports(data_dir, target_date)

    for warning in warnings:
        print(f"[WARN] {warning}", file=sys.stderr)

    title, body = build_digest(reports, target_date, reason)

    if args.dry_run:
        print(title)
        print(body)
        return 0

    channel_id = os.getenv("PUSHGO_CHANNEL_ID", "").strip()
    password = os.getenv("PUSHGO_PASSWORD", "").strip()
    pushgo_url = args.pushgo_url or os.getenv("PUSHGO_URL", DEFAULT_PUSHGO_URL)

    if not channel_id or not password:
        print("[ERROR] 缺少 PUSHGO_CHANNEL_ID 或 PUSHGO_PASSWORD", file=sys.stderr)
        return 1

    try:
        response = send_pushgo_notification(
            title=title,
            body=body,
            channel_id=channel_id,
            password=password,
            pushgo_url=pushgo_url,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] PushGo 推送成功: {response.get('data', {}).get('message_id', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
