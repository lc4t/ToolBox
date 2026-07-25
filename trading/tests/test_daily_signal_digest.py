import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock
from urllib import error
import os

import daily_signal_digest as digest


def make_report(symbol: str, name: str, report_date: str, action: str, **overrides):
    report = {
        "symbol": symbol,
        "name": name,
        "reportDate": report_date,
        "latestSignal": {
            "action": action,
            "timestamp": f"{report_date} 09:30:00",
            "prices": {"close": 10.123},
        },
        "positionInfo": None,
        "recentTrades": [],
    }
    report.update(overrides)
    return report


def make_rotation_report(strategy_id: str, strategy_name: str, report_date: str, **overrides):
    report = {
        "schemaVersion": "rotation-v1",
        "strategyId": strategy_id,
        "strategyName": strategy_name,
        "reportDate": report_date,
        "currentHolding": {
            "symbol": "159915.SZ",
            "name": "159915.SZ",
            "since": "2026-07-21",
            "entryPrice": 3.5,
            "currentPrice": 3.6,
            "unrealizedPnlPct": 2.86,
        },
        "nextSignal": {
            "action": "HOLD",
            "reason": "动量最强仍为 159915.SZ",
        },
        "rotations": [],
    }
    report.update(overrides)
    return report


class LoadReportsTests(unittest.TestCase):
    def test_load_reports_filters_today_and_skips_broken_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            today = date(2026, 3, 10)
            (base / "broken.json").write_text("{bad", encoding="utf-8")
            (base / "old.json").write_text(
                json.dumps(make_report("000002.SZ", "旧数据", "2026-03-09", "买入")),
                encoding="utf-8",
            )
            (base / "today.json").write_text(
                json.dumps(make_report("000001.SZ", "今日数据", "2026-03-10", "买入")),
                encoding="utf-8",
            )

            reports, warnings, reason = digest.load_reports(base, today)

            self.assertEqual(reason, None)
            self.assertEqual([report.symbol for report in reports], ["000001.SZ"])
            self.assertEqual(len(warnings), 1)

    def test_load_reports_returns_reason_when_no_matching_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "old.json").write_text(
                json.dumps(make_report("000002.SZ", "旧数据", "2026-03-09", "观望")),
                encoding="utf-8",
            )

            reports, warnings, reason = digest.load_reports(base, date(2026, 3, 10))

            self.assertEqual(reports, [])
            self.assertEqual(warnings, [])
            self.assertEqual(reason, "no_matching_reports")


class LoadRotationReportsTests(unittest.TestCase):
    def test_load_rotation_reports_filters_today_and_skips_broken_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            rotation_dir = base / "rotation"
            rotation_dir.mkdir()
            today = date(2026, 7, 25)
            (rotation_dir / "broken.json").write_text("{bad", encoding="utf-8")
            (rotation_dir / "old.json").write_text(
                json.dumps(make_rotation_report("momentum-top1", "动量轮动 Top-1", "2026-07-24")),
                encoding="utf-8",
            )
            (rotation_dir / "today.json").write_text(
                json.dumps(make_rotation_report("momentum-top1", "动量轮动 Top-1", "2026-07-25")),
                encoding="utf-8",
            )

            reports, warnings = digest.load_rotation_reports(base, today)

            self.assertEqual([r["strategyId"] for r in reports], ["momentum-top1"])
            self.assertEqual(len(warnings), 1)

    def test_load_rotation_reports_returns_empty_when_dir_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            reports, warnings = digest.load_rotation_reports(Path(temp_dir), date(2026, 7, 25))

            self.assertEqual(reports, [])
            self.assertEqual(warnings, [])


class FormatPnlLabelTests(unittest.TestCase):
    def test_positive_value_is_labeled_as_profit(self):
        self.assertEqual(digest.format_pnl_label(34.482), "浮盈34.48%")

    def test_negative_value_is_labeled_as_loss(self):
        self.assertEqual(digest.format_pnl_label(-3.2), "浮亏3.20%")

    def test_none_returns_placeholder(self):
        self.assertEqual(digest.format_pnl_label(None), "暂无")


class BaseStrategyNameTests(unittest.TestCase):
    def test_strips_trailing_candidate_pool_in_chinese_parens(self):
        self.assertEqual(
            digest.base_strategy_name("动量轮动 Top-1（创业板/纳指/黄金）"), "动量轮动 Top-1"
        )

    def test_strips_trailing_candidate_pool_in_ascii_parens(self):
        self.assertEqual(digest.base_strategy_name("Momentum Top-1(A/B/C)"), "Momentum Top-1")

    def test_returns_unchanged_when_no_parens(self):
        self.assertEqual(digest.base_strategy_name("动量轮动 Top-1"), "动量轮动 Top-1")


class ResolveSymbolNameTests(unittest.TestCase):
    def test_known_symbol_falls_back_to_chinese_name_when_raw_name_matches_symbol(self):
        self.assertEqual(digest.resolve_symbol_name("518880.SS", "518880.SS"), "华安易富黄金ETF")

    def test_unknown_symbol_falls_back_to_symbol_itself(self):
        self.assertEqual(digest.resolve_symbol_name("000001.SZ", "000001.SZ"), "000001.SZ")

    def test_prefers_real_raw_name_over_known_map(self):
        self.assertEqual(digest.resolve_symbol_name("518880.SS", "黄金ETF别名"), "黄金ETF别名")


class BuildDigestTests(unittest.TestCase):
    def test_build_digest_formats_all_actions(self):
        target_date = date(2026, 3, 10)
        reports = [
            digest.ReportRecord(
                source_path=Path("buy.json"),
                raw=make_report("000001.SZ", "买入标的", "2026-03-10", "买入"),
                action="买入",
                symbol="000001.SZ",
                name="买入标的",
            ),
            digest.ReportRecord(
                source_path=Path("sell.json"),
                raw=make_report(
                    "000002.SZ",
                    "卖出标的",
                    "2026-03-10",
                    "卖出",
                    latestSignal={
                        "action": "卖出",
                        "timestamp": "2026-03-10 09:30:00",
                        "prices": {"close": 11.321},
                    },
                    recentTrades=[
                        {"date": "2026-03-01 09:30:00", "action": "BUY", "price": 9.876},
                        {
                            "date": "2026-03-10 09:30:00",
                            "action": "SELL",
                            "price": 11.321,
                            "reason": "MA交叉卖出 [MA5=10.1 < MA20=10.5]",
                        },
                    ],
                ),
                action="卖出",
                symbol="000002.SZ",
                name="卖出标的",
            ),
            digest.ReportRecord(
                source_path=Path("hold.json"),
                raw=make_report(
                    "000003.SZ",
                    "持有标的",
                    "2026-03-10",
                    "持有",
                    positionInfo={
                        "entry_date": "2026-03-03 09:30:00",
                        "entry_price": 8.5,
                        "unrealized_pnl_pct": 7.89,
                    },
                ),
                action="持有",
                symbol="000003.SZ",
                name="持有标的",
            ),
            digest.ReportRecord(
                source_path=Path("watch.json"),
                raw=make_report(
                    "000004.SZ",
                    "观望标的",
                    "2026-03-10",
                    "观望",
                    recentTrades=[
                        {"date": "2026-02-01 09:30:00", "action": "BUY", "price": 6.123},
                        {"date": "2026-02-15 09:30:00", "action": "SELL", "price": 6.789},
                    ],
                ),
                action="观望",
                symbol="000004.SZ",
                name="观望标的",
            ),
        ]

        title, body = digest.build_digest(reports, target_date)

        self.assertEqual(title, "📊 [2026-03-10] 2个交易信号")
        self.assertIn("## 🎯 信号总览", body)
        self.assertIn("🟢 买入标的（000001.SZ）买入", body)
        self.assertIn("🔴 卖出标的（000002.SZ）卖出", body)
        self.assertIn("🟡 持有标的（000003.SZ）持有 浮盈7.89%", body)
        self.assertIn("⚪ 观望标的（000004.SZ）空仓", body)

        self.assertIn("### 🟢 买入标的（000001.SZ）· 买入", body)
        self.assertIn("买入信号：2026-03-10 09:30:00，价格：10.123", body)

        self.assertIn("### 🔴 卖出标的（000002.SZ）· 卖出", body)
        self.assertIn("买入信号：2026-03-01 09:30:00，价格：9.876", body)
        self.assertIn("卖出信号：2026-03-10 09:30:00，价格：11.321", body)
        self.assertIn("触发条件：MA交叉卖出 [MA5=10.1 < MA20=10.5]", body)

        self.assertIn("### 🟡 持有标的（000003.SZ）· 持有", body)
        self.assertIn("买入信号：2026-03-03 09:30:00，价格：8.500", body)
        self.assertIn("浮动盈亏：浮盈7.89%", body)

        self.assertIn("### ⚪ 观望标的（000004.SZ）· 空仓", body)
        self.assertIn("上次买入：2026-02-01 09:30:00，价格：6.123", body)
        self.assertIn("上次卖出：2026-02-15 09:30:00，价格：6.789", body)

    def test_build_digest_uses_no_signal_title_when_only_hold_and_watch(self):
        target_date = date(2026, 3, 10)
        reports = [
            digest.ReportRecord(
                source_path=Path("hold.json"),
                raw=make_report("000003.SZ", "持有标的", "2026-03-10", "持有"),
                action="持有",
                symbol="000003.SZ",
                name="持有标的",
            )
        ]

        title, body = digest.build_digest(reports, target_date)

        self.assertEqual(title, "📊 [2026-03-10] 无交易信号")
        self.assertIn("🟡 持有 1", body)

    def test_build_digest_uses_empty_title_when_no_data(self):
        title, body = digest.build_digest([], date(2026, 3, 10), "no_json")

        self.assertEqual(title, "[2026-03-10] 无任何数据可用")
        self.assertIn("未找到可用于汇总的当日 JSON 数据", body)

    def test_build_digest_includes_rotation_holding_with_pnl(self):
        target_date = date(2026, 7, 25)
        rotation_reports = [
            make_rotation_report("momentum-top1", "动量轮动 Top-1", "2026-07-25"),
        ]

        title, body = digest.build_digest([], target_date, "no_matching_reports", rotation_reports)

        self.assertEqual(title, "📊 [2026-07-25] 无交易信号")
        self.assertIn("🟡 动量轮动 Top-1 · 创业板ETF（momentum-top1）持有 浮盈2.86%", body)
        self.assertIn("### 🟡 动量轮动 Top-1 · 创业板ETF（momentum-top1）· 持有", body)
        self.assertIn(
            "当前持仓：159915.SZ（创业板ETF），入场：2026-07-21，入场价：3.500，当前价：3.600", body
        )
        self.assertIn("浮动盈亏：浮盈2.86%", body)

    def test_build_digest_rotation_strategy_name_drops_candidate_pool_suffix(self):
        target_date = date(2026, 7, 25)
        rotation_reports = [
            make_rotation_report(
                "momentum-top1", "动量轮动 Top-1（创业板/纳指/黄金）", "2026-07-25"
            ),
        ]

        _, body = digest.build_digest([], target_date, "no_matching_reports", rotation_reports)

        self.assertIn("🟡 动量轮动 Top-1 · 创业板ETF（momentum-top1）持有", body)
        self.assertNotIn("创业板/纳指/黄金", body)
        self.assertIn("下次预期动作：持有（动量最强仍为 159915.SZ）", body)

    def test_build_digest_rotation_buy_today(self):
        target_date = date(2026, 7, 25)
        rotation_reports = [
            make_rotation_report(
                "momentum-top1",
                "动量轮动 Top-1",
                "2026-07-25",
                currentHolding={
                    "symbol": "513100.SS",
                    "name": "513100.SS",
                    "since": "2026-07-25",
                    "entryPrice": 2.1,
                    "currentPrice": 2.1,
                    "unrealizedPnlPct": 0.0,
                },
            ),
        ]

        title, body = digest.build_digest([], target_date, "no_matching_reports", rotation_reports)

        self.assertEqual(title, "📊 [2026-07-25] 1个交易信号")
        self.assertIn("🟢 动量轮动 Top-1 · 国泰纳斯达克100ETF（momentum-top1）买入", body)

    def test_build_digest_rotation_exit_with_no_holding(self):
        target_date = date(2026, 7, 25)
        rotation_reports = [
            make_rotation_report(
                "momentum-top1",
                "动量轮动 Top-1",
                "2026-07-25",
                currentHolding=None,
                nextSignal={"action": "EXIT", "reason": "所有标的动量都低于阈值或为 NaN，下次预期空仓"},
            ),
        ]

        title, body = digest.build_digest([], target_date, "no_matching_reports", rotation_reports)

        self.assertEqual(title, "📊 [2026-07-25] 无交易信号")
        self.assertIn("⚪ 动量轮动 Top-1 · 空仓（momentum-top1）空仓", body)
        self.assertIn("当前持仓：空仓", body)
        self.assertIn("下次预期动作：清仓（所有标的动量都低于阈值或为 NaN，下次预期空仓）", body)

    def test_build_digest_rotation_sell_today(self):
        target_date = date(2026, 7, 25)
        rotation_reports = [
            make_rotation_report(
                "momentum-top1",
                "动量轮动 Top-1",
                "2026-07-25",
                currentHolding=None,
                rotations=[
                    {
                        "symbol": "159915.SZ",
                        "entryDate": "2026-07-10",
                        "exitDate": "2026-07-25",
                        "entryPrice": 3.5,
                        "exitPrice": 3.8,
                        "size": 1000,
                        "holdingDays": 15,
                        "periodReturnPct": 8.57,
                        "pnl": 300.0,
                    }
                ],
            ),
        ]

        title, body = digest.build_digest([], target_date, "no_matching_reports", rotation_reports)

        self.assertEqual(title, "📊 [2026-07-25] 1个交易信号")
        self.assertIn("🔴 动量轮动 Top-1 · 已清仓创业板ETF（momentum-top1）卖出 浮盈8.57%", body)
        self.assertIn("卖出标的：159915.SZ（创业板ETF），入场：2026-07-10，价格：3.500，出场价：3.800", body)
        self.assertIn("本段收益：浮盈8.57%", body)

    def test_build_digest_no_data_when_reports_and_rotation_both_empty(self):
        title, body = digest.build_digest([], date(2026, 3, 10), "no_json", [])

        self.assertEqual(title, "[2026-03-10] 无任何数据可用")
        self.assertIn("未找到可用于汇总的当日 JSON 数据", body)


class SendPushGoTests(unittest.TestCase):
    def test_send_pushgo_retries_until_success(self):
        attempts = []

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"success": true, "data": {"message_id": "msg-1"}}'

            def getcode(self):
                return self.status

        def fake_urlopen(req, timeout=15):
            attempts.append(timeout)
            if len(attempts) < 3:
                raise error.URLError("temporary")
            return Response()

        response = digest.send_pushgo_notification(
            title="title",
            body="body",
            channel_id="cid",
            password="pwd",
            pushgo_url="https://gateway.pushgo.dev/push",
            max_retries=3,
            retry_delay=0,
            urlopen=fake_urlopen,
            sleep_fn=lambda _: None,
        )

        self.assertEqual(len(attempts), 3)
        self.assertEqual(response["data"]["message_id"], "msg-1")

    def test_send_pushgo_raises_after_retry_exhausted(self):
        def fake_urlopen(req, timeout=15):
            raise error.URLError("down")

        with self.assertRaises(RuntimeError):
            digest.send_pushgo_notification(
                title="title",
                body="body",
                channel_id="cid",
                password="pwd",
                pushgo_url="https://gateway.pushgo.dev/push",
                max_retries=2,
                retry_delay=0,
                urlopen=fake_urlopen,
                sleep_fn=lambda _: None,
            )


class MainTests(unittest.TestCase):
    def test_load_dotenv_file_sets_missing_values_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = Path(temp_dir) / ".env"
            dotenv_path.write_text("PUSHGO_CHANNEL_ID=test-id\nPUSHGO_PASSWORD=test-pass\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"PUSHGO_PASSWORD": "existing"}, clear=False):
                os.environ.pop("PUSHGO_CHANNEL_ID", None)
                digest.load_dotenv_file(dotenv_path)
                self.assertEqual(os.environ["PUSHGO_CHANNEL_ID"], "test-id")
                self.assertEqual(os.environ["PUSHGO_PASSWORD"], "existing")

    def test_main_dry_run_prints_title_and_body(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "buy.json").write_text(
                json.dumps(make_report("000001.SZ", "买入标的", "2026-03-10", "买入")),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "daily_signal_digest.py",
                "--data-dir",
                temp_dir,
                "--date",
                "2026-03-10",
                "--dry-run",
            ]

            with mock.patch("sys.argv", argv), mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
                exit_code = digest.main()

            self.assertEqual(exit_code, 0)
            self.assertIn("[2026-03-10] 1个交易信号", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
