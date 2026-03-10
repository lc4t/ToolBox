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

        self.assertEqual(title, "[2026-03-10] 2个交易信号")
        self.assertIn("## [000001.SZ][买入标的] - [买入]", body)
        self.assertIn("买入信号：2026-03-10 09:30:00，价格：10.123", body)
        self.assertIn("买入信号：2026-03-01 09:30:00，价格：9.876", body)
        self.assertIn("卖出信号：2026-03-10 09:30:00，价格：11.321", body)
        self.assertIn("触发条件：MA交叉卖出 [MA5=10.1 < MA20=10.5]", body)
        self.assertIn("当前持仓买入信号：2026-03-03 09:30:00，价格：8.500，盈亏：7.89%", body)
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

        self.assertEqual(title, "[2026-03-10] 无交易信号")
        self.assertIn("- 持有：1", body)

    def test_build_digest_uses_empty_title_when_no_data(self):
        title, body = digest.build_digest([], date(2026, 3, 10), "no_json")

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
