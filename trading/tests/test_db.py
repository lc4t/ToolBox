import math
import os
import unittest

os.environ.setdefault("DB_PASSWORD", "unused")
os.environ.setdefault("DB_NAME", "unused")

import db  # noqa: E402


def make_record(**overrides):
    record = {
        "symbol": "159665.SZ",
        "date": "2026-07-24",
        "open_price": 2.566,
        "close_price": 2.648,
        "high": 2.706,
        "low": 2.52,
        "volume": 34837800,
        "amount": 92127003.0,
        "change": 0.022,
        "change_pct": 0.838,
    }
    record.update(overrides)
    return record


class CleanTradingDataTests(unittest.TestCase):
    def setUp(self):
        # DBClient.__init__ 只是构造 SQLAlchemy engine（惰性连接），不会真的连库
        self.client = db.DBClient(host="unused", user="unused", password="unused", database="unused", port=3306)

    def test_drops_records_with_nan_ohlc(self):
        good = make_record()
        bad = make_record(date="2026-07-25", close_price=float("nan"))

        cleaned = self.client._clean_trading_data([good, bad])

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["date"], "2026-07-24")

    def test_keeps_record_when_only_optional_field_is_nan(self):
        record = make_record(change=float("nan"), change_pct=float("nan"))

        cleaned = self.client._clean_trading_data([record])

        self.assertEqual(len(cleaned), 1)
        self.assertIsNone(cleaned[0]["change"])
        self.assertIsNone(cleaned[0]["change_pct"])
        # 核心价格字段没受影响
        self.assertEqual(cleaned[0]["close_price"], 2.648)

    def test_all_nan_ohlc_returns_empty(self):
        bad = make_record(
            open_price=float("nan"), close_price=float("nan"),
            high=float("nan"), low=float("nan"),
        )

        cleaned = self.client._clean_trading_data([bad])

        self.assertEqual(cleaned, [])

    def test_upsert_returns_false_without_touching_db_when_all_rows_bad(self):
        bad = make_record(close_price=float("nan"))

        result = self.client.upsert_trading_data([bad])

        self.assertFalse(result)

    def test_is_nan_helper(self):
        self.assertTrue(db.DBClient._is_nan(float("nan")))
        self.assertFalse(db.DBClient._is_nan(1.5))
        self.assertFalse(db.DBClient._is_nan(None))
        self.assertFalse(db.DBClient._is_nan("nan"))


if __name__ == "__main__":
    unittest.main()
