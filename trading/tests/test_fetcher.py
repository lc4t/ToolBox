import math
import os
import unittest
from unittest import mock

os.environ.setdefault("DB_PASSWORD", "unused")
os.environ.setdefault("DB_NAME", "unused")

import pandas as pd

from fetcher import ADataFetcher  # noqa: E402


class ADataFetcherNanHandlingTests(unittest.TestCase):
    def test_nan_change_fields_become_none_not_nan(self):
        # AData 对当天还没有涨跌幅统计的记录会返回 NaN；float('nan') 在 Python 里是
        # truthy，`if record["change"]` 这种写法会把 NaN 误判成"有值"直接透传下去，
        # 最终整批写入 MySQL 时因为 NaN 字面量报错、静默丢掉一整批历史数据。
        df = pd.DataFrame(
            [
                {
                    "trade_date": "2026-07-24",
                    "open": 2.566,
                    "close": 2.648,
                    "high": 2.706,
                    "low": 2.52,
                    "volume": 34837800,
                    "amount": float("nan"),
                    "change": float("nan"),
                    "change_pct": float("nan"),
                }
            ]
        )

        with mock.patch("fetcher.adata.fund.market.get_market_etf", return_value=df):
            result = ADataFetcher().fetch_data("159665.SZ", "2026-07-01", "2026-07-25")

        self.assertEqual(len(result), 1)
        record = result[0]
        self.assertIsNone(record["amount"])
        self.assertIsNone(record["change"])
        self.assertIsNone(record["change_pct"])
        # 确认不是残留了 NaN float（那样后面写库还是会炸）
        for field in ("amount", "change", "change_pct"):
            self.assertFalse(isinstance(record[field], float) and math.isnan(record[field]))

    def test_zero_change_is_preserved_not_treated_as_falsy(self):
        # 之前用 `if record["change"]` 判断，change=0.0 这种"真实的零涨幅"
        # 也会被误判成假值、错误地清成 None，这里顺带验证修复后不会再丢真实的 0。
        df = pd.DataFrame(
            [
                {
                    "trade_date": "2026-07-24",
                    "open": 2.566,
                    "close": 2.648,
                    "high": 2.706,
                    "low": 2.52,
                    "volume": 34837800,
                    "amount": 0.0,
                    "change": 0.0,
                    "change_pct": 0.0,
                }
            ]
        )

        with mock.patch("fetcher.adata.fund.market.get_market_etf", return_value=df):
            result = ADataFetcher().fetch_data("159665.SZ", "2026-07-01", "2026-07-25")

        record = result[0]
        self.assertEqual(record["amount"], 0.0)
        self.assertEqual(record["change"], 0.0)
        self.assertEqual(record["change_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
