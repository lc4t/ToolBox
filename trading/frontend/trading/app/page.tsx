"use client"

import { useEffect, useState } from 'react';
import QuantTradeReport from '../components/QuantTradeReport';
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

// 定义类型
interface TradeReport {
  symbol: string;
  name: string;
  reportDate: string;
  dateRange: {
    start: string;
    end: string;
  };
  latestSignal: {
    action: "观察" | "买入" | "卖出" | "持有";
    asset: string;
    price: number;
    timestamp: string;
  };
  annualReturns: Array<{
    year: number;
    value: number;
  }>;
  performanceMetrics: Array<{
    name: string;
    value: number;
    description: string;
  }>;
  riskMetrics: Array<{
    name: string;
    value: number;
    description: string;
  }>;
  marketIndicators: Array<{
    name: string;
    value: number;
    description: string;
  }>;
  recentTrades: Array<{
    date: string;
    action: string;
    price: number;
    quantity: number;
    value: number;
    profitLoss: number;
    totalValue: number;
    reason: string;
  }>;
  strategyParameters: Array<{
    name: string;
    value: any;
  }>;
  showStrategyParameters: boolean;
  positionInfo: {
    entryDate: string;
    entryPrice: number;
    quantity: number;
    currentValue: number;
    profitLoss: number;
    profitLossPercentage: number;
  } | null;
}

export default function Home() {
  const [reports, setReports] = useState<TradeReport[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 加载所有报告
    async function loadReports() {
      try {
        const response = await fetch('/api/reports');
        const data = await response.json();

        // 按报告日期降序排序
        const sortedReports = data.sort((a: TradeReport, b: TradeReport) =>
          new Date(b.reportDate).getTime() - new Date(a.reportDate).getTime()
        );

        setReports(sortedReports);
      } catch (error) {
        console.error('Failed to load reports:', error);
      } finally {
        setLoading(false);
      }
    }

    loadReports();
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center min-h-screen">
      <p>加载中...</p>
    </div>;
  }

  if (reports.length === 0) {
    return <div className="flex items-center justify-center min-h-screen">
      <p>没有可用的回测报告</p>
    </div>;
  }

  return (
    <main className="min-h-screen bg-background p-4">
      <Card className="mb-4">
        <CardHeader>
          <CardTitle>选择交易标的</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {reports.map((report, index) => (
            <Button
              key={report.symbol}
              onClick={() => setSelectedIndex(index)}
              variant={selectedIndex === index ? "default" : "outline"}
            >
              {report.name}（{report.latestSignal.action}）
            </Button>
          ))}
        </CardContent>
      </Card>
      <QuantTradeReport {...reports[selectedIndex]} />
    </main>
  );
}

