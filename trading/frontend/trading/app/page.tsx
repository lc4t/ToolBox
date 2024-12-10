"use client"

import { useState, useEffect } from 'react';
import QuantTradeReport from '../components/QuantTradeReport';
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

// 定义交易数据类型
interface TradeData {
  symbol: string;
  name: string;
  reportDate: string;
  dateRange: {
    start: string;
    end: string;
  };
  latestSignal: {
    action: "观察" | "卖出" | "买入" | "持有";
    asset: string;
    timestamp: string;
    prices?: {
      open: number;
      close: number;
      high: number;
      low: number;
    };
    price?: number;
  };
  positionInfo: {
    entryDate: string;
    entryPrice: number;
    quantity: number;
    currentValue: number;
    profitLoss: number;
    profitLossPercentage: number;
  } | null;
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
    value: number | string;
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
    value: string | number | boolean;
  }>;
  showStrategyParameters: boolean;
}

// 修改 normalizeData 函数，添加具体的类型
interface RawTradeData extends Omit<TradeData, 'latestSignal'> {
  latestSignal?: {
    action?: string;
    asset?: string;
    timestamp?: string;
    prices?: {
      open: number;
      close: number;
      high: number;
      low: number;
    };
    price?: number;
  };
}

// 确保 action 字段符合类型要求
const normalizeAction = (action: string): "观察" | "卖出" | "买入" | "持有" => {
  // 先统一转换为大写，以处理不同的大小写情况
  const upperAction = action.toUpperCase();
  switch (upperAction) {
    case "BUY":
      return "买入";
    case "SELL":
      return "卖出";
    case "HOLD":
      return "持有";
    case "持有":  // 处理已经是中文的情况
      return "持有";
    case "买入":
      return "买入";
    case "卖出":
      return "卖出";
    case "观察":
      return "观察";
    default:
      return "观察";
  }
};

// 规范化数据
const normalizeData = (data: RawTradeData): TradeData => {
  const latestSignal = data.latestSignal || {};
  
  // 如果有 price 字段但没有 prices 字段，创建一个默认的 prices 对象
  if (latestSignal.price && !latestSignal.prices) {
    latestSignal.prices = {
      open: latestSignal.price,
      close: latestSignal.price,
      high: latestSignal.price,
      low: latestSignal.price
    };
  }
  
  return {
    ...data,
    latestSignal: {
      ...latestSignal,
      // 确保必需字段有默认值
      action: normalizeAction(latestSignal.action || ''),
      asset: latestSignal.asset || data.symbol || 'Unknown',  // 使用 symbol 作为后备
      timestamp: latestSignal.timestamp || new Date().toISOString(),  // 使用当前时间作为后备
      prices: latestSignal.prices,
      price: latestSignal.price
    }
  };
};

export default function Home() {
  const [tradeData, setTradeData] = useState<TradeData[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadTradeData = async () => {
      try {
        setIsLoading(true);
        const response = await fetch('/api/trade-data');
        if (!response.ok) {
          throw new Error('Failed to fetch data');
        }
        const data = (await response.json()) as RawTradeData[];
        setTradeData(data.map(normalizeData));
      } catch (error) {
        console.error('Error loading trade data:', error);
        setError(error instanceof Error ? error.message : 'Failed to load data');
      } finally {
        setIsLoading(false);
      }
    };

    loadTradeData();
  }, []);

  if (isLoading) {
    return <div className="flex justify-center items-center min-h-screen">Loading...</div>;
  }

  if (error) {
    return <div className="flex justify-center items-center min-h-screen text-red-500">{error}</div>;
  }

  if (tradeData.length === 0) {
    return <div className="flex justify-center items-center min-h-screen">No data available</div>;
  }

  return (
    <main className="min-h-screen bg-background p-4">
      <Card className="mb-4">
        <CardHeader>
          <CardTitle>选择交易标的</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {tradeData.map((data, index) => (
            <Button
              key={data.symbol}
              onClick={() => setSelectedIndex(index)}
              variant={selectedIndex === index ? "default" : "outline"}
            >
              {data.name}（{data.latestSignal.action}）
            </Button>
          ))}
        </CardContent>
      </Card>
      <QuantTradeReport {...tradeData[selectedIndex]} />
    </main>
  );
}
