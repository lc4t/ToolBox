"use client"

import { useState, useEffect } from 'react';
import QuantTradeReport from '../components/QuantTradeReport';
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { config } from '../utils/config';

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
  returnMetrics: Array<{
    name: string;
    value: number | string;
    description: string;
  }>;
  riskMetrics: Array<{
    name: string;
    value: number | string;
    description: string;
  }>;
  riskAdjustedMetrics: Array<{
    name: string;
    value: number | string;
    description: string;
  }>;
  tradingMetrics: Array<{
    name: string;
    value: number | string;
    description: string;
  }>;
  positionMetrics: Array<{
    name: string;
    value: number | string;
    description: string;
  }>;
  benchmarkMetrics: Array<{
    name: string;
    value: number | string;
    description: string;
  }>;
  timeMetrics: Array<{
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
    returnMetrics: data.returnMetrics || [],
    riskMetrics: data.riskMetrics || [],
    riskAdjustedMetrics: data.riskAdjustedMetrics || [],
    tradingMetrics: data.tradingMetrics || [],
    positionMetrics: data.positionMetrics || [],
    benchmarkMetrics: data.benchmarkMetrics || [],
    timeMetrics: data.timeMetrics || [],
    latestSignal: {
      ...latestSignal,
      action: normalizeAction(latestSignal.action || ''),
      asset: latestSignal.asset || data.symbol || 'Unknown',
      timestamp: latestSignal.timestamp || new Date().toISOString(),
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
        if (!data || !Array.isArray(data) || data.length === 0) {
          throw new Error('No data available');
        }
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

  // 创建一个安全的获取当前数据的函数
  const getCurrentTradeData = (): TradeData | null => {
    if (!tradeData || !Array.isArray(tradeData) || tradeData.length === 0) {
      return null;
    }
    return tradeData[selectedIndex] || null;
  };

  if (isLoading) {
    return <div className="flex justify-center items-center min-h-screen">Loading...</div>;
  }

  if (error) {
    return <div className="flex justify-center items-center min-h-screen text-red-600">{error}</div>;
  }

  const currentData = getCurrentTradeData();
  if (!currentData) {
    return <div className="flex justify-center items-center min-h-screen">No data available</div>;
  }

  return (
    <main className="min-h-screen bg-background p-4">
      <Card className="mb-4">
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle>选择交易标的</CardTitle>
            <div className="text-sm text-muted-foreground bg-muted px-3 py-1 rounded-md">
              构建时间: {new Date(config.buildTime).toLocaleString('zh-CN', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
                hour12: false
              })}
            </div>
          </div>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {tradeData.map((data, index) => (
            <Button
              key={data.symbol}
              onClick={() => setSelectedIndex(index)}
              variant={selectedIndex === index ? "default" : "outline"}
            >
              {data.name}（{data.latestSignal?.action || '观察'}）
            </Button>
          ))}
        </CardContent>
      </Card>
      
      <QuantTradeReport 
        {...currentData}
        showStrategyParameters={currentData.showStrategyParameters || false}
      />
    </main>
  );
}

