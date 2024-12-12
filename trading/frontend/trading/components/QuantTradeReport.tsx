import React, { useState } from 'react';
import { formatDate } from '../utils/formatDate';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

interface TradeSignal {
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
}

interface PositionInfo {
  entryDate: string;
  entryPrice: number;
  quantity: number;
  currentValue: number;
  profitLoss: number;
  profitLossPercentage: number;
}

interface Metric {
  name: string;
  value: string | number;
  description?: string;
}

interface Trade {
  date: string;
  action: string;
  price: number;
  quantity: number;
  value: number;
  profitLoss: number;
  profitLossPercentage?: number;
  totalValue: number;
  reason: string;
  entryPrice?: number;
}

interface StrategyParameter {
  name: string;
  value: string | number | boolean;
}

interface QuantTradeReportProps {
  symbol: string;
  name: string;
  reportDate: string;
  dateRange: { start: string; end: string };
  latestSignal: TradeSignal;
  positionInfo: PositionInfo | null;
  annualReturns: { year: number; value: number }[];
  returnMetrics: Metric[];
  riskMetrics: Metric[];
  riskAdjustedMetrics: Metric[];
  tradingMetrics: Metric[];
  positionMetrics: Metric[];
  benchmarkMetrics: Metric[];
  timeMetrics: Metric[];
  recentTrades: Trade[];
  strategyParameters: StrategyParameter[];
  showStrategyParameters: boolean;
}

export default function QuantTradeReport({
  symbol,
  name,
  reportDate,
  dateRange,
  latestSignal,
  positionInfo,
  annualReturns,
  returnMetrics,
  riskMetrics,
  riskAdjustedMetrics,
  tradingMetrics,
  positionMetrics,
  benchmarkMetrics,
  timeMetrics,
  recentTrades,
  strategyParameters,
  showStrategyParameters
}: QuantTradeReportProps) {
  return (
    <div className="container mx-auto p-4 space-y-6">
      <Card>
        <CardHeader className="bg-primary text-primary-foreground">
          <CardTitle className="text-2xl">
            【{latestSignal.action}】{name}({symbol}) - {reportDate}
          </CardTitle>
          <p>{formatDate(dateRange.start)} 至 {formatDate(dateRange.end)}</p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
            <LatestSignalSection signal={latestSignal} />
            <PositionInfoSection info={positionInfo} />
          </div>
        </CardContent>
      </Card>

      <AnnualReturnsSection returns={annualReturns} />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricsSection title="收益指标" metrics={returnMetrics} />
        <MetricsSection title="风险指标" metrics={riskMetrics} />
        <MetricsSection title="风险调整收益" metrics={riskAdjustedMetrics} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricsSection title="交易统计" metrics={tradingMetrics} />
        <MetricsSection title="持仓特征" metrics={positionMetrics} />
        <MetricsSection title="时间统计" metrics={timeMetrics} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <MetricsSection title="基准对比" metrics={benchmarkMetrics} />
      </div>

      <RecentTradesSection trades={recentTrades} />
      <StrategyParametersSection parameters={strategyParameters} show={showStrategyParameters} />
    </div>
  );
}

function LatestSignalSection({ signal }: { signal: TradeSignal }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>最新信号</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <div>
              <p className="text-lg font-semibold">{signal.asset}</p>
              <p className="text-sm text-muted-foreground">
                {formatDate(signal.timestamp)} 交易日
              </p>
            </div>
            <Badge
              variant={
                signal.action === '买入'
                  ? 'default'
                  : signal.action === '卖出'
                  ? 'destructive'
                  : 'secondary'
              }
            >
              {signal.action}
            </Badge>
          </div>
          
          {signal.prices ? (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">开盘</p>
                <p className="text-lg font-semibold">
                  {signal.prices.open > 0 ? `¥${signal.prices.open.toFixed(3)}` : '暂无'}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">收盘</p>
                <p className="text-lg font-semibold">
                  {signal.prices.close > 0 ? `¥${signal.prices.close.toFixed(3)}` : '暂无'}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">最高</p>
                <p className="text-lg font-semibold text-green-600">
                  {signal.prices.high > 0 ? `¥${signal.prices.high.toFixed(3)}` : '暂无'}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">最低</p>
                <p className="text-lg font-semibold text-red-600">
                  {signal.prices.low > 0 ? `¥${signal.prices.low.toFixed(3)}` : '暂无'}
                </p>
              </div>
            </div>
          ) : (
            <div>
              <p className="text-sm text-muted-foreground">最新价格</p>
              <p className="text-lg font-semibold">
                {signal.price ? `¥${signal.price.toFixed(3)}` : '暂无价格'}
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function PositionInfoSection({ info }: { info: PositionInfo | null }) {
  if (!info) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>当前持仓</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <div>
            <p className="text-sm text-muted-foreground">买入日期:</p>
            <p className="text-lg font-semibold">{formatDate(info.entryDate)}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">买入价格:</p>
            <p className="text-lg font-semibold">¥{info.entryPrice.toFixed(3)}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">持仓成本:</p>
            <p className="text-lg font-semibold">{info.quantity.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">当前价值:</p>
            <p className="text-lg font-semibold">¥{info.currentValue.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">浮动盈亏:</p>
            <p className={`text-lg font-semibold ${info.profitLoss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              ¥{info.profitLoss.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">收益率:</p>
            <p className={`text-lg font-semibold ${info.profitLossPercentage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {info.profitLossPercentage >= 0 ? '+' : ''}{info.profitLossPercentage.toFixed(2)}%
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function AnnualReturnsSection({ returns }: { returns: { year: number; value: number }[] }) {
  if (!returns || returns.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>年度收益率</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center text-muted-foreground">暂无数据</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>年度收益率</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>年度</TableHead>
              <TableHead>收益率</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {returns.map((item, index) => (
              <TableRow key={index}>
                <TableCell>{item.year}年</TableCell>
                <TableCell className={item.value >= 0 ? 'text-green-600' : 'text-red-600'}>
                  {item.value >= 0 ? '+' : ''}{item.value.toFixed(2)}%
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function MetricsSection({ title, metrics }: { title: string; metrics: Metric[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {metrics.map((metric, index) => (
            <div key={index} className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="font-medium">{metric.name}</span>
                <span className={
                  typeof metric.value === 'number' 
                    ? metric.value > 0 ? 'text-green-600' : metric.value < 0 ? 'text-red-600' : ''
                    : ''
                }>
                  {typeof metric.value === 'number' ? metric.value.toFixed(2) : metric.value}
                </span>
              </div>
              {metric.description && (
                <p className="text-sm text-muted-foreground">{metric.description}</p>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function RecentTradesSection({ trades }: { trades: Trade[] }) {
  const [showAllTrades, setShowAllTrades] = useState(false);
  const [sortConfig, setSortConfig] = useState<{
    key: keyof Trade;
    direction: 'asc' | 'desc';
  }>({ key: 'date', direction: 'desc' });

  const getActionDisplay = (action: string) => {
    return action === 'BUY' ? '买入' : action === 'SELL' ? '卖出' : action;
  };

  const getVariant = (action: string) => {
    return action === 'BUY' ? 'default' : action === 'SELL' ? 'destructive' : 'secondary';
  };

  const sortedTrades = [...trades].sort((a, b) => {
    const multiplier = sortConfig.direction === 'asc' ? 1 : -1;
    
    if (sortConfig.key === 'date') {
      return multiplier * (new Date(a.date).getTime() - new Date(b.date).getTime());
    }
    
    const aValue = a[sortConfig.key];
    const bValue = b[sortConfig.key];
    
    if (aValue == null && bValue == null) return 0;
    if (aValue == null) return 1;
    if (bValue == null) return -1;
    
    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return multiplier * (aValue - bValue);
    }
    
    if (typeof aValue === 'string' && typeof bValue === 'string') {
      return multiplier * aValue.localeCompare(bValue);
    }
    
    return 0;
  });

  const displayTrades = showAllTrades ? sortedTrades : sortedTrades.slice(0, 10);

  const handleSort = (key: keyof Trade) => {
    setSortConfig(current => ({
      key,
      direction: current.key === key && current.direction === 'desc' ? 'asc' : 'desc'
    }));
  };

  const formatProfitLoss = (trade: Trade) => {
    if (trade.action !== 'SELL') return '—';
    
    const pnlText = `¥${trade.profitLoss.toFixed(2)}`;
    const pnlPercentage = trade.profitLossPercentage;
    
    const className = trade.profitLoss >= 0 ? 'text-green-600' : 'text-red-600';
    
    return (
      <div className={className}>
        {pnlText}
        {pnlPercentage != null && (
          <span className="ml-1">
            ({pnlPercentage >= 0 ? '+' : ''}{pnlPercentage.toFixed(2)}%)
          </span>
        )}
      </div>
    );
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle>交易记录</CardTitle>
          <Button
            variant="outline"
            onClick={() => setShowAllTrades(!showAllTrades)}
          >
            {showAllTrades ? '显示最近交易' : '显示所有交易'}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead onClick={() => handleSort('date')} className="cursor-pointer">
                  日期 {sortConfig.key === 'date' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                </TableHead>
                <TableHead>操作</TableHead>
                <TableHead onClick={() => handleSort('price')} className="cursor-pointer">
                  价格 {sortConfig.key === 'price' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                </TableHead>
                <TableHead onClick={() => handleSort('quantity')} className="cursor-pointer">
                  数量 {sortConfig.key === 'quantity' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                </TableHead>
                <TableHead onClick={() => handleSort('value')} className="cursor-pointer">
                  价值 {sortConfig.key === 'value' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                </TableHead>
                <TableHead onClick={() => handleSort('profitLoss')} className="cursor-pointer">
                  盈亏 {sortConfig.key === 'profitLoss' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                </TableHead>
                <TableHead onClick={() => handleSort('totalValue')} className="cursor-pointer">
                  总价值 {sortConfig.key === 'totalValue' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                </TableHead>
                <TableHead>原因</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {displayTrades.map((trade, index) => (
                <TableRow key={index}>
                  <TableCell>{formatDate(trade.date)}</TableCell>
                  <TableCell>
                    <Badge variant={getVariant(trade.action)}>
                      {getActionDisplay(trade.action)}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    ¥{trade.price.toFixed(3)}
                    {trade.action === 'SELL' && trade.entryPrice && (
                      <div className="text-xs text-muted-foreground">
                        买入: ¥{trade.entryPrice.toFixed(3)}
                      </div>
                    )}
                  </TableCell>
                  <TableCell>{trade.quantity.toLocaleString()}</TableCell>
                  <TableCell>¥{trade.value.toFixed(2)}</TableCell>
                  <TableCell>
                    {formatProfitLoss(trade)}
                  </TableCell>
                  <TableCell>¥{trade.totalValue.toFixed(2)}</TableCell>
                  <TableCell className="max-w-md truncate" title={trade.reason}>
                    {trade.reason}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <div className="mt-4 text-sm text-muted-foreground text-right">
          共 {trades.length} 笔交易，当前显示 {displayTrades.length} 笔
        </div>
      </CardContent>
    </Card>
  );
}

function StrategyParametersSection({ parameters, show }: { parameters: StrategyParameter[]; show: boolean }) {
  if (!show) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>策略参数</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>参数</TableHead>
              <TableHead>值</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {parameters.map((param, index) => (
              <TableRow key={index}>
                <TableCell>{param.name}</TableCell>
                <TableCell>{typeof param.value === 'boolean' ? (param.value ? '是' : '否') : param.value.toString()}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

