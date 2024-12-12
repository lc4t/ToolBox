import React, { useState } from 'react';
import { formatDate } from '../utils/formatDate';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { formatMoney } from '../utils/formatMoney';

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
  entryDate?: string;
  entry_date?: string;
  entryPrice?: number;
  entry_price?: number;
  quantity?: number;
  position_size?: number;
  currentValue?: number;
  current_value?: number;
  profitLoss?: number;
  unrealized_pnl?: number;
  profitLossPercentage?: number;
  unrealized_pnl_pct?: number;
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
            <LatestSignalSection signal={latestSignal} symbol={symbol} />
            <PositionInfoSection info={positionInfo} symbol={symbol} />
          </div>
        </CardContent>
      </Card>

      <AnnualReturnsSection returns={annualReturns} />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricsSection title="收益指标" metrics={returnMetrics} symbol={symbol} />
        <MetricsSection title="风险指标" metrics={riskMetrics} symbol={symbol} />
        <MetricsSection title="风险调整收益" metrics={riskAdjustedMetrics} symbol={symbol} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricsSection title="交易统计" metrics={tradingMetrics} symbol={symbol} />
        <MetricsSection title="持仓特征" metrics={positionMetrics} symbol={symbol} />
        <MetricsSection title="时间统计" metrics={timeMetrics} symbol={symbol} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <MetricsSection title="基准对比" metrics={benchmarkMetrics} symbol={symbol} />
      </div>

      <RecentTradesSection trades={recentTrades} symbol={symbol} />
      <StrategyParametersSection parameters={strategyParameters} show={showStrategyParameters} />
    </div>
  );
}

function LatestSignalSection({ signal, symbol }: { signal: TradeSignal, symbol: string }) {
  const getBadgeVariant = (action: string) => {
    switch (action) {
      case "持有":
        return "hold";
      case "买入":
        return "buy";
      case "卖出":
        return "sell";
      case "观察":
      default:
        return "watch";
    }
  };

  return (
    <div>
      <h3 className="text-lg font-semibold mb-4">最新信号</h3>
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <div>
            <p className="text-lg font-semibold">{signal.asset}</p>
            <p className="text-sm text-muted-foreground">
              {formatDate(signal.timestamp)} 交易日
            </p>
          </div>
          <Badge variant={getBadgeVariant(signal.action)}>
            {signal.action}
          </Badge>
        </div>

        {signal.prices ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-muted-foreground">开盘</p>
              <p className="text-lg font-semibold">
                {signal.prices.open > 0 ? formatMoney(signal.prices.open, symbol) : '暂无'}
              </p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">最高</p>
              <p className="text-lg font-semibold text-green-600">
                {signal.prices.high > 0 ? formatMoney(signal.prices.high, symbol) : '暂无'}
              </p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">最低</p>
              <p className="text-lg font-semibold text-red-600">
                {signal.prices.low > 0 ? formatMoney(signal.prices.low, symbol) : '暂无'}
              </p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">收盘</p>
              <p className="text-lg font-semibold">
                {signal.prices.close > 0 ? formatMoney(signal.prices.close, symbol) : '暂无'}
              </p>
            </div>
          </div>
        ) : signal.price ? (
          <div>
            <p className="text-sm text-muted-foreground">最新价格</p>
            <p className="text-lg font-semibold">
              {formatMoney(signal.price, symbol)}
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function PositionInfoSection({ info, symbol }: { info: PositionInfo | null, symbol: string }) {
  if (!info) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>当前持仓</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">暂无持仓</p>
        </CardContent>
      </Card>
    );
  }

  // 确保所有需要的字段都存在并格式化
  const entryDate = info.entryDate || info.entry_date || '';
  const entryPrice = info.entryPrice || info.entry_price || 0;
  const quantity = info.quantity || info.position_size || 0;
  const currentValue = info.currentValue || info.current_value || 0;
  const profitLoss = info.profitLoss || info.unrealized_pnl || 0;
  const profitLossPercentage = info.profitLossPercentage || info.unrealized_pnl_pct || 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>当前持仓</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <div>
            <p className="text-sm text-muted-foreground">买入日期:</p>
            <p className="text-lg font-semibold">{entryDate ? formatDate(entryDate) : '暂无'}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">买入价格:</p>
            <p className="text-lg font-semibold">{formatMoney(entryPrice, symbol)}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">持仓数量:</p>
            <p className="text-lg font-semibold">{quantity.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">当前价值:</p>
            <p className="text-lg font-semibold">{formatMoney(currentValue, symbol)}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">浮动盈亏:</p>
            <p className={`text-lg font-semibold ${profitLoss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatMoney(profitLoss, symbol)}
            </p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">收益率:</p>
            <p className={`text-lg font-semibold ${profitLossPercentage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {profitLossPercentage >= 0 ? '+' : ''}{profitLossPercentage.toFixed(2)}%
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

function MetricsSection({ title, metrics, symbol }: { title: string; metrics: Metric[]; symbol: string }) {
  const formatValue = (metric: Metric) => {
    const value = metric.value;

    if (typeof value === 'string' && value.includes('天')) {
      return value;
    }

    const percentageMetrics = [
      '年化收益率', '复合年化收益', 'Alpha', '最大回撤', '当前回撤',
      '波动率', '胜率', '持仓比例'
    ];

    const moneyMetrics = [
      '总盈亏', '平均盈利', '平均亏损', '最大亏损'
    ];

    const ratioMetrics = [
      '最新净值', '夏普比率', '索提诺比率', 'Calmar比率', 'VWR', 'SQN',
      'Beta系数', '盈亏比'
    ];

    const countMetrics = [
      '总交易次数', '盈利交易', '亏损交易', '最大连胜', '最大连亏'
    ];

    const dayMetrics = [
      '运行天数', '平均持仓'
    ];

    if (typeof value === 'number') {
      if (percentageMetrics.includes(metric.name)) {
        return `${value.toFixed(2)}%`;
      } else if (moneyMetrics.includes(metric.name)) {
        return formatMoney(value, symbol);
      } else if (ratioMetrics.includes(metric.name)) {
        return value.toFixed(2);
      } else if (countMetrics.includes(metric.name)) {
        return value.toLocaleString();
      } else if (dayMetrics.includes(metric.name)) {
        return `${value}天`;
      }
    }

    return value;
  };

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
                  {formatValue(metric)}
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

function RecentTradesSection({ trades, symbol }: { trades: Trade[]; symbol: string }) {
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

    const pnlText = formatMoney(trade.profitLoss, symbol);
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
                  <TableCell>
                    {formatMoney(trade.value, symbol)}
                  </TableCell>
                  <TableCell>
                    {formatProfitLoss(trade)}
                  </TableCell>
                  <TableCell>
                    {formatMoney(trade.totalValue, symbol)}
                  </TableCell>
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

