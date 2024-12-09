import React from 'react';
import { formatDate } from '../utils/formatDate';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"

interface TradeSignal {
  action: '买入' | '卖出' | '观察' | '持有';
  asset: string;
  price: number;
  timestamp: string;
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
  action: 'BUY' | 'SELL';
  price: number;
  quantity: number;
  value: number;
  profitLoss: number;
  totalValue: number;
  reason: string;
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
  performanceMetrics: Metric[];
  riskMetrics: Metric[];
  marketIndicators: Metric[];
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
  performanceMetrics,
  riskMetrics,
  marketIndicators,
  recentTrades,
  strategyParameters,
  showStrategyParameters
}: QuantTradeReportProps) {
  return (
    <div className="container mx-auto p-4 space-y-6">
      <Card>
        <CardHeader className="bg-primary text-primary-foreground">
          <CardTitle className="text-2xl">【{latestSignal.action === '观望' ? '观察' : '持有'}】{name}({symbol}) - {reportDate}</CardTitle>
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
      <MetricsSection title="性能指标" metrics={performanceMetrics} />
      <MetricsSection title="风险指标" metrics={riskMetrics} />
      <MetricsSection title="市场指标" metrics={marketIndicators} />

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
        <div className="flex justify-between items-center">
          <div>
            <p className="text-lg font-semibold">{signal.asset}</p>
            <p className="text-sm text-muted-foreground">{formatDate(signal.timestamp)}</p>
          </div>
          <div>
            <Badge
              variant={
                signal.action === '买入'
                  ? 'default'
                  : signal.action === '卖出'
                    ? 'destructive'
                    : signal.action === '平仓'
                      ? 'outline'
                      : 'secondary'
              }
            >
              {signal.action}
            </Badge>
            <p className="text-lg font-bold mt-2">¥{signal.price.toFixed(3)}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function PositionInfoSection({ info }: { info: PositionInfo | null }) {
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

  return (
    <Card>
      <CardHeader>
        <CardTitle>当前持仓</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
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
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>指标</TableHead>
              <TableHead>值</TableHead>
              <TableHead>描述</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {metrics.map((metric, index) => (
              <TableRow key={index}>
                <TableCell>{metric.name}</TableCell>
                <TableCell>{typeof metric.value === 'number' ? metric.value.toFixed(2) : metric.value}</TableCell>
                <TableCell>{metric.description}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function RecentTradesSection({ trades }: { trades: Trade[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>最近交易</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>日期</TableHead>
              <TableHead>操作</TableHead>
              <TableHead>价格</TableHead>
              <TableHead>数量</TableHead>
              <TableHead>价值</TableHead>
              <TableHead>盈亏</TableHead>
              <TableHead>总价值</TableHead>
              <TableHead>原因</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {trades.map((trade, index) => (
              <TableRow key={index}>
                <TableCell>{formatDate(trade.date)}</TableCell>
                <TableCell>
                  <Badge variant={trade.action === 'BUY' ? 'default' : 'destructive'}>
                    {trade.action === 'BUY' ? '买入' : '卖出'}
                  </Badge>
                </TableCell>
                <TableCell>¥{trade.price.toFixed(3)}</TableCell>
                <TableCell>{trade.quantity.toLocaleString()}</TableCell>
                <TableCell>¥{trade.value.toFixed(2)}</TableCell>
                <TableCell className={trade.profitLoss >= 0 ? 'text-green-600' : 'text-red-600'}>
                  ¥{trade.profitLoss.toFixed(2)}
                </TableCell>
                <TableCell>¥{trade.totalValue.toFixed(2)}</TableCell>
                <TableCell>{trade.reason}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
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

