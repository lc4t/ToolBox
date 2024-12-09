"use client"

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"

interface TradeSignal {
  action: 'BUY' | 'SELL' | 'HOLD'
  asset: string
  price: number
  timestamp: string
  reason: string
}

interface PositionInfo {
  initialCapital: number
  currentValue: number
  totalReturn: number
}

interface Metric {
  name: string
  value: string | number
  description: string
}

interface Trade {
  date: string
  action: 'BUY' | 'SELL'
  price: number
  quantity: number
  value: number
  pnl: number
  totalValue: number
  reason: string
}

interface StrategyParameter {
  name: string
  value: string | number
}

interface QuantTradeReportData {
  symbol: string
  latestSignal: TradeSignal
  positionInfo: PositionInfo
  performanceMetrics: Metric[]
  marketIndicators: Metric[]
  riskMetrics: Metric[]
  recentTrades: Trade[]
  strategyParameters: StrategyParameter[]
}

export default function QuantTradeReport() {
  const [data, setData] = useState<QuantTradeReportData | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      const response = await fetch('/api/quantTradeData')
      const jsonData = await response.json()
      setData(jsonData)
    }

    fetchData()
  }, [])

  if (!data) return <div>Loading...</div>

  return (
    <div className="container mx-auto p-4 space-y-6">
      <Card>
        <CardHeader className="bg-primary text-primary-foreground">
          <CardTitle className="text-2xl">量化交易策略报告 - {data.symbol}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <LatestSignalSection signal={data.latestSignal} />
            <PositionInfoSection info={data.positionInfo} />
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <MetricsSection title="性能指标" metrics={data.performanceMetrics} />
        <MetricsSection title="市场指标" metrics={data.marketIndicators} />
        <MetricsSection title="风险指标" metrics={data.riskMetrics} />
      </div>

      <RecentTradesSection trades={data.recentTrades} />
      <StrategyParametersSection parameters={data.strategyParameters} />
    </div>
  )
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
            <p className="text-sm text-muted-foreground">{new Date(signal.timestamp).toLocaleString()}</p>
          </div>
          <div>
            <Badge variant={signal.action === 'BUY' ? 'default' : signal.action === 'SELL' ? 'destructive' : 'secondary'}>
              {signal.action === 'BUY' ? '买入' : signal.action === 'SELL' ? '卖出' : '观望'}
            </Badge>
            <p className="text-lg font-bold mt-2">¥{signal.price.toFixed(2)}</p>
          </div>
        </div>
        <p className="text-sm mt-2">{signal.reason}</p>
      </CardContent>
    </Card>
  )
}

function PositionInfoSection({ info }: { info: PositionInfo }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>持仓信息</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-muted-foreground">初始资金</p>
            <p className="text-lg font-semibold">¥{info.initialCapital.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">当前价值</p>
            <p className="text-lg font-semibold">¥{info.currentValue.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">总收益</p>
            <p className={`text-lg font-semibold ${info.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {info.totalReturn >= 0 ? '+' : ''}{info.totalReturn.toFixed(2)}%
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function MetricsSection({ title, metrics }: { title: string, metrics: Metric[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[300px]">
          <div className="space-y-4">
            {metrics.map((metric, index) => (
              <div key={index} className="p-4 border rounded">
                <p className="text-sm font-semibold">{metric.name}</p>
                <p className="text-lg font-bold">{metric.value}</p>
                <p className="text-xs text-muted-foreground">{metric.description}</p>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

function RecentTradesSection({ trades }: { trades: Trade[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>最近交易</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px]">
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
                  <TableCell>{trade.date}</TableCell>
                  <TableCell>
                    <Badge variant={trade.action === 'BUY' ? 'default' : 'destructive'}>
                      {trade.action === 'BUY' ? '买入' : '卖出'}
                    </Badge>
                  </TableCell>
                  <TableCell>¥{trade.price.toFixed(2)}</TableCell>
                  <TableCell>{trade.quantity}</TableCell>
                  <TableCell>¥{trade.value.toFixed(2)}</TableCell>
                  <TableCell className={trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}>
                    ¥{trade.pnl.toFixed(2)}
                  </TableCell>
                  <TableCell>¥{trade.totalValue.toFixed(2)}</TableCell>
                  <TableCell>{trade.reason}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

function StrategyParametersSection({ parameters }: { parameters: StrategyParameter[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>策略参数</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[300px]">
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
                  <TableCell>{param.value}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}

