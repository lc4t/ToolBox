# 量化交易系统

一个完整的量化交易系统，包含数据同步、回测和前端展示功能。系统使用 MySQL 存储数据，支持多种交易策略和回测分析。

## 环境配置

### 后端环境配置

1. 安装 Python 依赖

```bash
# 使用 uv 包管理器安装依赖
uv sync
```

2. 配置环境变量
   复制 `.env.example` 到 `.env`，并填写配置

## 数据同步

### 初始化数据库

```bash
# uv run python db.py init
# 暂无
```

### 同步历史数据

```bash
# 同步指定日期范围的数据
uv run python fetcher.py --start-date 2023-01-01 --end-date 2023-12-31 --symbol XXX
# 特别地，如果你不指定任何时间和symbol，会从DB的symbol_info中获取
```

## 回测系统

### 运行单个策略回测

```bash
# 运行双均线策略回测
uv run python backtest.py \
    --symbol 159915.SZ \
    --start-date 2023-01-01
```

### 参数优化

支持的参数：

-   `short_period`: 短期均线周期
-   `long_period`: 长期均线周期
-   `chandelier-period`: 吊灯止损周期
-   `chandelier-multiplier`: 吊灯止损乘数
-   `adr-period`: ADR 周期
-   `adr-multiplier`: ADR 乘数
-   `initial-capital`: 初始资金
-   `start-date`: 开始日期
-   `output-json`: 输出 json 文件路径

例子：

`uv run python backtest.py 159915.SZ --initial-capital 50000 --start-date 2022-01-01 --use-ma --ma-short 5 --ma-long 8 --use-chandelier --chandelier-multiplier 1.5 --chandelier-period 15 --output-json frontend/trading/data/159915.SZ.json --notify email --email-to lc4t0.0@gmail.com`

`uv run python backtest.py 159915.SZ --initial-capital 50000 --start-date 2022-01-01 --use-ma --ma-short 5 --ma-long 8-15 --use-chandelier --chandelier-multiplier 1.5 --chandelier-period 15-60`

### 分析结果

回测结果将保存在 `logs/backtest.log` 中，包含以下信息：

-   交易记录
-   收益分析
-   风险指标
-   参数优化结果

### 检查结果

-   建议使用 output-json 参数将结果输出到 frontend/trading/data/SYMBOL.json 中
-   然后使用`./deploy_to_worker.sh dev`启动前端，访问`http://localhost:8787/`查看结果

## 交易信号日报

当 JSON 已生成后，可以运行独立汇总命令，把当天所有标的的状态汇总后推送到 PushGo。

### 环境变量

在 `.env` 中增加以下配置：

```bash
PUSHGO_CHANNEL_ID=your-channel-id
PUSHGO_PASSWORD=your-channel-password
# 可选，默认使用官方网关
PUSHGO_URL=https://gateway.pushgo.dev/push
```

### 命令

```bash
uv run python daily_signal_digest.py
```

可选参数：

-   `--data-dir frontend/trading/data`
-   `--date 2026-03-10`
-   `--pushgo-url https://gateway.pushgo.dev/push`
-   `--max-retries 3`
-   `--retry-delay 5`
-   `--dry-run`

### 行为说明

-   默认只扫描 `frontend/trading/data/*.json`
-   只汇总 `reportDate` 等于当天日期的 JSON
-   标题只统计 `买入` 和 `卖出`，`持有` 与 `观望` 仍会展示详情
-   如果没有 JSON，或没有任何当天 JSON，会发送一条 `无任何数据可用` 的通知
-   如果 PushGo 推送失败，会按参数重试；超过重试次数后命令退出非 0

### Dry Run

本地验证格式时可以只输出内容，不发送通知：

```bash
uv run python daily_signal_digest.py --dry-run --date 2026-03-10
```

### GitHub Workflow 调用示例

保持回测脚本不变，只在 JSON 产物生成完成后追加一个独立步骤：

```yaml
- name: Send daily signal digest
  working-directory: trading
  run: uv run python daily_signal_digest.py
  env:
    PUSHGO_CHANNEL_ID: ${{ secrets.PUSHGO_CHANNEL_ID }}
    PUSHGO_PASSWORD: ${{ secrets.PUSHGO_PASSWORD }}
```
