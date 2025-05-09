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
