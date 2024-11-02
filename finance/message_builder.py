from typing import Any, Dict, Optional


from finance.config import get_etf_name


def build_title(signal_data: Dict[str, Any]) -> str:
    """
    构建统一的通知标题

    Args:
        signal_data: 信号数据字典，必须包含 date, signal_type, name, symbol 字段

    Returns:
        格式化的标题字符串

    Raises:
        ValueError: 当无法获取ETF中文名称时抛出异常
    """
    symbol = signal_data["symbol"]
    name = get_etf_name(symbol)
    if not name:
        raise ValueError(f"无法获取ETF {symbol} 的中文名称")

    return (
        f"{name}({symbol}) - "
        f"{signal_data['date']} - "
        f"{signal_data['signal_type']}"
    )


def build_message(
    signal_data: Dict[str, Any],
    holding_data: Optional[Dict[str, Any]] = None,
    message_type: str = "markdown",
) -> Dict[str, Any]:
    """
    构建通知消息内容

    Args:
        signal_data: 信号数据字典
        holding_data: 持仓数据字典（如果有持仓）
        message_type: 消息类型（markdown/html/text）

    Returns:
        包含消息内容的字典

    Raises:
        ValueError: 当无法获取ETF中文名称时抛出异常
    """
    # 确保能获取到ETF中文名称
    symbol = signal_data["symbol"]
    name = get_etf_name(symbol)
    if not name:
        raise ValueError(f"无法获取ETF {symbol} 的中文名称")

    # 更新signal_data中的name字段
    signal_data["name"] = name

    # 准备持仓详情
    if holding_data:
        holding_details = (
            f"买入时间: {holding_data['买入时间']}\n"
            f"买入价格: {holding_data['买入价格']:.3f}\n"
            f"当前收益: {holding_data['当前收益']:.2%}\n"
            f"最高价格: {holding_data['最高价格']:.3f}\n"
            f"当前价格: {holding_data['当前价格']:.3f}"
        )
    else:
        holding_details = "当前无持仓"

    # 为持有和买入信号添加次日止损提示
    next_day_notice = ""
    if signal_data["signal_type"] in ["持有", "买入"]:
        current_price = signal_data["price"]
        atr = signal_data["atr"]
        atr_multiplier = signal_data["atr_multiplier"]
        stop_loss_pct = signal_data["stop_loss"]

        # 计算两种止损价格
        atr_stop = current_price - (atr * atr_multiplier)
        pct_stop = current_price * (1 - stop_loss_pct)
        final_stop = max(atr_stop, pct_stop)

        next_day_notice = (
            f"\n次日止损提示:\n"
            f"当前价格: {current_price:.3f}\n"
            f"ATR止损价: {atr_stop:.3f} (当前价格 - {atr_multiplier}倍ATR)\n"
            f"比例止损价: {pct_stop:.3f} (当前价格 × (1 - {stop_loss_pct:.1%}))\n"
            f"最终止损价: {final_stop:.3f} (取两者较高者)\n\n"
            f"注意：如果次日价格在不超过当前价格的情况下回落到 {final_stop:.3f}，"
            f"将触发止损卖出。如果价格创新高，止损价格会相应提高。"
        )

    # 构建基础消息内容
    base_content = (
        f"ETF交易信号 - {signal_data['name']} ({signal_data['symbol']})\n\n"
        f"日期: {signal_data['date']}\n"
        f"当前价格: {signal_data['price']:.3f}\n\n"
        f"信号类型: {signal_data['signal_type']}\n\n"
        f"信号详情:\n{signal_data['signal_details']}\n\n"
        f"持仓状态:\n{signal_data['position_details']}\n"
        f"{holding_details}\n\n"
        f"策略参数:\n"
        f"- MA{signal_data['short_period']}: {signal_data['sma_short']:.3f}\n"
        f"- MA{signal_data['long_period']}: {signal_data['sma_long']:.3f}\n"
        f"- ATR({signal_data['atr_period']}): {signal_data['atr']:.3f}\n"
        f"- ATR倍数: {signal_data['atr_multiplier']}\n"
        f"- 止损比例: {signal_data['stop_loss']:.1%}\n\n"
        f"说明: {signal_data['signal_description']}\n\n"
        f"{next_day_notice}\n\n"
    )

    # 添加交易记录（如果有）
    if "trade_table" in signal_data:
        base_content += f"{signal_data['trade_table']}\n\n"

    base_content += "注意：此消息由自动交易系统生成，仅供参考。请结合市场情况自行判断。"

    # 使用统一的标题生成函数
    title = build_title(signal_data)
    level = _get_signal_level(signal_data["signal_type"])

    # 根据消息类型返回不同格式的内容
    if message_type == "html":  # 邮件使用 HTML
        html_content = _to_html(
            base_content, signal_data, holding_details, next_day_notice
        )
        return {
            "title": title,
            "content": html_content,
            "level": level,
            "msgtype": "html",
        }
    elif message_type == "markdown":  # 企业微信使用 markdown
        markdown_content = _to_markdown(base_content)
        return {
            "title": title,
            "content": markdown_content,
            "level": level,
            "msgtype": "markdown",
        }
    else:  # 纯文本
        return {
            "title": title,
            "content": base_content,
            "level": level,
            "msgtype": "text",
        }


def _to_html(
    base_content: str,
    signal_data: Dict[str, Any],
    holding_details: str,
    next_day_notice: str,
) -> str:
    """将基础内容转换为HTML格式"""
    # 构建基本HTML内容
    base_html = [
        "<h1>ETF交易信号 - ",
        f"{signal_data['name']} ({signal_data['symbol']})",
        "</h1>",
        '<div class="signal-box">',
        "<p>日期: ",
        str(signal_data["date"]),
        "</p>",
        "<p>当前价格: ",
        f"{signal_data['price']:.3f}",
        "</p>",
        "<p>信号类型: <span class='",
        signal_data["signal_type"].lower(),
        "'>",
        signal_data["signal_type"],
        "</span></p>",
        "</div>",
        "<h2>信号详情</h2>",
        '<div class="signal-box">',
        signal_data["signal_details"].replace("\n", "<br>"),
        "</div>",
        "<h2>持仓状态</h2>",
        '<div class="signal-box">',
        signal_data["position_details"],
        "<br>",
        holding_details.replace("\n", "<br>"),
        "</div>",
        "<h2>策略参数</h2>",
        '<table class="params-table">',
        "<tr><th>参数</th><th>值</th></tr>",
        "<tr><td>MA",
        str(signal_data["short_period"]),
        "</td><td>",
        f"{signal_data['sma_short']:.3f}",
        "</td></tr>",
        "<tr><td>MA",
        str(signal_data["long_period"]),
        "</td><td>",
        f"{signal_data['sma_long']:.3f}",
        "</td></tr>",
        "<tr><td>ATR(",
        str(signal_data["atr_period"]),
        ")</td><td>",
        f"{signal_data['atr']:.3f}",
        "</td></tr>",
        "<tr><td>ATR倍数</td><td>",
        str(signal_data["atr_multiplier"]),
        "</td></tr>",
        "<tr><td>止损比例</td><td>",
        f"{signal_data['stop_loss']:.1%}",
        "</td></tr>",
        "</table>",
        "<h2>说明</h2>",
        '<div class="signal-box">',
        signal_data["signal_description"],
        "</div>",
    ]
    base_html = "".join(base_html)

    # 添加次日止损提示（如果有）
    stop_loss_html = ""
    if next_day_notice:
        stop_loss_html = "".join(
            [
                "<h2>次日止损提示</h2>",
                '<div class="notice">',
                next_day_notice.replace("\n", "<br>"),
                "</div>",
            ]
        )

    # 添加交易记录（如果有）
    trade_html = ""
    if "trade_table" in signal_data:
        trade_html = "".join(
            [
                "<h2>交易记录</h2>",
                _ascii_table_to_html(signal_data["trade_table"]),
            ]
        )

    # 添加注意事项
    notice_html = "".join(
        [
            '<div class="notice">',
            "注意：此消息由自动交易系统生成，仅供参考。请结合市场情况自行判断。",
            "</div>",
        ]
    )

    # 组合所有HTML部分
    style = """
        <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
        h1 { color: #1a73e8; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 20px; }
        h2 { color: #666; margin-top: 25px; margin-bottom: 15px; }
        .signal-box { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; 
                     border-left: 4px solid #1a73e8; }
        .buy { color: #1e88e5; font-weight: bold; }
        .sell { color: #e53935; font-weight: bold; }
        .hold { color: #43a047; font-weight: bold; }
        .params-table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        .params-table th, .params-table td { border: 1px solid #dee2e6; padding: 12px; 
                                           text-align: left; }
        .params-table th { background-color: #f8f9fa; font-weight: bold; }
        .trade-table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 14px; }
        .trade-table th { background-color: #f8f9fa; border: 1px solid #dee2e6; padding: 12px; 
                         text-align: left; font-weight: bold; }
        .trade-table td { border: 1px solid #dee2e6; padding: 12px; }
        .trade-table tr:nth-child(even) { background-color: #f8f9fa; }
        .trade-table .number { text-align: right; font-family: monospace; }
        .notice { background-color: #fff3e0; padding: 15px; border-radius: 5px; 
                 border-left: 4px solid #ff9800; margin: 15px 0; }
        .signal-type { font-size: 1.2em; font-weight: bold; padding: 5px 10px; border-radius: 3px; 
                      background-color: #e8f0fe; display: inline-block; margin: 10px 0; }
        </style>
    """

    return "".join(
        [
            "<html>",
            "<head>",
            style,
            "</head>",
            "<body>",
            base_html,
            stop_loss_html,
            trade_html,
            notice_html,
            "</body>",
            "</html>",
        ]
    )


def _ascii_table_to_html(ascii_table: str) -> str:
    """将ASCII表格转换为HTML表格"""
    if not ascii_table or "没有发生交易" in ascii_table:
        return "<p>没有发生交易</p>"

    rows = ascii_table.split("\n")
    html_table = '<table class="trade-table">\n'

    # 处理表头
    header_row = [cell.strip() for cell in rows[1].split("|")[1:-1]]
    html_table += "<thead>\n<tr>\n"
    for header in header_row:
        html_table += f"<th>{header}</th>\n"
    html_table += "</tr>\n</thead>\n<tbody>\n"

    # 处理数据行
    for row in rows[3:-1]:  # 跳过表头和分隔线
        if "+-" in row:  # 跳过分线
            continue
        cells = [cell.strip() for cell in row.split("|")[1:-1]]
        html_table += "<tr>\n"
        for i, cell in enumerate(cells):
            # 根据列的类型添加不同的样式
            if i == 0:  # 类型列
                class_name = cell.lower()
                html_table += f'<td class="{class_name}">{cell}</td>\n'
            elif i in [2, 4, 5, 6, 7]:  # 数值列
                html_table += f'<td class="number">{cell}</td>\n'
            else:
                html_table += f"<td>{cell}</td>\n"
        html_table += "</tr>\n"

    html_table += "</tbody>\n</table>"
    return html_table


def _get_signal_level(signal_type: str) -> str:
    """获取信号的通知级别"""
    return {
        "买入": "active",
        "卖出": "timeSensitive",
        "持有": "passive",
    }.get(signal_type, "passive")


def _to_markdown(text: str) -> str:
    """将普通文本转换为markdown格式"""
    lines = text.split("\n")
    formatted_lines = []

    # 特殊处理第一行（标题行）
    if lines:
        first_line = lines[0]
        if "ETF交易信号" in first_line:
            # 保持原始格式，确保中文名称和代码都显示
            formatted_lines.append(f"# {first_line}")
        else:
            formatted_lines.append(first_line)

    # 处理剩余行
    for line in lines[1:]:
        # 替换章节标题
        line = line.replace("信号详情:", "### 信号详情")
        line = line.replace("持仓状态:", "### 持仓状态")
        line = line.replace("策略参数:", "### 策略参数")
        line = line.replace("说明:", "### 说明")
        line = line.replace("次日止损提示:", "### 次日止损提示")
        line = line.replace("交易记录:", "### 交易记录")

        # 添加信号类型的颜色标记
        line = line.replace("买入", '<font color="info">买入</font>')
        line = line.replace("卖出", '<font color="warning">卖出</font>')
        line = line.replace("持有", '<font color="comment">持有</font>')

        formatted_lines.append(line)

    return "\n".join(formatted_lines)
