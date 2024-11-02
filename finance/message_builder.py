from typing import Any, Dict, Optional

from finance.config import get_etf_name
from finance.logger import logger


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
        包含消息容的字典

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
    if signal_data.get("position_details", "").startswith("当前持仓"):
        holding_details = signal_data["position_details"]  # 直接使用完整的持仓信息
    else:
        holding_details = "当前无持仓"

    # 构建基础消息内容
    base_content = (
        f"ETF交易信号 - {signal_data['name']} ({signal_data['symbol']}) - {signal_data['signal_type']}\n\n"
        f"日期: {signal_data['date']}\n"
        f"回测开始: {signal_data['start_date']}\n"
        f"当前价格: {signal_data['price']:.3f}\n\n"
        f"信号类型: {signal_data['signal_type']}\n\n"
        f"信号详情:\n{signal_data['signal_details']}\n\n"  # 这里直接使用了signal_details
        f"持仓状态:\n{holding_details}\n\n"
        f"策略参数:\n"
        f"- MA{signal_data['short_period']}: {signal_data['sma_short']:.3f}\n"
        f"- MA{signal_data['long_period']}: {signal_data['sma_long']:.3f}\n"
        f"- ATR({signal_data['atr_period']}): {signal_data['atr']:.3f}\n"
        f"- ATR倍数: {signal_data['atr_multiplier']}\n"
        f"- 止损比例: {signal_data['stop_loss']:.1%}\n"
    )

    # 移除多余的说明
    if signal_data["signal_description"] != "策略回测结果":
        base_content += f"说明: {signal_data['signal_description']}\n\n"

    # 添加交易记录（如果有）
    if "trade_table" in signal_data:
        base_content += f"{signal_data['trade_table']}\n\n"

    base_content += "注意：此消息由自动交易系统生成，仅供参考。请结合市场情况自行判断。"

    # 使用统一的标题生成函数
    title = build_title(signal_data)
    level = _get_signal_level(signal_data["signal_type"])

    # 根据消息类型返回不同格式的内容
    if message_type == "html":  # 邮件使用 HTML
        html_content = _to_html(base_content)  # 只传入基础内容，让函数只负责格式转换
        return {
            "title": title,
            "content": html_content,
            "level": level,
            "msgtype": "html",
        }
    elif message_type == "markdown":  # 企业微信使用 markdown
        markdown_content = _to_markdown(
            base_content
        )  # 只传入基础内容，让函数只负责格式转换
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


def _to_html(base_content: str) -> str:
    """将基础内容转换为HTML格式，只负责格式转换"""
    # 添加样式
    style = """
        <style>
        body { 
            font-family: Arial, sans-serif; 
            line-height: 1.6; 
            margin: 0; 
            padding: 20px; 
            color: #333; 
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: #fff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #1a73e8; 
            border-bottom: 2px solid #eee; 
            padding-bottom: 10px; 
            margin-bottom: 20px;
            font-size: 24px;
        }
        h2 { 
            color: #666; 
            margin-top: 25px; 
            margin-bottom: 15px;
            font-size: 20px;
            border-left: 4px solid #1a73e8;
            padding-left: 10px;
        }
        .section {
            background-color: #fff;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
        }
        .signal-box { 
            background-color: #f8f9fa; 
            padding: 20px; 
            border-radius: 5px; 
            margin: 15px 0;
            border-left: 4px solid #1a73e8;
        }
        .buy { 
            color: #1e88e5; 
            font-weight: bold;
            background-color: #e3f2fd;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .sell { 
            color: #e53935; 
            font-weight: bold;
            background-color: #ffebee;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .hold { 
            color: #43a047; 
            font-weight: bold;
            background-color: #e8f5e9;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .notice { 
            background-color: #fff3e0; 
            padding: 15px; 
            border-radius: 5px;
            border-left: 4px solid #ff9800; 
            margin: 15px 0;
        }
        .params {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }
        .param-item {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }
        .trade-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
            background-color: #fff;
        }
        .trade-table th {
            background-color: #1a73e8;
            color: white;
            padding: 12px;
            text-align: left;
            border: 1px solid #e0e0e0;
        }
        .trade-table td {
            padding: 10px;
            border: 1px solid #e0e0e0;
            text-align: left;
        }
        .trade-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .trade-table tr:hover {
            background-color: #f5f5f5;
        }
        .trade-table .number {
            text-align: right;
        }
        .performance {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .performance-item {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }
        .performance-item .label {
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }
        .performance-item .value {
            font-size: 18px;
            font-weight: bold;
            color: #1a73e8;
        }
        </style>
    """

    # 将基础内容按章节分割
    sections = base_content.split("\n\n")
    formatted_sections = []

    for section in sections:
        if section.startswith("ETF交易信号"):
            # 标题部分
            formatted_sections.append(f"<h1>{section}</h1>")
        elif "信号详情:" in section:
            # 信号详情部分
            title, content = section.split(":", 1)
            content_html = content.replace("\n", "<br>")
            formatted_sections.append(
                f'<div class="section"><h2>{title}</h2>{content_html}</div>'
            )
        elif "持仓状态:" in section:
            # 持仓状态部分
            title, content = section.split(":", 1)
            content_html = content.replace("\n", "<br>")
            formatted_sections.append(
                f'<div class="section"><h2>{title}</h2>{content_html}</div>'
            )
        elif "策略参数:" in section:
            # 策略参数部分
            title, content = section.split(":", 1)
            params = content.strip().split("\n")
            params_html = '<div class="params">'
            for param in params:
                if param.strip():
                    params_html += f'<div class="param-item">{param}</div>'
            params_html += "</div>"
            formatted_sections.append(
                f'<div class="section"><h2>{title}</h2>{params_html}</div>'
            )
        elif "策略表现:" in section:
            # 策略表现部分
            title, content = section.split(":", 1)
            metrics = content.strip().split("\n")
            metrics_html = '<div class="performance">'
            for metric in metrics:
                if metric.strip():
                    # 检查是否是有效的指标行
                    if ": " in metric:
                        label, value = metric.replace("- ", "").split(": ", 1)
                        metrics_html += f"""
                            <div class="performance-item">
                                <div class="label">{label}</div>
                                <div class="value">{value}</div>
                            </div>
                        """
                    else:
                        # 如果不是标准格式的指标行，作为普通文本显示
                        metrics_html += (
                            f'<div class="text">{metric.replace("- ", "")}</div>'
                        )
            metrics_html += "</div>"
            formatted_sections.append(
                f'<div class="section"><h2>{title}</h2>{metrics_html}</div>'
            )
        elif "交易记录:" in section:
            # 交易记录部分
            title, content = section.split(":", 1)
            # 将ASCII表格转换为HTML表格
            rows = content.strip().split("\n")
            table_html = '<table class="trade-table">'

            # 处理表头
            header_row = rows[1].strip()  # 第二行是表头
            headers = [h.strip() for h in header_row.split("|") if h.strip()]
            table_html += "<thead><tr>"
            for header in headers:
                table_html += f"<th>{header}</th>"
            table_html += "</tr></thead>"

            # 处理数据行
            table_html += "<tbody>"
            for row in rows[3:-1]:  # 跳过表头和分隔行
                if row.strip() and not all(c in "+-|" for c in row):  # 跳过分隔行
                    cells = [cell.strip() for cell in row.split("|") if cell.strip()]
                    table_html += "<tr>"
                    # 根据列的类型设置不同的样式
                    for i, cell in enumerate(cells):
                        # 数字列使用右对齐
                        if i in [
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                        ]:  # 价格、数量、资金、手续费、收益、剩余资金列
                            table_html += f'<td class="number">{cell}</td>'
                        else:
                            table_html += f"<td>{cell}</td>"
                    table_html += "</tr>"
            table_html += "</tbody></table>"

            formatted_sections.append(
                f'<div class="section"><h2>{title}</h2>{table_html}</div>'
            )
        elif "建议买入" in section:
            # 建议买入部分，作为普通内容处理
            content_html = section.replace("\n", "<br>")
            formatted_sections.append(f'<div class="section">{content_html}</div>')
        else:
            # 其他内容
            content_html = section.replace("\n", "<br>")
            formatted_sections.append(f'<div class="section">{content_html}</div>')

    # 替换信号类型的标记
    content_html = "\n".join(formatted_sections)
    content_html = content_html.replace("买入", '<span class="buy">买入</span>')
    content_html = content_html.replace("卖出", '<span class="sell">卖出</span>')
    content_html = content_html.replace("持有", '<span class="hold">持有</span>')

    # 组装完整的HTML
    return "".join(
        [
            "<html>",
            "<head>",
            style,
            "</head>",
            "<body>",
            '<div class="container">',
            content_html,
            "</div>",
            "</body>",
            "</html>",
        ]
    )


def _to_markdown(text: str) -> str:
    """将基础内容转换为markdown格式，只负责格式转换"""
    lines = text.split("\n")
    formatted_lines = []

    # 特殊处理第一行（标题行）
    if lines:
        first_line = lines[0]
        if "ETF交易信号" in first_line:
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


def _get_signal_level(signal_type: str) -> str:
    """获取信号的通知级别"""
    return {
        "买入": "active",
        "卖出": "timeSensitive",
        "持有": "passive",
    }.get(signal_type, "passive")
