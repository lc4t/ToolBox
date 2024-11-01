from typing import Dict, Optional


def build_message(
    signal_data: Dict,
    holding_data: Optional[Dict] = None,
    message_type: str = "markdown",
) -> Dict:
    """
    构建通知消息内容

    Args:
        signal_data: 信号数据字典
        holding_data: 持仓数据字典（如果有持仓）
        message_type: 消息类型（markdown/html/text）

    Returns:
        包含消息内容的字典
    """
    # 准备持仓详情
    if holding_data:
        holding_details = (
            f"买入时间: {holding_data['买入时间']}\n"
            f"买入价格: {holding_data['买入价格']:.3f}\n"
            f"当前收益: {holding_data['当前收益']:.2%}\n"
            f"最高价格: {holding_data['最高价格']:.3f}\n"
            f"止损价格: {holding_data['止损价格']:.3f}"
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
        f"ETF交易信号 - {signal_data['symbol']}\n\n"
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

    # 修改标题格式
    title = (
        f"ETF-{signal_data['date']}-"
        f"{signal_data['signal_type']}-"
        f"{signal_data.get('name', signal_data['symbol'])}"  # 如果有中文名称就用中文名称，否则用代码
    )
    level = _get_signal_level(signal_data["signal_type"])

    if message_type == "markdown":  # 企业微信使用 markdown
        return {
            "title": title,
            "content": _to_markdown(base_content),
            "level": level,
        }
    elif message_type == "html":  # 邮件使用 HTML
        html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                }}
                h1 {{
                    color: #333;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                h3 {{
                    color: #666;
                    margin-top: 20px;
                }}
                .buy {{
                    color: #1e88e5;
                    font-weight: bold;
                }}
                .sell {{
                    color: #e53935;
                    font-weight: bold;
                }}
                .hold {{
                    color: #43a047;
                    font-weight: bold;
                }}
                .signal-box {{
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .trade-table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 15px 0;
                    font-size: 14px;
                }}
                .trade-table th {{
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    padding: 12px;
                    text-align: left;
                    font-weight: bold;
                    color: #495057;
                }}
                .trade-table td {{
                    border: 1px solid #dee2e6;
                    padding: 12px;
                    text-align: left;
                }}
                .trade-table tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .trade-table tr:hover {{
                    background-color: #f2f2f2;
                }}
                .trade-table .number {{
                    text-align: right;
                    font-family: monospace;
                }}
                .notice {{
                    background-color: #fff3e0;
                    padding: 15px;
                    border-left: 4px solid #ff9800;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            {_to_html(base_content)}
        </body>
        </html>
        """
        return {
            "title": title,
            "content": html_content,
            "level": level,
        }
    else:  # 纯文本
        return {
            "title": title,
            "content": base_content,
            "level": level,
        }


def _to_html(text: str) -> str:
    """将普通文本转换为HTML格式"""
    # 基本格式转换
    text = text.replace("\n", "<br>")

    # 添加标题样式
    text = text.replace("ETF交易信号", "<h1>ETF交易信号</h1>")
    text = text.replace("信号详情:", "<h3>信号详情</h3>")
    text = text.replace("持仓状态:", "<h3>持仓状态</h3>")
    text = text.replace("策略参数:", "<h3>策略参数</h3>")
    text = text.replace("说明:", "<h3>说明</h3>")
    text = text.replace("次日止损提示:", "<h3>次日止损提示</h3>")
    text = text.replace("交易记录:", "<h3>交易记录</h3>")

    # 添加信号类型样式
    text = text.replace("买入", '<span class="buy">买入</span>')
    text = text.replace("卖出", '<span class="sell">卖出</span>')
    text = text.replace("持有", '<span class="hold">持有</span>')

    # 添加信号详情样式
    text = text.replace("信号详情:", '<div class="signal-box">信号详情:')
    text = text.replace("策略参数:", '</div><div class="signal-box">策略参数:')
    text = text.replace("说明:", '</div><div class="signal-box">说明:')

    # 美化交易记录表格
    if "+---------+" in text:  # 检测是否存在表格边框
        # 提取表格内容
        table_start = text.find("+---------+")
        table_end = text.find("\n\n", table_start)
        if table_end == -1:  # 如果是最后一部分
            table_end = len(text)
        table_text = text[table_start:table_end]

        # 将ASCII表格转换为HTML表格
        rows = table_text.split("\n")
        html_table = '<table class="trade-table">\n'

        # 处理表头
        header_row = rows[1]  # 第二行是表头
        headers = [cell.strip() for cell in header_row.split("|")[1:-1]]
        html_table += "<thead>\n<tr>\n"
        for header in headers:
            html_table += f"<th>{header}</th>\n"
        html_table += "</tr>\n</thead>\n<tbody>\n"

        # 处理数据行
        for row in rows[3:-1]:  # 跳过表头和分隔线
            if "+---------+" in row:  # 跳过分隔线
                continue
            cells = [cell.strip() for cell in row.split("|")[1:-1]]
            html_table += "<tr>\n"
            for i, cell in enumerate(cells):
                # 根据列的类型添加不同的样式
                if i == 0:  # 类型列
                    if "买入" in cell:
                        html_table += f'<td class="buy">{cell}</td>\n'
                    elif "卖出" in cell:
                        html_table += f'<td class="sell">{cell}</td>\n'
                    else:
                        html_table += f"<td>{cell}</td>\n"
                elif i in [2, 4, 5, 6]:  # 数值列（价格、资金、手续费、收益）
                    html_table += f'<td class="number">{cell}</td>\n'
                else:
                    html_table += f"<td>{cell}</td>\n"
            html_table += "</tr>\n"

        html_table += "</tbody>\n</table>"

        # 替换原始表格
        text = text[:table_start] + html_table + text[table_end:]

    # 添加注意事项样式
    text = text.replace(
        "注意：此消息由自动交易系统生成",
        '</div><div class="notice">注意：此消息由自动交易系统生成',
    )
    text += "</div>"

    return text


def _to_markdown(text: str) -> str:
    """将普通文本转换为markdown格式"""
    # 添加markdown格式化
    text = text.replace("ETF交易信号", "# ETF交易信号")
    text = text.replace("信号详情:", "### 信号详情")
    text = text.replace("持仓状态:", "### 持仓状态")
    text = text.replace("策略参数:", "### 策略参数")
    text = text.replace("说明:", "### 说明")
    text = text.replace("次日止损提示:", "### 次日止损提示")

    # 添加颜色和格式
    text = text.replace("买入", '<font color="info">买入</font>')
    text = text.replace("卖出", '<font color="warning">卖出</font>')
    text = text.replace("持有", '<font color="comment">持有</font>')

    return text


def _get_signal_level(signal_type: str) -> str:
    """获取信号的通知级别"""
    return {
        "买入": "active",
        "卖出": "timeSensitive",
        "持有": "passive",
    }.get(signal_type, "passive")
