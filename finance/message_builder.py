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
        message_type: 消息类型（markdown/text）

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
        f"注意：此消息由自动交易系统生成，仅供参考。请结合市场情况自行判断。"
    )

    # 根据消息类型返回不同格式
    if message_type == "markdown":
        return {
            "title": f"ETF交易信号 - {signal_data['symbol']} - {signal_data['signal_type']}",
            "content": _to_markdown(base_content),
            "level": _get_signal_level(signal_data["signal_type"]),
        }
    else:
        return {
            "title": f"ETF交易信号 - {signal_data['symbol']} - {signal_data['signal_type']}",
            "content": base_content,
            "level": _get_signal_level(signal_data["signal_type"]),
        }


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
