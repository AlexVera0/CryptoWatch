# ============================================================
# src/monitor/pump_alert.py — 暴涨暴跌预警
#
# 每次K线收盘扫描时，额外检查：
#   1h涨幅 > 30%（WARNING）→ 发预警邮件
#   1h涨幅 > 40%（CRITICAL）→ 发紧急邮件
# 独立于做空信号，不影响主监控流程
# ============================================================

from __future__ import annotations
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger

import config


def check_sudden_pump(
    symbol: str,
    klines: pd.DataFrame,
    probability: float,
    win_rate: float,
    indicator_checks: Dict[str, bool],
) -> Optional[Dict[str, Any]]:
    """
    检查短期暴涨（从最新K线计算 30m 和 1h 涨幅）。
    
    参数：
        klines: K线数据 (必须 >= 2 根)
        probability: AI预测做空概率
        win_rate: 预期胜率
        indicator_checks: 指标状态字典
    
    返回：如果触发预警返回 dict，否则 None
    """
    if len(klines) < 2:
        return None

    # 最新收盘价
    current_price = float(klines['close'].iloc[-1])
    
    # 30分钟涨幅 (当前K线的最低点到收盘价)
    low_30m = float(klines['low'].iloc[-1])
    pump_30m = (current_price - low_30m) / low_30m * 100 if low_30m > 0 else 0

    # 1小时涨幅 (最近2根K线的最低点到收盘价)
    low_1h = float(klines['low'].iloc[-2:].min())
    pump_1h = (current_price - low_1h) / low_1h * 100 if low_1h > 0 else 0

    # 取最大涨幅作为判断依据
    max_pump = max(pump_30m, pump_1h)
    pump_period = "30分钟" if pump_30m > pump_1h else "1小时"

    level = None
    if max_pump >= config.PUMP_CRITICAL_THRESHOLD:
        level = "critical"
    elif max_pump >= config.PUMP_WARNING_THRESHOLD:
        level = "warning"

    if level:
        direction = "暴涨 🚀" if max_pump > 0 else "暴跌 🔴"
        
        # 统计指标
        pass_count = sum(indicator_checks.values())
        
        return {
            "symbol": symbol,
            "level": level,
            "direction": direction,
            "max_pump": max_pump,
            "pump_period": pump_period,
            "price": current_price,
            "probability": probability,
            "win_rate": win_rate,
            "pass_count": pass_count,
            "total_indicators": len(indicator_checks),
            "indicator_checks": indicator_checks,
        }
    return None

def build_pump_email_body(alerts: List[Dict[str, Any]]) -> tuple[str, str]:
    """构建暴涨预警邮件（带AI预测信息）"""
    now_cst = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts_sorted = sorted(alerts, key=lambda x: abs(x["max_pump"]), reverse=True)

    critical_count = sum(1 for a in alerts if a["level"] == "critical")
    warning_count = sum(1 for a in alerts if a["level"] == "warning")

    if critical_count > 0:
        subject = f"🚨 紧急行情：{critical_count}个合约短期暴涨 | 微信用户AB699Q | {now_cst}"
    else:
        subject = f"⚠️ 异动预警：{warning_count}个合约短期暴涨 | 微信用户AB699Q | {now_cst}"

    lines = []
    lines.append("════════════════════════════════════")
    lines.append(f"  🚨 短期异动榜单 — 微信用户AB699Q专属")
    lines.append("════════════════════════════════════")
    lines.append("")
    lines.append(f"检测时间：{now_cst} CST")
    lines.append(f"触发合约：{len(alerts)} 个 (阈值>{config.PUMP_WARNING_THRESHOLD}%)")
    lines.append("")

    for a in alerts_sorted:
        tag = "🚨 紧急" if a["level"] == "critical" else "⚠️ 预警"
        lines.append(f"► {tag} {a['symbol']} ({a['pump_period']} {a['direction']})")
        lines.append(f"   最大涨幅：{a['max_pump']:+.1f}%")
        lines.append(f"   当前价格：{a['price']:.6g} USDT")
        
        # 加上AI指标
        lines.append(f"   做空胜率：{a['win_rate']*100:.1f}% (AI概率: {a['probability']*100:.1f}%)")
        lines.append(f"   符合指标：{a['pass_count']}/{a['total_indicators']}")
        
        for name, passed in a['indicator_checks'].items():
            mark = "✓" if passed else "✗"
            lines.append(f"      {mark} {name}")
        lines.append("")

    lines.append("────────────────────────────────────")
    lines.append("💡 操作建议：")
    lines.append(" 1. 暴涨合约极易出现急涨急跌，请查看做空胜率是否达标。")
    lines.append(" 2. 若【符合指标】≥3且胜率较高，可考虑入场空单。")
    lines.append(" 3. 顺势拉升不建议逆势抗单，务必带好止损。")
    lines.append("════════════════════════════════════")

    return subject, "\n".join(lines)
