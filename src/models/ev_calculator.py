# ============================================================
# src/models/ev_calculator.py — EV 预期收益计算（要求49）
#
# EV = 胜率×平均盈利 - (1-胜率)×平均亏损 - 手续费 - funding - 滑点
# 基于真实分布计算，非均值假设
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional
from loguru import logger

import config


def compute_dynamic_slippage(atr: float, price: float) -> float:
    """
    动态滑点（要求回测成本）。
    滑点 = max(基础滑点, ATR × 系数 / 价格)
    大波动行情下 ATR 变大，滑点自动增加。
    """
    if price <= 0 or np.isnan(atr) or atr <= 0:
        return config.BASE_SLIPPAGE_RATE
    atr_slippage = (atr * config.ATR_SLIPPAGE_MULTIPLIER) / price
    return max(config.BASE_SLIPPAGE_RATE, atr_slippage)


def compute_funding_cost(funding_rates: np.ndarray, hold_bars: int) -> float:
    """
    做空持仓期间累计 funding（做空：正funding=收益，负=成本）。
    按实际持仓时间累计，不固定8小时。
    """
    if len(funding_rates) == 0:
        return 0.0
    bars = min(hold_bars, len(funding_rates))
    return float(np.sum(funding_rates[:bars]))


def compute_ev(
    probability: float,
    backtest_pnl: np.ndarray,
    atr: float,
    price: float,
    funding_rates: np.ndarray,
    avg_hold_bars: int,
    side: str = "short",
) -> Dict[str, float]:
    """
    基于真实分布计算 EV（要求49）。

    EV = 胜率×平均盈利 − (1−胜率)×平均亏损 − 手续费 − funding成本 − 滑点

    参数：
        probability   : 模型校准后的做空概率
        backtest_pnl  : 历史回测所有交易 PnL 数组（比例，不含成本）
        atr           : 当前 ATR(14)
        price         : 当前价格
        funding_rates : 预期持仓期间的 funding rate 序列
        avg_hold_bars : 平均持仓K线数
        side          : "short" 或 "long"
    """
    if len(backtest_pnl) == 0:
        return {"ev": 0.0, "win_rate": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "is_positive": False}

    win_mask = backtest_pnl > 0
    win_pnl = backtest_pnl[win_mask]
    loss_pnl = backtest_pnl[~win_mask]

    win_rate = float(probability)                               # 校准后概率
    avg_win = float(np.mean(win_pnl)) if len(win_pnl) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(loss_pnl))) if len(loss_pnl) > 0 else 0.0

    # ---- 成本计算 ----
    fee_cost = 2 * config.TAKER_FEE_RATE                       # 双边手续费
    slippage = 2 * compute_dynamic_slippage(atr, price)        # 双边滑点
    funding_net = compute_funding_cost(
        funding_rates[:avg_hold_bars], avg_hold_bars
    )
    if side != "short":
        funding_net = -funding_net

    # ---- EV 公式（要求49）----
    ev = (
        win_rate * avg_win
        - (1 - win_rate) * avg_loss
        - fee_cost
        - slippage
        + funding_net
    )

    return {
        "ev": round(ev, 6),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "fee_cost": round(fee_cost, 6),
        "slippage_cost": round(slippage, 6),
        "funding_net": round(funding_net, 6),
        "net_cost": round(fee_cost + slippage - funding_net, 6),
        "is_positive": ev > config.MIN_EV_THRESHOLD,
    }


def quick_ev_estimate(
    probability: float,
    atr: float,
    price: float,
    funding_rate: float = 0.0,
    hold_bars: int = None,
) -> Dict[str, float]:
    """
    无历史回测数据时的快速 EV 估算（基于 Triple-Barrier 障碍倍数）。
    标记 method=quick_estimate，提示精度低于真实分布计算。
    """
    if hold_bars is None:
        hold_bars = config.BARRIER_MAX_HOLD_BARS

    atr_rate = atr / price if price > 0 else 0.001
    avg_win = config.BARRIER_LOWER_MULTIPLIER * atr_rate
    avg_loss = config.BARRIER_UPPER_MULTIPLIER * atr_rate
    fee_cost = 2 * config.TAKER_FEE_RATE
    slippage = 2 * compute_dynamic_slippage(atr, price)
    funding_net = funding_rate * (hold_bars / 8)

    ev = probability * avg_win - (1 - probability) * avg_loss - fee_cost - slippage + funding_net

    return {
        "ev": round(ev, 6),
        "win_rate": round(probability, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "fee_cost": round(fee_cost, 6),
        "slippage_cost": round(slippage, 6),
        "funding_net": round(funding_net, 6),
        "is_positive": ev > config.MIN_EV_THRESHOLD,
        "method": "quick_estimate",
    }
