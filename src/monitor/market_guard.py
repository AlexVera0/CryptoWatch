# ============================================================
# src/monitor/market_guard.py — 极端行情保护（要求51）
#
# BTC 1小时涨跌幅 > X% 时，全局暂停所有信号
# 防止新闻/爆仓/黑天鹅行情下的误判
# ============================================================

from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Optional
import requests
from loguru import logger

import config
from src.utils.retry import with_retry


@with_retry(max_retries=3, base_delay=1.0, log_prefix="MarketGuard")
def _get_btc_kline() -> Optional[float]:
    """获取 BTC 最近1小时涨跌幅"""
    url = f"{config.BINANCE_FUTURES_BASE_URL}/fapi/v1/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 2}
    resp = requests.get(url, params=params, timeout=config.API_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if len(data) < 2:
        return None
    # 取最近完成的K线
    last_close = float(data[-2][4])
    prev_close = float(data[-3][4]) if len(data) >= 3 else float(data[-2][1])
    pct_change = abs((last_close - prev_close) / prev_close) * 100
    return pct_change


class MarketGuard:
    """
    极端行情保护器。

    逻辑：
    1. 每轮监控前检查 BTC 1h 涨跌幅
    2. 超过阈值 → 触发保护，记录冷却计数
    3. 冷却 N 根K线后自动恢复（要求51）
    """

    def __init__(self):
        self._halted: bool = False
        self._halt_bars_remaining: int = 0
        self._halt_reason: str = ""

    def check(self) -> bool:
        """
        检查是否需要暂停信号。
        返回 True = 正常，可以发信号；False = 极端行情，暂停
        """
        # 冷却倒计时
        if self._halted:
            self._halt_bars_remaining -= 1
            if self._halt_bars_remaining <= 0:
                self._halted = False
                logger.info("[market_guard] 极端行情冷却结束，恢复信号")
                return True
            else:
                logger.warning(
                    f"[market_guard] 极端行情保护中，剩余 {self._halt_bars_remaining} 根K线 | {self._halt_reason}"
                )
                return False

        # 检测 BTC 波动
        try:
            btc_pct = _get_btc_kline()
        except Exception as e:
            logger.warning(f"[market_guard] BTC波动检测失败，默认放行: {e}")
            return True

        if btc_pct is None:
            return True

        threshold = config.BTC_VOLATILE_THRESHOLD
        if btc_pct > threshold:
            self._halted = True
            self._halt_bars_remaining = config.BTC_VOLATILE_COOLDOWN_BARS
            self._halt_reason = f"BTC 1h波动 {btc_pct:.2f}% > {threshold}%"
            logger.warning(
                f"[market_guard] ⚠️ 极端行情！{self._halt_reason}，"
                f"暂停 {config.BTC_VOLATILE_COOLDOWN_BARS} 根K线"
            )
            return False

        logger.debug(f"[market_guard] BTC 1h波动={btc_pct:.2f}%，正常")
        return True

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason
