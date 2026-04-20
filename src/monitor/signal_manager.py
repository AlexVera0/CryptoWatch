# ============================================================
# src/monitor/signal_manager.py — 冷却机制（要求48）
#
# 冷却逻辑（非简单计时）：
#   信号触发 → 发送邮件 → 等待信号条件消失 → 再次出现才允许再次触发
#   即：同一合约必须先"退出"信号状态，才能再次触发
# ============================================================

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional
from loguru import logger


@dataclass
class SignalState:
    """单个合约的信号状态"""
    symbol: str
    is_active: bool = False          # 当前是否处于"信号已触发"状态
    last_trigger_time: Optional[datetime] = None  # 上次触发时间
    trigger_count: int = 0           # 累计触发次数
    last_condition_met: bool = False  # 上一轮是否满足条件（用于检测"消失→重现"）


class SignalManager:
    """
    信号冷却管理器。

    工作原理：
    1. 首次信号满足条件 → 触发 → 标记 is_active=True
    2. 信号条件消失 → last_condition_met=False，is_active 保持
    3. 信号条件再次满足 + 上一轮不满足 → 才允许再次触发
       （要求48：必须信号消失后重新出现，不是计时）
    """

    def __init__(self):
        self._states: Dict[str, SignalState] = {}

    def _get_state(self, symbol: str) -> SignalState:
        if symbol not in self._states:
            self._states[symbol] = SignalState(symbol=symbol)
        return self._states[symbol]

    def should_trigger(self, symbol: str, condition_met: bool) -> bool:
        """
        判断是否应该触发信号。

        参数：
            symbol       : 合约代码
            condition_met: 本轮信号条件是否满足（概率>门槛 且 EV>0）

        返回：
            True = 可以触发（发邮件/记录），False = 冷却中或条件不满足
        """
        state = self._get_state(symbol)

        if not condition_met:
            # 条件不满足 → 标记信号消失（解除冷却锁）
            if state.last_condition_met:
                logger.debug(f"[signal] {symbol} 信号消失，冷却解除")
            state.last_condition_met = False
            state.is_active = False
            return False

        # 条件满足
        if not state.is_active and not state.last_condition_met:
            # 信号全新出现（之前未激活）→ 允许触发
            state.is_active = True
            state.last_condition_met = True
            state.last_trigger_time = datetime.now(timezone.utc)
            state.trigger_count += 1
            logger.info(f"[signal] {symbol} 信号触发（第{state.trigger_count}次）")
            return True

        if state.is_active and state.last_condition_met:
            # 信号持续存在 → 冷却中，不重复触发
            logger.debug(f"[signal] {symbol} 信号持续但已触发，冷却中...")
            state.last_condition_met = True
            return False

        # 其他状态（不应发生）
        state.last_condition_met = condition_met
        return False

    def reset(self, symbol: str):
        """手动重置某合约的信号状态"""
        if symbol in self._states:
            self._states[symbol] = SignalState(symbol=symbol)

    def get_status(self) -> Dict[str, dict]:
        """获取所有合约的信号状态摘要"""
        return {
            sym: {
                "is_active": s.is_active,
                "trigger_count": s.trigger_count,
                "last_trigger": str(s.last_trigger_time) if s.last_trigger_time else None,
                "last_condition_met": s.last_condition_met,
            }
            for sym, s in self._states.items()
        }
