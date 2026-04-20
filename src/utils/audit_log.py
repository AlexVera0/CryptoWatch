# ============================================================
# src/utils/audit_log.py — 信号审计日志（要求52）
# 每次信号触发时，保存完整 feature snapshot、模型版本、
# 输入概率、EV，用于后续复盘和审计
# ============================================================

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger

import config


def _get_snapshot_path(symbol: str, signal_time: datetime) -> Path:
    """生成审计快照文件路径（按日期分目录）"""
    date_str = signal_time.strftime("%Y-%m-%d")
    time_str = signal_time.strftime("%H%M%S")
    dir_path = config.SIGNAL_SNAPSHOT_DIR / date_str
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / f"{symbol}_{time_str}.json"


def save_signal_snapshot(
    symbol: str,
    signal_type: str,            # "short" 或 "long"
    probability: float,          # 模型输出概率
    ev: float,                   # 预期收益 EV
    win_rate: float,             # 历史胜率
    model_version: str,          # 模型版本号
    feature_snapshot: Dict[str, Any],  # 完整特征值
    indicators_check: Dict[str, bool], # 各指标是否通过
    extra: Optional[Dict] = None,
) -> str:
    """
    保存信号完整快照到 JSON 文件。
    
    返回：快照文件路径（字符串）
    
    【设计理由】：
    每个信号都必须记录完整上下文，这样在模型退化或事后复盘时，
    可以精确还原当时的决策依据，找到问题根源。
    """
    now = datetime.now(timezone.utc)

    # 生成特征快照的哈希（用于数据完整性校验）
    feature_json = json.dumps(feature_snapshot, sort_keys=True, default=str)
    feature_hash = hashlib.md5(feature_json.encode()).hexdigest()[:8]

    snapshot = {
        # ---- 基础信息 ----
        "snapshot_id": f"{symbol}_{now.strftime('%Y%m%d_%H%M%S')}_{feature_hash}",
        "symbol": symbol,
        "signal_type": signal_type,
        "timestamp_utc": now.isoformat(),
        "timestamp_cst": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 北京时间

        # ---- 模型决策信息 ----
        "model_version": model_version,
        "probability": round(probability, 6),
        "ev": round(ev, 6),
        "win_rate": round(win_rate, 4),

        # ---- 指标通过情况 ----
        "indicators_check": indicators_check,
        "indicators_pass_count": sum(indicators_check.values()),
        "indicators_total_count": len(indicators_check),

        # ---- 完整特征快照 ----
        "feature_hash": feature_hash,
        "feature_snapshot": feature_snapshot,

        # ---- 配置信息（方便复盘） ----
        "config_snapshot": {
            "cache_version": config.CACHE_VERSION,
            "kline_interval": config.KLINE_INTERVAL,
            "taker_fee_rate": config.TAKER_FEE_RATE,
            "min_ev_threshold": config.MIN_EV_THRESHOLD,
            "min_short_prob": config.MIN_SHORT_PROB,
            "btc_volatile_threshold": config.BTC_VOLATILE_THRESHOLD,
        },

        # ---- 附加信息 ----
        **(extra or {}),
    }

    # 保存 JSON 文件
    path = _get_snapshot_path(symbol, now)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)

    logger.debug(f"[audit_log] 信号快照已保存: {path}")
    return str(path)


def load_recent_snapshots(symbol: str, days: int = 7) -> list[dict]:
    """
    加载最近 N 天的信号快照（用于复盘分析）。
    
    参数：
        symbol: 合约代码
        days  : 加载最近几天的数据
    """
    snapshots = []
    base_dir = config.SIGNAL_SNAPSHOT_DIR

    if not base_dir.exists():
        return snapshots

    # 遍历日期目录
    for date_dir in sorted(base_dir.iterdir(), reverse=True)[:days]:
        if not date_dir.is_dir():
            continue
        for snap_file in date_dir.glob(f"{symbol}_*.json"):
            try:
                with open(snap_file, "r", encoding="utf-8") as f:
                    snapshots.append(json.load(f))
            except Exception as e:
                logger.warning(f"[audit_log] 读取快照失败: {snap_file}, {e}")

    return sorted(snapshots, key=lambda x: x.get("timestamp_utc", ""), reverse=True)
