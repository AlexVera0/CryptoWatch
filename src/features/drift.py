# ============================================================
# src/features/drift.py — Concept Drift 检测（PSI）
#
# PSI（Population Stability Index）用于检测特征分布漂移：
# 训练时的特征分布 vs 实盘实时分布
# PSI < 0.1  : 分布稳定，正常运行
# PSI 0.1-0.25: 轻微漂移，警告
# PSI > 0.25 : 严重漂移，建议暂停该合约信号并重训模型
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from loguru import logger

import config


def compute_psi_single(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    计算单个特征的 PSI 值。
    
    PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)
    
    参数：
        expected  : 训练集特征值（基准分布）
        actual    : 实盘特征值（当前分布）
        buckets   : 分桶数量
        epsilon   : 防止除零的最小值
    
    返回：
        PSI 值（越大表示漂移越严重）
    """
    # 去除 NaN
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # 基于训练集数据确定分桶边界
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)  # 去除重复边界

    if len(breakpoints) < 2:
        return 0.0

    # 计算各桶比例
    def get_bucket_pcts(data, bps):
        counts = np.histogram(data, bins=bps)[0]
        total = counts.sum()
        pcts = counts / total if total > 0 else np.zeros(len(counts))
        return np.clip(pcts, epsilon, None)  # 防止为0导致log无穷

    expected_pcts = get_bucket_pcts(expected, breakpoints)
    actual_pcts = get_bucket_pcts(actual, breakpoints)

    # Normalize
    expected_pcts = expected_pcts / expected_pcts.sum()
    actual_pcts = actual_pcts / actual_pcts.sum()

    # PSI 计算
    psi = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))

    return float(psi)


def compute_feature_psi(
    train_features: pd.DataFrame,
    live_features: pd.DataFrame,
    feature_cols: list = None,
) -> Dict[str, float]:
    """
    计算所有特征的 PSI，返回字典。
    
    参数：
        train_features: 训练集特征（基准分布）
        live_features : 实盘最近特征窗口（当前分布）
        feature_cols  : 计算哪些特征的PSI（None=全部）
    
    返回：
        {feature_name: psi_value}
    """
    if feature_cols is None:
        feature_cols = [c for c in train_features.columns if not c.startswith("_")]

    psi_dict = {}
    for col in feature_cols:
        if col not in train_features.columns or col not in live_features.columns:
            continue

        psi = compute_psi_single(
            train_features[col].values,
            live_features[col].values,
        )
        psi_dict[col] = round(psi, 6)

    return psi_dict


def evaluate_drift_status(
    psi_dict: Dict[str, float],
    warning_threshold: float = None,
    critical_threshold: float = None,
) -> Tuple[str, Dict[str, str]]:
    """
    根据 PSI 评估 Concept Drift 状态。
    
    返回：
        (overall_status, feature_status_dict)
        overall_status: "stable" / "warning" / "critical"
        feature_status_dict: 每个特征的状态
    """
    if warning_threshold is None:
        warning_threshold = config.PSI_WARNING_THRESHOLD
    if critical_threshold is None:
        critical_threshold = config.PSI_CRITICAL_THRESHOLD

    feature_status = {}
    has_warning = False
    has_critical = False

    for feat, psi_val in psi_dict.items():
        if psi_val >= critical_threshold:
            feature_status[feat] = "critical"
            has_critical = True
        elif psi_val >= warning_threshold:
            feature_status[feat] = "warning"
            has_warning = True
        else:
            feature_status[feat] = "stable"

    if has_critical:
        overall = "critical"
        logger.warning(
            f"[drift] ⚠️ Concept Drift 严重！"
            f"以下特征PSI超过{critical_threshold}: "
            f"{[k for k,v in feature_status.items() if v=='critical']}"
        )
    elif has_warning:
        overall = "warning"
        logger.info(
            f"[drift] 轻微漂移，以下特征PSI在({warning_threshold},{critical_threshold}): "
            f"{[k for k,v in feature_status.items() if v=='warning']}"
        )
    else:
        overall = "stable"

    return overall, feature_status


def check_symbol_drift(
    symbol: str,
    train_features: pd.DataFrame,
    recent_features: pd.DataFrame,
    window: int = 168,  # 最近168根K线（约1周）
) -> dict:
    """
    对单个合约进行 Concept Drift 检测。
    
    参数：
        symbol         : 合约代码
        train_features : 训练集特征
        recent_features: 实盘最近特征
        window         : 取最近多少根K线作为"当前分布"
    
    返回：
        {
          "symbol": ...,
          "overall_status": "stable/warning/critical",
          "psi_dict": {...},
          "feature_status": {...},
          "max_psi": ...,
          "action": "continue/watch/pause",
        }
    """
    # 取最近 window 根K线作为当前分布
    live_window = recent_features.tail(window)

    psi_dict = compute_feature_psi(train_features, live_window)
    overall_status, feature_status = evaluate_drift_status(psi_dict)

    max_psi = max(psi_dict.values()) if psi_dict else 0.0
    avg_psi = np.mean(list(psi_dict.values())) if psi_dict else 0.0

    # 决策建议
    if overall_status == "critical":
        action = "pause"   # 暂停此合约信号，建议重新训练
    elif overall_status == "warning":
        action = "watch"   # 监控，提高警惕
    else:
        action = "continue"

    result = {
        "symbol": symbol,
        "overall_status": overall_status,
        "psi_dict": psi_dict,
        "feature_status": feature_status,
        "max_psi": round(max_psi, 6),
        "avg_psi": round(avg_psi, 6),
        "action": action,
    }

    logger.debug(
        f"[drift] {symbol}: {overall_status.upper()} | "
        f"max_psi={max_psi:.4f} | action={action}"
    )

    return result
