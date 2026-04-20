# ============================================================
# src/models/predictor.py — 实盘预测（特征列严格对齐）
#
# 要求50：实盘特征列顺序必须和训练完全一致
#         缺失列补0，多余列丢弃，绝不错位
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from loguru import logger

import config
from src.features.engineering import align_feature_columns


def predict_short_probability(
    symbol: str,
    features_row: pd.Series,
) -> Optional[Tuple[float, str]]:
    """
    实盘预测：对单根K线计算做空概率。

    参数：
        symbol       : 合约代码
        features_row : 单行特征（pd.Series，index=特征名）

    返回：
        (probability, model_version) 或 None（模型不存在）

    【要求50 实现】：
    1. 按 config.FEATURE_COLUMNS 严格排序
    2. 缺失列填0，多余列丢弃
    3. 类型统一转 float32，防止精度差异
    """
    from src.models.trainer import load_model
    result = load_model(symbol)
    if result is None:
        return None
    model, meta = result

    expected_cols = meta.get("feature_columns", config.FEATURE_COLUMNS)
    model_version = meta.get("model_version", "unknown")

    # ---- 严格列对齐（要求50）----
    row_dict = features_row.to_dict() if hasattr(features_row, "to_dict") else dict(features_row)

    aligned = []
    missing_cols = []
    for col in expected_cols:
        val = row_dict.get(col, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            aligned.append(0.0)   # 缺失列补0
            missing_cols.append(col)
        else:
            aligned.append(float(val))

    if missing_cols:
        logger.warning(f"[predictor] {symbol} 缺失特征列 {missing_cols}，已补0")

    X = np.array([aligned], dtype=np.float32)

    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception as e:
        logger.error(f"[predictor] {symbol} 预测失败: {e}")
        return None

    logger.debug(f"[predictor] {symbol} 做空概率={prob:.4f} 版本={model_version}")
    return prob, model_version


def check_indicator_signals(features_row: pd.Series) -> Dict[str, bool]:
    """
    逐项检查技术指标信号（用于邮件报告中的指标通过情况）。

    返回：{指标名: 是否通过}
    """
    checks = {}

    rsi = features_row.get("rsi_14", 50.0)
    checks["RSI超买(>70)"] = float(rsi) > 70.0

    macd_hist = features_row.get("macd_hist", 0.0)
    checks["MACD死叉(hist<0)"] = float(macd_hist) < 0.0

    bb_pct = features_row.get("bb_pct", 0.5)
    checks["BB超买(>0.9)"] = float(bb_pct) > 0.9

    funding = features_row.get("funding_rate", 0.0)
    checks["Funding偏高(>0.01%)"] = float(funding) > 0.0001

    oi_change = features_row.get("oi_change_1h", 0.0)
    checks["OI下降(<0)"] = float(oi_change) < 0.0

    volume_ratio = features_row.get("volume_ratio", 1.0)
    checks["量能异常(>2x均量)"] = float(volume_ratio) > 2.0

    return checks
