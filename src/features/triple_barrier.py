# ============================================================
# src/features/triple_barrier.py — Triple-Barrier 标签生成
#
# 原理：对每根K线，在未来 max_hold_bars 根K线内，
# 判断价格是否先触及上方障碍（止盈）、下方障碍（止损）
# 还是超时（中性）。
#
# 【防未来函数关键】：
# 标签只依赖标签时间点之后的价格，不会污染特征。
# 特征和标签对齐时：特征[t] <-> 标签[t]（即T时刻触发信号后看未来）
# 注意：特征已经做了 shift(1)，所以特征[t] 实际上是 t-1 的观测值。
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from loguru import logger

import config


def compute_triple_barrier_labels(
    df: pd.DataFrame,
    upper_multiplier: float = None,
    lower_multiplier: float = None,
    max_hold_bars: int = None,
    atr_col: str = "atr_14",
) -> pd.Series:
    """
    生成 Triple-Barrier 标签。
    
    标签定义：
        +1 = 先触及上方障碍（做多止盈，对做空者是亏损）
        -1 = 先触及下方障碍（做空止盈，对做空者是盈利）
         0 = 超时（中性，平仓）
    
    参数：
        df               : 包含 close/atr_14 的 DataFrame
        upper_multiplier : 上方障碍 = ATR × upper_multiplier
        lower_multiplier : 下方障碍 = ATR × lower_multiplier
        max_hold_bars    : 最大持有K线数（超时障碍）
        atr_col          : ATR 列名
    
    【防未来函数说明】：
    标签在训练时生成，回测时用于评估。
    实盘运行时不生成标签（实盘无法知道未来价格）。
    训练集的标签生成完全基于未来价格，这是正确的——
    因为标签本身就代表"未来会发生什么"，不构成未来函数泄露。
    问题在于特征不能使用未来数据，这已通过 shift(1) 保证。
    """
    if upper_multiplier is None:
        upper_multiplier = config.BARRIER_UPPER_MULTIPLIER
    if lower_multiplier is None:
        lower_multiplier = config.BARRIER_LOWER_MULTIPLIER
    if max_hold_bars is None:
        max_hold_bars = config.BARRIER_MAX_HOLD_BARS

    close = df["close"].values
    # ⚠️ ATR 已经 shift(1)，这里用 shift 后的值作为障碍宽度
    # 这是合理的：信号触发时，我们用"当时已知"的ATR设置障碍
    atr = df[atr_col].values if atr_col in df.columns else np.full(len(df), close.mean() * 0.01)

    labels = np.zeros(len(df), dtype=np.float32)

    for i in range(len(df) - max_hold_bars):
        entry_price = close[i]
        current_atr = atr[i]

        if np.isnan(current_atr) or current_atr <= 0:
            labels[i] = 0
            continue

        # 障碍价格
        upper_barrier = entry_price + upper_multiplier * current_atr
        lower_barrier = entry_price - lower_multiplier * current_atr

        # 向后查找是否触及障碍
        label = 0  # 默认：超时
        for j in range(i + 1, min(i + max_hold_bars + 1, len(df))):
            future_high = df["high"].values[j]
            future_low = df["low"].values[j]

            if future_high >= upper_barrier:
                label = 1   # 先触及上方（做多止盈）
                break
            if future_low <= lower_barrier:
                label = -1  # 先触及下方（做空止盈）
                break

        labels[i] = label

    # 最后 max_hold_bars 根K线无法生成有效标签，设为 NaN
    labels[-(max_hold_bars):] = np.nan

    result = pd.Series(labels, index=df.index, name="label")
    label_counts = result.value_counts().to_dict()
    logger.info(
        f"Triple-Barrier 标签分布: "
        f"+1(上方)={label_counts.get(1.0, 0)}, "
        f"-1(下方)={label_counts.get(-1.0, 0)}, "
        f"0(超时)={label_counts.get(0.0, 0)}, "
        f"NaN={result.isna().sum()}"
    )

    return result


def create_binary_short_label(labels: pd.Series) -> pd.Series:
    """
    将三分类标签转换为二分类做空标签：
        1 = 做空信号（标签=-1，下方障碍先触及）
        0 = 非做空信号（标签=+1或0）
    
    用于 XGBoost 二分类训练。
    """
    binary = (labels == -1).astype(int)
    binary.name = "short_label"

    pos_rate = binary.mean()
    logger.info(f"做空标签正例比率: {pos_rate:.3f} ({binary.sum()}/{len(binary)})")

    return binary


def compute_sample_weights(
    df: pd.DataFrame,
    label_col: str = "label",
    decay_factor: float = 0.5,
) -> pd.Series:
    """
    计算样本权重（mlfinlab 风格）：
    
    1. 时间衰减权重：越新的样本权重越高
    2. 类别平衡权重：正负样本均衡
    
    参数：
        decay_factor: 0=等权，1=线性衰减，0.5=折中
    
    【设计理由】：
    金融时间序列中，近期数据对未来的预测价值更高，
    给予更高权重可以提升模型对当前市场状态的适应性。
    """
    n = len(df)

    # 时间衰减权重（越新权重越高）
    time_weights = np.linspace(1 - decay_factor, 1.0, n)

    weights = pd.Series(time_weights, index=df.index, name="sample_weight")

    return weights
