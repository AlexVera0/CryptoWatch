# ============================================================
# src/features/engineering.py — 特征工程（严防未来函数）
#
# 核心防御三原则：
#   1. 所有指标计算完成后统一 shift(1)，确保T时刻只能看到T-1的特征
#   2. Funding/OI 已在 downloader 中 ffill（只向过去），此处无需再处理
#   3. 特征列顺序严格遵循 config.FEATURE_COLUMNS，实盘与训练完全一致
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional
from loguru import logger

import config


def compute_features(
    klines: pd.DataFrame,
    funding: pd.Series,
    oi: pd.DataFrame,
) -> pd.DataFrame:
    """
    计算完整特征集。
    
    【防未来函数核心逻辑】：
    所有技术指标（RSI/MACD/ATR等）本身基于历史数据计算，
    但在和标签对齐时必须做 shift(1)：
      - T 时刻的标签 = T 时刻K线走势
      - T 时刻可用的特征 = T-1 时刻K线关闭后计算的指标
    因此最终特征 DataFrame 中所有技术指标列都需要 shift(1)。
    
    参数：
        klines  : K线数据（open/high/low/close/volume）
        funding : 对齐到K线时间的 Funding Rate
        oi      : 对齐到K线时间的 OI 数据
    
    返回：
        特征 DataFrame（列顺序严格按 config.FEATURE_COLUMNS）
    """
    df = klines.copy()

    # ---- 基础 OHLCV（不需要shift，标签生成后会统一处理）----
    # 注意：open/high/low/close/volume 代表"本根K线"的数据
    # 在 Triple-Barrier 标签中，这些作为"当前状态"，不需要 shift

    # ============================================================
    # 技术指标计算（使用 pandas_ta）
    # ============================================================

    # ---- RSI ----
    df["rsi_14"] = ta.rsi(df["close"], length=config.RSI_PERIOD)

    # ---- MACD ----
    macd_result = ta.macd(
        df["close"],
        fast=config.MACD_FAST,
        slow=config.MACD_SLOW,
        signal=config.MACD_SIGNAL,
    )
    if macd_result is not None and not macd_result.empty:
        df["macd"] = macd_result.iloc[:, 0]         # MACD线
        df["macd_signal"] = macd_result.iloc[:, 1]  # 信号线
        df["macd_hist"] = macd_result.iloc[:, 2]    # 柱状图
    else:
        df["macd"] = np.nan
        df["macd_signal"] = np.nan
        df["macd_hist"] = np.nan

    # ---- ATR（用于动态滑点和障碍设置）----
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=config.ATR_PERIOD)

    # ---- 布林带 ----
    bb_result = ta.bbands(
        df["close"],
        length=config.BBANDS_PERIOD,
        std=config.BBANDS_STD,
    )
    if bb_result is not None and not bb_result.empty:
        df["bb_upper"] = bb_result.iloc[:, 0]   # 上轨
        df["bb_middle"] = bb_result.iloc[:, 1]  # 中轨（MA）
        df["bb_lower"] = bb_result.iloc[:, 2]   # 下轨
        # 价格在布林带中的相对位置（0=下轨，1=上轨）
        band_width = df["bb_upper"] - df["bb_lower"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / band_width.replace(0, np.nan)
    else:
        df["bb_upper"] = np.nan
        df["bb_middle"] = np.nan
        df["bb_lower"] = np.nan
        df["bb_pct"] = np.nan

    # ---- 成交量均线和比率 ----
    df["volume_ma_20"] = ta.sma(df["volume"], length=config.VOLUME_MA_PERIOD)
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"].replace(0, np.nan)

    # ---- 价格动量（收益率）----
    df["return_1h"] = df["close"].pct_change(1)   # 1小时收益率
    df["return_4h"] = df["close"].pct_change(4)   # 4小时收益率
    df["return_24h"] = df["close"].pct_change(24) # 24小时收益率

    # ---- 衍生特征 ----
    df["high_low_ratio"] = df["high"] / df["low"].replace(0, np.nan)      # 振幅
    df["close_open_ratio"] = df["close"] / df["open"].replace(0, np.nan)  # 实体方向

    # ============================================================
    # 合并 Funding Rate 特征（已经是只向过去ffill的，安全）
    # ============================================================
    df["funding_rate"] = funding.reindex(df.index).fillna(0.0)

    # Funding Rate 滚动均值（过去8期 = 64小时）
    df["funding_rate_ma8"] = df["funding_rate"].rolling(8, min_periods=1).mean()

    # 过去24小时累计 funding（做空者实际成本）
    df["funding_cumsum_24h"] = df["funding_rate"].rolling(24, min_periods=1).sum()

    # ============================================================
    # 合并 OI 特征（已经是只向过去ffill的，安全）
    # ============================================================
    if not oi.empty and "oi_value" in oi.columns:
        df["oi_value"] = oi["oi_value"].reindex(df.index).ffill().fillna(0.0)
        df["oi_change_1h"] = df["oi_value"].pct_change(1).replace([np.inf, -np.inf], 0.0)
        df["oi_change_4h"] = df["oi_value"].pct_change(4).replace([np.inf, -np.inf], 0.0)
    else:
        df["oi_value"] = 0.0
        df["oi_change_1h"] = 0.0
        df["oi_change_4h"] = 0.0

    # ============================================================
    # ⚠️ 关键：统一 shift(1) 防未来函数
    # ============================================================
    # 所有技术指标和衍生特征需要向后移动1位
    # 确保：T时刻的预测，只能使用 T-1 时刻K线收盘后的数据
    # 原始 OHLCV 保留（用于 Triple-Barrier 标签计算）
    INDICATOR_COLS = [
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "atr_14", "bb_upper", "bb_middle", "bb_lower", "bb_pct",
        "volume_ma_20", "volume_ratio",
        "return_1h", "return_4h", "return_24h",
        "funding_rate", "funding_rate_ma8", "funding_cumsum_24h",
        "oi_value", "oi_change_1h", "oi_change_4h",
        "high_low_ratio", "close_open_ratio",
    ]

    for col in INDICATOR_COLS:
        if col in df.columns:
            df[col] = df[col].shift(1)  # ⚠️ 防未来函数核心操作

    # 去掉头部 NaN（shift 导致的）
    df = df.dropna(subset=["rsi_14", "macd", "atr_14"])

    logger.debug(f"特征计算完成: {len(df)} 行，{len(df.columns)} 列")
    return df


def align_feature_columns(
    df: pd.DataFrame,
    expected_columns: list = None,
) -> pd.DataFrame:
    """
    实盘特征列顺序对齐（要求50）。
    
    训练时固定特征列顺序，实盘预测时必须完全一致。
    缺失列补 0，多余列丢弃。
    
    参数：
        df              : 实盘计算出的特征 DataFrame
        expected_columns: 期望的列顺序（None=使用config.FEATURE_COLUMNS）
    
    返回：
        列顺序与训练完全一致的 DataFrame
    """
    if expected_columns is None:
        expected_columns = config.FEATURE_COLUMNS

    result = pd.DataFrame(index=df.index)

    for col in expected_columns:
        if col in df.columns:
            result[col] = df[col]
        else:
            # 缺失列补0（要求50：缺失列必须补0，不能错位）
            logger.warning(f"[features] 缺失特征列 '{col}'，补0处理")
            result[col] = 0.0

    return result
