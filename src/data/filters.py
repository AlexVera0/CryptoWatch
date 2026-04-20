# ============================================================
# src/data/filters.py — 数据过滤器
# 包含：
#   1. 新币样本不足过滤（要求43）
#   2. Survivorship bias 剔除（要求43）
#   3. 流动性三重过滤：volume + spread + OI（实时监控要求）
#   4. Research模式：成交额前N筛选
# ============================================================

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from loguru import logger

import config


def filter_new_coins(
    symbol: str,
    df: pd.DataFrame,
    min_samples: int = None,
) -> Optional[pd.DataFrame]:
    """
    新币样本不足过滤（要求43）。
    
    上线不久的新币历史数据太少，统计规律不可靠，必须过滤。
    
    参数：
        symbol     : 合约代码（用于日志）
        df         : K线数据 DataFrame
        min_samples: 最少需要的K线数量（None=使用config默认值）
    
    返回：
        DataFrame（有效）或 None（过滤掉）
    
    【防 Survivorship Bias 说明】：
    同样地，回测时不能使用"当前仍在上市"的逻辑来筛选合约，
    必须基于该时间点真实存在的合约列表进行回测。
    """
    if min_samples is None:
        min_samples = config.MIN_KLINE_SAMPLES

    if df is None or len(df) < min_samples:
        logger.info(
            f"[filter] {symbol} 样本不足，跳过"
            f"（{len(df) if df is not None else 0} < {min_samples}）"
        )
        return None

    return df


def filter_survivorship_bias(
    symbols_at_time: List[str],
    all_fetched_symbols: List[str],
) -> List[str]:
    """
    回测时剔除 survivorship bias（要求43）。
    
    回测某一历史时间点时，只使用该时间点之前已经上市的合约。
    不能使用"当前还在交易"的合约列表（否则会自动排除已下架的亏损币种）。
    
    参数：
        symbols_at_time    : 特定历史时间点上实际存在的合约列表
        all_fetched_symbols: 当前获取的所有合约列表
    
    返回：
        过滤后的合约列表（只保留当时真实存在的）
    
    【注意】：
    在全量回测时，必须维护一个历史合约上下市记录表。
    此函数是接口占位，完整实现依赖历史数据。
    """
    # 将 all_fetched_symbols 限制为 symbols_at_time 中实际存在的
    valid = [s for s in all_fetched_symbols if s in set(symbols_at_time)]
    removed = len(all_fetched_symbols) - len(valid)
    if removed > 0:
        logger.debug(f"[filter] 剔除 {removed} 个回测时间点不存在的合约（防 survivorship bias）")
    return valid


def filter_by_liquidity(
    ticker_data: Dict,
    oi_value: Optional[float] = None,
) -> bool:
    """
    流动性三重过滤（实时监控要求）：
      1. 24小时成交额 >= MIN_QUOTE_VOLUME_24H
      2. bid-ask spread <= MAX_SPREAD_RATE
      3. OI（持仓量）>= MIN_OI_VALUE
    
    参数：
        ticker_data: Binance /fapi/v1/ticker/24hr 返回的单合约数据
        oi_value   : OI名义价值（USDT），从 OI 数据中获取
    
    返回：
        True = 通过流动性过滤；False = 流动性不足，跳过
    """
    symbol = ticker_data.get("symbol", "?")

    # ---- 过滤1：成交额 ----
    quote_volume = float(ticker_data.get("quoteVolume", 0))
    if quote_volume < config.MIN_QUOTE_VOLUME_24H:
        logger.debug(
            f"[filter] {symbol} 成交额不足: "
            f"{quote_volume/1e6:.1f}M < {config.MIN_QUOTE_VOLUME_24H/1e6:.0f}M USDT"
        )
        return False

    # ---- 过滤2：bid-ask spread ----
    bid_price = float(ticker_data.get("bidPrice", 0))
    ask_price = float(ticker_data.get("askPrice", 0))
    mid_price = (bid_price + ask_price) / 2
    if mid_price > 0:
        spread_rate = (ask_price - bid_price) / mid_price
        if spread_rate > config.MAX_SPREAD_RATE:
            logger.debug(
                f"[filter] {symbol} spread 过大: "
                f"{spread_rate*100:.3f}% > {config.MAX_SPREAD_RATE*100:.2f}%"
            )
            return False

    # ---- 过滤3：OI 持仓量 ----
    if oi_value is not None and oi_value < config.MIN_OI_VALUE:
        logger.debug(
            f"[filter] {symbol} OI不足: "
            f"{oi_value/1e6:.1f}M < {config.MIN_OI_VALUE/1e6:.0f}M USDT"
        )
        return False

    return True


def select_top_n_by_volume(
    tickers: List[Dict],
    top_n: int = None,
) -> List[str]:
    """
    Research模式：按24h成交额选取前N个USDT永续合约（要求1）。
    
    参数：
        tickers: /fapi/v1/ticker/24hr 返回的所有合约数据
        top_n  : 选取前N个（None=使用config.RESEARCH_TOP_N）
    
    返回：
        合约代码列表（按成交额降序排列）
    """
    if top_n is None:
        top_n = config.RESEARCH_TOP_N

    # 只保留 USDT 计价合约
    usdt_tickers = [
        t for t in tickers
        if t.get("symbol", "").endswith("USDT")
    ]

    # 按 24h 成交额降序排序
    sorted_tickers = sorted(
        usdt_tickers,
        key=lambda x: float(x.get("quoteVolume", 0)),
        reverse=True
    )

    top_symbols = [t["symbol"] for t in sorted_tickers[:top_n]]
    logger.info(f"[filter] Research模式：按成交额选取前{top_n}个合约，前3: {top_symbols[:3]}")

    return top_symbols

def select_top_gainers(
    tickers: List[Dict],
    top_n: int = None,
) -> List[str]:
    """
    按24h涨跌幅选取涨幅榜前N个USDT永续合约。
    
    参数：
        tickers: /fapi/v1/ticker/24hr 返回的所有合约数据
        top_n  : 选取前N个（None=使用config.RESEARCH_TOP_N）
    
    返回：
        合约代码列表（按涨幅降序排列）
    """
    if top_n is None:
        top_n = config.RESEARCH_TOP_N

    # 只保留 USDT 计价合约
    usdt_tickers = [
        t for t in tickers
        if t.get("symbol", "").endswith("USDT")
    ]

    # 按 24h 涨幅降序排序 (priceChangePercent)
    sorted_tickers = sorted(
        usdt_tickers,
        key=lambda x: float(x.get("priceChangePercent", 0)),
        reverse=True
    )

    top_symbols = [t["symbol"] for t in sorted_tickers[:top_n]]
    logger.info(f"[filter] Research模式：按涨幅榜选取前{top_n}个合约，前3: {top_symbols[:3]}")

    return top_symbols
