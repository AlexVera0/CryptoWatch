# ============================================================
# src/data/downloader.py — 数据下载模块
# 包含：
#   1. K线数据下载（OHLCV）
#   2. Funding Rate 数据下载
#   3. OI（持仓量）数据下载
#   4. parquet 缓存 + 文件锁（要求46）
#   5. Funding/OI 严格 resample + 只向过去 forward-fill（要求45）
#   6. 获取 USDT 永续合约列表
# ============================================================

from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import requests
from filelock import FileLock, Timeout
from loguru import logger

import config
from src.data.cache_manager import (
    get_cache_path, is_cache_valid, save_with_metadata, load_without_metadata
)
from src.data.filters import filter_new_coins
from src.utils.retry import with_retry


# ============================================================
# 内部辅助函数
# ============================================================

@with_retry(max_retries=config.API_MAX_RETRIES, base_delay=config.API_RETRY_BASE_DELAY,
            log_prefix="BinanceAPI")
def _get(url: str, params: dict = None) -> dict | list:
    """带重试的 Binance REST GET 请求"""
    resp = requests.get(url, params=params, timeout=config.API_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _ms_to_dt(ms: int) -> datetime:
    """毫秒时间戳转 UTC datetime"""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _dt_to_ms(dt: datetime) -> int:
    """datetime 转毫秒时间戳"""
    return int(dt.timestamp() * 1000)


# ============================================================
# 合约列表
# ============================================================

def get_all_usdt_perpetual_symbols() -> List[str]:
    """
    获取 Binance 所有 USDT 计价永续合约代码列表。
    从 /fapi/v1/exchangeInfo 动态读取（非硬编码）。
    只返回状态为 TRADING 的活跃合约。
    """
    url = f"{config.BINANCE_FUTURES_BASE_URL}/fapi/v1/exchangeInfo"
    data = _get(url)

    symbols = [
        s["symbol"]
        for s in data.get("symbols", [])
        if (
            s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"
        )
    ]

    logger.info(f"[downloader] 获取到 {len(symbols)} 个 USDT 永续合约")
    return symbols


def get_24h_tickers() -> List[Dict]:
    """
    获取所有 USDT 永续合约的 24h 行情（用于流动性过滤和Research模式选币）。
    返回 /fapi/v1/ticker/24hr 的完整列表。
    """
    url = f"{config.BINANCE_FUTURES_BASE_URL}/fapi/v1/ticker/24hr"
    tickers = _get(url)

    # 只保留 USDT 合约
    usdt_tickers = [t for t in tickers if t.get("symbol", "").endswith("USDT")]
    return usdt_tickers


# ============================================================
# K线数据下载
# ============================================================

def _download_klines_raw(
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """
    下载单个合约的原始K线数据。
    Binance 每次最多返回 1500 根K线，自动分页。
    
    【防数据错位】：
    - 所有时间戳使用 UTC
    - 确保 open_time 严格单调递增
    - 去重处理
    """
    url = f"{config.BINANCE_FUTURES_BASE_URL}/fapi/v1/klines"
    all_klines = []

    current_start = _dt_to_ms(start_dt)
    end_ms = _dt_to_ms(end_dt)

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1500,
        }

        data = _get(url, params)

        if not data:
            break

        all_klines.extend(data)

        # 下一页从最后一根K线的 close_time + 1ms 开始
        last_open_time = data[-1][0]
        current_start = last_open_time + 1

        # 防止无限循环
        if len(data) < 1500:
            break

        # 避免超频
        time.sleep(0.1)

    if not all_klines:
        logger.warning(f"[downloader] {symbol} 未下载到K线数据")
        return pd.DataFrame()

    # 转换为 DataFrame
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ]
    df = pd.DataFrame(all_klines, columns=columns)

    # 类型转换
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    # 设置时间索引
    df = df.set_index("open_time")
    df = df[["open", "high", "low", "close", "volume", "quote_volume"]]

    # 去重（防止分页重叠）
    df = df[~df.index.duplicated(keep="last")]

    # 确保时间单调递增
    df = df.sort_index()

    return df


def download_klines(
    symbol: str,
    interval: str = None,
    history_days: int = None,
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    下载K线数据，带 parquet 缓存 + 文件锁。
    
    【文件锁（要求46）】：
    使用 filelock 确保多进程/多线程环境下，
    同一个文件不会被并发写入（防止 parquet 损坏）。
    
    返回：
        DataFrame（列：open/high/low/close/volume）或 None（下载失败）
    """
    if interval is None:
        interval = config.KLINE_INTERVAL
    if history_days is None:
        history_days = config.KLINE_HISTORY_DAYS

    cache_path = get_cache_path(symbol, "kline", interval)
    lock_path = cache_path.with_suffix(".lock")

    # 尝试使用缓存（不需要锁）
    if not force_refresh and is_cache_valid(cache_path, max_age_hours=1):
        try:
            df = load_without_metadata(cache_path)
            logger.debug(f"[downloader] {symbol} 使用缓存K线数据 ({len(df)} 行)")
            return df
        except Exception as e:
            logger.warning(f"[downloader] {symbol} 读取缓存失败，重新下载: {e}")

    # 下载新数据（加文件锁防止并发写入）
    try:
        with FileLock(str(lock_path), timeout=30):
            # 二次检查（锁内再验证，防止等待期间其他进程已经写入）
            if not force_refresh and is_cache_valid(cache_path, max_age_hours=1):
                return load_without_metadata(cache_path)

            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=history_days)

            logger.info(f"[downloader] 下载 {symbol} K线 {interval} ({history_days}天)...")
            df = _download_klines_raw(symbol, interval, start_dt, end_dt)

            if df.empty:
                return None

            # 新币过滤
            df = filter_new_coins(symbol, df, config.MIN_KLINE_SAMPLES)
            if df is None:
                return None

            # 保存缓存
            save_with_metadata(df, cache_path)
            logger.info(f"[downloader] {symbol} K线下载完成: {len(df)} 根")
            return df

    except Timeout:
        logger.error(f"[downloader] {symbol} 获取文件锁超时，跳过")
        return None
    except Exception as e:
        logger.error(f"[downloader] {symbol} K线下载失败: {e}")
        return None


# ============================================================
# Funding Rate 数据下载
# ============================================================

def _download_funding_raw(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """下载原始 Funding Rate 数据"""
    url = f"{config.BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate"
    all_records = []

    current_start = _dt_to_ms(start_dt)
    end_ms = _dt_to_ms(end_dt)

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        data = _get(url, params)

        if not data:
            break

        all_records.extend(data)

        if len(data) < 1000:
            break

        # 下一页
        current_start = data[-1]["fundingTime"] + 1
        time.sleep(0.1)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df.set_index("fundingTime")[["fundingRate"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    return df


def download_funding_rate(
    symbol: str,
    kline_index: pd.DatetimeIndex,
    history_days: int = None,
    force_refresh: bool = False,
) -> pd.Series:
    """
    下载 Funding Rate 并对齐到 K 线时间序列（要求45）。
    
    【严格 resample + 只向过去 forward-fill（要求45）】：
    Funding Rate 每8小时更新一次，而K线可能是1小时的。
    必须：
      1. 重采样到K线时间粒度
      2. 使用 ffill()（向前填充，即用过去的值填充）
      3. 绝对禁止 bfill()（向未来填充，会引入未来信息）
    
    返回：
        与 kline_index 对齐的 Funding Rate Series
    """
    if history_days is None:
        history_days = config.KLINE_HISTORY_DAYS

    cache_path = get_cache_path(symbol, "funding", "8h")
    lock_path = cache_path.with_suffix(".lock")

    # 尝试使用缓存
    if not force_refresh and is_cache_valid(cache_path, max_age_hours=8):
        try:
            raw_df = load_without_metadata(cache_path)
        except Exception:
            raw_df = pd.DataFrame()
    else:
        # 下载并缓存
        try:
            with FileLock(str(lock_path), timeout=30):
                if not force_refresh and is_cache_valid(cache_path, max_age_hours=8):
                    raw_df = load_without_metadata(cache_path)
                else:
                    end_dt = datetime.now(timezone.utc)
                    start_dt = end_dt - timedelta(days=history_days)
                    logger.info(f"[downloader] 下载 {symbol} Funding Rate...")
                    raw_df = _download_funding_raw(symbol, start_dt, end_dt)
                    if not raw_df.empty:
                        save_with_metadata(raw_df, cache_path)
        except Exception as e:
            logger.warning(f"[downloader] {symbol} Funding Rate 下载失败: {e}")
            raw_df = pd.DataFrame()

    if raw_df.empty:
        # 返回全零序列（防止下游 NaN 传播）
        return pd.Series(0.0, index=kline_index, name="funding_rate")

    # ---- 严格对齐到 K 线时间序列（只向过去填充）----
    # 将 funding rate 重采样到 K 线频率
    # 先 reindex 到 kline_index，然后只用 ffill（不用 bfill）
    funding_series = raw_df["fundingRate"].copy()
    funding_series = funding_series.reindex(
        funding_series.index.union(kline_index)
    )

    # ⚠️ 关键：只向过去填充（ffill），严禁 bfill
    # ffill 表示：用最近已知的 funding 值填充后续空缺
    # 这样每根K线看到的 funding 都是"当时已经公开"的值
    funding_series = funding_series.ffill()

    # 截取到 kline_index 对应的时间点
    funding_aligned = funding_series.reindex(kline_index)

    # 仍有 NaN 的（K线时间早于第一条funding数据）填为0
    funding_aligned = funding_aligned.fillna(0.0)
    funding_aligned.name = "funding_rate"

    return funding_aligned


# ============================================================
# OI（持仓量）数据下载
# ============================================================

def _download_oi_raw(symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """下载原始 OI 数据"""
    url = f"{config.BINANCE_FUTURES_BASE_URL}/futures/data/openInterestHist"
    all_records = []

    current_start = _dt_to_ms(start_dt)
    end_ms = _dt_to_ms(end_dt)

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "period": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 500,
        }
        try:
            data = _get(url, params)
        except Exception as e:
            logger.warning(f"[downloader] {symbol} OI 请求失败: {e}")
            break

        if not data:
            break

        all_records.extend(data)

        if len(data) < 500:
            break

        current_start = data[-1]["timestamp"] + 1
        time.sleep(0.1)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
    df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
    df = df.set_index("timestamp")[["sumOpenInterest", "sumOpenInterestValue"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    return df


def download_open_interest(
    symbol: str,
    kline_index: pd.DatetimeIndex,
    interval: str = None,
    history_days: int = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    下载 OI 并对齐到 K 线时间序列（要求45）。
    同样只向过去 ffill，严禁 bfill。
    
    返回：
        包含 oi_value（名义价值USDT）和 oi_contracts（合约数）的 DataFrame
    """
    if interval is None:
        interval = config.OI_INTERVAL
    if history_days is None:
        history_days = config.KLINE_HISTORY_DAYS

    # OI 历史数据最多只有30天，调整范围
    history_days = min(history_days, 30)

    cache_path = get_cache_path(symbol, "oi", interval)
    lock_path = cache_path.with_suffix(".lock")

    raw_df = pd.DataFrame()

    if not force_refresh and is_cache_valid(cache_path, max_age_hours=1):
        try:
            raw_df = load_without_metadata(cache_path)
        except Exception:
            raw_df = pd.DataFrame()

    if raw_df.empty:
        try:
            with FileLock(str(lock_path), timeout=30):
                end_dt = datetime.now(timezone.utc)
                start_dt = end_dt - timedelta(days=history_days)
                logger.info(f"[downloader] 下载 {symbol} OI...")
                raw_df = _download_oi_raw(symbol, interval, start_dt, end_dt)
                if not raw_df.empty:
                    save_with_metadata(raw_df, cache_path)
        except Exception as e:
            logger.warning(f"[downloader] {symbol} OI 下载失败: {e}")

    if raw_df.empty:
        return pd.DataFrame(
            {"oi_value": 0.0, "oi_contracts": 0.0},
            index=kline_index,
        )

    # ---- 只向过去 ffill（要求45）----
    oi_value = raw_df["sumOpenInterestValue"].reindex(
        raw_df.index.union(kline_index)
    ).ffill().reindex(kline_index).fillna(0.0)

    oi_contracts = raw_df["sumOpenInterest"].reindex(
        raw_df.index.union(kline_index)
    ).ffill().reindex(kline_index).fillna(0.0)

    result = pd.DataFrame({
        "oi_value": oi_value,
        "oi_contracts": oi_contracts,
    }, index=kline_index)

    return result


# ============================================================
# 批量下载（多合约）
# ============================================================

def download_symbol_data(
    symbol: str,
    force_refresh: bool = False,
) -> Optional[dict]:
    """
    下载单个合约的完整数据（K线 + Funding + OI）。
    
    返回：
        {
            "klines": DataFrame,
            "funding": Series,
            "oi": DataFrame,
        } 或 None（下载失败）
    """
    klines = download_klines(symbol, force_refresh=force_refresh)

    if klines is None or klines.empty:
        return None

    kline_index = klines.index

    funding = download_funding_rate(symbol, kline_index, force_refresh=force_refresh)
    oi = download_open_interest(symbol, kline_index, force_refresh=force_refresh)

    return {
        "klines": klines,
        "funding": funding,
        "oi": oi,
    }
