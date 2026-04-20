# ============================================================
# src/utils/precision.py — 动态精度读取（要求42）
# 从 /fapi/v1/exchangeInfo 读取每个合约的精度信息
# 防止实盘下单时出现精度错误（如小数位数不符合交易所要求）
# ============================================================

from __future__ import annotations
import requests
from typing import Dict, Optional
from dataclasses import dataclass, field
from loguru import logger

import config
from src.utils.retry import with_retry


@dataclass
class SymbolPrecision:
    """单个合约的精度信息"""
    symbol: str
    price_precision: int        # 价格精度（小数位数）
    quantity_precision: int     # 数量精度（小数位数）
    tick_size: float            # 价格最小变动单位（price filter）
    lot_size: float             # 数量最小变动单位（lot size filter）
    min_notional: float         # 最小名义价值（USDT）
    contract_size: float = 1.0  # 合约面值（默认1）


# 全局缓存，避免重复请求
_precision_cache: Dict[str, SymbolPrecision] = {}


@with_retry(max_retries=5, base_delay=1.0, log_prefix="exchangeInfo")
def _fetch_exchange_info() -> dict:
    """
    从 Binance USDⓈ-M Futures 获取交易对信息。
    包含所有合约的精度、过滤器等元数据。
    """
    url = f"{config.BINANCE_FUTURES_BASE_URL}/fapi/v1/exchangeInfo"
    resp = requests.get(url, timeout=config.API_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def load_all_precisions(force_refresh: bool = False) -> Dict[str, SymbolPrecision]:
    """
    加载所有 USDT 永续合约的精度信息。
    
    参数：
        force_refresh: True = 强制重新从 API 获取（忽略缓存）
    
    返回：
        Dict[symbol, SymbolPrecision]
    """
    global _precision_cache

    # 已缓存且不强制刷新时，直接返回
    if _precision_cache and not force_refresh:
        return _precision_cache

    logger.info("从 exchangeInfo 动态读取合约精度信息...")
    data = _fetch_exchange_info()

    result: Dict[str, SymbolPrecision] = {}

    for sym_info in data.get("symbols", []):
        # 只处理 USDT 计价的永续合约
        if (
            sym_info.get("quoteAsset") != "USDT"
            or sym_info.get("contractType") != "PERPETUAL"
            or sym_info.get("status") != "TRADING"
        ):
            continue

        symbol = sym_info["symbol"]
        price_precision = int(sym_info.get("pricePrecision", 2))
        qty_precision = int(sym_info.get("quantityPrecision", 2))

        # 从 filters 中提取 tick_size 和 lot_size
        tick_size = 0.01
        lot_size = 0.001
        min_notional = 5.0

        for f in sym_info.get("filters", []):
            if f["filterType"] == "PRICE_FILTER":
                tick_size = float(f.get("tickSize", 0.01))
            elif f["filterType"] == "LOT_SIZE":
                lot_size = float(f.get("stepSize", 0.001))
            elif f["filterType"] == "MIN_NOTIONAL":
                min_notional = float(f.get("notional", 5.0))

        result[symbol] = SymbolPrecision(
            symbol=symbol,
            price_precision=price_precision,
            quantity_precision=qty_precision,
            tick_size=tick_size,
            lot_size=lot_size,
            min_notional=min_notional,
        )

    _precision_cache = result
    logger.info(f"成功加载 {len(result)} 个 USDT 永续合约精度信息")
    return result


def get_precision(symbol: str) -> Optional[SymbolPrecision]:
    """获取单个合约的精度信息（使用缓存）"""
    if not _precision_cache:
        load_all_precisions()
    return _precision_cache.get(symbol)


def round_price(symbol: str, price: float) -> float:
    """将价格按合约精度进行四舍五入"""
    prec = get_precision(symbol)
    if prec is None:
        return round(price, 2)
    return round(round(price / prec.tick_size) * prec.tick_size, prec.price_precision)


def round_quantity(symbol: str, qty: float) -> float:
    """将数量按合约精度进行四舍五入"""
    prec = get_precision(symbol)
    if prec is None:
        return round(qty, 3)
    return round(round(qty / prec.lot_size) * prec.lot_size, prec.quantity_precision)
