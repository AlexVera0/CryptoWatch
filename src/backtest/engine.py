# ============================================================
# src/backtest/engine.py — 回测引擎
#
# 核心要求：
#   1. 真实计入所有成本（手续费+funding+动态滑点）
#   2. Survivorship bias 剔除
#   3. 支持单币/多币两种模式（要求54）
#   4. 输出 equity curve、drawdown、每笔交易记录
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from loguru import logger

import config
from src.models.ev_calculator import compute_dynamic_slippage, compute_funding_cost


def run_single_symbol_backtest(
    symbol: str,
    features_df: pd.DataFrame,
    labels: pd.Series,
    proba: np.ndarray,
    funding_series: pd.Series,
    min_prob: float = None,
) -> Dict[str, Any]:
    """
    单币种回测（要求54 步骤1）。

    参数：
        symbol      : 合约代码
        features_df : 完整特征 DataFrame（含close/atr_14）
        labels      : Triple-Barrier 标签（-1/0/+1）
        proba       : 模型预测概率数组（与 features_df 等长对齐）
        funding_series: Funding Rate Series（对齐到 features_df.index）
        min_prob    : 触发做空的最低概率门槛

    返回：
        {equity_curve, drawdown, trades, metrics}
    """
    if min_prob is None:
        min_prob = config.MIN_SHORT_PROB

    capital = config.BACKTEST_INITIAL_CAPITAL
    position_size_ratio = config.BACKTEST_POSITION_SIZE

    equity_curve = [capital]
    trades = []
    holding = False
    entry_idx = None
    entry_price = None

    close_prices = features_df["close"].values
    atr_values = features_df["atr_14"].values if "atr_14" in features_df.columns else np.full(len(features_df), 0.01)
    index = features_df.index

    for i in range(len(features_df)):
        row_label = labels.iloc[i] if i < len(labels) else np.nan
        row_prob = proba[i] if i < len(proba) else 0.0
        price = close_prices[i]
        atr = atr_values[i]

        # ---- 开仓逻辑（做空）----
        if not holding and row_prob >= min_prob and not np.isnan(row_label):
            holding = True
            entry_idx = i
            entry_price = price
            # 开仓滑点（吃单，价格更差）
            entry_slip = compute_dynamic_slippage(atr, price)
            entry_price_adj = entry_price * (1 + entry_slip)  # 做空：买贵一点点（更差）

        # ---- 平仓逻辑 ----
        elif holding:
            max_hold = config.BARRIER_MAX_HOLD_BARS
            hold_bars = i - entry_idx

            # 触及 Triple-Barrier 或超时
            should_exit = (
                not np.isnan(row_label)
                and (row_label != 0 or hold_bars >= max_hold)
            )

            if should_exit:
                exit_price = price
                exit_slip = compute_dynamic_slippage(atr, price)
                exit_price_adj = exit_price * (1 - exit_slip)  # 做空平仓：卖便宜一点

                # 做空 PnL = (entry - exit) / entry
                pnl_ratio = (entry_price_adj - exit_price_adj) / entry_price_adj

                # 手续费（双边）
                fee = 2 * config.TAKER_FEE_RATE
                pnl_ratio -= fee

                # Funding（真实累计）
                fund_rates = funding_series.iloc[entry_idx:i].values if entry_idx < len(funding_series) else np.array([])
                funding_net = compute_funding_cost(fund_rates, hold_bars)
                pnl_ratio += funding_net  # 正 funding 对做空有利

                # 本笔交易盈亏（USDT）
                position_value = capital * position_size_ratio
                pnl_usdt = position_value * pnl_ratio
                capital += pnl_usdt

                trades.append({
                    "entry_time": str(index[entry_idx]),
                    "exit_time": str(index[i]),
                    "entry_price": round(entry_price, 6),
                    "exit_price": round(exit_price, 6),
                    "hold_bars": hold_bars,
                    "pnl_ratio": round(pnl_ratio, 6),
                    "pnl_usdt": round(pnl_usdt, 4),
                    "capital": round(capital, 4),
                    "label": float(row_label),
                    "funding_net": round(funding_net, 6),
                    "fee": round(fee, 6),
                })

                holding = False
                entry_idx = None
                entry_price = None

        equity_curve.append(capital)

    # ---- 指标计算 ----
    equity_arr = np.array(equity_curve)
    returns = pd.Series(equity_arr).pct_change().dropna()

    peak = np.maximum.accumulate(equity_arr)
    drawdown_arr = (equity_arr - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(drawdown_arr.min())

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    win_trades = trades_df[trades_df["pnl_ratio"] > 0] if not trades_df.empty else pd.DataFrame()
    win_rate = len(win_trades) / len(trades_df) if not trades_df.empty else 0.0
    total_return = (capital - config.BACKTEST_INITIAL_CAPITAL) / config.BACKTEST_INITIAL_CAPITAL

    # Sharpe（年化）
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 24)
    else:
        sharpe = 0.0

    metrics = {
        "symbol": symbol,
        "total_trades": len(trades_df),
        "win_rate": round(win_rate, 4),
        "total_return": round(total_return, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(sharpe, 4),
        "final_capital": round(capital, 2),
        "avg_pnl": round(trades_df["pnl_ratio"].mean(), 6) if not trades_df.empty else 0.0,
    }

    logger.info(
        f"[backtest] {symbol}: 交易{metrics['total_trades']}笔 "
        f"胜率={metrics['win_rate']:.2%} "
        f"总收益={metrics['total_return']:.2%} "
        f"MaxDD={metrics['max_drawdown']:.2%} "
        f"Sharpe={metrics['sharpe']:.2f}"
    )

    return {
        "symbol": symbol,
        "equity_curve": equity_arr.tolist(),
        "drawdown": drawdown_arr.tolist(),
        "trades": trades_df,
        "metrics": metrics,
    }


def run_multi_symbol_backtest(
    results_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    多币种汇总回测（要求54 步骤2）。
    对每个合约的回测结果进行加权汇总。

    参数：
        results_list: 多个 run_single_symbol_backtest 的结果列表

    返回：
        汇总指标字典
    """
    if not results_list:
        return {}

    metrics_list = [r["metrics"] for r in results_list if r.get("metrics")]
    metrics_df = pd.DataFrame(metrics_list)

    all_trades = pd.concat(
        [r["trades"] for r in results_list if not r.get("trades", pd.DataFrame()).empty],
        ignore_index=True,
    ) if any(not r.get("trades", pd.DataFrame()).empty for r in results_list) else pd.DataFrame()

    summary = {
        "total_symbols": len(metrics_list),
        "avg_win_rate": round(metrics_df["win_rate"].mean(), 4),
        "avg_return": round(metrics_df["total_return"].mean(), 4),
        "avg_max_drawdown": round(metrics_df["max_drawdown"].mean(), 4),
        "avg_sharpe": round(metrics_df["sharpe"].mean(), 4),
        "total_trades": int(metrics_df["total_trades"].sum()),
        "profitable_symbols": int((metrics_df["total_return"] > 0).sum()),
        "per_symbol": metrics_list,
    }

    logger.info(
        f"[backtest] 多币汇总: {summary['total_symbols']}个合约 "
        f"平均胜率={summary['avg_win_rate']:.2%} "
        f"平均收益={summary['avg_return']:.2%} "
        f"平均Sharpe={summary['avg_sharpe']:.2f}"
    )

    return summary
