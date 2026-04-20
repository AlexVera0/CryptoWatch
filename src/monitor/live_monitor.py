# ============================================================
# src/monitor/live_monitor.py — 实时监控主循环
#
# on_close 模式：K线收盘后延迟计算，确认K线最终数据
# Research/Full 双模式切换
# Dry-Run：只打印，不发邮件
# ============================================================

from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from loguru import logger

import config
from src.data.downloader import (
    get_all_usdt_perpetual_symbols, get_24h_tickers,
    download_symbol_data
)
from src.data.filters import filter_by_liquidity, select_top_n_by_volume
from src.features.engineering import compute_features, align_feature_columns
from src.features.drift import check_symbol_drift
from src.models.predictor import predict_short_probability, check_indicator_signals
from src.models.ev_calculator import quick_ev_estimate
from src.monitor.signal_manager import SignalManager
from src.monitor.market_guard import MarketGuard
from src.utils.audit_log import save_signal_snapshot
from src.utils.log_cleaner import run_all_cleanups


class LiveMonitor:
    """
    实时监控器，支持 research/full 双模式。

    流程：
        1. 等待下一根K线收盘（on_close 模式）
        2. 检查极端行情保护（BTC 1h 波动）
        3. 获取监控合约列表（research=前N，full=全部）
        4. 遍历每个合约：
           a. 流动性三重过滤
           b. 下载最新K线
           c. 计算特征（shift(1) 防未来函数）
           d. Concept Drift 检测（PSI）
           e. 模型预测概率
           f. EV 计算
           g. 冷却机制判断
           h. 触发信号 → 审计日志 → 发邮件
    """

    def __init__(
        self,
        mode: str = None,
        top_n: int = None,
        dry_run: bool = None,
        email_sender=None,
    ):
        self.mode = mode or config.RUN_MODE
        self.top_n = top_n or config.RESEARCH_TOP_N
        self.dry_run = dry_run if dry_run is not None else config.DRY_RUN

        self.signal_manager = SignalManager()
        self.market_guard = MarketGuard()
        self.email_sender = email_sender

        # 训练特征缓存（用于 Drift 检测）
        self._train_features_cache: Dict[str, pd.DataFrame] = {}

        logger.info(
            f"[monitor] 初始化完成 | 模式={self.mode} | "
            f"top_n={self.top_n} | dry_run={self.dry_run}"
        )

    def _get_symbols(self) -> List[str]:
        """根据模式获取监控合约列表"""
        tickers = get_24h_tickers()

        if self.mode == "research":
            from src.data.filters import select_top_gainers
            symbols = select_top_gainers(tickers, self.top_n)
            logger.info(f"[monitor] Research模式：监控涨幅榜前{self.top_n}个合约")
        else:
            symbols = get_all_usdt_perpetual_symbols()
            logger.info(f"[monitor] Full模式：监控全部 {len(symbols)} 个合约")

        return symbols

    def _wait_for_kline_close(self):
        """
        等待到下一根K线收盘后的延迟时刻（on_close 模式）。
        例如 30m K线在整点和半点收盘，等到此时刻 + ON_CLOSE_DELAY_SECONDS。
        """
        interval_str = config.KLINE_INTERVAL
        if interval_str.endswith("m"):
            interval_sec = int(interval_str[:-1]) * 60
        elif interval_str.endswith("h"):
            interval_sec = int(interval_str[:-1]) * 3600
        else:
            interval_sec = 3600  # 默认1h

        now = datetime.now(timezone.utc).timestamp()
        
        # 计算下一个周期的收盘时间戳（基于 unix epoch 开始，所以对齐是正确的）
        next_close = ((now // interval_sec) + 1) * interval_sec
        
        # 还要加上延迟
        target_time = next_close + config.ON_CLOSE_DELAY_SECONDS
        wait_seconds = target_time - now

        logger.info(
            f"[monitor] 等待 {config.KLINE_INTERVAL} K线收盘，"
            f"{wait_seconds:.0f}s 后开始计算"
            f"（{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC）"
        )
        time.sleep(wait_seconds)

    def _process_symbol(
        self,
        symbol: str,
        ticker_map: Dict[str, dict],
    ) -> Optional[Dict[str, Any]]:
        """处理单个合约的监控逻辑，返回信号结果或 None"""

        # ---- 1. 流动性三重过滤 ----
        ticker = ticker_map.get(symbol, {})
        if not filter_by_liquidity(ticker):
            return None

        # ---- 2. 下载最新数据（K线+Funding+OI，优先使用缓存）----
        data = download_symbol_data(symbol, force_refresh=False)
        if data is None:
            return None

        klines = data["klines"]
        funding = data["funding"]
        oi = data["oi"]

        if len(klines) < 50:
            return None

        # ---- 3. 计算特征（含 shift(1) 防未来函数）----
        try:
            features_df = compute_features(klines, funding, oi)
        except Exception as e:
            logger.warning(f"[monitor] {symbol} 特征计算失败: {e}")
            return None

        if features_df.empty:
            return None

        # 取最新一行（最近收盘K线的特征）
        latest_row = features_df.iloc[-1]

        # ---- 4. Concept Drift 检测 ----
        train_features = self._train_features_cache.get(symbol)
        drift_result = {"action": "continue", "overall_status": "stable"}
        if train_features is not None:
            drift_result = check_symbol_drift(symbol, train_features, features_df)
            if drift_result["action"] == "pause":
                logger.warning(f"[monitor] {symbol} Drift严重，跳过信号")
                return None

        # ---- 5. 模型预测（即时克隆模式） ----
        pred = predict_short_probability(symbol, latest_row)
        if pred is None:
            logger.info(f"[monitor] ⚠️ {symbol} 属于异动上榜新币，未找到模型，启动即时克隆训练...")
            from src.features.triple_barrier import compute_triple_barrier_labels, create_binary_short_label, compute_sample_weights
            from src.models.trainer import train_model
            try:
                labels = compute_triple_barrier_labels(features_df)
                binary_labels = create_binary_short_label(labels)
                sample_weights = compute_sample_weights(features_df)
                train_result = train_model(symbol, features_df, binary_labels, sample_weights)
                if train_result is not None:
                    # 重新预测
                    pred = predict_short_probability(symbol, latest_row)
                    logger.info(f"[monitor] ✅ {symbol} 即时克隆完成并预测成功")
            except Exception as e:
                logger.error(f"[monitor] {symbol} 即时克隆训练失败: {e}")

        if pred is None:
            logger.debug(f"[monitor] {symbol} 模型无法建立，跳过")
            return None

        probability, model_version = pred

        # ---- 6. EV 计算 ----
        atr = float(latest_row.get("atr_14", 0.01))
        price = float(klines["close"].iloc[-1])
        funding_arr = funding.values[-config.BARRIER_MAX_HOLD_BARS:]
        funding_rate = float(latest_row.get("funding_rate", 0.0))

        ev_result = quick_ev_estimate(
            probability=probability,
            atr=atr,
            price=price,
            funding_rate=funding_rate,
            hold_bars=config.BARRIER_MAX_HOLD_BARS,
        )

        # ---- 7. 指标检查 ----
        indicator_checks = check_indicator_signals(latest_row)
        pass_count = sum(indicator_checks.values())

        # ---- 8. 信号条件判断 ----
        condition_met = (
            probability >= config.MIN_SHORT_PROB
            and ev_result["is_positive"]
            and pass_count >= 3  # 至少3个指标通过
        )

        # ---- 9. 暴涨预警检测 (新增) ----
        from src.monitor.pump_alert import check_sudden_pump
        pump_alert = check_sudden_pump(
            symbol=symbol,
            klines=klines,
            probability=probability,
            win_rate=ev_result["win_rate"],
            indicator_checks=indicator_checks,
        )

        # ---- 10. 冷却机制（要求48）----
        should_trigger = self.signal_manager.should_trigger(symbol, condition_met)

        signal = None
        if should_trigger:
            # ---- 11. 触发信号 ----
            signal = {
                "symbol": symbol,
                "probability": probability,
                "model_version": model_version,
                "ev": ev_result["ev"],
                "win_rate": ev_result["win_rate"],
                "indicator_checks": indicator_checks,
                "pass_count": pass_count,
                "drift_status": drift_result["overall_status"],
                "price": price,
                "atr": atr,
                "funding_rate": funding_rate,
                "ev_detail": ev_result,
                "feature_snapshot": latest_row.to_dict(),
            }

            # 保存审计快照（要求52）
            try:
                snap_path = save_signal_snapshot(
                    symbol=symbol,
                    signal_type="short",
                    probability=probability,
                    ev=ev_result["ev"],
                    win_rate=ev_result["win_rate"],
                    model_version=model_version,
                    feature_snapshot=latest_row.to_dict(),
                    indicators_check=indicator_checks,
                    extra={"drift_status": drift_result["overall_status"]},
                )
                signal["snapshot_path"] = snap_path
            except Exception as e:
                logger.warning(f"[monitor] {symbol} 审计快照保存失败: {e}")

            logger.info(
                f"[monitor] ✅ 信号触发: {symbol} | "
                f"概率={probability:.2%} | EV={ev_result['ev']:.4%} | "
                f"指标={pass_count}/{len(indicator_checks)}"
            )

        if signal or pump_alert:
            return {"signal": signal, "pump_alert": pump_alert}
        return None

    def run_once(self) -> List[Dict[str, Any]]:
        """
        执行一轮监控扫描（不等待K线，立即扫描）。
        返回本轮触发的信号列表。
        """
        # ---- 极端行情保护（要求51）----
        if not self.market_guard.check():
            logger.warning("[monitor] 极端行情保护触发，本轮跳过所有信号")
            return []

        symbols = self._get_symbols()
        tickers = get_24h_tickers()
        ticker_map = {t["symbol"]: t for t in tickers}

        signals = []
        pump_alerts = []
        for symbol in symbols:
            try:
                result = self._process_symbol(symbol, ticker_map)
                if result is not None:
                    if result.get("signal"):
                        signals.append(result["signal"])
                    if result.get("pump_alert"):
                        pump_alerts.append(result["pump_alert"])
            except Exception as e:
                logger.error(f"[monitor] {symbol} 处理异常: {e}")

        # ---- 发送做空信号邮件 ----
        for sig in signals:
            if self.dry_run:
                logger.info(f"[DRY-RUN] 信号 {sig['symbol']} 不发送邮件")
            elif self.email_sender:
                try:
                    self.email_sender.send_short_signal(sig)
                except Exception as e:
                    logger.error(f"[monitor] {sig['symbol']} 发邮件失败: {e}")

        # ---- 发送暴涨预警邮件 ----
        if pump_alerts:
            if self.dry_run:
                logger.info(f"[DRY-RUN] 有 {len(pump_alerts)} 个暴涨预警，不发送邮件")
            elif self.email_sender:
                try:
                    from src.monitor.pump_alert import build_pump_email_body
                    subject, body = build_pump_email_body(pump_alerts)
                    self.email_sender.send_pump_alert(subject, body)
                except Exception as e:
                    logger.error(f"[monitor] 发送暴涨预警失败: {e}")

        logger.info(
            f"[monitor] 本轮扫描完成: {len(symbols)}个合约 | "
            f"做空信号: {len(signals)} 个 | 暴涨预警: {len(pump_alerts)} 个"
        )
        return signals

    def run_forever(self):
        """
        持续运行监控（on_close 模式）。
        每根K线收盘后扫描一次，并自动清理旧日志。
        """
        logger.info(f"[monitor] 启动持续监控 | 模式={self.mode} | dry_run={self.dry_run}")

        # ---- 启动成功后发邮件通知（要求：完好运行后通知）----
        if self.email_sender and not self.dry_run:
            try:
                symbols_preview = self._get_symbols()
                self.email_sender.send_startup_notification(
                    mode=self.mode,
                    symbols_count=len(symbols_preview),
                    top_n=self.top_n,
                )
            except Exception as e:
                logger.warning(f"[monitor] 启动通知发送失败: {e}")
        elif self.dry_run:
            logger.info("[monitor] Dry-Run模式：跳过启动通知邮件")

        # 每24轮（约24小时）执行一次日志清理
        _scan_count = 0

        while True:
            try:
                self._wait_for_kline_close()
                self.run_once()

                # 每24轮清理一次日志（约每天一次）
                _scan_count += 1
                if _scan_count % 24 == 0:
                    run_all_cleanups()
                    logger.info("[monitor] 定期日志清理完成")

            except KeyboardInterrupt:
                logger.info("[monitor] 收到中断信号，停止监控")
                break
            except Exception as e:
                logger.error(f"[monitor] 主循环异常: {e}，60s 后重试")
                time.sleep(60)
