# ============================================================
# main.py — CLI 入口
#
# 配置优先级（要求41）：CLI参数 > .env > config.py
# 支持命令：
#   python main.py --mode research --dry-run            # 研究模式监控
#   python main.py --mode full                          # 全市场监控
#   python main.py --mode research --backtest-only      # 仅回测
#   python main.py --mode research --symbols BTCUSDT    # 单币回测
#   python main.py --test-email                         # 测试邮箱配置
#   python main.py --validate-config                    # 检查配置
# ============================================================

import argparse
import sys
import os

# 修复 Windows 控制台下打印 Emoji 报错的问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from loguru import logger

# ---- 加载 .env（必须在 import config 之前）----
from dotenv import load_dotenv
load_dotenv(override=True)

import config


def setup_logging(level: str = None):
    """配置结构化日志（loguru）"""
    log_level = level or config.LOG_LEVEL
    logger.remove()
    # 控制台输出（带颜色）
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
        colorize=True,
    )
    # 文件输出（滚动）
    logger.add(
        config.LOG_DIR / "cryptowatch_{time:YYYY-MM-DD}.log",
        level=log_level,
        rotation=config.LOG_MAX_SIZE,
        retention=f"{config.LOG_RETENTION_DAYS} days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    )


def apply_cli_overrides(args: argparse.Namespace):
    """
    将 CLI 参数覆盖到 config（实现优先级：CLI > .env > config.py）。
    修改 config 模块的全局变量，后续所有模块读取 config 时生效。
    """
    if args.mode:
        config.RUN_MODE = args.mode
        logger.info(f"[CLI] RUN_MODE 覆盖为: {config.RUN_MODE}")

    if args.top_n:
        config.RESEARCH_TOP_N = args.top_n
        logger.info(f"[CLI] RESEARCH_TOP_N 覆盖为: {config.RESEARCH_TOP_N}")

    if args.dry_run is not None:
        config.DRY_RUN = args.dry_run
        logger.info(f"[CLI] DRY_RUN 覆盖为: {config.DRY_RUN}")

    if args.btc_threshold:
        config.BTC_VOLATILE_THRESHOLD = args.btc_threshold
        logger.info(f"[CLI] BTC_VOLATILE_THRESHOLD 覆盖为: {config.BTC_VOLATILE_THRESHOLD}%")


def cmd_validate_config():
    """验证配置完整性"""
    print("=" * 55)
    print("CryptoWatch V4 — 配置验证")
    print("=" * 55)
    issues = config.validate_config()
    print(f"  运行模式    : {config.RUN_MODE}")
    print(f"  研究子集    : 前 {config.RESEARCH_TOP_N} 个")
    print(f"  K线周期     : {config.KLINE_INTERVAL}")
    print(f"  Taker手续费 : {config.TAKER_FEE_RATE*100:.3f}%")
    print(f"  EV门槛      : {config.MIN_EV_THRESHOLD*100:.3f}%")
    print(f"  信号概率门槛: {config.MIN_SHORT_PROB*100:.0f}%")
    print(f"  BTC保护阈值 : {config.BTC_VOLATILE_THRESHOLD}%")
    print(f"  Dry-Run     : {config.DRY_RUN}")
    print(f"  缓存版本    : {config.CACHE_VERSION}")
    print("=" * 55)
    if issues:
        for i in issues:
            print(f"  {i}")
        sys.exit(1)
    else:
        print("  ✅ 配置验证通过")


def cmd_test_email():
    """测试邮箱配置"""
    from src.notify.email_sender import EmailSender
    sender = EmailSender()
    ok = sender.send_test_email()
    if ok:
        print("✅ 测试邮件发送成功！")
    else:
        print("❌ 测试邮件失败，请检查 .env 中的邮箱配置")
        sys.exit(1)


def cmd_backtest(args):
    """执行回测（单币或多币）"""
    import numpy as np
    from src.data.downloader import download_symbol_data, get_24h_tickers
    from src.data.filters import select_top_gainers
    from src.features.engineering import compute_features
    from src.features.triple_barrier import compute_triple_barrier_labels, create_binary_short_label, compute_sample_weights
    from src.models.trainer import train_model
    from src.backtest.engine import run_single_symbol_backtest, run_multi_symbol_backtest
    from src.report.html_report import generate_backtest_report

    # 确定回测合约列表
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        logger.info(f"[backtest] 单币回测: {symbols}")
    else:
        tickers = get_24h_tickers()
        symbols = select_top_gainers(tickers, config.RESEARCH_TOP_N)
        logger.info(f"[backtest] 多币回测: {len(symbols)} 个合约")

    results = []
    for symbol in symbols:
        logger.info(f"\n{'='*50}\n回测: {symbol}\n{'='*50}")

        data = download_symbol_data(symbol, force_refresh=args.force_refresh)
        if data is None:
            logger.warning(f"{symbol} 数据下载失败，跳过")
            continue

        klines, funding, oi = data["klines"], data["funding"], data["oi"]

        # 特征工程
        try:
            features_df = compute_features(klines, funding, oi)
        except Exception as e:
            logger.error(f"{symbol} 特征计算失败: {e}")
            continue

        # Triple-Barrier 标签
        labels = compute_triple_barrier_labels(features_df)
        binary_labels = create_binary_short_label(labels)
        sample_weights = compute_sample_weights(features_df)

        # 训练模型
        train_result = train_model(symbol, features_df, binary_labels, sample_weights)
        if train_result is None:
            continue

        # 回测
        model = train_result["model"]
        X_test = train_result["X_test"]
        y_test = train_result["y_test"]
        y_prob = train_result["y_prob_test"]

        # 在测试集上运行回测
        test_features = features_df.loc[X_test.index]
        test_labels = binary_labels.loc[X_test.index]
        test_funding = funding.reindex(X_test.index).fillna(0.0)

        bt_result = run_single_symbol_backtest(
            symbol=symbol,
            features_df=test_features,
            labels=test_labels,
            proba=y_prob,
            funding_series=test_funding,
        )
        results.append(bt_result)

        # 生成 HTML 报告
        generate_backtest_report(
            symbol=symbol,
            backtest_result=bt_result,
            y_true=y_test.values,
            y_prob=y_prob,
        )

    # 多币汇总
    if len(results) > 1:
        summary = run_multi_symbol_backtest(results)
        print("\n" + "="*55)
        print("多币回测汇总")
        print("="*55)
        for k, v in summary.items():
            if k != "per_symbol":
                print(f"  {k}: {v}")

    logger.info("[backtest] 回测完成，报告已保存到 logs/reports/")


def cmd_monitor(args):
    """启动实时监控"""
    from src.notify.email_sender import EmailSender
    from src.monitor.live_monitor import LiveMonitor

    sender = EmailSender()
    monitor = LiveMonitor(
        mode=config.RUN_MODE,
        top_n=config.RESEARCH_TOP_N,
        dry_run=config.DRY_RUN,
        email_sender=sender,
    )

    if args.run_once:
        logger.info("[main] 执行单次扫描...")
        signals = monitor.run_once()
        print(f"\n本次扫描触发信号: {len(signals)} 个")
        for s in signals:
            print(f"  ✅ {s['symbol']} | 概率={s['probability']:.2%} | EV={s['ev']:.4%}")
    else:
        logger.info("[main] 启动持续监控（on_close 模式）...")
        monitor.run_forever()


def main():
    parser = argparse.ArgumentParser(
        description="CryptoWatch V4 — 币安永续合约做空监控系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
上线检查流程（要求54，禁止跳步）：
  步骤1: python main.py --mode research --symbols BTCUSDT --backtest-only
  步骤2: python main.py --mode research --backtest-only
  步骤3: python main.py --mode research --dry-run
  步骤4: python main.py --mode full
        """
    )

    # ---- 运行模式（CLI优先级最高，覆盖 .env 和 config.py）----
    parser.add_argument("--mode", choices=["research", "full"],
                        help="运行模式: research=前N个合约, full=全市场（默认: config.py/RUN_MODE）")
    parser.add_argument("--top-n", type=int, dest="top_n",
                        help="Research模式子集数量（覆盖 config.RESEARCH_TOP_N）")
    parser.add_argument("--dry-run", action="store_true", default=None,
                        help="Dry-Run模式：只打印信号，不发邮件")
    parser.add_argument("--btc-threshold", type=float, dest="btc_threshold",
                        help="BTC 1h波动保护阈值百分比（覆盖 config.BTC_VOLATILE_THRESHOLD）")

    # ---- 功能命令 ----
    parser.add_argument("--backtest-only", action="store_true",
                        help="只执行回测，不启动监控")
    parser.add_argument("--symbols", type=str,
                        help="指定回测合约，逗号分隔，如 BTCUSDT,ETHUSDT")
    parser.add_argument("--force-refresh", action="store_true",
                        help="强制重新下载数据（忽略缓存）")
    parser.add_argument("--run-once", action="store_true",
                        help="监控模式：只扫描一次，不持续运行")
    parser.add_argument("--validate-config", action="store_true",
                        help="验证配置完整性后退出")
    parser.add_argument("--test-email", action="store_true",
                        help="发送测试邮件验证邮箱配置")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别（覆盖 config.LOG_LEVEL）")

    args = parser.parse_args()

    # ---- 初始化日志 ----
    setup_logging(args.log_level)

    # ---- CLI 参数覆盖 config（优先级：CLI > .env > config.py）----
    apply_cli_overrides(args)

    logger.info(f"CryptoWatch V4 启动 | 模式={config.RUN_MODE} | dry_run={config.DRY_RUN}")

    # ---- 执行对应命令 ----
    if args.validate_config:
        cmd_validate_config()

    elif args.test_email:
        cmd_test_email()

    elif args.backtest_only:
        cmd_backtest(args)

    else:
        # 默认：启动实时监控
        cmd_monitor(args)


if __name__ == "__main__":
    main()
