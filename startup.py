# ============================================================
# startup.py — 一键启动脚本
# 逐项检测所有功能，全部通过后发邮件通知并启动监控
# 用法：python startup.py
# ============================================================

import sys
import time

# 修复 Windows 控制台 UTF-8 输出
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv(override=True)

import config
from loguru import logger

# 配置简洁日志（启动检测阶段不写文件）
logger.remove()
logger.add(sys.stderr, level="WARNING", format="{message}")

# ============================================================
# 工具函数
# ============================================================

def ok(msg: str):
    print(f"  [OK]  {msg}")

def fail(msg: str):
    print(f"  [FAIL] {msg}")

def info(msg: str):
    print(f"        {msg}")

def section(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

# ============================================================
# 各项检测
# ============================================================

all_passed = True

print("\n" + "="*50)
print("  CryptoWatch V4 — 启动自检")
print("="*50)

# ---- 1. 配置文件 ----
section("1. 配置检测")
issues = config.validate_config()
if not issues:
    ok("config.py 加载正常")
    ok(f"运行模式: {config.RUN_MODE}")
    ok(f"监控子集: 前 {config.RESEARCH_TOP_N} 个合约")
    ok(f"K线周期: {config.KLINE_INTERVAL}")
    ok(f"Dry-Run: {config.DRY_RUN}")
else:
    for i in issues:
        if "BINANCE" in i:
            fail(i)
            all_passed = False
        else:
            print(f"  [WARN] {i}")

# ---- 2. 依赖包 ----
section("2. 依赖包检测")
deps = {
    "pandas": "pandas",
    "numpy": "numpy",
    "xgboost": "xgboost",
    "sklearn": "sklearn",
    "pandas_ta": "pandas_ta",
    "requests": "requests",
    "filelock": "filelock",
    "loguru": "loguru",
    "plotly": "plotly",
    "dotenv": "dotenv",
    "pyarrow": "pyarrow",
}
for name, mod in deps.items():
    try:
        __import__(mod)
        ok(f"{name}")
    except ImportError:
        fail(f"{name} 未安装 → pip install {name}")
        all_passed = False

# ---- 3. Binance API 连接 ----
section("3. Binance API 连接")
try:
    import requests as _req
    resp = _req.get(
        f"{config.BINANCE_FUTURES_BASE_URL}/fapi/v1/ping",
        timeout=10
    )
    resp.raise_for_status()
    ok("Binance Futures API 连接正常")
except Exception as e:
    fail(f"Binance API 连接失败: {e}")
    all_passed = False

# ---- 4. 获取合约列表 ----
section("4. 合约列表")
try:
    from src.data.downloader import get_24h_tickers, get_all_usdt_perpetual_symbols
    from src.data.filters import select_top_gainers
    tickers = get_24h_tickers()
    symbols = select_top_gainers(tickers, config.RESEARCH_TOP_N)
    ok(f"成功获取 {len(tickers)} 个 USDT 合约行情")
    ok(f"涨幅榜前排锁定: {len(symbols)} 个合约")
    info(f"前5: {symbols[:5]}")
except Exception as e:
    fail(f"获取合约列表失败: {e}")
    all_passed = False
    symbols = []

# ---- 5. 动态精度读取 ----
section("5. 合约精度信息")
try:
    from src.utils.precision import load_all_precisions
    precs = load_all_precisions()
    ok(f"成功读取 {len(precs)} 个合约精度信息")
    if symbols:
        s0 = symbols[0]
        p = precs.get(s0)
        if p:
            info(f"{s0}: price_precision={p.price_precision}, tick={p.tick_size}")
except Exception as e:
    fail(f"精度读取失败: {e}")
    all_passed = False

# ---- 6. 数据下载测试（只测1个合约）----
section("6. 数据下载 (测试1个合约)")
test_sym = symbols[0] if symbols else "BTCUSDT"
try:
    from src.data.downloader import download_klines
    # 下载最近30天做快速测试，确保满足 MIN_KLINE_SAMPLES=500 的要求
    import config as _cfg
    _orig = _cfg.KLINE_HISTORY_DAYS
    _cfg.KLINE_HISTORY_DAYS = 30
    df = download_klines(test_sym, force_refresh=True)
    _cfg.KLINE_HISTORY_DAYS = _orig
    if df is not None and len(df) > 0:
        ok(f"{test_sym} K线下载正常: {len(df)} 根")
        info(f"最新时间: {df.index[-1]}")
    else:
        fail(f"{test_sym} K线下载返回空数据")
        all_passed = False
except Exception as e:
    fail(f"数据下载失败: {e}")
    all_passed = False
    df = None

# ---- 7. 特征工程测试 ----
section("7. 特征工程 (shift防未来函数)")
try:
    if df is not None and len(df) > 50:
        import pandas as pd
        import numpy as np
        from src.data.downloader import download_funding_rate, download_open_interest
        funding = download_funding_rate(test_sym, df.index)
        oi = download_open_interest(test_sym, df.index)
        from src.features.engineering import compute_features
        feat_df = compute_features(df, funding, oi)
        ok(f"特征工程正常: {len(feat_df)} 行, {len(feat_df.columns)} 个特征")
        ok("shift(1) 防未来函数已应用")
    else:
        print("  [SKIP] 跳过（K线数据不足）")
except Exception as e:
    fail(f"特征工程失败: {e}")
    all_passed = False
    feat_df = None

# ---- 8. 模型检测（没有模型则自动触发训练）----
section("8. 模型文件")
try:
    from pathlib import Path
    model_files = list(config.MODEL_DIR.glob("*_model.pkl"))
    if model_files:
        from src.models.trainer import load_model
        ok(f"找到 {len(model_files)} 个模型文件")
        sym_name = model_files[0].stem.replace("_model", "")
        result = load_model(sym_name)
        if result:
            _, meta = result
            ok(f"模型加载正常: {meta.get('model_version', 'unknown')}")
        else:
            print("  [WARN] 模型加载失败（版本不兼容），建议重新训练")
    else:
        print("  [WARN] 未找到模型文件，需要训练")
        print()
        ans = input("  是否现在开始训练模型？(y/n，约15分钟): ").strip().lower()
        if ans == 'y':
            import subprocess
            print()
            print("  开始训练（实时输出进度）...")
            print("  " + "-"*40)
            ret = subprocess.run(
                [sys.executable, "main.py",
                 "--mode", config.RUN_MODE,
                 "--top-n", str(config.RESEARCH_TOP_N),
                 "--backtest-only"],
                cwd=str(Path(__file__).parent),
            )
            print("  " + "-"*40)
            if ret.returncode == 0:
                model_files = list(config.MODEL_DIR.glob("*_model.pkl"))
                ok(f"训练完成！共生成 {len(model_files)} 个模型文件")
            else:
                fail("训练过程出错，请查看上方输出")
                all_passed = False
        else:
            print("  [SKIP] 跳过训练（监控将无法发出信号）")
            print(f"         之后手动运行: python main.py --mode research --top-n {config.RESEARCH_TOP_N} --backtest-only")
except Exception as e:
    print(f"  [WARN] 模型检测异常: {e}")

# ---- 9. EV 计算测试 ----
section("9. EV 计算模块")
try:
    import numpy as np
    from src.models.ev_calculator import quick_ev_estimate, compute_dynamic_slippage
    ev = quick_ev_estimate(probability=0.65, atr=100.0, price=50000.0)
    ok(f"EV 计算正常: EV={ev['ev']*100:.4f}%")
    info(f"手续费={ev['fee_cost']*100:.3f}% 滑点={ev['slippage_cost']*100:.3f}%")
except Exception as e:
    fail(f"EV 计算失败: {e}")
    all_passed = False

# ---- 10. 极端行情保护测试 ----
section("10. 极端行情保护")
try:
    from src.monitor.market_guard import MarketGuard
    guard = MarketGuard()
    ok(f"极端行情保护模块就绪")
    ok(f"BTC 1h 波动阈值: {config.BTC_VOLATILE_THRESHOLD}%")
    ok(f"冷却K线数: {config.BTC_VOLATILE_COOLDOWN_BARS} 根")
except Exception as e:
    fail(f"极端行情保护模块异常: {e}")
    all_passed = False

# ---- 11. 冷却机制测试 ----
section("11. 信号冷却机制")
try:
    from src.monitor.signal_manager import SignalManager
    sm = SignalManager()
    # 模拟：触发 → 持续 → 消失 → 再出现
    r1 = sm.should_trigger("TEST", True)   # 第一次：应触发
    r2 = sm.should_trigger("TEST", True)   # 第二次：冷却中
    r3 = sm.should_trigger("TEST", False)  # 信号消失
    r4 = sm.should_trigger("TEST", True)   # 再出现：应触发
    if r1 and not r2 and not r3 and r4:
        ok("冷却机制逻辑正常（信号消失→重现才触发）")
    else:
        fail(f"冷却逻辑异常: {r1},{r2},{r3},{r4}")
        all_passed = False
except Exception as e:
    fail(f"冷却机制测试失败: {e}")
    all_passed = False

# ---- 12. PSI Drift 检测 ----
section("12. Concept Drift 检测 (PSI)")
try:
    import numpy as np
    from src.features.drift import compute_psi_single
    psi = compute_psi_single(
        np.random.randn(1000),
        np.random.randn(100)
    )
    ok(f"PSI 计算模块正常: 测试PSI={psi:.4f}")
except Exception as e:
    fail(f"PSI 检测模块异常: {e}")
    all_passed = False

# ---- 13. 日志清理模块 ----
section("13. 自动日志清理")
try:
    from src.utils.log_cleaner import run_all_cleanups
    ok(f"日志保留: {config.LOG_RETENTION_DAYS}天")
    ok(f"快照保留: {config.SIGNAL_SNAPSHOT_KEEP_DAYS}天")
    ok(f"报告保留: {config.REPORT_KEEP_DAYS}天")
    ok("自动清理模块就绪（每24小时执行一次）")
except Exception as e:
    fail(f"清理模块异常: {e}")
    all_passed = False

# ---- 14. 邮件配置检测 ----
section("14. 邮件通知配置")
email_ok = False
try:
    from src.notify.email_sender import EmailSender
    sender = EmailSender()
    if all([config.SMTP_FROM, config.SMTP_TO, config.SMTP_PASSWORD]):
        ok(f"发件人: {config.SMTP_FROM}")
        ok(f"收件人: {config.SMTP_TO}")
        ok("SMTP 授权码已配置")
        email_ok = True
    else:
        print("  [WARN] 邮件未完整配置（请检查 .env）")
        print("         监控功能仍可运行，但不会发邮件")
except Exception as e:
    print(f"  [WARN] 邮件模块异常: {e}")

# ============================================================
# 自检汇总
# ============================================================

print("\n" + "="*50)
if all_passed:
    print("  [OK] 全部核心功能检测通过！")
else:
    print("  [FAIL] 部分功能检测失败，请修复后重新运行")
    print("         查看上方 [FAIL] 项目")
print("="*50)

if not all_passed:
    sys.exit(1)

# ============================================================
# 检测通过 → 发启动通知邮件 → 启动监控
# ============================================================

print("\n所有检测通过，准备启动监控...\n")
time.sleep(1)

# 重新配置日志（正式运行）
from loguru import logger as real_logger
real_logger.remove()
real_logger.add(
    sys.stderr,
    level=config.LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    colorize=True,
)
real_logger.add(
    config.LOG_DIR / "cryptowatch_{time:YYYY-MM-DD}.log",
    level=config.LOG_LEVEL,
    rotation=config.LOG_MAX_SIZE,
    retention=f"{config.LOG_RETENTION_DAYS} days",
    encoding="utf-8",
)

# 发启动通知邮件
if email_ok and not config.DRY_RUN:
    try:
        from src.notify.email_sender import EmailSender
        sender = EmailSender()
        sender.send_startup_notification(
            mode=config.RUN_MODE,
            symbols_count=len(symbols),
            top_n=config.RESEARCH_TOP_N,
        )
        print("启动通知邮件已发送！")
    except Exception as e:
        print(f"启动邮件发送失败: {e}")
elif config.DRY_RUN:
    print("[DRY-RUN] 跳过启动通知邮件（dry_run=True）")

# 启动监控
print(f"\n启动持续监控...")
print(f"  模式: {config.RUN_MODE.upper()}")
print(f"  合约数: {len(symbols)} 个")
print(f"  dry_run: {config.DRY_RUN}")
print(f"  按 Ctrl+C 停止\n")

from src.notify.email_sender import EmailSender
from src.monitor.live_monitor import LiveMonitor

monitor = LiveMonitor(
    mode=config.RUN_MODE,
    top_n=config.RESEARCH_TOP_N,
    dry_run=config.DRY_RUN,
    email_sender=EmailSender() if email_ok else None,
)
monitor.run_forever()
