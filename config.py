# ============================================================
# CryptoWatch V4 — 业务配置文件
# 规则：
#   1. 所有敏感密钥 必须 放在 .env 文件中，此处只做读取
#   2. 配置优先级：CLI参数 > .env > 此文件默认值
#   3. 所有参数必须有中文注释说明
# ============================================================

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 修复 Windows 控制台下打印 Emoji 报错的问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 加载 .env 文件（.env 存在时覆盖系统环境变量）
load_dotenv(override=True)

# ============================================================
# 【路径配置】
# ============================================================
# 项目根目录
BASE_DIR = Path(__file__).parent.resolve()

# 数据缓存目录（parquet文件存放位置）
DATA_CACHE_DIR = BASE_DIR / "data" / "cache"

# 原始数据目录
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

# 模型保存目录
MODEL_DIR = BASE_DIR / "models"

# 日志目录
LOG_DIR = BASE_DIR / "logs"

# 信号审计快照目录（每次信号触发时保存完整特征快照）
SIGNAL_SNAPSHOT_DIR = BASE_DIR / "logs" / "signals"

# HTML报告输出目录
REPORT_DIR = BASE_DIR / "logs" / "reports"

# 自动创建必要目录
for _dir in [DATA_CACHE_DIR, DATA_RAW_DIR, MODEL_DIR, LOG_DIR,
             SIGNAL_SNAPSHOT_DIR, REPORT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 【Binance API 配置】（密钥从 .env 读取，禁止硬编码）
# ============================================================
BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")

# Binance Futures REST API 基础地址
BINANCE_FUTURES_BASE_URL: str = "https://fapi.binance.com"

# API 请求超时秒数
API_TIMEOUT: int = 15

# API 请求失败时的最大重试次数
API_MAX_RETRIES: int = 5

# API 重试初始等待秒数（指数退避：1s, 2s, 4s, 8s, 16s）
API_RETRY_BASE_DELAY: float = 1.0

# ============================================================
# 【运行模式配置】
# ============================================================
# 运行模式：
#   "research" = 研究模式，只处理成交额前 RESEARCH_TOP_N 的合约
#   "full"     = 全市场模式，处理所有 USDT 永续合约
# 优先级：CLI --mode 参数 > .env RUN_MODE > 此处默认值
RUN_MODE: str = os.getenv("RUN_MODE", "research")

# 选取涨幅榜前 N 的 USDT 永续合约
# 优先级：CLI --top-n > .env RESEARCH_TOP_N > 此处默认值
RESEARCH_TOP_N: int = int(os.getenv("RESEARCH_TOP_N", "5"))

# ============================================================
# 【数据下载配置】
# ============================================================
# K线时间周期（用于信号计算，改为30m平衡速度和假信号）
KLINE_INTERVAL: str = "30m"

# 下载历史K线的天数（用于训练和回测）
KLINE_HISTORY_DAYS: int = 365

# 新币上市后必须满足的最少K线数量（低于此值过滤）
MIN_KLINE_SAMPLES: int = 500

# Funding Rate 时间周期（Binance永续合约每8小时一次）
FUNDING_INTERVAL_HOURS: int = 8

# OI（持仓量）数据时间周期
OI_INTERVAL: str = "1h"

# 数据缓存版本（特征升级时修改此版本号，强制重新下载）
CACHE_VERSION: str = "v4.0"

# ============================================================
# 【特征工程配置】
# ============================================================
# RSI 计算周期
RSI_PERIOD: int = 14

# MACD 快线周期
MACD_FAST: int = 12

# MACD 慢线周期
MACD_SLOW: int = 26

# MACD 信号线周期
MACD_SIGNAL: int = 9

# ATR 计算周期（用于动态滑点和Triple-Barrier障碍设置）
ATR_PERIOD: int = 14

# 布林带周期
BBANDS_PERIOD: int = 20

# 布林带标准差倍数
BBANDS_STD: float = 2.0

# 成交量移动平均周期（用于流动性过滤）
VOLUME_MA_PERIOD: int = 20

# ============================================================
# 【Triple-Barrier 标签配置】
# ============================================================
# 上方障碍倍数（以ATR为单位，触及止盈）
BARRIER_UPPER_MULTIPLIER: float = 2.0

# 下方障碍倍数（以ATR为单位，触及止损）
BARRIER_LOWER_MULTIPLIER: float = 1.0

# 最大持有K线数（超时障碍）
BARRIER_MAX_HOLD_BARS: int = 24

# ============================================================
# 【模型训练配置】
# ============================================================
# XGBoost 超参数
XGB_PARAMS: dict = {
    "n_estimators": 300,           # 树的数量
    "max_depth": 4,                # 树的最大深度（防过拟合）
    "learning_rate": 0.05,         # 学习率
    "subsample": 0.8,              # 样本采样比例
    "colsample_bytree": 0.8,       # 特征采样比例
    "min_child_weight": 10,        # 最小叶节点样本权重（防过拟合）
    "gamma": 0.1,                  # 最小分裂增益
    "reg_alpha": 0.1,              # L1 正则化
    "reg_lambda": 1.0,             # L2 正则化
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

# Purged K-Fold CV 折数
CV_N_SPLITS: int = 5

# Purged CV 清除间隔（K线数，防止训练/验证集时间重叠）
CV_PURGE_BARS: int = 24

# Embargo 间隔（K线数，避免泄露）
CV_EMBARGO_BARS: int = 6

# 概率校准方法（"sigmoid" 或 "isotonic"）
CALIBRATION_METHOD: str = "sigmoid"

# 训练集最小样本数
MIN_TRAIN_SAMPLES: int = 200

# ============================================================
# 【EV 计算配置】（防成本低估核心）
# EV = 胜率×平均盈利 - (1-胜率)×平均亏损 - 手续费 - funding - 滑点
# ============================================================
# Binance USDⓈ-M 期货 Taker 手续费（百分比）
TAKER_FEE_RATE: float = float(os.getenv("TAKER_FEE_RATE", "0.0005"))  # 0.05%

# Maker 手续费（如果用限价单挂单）
MAKER_FEE_RATE: float = float(os.getenv("MAKER_FEE_RATE", "0.0002"))  # 0.02%

# 基础滑点（百分比，正常市况）
BASE_SLIPPAGE_RATE: float = 0.0003  # 0.03%

# ATR 倍数滑点系数（大波动时：滑点 = ATR × 此系数 / 价格）
ATR_SLIPPAGE_MULTIPLIER: float = 0.1

# EV 过滤最低门槛（低于此值的信号不发出）
MIN_EV_THRESHOLD: float = 0.0005  # 0.05%

# 最低做空信号概率门槛（模型输出概率低于此值不触发）
MIN_SHORT_PROB: float = 0.60

# ============================================================
# 【回测配置】
# ============================================================
# 回测初始资金（USDT）
BACKTEST_INITIAL_CAPITAL: float = 10000.0

# 每次开仓使用资金比例
BACKTEST_POSITION_SIZE: float = 0.1  # 10%

# 回测数据集分割：前 N% 用于训练，后 (1-N)% 用于回测验证
TRAIN_TEST_SPLIT_RATIO: float = 0.7

# ============================================================
# 【流动性过滤配置】（三重过滤：volume + spread + OI）
# ============================================================
# 24小时成交额最低要求（USDT）
MIN_QUOTE_VOLUME_24H: float = 10_000_000  # 1000万 USDT

# 最大允许 bid-ask spread 比例
MAX_SPREAD_RATE: float = 0.002  # 0.2%

# 最低 OI（持仓量，USDT计价）
MIN_OI_VALUE: float = 5_000_000  # 500万 USDT

# ============================================================
# 【实时监控配置】
# ============================================================
# K线关闭后延迟计算秒数（等待数据稳定）
ON_CLOSE_DELAY_SECONDS: int = 5

# 监控轮询间隔秒数（每个时间周期扫描一次）
MONITOR_POLL_INTERVAL_SECONDS: int = 60

# Dry-Run 模式（True = 只打印信号，不发邮件，不下单）
DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() == "true"

# ============================================================
# 【极端行情保护配置】（要求51）
# ============================================================
# BTC 1小时涨跌幅超过此阈值时，全局暂停信号发出
BTC_VOLATILE_THRESHOLD: float = float(
    os.getenv("BTC_VOLATILE_THRESHOLD", "3.0")
)  # 百分比，默认3%

# 极端行情保护触发后，冷却多少个K线才恢复
BTC_VOLATILE_COOLDOWN_BARS: int = 3

# ============================================================
# 【暴涨暴跌预警配置】
# ============================================================
# 1小时涨跌幅超过此值发送"预警"邮件（黄色）
PUMP_WARNING_THRESHOLD: float = 30.0   # 30%，注意风险

# 1小时涨跌幅超过此值发送"紧急"邮件（红色）
PUMP_CRITICAL_THRESHOLD: float = 40.0  # 40%，极端异常

# 预警监控范围：True=只监控research子集，False=监控全市场
PUMP_WATCH_ALL: bool = False  # 建议只看监控的合约，省资源


# ============================================================
# 【冷却机制配置】（要求48：信号消失→重新出现才触发）
# ============================================================
# 冷却机制说明：
#   触发信号 → 发送邮件 → 等待信号条件消失 → 条件再次满足 → 才能再次触发
#   不是简单计时，而是基于信号状态变化
# （此处无需数值配置，逻辑在 signal_manager.py 实现）

# ============================================================
# 【PSI Drift 检测配置】（Concept Drift，要求44）
# ============================================================
# PSI 警告阈值（超过此值表示分布漂移，需要重新训练）
PSI_WARNING_THRESHOLD: float = 0.1

# PSI 危险阈值（超过此值，暂停该合约信号）
PSI_CRITICAL_THRESHOLD: float = 0.25

# ============================================================
# 【邮件通知配置】（密钥从 .env 读取）
# ============================================================
# 发件人 QQ 邮箱地址
SMTP_FROM: str = os.getenv("ALERT_EMAIL_FROM", "")

# 收件人邮箱地址
SMTP_TO: str = os.getenv("ALERT_EMAIL_TO", "")

# QQ 邮箱 SMTP 授权码（不是QQ密码）
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")

# QQ 邮箱 SMTP 服务器
SMTP_HOST: str = "smtp.qq.com"

# QQ 邮箱 SMTP 端口（SSL）
SMTP_PORT: int = 465

# ============================================================
# 【日志配置】
# ============================================================
# 日志级别（DEBUG / INFO / WARNING / ERROR）
LOG_LEVEL: str = "INFO"

# 单个日志文件最大大小（字节）
LOG_MAX_SIZE: str = "20 MB"  # 从50MB改为20MB，节省磁盘

# 日志保留天数（超过自动删除）
LOG_RETENTION_DAYS: int = 1  # 1天，磁盘紧张，当天日志即可

# 信号快照保留天数
SIGNAL_SNAPSHOT_KEEP_DAYS: int = 1  # 1天快照

# HTML报告保留天数
REPORT_KEEP_DAYS: int = 3  # 报告留3天方便回顾

# ============================================================
# 【数据缓存版本控制】（要求44）
# ============================================================
# 特征列表（修改时必须同步升级 CACHE_VERSION）
# 此列表定义了模型训练和实盘预测使用的完整特征集
# 实盘特征列顺序必须和此列表完全一致（要求50）
FEATURE_COLUMNS: list = [
    # K线基础特征
    "open", "high", "low", "close", "volume",
    # 技术指标
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_pct",       # 价格在布林带中的位置
    "volume_ma_20",
    "volume_ratio",  # 当前成交量 / 20周期均量
    # 价格动量
    "return_1h",
    "return_4h",
    "return_24h",
    # Funding Rate 特征
    "funding_rate",
    "funding_rate_ma8",   # 过去8期平均funding
    "funding_cumsum_24h", # 过去24小时累计funding
    # OI 特征
    "oi_value",
    "oi_change_1h",   # OI 1小时变化率
    "oi_change_4h",   # OI 4小时变化率
    # 衍生特征
    "high_low_ratio",  # high/low 波动比
    "close_open_ratio", # close/open 涨跌
]

# ============================================================
# 【配置完整性验证】
# ============================================================
def validate_config() -> list[str]:
    """
    验证配置完整性，返回问题列表。
    空列表表示配置正常。
    """
    issues = []
    if not BINANCE_API_KEY:
        issues.append("❌ BINANCE_API_KEY 未配置（请检查 .env 文件）")
    if not SMTP_PASSWORD:
        issues.append("⚠️  SMTP_PASSWORD 未配置（邮件通知将不可用）")
    if not SMTP_FROM:
        issues.append("⚠️  ALERT_EMAIL_FROM 未配置（邮件通知将不可用）")
    if not SMTP_TO:
        issues.append("⚠️  ALERT_EMAIL_TO 未配置（邮件通知将不可用）")
    return issues


if __name__ == "__main__":
    # 直接运行此文件时，打印配置摘要
    issues = validate_config()
    print("=" * 60)
    print("CryptoWatch V4 配置摘要")
    print("=" * 60)
    print(f"运行模式    : {RUN_MODE}")
    print(f"研究子集    : 前 {RESEARCH_TOP_N} 个合约")
    print(f"K线周期     : {KLINE_INTERVAL}")
    print(f"历史天数    : {KLINE_HISTORY_DAYS} 天")
    print(f"Taker手续费 : {TAKER_FEE_RATE*100:.3f}%")
    print(f"EV门槛      : {MIN_EV_THRESHOLD*100:.3f}%")
    print(f"信号概率门槛: {MIN_SHORT_PROB*100:.0f}%")
    print(f"BTC波动保护 : {BTC_VOLATILE_THRESHOLD}%")
    print(f"Dry-Run     : {DRY_RUN}")
    print(f"缓存版本    : {CACHE_VERSION}")
    print("=" * 60)
    if issues:
        print("⚠️  配置问题：")
        for i in issues:
            print(f"   {i}")
    else:
        print("✅ 配置验证通过")
