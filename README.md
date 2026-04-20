# CryptoWatch V4 — Binance Perpetual Futures High-Prob Shorting Monitor

<p align="center">
  English | <a href="README_zh.md">简体中文</a>
</p>

---

> **Core Defensive Goals**: Anti-data-misalignment / Anti-hidden-future-function / Anti-cost-underestimation  
> The system core is not about "predicting right", but "maintaining EV > 0 under all deviations".

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Keys
```bash
# Copy template and fill in real keys
cp .env.example .env
```
Edit `.env`:
```env
BINANCE_API_KEY=YourBinanceAPIKey
BINANCE_API_SECRET=YourBinanceAPISecret
SMTP_PASSWORD=QQ_Email_SMTP_Authorization_Code (16 chars, NOT your password)
ALERT_EMAIL_FROM=your_email@qq.com
ALERT_EMAIL_TO=recipient_email@example.com
```

### 3. Validate Configuration
```bash
python main.py --validate-config
python main.py --test-email
```

### 4. Execution Pipeline (Must follow order)
```bash
# Step 1: Single symbol backtest (Mandatory first step)
python main.py --mode research --symbols BTCUSDT --backtest-only

# Step 2: Multi-symbol backtest (Research subset)
python main.py --mode research --backtest-only

# Step 3: Dry-Run monitoring (Observe for at least 1 day)
python main.py --mode research --dry-run

# Step 4: Full market monitoring (Confirm steps 1-3 before execution)
python main.py --mode full
```

---

## 🔧 Config Priority
Strict Priority: **CLI Arguments > .env File > config.py Defaults**

| Parameter | CLI | .env | config.py |
|------|-----|------|-----------|
| Run Mode | `--mode research/full` | `RUN_MODE=research` | `RUN_MODE = "research"` |
| Top N Count | `--top-n 50` | `RESEARCH_TOP_N=50` | `RESEARCH_TOP_N = 50` |
| BTC Threshold | `--btc-threshold 3.0` | `BTC_VOLATILE_THRESHOLD=3.0` | `BTC_VOLATILE_THRESHOLD = 3.0` |
| Dry-Run | `--dry-run` | `DRY_RUN=true` | `DRY_RUN = True` |

---

## 📧 Email Setup (SMTP)
1. Login to QQ Email → **Settings** → **Account**.
2. Find "POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV Service".
3. Enable **SMTP service**.
4. Get **Authorization Code** (16-character string).
5. Fill it into `SMTP_PASSWORD` in `.env`.

---

## 🔄 Research vs Full Mode
| Mode | Description | Use Case |
|------|------|---------|
| `research` | Monitors Top N symbols by volume | Dev, Test, Initial run |
| `full` | Monitors all USDT Perpetual pairs | Production (High resource) |

---

## 📊 HTML Reports
Reports are saved in `logs/reports/` after backtesting.
Required Charts for decision making:
- **Equity Curve**: Net value changes over time.
- **Drawdown**: Max historical drawdown analysis.
- **EV Distribution**: Real PnL distribution (not mean assumption).
- **Calibration Curve**: Verifies if model probability has physical meaning.
- **PSI Drift**: Feature distribution drift measurement.

---

## ⚖️ EV Formula
```
EV = WinRate × AvgProfit
   - (1 - WinRate) × AvgLoss
   - Taker Fee (Double-sided, default 0.05%×2)
   - Slippage (Double-sided, base 0.03% + ATR dynamic)
   - Funding Cost (Cumulative based on actual holding time)
```

---

## 🧊 Cooling Mechanism
Not a simple timer! Cooling is based on signal state changes:
1. Signal triggers → Email sent.
2. Signal persists → No duplicate alerts (Cooling).
3. Signal disappears (Prob < threshold) → Cooling reset.
4. Signal reappears → Allowed to trigger again.

---

## 🚨 Extreme Market Protection
If BTC 1h volatility exceeds threshold (default 3%):
- **Global Pause** on all signals.
- Automatic recovery after N candles (default 3).
- Prevents false positives during news/liquidation/black swan events.

---

## 🔍 Dynamic Precision
Reads `price_precision`, `quantity_precision`, `tick_size`, and `lot_size` dynamically from `/fapi/v1/exchangeInfo` to prevent `LOT_SIZE` errors in live trading.

---

## 📦 Cache Versioning
Each parquet cache contains `_cache_version`. If feature logic updates:
1. Change `CACHE_VERSION` in `config.py`.
2. System automatically forces re-download.

---

## 📝 Signal Audit Logs
JSON snapshots saved to `logs/signals/` on every trigger, including probability, EV, feature snapshots, and indicator checks.

---

## 🔴 Risk Warning
1. **History != Future**: Market regimes shift.
2. **Concept Drift**: Models decay; retrain regularly.
3. **Black Swan**: High risk of total liquidation in crypto.
4. **EV > 0 != Guaranteed Profit**: EV is a statistical expectation.
5. **Manual Confirmation**: Signals are for analysis; verify before trading.
6. **No Investment Advice**: Use at your own risk.

---

*CryptoWatch V4 | Anti-data-misalignment / Anti-future-function / Anti-cost-underestimation*
