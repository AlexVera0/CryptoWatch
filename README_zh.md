# CryptoWatch V4 — 币安永续合约高概率做空监控系统

<p align="center">
  <a href="README.md">English</a> | 简体中文
</p>

---

> **核心防御目标**：防数据错位 / 防隐形未来函数 / 防成本低估  
> 系统核心不是"预测对"，而是"在所有偏差下仍然 EV > 0"

## ⚡ 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置密钥
```bash
# 复制模板，填入真实密钥
cp .env.example .env
```
编辑 `.env`：
```env
BINANCE_API_KEY=你的BinanceAPIKey
BINANCE_API_SECRET=你的BinanceAPISecret
SMTP_PASSWORD=QQ邮箱SMTP授权码（16位，非QQ密码）
ALERT_EMAIL_FROM=你的QQ邮箱@qq.com
ALERT_EMAIL_TO=接收信号的邮箱
```

### 3. 验证配置
```bash
python main.py --validate-config
python main.py --test-email
```

### 4. 按上线流程执行（必须按顺序，禁止跳步）
```bash
# 步骤1：单币回测（必须先做）
python main.py --mode research --symbols BTCUSDT --backtest-only

# 步骤2：多币回测（研究子集）
python main.py --mode research --backtest-only

# 步骤3：子集实盘监控 Dry-Run（观察至少1天）
python main.py --mode research --dry-run

# 步骤4：全市场实盘监控（确认前三步无误后才执行）
python main.py --mode full
```

---

## 🔧 配置优先级说明
优先级严格为：**CLI 参数 > .env 文件 > config.py 默认值**

| 参数 | CLI | .env | config.py |
|------|-----|------|-----------|
| 运行模式 | `--mode research/full` | `RUN_MODE=research` | `RUN_MODE = "research"` |
| 子集数量 | `--top-n 50` | `RESEARCH_TOP_N=50` | `RESEARCH_TOP_N = 50` |
| BTC保护阈值 | `--btc-threshold 3.0` | `BTC_VOLATILE_THRESHOLD=3.0` | `BTC_VOLATILE_THRESHOLD = 3.0` |
| Dry-Run | `--dry-run` | `DRY_RUN=true` | `DRY_RUN = True` |

---

## 📊 HTML 报告说明
回测完成后，报告保存在 `logs/reports/` 目录。
包含 5 个必要图表：资金曲线、回撤、EV 分布、概率校准曲线、PSI Drift 图。

---

## ⚖️ EV 计算公式
```
EV = 胜率 × 平均盈利
   - (1 - 胜率) × 平均亏损
   - Taker手续费（双边，默认0.05%×2）
   - 滑点（双边，基础0.03% + ATR动态部分）
   - Funding成本（按实际持仓时间累计）
```

---

## 🚨 风险提醒
1. **历史回测不代表未来**：市场结构随时可能发生根本性变化。
2. **Concept Drift**：模型需定期重训，否则预测能力衰减。
3. **黑天鹅风险极高**：加密货币市场极端行情频发。
4. **EV>0 不保证盈利**：EV 是统计期望。
5. **必须手动最终确认**：任何信号触发后，必须人工判断。
6. **本系统不构成投资建议**：操作风险由用户自行承担。

---

*CryptoWatch V4 | 满足全部开发要求 | 防数据错位/防未来函数/防成本低估*
