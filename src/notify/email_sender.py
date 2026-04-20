# ============================================================
# src/notify/email_sender.py — QQ邮箱 SMTP 邮件发送
#
# 固定邮件模板（要求：邮件格式固定且简洁）：
#   做空代币：XXX
#   本次做空胜率：XX%
#   通过指标：4/6
#   ✓ RSI超买
#   ✗ MACD死叉
#
# 邮件包含：模型版本号 + 时间戳（实时监控要求）
# ============================================================

from __future__ import annotations
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Any, Optional
from loguru import logger

import config


class EmailSender:
    """QQ邮箱 SMTP 邮件发送器"""

    def __init__(self):
        self.smtp_host = config.SMTP_HOST
        self.smtp_port = config.SMTP_PORT
        self.smtp_from = config.SMTP_FROM
        self.smtp_to = config.SMTP_TO
        self.smtp_password = config.SMTP_PASSWORD

        if not all([self.smtp_from, self.smtp_to, self.smtp_password]):
            logger.warning("[email] 邮箱配置不完整，邮件功能将不可用")

    def _build_short_signal_body(self, signal: Dict[str, Any]) -> str:
        """
        构建固定格式的做空信号邮件正文（纯文本）。

        格式（固定，不可随意更改）：
        ────────────────────────────
        做空代币：XXXX
        本次做空胜率：XX.X%
        通过指标：N/M
        ✓ 已通过的指标
        ✗ 未通过的指标
        ...
        [模型/成本/风险信息]
        ────────────────────────────
        """
        symbol = signal.get("symbol", "?")
        probability = signal.get("probability", 0.0)
        win_rate = signal.get("win_rate", 0.0)
        ev = signal.get("ev", 0.0)
        model_version = signal.get("model_version", "unknown")
        indicator_checks: Dict[str, bool] = signal.get("indicator_checks", {})
        pass_count = signal.get("pass_count", 0)
        total_count = len(indicator_checks)
        price = signal.get("price", 0.0)
        funding_rate = signal.get("funding_rate", 0.0)
        drift_status = signal.get("drift_status", "stable")
        ev_detail = signal.get("ev_detail", {})

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        now_cst = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")

        # 指标列表
        indicator_lines = ""
        for name, passed in indicator_checks.items():
            mark = "✓" if passed else "✗"
            indicator_lines += f"  {mark} {name}\n"

        body = f"""
════════════════════════════════════
  🔴 做空信号提醒 — CryptoWatch V4
════════════════════════════════════

做空代币：{symbol}
当前价格：{price:.6g} USDT
本次做空胜率：{win_rate*100:.1f}%
模型置信度：{probability*100:.1f}%
预期收益(EV)：{ev*100:.3f}%

通过指标：{pass_count}/{total_count}
{indicator_lines}
────────────────────────────────────
成本明细：
  手续费（双边）：{ev_detail.get('fee_cost', 0)*100:.3f}%
  滑点（双边）  ：{ev_detail.get('slippage_cost', 0)*100:.3f}%
  Funding收益  ：{ev_detail.get('funding_net', 0)*100:.4f}%
  当前Funding  ：{funding_rate*100:.4f}%/8h

────────────────────────────────────
模型版本：{model_version}
漂移状态：{drift_status.upper()}
发信时间：{now_cst}
        ({now_utc})

════════════════════════════════════
⚠️  风险提醒：
• 历史回测不代表未来收益
• EV>0 不保证每笔交易盈利
• 高波动行情请手动确认后操作
• 本系统仅为辅助决策，不构成投资建议
════════════════════════════════════
"""
        return body.strip()

    def _build_html_body(self, signal: Dict[str, Any]) -> str:
        """构建 HTML 格式邮件（可选，与纯文本同时发送）"""
        symbol = signal.get("symbol", "?")
        probability = signal.get("probability", 0.0)
        win_rate = signal.get("win_rate", 0.0)
        ev = signal.get("ev", 0.0)
        model_version = signal.get("model_version", "unknown")
        indicator_checks = signal.get("indicator_checks", {})
        pass_count = signal.get("pass_count", 0)
        total_count = len(indicator_checks)
        price = signal.get("price", 0.0)
        drift_status = signal.get("drift_status", "stable")
        now_cst = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        indicator_rows = ""
        for name, passed in indicator_checks.items():
            color = "#27ae60" if passed else "#e74c3c"
            mark = "✓" if passed else "✗"
            indicator_rows += f"""
            <tr>
                <td style="color:{color};font-weight:bold;">{mark}</td>
                <td>{name}</td>
            </tr>"""

        drift_color = {"stable": "#27ae60", "warning": "#f39c12", "critical": "#e74c3c"}.get(drift_status, "#95a5a6")

        html = f"""
<html><body style="font-family:Arial,sans-serif;background:#0d1117;color:#c9d1d9;padding:20px;">
<div style="max-width:500px;margin:0 auto;background:#161b22;border-radius:10px;padding:24px;border:1px solid #30363d;">
  <h2 style="color:#e74c3c;margin:0 0 16px;">🔴 做空信号 — {symbol}</h2>
  <table style="width:100%;border-collapse:collapse;">
    <tr><td style="color:#8b949e;">当前价格</td><td style="font-weight:bold;">{price:.6g} USDT</td></tr>
    <tr><td style="color:#8b949e;">做空胜率</td><td style="font-weight:bold;color:#f39c12;">{win_rate*100:.1f}%</td></tr>
    <tr><td style="color:#8b949e;">模型置信度</td><td>{probability*100:.1f}%</td></tr>
    <tr><td style="color:#8b949e;">预期收益EV</td><td style="color:#27ae60;">{ev*100:.3f}%</td></tr>
    <tr><td style="color:#8b949e;">漂移状态</td><td style="color:{drift_color};">{drift_status.upper()}</td></tr>
    <tr><td style="color:#8b949e;">通过指标</td><td><b>{pass_count}/{total_count}</b></td></tr>
  </table>
  <hr style="border-color:#30363d;margin:16px 0;">
  <table style="width:100%;">{indicator_rows}</table>
  <hr style="border-color:#30363d;margin:16px 0;">
  <p style="font-size:11px;color:#8b949e;">模型: {model_version}<br>时间: {now_cst}</p>
  <p style="font-size:11px;color:#e74c3c;">⚠️ 仅供参考，不构成投资建议，操作需自行判断。</p>
</div>
</body></html>"""
        return html

    def send_short_signal(self, signal: Dict[str, Any]) -> bool:
        """
        发送做空信号邮件。
        返回 True=发送成功，False=失败。
        """
        if not all([self.smtp_from, self.smtp_to, self.smtp_password]):
            logger.error("[email] 邮箱配置不完整，无法发送")
            return False

        symbol = signal.get("symbol", "?")
        subject = f"🔴 做空信号：{symbol} | 胜率 {signal.get('win_rate', 0)*100:.1f}% | 微信用户AB699Q"

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.smtp_from
        msg["To"] = self.smtp_to

        # 纯文本（降级兼容）
        text_body = self._build_short_signal_body(signal)
        msg.attach(MIMEText(text_body, "plain", "utf-8"))

        # HTML 版本
        html_body = self._build_html_body(signal)
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context) as server:
                server.login(self.smtp_from, self.smtp_password)
                server.sendmail(self.smtp_from, self.smtp_to, msg.as_string())

            logger.info(f"[email] ✉️  已发送: {symbol} 做空信号")
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error("[email] SMTP 认证失败，请检查 .env 中的 SMTP_PASSWORD 是否为授权码（非QQ密码）")
            return False
        except Exception as e:
            logger.error(f"[email] 发送失败: {e}")
            return False

    def send_pump_alert(self, subject: str, body: str) -> bool:
        """发送暴涨暴跌预警邮件"""
        if not all([self.smtp_from, self.smtp_to, self.smtp_password]):
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.smtp_from
        msg["To"] = self.smtp_to
        msg.attach(MIMEText(body, "plain", "utf-8"))

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context) as server:
                server.login(self.smtp_from, self.smtp_password)
                server.sendmail(self.smtp_from, self.smtp_to, msg.as_string())
            logger.info(f"[email] ✉️  已发送暴涨暴跌预警")
            return True
        except Exception as e:
            logger.error(f"[email] 预警邮件发送失败: {e}")
            return False

    def send_test_email(self) -> bool:
        """发送测试邮件，验证邮箱配置是否正确"""
        test_signal = {
            "symbol": "TESTUSDT",
            "probability": 0.72,
            "win_rate": 0.68,
            "ev": 0.0025,
            "model_version": "v_test",
            "price": 1.0000,
            "funding_rate": 0.0001,
            "drift_status": "stable",
            "pass_count": 4,
            "indicator_checks": {
                "RSI超买(>70)": True,
                "MACD死叉(hist<0)": True,
                "BB超买(>0.9)": False,
                "Funding偏高(>0.01%)": True,
                "OI下降(<0)": True,
                "量能异常(>2x均量)": False,
            },
            "ev_detail": {
                "fee_cost": 0.001,
                "slippage_cost": 0.0006,
                "funding_net": 0.0001,
            },
        }
        logger.info("[email] 发送测试邮件...")
        return self.send_short_signal(test_signal)

    def send_startup_notification(self, mode: str, symbols_count: int, top_n: int) -> bool:
        """
        系统启动成功时发送通知邮件。
        告知用户：各功能已就绪，正在24小时监控中。
        """
        if not all([self.smtp_from, self.smtp_to, self.smtp_password]):
            logger.warning("[email] 邮箱未配置，跳过启动通知")
            return False

        now_cst = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"✅ 微信用户AB699Q专属 | CryptoWatch V4 已启动 | {now_cst}"

        body = f"""
════════════════════════════════════
  ✅ 微信用户AB699Q专属 — CryptoWatch 启动成功
════════════════════════════════════

系统已正常运行，正在24小时监控中。

📊 运行配置：
  模式          : {mode.upper()}
  监控合约数    : {symbols_count} 个
  子集数量      : 前 {top_n} 个（按成交额）
  启动时间      : {now_cst} CST

🔧 各功能状态：
  ✓ Binance API 连接正常
  ✓ 数据下载模块就绪
  ✓ 特征工程模块就绪（shift(1)防未来函数）
  ✓ XGBoost 模型已加载
  ✓ EV 计算模块就绪
  ✓ 极端行情保护（BTC波动>{config.BTC_VOLATILE_THRESHOLD}%暂停）
  ✓ 冷却机制（信号消失→重现才触发）
  ✓ 自动日志清理（7天滚动）
  ✓ 邮件通知就绪

📬 信号触发条件：
  模型做空概率 ≥ {config.MIN_SHORT_PROB*100:.0f}%
  预期收益 EV > 0
  至少 3/6 项指标通过

⏰ 监控节奏：
  每根K线收盘后（每小时整点+5秒）扫描一次
  触发信号时会立即发送邮件通知

════════════════════════════════════
  本邮件为系统自动发送，无需回复
════════════════════════════════════
""".strip()

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.smtp_from
        msg["To"] = self.smtp_to
        msg.attach(MIMEText(body, "plain", "utf-8"))

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context) as server:
                server.login(self.smtp_from, self.smtp_password)
                server.sendmail(self.smtp_from, self.smtp_to, msg.as_string())
            logger.info("[email] ✅ 启动通知邮件已发送")
            return True
        except Exception as e:
            logger.error(f"[email] 启动通知发送失败: {e}")
            return False
