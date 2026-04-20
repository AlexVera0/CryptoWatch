# src/report/html_report.py — HTML 报告生成（要求47）
# 包含：equity curve、drawdown、EV分布、概率校准曲线、PSI drift图

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from loguru import logger

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

import config


def _eq_dd_chart(equity: list, drawdown: list, symbol: str) -> str:
    if not PLOTLY_OK:
        return "<p>请安装 plotly</p>"
    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                        subplot_titles=["资金曲线", "回撤(%)"],
                        vertical_spacing=0.08)
    x = list(range(len(equity)))
    fig.add_trace(go.Scatter(x=x, y=equity, mode="lines", name="净值",
                             line=dict(color="#00d4ff", width=2),
                             fill="tozeroy", fillcolor="rgba(0,212,255,0.08)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[d*100 for d in drawdown], mode="lines",
                             name="回撤%", line=dict(color="#ff4757", width=1.5),
                             fill="tozeroy", fillcolor="rgba(255,71,87,0.15)"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=560,
                      paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font=dict(color="#c9d1d9"), title=f"{symbol} 回测")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _ev_chart(trades_df: pd.DataFrame) -> str:
    if not PLOTLY_OK or trades_df.empty:
        return ""
    pnl = trades_df["pnl_ratio"].values * 100
    fig = go.Figure(go.Histogram(x=pnl, nbinsx=50, marker_color="#f39c12", opacity=0.8))
    fig.add_vline(x=np.mean(pnl), line_dash="dash", line_color="#27ae60",
                  annotation_text=f"均值 {np.mean(pnl):.3f}%")
    fig.add_vline(x=0, line_color="#e74c3c", line_width=2)
    fig.update_layout(template="plotly_dark", height=360,
                      paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font=dict(color="#c9d1d9"), title="EV 分布（每笔PnL%）",
                      xaxis_title="PnL (%)", yaxis_title="频次")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _calib_chart(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> str:
    if not PLOTLY_OK or len(y_true) == 0:
        return ""
    bins = np.linspace(0, 1, n_bins + 1)
    centers, freqs = [], []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            centers.append((bins[i]+bins[i+1])/2)
            freqs.append(float(y_true[mask].mean()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(color="#8b949e", dash="dash"), name="完美校准"))
    fig.add_trace(go.Scatter(x=centers, y=freqs, mode="lines+markers",
                             line=dict(color="#00d4ff", width=2),
                             marker=dict(size=8), name="模型校准"))
    fig.update_layout(template="plotly_dark", height=360,
                      paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font=dict(color="#c9d1d9"), title="概率校准曲线 (Reliability Curve)",
                      xaxis_title="预测概率", yaxis_title="实际频率")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _psi_chart(psi_dict: Dict[str, float]) -> str:
    if not PLOTLY_OK or not psi_dict:
        return ""
    feats = list(psi_dict.keys())
    vals = list(psi_dict.values())
    colors = ["#e74c3c" if v >= config.PSI_CRITICAL_THRESHOLD
              else "#f39c12" if v >= config.PSI_WARNING_THRESHOLD
              else "#27ae60" for v in vals]
    fig = go.Figure(go.Bar(y=feats, x=vals, orientation="h", marker_color=colors))
    fig.add_vline(x=config.PSI_WARNING_THRESHOLD, line_dash="dash", line_color="#f39c12")
    fig.add_vline(x=config.PSI_CRITICAL_THRESHOLD, line_dash="dash", line_color="#e74c3c")
    fig.update_layout(template="plotly_dark", height=max(300, len(feats)*22+100),
                      paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                      font=dict(color="#c9d1d9"), title="Concept Drift — PSI 漂移图")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def generate_backtest_report(
    symbol: str,
    backtest_result: Dict[str, Any],
    y_true: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    psi_dict: Optional[Dict[str, float]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """生成完整 HTML 回测报告（要求47：必须包含5个图表）"""
    metrics = backtest_result.get("metrics", {})
    equity = backtest_result.get("equity_curve", [])
    dd = backtest_result.get("drawdown", [])
    trades_df = backtest_result.get("trades", pd.DataFrame())

    eq_html = _eq_dd_chart(equity, dd, symbol)
    ev_html = _ev_chart(trades_df)
    cal_html = _calib_chart(y_true, y_prob) if y_true is not None else ""
    psi_html = _psi_chart(psi_dict) if psi_dict else ""

    metrics_rows = "".join(
        f"<tr><td>{k}</td><td style='color:#00d4ff;font-weight:bold;'>{v}</td></tr>"
        for k, v in metrics.items() if k != "symbol"
    )
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html><html lang="zh-CN">
<head><meta charset="UTF-8"><title>CryptoWatch V4 — {symbol}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body{{background:#0d1117;color:#c9d1d9;font-family:Arial,sans-serif;padding:24px;}}
h1{{color:#00d4ff;border-bottom:2px solid #30363d;padding-bottom:10px;}}
h2{{color:#8b949e;font-size:15px;margin-top:28px;}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;margin:14px 0;}}
table{{width:100%;border-collapse:collapse;}}
td{{padding:7px 10px;border-bottom:1px solid #21262d;}}
td:first-child{{color:#8b949e;}}
.warn{{color:#e74c3c;font-size:13px;padding:12px;background:#1c1010;
       border-left:4px solid #e74c3c;border-radius:4px;margin-top:20px;}}
</style></head>
<body>
<h1>📊 CryptoWatch V4 — {symbol} 回测报告</h1>
<p style="color:#8b949e;">生成时间：{now_str} | 缓存版本：{config.CACHE_VERSION}</p>
<div class="card"><h2>📈 关键指标</h2><table>{metrics_rows}</table></div>
<div class="card"><h2>💹 资金曲线 & 回撤</h2>{eq_html}</div>
<div class="card"><h2>📊 EV 分布（每笔PnL%）</h2>{ev_html}</div>
{"<div class='card'><h2>🎯 概率校准曲线</h2>" + cal_html + "</div>" if cal_html else ""}
{"<div class='card'><h2>🔍 PSI Drift 图</h2>" + psi_html + "</div>" if psi_html else ""}
<div class="warn">⚠️ <b>风险提醒</b>：历史回测不代表未来。回测与实盘存在显著差距（延迟+滑点+不成交）。
Regime Shift、Concept Drift、黑天鹅风险极高，必须手动最终确认后方可实盘。</div>
</body></html>"""

    if output_path is None:
        output_path = config.REPORT_DIR / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"[report] HTML报告已生成: {output_path}")
    return output_path
