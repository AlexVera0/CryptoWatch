# ============================================================
# src/utils/log_cleaner.py — 自动日志清理
# 防止日志/缓存/快照累积过多占用磁盘
# ============================================================

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

import config


def clean_old_logs(keep_days: int = None):
    """清理超过 keep_days 天的日志文件"""
    if keep_days is None:
        keep_days = config.LOG_RETENTION_DAYS
    log_dir = config.LOG_DIR
    cutoff = datetime.now() - timedelta(days=keep_days)
    count = 0
    for f in log_dir.glob("*.log"):
        if f.stat().st_mtime < cutoff.timestamp():
            f.unlink()
            count += 1
    if count:
        logger.info(f"[cleaner] 清理旧日志 {count} 个（>{keep_days}天）")


def clean_old_signal_snapshots(keep_days: int = None):
    """清理超过 keep_days 天的信号快照目录"""
    if keep_days is None:
        keep_days = config.SIGNAL_SNAPSHOT_KEEP_DAYS
    snap_dir = config.SIGNAL_SNAPSHOT_DIR
    if not snap_dir.exists():
        return
    cutoff = datetime.now() - timedelta(days=keep_days)
    count = 0
    for date_dir in snap_dir.iterdir():
        if date_dir.is_dir() and date_dir.stat().st_mtime < cutoff.timestamp():
            shutil.rmtree(date_dir)
            count += 1
    if count:
        logger.info(f"[cleaner] 清理旧快照目录 {count} 个（>{keep_days}天）")


def clean_old_reports(keep_days: int = None):
    """清理超过 keep_days 天的 HTML 报告"""
    if keep_days is None:
        keep_days = config.REPORT_KEEP_DAYS
    report_dir = config.REPORT_DIR
    if not report_dir.exists():
        return
    cutoff = datetime.now() - timedelta(days=keep_days)
    count = 0
    for f in report_dir.glob("*.html"):
        if f.stat().st_mtime < cutoff.timestamp():
            f.unlink()
            count += 1
    if count:
        logger.info(f"[cleaner] 清理旧报告 {count} 个（>{keep_days}天）")



def clean_old_cache(keep_hours: int = 2):
    """清理过期 parquet 缓存（已由 cache_manager 控制，此处做兜底清理）"""
    pass  # parquet缓存已有版本控制，不做强制清理


def run_all_cleanups():
    """执行全部清理任务（每次K线收盘后调用一次）"""
    logger.debug("[cleaner] 执行定期清理...")
    clean_old_logs(keep_days=7)
    clean_old_signal_snapshots(keep_days=7)
    clean_old_reports(keep_days=14)
