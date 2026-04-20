# ============================================================
# src/data/cache_manager.py — 数据缓存版本控制（要求44）
# 每个 parquet 文件带 version 字段 + schema hash
# 特征升级后自动检测旧缓存并强制重新下载
# ============================================================

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from loguru import logger

import config


def _compute_schema_hash(columns: list[str], version: str) -> str:
    """
    计算特征列表的哈希值。
    当特征列发生变化时，哈希不同，触发缓存失效。
    """
    content = json.dumps({"columns": sorted(columns), "version": version})
    return hashlib.sha256(content.encode()).hexdigest()[:12]


# 当前版本的 schema hash（基于 FEATURE_COLUMNS + CACHE_VERSION）
CURRENT_SCHEMA_HASH = _compute_schema_hash(
    config.FEATURE_COLUMNS, config.CACHE_VERSION
)


def get_cache_path(
    symbol: str,
    data_type: str,  # "kline" / "funding" / "oi"
    interval: str = "1h",
) -> Path:
    """生成缓存文件路径"""
    return config.DATA_CACHE_DIR / data_type / f"{symbol}_{interval}.parquet"


def is_cache_valid(path: Path, max_age_hours: int = 1) -> bool:
    """
    检查缓存文件是否有效：
    1. 文件必须存在
    2. 版本号必须匹配
    3. schema hash 必须匹配
    4. 文件年龄不超过 max_age_hours
    """
    if not path.exists():
        return False

    try:
        # 读取 metadata（只读第一行元数据，不加载全部数据）
        df = pd.read_parquet(path, columns=["_cache_version", "_schema_hash"])
        if df.empty:
            return False

        cached_version = df["_cache_version"].iloc[0]
        cached_hash = df["_schema_hash"].iloc[0]

        # 版本号和 schema hash 必须都匹配
        if cached_version != config.CACHE_VERSION:
            logger.debug(f"[cache] {path.name} 版本不匹配({cached_version} vs {config.CACHE_VERSION})，需重建")
            return False

        if cached_hash != CURRENT_SCHEMA_HASH:
            logger.debug(f"[cache] {path.name} schema hash不匹配，需重建")
            return False

        # 检查文件修改时间
        import time
        file_age_hours = (time.time() - path.stat().st_mtime) / 3600
        if file_age_hours > max_age_hours:
            logger.debug(f"[cache] {path.name} 缓存已过期({file_age_hours:.1f}h > {max_age_hours}h)")
            return False

        return True

    except Exception as e:
        logger.warning(f"[cache] 读取缓存元数据失败: {path.name}, {e}")
        return False


def save_with_metadata(df: pd.DataFrame, path: Path) -> None:
    """
    保存 DataFrame 到 parquet，并附加版本和 schema 元数据。
    
    【设计要点】：
    在数据中附加 _cache_version 和 _schema_hash 列，
    这样读取时可以验证版本，防止特征升级后使用旧缓存导致错位。
    """
    # 确保目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["_cache_version"] = config.CACHE_VERSION
    df["_schema_hash"] = CURRENT_SCHEMA_HASH

    df.to_parquet(path, index=True, compression="snappy")
    logger.debug(f"[cache] 已保存: {path.name} ({len(df)} 行)")


def load_without_metadata(path: Path) -> pd.DataFrame:
    """
    读取 parquet 文件并去除版本元数据列。
    返回干净的业务数据。
    """
    df = pd.read_parquet(path)

    # 删除元数据列（不参与特征计算）
    meta_cols = [c for c in df.columns if c.startswith("_cache_")]
    df = df.drop(columns=meta_cols, errors="ignore")

    return df


def invalidate_all_cache(data_type: Optional[str] = None) -> int:
    """
    手动使缓存失效（删除 parquet 文件）。
    
    参数：
        data_type: None = 清除所有缓存；
                   "kline" / "funding" / "oi" = 只清除该类型
    返回：删除文件数量
    """
    count = 0
    if data_type:
        target_dir = config.DATA_CACHE_DIR / data_type
    else:
        target_dir = config.DATA_CACHE_DIR

    for f in target_dir.rglob("*.parquet"):
        f.unlink()
        count += 1
        logger.debug(f"[cache] 已删除: {f.name}")

    logger.info(f"[cache] 共清除 {count} 个缓存文件")
    return count
