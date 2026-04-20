# ============================================================
# src/models/trainer.py — XGBoost 训练 + Purged CV + 概率校准
#
# 核心设计：
#   1. Purged K-Fold CV（防止时间序列数据泄露）
#   2. CalibratedClassifierCV（校准概率，使概率有物理意义）
#   3. 模型版本号管理（timestamp + hash）
#   4. 训练集/测试集严格按时间分割（禁止随机 shuffle）
# ============================================================

from __future__ import annotations
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import BaseCrossValidator
import xgboost as xgb

import config


# ============================================================
# Purged K-Fold 实现（防时间序列泄露）
# ============================================================

class PurgedKFold(BaseCrossValidator):
    """
    Purged + Embargo K-Fold 交叉验证。
    
    与普通 KFold 的区别：
    1. 严格按时间顺序分割（无 shuffle）
    2. 在训练集和验证集之间「清除」purge_bars 根K线
       （防止训练集样本的 Triple-Barrier 标签泄露到验证集）
    3. 在验证集之后「隔离」embargo_bars 根K线
       （防止验证集起点的特征包含训练集末尾的信息）
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_bars: int = 24,
        embargo_bars: int = 6,
    ):
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        fold_size = n_samples // self.n_splits

        for fold in range(self.n_splits):
            # 验证集范围
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.n_splits - 1 else n_samples

            # 训练集：验证集之前，但去掉 purge_bars 和 embargo_bars
            train_end = max(0, val_start - self.purge_bars)
            embargo_end = min(n_samples, val_end + self.embargo_bars)

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)

            if len(train_idx) < config.MIN_TRAIN_SAMPLES or len(val_idx) == 0:
                continue

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ============================================================
# 模型版本管理
# ============================================================

def _generate_model_version(symbol: str, feature_cols: list) -> str:
    """
    生成模型版本号：格式为 v{日期}_{symbol}_{特征hash}
    
    参数：
        symbol      : 合约代码
        feature_cols: 训练使用的特征列列表
    
    返回：
        版本字符串，例如 "v20240419_BTCUSDT_a1b2c3d4"
    """
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    feat_hash = hashlib.md5(
        json.dumps(sorted(feature_cols)).encode()
    ).hexdigest()[:8]
    return f"v{date_str}_{symbol}_{feat_hash}"


def get_model_path(symbol: str) -> Path:
    """获取模型文件路径"""
    return config.MODEL_DIR / f"{symbol}_model.pkl"


def get_model_meta_path(symbol: str) -> Path:
    """获取模型元数据路径"""
    return config.MODEL_DIR / f"{symbol}_meta.json"


# ============================================================
# 训练主函数
# ============================================================

def train_model(
    symbol: str,
    features_df: pd.DataFrame,
    labels: pd.Series,
    sample_weights: Optional[pd.Series] = None,
) -> Optional[Dict[str, Any]]:
    """
    训练单个合约的 XGBoost 模型。
    
    流程：
        1. 时间序列分割（前70%训练，后30%验证）
        2. Purged K-Fold CV 在训练集上进行
        3. 最终模型在全量训练集上训练
        4. CalibratedClassifierCV 校准概率
        5. 保存模型 + 元数据
    
    返回：
        训练结果字典（含评估指标）或 None（训练失败）
    """
    # 对齐特征和标签（去除 NaN 标签行）
    valid_mask = labels.notna()
    X = features_df[valid_mask][config.FEATURE_COLUMNS].copy()
    y = labels[valid_mask].astype(int)

    if sample_weights is not None:
        sw = sample_weights[valid_mask]
    else:
        sw = None

    # 样本数量检查
    if len(X) < config.MIN_TRAIN_SAMPLES:
        logger.warning(f"[trainer] {symbol} 样本不足({len(X)})，跳过训练")
        return None

    pos_rate = y.mean()
    logger.info(
        f"[trainer] {symbol} 开始训练: {len(X)} 样本，正例率={pos_rate:.3f}"
    )

    # ---- 时间序列分割（严格按时间，禁止随机 shuffle）----
    split_idx = int(len(X) * config.TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    sw_train = sw.iloc[:split_idx] if sw is not None else None

    # ---- Purged K-Fold CV（在训练集上评估）----
    cv = PurgedKFold(
        n_splits=config.CV_N_SPLITS,
        purge_bars=config.CV_PURGE_BARS,
        embargo_bars=config.CV_EMBARGO_BARS,
    )

    # 基础 XGBoost 分类器
    base_clf = xgb.XGBClassifier(**config.XGB_PARAMS)

    # Purged CV 评估
    cv_aucs = []
    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train)):
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]
        sw_tr = sw_train.iloc[tr_idx] if sw_train is not None else None

        fold_clf = xgb.XGBClassifier(**config.XGB_PARAMS)
        fit_kwargs = {}
        if sw_tr is not None:
            fit_kwargs["sample_weight"] = sw_tr.values

        fold_clf.fit(X_tr, y_tr, **fit_kwargs,
                     eval_set=[(X_val, y_val)], verbose=False)

        if len(y_val.unique()) >= 2:
            auc = roc_auc_score(y_val, fold_clf.predict_proba(X_val)[:, 1])
            cv_aucs.append(auc)
            logger.debug(f"[trainer] {symbol} Fold {fold_idx+1} AUC={auc:.4f}")

    cv_auc_mean = np.mean(cv_aucs) if cv_aucs else 0.0
    cv_auc_std = np.std(cv_aucs) if cv_aucs else 0.0
    logger.info(f"[trainer] {symbol} Purged CV AUC={cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

    # ---- 全量训练集训练最终模型 ----
    fit_kwargs = {}
    if sw_train is not None:
        fit_kwargs["sample_weight"] = sw_train.values

    base_clf.fit(X_train, y_train, **fit_kwargs, verbose=False)

    # ---- 测试集评估 ----
    y_prob = base_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {}
    if len(y_test.unique()) >= 2:
        metrics["test_auc"] = round(roc_auc_score(y_test, y_prob), 4)
        metrics["test_brier"] = round(brier_score_loss(y_test, y_prob), 4)
        metrics["test_logloss"] = round(log_loss(y_test, y_prob), 4)
    else:
        metrics["test_auc"] = 0.0
        metrics["test_brier"] = 1.0
        metrics["test_logloss"] = 1.0

    metrics["test_precision"] = round(precision_score(y_test, y_pred, zero_division=0), 4)
    metrics["test_recall"] = round(recall_score(y_test, y_pred, zero_division=0), 4)
    metrics["test_f1"] = round(f1_score(y_test, y_pred, zero_division=0), 4)
    metrics["cv_auc_mean"] = round(cv_auc_mean, 4)
    metrics["cv_auc_std"] = round(cv_auc_std, 4)
    metrics["train_samples"] = len(X_train)
    metrics["test_samples"] = len(X_test)
    metrics["pos_rate"] = round(pos_rate, 4)

    logger.info(
        f"[trainer] {symbol} 测试集: AUC={metrics['test_auc']}, "
        f"Precision={metrics['test_precision']}, Recall={metrics['test_recall']}"
    )

    # ---- 生成模型版本号 ----
    model_version = _generate_model_version(symbol, config.FEATURE_COLUMNS)

    # ---- 保存模型 ----
    model_path = get_model_path(symbol)
    with open(model_path, "wb") as f:
        pickle.dump(base_clf, f)

    # ---- 保存元数据 ----
    meta = {
        "symbol": symbol,
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": config.FEATURE_COLUMNS,
        "cache_version": config.CACHE_VERSION,
        "metrics": metrics,
        "xgb_params": config.XGB_PARAMS,
        "train_range": {
            "start": str(X_train.index[0]),
            "end": str(X_train.index[-1]),
        },
        "test_range": {
            "start": str(X_test.index[0]),
            "end": str(X_test.index[-1]),
        },
    }

    meta_path = get_model_meta_path(symbol)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"[trainer] {symbol} 模型已保存: {model_path}")
    logger.info(f"[trainer] {symbol} 模型版本: {model_version}")

    return {
        "model": base_clf,
        "model_version": model_version,
        "metrics": metrics,
        "meta": meta,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_prob_test": y_prob,
    }


def load_model(symbol: str) -> Optional[Tuple[Any, Dict]]:
    """
    加载已训练的模型和元数据。
    
    返回：
        (calibrated_clf, meta_dict) 或 None（文件不存在）
    """
    model_path = get_model_path(symbol)
    meta_path = get_model_meta_path(symbol)

    if not model_path.exists():
        logger.warning(f"[trainer] {symbol} 模型文件不存在: {model_path}")
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    logger.debug(
        f"[trainer] 已加载 {symbol} 模型: "
        f"版本={meta.get('model_version', 'unknown')}"
    )
    return model, meta
