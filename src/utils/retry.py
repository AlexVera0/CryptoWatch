# ============================================================
# src/utils/retry.py — API 指数退避重试装饰器
# 防止网络波动导致的请求失败，自动重试并记录日志
# ============================================================

import time
import functools
from typing import Callable, Type, Tuple
from loguru import logger


def with_retry(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    log_prefix: str = "",
) -> Callable:
    """
    指数退避重试装饰器。
    
    策略：失败后等待 base_delay * 2^(attempt-1) 秒，最长不超过 max_delay。
    例：1s, 2s, 4s, 8s, 16s ...
    
    参数：
        max_retries   : 最大重试次数
        base_delay    : 初始等待秒数
        max_delay     : 最大等待秒数（防止等待过长）
        exceptions    : 捕获哪些异常类型触发重试
        log_prefix    : 日志前缀标识
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            prefix = log_prefix or func.__name__
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    # 计算等待时间（指数退避）
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

                    if attempt < max_retries:
                        logger.warning(
                            f"[{prefix}] 第 {attempt}/{max_retries} 次失败: {e}，"
                            f"{delay:.1f}s 后重试..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[{prefix}] 已达最大重试次数 ({max_retries})，"
                            f"最终失败: {e}"
                        )

            # 所有重试均失败，抛出最后一次异常
            raise last_exception

        return wrapper
    return decorator


def retry_request(
    func: Callable,
    *args,
    max_retries: int = 5,
    base_delay: float = 1.0,
    **kwargs,
):
    """
    函数式重试封装（不使用装饰器时调用）。
    
    用法：
        result = retry_request(requests.get, url, timeout=15)
    """
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            delay = min(base_delay * (2 ** (attempt - 1)), 60.0)
            if attempt < max_retries:
                logger.warning(
                    f"[retry_request] {func.__name__} 第{attempt}次失败: {e}，"
                    f"{delay:.1f}s后重试"
                )
                time.sleep(delay)
    raise last_exception
