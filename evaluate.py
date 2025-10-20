#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import sys
from math import sqrt
from typing import List, Tuple

RE_KEY = re.compile(r'^(\d+)_gt(\d+)$')  # 例: "29_gt145" -> 样本名=29, gt=145

def parse_json(path: str) -> Tuple[List[float], List[float]]:
    """读取JSON并返回(y_true, y_pred)列表"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    y_true, y_pred = [], []
    for k, v in data.items():
        m = RE_KEY.match(str(k))
        if not m:
            continue
        gt = float(m.group(2))
        pred = float(v)
        y_true.append(gt)
        y_pred.append(pred)
    return y_true, y_pred


def mae(y_true: List[float], y_pred: List[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))


def mape(y_true: List[float], y_pred: List[float]) -> float:
    """
    Mean Absolute Percentage Error (%)
    忽略真实值为0的样本
    """
    values = [abs((a - b) / a) for a, b in zip(y_true, y_pred) if a != 0]
    return sum(values) / len(values) * 100 if values else float('nan')


def rmspe(y_true: List[float], y_pred: List[float]) -> float:
    """
    Root Mean Squared Percentage Error (%)
    忽略真实值为0的样本
    """
    values = [((a - b) / a) ** 2 for a, b in zip(y_true, y_pred) if a != 0]
    return sqrt(sum(values) / len(values)) * 100 if values else float('nan')


def r2_score_safe(y_true: List[float], y_pred: List[float]) -> float:
    """R² 安全实现（兼容常量情况）"""
    n = len(y_true)
    if n == 0:
        raise ValueError("Empty y_true/y_pred.")
    y_mean = sum(y_true) / n
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))

    if ss_tot == 0.0:
        all_equal = all(yp == y_true[0] for yp in y_pred)
        return 1.0 if all_equal else 0.0
    return 1.0 - ss_res / ss_tot


def main():
    if len(sys.argv) != 2:
        print("用法: python calc_metrics.py <path_to_json>")
        sys.exit(1)

    path = sys.argv[1]
    y_true, y_pred = parse_json(path)

    if len(y_true) == 0:
        print("未解析到任何 (gt, pred) 对，请检查键名是否形如 '29_gt145'.")
        sys.exit(2)

    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    rmspe_val = rmspe(y_true, y_pred)
    r2_val = r2_score_safe(y_true, y_pred)

    print(f"样本数 : {len(y_true)}")
    print(f"MAE    : {mae_val:.6f}")
    print(f"RMSE   : {rmse_val:.6f}")
    print(f"MAPE   : {mape_val:.6f} %")
    print(f"RMSPE  : {rmspe_val:.6f} %")
    print(f"R²     : {r2_val:.6f}")


if __name__ == "__main__":
    main()
