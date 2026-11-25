"""用标注检索样本校准多信号证据闸，分值是可解释指标而非回答正确率。"""

import json
import math
from pathlib import Path


def evidence_signals(matches, relevance_floor):
    """从去重精排结果提取 Top1、有效证据数与 Top1/Top2 分差。"""
    unique, seen = [], set()
    for index, match in enumerate(matches):
        identity = match.get("chunk_id", index)
        score = float(match.get("score", 0))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("证据闸只接受 0 到 1 的精排相关性分。")
        if identity not in seen:
            seen.add(identity)
            unique.append(score)
    scores = sorted(unique, reverse=True)
    top1 = scores[0] if scores else 0
    return {
        "top1": top1,
        "effective_count": sum(s >= relevance_floor for s in scores),
        "margin": round(top1 - (scores[1] if len(scores) > 1 else 0), 10),
    }


def confidence_value(signals, weights):
    """将三个同尺度信号线性加权，有效数在三条处饱和以免大量弱片段抬分。"""
    features = (signals["top1"], min(signals["effective_count"], 3) / 3, signals["margin"])
    return sum(value * weight for value, weight in zip(features, weights, strict=True))


class EvidencePolicy:
    def __init__(self, artifact):
        """加载包含标注集来源、三项权重、有效分下限与接受阈值的校准产物。"""
        self.artifact = artifact
        if len(artifact["weights"]) != 3 or not math.isclose(sum(artifact["weights"]), 1):
            raise ValueError("证据置信度校准权重无效。")

    @classmethod
    def load(cls, path):
        """显式读取 UTF-8 校准数据，缺失时明确失败，不回退到拍脑袋阈值。"""
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def assess(self, matches):
        """返回证据分、各原始信号和闸结果；零有效证据始终拒绝。"""
        signals = evidence_signals(matches, self.artifact["relevance_floor"])
        value = confidence_value(signals, self.artifact["weights"])
        return {
            "evidence_confidence": value,
            "signals": signals,
            "threshold": self.artifact["threshold"],
            "accepted": bool(signals["effective_count"] and value >= self.artifact["threshold"]),
        }


def calibrate(rows, *, dataset_hash, metadata=None):
    """优先保留 ch04 可回答题的覆盖，再最小化平衡错误率，搜索经验分界点。

    三信号权重均大于零；同等错误率选择离样本更远的分界，保留误判明细供人工复核。
    开发集校准不视为独立效果评估，生成自评仍负责高相关但条件不足的情况。
    不通过拒绝已知多证据问题来压低模糊题误放；无法由检索信号区分的情况交给生成自评。
    """
    positives = [r for r in rows if not r["expected_refusal"]]
    negatives = [r for r in rows if r["expected_refusal"]]
    if not positives or not negatives:
        raise ValueError("校准集必须同时包含可回答与应拒答样本。")
    tops = sorted(evidence_signals(r["matches"], 0)["top1"] for r in positives)
    floors = sorted({tops[int((len(tops) - 1) * q)] for q in (0, 0.25, 0.5, 0.75)})
    best = None
    for floor in floors:
        signals = [evidence_signals(r["matches"], floor) for r in rows]
        for first in range(1, 9):
            for second in range(1, 10 - first):
                weights = [first / 10, second / 10, (10 - first - second) / 10]
                values = [confidence_value(s, weights) for s in signals]
                ordered = sorted(set(values))
                thresholds = [
                    0,
                    1.000001,
                    *[(a + b) / 2 for a, b in zip(ordered, ordered[1:], strict=False)],
                ]
                for threshold in thresholds:
                    accepted = [
                        bool(s["effective_count"] and v >= threshold)
                        for s, v in zip(signals, values, strict=True)
                    ]
                    false_accept = sum(
                        a and r["expected_refusal"] for a, r in zip(accepted, rows, strict=True)
                    )
                    false_reject = sum(
                        not a and not r["expected_refusal"]
                        for a, r in zip(accepted, rows, strict=True)
                    )
                    error = false_accept / len(negatives) + false_reject / len(positives)
                    clearance = min(abs(v - threshold) for v in values)
                    rank = (false_reject, round(error, 10), false_accept, -clearance, -weights[0])
                    if best is None or rank < best[0]:
                        best = (
                            rank,
                            {
                                "weights": weights,
                                "relevance_floor": floor,
                                "threshold": threshold,
                                "balanced_accuracy": 1 - error / 2,
                                "false_accept_ids": [
                                    r["id"]
                                    for a, r in zip(accepted, rows, strict=True)
                                    if a and r["expected_refusal"]
                                ],
                                "false_reject_ids": [
                                    r["id"]
                                    for a, r in zip(accepted, rows, strict=True)
                                    if not a and not r["expected_refusal"]
                                ],
                            },
                        )
    return {
        "version": 1,
        "dataset_hash": dataset_hash,
        "sample_count": len(rows),
        "metadata": metadata or {},
        "validation": "开发集校准，非独立留出评估",
        "objective": "先最小化可回答题误拦，再最小化平衡错误率；模糊高相关题由生成自评处理",
        **best[1],
    }
