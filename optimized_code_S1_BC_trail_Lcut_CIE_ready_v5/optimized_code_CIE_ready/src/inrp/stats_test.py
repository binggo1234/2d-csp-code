# stats_test.py
from __future__ import annotations
from typing import List, Dict, Any
import statistics

def _mean(xs): return statistics.mean(xs) if xs else 0.0
def _std(xs):  return statistics.pstdev(xs) if len(xs) > 1 else 0.0

def build_summary_stats(totals_all: List[Dict[str, Any]], variants: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for v in variants:
        data = [r for r in totals_all if r["variant"] == v]
        rows.append(dict(
            variant=v,
            n=len(data),
            N_board_mean=_mean([r["N_board"] for r in data]),
            N_board_std=_std([r["N_board"] for r in data]),
            U_mean=_mean([r["U_avg"] for r in data]),
            U_std=_std([r["U_avg"] for r in data]),
            L_shared_mean=_mean([r["L_shared_sum"] for r in data]),
            L_shared_std=_std([r["L_shared_sum"] for r in data]),
            N_lift_mean=_mean([r["N_lift_sum"] for r in data]),
            N_lift_std=_std([r["N_lift_sum"] for r in data]),
            L_air_mean=_mean([r["L_air_sum"] for r in data]),
            L_air_std=_std([r["L_air_sum"] for r in data]),
            L_cut_mean=_mean([r["L_cut_sum"] for r in data]),
            L_cut_std=_std([r["L_cut_sum"] for r in data]),
            L_cut_base_mean=_mean([r.get("L_cut_base_sum", 0.0) for r in data]),
            L_cut_base_std=_std([r.get("L_cut_base_sum", 0.0) for r in data]),
            L_kerf_extra_mean=_mean([r.get("L_kerf_extra_sum", 0.0) for r in data]),
            L_kerf_extra_std=_std([r.get("L_kerf_extra_sum", 0.0) for r in data]),
            L_lead_mean=_mean([r.get("L_lead_sum", 0.0) for r in data]),
            L_lead_std=_std([r.get("L_lead_sum", 0.0) for r in data]),
            T_est_mean=_mean([r["T_est_sum"] for r in data]),
            T_est_std=_std([r["T_est_sum"] for r in data]),
            runtime_mean=_mean([r.get("runtime_s", 0.0) for r in data]),
            runtime_std=_std([r.get("runtime_s", 0.0) for r in data]),
        ))

    # 可选：给 proposed_shared vs baseline 的 Wilcoxon p 值（如果安装了 scipy）
    try:
        from scipy.stats import wilcoxon
        def pval(metric: str, a: str, b: str) -> float:
            xa = [r[metric] for r in totals_all if r["variant"] == a]
            xb = [r[metric] for r in totals_all if r["variant"] == b]
            n = min(len(xa), len(xb))
            if n <= 1:
                return 1.0
            d = [xb[i] - xa[i] for i in range(n)]
            # 两边差异检验
            return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)

        for row in rows:
            if row["variant"] == "proposed_shared":
                row["p_T_vs_A"] = pval("T_est_sum", "baselineA", "proposed_shared")
                row["p_T_vs_B"] = pval("T_est_sum", "baselineB", "proposed_shared")
    except Exception:
        pass

    return rows
