import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from usable import run_portfolio_optimization


def sample_variance(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(np.var(arr, ddof=1))


def safe_mean(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.mean(arr))


def state_to_code(state):
    if state is None or pd.isna(state):
        return np.nan
    return int(str(state), 2)


def summarize_rows(rows):
    success_rows = [row for row in rows if row["success"]]
    states = [row["result"] for row in success_rows]
    state_counter = Counter(states)
    result_mode, result_mode_count = (None, 0)
    if state_counter:
        result_mode, result_mode_count = state_counter.most_common(1)[0]

    return {
        "runs": len(rows),
        "success_runs": len(success_rows),
        "min_loss_mean": safe_mean([row["min_loss"] for row in success_rows]),
        "min_loss_var": sample_variance([row["min_loss"] for row in success_rows]),
        "value_mean": safe_mean([row["value"] for row in success_rows]),
        "value_var": sample_variance([row["value"] for row in success_rows]),
        "prob_mean": safe_mean([row["prob"] for row in success_rows]),
        "prob_var": sample_variance([row["prob"] for row in success_rows]),
        "elapsed_s_mean": safe_mean([row["elapsed_s"] for row in success_rows]),
        "elapsed_s_var": sample_variance([row["elapsed_s"] for row in success_rows]),
        "result_mode": result_mode,
        "result_mode_count": result_mode_count,
        "result_code_mean": safe_mean([state_to_code(row["result"]) for row in success_rows]),
        "result_code_var": sample_variance([state_to_code(row["result"]) for row in success_rows]),
    }


def plot_metric(summary_df, metric_key, y_label, title, save_path):
    x_values = summary_df["maxiter"].to_numpy()
    mean_values = summary_df[f"{metric_key}_mean"].to_numpy(dtype=float)
    var_values = summary_df[f"{metric_key}_var"].to_numpy(dtype=float)
    std_values = np.sqrt(np.maximum(var_values, 0.0))

    plt.figure(figsize=(12, 8))
    plt.errorbar(
        x_values,
        mean_values,
        yerr=std_values,
        fmt="bo-",
        linewidth=2,
        markersize=7,
        capsize=5,
        label=f"{y_label} mean ± std",
    )
    plt.xlabel("Max Iterations", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"图像已保存到: {save_path}")


def run_maxiter_scan(
    maxiter_values=(50, 100, 200, 300, 500),
    repeat_count=10,
    seed_base=123456,
    output_dir="./figures",
):
    """Scan maxiter and measure probability, runtime, value, and loss."""
    os.makedirs(output_dir, exist_ok=True)

    raw_rows = []
    summary_rows = []

    print("=" * 80)
    print("开始 maxiter 参数扫描实验")
    print(f"maxiter 范围: {list(maxiter_values)}")
    print(f"每个 maxiter 重复运行 {repeat_count} 次")
    print("记录指标: min_loss, value, probability, elapsed_s, result")
    print("=" * 80)

    for maxiter in maxiter_values:
        current_rows = []
        print(f"\n运行 maxiter={maxiter}")

        for repeat_id in range(repeat_count):
            seed = seed_base + repeat_id
            print(f"  repeat {repeat_id + 1}/{repeat_count}, seed={seed}")

            try:
                min_loss, state, value, prob, elapsed_s = run_portfolio_optimization(
                    budget=4,
                    num_assets=6,
                    num_slices=1,
                    theta1=1.0,
                    half_q=0.25,
                    eta=6,
                    seed=seed,
                    optimizer_choice=False,
                    maxiter=maxiter,
                    layers=4,
                    file_path="./data/stock_data.xlsx",
                )
                row = {
                    "maxiter": maxiter,
                    "repeat": repeat_id + 1,
                    "seed": seed,
                    "success": True,
                    "min_loss": float(min_loss),
                    "value": float(np.real(value)),
                    "prob": float(prob),
                    "elapsed_s": float(elapsed_s),
                    "result": state,
                    "error": "",
                }
            except Exception as exc:
                row = {
                    "maxiter": maxiter,
                    "repeat": repeat_id + 1,
                    "seed": seed,
                    "success": False,
                    "min_loss": np.nan,
                    "value": np.nan,
                    "prob": np.nan,
                    "elapsed_s": np.nan,
                    "result": None,
                    "error": str(exc),
                }
                print(f"  failed: {exc}")

            raw_rows.append(row)
            current_rows.append(row)

        summary = summarize_rows(current_rows)
        summary["maxiter"] = maxiter
        summary_rows.append(summary)
        print(
            "  summary: "
            f"min_loss_mean={summary['min_loss_mean']:.8f}, "
            f"prob_mean={summary['prob_mean']:.8f}, "
            f"value_mean={summary['value_mean']:.8f}, "
            f"elapsed_s_mean={summary['elapsed_s_mean']:.2f}, "
            f"mode={summary['result_mode']} ({summary['result_mode_count']}/{summary['success_runs']})"
        )

    raw_df = pd.DataFrame(raw_rows)
    summary_df = pd.DataFrame(summary_rows)

    raw_path = os.path.join(output_dir, "maxiter_scan_raw.csv")
    summary_path = os.path.join(output_dir, "maxiter_scan_summary.csv")
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_metric(
        summary_df,
        "prob",
        "Probability",
        f"Effect of Maxiter on Probability (eta=6, layers=4, repeated {repeat_count} times)",
        os.path.join(output_dir, "maxiter_vs_prob_mean.png"),
    )
    plot_metric(
        summary_df,
        "elapsed_s",
        "Elapsed Time (s)",
        f"Effect of Maxiter on Runtime (eta=6, layers=4, repeated {repeat_count} times)",
        os.path.join(output_dir, "maxiter_vs_elapsed_s_mean.png"),
    )
    plot_metric(
        summary_df,
        "value",
        "Optimal Value",
        f"Effect of Maxiter on Optimal Value (eta=6, layers=4, repeated {repeat_count} times)",
        os.path.join(output_dir, "maxiter_vs_value_mean.png"),
    )
    plot_metric(
        summary_df,
        "min_loss",
        "Minimum Loss",
        f"Effect of Maxiter on Minimum Loss (eta=6, layers=4, repeated {repeat_count} times)",
        os.path.join(output_dir, "maxiter_vs_min_loss_mean.png"),
    )
    plot_metric(
        summary_df,
        "result_code",
        "State Code",
        f"Effect of Maxiter on Result State Code (eta=6, layers=4, repeated {repeat_count} times)",
        os.path.join(output_dir, "maxiter_vs_result_code_mean.png"),
    )

    report_path = os.path.join(output_dir, "maxiter_scan_report.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("Maxiter Scan Report\n")
        file.write("=" * 80 + "\n\n")
        file.write(f"maxiter values: {list(maxiter_values)}\n")
        file.write(f"repeat count: {repeat_count}\n")
        file.write("fixed eta: 6\n")
        file.write("fixed layers: 4\n")
        file.write("variance: sample variance, ddof=1\n")
        file.write("plots: mean values with standard deviation error bars\n\n")
        for _, row in summary_df.iterrows():
            file.write(
                f"maxiter={row['maxiter']}: "
                f"min_loss_mean={row['min_loss_mean']:.8f}, min_loss_var={row['min_loss_var']:.8f}, "
                f"value_mean={row['value_mean']:.8f}, value_var={row['value_var']:.8f}, "
                f"prob_mean={row['prob_mean']:.8f}, prob_var={row['prob_var']:.8f}, "
                f"elapsed_s_mean={row['elapsed_s_mean']:.2f}, elapsed_s_var={row['elapsed_s_var']:.2f}, "
                f"result_mode={row['result_mode']} ({int(row['result_mode_count'])}/{int(row['success_runs'])})\n"
            )

    print("\n" + "=" * 80)
    print("maxiter 参数扫描实验完成")
    print(f"逐次运行结果: {raw_path}")
    print(f"统计结果: {summary_path}")
    print(f"报告: {report_path}")
    print("=" * 80)

    return raw_df, summary_df


if __name__ == "__main__":
    run_maxiter_scan()
