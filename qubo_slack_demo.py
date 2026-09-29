import argparse
from itertools import product

import numpy as np
import pandas as pd


def data_preprocessing(file_path):
    df = pd.read_excel(file_path)
    data = df.values
    returns = np.zeros([data.shape[0] - 1, data.shape[1]])
    for i in range(data.shape[0] - 1):
        returns[i, :] = (data[i + 1, :] - data[i, :]) / data[i, :]
    exp_ret = np.mean(returns, axis=0)
    cov_mat = np.cov(returns, rowvar=False)
    return exp_ret, cov_mat


def bits_needed(max_value):
    if max_value < 0:
        raise ValueError("max_value must be non-negative")
    bits = 1
    while 2**bits - 1 < max_value:
        bits += 1
    return bits


def decode_binary(bits, widths):
    values = []
    cursor = 0
    for width in widths:
        value = 0
        for offset in range(width):
            value += (2**offset) * bits[cursor + offset]
        values.append(value)
        cursor += width
    return np.array(values, dtype=int), cursor


def evaluate_objective(y, slack, exp_ret, cov_mat, prices, capital, risk_lambda, eta):
    invested = float(np.dot(prices, y))
    revenue = float(np.dot(exp_ret, y))
    risk = float(0.5 * risk_lambda * (y @ cov_mat @ y))
    penalty = float(eta * (capital - invested - slack) ** 2)
    score = revenue - risk - penalty
    return {
        "score": score,
        "revenue": revenue,
        "risk": risk,
        "penalty": penalty,
        "invested": invested,
        "unused_cash": capital - invested,
        "slack": float(slack),
        "constraint_error": capital - invested - slack,
        "feasible_budget": invested <= capital,
    }


def enumerate_qubo(exp_ret, cov_mat, prices, capital, max_shares, risk_lambda, eta):
    asset_widths = [bits_needed(max_share) for max_share in max_shares]
    max_slack = int(capital)
    slack_width = bits_needed(max_slack)
    total_bits = sum(asset_widths) + slack_width

    if total_bits > 24:
        raise ValueError(
            f"Enumeration would require 2^{total_bits} states. "
            "Reduce max_shares/capital or switch to an optimizer."
        )

    rows = []
    for bits in product([0, 1], repeat=total_bits):
        y, cursor = decode_binary(bits, asset_widths)
        slack_bits = bits[cursor:]
        slack = int(sum((2**idx) * bit for idx, bit in enumerate(slack_bits)))
        metrics = evaluate_objective(
            y=y,
            slack=slack,
            exp_ret=exp_ret,
            cov_mat=cov_mat,
            prices=prices,
            capital=capital,
            risk_lambda=risk_lambda,
            eta=eta,
        )
        rows.append(
            {
                "bits": "".join(str(bit) for bit in bits),
                "shares": " ".join(str(v) for v in y),
                **metrics,
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    return result, asset_widths, slack_width


def main():
    parser = argparse.ArgumentParser(
        description="Enumerate a QUBO model with multi-share holdings and slack variable."
    )
    parser.add_argument("--data", default="./data/stock_data.xlsx")
    parser.add_argument("--capital", type=int, default=10)
    parser.add_argument("--prices", default="2,3,2,4,3,2")
    parser.add_argument("--max-shares", default="2,2,2,2,2,2")
    parser.add_argument("--risk-lambda", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=6.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="./output/qubo_slack_results.csv")
    args = parser.parse_args()

    exp_ret, cov_mat = data_preprocessing(args.data)
    prices = np.array([int(x) for x in args.prices.split(",")], dtype=int)
    max_shares = np.array([int(x) for x in args.max_shares.split(",")], dtype=int)

    if len(prices) != len(exp_ret):
        raise ValueError("prices length must match number of assets")
    if len(max_shares) != len(exp_ret):
        raise ValueError("max-shares length must match number of assets")

    result, asset_widths, slack_width = enumerate_qubo(
        exp_ret=exp_ret,
        cov_mat=cov_mat,
        prices=prices,
        capital=args.capital,
        max_shares=max_shares,
        risk_lambda=args.risk_lambda,
        eta=args.eta,
    )

    result.to_csv(args.output, index=False)

    print("QUBO slack model enumeration complete.")
    print(f"capital: {args.capital}")
    print(f"prices: {prices.tolist()}")
    print(f"max_shares: {max_shares.tolist()}")
    print(f"asset bit widths: {asset_widths}")
    print(f"slack bit width: {slack_width}")
    print(f"total enumerated states: {len(result)}")
    print(f"full result csv: {args.output}")
    print()
    print("Top solutions:")
    cols = [
        "shares",
        "slack",
        "invested",
        "unused_cash",
        "constraint_error",
        "score",
        "revenue",
        "risk",
        "penalty",
        "feasible_budget",
    ]
    print(result[cols].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
