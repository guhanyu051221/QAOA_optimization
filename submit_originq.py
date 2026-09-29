import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector


DEFAULT_SOLUTION = [
    0.03572766,
    0.11688645,
    0.02639345,
    0.22500591,
    -0.00097934,
    -0.03978164,
    -0.26076253,
    -0.15046207,
]


def data_preprocessing(file_path):
    """Read stock prices and return expected returns plus covariance matrix."""
    df = pd.read_excel(file_path)
    data = df.values

    returns = np.zeros([data.shape[0] - 1, data.shape[1]])
    for i in range(data.shape[0] - 1):
        returns[i, :] = (data[i + 1, :] - data[i, :]) / data[i, :]

    exp_ret = np.mean(returns, axis=0)
    cov_mat = np.cov(returns, rowvar=False)
    return exp_ret, cov_mat


def calc_coefficients(exp_ret, cov_mat, budget, eta, theta1=1.0, half_q=0.25):
    """Build the Ising coefficients h and J used by the QAOA cost layer."""
    num_qubits = len(exp_ret)
    j_mat = np.zeros((num_qubits, num_qubits))

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            j_mat[i, j] = half_q * cov_mat[i, j] / 2.0 + eta / 2.0
            j_mat[j, i] = j_mat[i, j]

    row_sum = np.sum(cov_mat, axis=1)
    h_vec = np.zeros(num_qubits)
    for i in range(num_qubits):
        h_vec[i] = (
            -half_q * row_sum[i] / 2.0
            + theta1 * exp_ret[i] / 2.0
            + eta * (budget - num_qubits / 2.0)
        )

    return h_vec, j_mat


def build_qaoa_circuit(h_vec, j_mat, solution, layers, with_measurements=False):
    """Create a fixed-parameter QAOA circuit ready for local simulation or export."""
    num_qubits = len(h_vec)
    expected_params = 2 * layers
    if len(solution) != expected_params:
        raise ValueError(
            f"Expected {expected_params} parameters for layers={layers}, "
            f"but got {len(solution)}."
        )

    beta = solution[:layers]
    gamma = solution[layers:]

    qc = QuantumCircuit(num_qubits, num_qubits if with_measurements else 0)

    for qubit in range(num_qubits):
        qc.h(qubit)

    for layer in range(layers):
        for qubit in range(num_qubits):
            qc.rz(2 * gamma[layer] * h_vec[qubit], qubit)

        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                if abs(j_mat[i, j]) > 0:
                    qc.rzz(2 * gamma[layer] * j_mat[i, j], i, j)

        for qubit in range(num_qubits):
            qc.rx(2 * beta[layer], qubit)

    if with_measurements:
        qc.measure(range(num_qubits), range(num_qubits))

    return qc


def ideal_probabilities(circuit):
    """Return ideal statevector probabilities in the report's bit order."""
    state = Statevector.from_instruction(circuit)
    raw_probs = state.probabilities_dict()

    # Qiskit prints bitstrings in little-endian order relative to the portfolio
    # convention used in the report, so reverse the string for readability.
    probs = {bitstring[::-1]: probability for bitstring, probability in raw_probs.items()}
    return dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))


def write_probability_csv(path, probabilities):
    """Save ideal probabilities to CSV for later comparison with hardware counts."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["selection", "ideal_probability"])
        for selection, probability in probabilities.items():
            writer.writerow([selection, f"{probability:.10f}"])


def circuit_to_originir(circuit):
    """Convert the portable Qiskit circuit to simple OriginIR syntax."""
    lines = [f"QINIT {circuit.num_qubits}", f"CREG {circuit.num_clbits}"]

    for instruction, qargs, cargs in circuit.data:
        name = instruction.name.lower()
        qubits = [circuit.find_bit(qubit).index for qubit in qargs]
        cbits = [circuit.find_bit(cbit).index for cbit in cargs]

        if name == "h":
            lines.append(f"H q[{qubits[0]}]")
        elif name == "rx":
            lines.append(f"RX q[{qubits[0]}],({float(instruction.params[0]):.15f})")
        elif name == "rz":
            lines.append(f"RZ q[{qubits[0]}],({float(instruction.params[0]):.15f})")
        elif name == "cx":
            lines.append(f"CNOT q[{qubits[0]}],q[{qubits[1]}]")
        elif name == "measure":
            lines.append(f"MEASURE q[{qubits[0]}],c[{cbits[0]}]")
        elif name == "barrier":
            continue
        else:
            raise ValueError(f"Unsupported gate for OriginIR export: {instruction.name}")

    return "\n".join(lines) + "\n"


def parse_solution(text):
    """Parse comma-separated or space-separated QAOA parameters."""
    if not text:
        return DEFAULT_SOLUTION
    parts = text.replace(",", " ").split()
    return [float(part) for part in parts]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a fixed-parameter QAOA circuit for OriginQ/OriginQC-style "
            "cloud submission and local theory comparison."
        )
    )
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--eta", type=float, default=6.0)
    parser.add_argument("--data", default="./data/stock_data.xlsx")
    parser.add_argument(
        "--solution",
        default=None,
        help=(
            "QAOA parameters. Use comma or space separated values. "
            "Default is the layers=4, eta=6, maxiter=200 solution from the report."
        ),
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--output-dir", default="./output/originq")
    args = parser.parse_args()

    solution = parse_solution(args.solution)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_ret, cov_mat = data_preprocessing(args.data)
    h_vec, j_mat = calc_coefficients(exp_ret, cov_mat, args.budget, args.eta)

    ideal_circuit = build_qaoa_circuit(
        h_vec=h_vec,
        j_mat=j_mat,
        solution=solution,
        layers=args.layers,
        with_measurements=False,
    )
    measured_circuit = build_qaoa_circuit(
        h_vec=h_vec,
        j_mat=j_mat,
        solution=solution,
        layers=args.layers,
        with_measurements=True,
    )

    # Most cloud platforms do not accept RZZ directly. Transpiling to a small
    # basis gives a more portable OpenQASM file using cx/rz/rx/h/measure.
    portable_circuit = transpile(
        measured_circuit,
        basis_gates=["h", "rx", "rz", "cx", "measure"],
        optimization_level=1,
    )

    qasm_path = output_dir / "qaoa_fixed_params.qasm"
    originir_path = output_dir / "qaoa_fixed_params.originir"
    theory_csv_path = output_dir / "ideal_probabilities.csv"
    summary_path = output_dir / "submission_notes.txt"

    qasm_path.write_text(portable_circuit.qasm(), encoding="utf-8")
    originir_path.write_text(circuit_to_originir(portable_circuit), encoding="utf-8")

    probabilities = ideal_probabilities(ideal_circuit)
    write_probability_csv(theory_csv_path, probabilities)

    best_selection, best_probability = next(iter(probabilities.items()))
    summary_path.write_text(
        "\n".join(
            [
                "Fixed-parameter QAOA export",
                f"budget = {args.budget}",
                f"layers = {args.layers}",
                f"eta = {args.eta}",
                f"shots suggested for hardware = {args.shots}",
                f"parameters = {solution}",
                f"highest ideal-probability selection = {best_selection}",
                f"highest ideal probability = {best_probability:.10f}",
                "",
                "For the OriginIR editor, paste qaoa_fixed_params.originir.",
                "Upload qaoa_fixed_params.qasm only if the cloud platform accepts OpenQASM.",
                "After hardware execution, compare hardware counts/shots with ideal_probabilities.csv.",
            ]
        ),
        encoding="utf-8",
    )

    print("Prepared fixed-parameter QAOA circuit.")
    print(f"OpenQASM file: {qasm_path}")
    print(f"OriginIR file: {originir_path}")
    print(f"Ideal probability CSV: {theory_csv_path}")
    print(f"Notes: {summary_path}")
    print(f"Highest ideal-probability selection: {best_selection}")
    print(f"Highest ideal probability: {best_probability:.10f}")


if __name__ == "__main__":
    main()
