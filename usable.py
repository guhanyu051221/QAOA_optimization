import argparse
import time
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.circuit import Parameter
from qiskit import transpile
from qiskit_aer import AerSimulator
import numpy as np
from qiskit.opflow import PauliSumOp
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter


def data_preprocessing(file_path):
    df = pd.read_excel(file_path)
    data = df.values
    RoR = np.zeros([data.shape[0]-1,data.shape[1]]) 
    for i in range(data.shape[0]-1):
        RoR[i,:] = (data[i+1,:]-data[i,:])/data[i,:]
    exp_ret = pd.Series(np.mean(RoR, axis=0))
    cov_mat = pd.DataFrame(np.cov(RoR, rowvar=False))
    return exp_ret, cov_mat

def calc_J(num_qubits, half_q, cov_mat, eta):
    J = np.zeros((num_qubits, num_qubits))
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            J[i, j] = half_q * cov_mat[i, j] / 2.0 + eta / 2.0
            J[j, i] = J[i, j]
    return J

def calc_h(num_qubits, half_q, cov_mat, theta1, exp_ret, eta, budget):
    h = np.zeros(num_qubits)
    row_sum = np.sum(cov_mat, axis=1)
    for i in range(num_qubits):
        h[i] = -half_q * row_sum[i] / 2.0 + theta1 * exp_ret[i] / 2.0 + eta * (budget - num_qubits / 2.0)
    return h

def insert_RX(num_qubits):
    qc = QuantumCircuit(num_qubits)
    beta = Parameter('beta_tmp')
    for qubit in range(num_qubits):
        qc.rx(2 * beta, qubit)
    qc.barrier()
    return qc

def insert_RZ(num_qubits, h):
    qc = QuantumCircuit(num_qubits)
    gamma = Parameter('gamma_tmp')
    for qubit in range(num_qubits):
        qc.rz(2 * gamma * h[qubit], qubit)
    qc.barrier()
    return qc

def insert_RZZ(gamma, J, num_qubits):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if np.abs(J[i, j]) > 0:
                qc.rzz(2 * gamma * J[i, j], i, j)
    qc.barrier()
    return qc

def insert_H(num_qubits):
    qc = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        qc.h(qubit)
    qc.barrier()
    return qc

def get_Pauli(num_qubits, index, type):
    if type == 'Z':
        assert len(index) == 1
        index = index[0]
        assert index >= 0 and index <= num_qubits - 1
        _Pauli = ['I'] * (num_qubits - 1)
        _Pauli.insert(index, 'Z')
        _Pauli = ''.join(_Pauli)
        return _Pauli
    elif type == 'ZZ':
        assert len(index) == 2
        _Pauli = ['I'] * (num_qubits - 2)
        for i in range(len(index)):
            assert index[i] >= 0 and index[i] <= num_qubits - 1
            _Pauli.insert(index[i], 'Z')
        _Pauli = ''.join(_Pauli)
        return _Pauli
    else:
        raise AssertionError()

def problem_PauliOperator(num_qubits, h, J):
    Pauli_h_list = []
    for i in range(num_qubits):
        Pauli_h_list.append((get_Pauli(num_qubits, [i], 'Z'), h[i]))
    Pauli_h = PauliSumOp.from_list(Pauli_h_list, coeff=1.0)
    Pauli_J_list = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            Pauli_J_list.append((get_Pauli(num_qubits, [i,j], 'ZZ'), J[i][j]))
    Pauli_J = PauliSumOp.from_list(Pauli_J_list, coeff=1.0)
    Pauli_sum = Pauli_h + Pauli_J
    return Pauli_h, Pauli_J, Pauli_sum

def oneCircuit(num_qubits, h, J, beta, gamma):
    qc = QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        qc.rz(2 * gamma * h[qubit], qubit)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if np.abs(J[i, j]) > 0:
                qc.rzz(2 * gamma * J[i, j], i, j)
    qc.barrier()
    for qubit in range(num_qubits):
        qc.rx(2 * beta, qubit)
    qc.barrier()
    return qc

def get_expectation(circuit, para_list, Hamiltonian, simulator, num_qubits):
    def execute_circ(theta):
        qc = QuantumCircuit(num_qubits)
        p = len(theta) // 2
        beta = theta[:p]
        gamma = theta[p:]
        para_dict = {}
        for i in range(p):
            para_dict[para_list[i]] = beta[i]
            para_dict[para_list[i+p]] = gamma[i]
        qc.append(circuit, [i for i in range(0, num_qubits)])
        qc.assign_parameters(para_dict, inplace=True)
        circ = transpile(qc, simulator)
        result = simulator.run(circ).result()
        _statevector = result.get_statevector(circ)
        loss = _statevector.expectation_value(Hamiltonian)
        assert np.imag(loss) < 1e-10
        return np.real(loss)
    return execute_circ

def str_to_statevector(string):
    string = string[::-1]
    dec = int(string, 2)
    state = np.zeros(2 ** len(string))
    state[dec] = 1.0
    return state[None,:]

def print_config(budget, num_slices, eta, layers):
    print('%%%%%%%%%%%%%%%%%%%% Configuration %%%%%%%%%%%%%%%%%%%%')
    print('budget: %d, g: %d, eta: %f, layers: %d' % (budget, num_slices, eta, layers))

def print_result(circuit, Hamiltonian, para_list, solution, simulator, num_qubits):
    qc = QuantumCircuit(num_qubits)
    p = len(solution) // 2
    beta = solution[:p]
    gamma = solution[p:]
    para_dict = {}
    for i in range(p):
        para_dict[para_list[i]] = beta[i]
        para_dict[para_list[i + p]] = gamma[i]
    qc.append(circuit, [i for i in range(0, num_qubits)])
    qc.assign_parameters(para_dict, inplace=True)
    circ = transpile(qc, simulator)
    result = simulator.run(circ).result()
    statevector = result.get_statevector(circ)  
    statevector_dict = statevector.to_dict()
    
    # 计算概率
    probabilities = {}
    for i in statevector_dict:
        probabilities[i] = np.abs(np.array(statevector_dict[i])) ** 2
    
    # 按概率排序
    sorted_results = sorted(probabilities.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    
    # 构建矩阵用于计算期望值
    mm = []
    for i in range(len(sorted_results)):
        x, prob = sorted_results[i]
        mm.append(str_to_statevector(x))
    mm = np.concatenate(mm, axis=0)
    value_mm = np.sum((mm @ Hamiltonian) * mm, axis=1)
    
    # 找到最小值的索引
    min_index = np.argmin(value_mm)
    best_state, best_prob = sorted_results[min_index]
    best_value = value_mm[min_index]
    
    # 打印结果
    print("{}\t{:.8f}\t{:.6f}".format(best_state[::-1], best_value, best_prob))
    
    return best_state[::-1], best_value, best_prob

class callback:
    def __init__(self, step_size: int):
        self.step_size = step_size
        self.full_values = []
        self._values = []
        self.values = []
    def __call__(self, nfev, parameters, value, stepsize, accepted):
        self.full_values.append(value)
        self._values.append(value)
        if len(self._values) == self.step_size:
            last_value = self._values[-1]
            self.values.append(last_value)
            self._values = []
            return self.values

def print_loss(callback_func):
    loss_ls = callback_func.values
    min_loss = min(loss_ls)
    print("%.10f" % min_loss)
    return min_loss


def run_portfolio_optimization(
    budget=4,
    num_assets=6,
    num_slices=1,
    theta1=1.0,
    half_q=0.25,
    eta=1.0,
    seed=123456,
    optimizer_choice=False,
    maxiter=300,
    Gf=1.0,
    lr=0.01,
    layers=3,
    epochs=1,
    file_path="./data/stock_data.xlsx"
):
    """
    运行投资组合优化问题
    
    参数:
        budget: 总资产
        num_assets: 资产数量
        num_slices: 表示一个资产所需的二进制位数
        theta1: 线性项系数
        half_q: 二次项系数
        eta: 拉格朗日项系数
        seed: 随机种子
        optimizer_choice: 是否使用scipy优化器(True)或SPSA(False)
        maxiter: 最大迭代次数
        Gf: 粒度
        lr: 初始学习率
        layers: QAOA层数
        epochs: 训练轮数
        file_path: 数据文件路径
    
    返回:
        min_loss: 最小损失值
        result: 最优结果字符串
        value: 最优值
    """
    
    # 参数处理
    Gf = 1.0 / budget
    num_qubits = num_assets * num_slices
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 打印配置
    print_config(budget, num_slices, eta, layers)
    
    # 加载数据
    exp_ret, cov_mat = data_preprocessing(file_path)
    exp_ret = exp_ret.to_numpy()
    cov_mat = cov_mat.to_numpy()
    
    # 计算J和h
    J = calc_J(num_qubits, half_q, cov_mat, eta)
    h = calc_h(num_qubits, half_q, cov_mat, theta1, exp_ret, eta, budget)
    
    # 构建Pauli算子
    Pauli_h, Pauli_J, Pauli_sum = problem_PauliOperator(num_qubits, h, J)
    
    # 设置模拟器
    simulator = AerSimulator(method='statevector')
    simulator.set_options(
        max_parallel_threads=0,
        max_parallel_experiments=1,
        max_parallel_shots=0,
        statevector_parallel_threshold=14
    )
    
    # 构建量子电路
    qc = QuantumCircuit(num_qubits)
    beta = []
    gamma = []
    para_list = []
    for i in range(layers):
        name = "β%d" % i
        beta.append(Parameter(name))
        name = "γ%d" % i
        gamma.append(Parameter(name))
    
    para_list = beta + gamma
    qc.append(insert_H(num_qubits), [i for i in range(0, num_qubits)])
    for i in range(layers):
        qc.append(oneCircuit(num_qubits, h, J, beta[i], gamma[i]), [i for i in range(0, num_qubits)])
    qc.save_statevector()
    
    print('\nCircuit Initialization Complete! Start Training...', flush=True)
    
    # 获取期望值函数
    expectation = get_expectation(qc, para_list, Pauli_sum, simulator, num_qubits)
    
    start = time.time()
    
    if optimizer_choice:
        # 使用scipy优化器
        res = minimize(expectation,
                       np.random.uniform(0, 0.001 * np.pi, size=layers * 2),
                       method='COBYLA',
                       options={'maxiter': maxiter})
        print(res)
        solution = res.x
        min_loss = res.fun
    else:
        # 使用SPSA优化器
        step_size = 1
        callback_func = callback(step_size)
        optimizer = SPSA(maxiter=maxiter, blocking=True, second_order=True, callback=callback_func)
        res = optimizer.optimize(num_vars=layers * 2, 
                                objective_function=expectation, 
                                initial_point=np.random.uniform(0, 0.001*np.pi, size=layers * 2))
        solution = res[0]
        min_loss = print_loss(callback_func)
    
    elapsed_s = time.time() - start
    print("\nTraining done! Total elapsed time:{:.2f}s".format(elapsed_s))
    
    # 获取结果
    state, value,prob = print_result(qc, Pauli_sum.to_matrix(), para_list, solution, simulator, num_qubits)
    
    return min_loss, state, value, prob, elapsed_s


def _safe_mean(values):
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def _safe_var(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(np.var(arr, ddof=1))


def _state_to_code(state):
    if state is None or pd.isna(state):
        return np.nan
    return int(str(state), 2)


def summarize_repeated_runs(rows):
    """Summarize 10 repeated runs for one parameter value."""
    success_rows = [row for row in rows if row["success"]]
    states = [row["result"] for row in success_rows]
    state_counter = Counter(states)
    mode_state, mode_count = (None, 0)
    if state_counter:
        mode_state, mode_count = state_counter.most_common(1)[0]

    return {
        "runs": len(rows),
        "success_runs": len(success_rows),
        "min_loss_mean": _safe_mean([row["min_loss"] for row in success_rows]),
        "min_loss_var": _safe_var([row["min_loss"] for row in success_rows]),
        "value_mean": _safe_mean([row["value"] for row in success_rows]),
        "value_var": _safe_var([row["value"] for row in success_rows]),
        "prob_mean": _safe_mean([row["prob"] for row in success_rows]),
        "prob_var": _safe_var([row["prob"] for row in success_rows]),
        "elapsed_s_mean": _safe_mean([row["elapsed_s"] for row in success_rows]),
        "elapsed_s_var": _safe_var([row["elapsed_s"] for row in success_rows]),
        "result_mode": mode_state,
        "result_mode_count": mode_count,
        "result_code_mean": _safe_mean([_state_to_code(row["result"]) for row in success_rows]),
        "result_code_var": _safe_var([_state_to_code(row["result"]) for row in success_rows]),
    }


def run_repeated_parameter_point(repeat_count=10, seed_base=123456, **kwargs):
    """Run one parameter setting repeatedly and return raw run rows plus summary."""
    rows = []
    for repeat_id in range(repeat_count):
        run_seed = seed_base + repeat_id
        print(f"    repeat {repeat_id + 1}/{repeat_count}, seed={run_seed}")
        try:
            min_loss, state, value, prob, elapsed_s = run_portfolio_optimization(
                seed=run_seed,
                **kwargs
            )
            rows.append({
                "repeat": repeat_id + 1,
                "seed": run_seed,
                "success": True,
                "min_loss": float(min_loss),
                "value": float(np.real(value)),
                "prob": float(prob),
                "elapsed_s": float(elapsed_s),
                "result": state,
                "error": "",
            })
        except Exception as exc:
            print(f"    failed: {exc}")
            rows.append({
                "repeat": repeat_id + 1,
                "seed": run_seed,
                "success": False,
                "min_loss": np.nan,
                "value": np.nan,
                "prob": np.nan,
                "elapsed_s": np.nan,
                "result": None,
                "error": str(exc),
            })
    return rows, summarize_repeated_runs(rows)


def plot_parameter_relationship(x_values, mean_values, var_values, x_label, y_label, title, save_path):
    """Plot parameter relationship using means, with standard deviation error bars."""
    plt.figure(figsize=(12, 8))
    mean_values = np.array(mean_values, dtype=float)
    var_values = np.array(var_values, dtype=float)
    std_values = np.sqrt(np.maximum(var_values, 0.0))

    plt.errorbar(
        x_values,
        mean_values,
        yerr=std_values,
        fmt='bo-',
        linewidth=2,
        markersize=7,
        capsize=5,
        label=f"{y_label} mean ± std",
    )
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像已保存到: {save_path}")


def run_parameter_scan():
    """运行完整的参数扫描实验。每个参数点重复10次，使用均值作图并计算方差。"""

    # 创建figures目录
    figures_dir = "./figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # 定义参数扫描范围
    eta_values = np.arange(0, 11, 2)  # 0, 2, 4, 6, 8, 10
    layer_values = range(1, 7)  # 1, 2, 3, 4, 5, 6
    maxiter_values = [50, 100, 200, 300, 500]
    repeat_count = 10
    seed_base = 123456
    
    # 默认值
    default_eta = 6
    default_layer = 4
    default_maxiter = 250
    
    # 存储所有实验结果
    all_results = {
        'eta_scan': {},      # 改变eta时的结果
        'layer_scan': {},    # 改变layer时的结果
        'maxiter_scan': {}   # 改变maxiter时的结果
    }
    
    print("="*80)
    print("开始参数扫描实验")
    print(f"每个参数点重复运行 {repeat_count} 次，并统计均值和方差")
    print("="*80)

    def run_scan(scan_name, param_name, param_values, fixed_kwargs):
        print(f"\n{scan_name}")
        print(f"  {param_name}范围: {list(param_values)}")
        print("-"*60)

        raw_rows = []
        summary_rows = []
        for param_value in param_values:
            print(f"\n运行 {param_name}={param_value}...")
            kwargs = dict(fixed_kwargs)
            kwargs[param_name] = param_value
            rows, summary = run_repeated_parameter_point(
                repeat_count=repeat_count,
                seed_base=seed_base,
                **kwargs
            )
            for row in rows:
                row[param_name] = param_value
                raw_rows.append(row)
            summary[param_name] = param_value
            summary_rows.append(summary)
            print(
                "  summary: "
                f"min_loss_mean={summary['min_loss_mean']:.8f}, "
                f"min_loss_var={summary['min_loss_var']:.8f}, "
                f"prob_mean={summary['prob_mean']:.8f}, "
                f"prob_var={summary['prob_var']:.8f}, "
                f"mode={summary['result_mode']} ({summary['result_mode_count']}/{summary['success_runs']})"
            )

        return pd.DataFrame(raw_rows), pd.DataFrame(summary_rows)

    base_kwargs = dict(
        budget=4,
        num_assets=6,
        num_slices=1,
        theta1=1.0,
        half_q=0.25,
        optimizer_choice=False,
        file_path="./data/stock_data.xlsx",
    )

    eta_raw, eta_summary = run_scan(
        scan_name="实验组1: 扫描eta参数",
        param_name="eta",
        param_values=eta_values,
        fixed_kwargs=dict(
            base_kwargs,
            layers=default_layer,
            maxiter=default_maxiter,
        ),
    )

    layer_raw, layer_summary = run_scan(
        scan_name="实验组2: 扫描layers参数",
        param_name="layers",
        param_values=layer_values,
        fixed_kwargs=dict(
            base_kwargs,
            eta=default_eta,
            maxiter=default_maxiter,
        ),
    )

    maxiter_raw, maxiter_summary = run_scan(
        scan_name="实验组3: 扫描maxiter参数",
        param_name="maxiter",
        param_values=maxiter_values,
        fixed_kwargs=dict(
            base_kwargs,
            eta=default_eta,
            layers=default_layer,
        ),
    )

    # 保存详细结果和统计摘要
    eta_raw.to_csv(os.path.join(figures_dir, 'eta_scan_raw.csv'), index=False)
    eta_summary.to_csv(os.path.join(figures_dir, 'eta_scan_summary.csv'), index=False)
    layer_raw.to_csv(os.path.join(figures_dir, 'layer_scan_raw.csv'), index=False)
    layer_summary.to_csv(os.path.join(figures_dir, 'layer_scan_summary.csv'), index=False)
    maxiter_raw.to_csv(os.path.join(figures_dir, 'maxiter_scan_raw.csv'), index=False)
    maxiter_summary.to_csv(os.path.join(figures_dir, 'maxiter_scan_summary.csv'), index=False)

    all_results = {
        'eta_scan': eta_summary,
        'layer_scan': layer_summary,
        'maxiter_scan': maxiter_summary,
    }

    # ==================== 生成12张图 ====================
    print("\n\n" + "="*80)
    print("开始生成图像 (共12张，均值作图，误差棒为标准差)")
    print("="*80)

    metric_specs = [
        ('min_loss', 'Minimum Loss', 'min_loss'),
        ('value', 'Optimal Value', 'value'),
        ('prob', 'Probability', 'probability'),
        ('result_code', 'State Code', 'state_code'),
    ]

    scan_specs = [
        ('eta', eta_summary, 'Eta', f'layers={default_layer}, maxiter={default_maxiter}', 'eta'),
        ('layers', layer_summary, 'Number of Layers', f'eta={default_eta}, maxiter={default_maxiter}', 'layer'),
        ('maxiter', maxiter_summary, 'Max Iterations', f'eta={default_eta}, layers={default_layer}', 'maxiter'),
    ]

    for param_col, summary_df, x_label, condition_text, file_prefix in scan_specs:
        x_values = summary_df[param_col].to_numpy()
        for metric_key, metric_name, y_label in metric_specs:
            mean_col = f"{metric_key}_mean"
            var_col = f"{metric_key}_var"
            plot_parameter_relationship(
                x_values=x_values,
                mean_values=summary_df[mean_col].to_numpy(),
                var_values=summary_df[var_col].to_numpy(),
                x_label=x_label,
                y_label=y_label,
                title=f'Effect of {x_label} on {metric_name} ({condition_text}, repeated {repeat_count} times)',
                save_path=os.path.join(figures_dir, f'{file_prefix}_vs_{metric_key}_mean.png')
            )

    # 兼容旧报告文件名：用均值图覆盖对应旧文件名，便于Markdown直接显示新图
    plot_parameter_relationship(
        eta_summary['eta'].to_numpy(),
        eta_summary['prob_mean'].to_numpy(),
        eta_summary['prob_var'].to_numpy(),
        'Eta',
        'probability',
        f'Effect of Eta on Probability (layers={default_layer}, maxiter={default_maxiter}, repeated {repeat_count} times)',
        os.path.join(figures_dir, 'eta_sweep_layers4_maxiter250_mean.png')
    )
    plot_parameter_relationship(
        layer_summary['layers'].to_numpy(),
        layer_summary['prob_mean'].to_numpy(),
        layer_summary['prob_var'].to_numpy(),
        'Number of Layers',
        'probability',
        f'Effect of Layers on Probability (eta={default_eta}, maxiter={default_maxiter}, repeated {repeat_count} times)',
        os.path.join(figures_dir, 'layers_sweep_eta6_maxiter250_mean.png')
    )
    plot_parameter_relationship(
        maxiter_summary['maxiter'].to_numpy(),
        maxiter_summary['prob_mean'].to_numpy(),
        maxiter_summary['prob_var'].to_numpy(),
        'Max Iterations',
        'probability',
        f'Effect of Maxiter on Probability (eta={default_eta}, layers={default_layer}, repeated {repeat_count} times)',
        os.path.join(figures_dir, 'maxiter_sweep_layers4_eta6_mean.png')
    )
    plot_parameter_relationship(
        maxiter_summary['maxiter'].to_numpy(),
        maxiter_summary['elapsed_s_mean'].to_numpy(),
        maxiter_summary['elapsed_s_var'].to_numpy(),
        'Max Iterations',
        'elapsed_s',
        f'Effect of Maxiter on Runtime (eta={default_eta}, layers={default_layer}, repeated {repeat_count} times)',
        os.path.join(figures_dir, 'maxiter_vs_elapsed_s_mean.png')
    )

    # 生成综合报告
    print("\n" + "="*80)
    print("生成综合报告")
    print("="*80)

    report_path = os.path.join(figures_dir, 'parameter_scan_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Parameter Scan Experiment Report\n")
        f.write("="*80 + "\n\n")
        f.write("Experiment Configuration:\n")
        f.write(f"  Repeats per parameter: {repeat_count}\n")
        f.write(f"  Seed base: {seed_base}\n")
        f.write(f"  Default eta: {default_eta}\n")
        f.write(f"  Default layers: {default_layer}\n")
        f.write(f"  Default maxiter: {default_maxiter}\n\n")
        f.write("Scan Ranges:\n")
        f.write(f"  Eta: {list(eta_values)}\n")
        f.write(f"  Layers: {list(layer_values)}\n")
        f.write(f"  Maxiter: {list(maxiter_values)}\n\n")
        f.write("Variance uses sample variance, ddof=1.\n")
        f.write("Plots use mean values; error bars are standard deviations.\n\n")

        def write_summary(title, df, param_col):
            f.write("="*80 + "\n")
            f.write(title + "\n")
            f.write("="*80 + "\n")
            for _, row in df.iterrows():
                f.write(
                    f"{param_col}={row[param_col]}: "
                    f"min_loss_mean={row['min_loss_mean']:.8f}, min_loss_var={row['min_loss_var']:.8f}, "
                    f"value_mean={row['value_mean']:.8f}, value_var={row['value_var']:.8f}, "
                    f"prob_mean={row['prob_mean']:.8f}, prob_var={row['prob_var']:.8f}, "
                    f"elapsed_s_mean={row['elapsed_s_mean']:.2f}, elapsed_s_var={row['elapsed_s_var']:.2f}, "
                    f"result_mode={row['result_mode']} ({int(row['result_mode_count'])}/{int(row['success_runs'])})\n"
                )
            best_idx = df['min_loss_mean'].astype(float).idxmin()
            best = df.loc[best_idx]
            f.write(
                f"Best by mean min_loss: {param_col}={best[param_col]} "
                f"(mean={best['min_loss_mean']:.8f}, var={best['min_loss_var']:.8f})\n\n"
            )

        write_summary("Eta Scan", eta_summary, "eta")
        write_summary("Layer Scan", layer_summary, "layers")
        write_summary("Maxiter Scan", maxiter_summary, "maxiter")

    print(f"综合报告已保存到: {report_path}")
    print("\n" + "="*80)
    print("所有实验完成！")
    print("总共生成 15 张图片：12 张指标均值图 + 3 张报告用 probability 均值图")
    print(f"结果保存在: {figures_dir}")
    print("="*80)

    return all_results


# 修改主程序，调用参数扫描函数
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import numpy as np
    
    # 运行完整的参数扫描实验
    results = run_parameter_scan()
