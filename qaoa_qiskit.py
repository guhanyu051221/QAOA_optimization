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


def data_preprocessing(file_path):
    '''
    数据预处理函数。

    作用：
    读取 Excel 中的股票收盘价数据，把价格序列转换成收益率序列，
    然后计算投资组合模型需要的两个核心输入：
    1. 期望收益向量 exp_ret：每只股票平均每天能带来的收益；
    2. 协方差矩阵 cov_mat：描述股票之间风险和联动关系的矩阵。
    '''
    # 读取 Excel 文件。每一列代表一只股票，每一行代表一个时间点的收盘价。
    df = pd.read_excel(file_path)

    # 将 pandas 的 DataFrame 转换为 numpy 数组，方便后续做数值计算。
    data = df.values

    # closing price --> rate of return
    # RoR 表示收益率矩阵。若原始价格有 T 行，则相邻两天之间只能得到 T-1 个收益率。
    RoR = np.zeros([data.shape[0]-1,data.shape[1]])   # Rate of return
    for i in range(data.shape[0]-1):
        # 单期收益率公式：(下一期价格 - 当前价格) / 当前价格。
        # 这样可以把不同价格尺度的股票统一转化为“涨跌比例”。
        RoR[i,:] = (data[i+1,:]-data[i,:])/data[i,:]

    # 期望收益向量 μ：对每一只股票的收益率按时间取平均。
    # 在均值-方差模型中，μ 越大，说明这只股票历史平均收益越高。
    exp_ret = pd.Series(np.mean(RoR, axis=0))

    # 协方差矩阵 Σ：描述股票收益率之间的共同波动情况。
    # 对角线元素是单只股票自己的方差，表示波动风险；
    # 非对角线元素表示两只股票是否倾向于同涨同跌。
    cov_mat = pd.DataFrame(np.cov(RoR, rowvar=False))



    return exp_ret, cov_mat


def calc_J():
    '''
    计算哈密顿量中 ZZ 项的系数矩阵 J。

    作用：
    在 QAOA 中，两只股票之间的相互影响会被编码成 Z_i Z_j 项。
    对应到量子电路中，Z_i Z_j 项由 RZZ 门实现，所以这里计算的是
    每一对 qubit 对应 RZZ 门的权重。
    '''
    # J[i, j] 表示第 i 个 qubit 和第 j 个 qubit 之间的 ZZ 耦合强度。
    J = np.zeros((num_qubits, num_qubits))

    # 只遍历 i < j，是因为 (i, j) 和 (j, i) 表示同一对 qubit，
    # 不需要重复添加两次 RZZ 门。
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            # half_q * cov_mat[i, j] / 2.0 来自风险项 x^TΣx 映射后的二次项；
            # eta / 2.0 来自预算约束惩罚项 (B - 1^T x)^2 映射后的二次项。
            # 二者相加得到 Z_i Z_j 前面的总系数。
            J[i, j] = half_q * cov_mat[i, j] / 2.0 + eta / 2.0

            # J 矩阵是对称的，因此同时填充 J[j, i]，方便后续读取。
            J[j, i] = J[i, j]



    return J

def calc_h():
    '''
    计算哈密顿量中 Z 项的系数向量 h。

    作用：
    每一只股票是否被选择，对应一个二进制变量 x_i；
    在量子模型中，x_i 会被映射为 Pauli-Z 算符。
    因此每只股票自身的收益、风险贡献和预算惩罚的一次项，
    最终会合并成一个 Z_i 项的系数 h_i。
    '''
    # h[i] 表示第 i 个 qubit 上 Z_i 项的权重，对应量子电路中的 RZ 门角度。
    h = np.zeros(num_qubits)

    # 对协方差矩阵按行求和，得到第 i 只股票与所有股票的总体风险联动。
    row_sum = np.sum(cov_mat, axis=1)
    for i in range(num_qubits):
        # 第一项：风险项映射后产生的一次项，风险越大，对目标函数越不利；
        # 第二项：收益项映射后产生的一次项，收益越高，对目标函数越有利；
        # 第三项：预算约束惩罚项映射后产生的一次项，用于推动选择数量接近 budget。
        h[i] = -half_q * row_sum[i] / 2.0 + theta1 * exp_ret[i] / 2.0 + eta * (budget - num_qubits / 2.0)

    return h

def insert_RX():
    '''
    构造 QAOA 的混合层 RX。

    作用：
    RX 门让每个 qubit 在 |0> 和 |1> 之间发生旋转，
    也就是让算法在“不选择某只股票”和“选择某只股票”之间继续搜索。
    如果没有混合层，量子态只会积累相位，测量概率很难向更优解转移。
    '''
    qc = QuantumCircuit(num_qubits)

    # beta 是 QAOA 的可训练参数之一，由后面的经典优化器负责寻找合适取值。
    beta = Parameter('beta_tmp')

    # 对每一个 qubit 都施加 RX(2β)，表示所有股票都允许在选/不选之间调整概率。
    for qubit in range(num_qubits):
        qc.rx(2 * beta, qubit)

    # barrier 只用于分隔电路层，便于阅读电路图，不改变计算结果。
    qc.barrier()


    return qc

def insert_RZ():
    '''
    构造损失哈密顿量中的单体 Z 项电路。

    作用：
    RZ 门用于实现 e^{-i γ h_i Z_i} 这一类演化。
    在本项目中，它编码的是单只股票自身的收益、风险一次项和预算惩罚一次项。
    '''
    qc = QuantumCircuit(num_qubits)

    # gamma 是 QAOA 的另一个可训练参数，用来控制损失层的演化强度。
    gamma = Parameter('gamma_tmp')

    # 重新计算 h 系数，h[i] 决定第 i 个 qubit 的 RZ 旋转角度。
    local_h = calc_h()

    # 对每一个 qubit 施加 RZ(2γh_i)。
    # 这里的 2 来自 Qiskit 中 RZ 门定义与指数演化形式之间的换算。
    for qubit in range(num_qubits):
        qc.rz(2 * gamma * local_h[qubit], qubit)
    qc.barrier()


    return qc

def insert_RZZ(gamma, J):
    '''
    构造损失哈密顿量中的二体 ZZ 项电路。

    作用：
    RZZ 门用于实现 e^{-i γ J_ij Z_i Z_j} 这一类演化。
    在投资组合问题中，ZZ 项主要表示两只股票之间的风险联动，
    以及预算惩罚项展开后产生的“两两选择数量”约束。

    注：
    (1, 3) 与 (3, 1) 是同一对 qubit，所以每两个 qubit 之间只添加一次 RZZ 门。
    '''
    qc = QuantumCircuit(num_qubits)

    # 遍历所有不同的 qubit 对，给每一对添加由 J[i, j] 决定角度的 RZZ 门。
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            # 如果系数为 0，说明这两个 qubit 之间没有有效耦合，可以跳过。
            if np.abs(J[i, j]) > 0:
                # RZZ(2γJ_ij) 对应哈密顿量中的 J_ij Z_i Z_j 项。
                qc.rzz(2 * gamma * J[i, j], i, j)


    qc.barrier()
    return qc

def insert_H():
    '''
    构造 QAOA 的初始叠加层。

    作用：
    Hadamard 门会把单个 qubit 从 |0> 变成 |0> 和 |1> 的均匀叠加。
    对所有 qubit 添加 H 门后，量子态会同时包含所有可能的股票选择方案。
    例如 6 个 qubit 会同时覆盖 2^6 = 64 种投资组合。
    '''
    qc = QuantumCircuit(num_qubits)

    # 每个 qubit 初始默认是 |0>，加 H 门后变成“选/不选”的均匀叠加。
    for qubit in range(num_qubits):
        qc.h(qubit)
    qc.barrier()



    return qc


# 接下来的两个函数用于拼接任务对应的哈密顿量。
# 在 Qiskit 中，哈密顿量可以用 Pauli 字符串加系数的形式表示，
# 例如 "ZIIZ" 表示第 0 和第 3 个 qubit 上有 Z 算符，其余位置是单位算符 I。
def get_Pauli(index, type):
    '''
    根据 qubit 下标生成 Pauli 字符串。

    作用：
    把数学表达式中的 Z_i 或 Z_i Z_j 转换成 Qiskit 可以识别的字符串形式。
    例如在 4 个 qubit 的系统中，index=[1], type='Z' 可以生成类似 "IZII" 的项；
    index=[0, 2], type='ZZ' 可以生成类似 "ZIZI" 的项。
    '''
    if type == 'Z':
        # 单体 Z 项必须只指定一个 qubit 下标。
        assert len(index) == 1
        index = index[0]

        # 确保 qubit 下标没有越界。
        assert index >= 0 and index <= num_qubits - 1

        # 先创建 num_qubits - 1 个单位算符 I，再在指定位置插入 Z。
        # I 表示这个位置上的 qubit 不参与该 Pauli 项。
        _Pauli = ['I'] * (num_qubits - 1)
        _Pauli.insert(index, 'Z')
        _Pauli = ''.join(_Pauli)
        return _Pauli
    elif type == 'ZZ':
        # 二体 ZZ 项必须指定两个 qubit 下标。
        assert len(index) == 2

        # 先创建 num_qubits - 2 个 I，再把两个 Z 插入指定位置。
        _Pauli = ['I'] * (num_qubits - 2)
        for i in range(len(index)):
            assert index[i] >= 0 and index[i] <= num_qubits - 1
            _Pauli.insert(index[i], 'Z')
        _Pauli = ''.join(_Pauli)
        return _Pauli
    else:
        # 当前程序只支持 Z 和 ZZ 两种 Pauli 项。
        raise AssertionError()

def problem_PauliOperator(h, J):
    '''
    构造完整的问题哈密顿量。

    作用：
    根据前面算出的 h 和 J，把投资组合优化问题写成
    H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j。
    这个 H_C 就是 QAOA 的 cost Hamiltonian，
    后续优化器会尝试最小化它的期望值。
    '''
    # 构造所有单体 Z 项。每一项由一个 Pauli 字符串和对应系数组成。
    Pauli_h_list = []
    for i in range(num_qubits):
        Pauli_h_list.append((get_Pauli([i], 'Z'), h[i]))

    # 将列表转换为 Qiskit 的 PauliSumOp 对象，表示 Σ h_i Z_i。
    Pauli_h = PauliSumOp.from_list(Pauli_h_list, coeff=1.0)

    # 构造所有二体 ZZ 项。每一项表示一对 qubit 之间的耦合关系。
    Pauli_J_list = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            Pauli_J_list.append((get_Pauli([i,j], 'ZZ'), J[i][j]))

    # 将列表转换为 PauliSumOp，表示 Σ J_ij Z_i Z_j。
    Pauli_J = PauliSumOp.from_list(Pauli_J_list, coeff=1.0)

    # 完整哈密顿量由 Z 项和 ZZ 项相加得到。
    Pauli_sum = Pauli_h + Pauli_J

    return Pauli_h, Pauli_J, Pauli_sum



def oneCircuit(h, J, beta, gamma):
    '''
    构造一层 QAOA 电路。

    作用：
    一层 QAOA 由两部分组成：
    1. 损失层 cost layer：根据哈密顿量 H_C 给不同投资组合积累不同相位；
    2. 混合层 mixer layer：通过 RX 门让概率在不同组合之间流动。

    多层 QAOA 就是把这个函数构造出的结构重复多次。
    '''
    # 创建一个只包含 num_qubits 个量子比特的新电路，用来表示单层 QAOA。
    qc = QuantumCircuit(num_qubits)

    # 损失层第一部分：实现所有单体 Z_i 项。
    # h[qubit] 是该 qubit 对应的哈密顿量系数，gamma 控制损失层整体强度。
    for qubit in range(num_qubits):
        qc.rz(2 * gamma * h[qubit], qubit)

    # 损失层第二部分：实现所有二体 Z_i Z_j 项。
    # J[i, j] 表示两只股票之间的联动风险和预算惩罚耦合。
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if np.abs(J[i, j]) > 0:
                qc.rzz(2 * gamma * J[i, j], i, j)
    qc.barrier()

    # 混合层：对所有 qubit 加 RX 门。
    # beta 控制混合强度，帮助算法从当前组合搜索到其他组合。
    for qubit in range(num_qubits):
        qc.rx(2 * beta, qubit)
    qc.barrier()


    return qc

def get_expectation(circuit, para_list, Hamiltonian):
    '''
    生成目标函数，用于经典优化器调用。

    作用：
    QAOA 的参数 beta 和 gamma 需要由经典优化器反复试探。
    这个函数返回一个内部函数 execute_circ(theta)，
    优化器每给出一组 theta，它就运行一次量子电路，
    计算当前量子态在哈密顿量 Hamiltonian 下的期望值，也就是 loss。

    优化目标：
    让 loss 尽可能小，对应找到哈密顿量能量最低的投资组合。
    '''

    def execute_circ(theta):
        # theta 为线路中所有可变参数，前一半为 beta，后一半为 gamma。
        qc = QuantumCircuit(num_qubits)

        # p 表示 QAOA 层数，也就是 beta 和 gamma 各自的参数个数。
        p = len(theta) // 2
        beta = theta[:p]
        gamma = theta[p:]

        # 构造参数字典，把符号参数 Parameter 映射到当前优化器给出的具体数值。
        para_dict = {}
        for i in range(p):
            para_dict[para_list[i]] = beta[i]
            para_dict[para_list[i+p]] = gamma[i]

        # 将已经搭好的参数化 QAOA 电路接到当前电路中。
        qc.append(circuit, [i for i in range(0, num_qubits)])

        # 把 beta/gamma 的具体数值代入量子电路。
        qc.assign_parameters(para_dict, inplace=True)

        # transpile 会把电路转换成模拟器更容易执行的形式。
        circ = transpile(qc, simulator)

        # 在 statevector 模拟器上运行电路，得到完整量子态。
        result = simulator.run(circ).result()
        _statevector = result.get_statevector(circ)

        # 利用当前状态向量计算 loss。
        # 这里的期望值 <ψ|H|ψ> 表示当前量子态对应的平均能量。
        loss = _statevector.expectation_value(Hamiltonian)

        # 理论上哈密顿量期望值应该是实数；这里检查虚部是否只是数值误差。
        assert np.imag(loss) < 1e-10
        return np.real(loss)

    return execute_circ

def str_to_statevector(string):
    '''
    将一个二进制字符串转换为对应的基态向量。

    作用：
    例如 "110101" 表示一个具体投资组合。
    为了计算该组合对应的哈密顿量能量，需要把字符串变成 one-hot 形式的状态向量。
    '''
    # Qiskit 输出 bitstring 的顺序与常见阅读顺序相反，因此这里先反转字符串。
    string = string[::-1]

    # 将二进制字符串转换为十进制下标。
    dec = int(string, 2)

    # 创建长度为 2^n 的零向量，其中 n 是 qubit 数。
    state = np.zeros(2 ** len(string))

    # 在该 bitstring 对应的位置填 1，表示一个确定的计算基态。
    state[dec] = 1.0
    return state[None,:]

def print_config():
    '''
    打印当前实验配置。

    作用：
    方便在程序输出中确认预算、编码粒度、惩罚系数和 QAOA 层数等关键参数。
    '''
    print('%%%%%%%%%%%%%%%%%%%% Configuration %%%%%%%%%%%%%%%%%%%%')
    print('budget: %d, g: %d, eta: %f, layers: %d' % (budget, num_slices, eta, layers))

def print_result(circuit, Hamiltonian, para_list, solution):
    '''
    根据优化得到的参数，输出最终量子态中的候选解及其概率。

    作用：
    训练结束后，把最优 beta/gamma 代入 QAOA 电路，
    得到最终 statevector，并计算每个投资组合的测量概率和能量值。
    能量最低的 bitstring 就是哈密顿量意义下的最优解。
    '''
    # 创建电路并准备代入已经优化好的参数。
    qc = QuantumCircuit(num_qubits)

    # solution 前半部分是 beta，后半部分是 gamma。
    p = len(solution) // 2
    beta = solution[:p]
    gamma = solution[p:]

    # 将符号参数与具体优化结果对应起来。
    para_dict = {}
    for i in range(p):
        para_dict[para_list[i]] = beta[i]
        para_dict[para_list[i + p]] = gamma[i]

    # 运行最终参数下的 QAOA 电路。
    qc.append(circuit, [i for i in range(0, num_qubits)])
    qc.assign_parameters(para_dict, inplace=True)
    circ = transpile(qc, simulator)
    result = simulator.run(circ).result()

    # 获取最终量子态。statevector 中每个基态的振幅平方就是测量概率。
    statevector = result.get_statevector(circ)  # innner product of statevector_dagger and statevector is 1
    statevector = statevector.to_dict()
    a = 0
    for i in statevector:
        # 将复振幅转换为概率：probability = |amplitude|^2。
        statevector[i] = np.abs(np.array(statevector[i])) ** 2
        a = a + statevector[i]
    # print('a: %f' % a)
    # print(statevector)

    # 按概率从高到低排序，方便观察最可能被测量到的组合。
    result = sorted(statevector.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # print(result)

    # 把每个 bitstring 转换成基态向量，用于直接计算该组合的哈密顿量能量。
    mm = []
    for i in range(len(result)):
        x, _ = result[i]
        mm.append(str_to_statevector(x))
    mm = np.concatenate(mm, axis=0)

    # 对每个确定的 bitstring 计算能量值 <x|H|x>。
    value_mm = np.sum((mm @ Hamiltonian) * mm, axis=1)

    # 能量最小的组合就是当前哈密顿量下的最优组合。
    min_index = np.argmin(value_mm)
    print("\nOptimal: selection {}, value {:.8f}".format(result[min_index][0][::-1], value_mm[min_index]))

    print("\n----------------- Full result ---------------------", flush=True)
    print("rank\tselection\tvalue\t\tprobability")
    print("---------------------------------------------------", flush=True)
    value_save = []
    probability_save = []
    utility_save = []
    for i in range(len(result)):
        x, probability = result[i]
        value = value_mm[i]
        assert np.imag(value) < 1e-10
        value = np.real(value)
        # value = portfolio.to_quadratic_program().objective.evaluate(x)
        # 输出时再次反转 bitstring，使其与通常的股票编号顺序一致。
        print("%d\t%-10s\t%.8f\t\t%.8f" % (i, x[::-1], value, probability), flush=True)
        # 保存实验结果文件，文件名包含预算、层数和 eta，便于区分不同实验。
        # 当前代码中 value_save/probability_save/utility_save 没有继续追加数据，
        # 因此保存的是预留结构，主要用于后续扩展。
        np.savez("./output/budget_{}_layers_{}_eta_{}.npz".format(budget, layers, eta), value=np.array(value_save), \
        probability=np.array(probability_save), utility=np.array(utility_save))

class callback:
    '''
    SPSA 优化器的回调记录器。

    作用：
    SPSA 每次迭代会调用该对象，把当前 loss 记录下来。
    后续 print_loss 可以利用这些记录展示训练过程中的 loss 变化。
    '''
    def __init__(self, step_size: int):
        # step_size 表示每隔多少次函数调用记录一次展示用 loss。
        self.step_size = step_size

        # full_values 保存所有 loss。
        self.full_values = []

        # _values 临时缓存最近 step_size 次 loss。
        self._values = []

        # values 保存用于打印的 loss 序列。
        self.values = []

    def __call__(self, nfev, parameters, value, stepsize, accepted):
        # 每次优化器调用 callback 时，value 就是当前参数下的 loss。
        self.full_values.append(value)
        self._values.append(value)
        if len(self._values) == self.step_size:
            last_value = self._values[-1]
            self.values.append(last_value)
            self._values = []
            return self.values

def print_loss(res):
    '''
    打印优化过程中的 loss 变化。

    作用：
    通过 loss 的变化可以观察优化器是否在逐步降低哈密顿量期望值。
    loss 越低，说明当前 QAOA 参数对应的量子态越接近低能量解。
    '''
    print('%%%%%%%%%%%%%%%%%%%% Optimization Output %%%%%%%%%%%%%%%%%%%%')
    loss_ls = callback_func.values
    print('minimal loss: %s, \nmaxIter: %d, func_eval: %d' % (res[1], len(callback_func.full_values), res[2]))
    print("Parameters Found:", res[0])
    print("\n----------------- Loss (%d steps from %d iterations) -----------------" % (len(loss_ls), len(callback_func.full_values)))
    print("iter\t\tloss")
    print("------------------------------------------------------------------------")
    for i in range(len(loss_ls)):
        loss = loss_ls[i]
        # value = portfolio.to_quadratic_program().objective.evaluate(x)
        print("%d\t\t%.10f" % (i, loss))


if __name__ == '__main__':
    # 主程序入口。
    # 整体流程：
    # 1. 读取命令行参数；
    # 2. 从股票数据中计算 μ 和 Σ；
    # 3. 构造问题哈密顿量 H_C；
    # 4. 搭建参数化 QAOA 电路；
    # 5. 用经典优化器寻找 beta/gamma；
    # 6. 输出最终投资组合及其概率。

    # 初始化参数
    parser = argparse.ArgumentParser()

    # budget 表示希望最终选择的股票数量。例如 budget=4 表示希望买入 4 只股票。
    parser.add_argument('--budget', type=int, default=4, help='Total assets.')

    # num_assets 表示候选股票数量。本实验默认 6 只股票。
    parser.add_argument('--num_assets', type=int, default=6, help='The number of assets.')

    # g 表示每只资产用多少个二进制位编码。
    # 当前默认 g=1，即每只股票只有买/不买两种状态。
    parser.add_argument('--g', type=int, default=1, help='The number of binary bits required to represent one asset.')

    # theta1 是收益项系数，控制收益在目标函数中的权重。
    parser.add_argument('--theta1', type=float, default=1.0, help='Coefficient of the linear term.')

    # half_q 是风险项系数的一部分，控制风险惩罚强度。
    parser.add_argument('--half_q', type=float, default=0.25, help='Coefficient of the quadratic term.')

    # eta 是预算约束惩罚系数。
    # eta 越大，选择数量偏离 budget 时受到的惩罚越强。
    parser.add_argument('--eta', type=float, default=1.0, help='Coefficient of the Lagrangian term.')

    # seed 固定随机数种子，方便复现实验结果。
    parser.add_argument('--seed', type=int, default=123456, help='Randon seed.')

    # optimizer 为 True 时使用 scipy 的 COBYLA；否则默认使用 Qiskit 的 SPSA。
    parser.add_argument('--optimizer', action='store_true', default=False, help='use scipy optimizer.')

    # maxiter 表示优化器最大迭代次数。
    parser.add_argument('--maxiter', type=int, default=300, help='max iterations.')
    parser.add_argument('--Gf', type=float, default=1.0, help='Granularity.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')

    # layers 表示 QAOA 层数。层数越多，电路表达能力越强，但优化也更困难。
    parser.add_argument('--layers', type=int, default=3, help='The number of QAOA layers.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    args = parser.parse_args()

    # 将命令行参数保存到全局变量中，供各个函数使用。
    budget = args.budget
    Gf = 1.0 / budget
    theta1 = 1.0
    half_q = 0.25
    eta = args.eta
    num_assets = args.num_assets
    num_slices = args.g  # The number of binary bits required to represent one asset
    layers = args.layers

    print_config()

    # 当前编码方式下，总 qubit 数 = 股票数量 × 每只股票使用的二进制位数。
    # 默认 num_assets=6, num_slices=1，因此使用 6 个 qubit。
    num_qubits = num_assets * num_slices

    # 固定随机种子，使优化器初始参数可复现。
    np.random.seed(args.seed)

    # 读取收益和方差
    file_path = "./data/stock_data.xlsx"
    exp_ret, cov_mat = data_preprocessing(file_path)

    # 将 pandas 对象转成 numpy 数组，便于后续下标访问和矩阵运算。
    exp_ret = exp_ret.to_numpy()
    cov_mat = cov_mat.to_numpy()

    # 计算所给问题对应的哈密尔顿量的系数
    J = calc_J()
    h = calc_h()

    # 计算所给问题对应的哈密尔顿量
    Pauli_h, Pauli_J, Pauli_sum = problem_PauliOperator(h, J)
    
    # 初始化量子虚拟机。
    # statevector 方法会保存完整量子态，适合本实验这种 qubit 数较少的仿真任务。
    simulator = AerSimulator(method = 'statevector')

    # 比特数较大时可以调整这里的参数进行并行计算，本次任务线路较小不需要修改。
    simulator.set_options(
        # default parameters
    max_parallel_threads = 0,
    max_parallel_experiments = 1,
    max_parallel_shots = 0,
    statevector_parallel_threshold = 14
)
    
    # 创建主量子电路，后续会把初始层和多层 QAOA 电路依次 append 进去。
    qc = QuantumCircuit(num_qubits)

    # 配置待优化参数
    beta = []
    gamma = []
    para_list = []
    for i in range(layers):
        # 每一层 QAOA 都有一个 beta 和一个 gamma。
        # beta 控制混合层 RX，gamma 控制损失层 RZ/RZZ。
        name = "β%d" % i
        beta.append(Parameter(name))
        name = "γ%d" % i
        gamma.append(Parameter(name))

    # para_list 统一保存所有参数，前半部分是 beta，后半部分是 gamma。
    para_list = beta + gamma

    # 构建QAOA
    # 第一步：加入 Hadamard 初始层，让所有投资组合处于均匀叠加中。
    qc.append(insert_H(), [i for i in range(0, num_qubits)])

    # 第二步：重复 layers 次，加入 QAOA 的损失层和混合层。
    for i in range(layers):
        qc.append(oneCircuit(h, J, beta[i], gamma[i]), [i for i in range(0, num_qubits)])

    # 保存最终 statevector，方便后续读取完整量子态。
    qc.save_statevector()
    print('\nCircuit Initialization Complete! Start Training...', flush=True)


    
    # 计算loss
    # expectation 是优化器实际最小化的目标函数。
    # 它接收一组 beta/gamma 参数，返回当前量子态的哈密顿量期望值。
    expectation = get_expectation(qc, para_list, Pauli_sum)

    # 优化参数
    start = time.time()
    if args.optimizer:
        # 利用外部优化器
        # COBYLA 是一种不需要梯度的经典优化算法，适合这类黑盒目标函数。
        res = minimize(expectation,
                       np.random.uniform(0, 0.001 * np.pi, size=layers * 2),
                       method='COBYLA',
                       options={'maxiter': args.maxiter})
        print('\nTraining Done! The output of optimizer: ')
        print(res)
        solution = res.x
    else:
        # 利用qiskit自带优化器
        # optimizer = COBYLA(maxiter=args.maxiter, tol=0.0001)
        # res = optimizer.optimize(num_vars=layers * 2, objective_function=expectation, initial_point=np.random.uniform(0, np.pi, size=layers * 2))
        step_size = 1 # 每隔step_size个iterations打印一次loss
        callback_func = callback(step_size)

        # SPSA 也是一种适合量子变分算法的无梯度优化器。
        # 它通过随机扰动估计参数更新方向，常用于含噪声或仿真开销较高的量子优化。
        optimizer = SPSA(maxiter=args.maxiter, blocking=True, second_order=True, callback=callback_func)

        # 初始参数取很小的随机值，长度为 layers * 2，因为每层有 beta 和 gamma 两个参数。
        res = optimizer.optimize(num_vars=layers * 2, objective_function=expectation, initial_point=np.random.uniform(0, 0.001*np.pi, size=layers * 2))
        solution = res[0]

        # 打印loss的变化
        print_loss(res)

    print("\nTraining done! Total elapsed time:{:.2f}s".format(time.time()-start))

    # 打印结果
    # 将最优参数代回电路，输出各投资组合的能量和测量概率。
    print_result(qc, Pauli_sum.to_matrix(), para_list, solution)
