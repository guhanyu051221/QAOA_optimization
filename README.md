# QAOA Portfolio Optimization

本项目使用 QAOA 算法求解 6 只股票的均值-方差投资组合优化问题。核心脚本是 `qaoa_qiskit.py`，`usable.py` 用于批量更改运算参数并绘图，辅助脚本 `submit_originq.py` 用于导出可提交到量子云平台的线路文件。

## 从打开环境到运行脚本

以下命令适用于 Windows PowerShell。

### 1. 打开 PowerShell 并进入项目目录

如果当前在仓库根目录 `D:\2B\GKC2\QAOA\QAOA`，执行：

```powershell
cd .\QAOA_optimization
```

如果从其他位置打开 PowerShell，可以直接进入完整路径：

```powershell
cd D:\2B\GKC2\QAOA\QAOA\QAOA_optimization
```

### 2. 创建 conda 环境

第一次运行项目前需要创建环境。已经创建过 `fer2` 环境的话，可以跳过这一步。

```powershell
conda create -n fer2 python=3.8
```

### 3. 激活环境

每次重新打开 PowerShell 后，都需要先激活环境：

```powershell
conda activate fer2
```

### 4. 安装依赖

第一次运行项目前安装依赖。已经安装过的话，可以跳过这一步。

```powershell
pip install -r .\requirements.txt
```

如果需要重新绘图或本地环境缺少 `matplotlib`，额外执行：

```powershell
pip install matplotlib
```

如果安装时遇到 SSL 或网络源连接问题，可以尝试：

```powershell
pip install matplotlib --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org
```

### 5. 运行 QAOA 主脚本

推荐用于答辩展示的代表性运行命令：

```powershell
python .\qaoa_qiskit.py --budget 4 --layers 4 --eta 6 --maxiter 200
```

主脚本常用参数含义：

```text
--budget   希望选择的股票数量，本项目通常为 4
--layers   QAOA 层数，层数越高线路表达能力越强，但运行更慢
--eta      预算约束惩罚系数
--maxiter  SPSA 优化器最大迭代次数
```

### 6. 导出量子云平台线路文件

如果需要生成 OriginQ/量子云平台可用的线路文件，运行：

```powershell
python .\submit_originq.py --budget 4 --layers 4 --eta 6 --shots 1000
```

运行后会在 `output\originq` 目录下生成：

```text
qaoa_fixed_params.qasm
qaoa_fixed_params.originir
ideal_probabilities.csv
submission_notes.txt
```

其中 `qaoa_fixed_params.originir` 可以用于 OriginIR 编辑器，`ideal_probabilities.csv` 用来和真实量子云平台采样结果对比。

### 7. 批量扫描参数并绘图

如果需要自动改变不同运算参数并生成图像，运行：

```powershell
python .\usable.py
```

`usable.py` 会自动完成三组实验。每一个参数点会重复运行 10 次，并统计均值和样本方差：

```text
1. 扫描 eta：eta = 0, 2, 4, 6, 8, 10
2. 扫描 QAOA 层数：layers = 1, 2, 3, 4, 5, 6
3. 扫描最大迭代次数：maxiter = 50, 100, 200, 300, 500
```

每组实验都会记录：

```text
min_loss_mean / min_loss_var   最小 loss 的均值和方差
value_mean / value_var         最优 bitstring 能量的均值和方差
prob_mean / prob_var           最优 bitstring 概率的均值和方差
result_mode                    10 次运行中出现最多的最优 bitstring
result_mode_count              result_mode 出现次数
```

运行结束后，结果会保存到 `figures` 目录，包括：

```text
eta_scan_raw.csv
eta_scan_summary.csv
layer_scan_raw.csv
layer_scan_summary.csv
maxiter_scan_raw.csv
maxiter_scan_summary.csv
parameter_scan_report.txt
eta_vs_min_loss_mean.png
eta_vs_value_mean.png
eta_vs_prob_mean.png
eta_vs_result_code_mean.png
layer_vs_min_loss_mean.png
layer_vs_value_mean.png
layer_vs_prob_mean.png
layer_vs_result_code_mean.png
maxiter_vs_min_loss_mean.png
maxiter_vs_value_mean.png
maxiter_vs_prob_mean.png
maxiter_vs_result_code_mean.png
maxiter_vs_elapsed_s_mean.png
eta_sweep_layers4_maxiter250_mean.png
layers_sweep_eta6_maxiter250_mean.png
maxiter_sweep_layers4_eta6_mean.png
```

图像默认使用 10 次运行的均值作图，误差棒表示标准差。

注意：`usable.py` 会连续运行多次 QAOA 训练，耗时明显长于单次运行 `qaoa_qiskit.py`。

### 8. 单独扫描 maxiter

如果只需要分析 `maxiter` 对 probability、训练时间、value 和 min_loss 的影响，不想重新运行完整参数扫描，可以执行：

```powershell
python .\usable_maxiter.py
```

该脚本只扫描：

```text
maxiter = 50, 100, 200, 300, 500
```

每个 `maxiter` 重复运行 10 次，统计各指标均值和方差，并生成：

```text
figures/maxiter_scan_raw.csv
figures/maxiter_scan_summary.csv
figures/maxiter_scan_report.txt
figures/maxiter_vs_prob_mean.png
figures/maxiter_vs_elapsed_s_mean.png
figures/maxiter_vs_value_mean.png
figures/maxiter_vs_min_loss_mean.png
figures/maxiter_vs_result_code_mean.png
```

### 9. 运行更现实的 QUBO 松弛变量模型

实验报告第七部分讨论了带资金不等式约束的多股持仓模型。对应示例脚本为：

```powershell
python .\qubo_slack_demo.py
```

该脚本会枚举一个小规模 QUBO 示例，其中：

```text
y_i      第 i 只股票买入股数
p_i      第 i 只股票价格
C        总资金上限
s        松弛变量，表示未使用资金
```

默认输出：

```text
output/qubo_slack_results.csv
```

也可以调整资金、价格和最大持仓：

```powershell
python .\qubo_slack_demo.py --capital 10 --prices 2,3,2,4,3,2 --max-shares 2,2,2,2,2,2 --eta 6 --top-k 10
```

## 项目结构

```text
qaoa_qiskit.py          QAOA 主程序，负责单次训练并输出最优组合
usable.py               批量扫描 eta、layers、maxiter，并生成 CSV 与图像
usable_maxiter.py       单独扫描 maxiter 对 probability、时间、value 和 loss 的影响
qubo_slack_demo.py      演示第七部分带松弛变量 s 的多股持仓 QUBO 模型
submit_originq.py       导出固定参数线路到 OpenQASM / OriginIR
requirements.txt        Python 依赖列表
data/stock_data.xlsx    股票收盘价数据
output/                 运行结果、图像和导出线路
figures/                usable.py 生成的参数扫描图像和详细结果
实验报告2.md            实验报告 Markdown 版本
答辩稿.md               中文答辩稿
```

## 输出说明

- `output/`：早期单次运行或手动扫描保存的结果、图像和中间文件。
- `output/originq/`：量子云平台提交相关文件。
- `figures/*_raw.csv`：`usable.py` 每个参数点重复 10 次的逐次运行结果。
- `figures/*_summary.csv`：`usable.py` 统计出的均值、方差、最常见最优组合及出现次数。
- `figures/*_mean.png`：使用 10 次运行均值作图的参数关系图，误差棒表示标准差。
- `figures/maxiter_vs_elapsed_s_mean.png`：`maxiter` 与平均训练时间的关系图。
- `figures/parameter_scan_report.txt`：自动生成的参数扫描综合报告。
