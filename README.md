# phy-layer-sim-amc

纯 Python(NumPy/SciPy)实现的**数字通信物理层全链路仿真平台**，外加一个
**深度学习自动调制识别(AMC)** 模块。覆盖：信道编码、数字调制、脉冲成型、
同步、均衡、OFDM(802.11a 风格帧结构)、信道估计，以及基于 1D-ResNet 的 I/Q
调制分类。

讲解文档站(逐模块原理 → 代码走读 → 面试问答)：<https://yzml1507.github.io/phy-layer-sim-amc/>

## 快速开始

```bash
pip install -e ".[dev]"          # 物理层仿真
pip install -e ".[dev,amc]"      # 另装 PyTorch(AMC 模块需要)
pytest -q                        # 跑测试(含与理论 BER 曲线的数值验证)
python experiments/exp02_ber_awgn.py   # 生成 BER-SNR 曲线图
```

## 目录结构

```
src/phylayer/      物理层仿真库(纯 NumPy/SciPy + torch 仅用于 amc.py)
  modulation.py    BPSK/QPSK/16QAM/64QAM Gray 映射调制解调(含软判决 LLR)
  pulse_shaping.py 根升余弦(RRC)成型与匹配滤波
  coding.py        (2,1,7) 卷积码 + 软/硬判决 Viterbi、比特交织器
  channels.py      AWGN、瑞利/莱斯多径、载波频偏、相位噪声、采样延迟
  sync.py          ZC 前导生成、两段式帧结构、相关定帧、Moose CFO、LS 信道估计
  burst.py         单载波突发发射/接收全流程(同步→均衡→软解调→译码)
  equalizer.py     ZF/MMSE 时域均衡器(全延迟扫描)
  ofdm.py          OFDM 收发机(S&C+匹配滤波定时、CFO、导频 CPE、信道估计)
  iqdata.py        带标注 I/Q 数据集生成(受损信道链,供 AMC)
  amc.py           1D-ResNet 调制分类(PyTorch,可选依赖)
  metrics.py       BER / SER
experiments/       可复现实验脚本,结果图输出到 docs/assets/
  exp01 星座图与 EVM       exp02 AWGN BER vs 理论
  exp03 单载波突发链路      exp04 OFDM 链路对比
  exp05 AMC 训练与评估
docs/              逐模块中文讲解文档(MkDocs Material)
  modules/         01-总览 ... 10-AMC
  interview/       面试追问手册 + 简历/PPT 素材
tests/             与理论值对齐的数值验证测试(56 项, 含理论界回归)
notebooks/         Kaggle 一键训练 notebook(amc_kaggle.ipynb)
```

## 两条链路

**单载波突发链路**：比特 → (2,1,7)卷积编码 → 交织 → QPSK/16QAM → RRC
成型 → `[ZC短前导×2 | ZC长前导 | payload]` → 信道 → 帧检测 → 匹配滤波 →
精同步 → Moose CFO → LS 信道估计 → MMSE 均衡 → 判决引导相位跟踪 →
软解调 → Viterbi 译码。

**OFDM 链路**：比特 → 调制 → 58 数据载波 + 4 导频映射 → IFFT → CP →
`[S&C 前导 | 全载波训练符号 | 数据符号]` → 信道 → 匹配滤波定帧 →
CFO 估计 → 训练符号 LS 信道估计(IFFT 截断降噪) → 逐符号导频 CPE 校正 →
逐载波均衡。

**AMC**：`iqdata` 受损信道链数据工厂 → 1D-ResNet → 5 类调制识别，
输出混淆矩阵与准确率-SNR 曲线。
