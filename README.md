# phy-layer-sim-amc

纯 Python(NumPy/SciPy)实现的**数字通信物理层全链路仿真平台**，外加一个
**深度学习自动调制识别(AMC)** 模块。覆盖：信道编码、数字调制、脉冲成型、
同步、均衡、OFDM(5G NR 风格参数)、信道估计，以及基于 CNN/ResNet 的 I/Q
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
src/phylayer/    物理层仿真库(纯 NumPy/SciPy)
  modulation.py    BPSK/QPSK/16QAM/64QAM Gray 映射调制解调
  pulse_shaping.py 根升余弦(RRC)成型与匹配滤波
  coding.py        卷积码编码 + Viterbi 译码(硬/软判决)
  channels.py      AWGN、多径衰落、载波频偏、相位噪声
  sync.py          前导辅助帧定时、CFO 估计、Costas 相位跟踪
  equalizer.py     ZF/MMSE 均衡
  ofdm.py          OFDM 收发机(子载波映射、CP、导频、Schmidl-Cox 定时)
  chan_est.py      LS / DFT 插值 / LMMSE 信道估计
  metrics.py       BER / SER / EVM
  iqdata.py        带标注 I/Q 数据集生成(供 AMC)
amc/             深度学习调制识别(PyTorch)
experiments/     可复现实验脚本，输出全部结果图
docs/            逐模块中文讲解文档(MkDocs)
tests/           与理论值对齐的数值验证测试
notebooks/       Colab / Kaggle 一键全量训练
```
