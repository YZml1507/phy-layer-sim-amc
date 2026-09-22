# PHY-Layer-Sim-AMC 讲解文档

用 Python 从零实现一条完整的数字通信物理层收发链路（单载波 + OFDM），
并在链路产生的 I/Q 数据上训练深度网络做自动调制识别（AMC）。

## 这个网站怎么用

- **模块讲解**：按链路顺序逐页讲解，每页结构相同——
  「它解决什么问题 → 原理（含推导）→ 代码走读 → 面试可能怎么问」。
- **[交互演示](demo.html)**：浏览器里实时跑链路仿真，拖滑块看星座图、
  眼图、BER 曲线、多径 ISI 和 MMSE 均衡效果。
- **面试手册**：预判追问与应答、简历条目、5 分钟 PPT 大纲。
- 代码仓库：[github.com/YZml1507/phy-layer-sim-amc](https://github.com/YZml1507/phy-layer-sim-amc)

## 链路总览

```text
信源比特 → 信道编码(卷积码) → 交织 → 数字调制(BPSK/QPSK/16QAM/64QAM)
  → 成型滤波(根升余弦 RRC) → 上采样
  → 信道(AWGN / 多径瑞利 / 载波频偏 / 相位噪声)
  → 接收机: 短前导粗检 → 匹配滤波 → 长前导细同步(相位×符号网格+首径)
  → Moose CFO → LS信道估计 → MMSE均衡 → Costas相位跟踪 → LLR → Viterbi
  → 输出: BER-SNR 曲线 / 星座图 / 眼图 / EVM

OFDM 分支: 比特 → 调制 → 子载波映射+导频 → IFFT → 加CP → 信道
  → 去CP → Schmidl-Cox 定时 → CFO 估计补偿 → FFT
  → LS/DFT/MMSE 信道估计 → 均衡 → 解调 → BER

AMC 支线: 链路批量生成带标注 I/Q 样本(调制类型 × SNR)
  → CNN/ResNet1D 训练 → 准确率-SNR 曲线 + 混淆矩阵
```
