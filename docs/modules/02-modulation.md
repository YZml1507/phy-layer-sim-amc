# 02 数字调制与解调

> 代码：`src/phylayer/modulation.py`，核心类 `Modem`
> 实验：`experiments/exp01_constellations.py`、`exp02_ber_awgn.py`

## 它解决什么问题

比特是 0/1，信道里跑的是连续波形。调制就是把 $k$ 个比特映射成一个复数符号
$s = a_I + j a_Q$，让波形在一个符号周期 $T_s$ 里携带 $\log_2 M$ 个比特。

## 原理

### M-QAM = 两路独立的 PAM

方形 M-QAM 的星座图是 $\sqrt{M} \times \sqrt{M}$ 的网格。关键观察：**I 路和 Q 路各自独立做一次 $\sqrt{M}$-PAM**。所以 16QAM 就是"两路 4-PAM 正交叠加"——这带来两个工程红利：

1. 调制/解调可以按轴分别做，代码极简；
2. 软信息（LLR）计算也按轴分解，不用在 16 个点上做二维搜索。

每轴的 PAM 电平取奇对称电平 $\{-(m-1), -(m-3), \dots, (m-1)\}$，Gray 码标记。

### Gray 码为什么重要

判决出错时，大概率错到**相邻**星座点。Gray 映射保证相邻点只差 1 个比特，于是"符号错"≈"只错 1 bit"——这让 BER 最小化。代码里用经典的反射 Gray：

```python
gray_encode(n) = n ^ (n >> 1)      # 自然序号 -> 格雷码标签
gray_decode(g) = 前缀异或还原       # 格雷码标签 -> 自然序号
```

调制时：比特组 →（当作 Gray 标签）→ `gray_decode` → 电平序号 → 电平值。
解调时：找最近电平 → `gray_encode` → 比特组。方向别搞反。

### 能量归一化

M-QAM 的 I/Q 电平方差为 $E[a^2] = (m^2-1)/3$（$m=\sqrt{M}$）。符号能量
$E[|s|^2] = 2(m^2-1)/3$，所以乘缩放因子 $\sqrt{3/(2(M-1))}$ 使平均符号能量为 1。

### 硬判决与软判决（LLR）

- **硬判决**：每个轴找最近的 PAM 电平 → Gray 编码 → 比特。
- **软判决（LLR）**：给译码器"这个比特是 0 还是 1、有多大把握"：

$$
\mathrm{LLR}(b_i)=\log\frac{\sum_{s:b_i=0}\exp\!\left(-\dfrac{|r-s|^2}{\sigma^2}\right)}{\sum_{s:b_i=1}\exp\!\left(-\dfrac{|r-s|^2}{\sigma^2}\right)}
$$

项目里用 `scipy.special.logsumexp` 做数值稳定的求和，且利用轴分解只在
$\sqrt{M}$ 个电平上求和。LLR 送 Viterbi 软译码能多赚约 2 dB（见模块 04）。

### 理论 BER

方形 M-QAM 在 AWGN 下的近似 BER（QPSK 为精确式）：

$$
P_b \approx \frac{4}{\log_2 M}\left(1-\frac{1}{\sqrt{M}}\right)
Q\!\left(\sqrt{\frac{3\log_2 M}{M-1}\cdot\frac{E_b}{N_0}}\right)
$$

`Modem.theoretical_ber_awgn()` 实现该式，`exp02` 用它做基准线。

## 代码走读（按调用顺序）

```python
m = Modem(16)                # bits_per_symbol=4, PAM 电平 {-3,-1,1,3}
syms = m.modulate(bits)      # bits.reshape(-1,4) -> I 2bit + Q 2bit -> 电平
rx   = awgn(syms, snr_db)    # 见模块05：sigma^2 = E|s|^2 / 10^(snr/10)
bits_hat = m.demodulate(rx)  # 每轴最近电平 -> gray_encode -> 比特
llr  = m.demodulate_llr(rx, sigma2)   # 软判决，给 Viterbi 用
```

## 面试可能怎么问

- **为什么 QPSK 和 BPSK 的 BER 曲线重合？** 都按每比特看：QPSK 每轴就是一路
  BPSK，I/Q 噪声互不串扰。仿真图里两条线确实重合。
- **16QAM 和 QPSK 同样 Eb/N0 下谁 BER 高？为什么？** 16QAM 高——星座点更密、
  最小欧氏距离更小（$d_{min}^2 = 4/(M-1)$ 量级），抗噪声余量小。代价换来频谱效率翻倍。
- **调制前为什么要 Gray 映射？** 见上文"Gray 码为什么重要"。
- **软判决比硬判决好多少？** 卷积码场景约 2 dB 编码增益差。
- **OQPSK、MSK 听过吗？** 恒包络/相位连续的变体，功放开环非线性场景用——本
  项目用线性调制，可提一句作为扩展方向。
