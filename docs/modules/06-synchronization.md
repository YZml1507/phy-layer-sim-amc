# 06 同步（定时 / 载波）

> 代码：`src/phylayer/sync.py`，组装在 `burst.py` 的 `rx_burst` 里
> 帧结构：`[短前导: ZC32 ×2 相同两段] [长前导: ZC64] [编码交织调制 payload]`

## 它解决什么问题

发射机发出去的每一个符号，接收机都要回答三个问题才知道它"是谁"：

1. **帧在哪？**（frame detection）——突发波形什么时候开始；
2. **每个符号在波形的哪个采样点上？**（symbol timing）——最佳采样相位；
3. **星座转了多少？**（CFO + 残余相位）——频偏把星座持续旋转。

这套接收机对标 802.11a 的两级前导思路：**短前导（重复结构）负责"粗略、
抗噪"的检测和 CFO；长前导（单条 ZC，无周期模糊）负责"精确"的细对齐和
信道估计**。为什么不只用一段重复前导？——重复结构相关峰在整数符号上存在
模糊（前半段也对得上），细对齐时会锁定到错误位置。

## 原理

### Zadoff-Chu 序列：同步前导的最佳选择

ZC 序列 $z[n] = e^{-j\pi r n^2 / N}$（LTE 主同步序列同款）的循环自相关是
理想冲激：

$$R_{zz}[\tau] = \sum_n z[n]\,z^*[n-\tau] = \begin{cases}N & \tau=0\\ 0 & \tau\neq0\end{cases}$$

意味着：把接收信号与本地 ZC 互相关，峰位置 = 前导到达时刻，**多径时每一径
都出一个峰**（峰间距 = 径延迟），而且恒定包络、频域平坦（做信道估计也合适）。

### 接收机流水线（`rx_burst`）

```text
波形 ──► frame_detect ──► locate_preamble ──► estimate_cfo ──► derotate
         (波形级粗检)       (τ×符号偏移网格)     (Moose 短前导)      (反旋转)
     ──► estimate_channel_ls ──► MMSE 均衡 ──► DD 相位跟踪 ──► LLR ──► Viterbi
         (长前导 LS)            (模块07)        (残余相位)
```

**第 1 步 frame_detect**：波形级互相关短前导取峰。多径让峰"胖"几样点，
无所谓——细同步在后面。

**第 2 步 locate_preamble**（本项目最花功夫的一步）：对 $\tau \in 0..sps-1$
个采样相位逐一抽取符号流、与长前导互相关，取 $|corr|$ 最大的相位；再从峰
向左扫到**连续簇的最左径**（允许 ≤2 个符号的凹陷）：

- 为什么取"首径"不是"最强径"？首径对应帧的真实起点；最强径是信道的
  $\arg\max_j |h_j|$，把它当起点会让整帧平移 $\arg\max$ 个符号——帧结构
  被破坏，payload 切错位置。
- 为什么允许凹陷？多径簇中相邻抽头之间相关值可能掉到阈值下，不能见坑就停。

**第 3 步 estimate_cfo**（Moose 算法）：短前导的两段相同 ZC32 在 CFO 下变成
$x_2[k] = x_1[k] e^{j2\pi\Delta f\cdot32\cdot sps}$，求互相关相角即得

$$\widehat{\Delta f} = \frac{\angle\left(\sum_k x_2[k]x_1^*[k]\right)}{2\pi\cdot32\cdot sps}$$

去模糊范围 $|\Delta f| < 1/(64\cdot sps)$——本项目 $8\times10^{-4}$ 远在范围内。

**第 4 步 derotate**：整帧乘 $e^{-j2\pi\hat f k}$ 反旋转。

**第 5 步 estimate_channel_ls**：在长前导上解 LS $y = Ph$（$P$ 为下三角
Toeplitz——帧前没有符号拖尾）。**注意参考必须用整个前导（短+长）**：长前导
的前面是短前导而非零，模型里要把短前导尾部的泄漏写进去，否则残差被当成噪声
导致 $\hat\sigma^2$ 高估一个数量级，MMSE 均衡直接退化（这是实测踩过的坑）。

**第 6 步 decision_directed_phase**：判决反馈 Costas——每符号判决到最近
星座点，残差相位 $\angle(z \cdot d^*)$ 以步长 $\alpha=0.15$ 累进跟踪。吃掉
CFO 估计残差 + 相位噪声这些"慢漂移"。

## 代码走读

- `zc_sequence(64)`：$e^{-j\pi r n^2/N}$，$r=25$。
- `frame_detect`：`np.convolve(rx, conj(pre[::-1]))` 取 argmax 再回退
  $4\cdot sps$ 个样点余量。
- `locate_preamble(z, long_pre, sps, gd, offset_syms=64)`：返回
  `(decimated_stream, d0)`。
- `estimate_cfo(rx_short, half_syms, sps)`：符号级 Moose。
- `estimate_channel_ls(pre_rx, pre_tx, n_taps)`：Toeplitz `lstsq`。
- `decision_directed_phase(syms, modem.nearest, alpha)`：逐符号相位锁相。

## 面试可能怎么问

- **为什么同步要分粗/细两级？** 粗同步要"弱信号也能检测到帧的存在"
  （波形级匹配滤波，采样点精度）；细同步要"每个符号采在最佳点"
  （符号相位精度）。一个模块同时满足两者成本高、鲁棒性差，所以分级。
- **为什么 LTE/5G 都用 ZC 序列做同步？** 理想循环自相关 + 恒定包络
  （频域也平坦，能同时当信道估计训练序列用），还天然区分多径。
- **CFO 估计的两段重复前导为什么好用？** 两段相同结构在 CFO 下的相位差
  正比于频偏和间距，一除就出来——这就是 Moose 方法，802.11a 的 CFO 估计
  原型。
- **符号定时偏差一点点会怎样？** 在匹配滤波输出上采错位置 → 收的不是
  Nyquist 无 ISI 点 → 星座图散开（信噪比等效损失）。
- **细同步找首径还是最强径？** 首径（帧起点语义）。最强径只是信道最强
  抽头，跟"帧从哪开始"不是一回事。
- **如果 CFO 大到超出模糊范围？** 需先用更短的重复结构（两段间隔更短，
  模糊范围更大）粗估，再细估——多级 CFO 估计。
