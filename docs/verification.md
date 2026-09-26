# 附录：每条数字怎么复现

文档里所有定量声称都能用一条命令复核。被问"这个数字哪来的"时，这张表就是答案——
大部分声称已固化成 CI 测试（`pytest tests/test_theory_bounds.py`），其余用一行
Python 或实验脚本即可复现。

## 已被 CI 测试固化的声称

| 声称 | 文档位置 | 复核方式 |
|---|---|---|
| Moose CFO 估计方差达渐近界 `std(f̂)=1/(2π·L·sps·√(L·Es/N0))`（3.5e-5 cyc/sample ≈ 0.10°/sym，实测/界≈1.02） | 06、手册深挖预案 | `pytest -k test_cfo_estimator_reaches_bound` |
| LS 信道估计每抽头 MSE ≈ σ²/Ntr（均值 1.0~1.05） | 09 | `pytest -k test_ls_estimator_per_tap_mse` |
| severe 信道 ZF 噪声增强 ×6.6 vs MMSE ×3.9 | 07 | `pytest -k test_mmse_beats_zf` |
| 16QAM 近似式与精确式比值=1.000（4~16dB） | 02 | `pytest -k test_16qam_approx` |
| 171/133 卷积码 d_free=10 | 04 | `pytest -k test_conv_code_dfree`（短输入全枚举） |
| OFDM ICI 闭式 `P_ICI=1−|s_0|²≈(πε)²/3`（ε=0.192→−9.4dB） | 08 | `pytest -k test_ofdm_ici_closed_form` |
| 相干带宽 B_c≈1/(2πσ_τ) ≫ 载波间隔 1/64（5~9 倍） | 05 | `pytest -k test_coherence_bandwidth` |
| Ricean K 深衰落概率：K=0→9.5%、K=3→2.8%、K=10→0.1% | 05 | `pytest -k test_rician_k` |
| 自然二进制相邻电平错位均值 4/3、11/7（Gray=1 → BER +33%/+57%） | 02 | `pytest -k test_natural_binary_neighbor` |

## 已有功能测试覆盖的声称

| 声称 | 复核方式 |
|---|---|
| RRC 收发级联=升余弦、抽样点 ISI<1e-4 | `pytest -k test_rrc_cascade_is_nyquist` |
| 各调制仿真 BER 压理论曲线 | `pytest -k test_ber_matches_theory`（蒙特卡洛对拍） |
| 软判决 Viterbi 优于硬判决（工作区 ~2dB） | `pytest -k test_soft_beats_hard` / `test_coding_gain` |
| 成型链路 BER 与理论一致 | `pytest -k test_shaped_ber` |
| 突发链路多径+CFO 低误码 | `pytest -k test_burst_multipath_cfo` |
| OFDM 干净信道回环零误符号 | `pytest -k test_loopback_clean` |

## 一句话命令复核的声称

| 声称 | 复核命令 |
|---|---|
| AMCNet 总参数 ~33 万 | `python3 -c "from phylayer.amc import AMCNet; print(sum(p.numel() for p in AMCNet(5).parameters()))"` → 332165 |
| QPSK+RRC(β=0.35) 谱效率 ~1.48 bit/s/Hz | 直接算：2 bit/sym ÷ 1.35（每符号带宽 (1+β)/Ts，符号速率 1/Ts） |
| CFO 每符号转角 2.3°（8e-4 cyc/sample, sps=8） | 2π×8e-4×8 rad = 0.0402 rad ≈ 2.3° |
| 匹配滤波输出符号 SNR = Es/N0 | 03 文档"定量结论"段推导 + `test_shaped_ber_matches_theory` 间接验证 |

## 实验脚本级声称（分钟级复现）

| 声称 | 复核方式 |
|---|---|
| AWGN 下仿真 BER 逐点压理论线 | `python3 experiments/exp02_ber_awgn.py` → `docs/assets/ber_awgn.png` |
| 突发链路编码增益曲线 | `python3 experiments/exp03_burst_link.py` → `docs/assets/burst_ber.png` |
| OFDM vs 单载波：低 SNR 占优、高 SNR 残留 ~1% | `python3 experiments/exp04_ofdm_link.py` → `docs/assets/ofdm_ber.png` |
| AMC 混合 SNR ~76% / 高 SNR ~85%（CPU 18 epoch） | `python3 experiments/exp05_amc.py` → `docs/assets/amc_results.png` |
| AMC Kaggle T4 全量 76.9% / ~88% | `notebooks/amc_kaggle.ipynb`（Kaggle → GPU T4 → Run all） |
| 交互演示页端到端行为 | 浏览器打开文档站"交互演示"页；无头截图命令见 demo.html 注释 |
