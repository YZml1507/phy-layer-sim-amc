"""docs/demo.html 内嵌 JS 仿真的数值回归——demo 的 JS 链路和 src/phylayer 是两套
独立实现，本测试把 demo 的 <script> 数学段抽出来在 node 下实跑，断言其数值行为
与文档声称一致（纯 AWGN 贴理论、均衡收敛、CFO 毁链、EVM≈10^(-SNR/20)）。

本地无 node 时 skip；GitHub Actions ubuntu 镜像自带 node，CI 中真实执行。
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

DEMO = Path(__file__).parent.parent / "docs" / "demo.html"

DRIVER = r"""
function run(p){seed=20240901;return simulate(p);}
const out={};
for(const m of ["bpsk","qpsk","8psk","16qam","64qam"]){
  for(const s of [10,16]){
    const r=run({mod:m,snr:s,cfo:0,beta:0.35,mp:"none",dt:0,eq:false});
    const th=theoryBer(r.sch, s-10*Math.log10(r.sch.k));
    out["awgn_"+m+"_"+s]={meas:r.ber,theory:th,ser:r.ser,evm:r.evm,nBits:r.nE*r.sch.k};
  }
}
const off=run({mod:"qpsk",snr:16,cfo:0,beta:0.35,mp:"severe",dt:0,eq:false});
const on=run({mod:"qpsk",snr:16,cfo:0,beta:0.35,mp:"severe",dt:0,eq:true});
out.eq={off:off.ser,on:on.ser};
const cf=run({mod:"qpsk",snr:24,cfo:8e-3,beta:0.35,mp:"none",dt:0,eq:false});
out.cfo={ser:cf.ser,evm:cf.evm};
const d1=run({mod:"qpsk",snr:10,cfo:0,beta:0.35,mp:"mild",dt:0,eq:false});
const d2=run({mod:"qpsk",snr:10,cfo:0,beta:0.35,mp:"mild",dt:0,eq:false});
out.det={same:d1.ser===d2.ser&&d1.ber===d2.ber};
console.log("RESULTS:"+JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def demo_results():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    html = DEMO.read_text()
    m = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
    assert m, "demo.html 缺 <script> 块"
    math_js = m.group(1).split("// ---------- 主循环")[0]
    assert "function simulate" in math_js and "function theoryBer" in math_js, (
        "demo.html 数学段标记缺失——若重构过 demo 请同步更新本测试的切分标记"
    )
    tmp = Path(__file__).parent / "_demo_driver.js"
    tmp.write_text(math_js + DRIVER)
    try:
        proc = subprocess.run(
            [node, str(tmp)], capture_output=True, text=True, timeout=120,
            check=False,
        )
    finally:
        tmp.unlink()
    assert proc.returncode == 0, f"node 执行失败: {proc.stderr[:500]}"
    line = [x for x in proc.stdout.splitlines() if x.startswith("RESULTS:")]
    assert line, f"driver 无输出: {proc.stdout[:300]} {proc.stderr[:300]}"
    return json.loads(line[0][8:])


def test_awgn_ber_tracks_theory(demo_results):
    """纯 AWGN：实测 BER 与理论式同量级（固定种子，4096 符号）。"""
    for key, r in demo_results.items():
        if not key.startswith("awgn_"):
            continue
        # 蒙特卡洛分辨率：期望错误数 <5 时 meas=0 完全合法（分辨不出更小的 BER）
        exp_err = r["theory"] * r["nBits"]
        if exp_err < 5:
            assert r["meas"] < 5e-3, f"{key}: theory≈0 但 meas={r['meas']:.2e}"
        else:
            assert r["theory"] / 6 < r["meas"] < r["theory"] * 6, (
                f"{key}: meas={r['meas']:.3e} theory={r['theory']:.3e} 偏离>6x"
            )
        assert r["meas"] <= r["ser"] + 1e-12, f"{key}: BER>SER 违反格雷一致"


def test_evm_matches_snr(demo_results):
    """纯 AWGN 下 EVM ≈ 10^(-SNR/20)：10dB→~0.32，16dB→~0.16。"""
    for snr in (10, 16):
        evm = demo_results[f"awgn_qpsk_{snr}"]["evm"]
        assert abs(evm - 10 ** (-snr / 20)) < 0.05, f"snr={snr} evm={evm:.3f}"


def test_mmse_eq_recovers_severe_isi(demo_results):
    """severe 多径：均衡 SER 从 ~27% 收敛到 <0.5%。"""
    assert demo_results["eq"]["off"] > 0.1
    assert demo_results["eq"]["on"] < 0.005


def test_cfo_destroys_link(demo_results):
    """CFO 8e-3 cyc/sample（≈11.5°/sym）下 QPSK SER 接近随机。"""
    assert demo_results["cfo"]["ser"] > 0.3


def _python_chain(mod_name: str, esn0_db: float, n_sym: int = 4096, seed: int = 0):
    """Python 侧镜像 demo.html simulate()：调制→上采样→RRC→AWGN→匹配滤波→抽取→硬判决。

    噪声口径对照：JS `nVar=sigPow*sps/10^(snr/10)`（sigPow≈Es/sps）
    等价于 `awgn(x, snr_for_waveform(esn0, sps))` 的 σ²=sigPow/10^(esn0−10log10 sps)/10。
    """
    from phylayer.channels import awgn, snr_for_waveform
    from phylayer.metrics import ber, ser
    from phylayer.modulation import Modem
    from phylayer.pulse_shaping import fir_filter, rrc_taps, upsample

    order = {"bpsk": 2, "qpsk": 4, "16qam": 16, "64qam": 64}[mod_name]
    sps, span, beta = 4, 9, 0.35
    # JS 用 rrcTaps(beta,8,SPS) 允许偶数 span（33 抽头）；Python rrc_taps 要求
    # 奇数 span，取 9（36 抽头）。截断窗差一个符号周期，不影响 ISI-free 采样判决。
    rng = np.random.default_rng(seed)
    m = Modem(order)
    bits = rng.integers(0, 2, n_sym * m.bits_per_symbol)
    syms = m.modulate(bits)
    h = rrc_taps(beta, span, sps)
    x = fir_filter(upsample(syms, sps), h)
    y = awgn(x, snr_for_waveform(esn0_db, sps), rng)
    z = fir_filter(y, h)[len(h) - 1 :: sps][:n_sym]  # gd=len(h)-1，同 JS
    return ber(bits, m.demodulate(z)), ser(syms, m.nearest(z))


def test_js_python_parity(demo_results):
    """demo JS 与 src/phylayer 是两套独立实现：同场景 BER/SER 应对拍一致。

    期望错误数 ≥20 才断言（更低时分不清实现差异和蒙特卡洛抖动）；
    8PSK 不在 Python Modem 支持集（方形 QAM）内，不在此对拍。
    """
    for key, r in demo_results.items():
        if not key.startswith("awgn_"):
            continue
        name = key.split("_")[1]
        if name not in ("bpsk", "qpsk", "16qam", "64qam"):
            continue
        if r["theory"] * r["nBits"] < 20:
            continue
        esn0_db = float(key.split("_")[2])
        ber_py, ser_py = _python_chain(name, esn0_db)
        # 独立种子、相同 SNR/链路：BER 与 SER 量级应一致（容忍带 ~4x）
        for js_v, py_v, what in ((r["meas"], ber_py, "BER"), (r["ser"], ser_py, "SER")):
            lo, hi = min(js_v, py_v), max(js_v, py_v)
            assert lo > 0 and hi / lo < 4, (
                f"{key} {what} 对拍失败: js={js_v:.3e} python={py_v:.3e}"
            )


def test_deterministic_seed(demo_results):
    """同一 seed 两次 simulate 结果逐位一致（可复现性）。"""
    assert demo_results["det"]["same"]
