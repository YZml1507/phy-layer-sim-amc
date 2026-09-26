"""生成复试 5 分钟陈述 PPT（phy-layer-sim-amc 项目）。

用法: python3 tools/build_ppt.py  → 输出 docs/interview/phy-layer-sim-amc-复试陈述.pptx

设计：16:9 白底学术风，8 页对应 docs/interview/resume-ppt.md 的逐页口播稿
（P1–P7 + 一页备用数字页）。图中所有数字以 verification.md 为准——改数字
请同步改口播稿与速查卡。
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
OUT = ROOT / "docs" / "interview" / "phy-layer-sim-amc-复试陈述.pptx"

NAVY = RGBColor(0x1B, 0x2A, 0x4A)
ACCENT = RGBColor(0x2E, 0x6F, 0xB7)
GRAY = RGBColor(0x55, 0x5F, 0x6E)
LIGHT = RGBColor(0xEE, 0xF2, 0xF7)
FONT = "Microsoft YaHei"

SW, SH = Inches(13.333), Inches(7.5)
prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def box(slide, l, t, w, h, fill=None, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sh.shadow.inherit = False
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    return sh


def text(slide, l, t, w, h, runs, size=18, color=GRAY, bold=False,
         align=PP_ALIGN.LEFT, leading=1.12):
    """runs: str 或 [(text, {bold,color,size}), ...] 的段落列表。"""
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    if isinstance(runs, str):
        runs = [runs]
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = leading
        p.space_after = Pt(6)
        if isinstance(para, str):
            para = [(para, {})]
        for txt, st in para:
            r = p.add_run()
            r.text = txt
            r.font.name = FONT
            r.font.size = Pt(st.get("size", size))
            r.font.bold = st.get("bold", bold)
            r.font.color.rgb = st.get("color", color)
    return tb


def header(slide, title, subtitle=None, num=None):
    box(slide, 0, 0, SW, Inches(0.14), fill=ACCENT)
    text(slide, Inches(0.6), Inches(0.35), Inches(11.5), Inches(0.7),
         title, size=30, color=NAVY, bold=True)
    if subtitle:
        text(slide, Inches(0.62), Inches(1.05), Inches(12), Inches(0.45),
             subtitle, size=15, color=GRAY)
    if num:
        text(slide, SW - Inches(0.9), SH - Inches(0.5), Inches(0.6),
             Inches(0.4), str(num), size=12, color=GRAY, align=PP_ALIGN.RIGHT)


def pic(slide, path, l, t, w=None, h=None, border=True):
    p = slide.shapes.add_picture(str(path), l, t, width=w, height=h)
    if border:
        p.line.color.rgb = RGBColor(0xD5, 0xDC, 0xE4)
        p.line.width = Pt(0.75)
    return p


def pic_fit(slide, path, l, t, max_w, max_h, border=True):
    """等比缩放图片放进 max_w×max_h 区域并居中。"""
    from PIL import Image
    iw, ih = Image.open(path).size
    s = min(max_w / iw, max_h / ih)
    w, h = Emu(int(iw * s)), Emu(int(ih * s))
    return pic(slide, path, l + Emu(int((max_w - w) / 2)),
               t + Emu(int((max_h - h) / 2)), w=w, h=h, border=border)


# ---------------- S1 封面 ----------------
s = prs.slides.add_slide(BLANK)
box(s, 0, 0, SW, SH, fill=NAVY)
box(s, 0, Inches(4.9), SW, Pt(2.2), fill=ACCENT)
text(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(1.2),
     "数字通信物理层仿真平台 + 智能调制识别", size=40,
     color=RGBColor(0xFF, 0xFF, 0xFF), bold=True)
text(s, Inches(0.92), Inches(3.35), Inches(11.5), Inches(1.0),
     "从比特到比特：编码·调制·成型·同步·均衡·OFDM 全链路 NumPy 手写实现，"
     "链路产出数据训练 1D-ResNet 自动调制分类", size=17,
     color=RGBColor(0xBF, 0xD3, 0xEA))
text(s, Inches(0.92), Inches(5.3), Inches(11.5), Inches(0.6),
     "汇报人：XXX    报考：信息与通信工程", size=15,
     color=RGBColor(0x8F, 0xA8, 0xC8))

# ---------------- S2 架构 ----------------
s = prs.slides.add_slide(BLANK)
header(s, "一条链路，每一级撤销一种信道损伤", "代码按“一级一个问题”组织——每个模块都能单独展开讲", 2)

tx = ["比特源", "卷积码 171/133", "交织", "调制 QPSK/16QAM", "RRC 成型 β=0.35"]
ch = ["多径衰落", "载波频偏 CFO", "AWGN", "定时偏差"]
rx = ["帧同步(相关)", "Moose CFO", "LS 信道估计", "MMSE 均衡", "相位跟踪", "Viterbi 译码"]

y0, bw, bh, gap = Inches(2.15), Inches(1.72), Inches(0.62), Inches(0.12)
for i, t in enumerate(tx):
    b = box(s, Inches(0.55) + i * (bw + gap), y0, bw, bh, fill=LIGHT)
    text(s, Inches(0.55) + i * (bw + gap), y0 + Inches(0.08), bw, bh,
         t, size=12.5, color=NAVY, align=PP_ALIGN.CENTER)
text(s, Inches(0.55), y0 - Inches(0.4), Inches(2), Inches(0.35), "发射端",
     size=14, color=ACCENT, bold=True)

y1 = y0 + bh + Inches(0.28)
for i, t in enumerate(ch):
    b = box(s, Inches(1.4) + i * (bw + gap + Inches(0.35)), y1, bw + Inches(0.3), bh,
            fill=RGBColor(0xFB, 0xEA, 0xDA))
    text(s, Inches(1.4) + i * (bw + gap + Inches(0.35)), y1 + Inches(0.08),
         bw + Inches(0.3), bh, t, size=12.5, color=RGBColor(0x9A, 0x4A, 0x1E),
         align=PP_ALIGN.CENTER)
text(s, Inches(0.55), y1 + Inches(0.1), Inches(0.8), Inches(0.4), "信道",
     size=14, color=RGBColor(0x9A, 0x4A, 0x1E), bold=True)

y2 = y1 + bh + Inches(0.28)
bw2 = Inches(1.55)
gap2 = Inches(0.14)
for i, t in enumerate(rx):
    b = box(s, Inches(0.55) + i * (bw2 + gap2), y2, bw2, bh,
            fill=RGBColor(0xE3, 0xEE, 0xFA))
    text(s, Inches(0.55) + i * (bw2 + gap2), y2 + Inches(0.06), bw2, bh,
         t, size=11.5, color=ACCENT, align=PP_ALIGN.CENTER)
text(s, Inches(0.55), y2 - Inches(0.4), Inches(2), Inches(0.35), "接收端",
     size=14, color=ACCENT, bold=True)

text(s, Inches(0.55), Inches(5.15), Inches(12.2), Inches(1.9), [
    [("OFDM 支线：", {"bold": True, "color": NAVY}),
     ("N_fft=64 / CP=16 / 导频 {6,18,42,54}——同一批损伤在频域对付："
      "Schmidl-Cox 定时 → CFO → 逐载波信道估计 → CPE 相位跟踪", {})],
    [("AMC 支线：", {"bold": True, "color": NAVY}),
     ("链路反过来当数据工厂，生成带标注 I/Q 样本训练 1D-ResNet 识别调制方式", {})],
    [("面试点：", {"bold": True, "color": RGBColor(0x9A, 0x4A, 0x1E)}),
     ("接收端每一级对应课本一章——同步(06)、信道估计(09)、均衡(07)、译码(04)", {})],
], size=15)

# ---------------- S3 接收机深水区 ----------------
s = prs.slides.add_slide(BLANK)
header(s, "接收机流水线：四个真实的坑", "突发链路：定帧 → CFO → 信道估计 → 均衡 → 相位跟踪 → 译码", 3)
pic_fit(s, ASSETS / "burst_internals.png", Inches(0.5), Inches(1.7),
        Inches(12.3), Inches(3.0))
text(s, Inches(0.55), Inches(4.95), Inches(12.2), Inches(2.2), [
    [("调通过程踩过的四个坑：", {"bold": True, "color": NAVY}),
     ("伪相关峰（短前导周期性）→ 换长前导定帧；周期模糊 → 双段 ZC；", {}),
     ("噪声泄漏 → 信道估计按门限截断；首径选错 → 相关峰需对齐长前导", {})],
    [("估计器对照理论界验证过：", {"bold": True, "color": ACCENT}),
     ("CFO 估计 std=3.5e-5 周/样点，离渐近界仅差 2%（达界）；"
      "LS 信道估计每抽头方差 σ²/Ntr，实测/界 = 1.0~1.05", {})],
], size=15)

# ---------------- S4 验证 ----------------
s = prs.slides.add_slide(BLANK)
header(s, "验证：仿真压理论曲线，增益可定量", "BER 逐点压理论 → 仿真没写错；每条数字都能一条命令重算", 4)
w3, h3 = Inches(4.05), Inches(3.35)
labels = ["BER vs 理论（纯 AWGN 压线）", "编码增益：软判 Viterbi ≈ +2dB",
          "OFDM vs 单载波（同信道对比）"]
for i, (f, lab) in enumerate(zip(
        ["ber_awgn.png", "burst_ber.png", "ofdm_ber.png"], labels)):
    l = Inches(0.45) + i * (w3 + Inches(0.22))
    pic_fit(s, ASSETS / f, l, Inches(1.75), w3, h3)
    text(s, l, Inches(5.15), w3, Inches(0.5), lab, size=12.5,
         color=GRAY, align=PP_ALIGN.CENTER)
text(s, Inches(0.55), Inches(5.8), Inches(12.2), Inches(1.4), [
    [("低 SNR 下 OFDM 反而优于单载波", {"bold": True, "color": ACCENT}),
     ("——有限长 MMSE 均衡器退化为匹配滤波，残留 ISI 除不掉；"
      "OFDM 把 ISI 化成逐载波平坦衰落", {})],
    [("CFO 不补偿 → ICI 闭式 ", {"bold": True, "color": NAVY}),
     ("P=(πε)²/3：ε=0.192 → −9.4dB，64QAM ~70% 报废；补偿后残余 −48dB", {})],
], size=14.5)

# ---------------- S5 AMC ----------------
s = prs.slides.add_slide(BLANK)
header(s, "链路当数据工厂：1D-ResNet 调制识别", "Kaggle T4 全量训练 · 5 类调制 · 332K 参数", 5)
pic_fit(s, ASSETS / "amc_eval_gpu.png", Inches(0.5), Inches(1.75),
        Inches(9.2), Inches(4.2))
text(s, Inches(9.9), Inches(1.9), Inches(3.0), Inches(4.5), [
    [("76.9%", {"bold": True, "color": ACCENT, "size": 34})],
    "混合 SNR 准确率（高 SNR 端 ~88%）",
    "",
    [("混淆结构 = 物理特征", {"bold": True, "color": NAVY})],
    "网络自己学出 PSK 族内、方形 QAM 族内的混淆对——与手工统计特征的教科书结论一致",
], size=14)

# ---------------- S6 工程素养 ----------------
s = prs.slides.add_slide(BLANK)
header(s, "工程习惯：每个数字都能一条命令重算", None, 6)
text(s, Inches(0.9), Inches(2.0), Inches(11.6), Inches(4.6), [
    [("63 项测试", {"bold": True, "color": ACCENT, "size": 20}),
     ("  —— 57 项 Python 数值验证（贴理论曲线/理论界）+ 6 项 JS 回归（CI 里 node 实跑演示页真实代码）", {"size": 16})],
    [("理论界回归", {"bold": True, "color": ACCENT, "size": 20}),
     ("  —— CFO 达界、LS 每抽头 σ²/Ntr、d_free=10 枚举、ICI 闭式……全部固化成 CI 断言", {"size": 16})],
    [("文档站", {"bold": True, "color": ACCENT, "size": 20}),
     ("  —— 10 篇模块讲解 + 面试手册 + 交互演示页（mkdocs，push 自动部署）", {"size": 16})],
    [("数字复核清单", {"bold": True, "color": ACCENT, "size": 20}),
     ("  —— 文档里每条定量声称对应一条复现命令，任何人可一键验证", {"size": 16})],
], leading=1.25)

# ---------------- S7 总结 ----------------
s = prs.slides.add_slide(BLANK)
header(s, "总结与下一步", None, 7)
text(s, Inches(0.9), Inches(2.0), Inches(11.6), Inches(4.6), [
    [("做了什么：", {"bold": True, "color": NAVY, "size": 20}),
     ("把课本理论落成一条可验证的完整链路（突发单载波 + OFDM），并外接 AI 调制识别模块", {"size": 16})],
    "",
    [("下一步：", {"bold": True, "color": ACCENT, "size": 20})],
    "  · LDPC / Polar 现代信道编码",
    "  · RadioML2016/2018 公开数据集对照实验",
    "  · SDR（USRP/PlutoSDR）实物收发验证",
    "",
    [("我对物理层链路方向很感兴趣，希望在研究生阶段继续深入。",
      {"color": NAVY, "size": 17, "bold": True})],
], leading=1.25)

# ---------------- S8 备用：速查数字 ----------------
s = prs.slides.add_slide(BLANK)
header(s, "备用页 · 关键数字速查", "问答环节用——每条都有 CI 测试或复现命令背书", 8)
rows = [
    ("CFO 估计", "std 3.5e-5 周/样点 ≈ 0.10°/符号，实测/界=1.02（达界）"),
    ("ICI 代价", "不补偿 ε=0.192→−9.4dB（64QAM ~70% 报废）；补偿后 −48dB"),
    ("均衡收益", "severe 多径 SER 27.4%→0.024%；ZF 噪声增强 ×6.6 vs MMSE ×3.9"),
    ("编码增益", "软判 Viterbi ≈ +2dB；d_free=10 → 渐近 ~7dB"),
    ("相干带宽", "severe σ_τ=2.01 样点→B_c≈0.08，载波间隔 1/64，余量 5~9×"),
    ("Ricean K", "P(|h|²<−10dB)：K=0→9.5% / K=3→2.8% / K=10→0.07%"),
    ("AMC", "混合 SNR 76.9%、高 SNR ~88%；AMCNet(5) 参数 332,165"),
    ("OFDM 配置", "N_fft=64、CP=16、62 可用载波、4 导频→58 数据载波"),
]
y = Inches(1.75)
for i, (k, v) in enumerate(rows):
    fill = LIGHT if i % 2 == 0 else None
    box(s, Inches(0.7), y, Inches(2.2), Inches(0.56), fill=fill)
    box(s, Inches(2.9), y, Inches(9.7), Inches(0.56), fill=fill)
    text(s, Inches(0.85), y + Inches(0.09), Inches(2.0), Inches(0.4), k,
         size=13.5, color=NAVY, bold=True)
    text(s, Inches(3.05), y + Inches(0.09), Inches(9.4), Inches(0.4), v,
         size=13.5, color=GRAY)
    y += Inches(0.62)

# ---------------- 演讲者备注（演示者视图可见的提词） ----------------
NOTES = [
    "各位老师好，我的项目是把通信原理里一条完整链路从比特到比特实现出来——"
    "信道编码、调制、成型、同步、均衡、OFDM，再让这条链路自己产数据，训练"
    "神经网络识别调制方式。物理层链路全部用 NumPy 手写（AMC 用 PyTorch），"
    "没有调通信库。（~20 秒）",
    "动机很简单：课上学了一堆公式，但没见过它们连成一条链路的样子。左边发射、"
    "右边接收，接收端每一级对应撤销一种信道损伤。我按'一级一个问题'组织代码，"
    "所以每个模块都能单独讲透。（~30 秒；钩子：模块划分=课本章节）",
    "难点都在接收端，挑两个讲。突发接收机流水线：相关定帧→两段 ZC 前导 Moose "
    "CFO→LS 信道估计→MMSE 均衡→判决引导相位跟踪。调通过程踩过伪相关峰、周期"
    "模糊、噪声泄漏、首径选错四个真实的坑。（~60 秒；深挖：CFO std 3.5e-5 "
    "周/样点，离渐近界差 2%；LS 每抽头 σ²/Ntr）",
    "左边 BER 逐点压理论曲线说明仿真没写错；中间软判决 Viterbi 工作区增益约 "
    "2dB；右边 OFDM 与单载波同信道对比——低 SNR 下 OFDM 反而更好（有限长 "
    "MMSE 均衡退化残留 ISI；OFDM 把 ISI 化成逐载波平坦衰落）；高 SNR OFDM "
    "~1% 残留来自深衰落载波噪声放大。（~60 秒；钩子：CFO 不补偿 → ICI 闭式 "
    "(πε)²/3，ε=0.192→−9.4dB，补偿后 −48dB）",
    "链路当数据工厂，喂给 1D-ResNet 做调制识别。Kaggle GPU 全量训练混合 SNR "
    "76.9%、高 SNR 端 ~88%。混淆矩阵结构最有意思：网络自己学出 PSK 族内、"
    "QAM 族内混淆对——学到的是物理特征。（~40 秒）",
    "工程习惯：63 项测试、逐模块中文文档站、CI、可复现实验脚本，还有一页"
    "'每条数字怎么复核'的清单——文档里每个数字都能一条命令重算。（~10 秒）",
    "总结：项目把课本理论落成可验证链路+AI 识别模块。下一步：LDPC/Polar、"
    "RadioML 公开数据集对照、SDR 实物验证。我对物理层链路方向很感兴趣，"
    "希望研究生阶段继续深入。（~20 秒）",
    "备用页：被问任何一条数字都指到 verification.md 的复现命令。",
]
for slide, note in zip(prs.slides, NOTES):
    slide.notes_slide.notes_text_frame.text = note

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"written: {OUT} ({OUT.stat().st_size//1024} KB)")
