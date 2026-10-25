# -*- coding: utf-8 -*-
"""Generate M4-GPT architecture PPT with python-pptx."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ====== Color Palette (Tech/Futuristic) ======
BG_DARK   = RGBColor(0x11, 0x18, 0x27)
BG_PANEL  = RGBColor(0x1C, 0x25, 0x38)
BG_CARD   = RGBColor(0x24, 0x2E, 0x44)
ACCENT    = RGBColor(0x00, 0xD4, 0xAA)
ACCENT2   = RGBColor(0x00, 0x94, 0xD6)
TEXT      = RGBColor(0xE8, 0xEC, 0xF4)
TEXT_DIM  = RGBColor(0x8B, 0x95, 0xA5)
TEXT_ACC  = RGBColor(0x00, 0xD4, 0xAA)
RED       = RGBColor(0xFF, 0x4D, 0x6A)
ORANGE    = RGBColor(0xFF, 0xA0, 0x3D)

# ====== Slide dimensions ======
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H


def add_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_shape(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1.5)
    else:
        shape.line.fill.background()
    return shape


def add_rect(slide, left, top, width, height, fill_color):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    return shape


def add_text(slide, left, top, width, height, text, size=14, color=TEXT, bold=False, align=PP_ALIGN.LEFT, font_name="Calibri"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = font_name
    return tb


def add_arrow(slide, left, top, width, height, color=ACCENT):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_down_arrow(slide, left, top, width, height, color=ACCENT):
    shape = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


# ============================================================
# SLIDE 1: COVER
# ============================================================
slide1 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide1, BG_DARK)

add_rect(slide1, Inches(0), Inches(2.8), Inches(13.333), Inches(0.06), ACCENT)

add_text(slide1, Inches(1), Inches(1.5), Inches(11.333), Inches(1.2),
         "M4-GPT", size=54, color=TEXT, bold=True, align=PP_ALIGN.CENTER, font_name="Georgia")

add_text(slide1, Inches(1), Inches(3.2), Inches(11.333), Inches(0.6),
         "毫米波雷达点云理解\nVision-Language 大模型",
         size=20, color=TEXT_DIM, bold=False, align=PP_ALIGN.CENTER)

add_text(slide1, Inches(1), Inches(4.2), Inches(11.333), Inches(0.5),
         "点云 → 三视图图像 → LoRA 微调 Qwen2-VL → 动作问答",
         size=14, color=TEXT_ACC, bold=False, align=PP_ALIGN.CENTER)

add_text(slide1, Inches(1), Inches(6.8), Inches(11.333), Inches(0.4),
         "ASweicon  |  M4Human 数据集  |  Qwen2-VL-2B-Instruct",
         size=11, color=TEXT_DIM, align=PP_ALIGN.CENTER)

add_rect(slide1, Inches(0), Inches(0), Inches(0.06), Inches(7.5), ACCENT)
add_rect(slide1, Inches(13.27), Inches(0), Inches(0.06), Inches(7.5), ACCENT2)


# ============================================================
# SLIDE 2: 渲染管线详解（中文）
# ============================================================
slide2 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide2, BG_DARK)

add_text(slide2, Inches(0.5), Inches(0.2), Inches(12.333), Inches(0.6),
         "点云 → 图像渲染管线详解", size=28, color=TEXT, bold=True, font_name="Georgia")
add_rect(slide2, Inches(0.5), Inches(0.75), Inches(2.8), Inches(0.04), ACCENT)

# ---- Step 1: 原始数据读取 ----
add_text(slide2, Inches(0.5), Inches(1.0), Inches(3), Inches(0.35),
         "Step 1: 读取雷达点云", size=14, color=ACCENT, bold=True)

s1 = add_shape(slide2, Inches(0.5), Inches(1.4), Inches(3.5), Inches(2.2), BG_PANEL, ACCENT)
add_text(slide2, Inches(0.7), Inches(1.5), Inches(3.1), Inches(0.3),
         "M4Human LMDB 数据库", size=12, color=TEXT, bold=True)
add_text(slide2, Inches(0.7), Inches(1.85), Inches(3.1), Inches(0.28),
         "键格式: [被试ID, 动作ID, 帧ID]", size=10, color=TEXT_DIM)
add_text(slide2, Inches(0.7), Inches(2.1), Inches(3.1), Inches(0.28),
         "每帧 ~400-600 个点", size=10, color=TEXT_DIM)
add_text(slide2, Inches(0.7), Inches(2.35), Inches(3.1), Inches(0.28),
         "物理范围:", size=10, color=TEXT_ACC)
add_text(slide2, Inches(0.7), Inches(2.55), Inches(3.1), Inches(0.28),
         "x: -3m ~ 2.5m,  y: 0m ~ 6m,  z: 0m ~ 3m", size=9, color=TEXT_DIM)
add_text(slide2, Inches(0.7), Inches(2.75), Inches(3.1), Inches(0.28),
         "数据列: x, y, z, intensity（强度）", size=9, color=TEXT_DIM)

add_arrow(slide2, Inches(4.1), Inches(2.2), Inches(0.5), Inches(0.35), ACCENT)

# ---- Step 2: 取 T=4 帧窗口 ----
add_text(slide2, Inches(4.8), Inches(1.0), Inches(3), Inches(0.35),
         "Step 2: 取 T=4 帧时序窗口", size=14, color=ACCENT, bold=True)

s2 = add_shape(slide2, Inches(4.8), Inches(1.4), Inches(3.5), Inches(2.2), BG_PANEL, ACCENT)
add_text(slide2, Inches(5.0), Inches(1.5), Inches(3.1), Inches(0.3),
         "时间窗口截取", size=12, color=TEXT, bold=True)
add_text(slide2, Inches(5.0), Inches(1.85), Inches(3.1), Inches(0.28),
         "选帧: 每个 (被试, 动作) 组合", size=10, color=TEXT_DIM)
add_text(slide2, Inches(5.0), Inches(2.1), Inches(3.1), Inches(0.28),
         "取 K 帧（均匀分布在序列中）", size=10, color=TEXT_DIM)
add_text(slide2, Inches(5.0), Inches(2.35), Inches(3.1), Inches(0.28),
         "每帧往前取 T=4 帧连续窗口", size=10, color=TEXT_ACC)
add_text(slide2, Inches(5.0), Inches(2.55), Inches(3.1), Inches(0.28),
         "帧率 ≈12Hz → 4帧 ≈0.33秒", size=9, color=TEXT_DIM)
add_text(slide2, Inches(5.0), Inches(2.75), Inches(3.1), Inches(0.28),
         "覆盖完整动作周期的一小段", size=9, color=TEXT_DIM)

add_arrow(slide2, Inches(8.4), Inches(2.2), Inches(0.5), Inches(0.35), ACCENT)

# ---- Step 3: 三视图投影 + 时间颜色 ----
add_text(slide2, Inches(9.1), Inches(1.0), Inches(4), Inches(0.35),
         "Step 3: 三视图正交投影 + 时间颜色", size=14, color=ACCENT, bold=True)

s3 = add_shape(slide2, Inches(9.1), Inches(1.4), Inches(3.8), Inches(2.2), BG_PANEL, ACCENT)
add_text(slide2, Inches(9.3), Inches(1.5), Inches(3.4), Inches(0.3),
         "三个正交视图", size=12, color=TEXT, bold=True)

views = [("俯视图 (x-y)", "鸟瞰"), ("正视图 (x-z)", "侧视"), ("侧视图 (y-z)", "正视")]
for i, (name, desc) in enumerate(views):
    y = 1.9 + i * 0.28
    add_text(slide2, Inches(9.3), Inches(y), Inches(1.6), Inches(0.25),
             f"● {name}", size=10, color=TEXT_ACC)
    add_text(slide2, Inches(10.9), Inches(y), Inches(1.8), Inches(0.25),
             desc, size=10, color=TEXT_DIM)

# 时间颜色
add_text(slide2, Inches(9.3), Inches(2.75), Inches(3.4), Inches(0.28),
         "时间颜色编码（旧 → 新）:", size=10, color=TEXT_ACC)
colors_cn = [("t-3 最早", RGBColor(0x2B, 0x83, 0xBA)), ("t-2", RGBColor(0xAB, 0xDD, 0xA4)),
             ("t-1", RGBColor(0xFD, 0xAE, 0x61)), ("t 最新", RGBColor(0xD7, 0x19, 0x1C))]
for i, (label, col) in enumerate(colors_cn):
    x = 9.3 + i * 0.92
    dot = slide2.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(3.0), Inches(0.2), Inches(0.2))
    dot.fill.solid()
    dot.fill.fore_color.rgb = col
    dot.line.fill.background()
    add_text(slide2, Inches(x+0.22), Inches(2.98), Inches(0.6), Inches(0.2),
             label, size=8, color=TEXT_DIM)

# ---- Step 4: 合成 784×784 图像 ----
add_down_arrow(slide2, Inches(3.0), Inches(3.7), Inches(0.4), Inches(0.5), ACCENT)

add_text(slide2, Inches(0.5), Inches(4.3), Inches(3), Inches(0.35),
         "Step 4: 合成输出图像", size=14, color=ACCENT, bold=True)

s4 = add_shape(slide2, Inches(0.5), Inches(4.7), Inches(5.5), Inches(2.5), BG_PANEL, ACCENT)
add_text(slide2, Inches(0.7), Inches(4.8), Inches(5.1), Inches(0.3),
         "784 × 784 PNG", size=14, color=TEXT, bold=True)

add_text(slide2, Inches(0.7), Inches(5.2), Inches(5.1), Inches(0.28),
         "画布分配:", size=11, color=TEXT_ACC, bold=True)
add_text(slide2, Inches(0.7), Inches(5.45), Inches(5.1), Inches(0.25),
         "┌──────────┬──────────┐", size=10, color=TEXT_DIM)
add_text(slide2, Inches(0.7), Inches(5.65), Inches(5.1), Inches(0.25),
         "│ 俯视图   │ 正视图   │  ← 三视图区域 (2×2 网格)", size=10, color=TEXT_DIM)
add_text(slide2, Inches(0.7), Inches(5.85), Inches(5.1), Inches(0.25),
         "├──────────┼──────────┤", size=10, color=TEXT_DIM)
add_text(slide2, Inches(0.7), Inches(6.05), Inches(5.1), Inches(0.25),
         "│ 侧视图   │ 图例面板 │  ← 图例: 时间顺序 t-3→t", size=10, color=TEXT_DIM)
add_text(slide2, Inches(0.7), Inches(6.25), Inches(5.1), Inches(0.25),
         "└──────────┴──────────┘", size=10, color=TEXT_DIM)
add_text(slide2, Inches(0.7), Inches(6.5), Inches(5.1), Inches(0.2),
         "分辨率选择: 784 = 28 × 28 → ViT patch size = 14 → 784 visual tokens", size=9, color=TEXT_ACC)

# 右侧：渲染特性
add_text(slide2, Inches(6.2), Inches(4.3), Inches(3), Inches(0.35),
         "渲染特性与局限", size=14, color=ORANGE, bold=True)

feat_card = add_shape(slide2, Inches(6.2), Inches(4.7), Inches(6.8), Inches(2.5), BG_PANEL, ORANGE)

features = [
    ("✅ 优点", "", ACCENT),
    ("三视图保留空间结构", "x/y/z 三个方向的投影互不丢失", TEXT_DIM),
    ("时间颜色编码保留运动信息", "4帧用4种颜色，颜色渐变 = 时间顺序", TEXT_DIM),
    ("黑底 + 无标签", "图例面板只画时间不写被试/动作，防止标签泄漏", TEXT_DIM),
    ("⚠️ 局限", "", RED),
    ("intensity 通道未使用", "只用 xyz 三列点坐标，强度信息被丢弃", TEXT_DIM),
    ("CPU 渲染 (matplotlib)", "不走 GPU，matplotlib 散点图逐个像素算", TEXT_DIM),
    ("分辨率损失", "原始点云 ~400点/帧 → 784×784 → 每帧变成散点图", TEXT_DIM),
]
row = 0
for title, detail, col in features:
    y = 4.8 + row * 0.3
    add_text(slide2, Inches(6.4), Inches(y), Inches(1.6), Inches(0.28),
             title, size=10, color=col, bold=True)
    if detail:
        add_text(slide2, Inches(8.0), Inches(y), Inches(4.8), Inches(0.28),
                 detail, size=9, color=TEXT_DIM)
    row += 1

add_rect(slide2, Inches(0), Inches(0), Inches(0.06), Inches(7.5), ACCENT)


# ============================================================
# SLIDE 3: 整体管线架构图（中文）
# ============================================================
slide3 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide3, BG_DARK)

add_text(slide3, Inches(0.5), Inches(0.2), Inches(12.333), Inches(0.6),
         "M4-GPT 完整处理管线", size=28, color=TEXT, bold=True, font_name="Georgia")
add_rect(slide3, Inches(0.5), Inches(0.75), Inches(2), Inches(0.04), ACCENT)

# ---- Phase 0 ----
add_text(slide3, Inches(0.5), Inches(1.0), Inches(2), Inches(0.35),
         "阶段 0: 原始数据", size=13, color=ACCENT, bold=True)

c0 = add_shape(slide3, Inches(0.5), Inches(1.4), Inches(2.8), Inches(1.8), BG_PANEL, ACCENT)
add_text(slide3, Inches(0.7), Inches(1.5), Inches(2.4), Inches(0.3),
         "M4Human LMDB", size=13, color=TEXT, bold=True)
add_text(slide3, Inches(0.7), Inches(1.85), Inches(2.4), Inches(0.3),
         "radar_pc.lmdb", size=11, color=TEXT_DIM)
add_text(slide3, Inches(0.7), Inches(2.15), Inches(2.4), Inches(0.3),
         "毫米波雷达点云", size=10, color=TEXT_DIM)
add_text(slide3, Inches(0.7), Inches(2.4), Inches(2.4), Inches(0.3),
         "661K帧, 999序列, 50动作", size=9, color=TEXT_DIM)
add_text(slide3, Inches(0.7), Inches(2.6), Inches(2.4), Inches(0.3),
         "20被试 × Vayyar vTrigB 雷达", size=9, color=TEXT_ACC)

add_arrow(slide3, Inches(3.4), Inches(2.1), Inches(0.5), Inches(0.35), ACCENT)

# ---- Phase 1 ----
add_text(slide3, Inches(4.1), Inches(1.0), Inches(3), Inches(0.35),
         "阶段 1: 点云→图像渲染", size=13, color=ACCENT, bold=True)

c1 = add_shape(slide3, Inches(4.1), Inches(1.4), Inches(2.8), Inches(1.8), BG_PANEL, ACCENT)
add_text(slide3, Inches(4.3), Inches(1.5), Inches(2.4), Inches(0.3),
         "三视图 + 时间颜色", size=12, color=TEXT, bold=True)
add_text(slide3, Inches(4.3), Inches(1.85), Inches(2.4), Inches(0.3),
         "T=4帧连续窗口", size=10, color=TEXT_DIM)
add_text(slide3, Inches(4.3), Inches(2.1), Inches(2.4), Inches(0.3),
         "BEV / Front / Side", size=10, color=TEXT_DIM)
add_text(slide3, Inches(4.3), Inches(2.35), Inches(2.4), Inches(0.3),
         "蓝→绿→橙→红 时间编码", size=10, color=TEXT_ACC)
add_text(slide3, Inches(4.3), Inches(2.55), Inches(2.4), Inches(0.3),
         "784×784 PNG", size=10, color=TEXT_DIM)

add_arrow(slide3, Inches(7.0), Inches(2.1), Inches(0.5), Inches(0.35), ACCENT)

# ---- Phase 2 ----
add_text(slide3, Inches(7.7), Inches(1.0), Inches(3), Inches(0.35),
         "阶段 2: 图像→问答数据", size=13, color=ACCENT, bold=True)

c2 = add_shape(slide3, Inches(7.7), Inches(1.4), Inches(2.8), Inches(1.8), BG_PANEL, ACCENT)
add_text(slide3, Inches(7.9), Inches(1.5), Inches(2.4), Inches(0.3),
         "每张图 × 3 种问答", size=12, color=TEXT, bold=True)
add_text(slide3, Inches(7.9), Inches(1.85), Inches(2.4), Inches(0.3),
         "细粒度: 50类动作名", size=10, color=TEXT_DIM)
add_text(slide3, Inches(7.9), Inches(2.1), Inches(2.4), Inches(0.3),
         "粗粒度: 3类 (原地/坐姿/移动)", size=10, color=TEXT_DIM)
add_text(slide3, Inches(7.9), Inches(2.35), Inches(2.4), Inches(0.3),
         "描述: 模板句", size=10, color=TEXT_DIM)
add_text(slide3, Inches(7.9), Inches(2.55), Inches(2.4), Inches(0.3),
         "输出 train/val/test.jsonl", size=10, color=TEXT_ACC)

add_down_arrow(slide3, Inches(9.1), Inches(3.3), Inches(0.4), Inches(0.5), ACCENT)

# ---- Phase 3 ----
add_text(slide3, Inches(0.5), Inches(4.0), Inches(3), Inches(0.35),
         "阶段 3: LoRA 微调训练", size=13, color=ACCENT, bold=True)

c3 = add_shape(slide3, Inches(0.5), Inches(4.4), Inches(6.0), Inches(2.6), BG_PANEL, ACCENT)

add_text(slide3, Inches(0.7), Inches(4.55), Inches(5.6), Inches(0.3),
         "Qwen2-VL-2B-Instruct (bf16, 冻结全部参数)", size=12, color=TEXT, bold=True)

# Vision Tower
add_rect(slide3, Inches(0.8), Inches(4.95), Inches(2.7), Inches(1.8), BG_CARD)
add_text(slide3, Inches(0.9), Inches(5.0), Inches(2.5), Inches(0.25),
         "视觉塔 (ViT-L)", size=10, color=TEXT, bold=True)
add_text(slide3, Inches(0.9), Inches(5.25), Inches(2.5), Inches(0.25),
         "24 blocks, 冻结", size=9, color=TEXT_DIM)
add_text(slide3, Inches(0.9), Inches(5.5), Inches(2.5), Inches(0.25),
         "🔧 LoRA (r=16):", size=9, color=TEXT_ACC)
add_text(slide3, Inches(0.9), Inches(5.75), Inches(2.5), Inches(0.25),
         "  qkv, attn.proj, fc1, fc2", size=9, color=RED)
add_text(slide3, Inches(0.9), Inches(6.2), Inches(2.5), Inches(0.25),
         "输出: 784 个视觉 token", size=9, color=TEXT_DIM)

# LLM
add_rect(slide3, Inches(3.6), Inches(4.95), Inches(2.8), Inches(1.8), BG_CARD)
add_text(slide3, Inches(3.7), Inches(5.0), Inches(2.6), Inches(0.25),
         "语言模型 (Qwen2)", size=10, color=TEXT, bold=True)
add_text(slide3, Inches(3.7), Inches(5.25), Inches(2.6), Inches(0.25),
         "28 blocks, base 冻结", size=9, color=TEXT_DIM)
add_text(slide3, Inches(3.7), Inches(5.5), Inches(2.6), Inches(0.25),
         "🔧 LoRA (r=16):", size=9, color=TEXT_ACC)
add_text(slide3, Inches(3.7), Inches(5.75), Inches(2.6), Inches(0.25),
         "  q/k/v/o/gate/up/down_proj", size=9, color=RED)
add_text(slide3, Inches(3.7), Inches(6.2), Inches(2.6), Inches(0.25),
         "训练: AdamW lr=1e-4", size=9, color=TEXT_DIM)

# LoRA badge
badge = add_shape(slide3, Inches(2.8), Inches(5.5), Inches(1.6), Inches(0.9), RED)
add_text(slide3, Inches(2.85), Inches(5.55), Inches(1.5), Inches(0.3),
         "🔧 LoRA", size=14, color=TEXT, bold=True, align=PP_ALIGN.CENTER)
add_text(slide3, Inches(2.85), Inches(5.85), Inches(1.5), Inches(0.4),
         "~0.1% 可训练参数", size=8, color=TEXT, align=PP_ALIGN.CENTER)

add_arrow(slide3, Inches(6.1), Inches(5.6), Inches(0.5), Inches(0.35), ACCENT)

# ---- Phase 4 ----
add_text(slide3, Inches(6.8), Inches(4.0), Inches(3), Inches(0.35),
         "阶段 4: 似然打分评估", size=13, color=ACCENT, bold=True)

c4 = add_shape(slide3, Inches(6.8), Inches(4.4), Inches(6.0), Inches(2.6), BG_PANEL, ACCENT)

add_text(slide3, Inches(7.0), Inches(4.55), Inches(5.6), Inches(0.3),
         "每张测试图的评估流程:", size=11, color=TEXT, bold=True)

steps = [
    ("①", "视觉塔前向（每个 clip 只算一次，缓存）", ACCENT),
    ("②", "50 个候选动作名 → 拼接 50 条 prompt", TEXT_DIM),
    ("③", "批量计算答案 span 的平均对数似然", TEXT_DIM),
    ("④", "取最高 log prob 的候选 = 预测答案", ACCENT),
]
for i, (n, txt, col) in enumerate(steps):
    y = 4.9 + i * 0.42
    circle = slide3.shapes.add_shape(MSO_SHAPE.OVAL, Inches(7.1), Inches(y), Inches(0.3), Inches(0.3))
    circle.fill.solid()
    circle.fill.fore_color.rgb = ACCENT
    circle.line.fill.background()
    add_text(slide3, Inches(7.1), Inches(y), Inches(0.3), Inches(0.3),
             n, size=11, color=BG_DARK, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide3, Inches(7.5), Inches(y+0.02), Inches(5.0), Inches(0.3),
             txt, size=10, color=col)

metbox = add_shape(slide3, Inches(9.5), Inches(5.8), Inches(3.0), Inches(1.0), BG_CARD)
add_text(slide3, Inches(9.6), Inches(5.85), Inches(2.8), Inches(0.25),
         "评估指标:", size=10, color=TEXT, bold=True)
add_text(slide3, Inches(9.6), Inches(6.1), Inches(2.8), Inches(0.25),
         "Top-1 / Top-5 准确率", size=9, color=TEXT_ACC)
add_text(slide3, Inches(9.6), Inches(6.3), Inches(2.8), Inches(0.25),
         "Per-Action 混淆矩阵", size=9, color=TEXT_DIM)

add_rect(slide3, Inches(0), Inches(0), Inches(0.06), Inches(7.5), ACCENT)


# ============================================================
# SLIDE 4: LoRA 原理（中文）
# ============================================================
slide4 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide4, BG_DARK)

add_text(slide4, Inches(0.5), Inches(0.2), Inches(12.333), Inches(0.6),
         "LoRA 微调原理", size=28, color=TEXT, bold=True, font_name="Georgia")
add_rect(slide4, Inches(0.5), Inches(0.75), Inches(2), Inches(0.04), ACCENT)

# 数学公式
formula_card = add_shape(slide4, Inches(0.5), Inches(1.1), Inches(5.5), Inches(2.5), BG_PANEL, ACCENT)
add_text(slide4, Inches(0.7), Inches(1.2), Inches(5.1), Inches(0.3),
         "LoRA 数学公式", size=14, color=TEXT, bold=True)

formula_lines = [
    "原始层（冻结）:",
    "  y = W · x",
    "  W ∈ R^{d×d}  (d 通常 2048 或 4096)",
    "",
    "LoRA 注入后（可训练）:",
    "  y = W · x + (α/r) · B · A · x",
    "  A ∈ R^{r×d},  B ∈ R^{d×r}",
    "  r = 16  (低秩，远小于 d)",
    "",
    "初始化: B = 0 → LoRA 分支 = 0, 不影响原始输出",
    "可训练参数: 每层仅 2·r·d ≈ 0.1% of W",
]
for i, line in enumerate(formula_lines):
    is_formula = "=" in line and not line.startswith("  W") and not line.startswith("  r")
    add_text(slide4, Inches(0.7), Inches(1.6 + i*0.18), Inches(5.1), Inches(0.2),
             line, size=10,
             color=TEXT_ACC if is_formula else TEXT if line and not line.startswith("  ") else TEXT_DIM,
             bold=line.startswith("原始层") or line.startswith("LoRA 注入") or line.startswith("初始化"))

# 图示
vis_card = add_shape(slide4, Inches(6.3), Inches(1.1), Inches(6.5), Inches(5.5), BG_PANEL, ACCENT)
add_text(slide4, Inches(6.5), Inches(1.2), Inches(6.1), Inches(0.3),
         "LoRA 层结构示意", size=14, color=TEXT, bold=True)

# Before
add_text(slide4, Inches(6.5), Inches(1.55), Inches(6.1), Inches(0.25),
         "注入前（冻结）:", size=11, color=TEXT_DIM, bold=True)
inp = add_shape(slide4, Inches(6.7), Inches(1.85), Inches(0.9), Inches(0.5), BG_CARD)
add_text(slide4, Inches(6.7), Inches(1.88), Inches(0.9), Inches(0.45),
         "x", size=14, color=TEXT, bold=True, align=PP_ALIGN.CENTER)
w = add_shape(slide4, Inches(7.7), Inches(1.85), Inches(1.2), Inches(0.5), RED)
add_text(slide4, Inches(7.7), Inches(1.88), Inches(1.2), Inches(0.45),
         "W (冻结)", size=11, color=TEXT, bold=True, align=PP_ALIGN.CENTER)
add_arrow(slide4, Inches(6.85), Inches(2.0), Inches(0.8), Inches(0.25), TEXT_DIM)
out = add_shape(slide4, Inches(9.1), Inches(1.85), Inches(0.9), Inches(0.5), BG_CARD)
add_text(slide4, Inches(9.1), Inches(1.88), Inches(0.9), Inches(0.45),
         "y", size=14, color=TEXT, bold=True, align=PP_ALIGN.CENTER)

# After
add_text(slide4, Inches(6.5), Inches(2.6), Inches(6.1), Inches(0.25),
         "注入后（可训练）:", size=11, color=TEXT_ACC, bold=True)
inp2 = add_shape(slide4, Inches(6.7), Inches(3.0), Inches(0.9), Inches(0.5), BG_CARD)
add_text(slide4, Inches(6.7), Inches(3.03), Inches(0.9), Inches(0.45),
         "x", size=14, color=TEXT, bold=True, align=PP_ALIGN.CENTER)
w2 = add_shape(slide4, Inches(7.7), Inches(3.0), Inches(1.2), Inches(0.5), RED)
add_text(slide4, Inches(7.7), Inches(3.03), Inches(1.2), Inches(0.45),
         "W (冻结)", size=11, color=TEXT, bold=True, align=PP_ALIGN.CENTER)
a = add_shape(slide4, Inches(7.7), Inches(3.7), Inches(0.55), Inches(0.5), ACCENT)
add_text(slide4, Inches(7.7), Inches(3.73), Inches(0.55), Inches(0.45),
         "A", size=14, color=BG_DARK, bold=True, align=PP_ALIGN.CENTER)
b = add_shape(slide4, Inches(8.4), Inches(3.7), Inches(0.55), Inches(0.5), ACCENT2)
add_text(slide4, Inches(8.4), Inches(3.73), Inches(0.55), Inches(0.45),
         "B", size=14, color=TEXT, bold=True, align=PP_ALIGN.CENTER)
out2 = add_shape(slide4, Inches(9.4), Inches(3.0), Inches(0.9), Inches(0.5), BG_CARD)
add_text(slide4, Inches(9.4), Inches(3.03), Inches(0.9), Inches(0.45),
         "y", size=14, color=TEXT, bold=True, align=PP_ALIGN.CENTER)

add_arrow(slide4, Inches(6.85), Inches(3.15), Inches(0.8), Inches(0.25), TEXT_DIM)
add_arrow(slide4, Inches(6.85), Inches(3.85), Inches(0.8), Inches(0.25), ACCENT)
add_arrow(slide4, Inches(8.25), Inches(3.85), Inches(0.15), Inches(0.25), ACCENT)
add_down_arrow(slide4, Inches(8.68), Inches(4.2), Inches(0.3), Inches(0.4), ACCENT)
add_arrow(slide4, Inches(8.95), Inches(3.15), Inches(0.4), Inches(0.25), TEXT_DIM)
add_text(slide4, Inches(8.95), Inches(3.4), Inches(0.6), Inches(0.2),
         "+", size=14, color=TEXT_ACC, bold=True, align=PP_ALIGN.CENTER)

# 统计表
stats = [
    ("组件", "LoRA 层数", "状态"),
    ("视觉塔 (ViT-L)", "24 × 4 = 96", "🔧 LoRA (r=16)"),
    ("语言模型 (28 blocks)", "28 × 7 = 196", "🔧 LoRA (r=16)"),
    ("LoRA 总层数", "308", "可训练"),
    ("Base model 参数", "~2B", "冻结"),
    ("LoRA 可训练参数", "~0.1%", "可训练"),
]
for i, (col1, col2, col3) in enumerate(stats):
    y = 4.5 + i * 0.35
    c1 = TEXT_ACC if i == 0 else TEXT
    c2 = TEXT_ACC if i == 0 else TEXT_DIM
    c3 = TEXT_ACC if i == 0 else ACCENT if "LoRA" in col3 else TEXT_DIM
    add_text(slide4, Inches(6.5), Inches(y), Inches(2.5), Inches(0.3),
             col1, size=10, color=c1, bold=(i==0))
    add_text(slide4, Inches(9.0), Inches(y), Inches(1.5), Inches(0.3),
             col2, size=10, color=c2, bold=(i==0))
    add_text(slide4, Inches(10.5), Inches(y), Inches(2.3), Inches(0.3),
             col3, size=10, color=c3, bold=(i==0))

add_rect(slide4, Inches(0), Inches(0), Inches(0.06), Inches(7.5), ACCENT)


# ============================================================
# SLIDE 5: 训练配置 + 评估流程（中文）
# ============================================================
slide5 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide5, BG_DARK)

add_text(slide5, Inches(0.5), Inches(0.2), Inches(12.333), Inches(0.6),
         "训练配置与评估流程", size=28, color=TEXT, bold=True, font_name="Georgia")
add_rect(slide5, Inches(0.5), Inches(0.75), Inches(2.8), Inches(0.04), ACCENT)

# 左：训练配置
cfg_card = add_shape(slide5, Inches(0.5), Inches(1.1), Inches(6.0), Inches(5.8), BG_PANEL, ACCENT)
add_text(slide5, Inches(0.7), Inches(1.2), Inches(5.6), Inches(0.3),
         "训练超参数配置", size=14, color=TEXT, bold=True)

configs = [
    ("基础模型", "Qwen2-VL-2B-Instruct"),
    ("精度", "bf16"),
    ("LoRA rank (r)", "16"),
    ("LoRA alpha", "32"),
    ("LoRA dropout", "0.05"),
    ("Batch size", "4 (梯度累积 2, 等效 8)"),
    ("学习率", "1e-4"),
    ("学习率调度", "cosine + warmup (3%)"),
    ("梯度裁剪", "1.0"),
    ("梯度检查点", "开启（省显存）"),
    ("优化器", "AdamW"),
    ("最大步数", "3000 (full 数据集)"),
    ("LoRA 目标层", "LLM: q/k/v/o/gate/up/down_proj"),
    ("", "ViT: qkv/attn.proj/fc1/fc2"),
]
for i, (k, v) in enumerate(configs):
    y = 1.6 + i * 0.32
    add_text(slide5, Inches(0.7), Inches(y), Inches(2.8), Inches(0.28),
             k, size=10, color=TEXT_DIM)
    col = TEXT_ACC if i < 11 else RED
    add_text(slide5, Inches(3.5), Inches(y), Inches(2.8), Inches(0.28),
             v, size=10, color=col)

# 右：评估流程
eval_card = add_shape(slide5, Inches(6.8), Inches(1.1), Inches(6.0), Inches(5.8), BG_PANEL, ACCENT)
add_text(slide5, Inches(7.0), Inches(1.2), Inches(5.6), Inches(0.3),
         "似然打分评估流程", size=14, color=TEXT, bold=True)

pipeline = [
    ("①", "加载基础模型 + LoRA adapter", "PeftModel.from_pretrained", ACCENT),
    ("②", "每个 clip 视觉塔前向一次", "缓存视觉 token, --cbatch 5 批处理", TEXT_DIM),
    ("③", "50 个候选动作名拼 50 条 prompt", "批量计算避免重复前向", ACCENT),
    ("④", "计算答案 span 的平均 log prob", "似然越高 = 模型越确定", TEXT_DIM),
    ("⑤", "取 argmax = 预测动作", "与 GT 对比算 Top-1/Top-5", ACCENT),
]
for i, (n, title, detail, col) in enumerate(pipeline):
    y = 1.6 + i * 0.9
    circle = slide5.shapes.add_shape(MSO_SHAPE.OVAL, Inches(7.0), Inches(y+0.05), Inches(0.35), Inches(0.35))
    circle.fill.solid()
    circle.fill.fore_color.rgb = ACCENT
    circle.line.fill.background()
    add_text(slide5, Inches(7.0), Inches(y+0.05), Inches(0.35), Inches(0.35),
             n, size=13, color=BG_DARK, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide5, Inches(7.5), Inches(y), Inches(5.0), Inches(0.3),
             title, size=12, color=TEXT, bold=True)
    add_text(slide5, Inches(7.5), Inches(y+0.3), Inches(5.0), Inches(0.3),
             detail, size=10, color=col)

metbox2 = add_shape(slide5, Inches(7.0), Inches(6.3), Inches(5.6), Inches(0.5), ACCENT2)
add_text(slide5, Inches(7.1), Inches(6.32), Inches(5.4), Inches(0.45),
         "评估指标: Top-1 准确率  |  Top-5 准确率  |  Per-Action 混淆矩阵",
         size=11, color=TEXT, bold=True, align=PP_ALIGN.CENTER)

add_rect(slide5, Inches(0), Inches(0), Inches(0.06), Inches(7.5), ACCENT)


# ============================================================
# SLIDE 6: 诊断发现 + 实验路线（中文）
# ============================================================
slide6 = prs.slides.add_slide(prs.slide_layouts[6])
add_bg(slide6, BG_DARK)

add_text(slide6, Inches(0.5), Inches(0.2), Inches(12.333), Inches(0.6),
         "关键发现与后续路线图", size=28, color=TEXT, bold=True, font_name="Georgia")
add_rect(slide6, Inches(0.5), Inches(0.75), Inches(2), Inches(0.04), ACCENT)

# 左：诊断发现
find_card = add_shape(slide6, Inches(0.5), Inches(1.1), Inches(6.0), Inches(5.8), BG_PANEL, RED)
add_text(slide6, Inches(0.7), Inches(1.2), Inches(5.6), Inches(0.3),
         "诊断关键发现 (E1-E3)", size=14, color=TEXT, bold=True)

findings = [
    ("📊", "Loss 0.5 但评估 ≈ 2%", "coarse/desc QA 拉低平均 loss, fine 任务没学会", RED),
    ("🗂️", "Mini 数据集太小", "5,400 QA 条, full 模式应有 38,400 条", TEXT_ACC),
    ("🔄", "混合 QA 目标", "3 种问答混合训练, 简单的 coarse 掩盖了 fine 的失败", TEXT_DIM),
    ("🏷️", "6 个占位动作名", "13/24/25/38/49/50 污染 12% 的标签", TEXT_DIM),
    ("🚪", "Merger 层冻结", "视觉塔未适应雷达图像分布", TEXT_DIM),
    ("📡", "Intensity 通道未用", "只用 xyz 三列, 强度信息被丢弃", TEXT_DIM),
]
for i, (icon, title, detail, col) in enumerate(findings):
    y = 1.6 + i * 0.72
    ic = slide6.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.7), Inches(y), Inches(0.4), Inches(0.4))
    ic.fill.solid()
    ic.fill.fore_color.rgb = col
    ic.line.fill.background()
    add_text(slide6, Inches(0.7), Inches(y), Inches(0.4), Inches(0.4),
             icon, size=12, color=BG_DARK, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide6, Inches(1.2), Inches(y), Inches(5.1), Inches(0.28),
             title, size=11, color=TEXT, bold=True)
    add_text(slide6, Inches(1.2), Inches(y+0.3), Inches(5.1), Inches(0.3),
             detail, size=9, color=TEXT_DIM)

# 右：路线图
road_card = add_shape(slide6, Inches(6.8), Inches(1.1), Inches(6.0), Inches(5.8), BG_PANEL, ACCENT)
add_text(slide6, Inches(7.0), Inches(1.2), Inches(5.6), Inches(0.3),
         "后续实验路线图", size=14, color=TEXT, bold=True)

roadmap = [
    ("E1-E2", "诊断评估", "训练集 top-1 / 粗粒度 top-1 → 判断分支 A 或 B", True),
    ("E4", "重训", "修 6 占位名 + full 数据 + --task fine + ≥3000 步", True),
    ("E5", "监督基线", "P4Transformer 同 split → 上限参考", False),
    ("E6", "表示消融", "RPC-xyz vs +intensity / T=4 vs 16 / merger 解冻", False),
    ("E7", "PointNet++ 编码器", "若 E4 失败: 专用点云编码器 → Qwen LLM", False),
    ("E8", "多轮对话", "任务 D 数据构造 + SFT on fine-tuned model", False),
]
for i, (stage, title, detail, done) in enumerate(roadmap):
    y = 1.6 + i * 0.72
    status_col = ACCENT if done else TEXT_DIM
    status_icon = "✓" if done else "○"
    add_text(slide6, Inches(7.0), Inches(y), Inches(0.5), Inches(0.4),
             status_icon, size=16, color=status_col, bold=True)
    stage_box = add_shape(slide6, Inches(7.6), Inches(y+0.02), Inches(0.8), Inches(0.35), status_col)
    add_text(slide6, Inches(7.6), Inches(y+0.02), Inches(0.8), Inches(0.35),
             stage, size=9, color=BG_DARK, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide6, Inches(8.5), Inches(y), Inches(4.1), Inches(0.28),
             title, size=11, color=TEXT, bold=True)
    add_text(slide6, Inches(8.5), Inches(y+0.3), Inches(4.1), Inches(0.3),
             detail, size=9, color=TEXT_DIM)

add_rect(slide6, Inches(0), Inches(0), Inches(0.06), Inches(7.5), ACCENT)


# ====== SAVE ======
out_path = "/Users/weicongbing/Desktop/m4gpt/m4gpt/m4gpt_architecture.pptx"
prs.save(out_path)
print(f"SAVED: {out_path}")
