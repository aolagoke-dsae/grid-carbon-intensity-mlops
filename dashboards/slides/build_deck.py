"""Build the editable PowerPoint deck from dashboards/slides/deck_data.json.

Two steps, from the repository root:
    python dashboards/slides/export_deck_data.py     (project environment)
    python dashboards/slides/build_deck.py           (needs python-pptx and segno only)

Add --interview for the interview version, or give an output path. By default the decks are
written to dashboards/design/, which is not tracked. A profile photo is used if
dashboards/design/profile_photo.jpg exists; otherwise the slide keeps a placeholder.

Text is real text, diagrams are native shapes and charts are native PowerPoint charts, so
everything can be edited in PowerPoint. Colours and fonts come from the deck's theme.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import segno
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (
    XL_CHART_TYPE,
    XL_LABEL_POSITION,
    XL_LEGEND_POSITION,
    XL_MARKER_STYLE,
    XL_TICK_LABEL_POSITION,
)
from pptx.enum.dml import MSO_LINE, MSO_THEME_COLOR
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

D = json.loads(Path("dashboards/slides/deck_data.json").read_text(encoding="utf-8"))
F = D["facts"]  # figures quoted in text, looked up so that none is typed in here
TREES, WINDOW = D["method"]["trees"], D["method"]["window_days"]
MULTIPLE = {2: "Twice", 3: "Three times", 4: "Four times"}[int(F["largest_multiple"])]
INTERVIEW = "--interview" in sys.argv  # the interview deck: skills and lessons up front
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
_default = "grid_carbon_intensity_forecast_interview.pptx" if INTERVIEW else "grid_carbon_intensity_forecast_v8.pptx"
OUT = Path(_args[0] if _args else f"dashboards/design/{_default}")
OUT.parent.mkdir(parents=True, exist_ok=True)

THEME = {
    "dk1": "1F2328", "lt1": "FFFFFF", "dk2": "0F3D3E", "lt2": "F2F5F4",
    "accent1": "D95F02",  # DE-LU
    "accent2": "1B9E77",  # FR
    "accent3": "0F3D3E",  # primary (pine)
    "accent4": "57606A",  # secondary text
    "accent5": "8C959F",  # muted
    "accent6": "2B5C96",  # upstream repository
    "hlink": "2B5C96", "folHlink": "57606A",
}
FONT = "Calibri"
TC = {
    "tx1": MSO_THEME_COLOR.TEXT_1, "bg1": MSO_THEME_COLOR.BACKGROUND_1, "tx2": MSO_THEME_COLOR.TEXT_2,
    "bg2": MSO_THEME_COLOR.BACKGROUND_2, "de": MSO_THEME_COLOR.ACCENT_1, "fr": MSO_THEME_COLOR.ACCENT_2,
    "pri": MSO_THEME_COLOR.ACCENT_3, "sec": MSO_THEME_COLOR.ACCENT_4, "mut": MSO_THEME_COLOR.ACCENT_5,
    "up": MSO_THEME_COLOR.ACCENT_6,
}
GRID = RGBColor(0xE1, 0xE5, 0xE8)
SOURCE = (
    "Source: ENTSO-E Transparency Platform; data: 50Hertz, Amprion, TenneT, TransnetBW (DE-LU) and RTE (FR). "
    "Intensity values and forecasts are the author's own computations."
)
QUESTIONS = ["Can we forecast it?", "Can we trust the interval?", "Does acting on it pay?"]
ZL = {"DE-LU": "de", "FR": "fr"}

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
LAYOUT = prs.slide_layouts[5]  # "Title Only"


def patch_theme() -> None:
    """Write the deck's colours and fonts into the theme, so a restyle is one change."""
    part = prs.slide_master.part.part_related_by(RT.THEME)
    root = etree.fromstring(part.blob)
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    root.set("name", "Grid carbon intensity")
    scheme = root.find(".//a:clrScheme", ns)
    scheme.set("name", "Grid carbon intensity")
    for name, value in THEME.items():
        node = scheme.find(f"a:{name}", ns)
        for child in list(node):
            node.remove(child)
        etree.SubElement(node, f"{{{ns['a']}}}srgbClr").set("val", value)
    for tag in ("majorFont", "minorFont"):
        root.find(f".//a:{tag}/a:latin", ns).set("typeface", FONT)
    part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def colour(target, key: str) -> None:
    target.theme_color = TC[key]


def text(slide, x, y, w, h, paras, size=14, bold=False, color="tx1", anchor="top", align="l", name=None, spacing=None):
    """A text box. `paras` is a string or a list; each item is a string or a list of (text, options) runs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[anchor]
    fill_paragraphs(tf, paras, size, bold, color, align, spacing)
    return box


def fill_paragraphs(tf, paras, size, bold, color, align="l", spacing=None):
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        if spacing is not None:
            p.space_after = Pt(spacing)
        runs = [(para, {})] if isinstance(para, str) else para
        for run_text, opt in runs:
            run = p.add_run()
            run.text = run_text
            run.font.size = Pt(opt.get("size", size))
            run.font.bold = opt.get("bold", bold)
            colour(run.font.color, opt.get("color", color))


def shape(slide, kind, x, y, w, h, fill="bg2", line=None, name=None, line_width=1.25):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        colour(s.fill.fore_color, fill)
    if line is None:
        s.line.fill.background()
    else:
        colour(s.line.color, line)
        s.line.width = Pt(line_width)
    return s


def card(slide, x, y, w, h, fill="bg2", line=None, name=None):
    s = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill, line=line, name=name)
    s.adjustments[0] = min(0.12, 0.09 / min(w, h))
    return s


def badge(slide, x, y, d, label, fill="pri", color="bg1", size=14):
    s = shape(slide, MSO_SHAPE.OVAL, x, y, d, d, fill=fill, name=f"Number {label}")
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    fill_paragraphs(tf, label, size, True, color, "c")
    return s


def label_shape(s, paras, size=12, bold=False, color="tx1", align="c"):
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    fill_paragraphs(tf, paras, size, bold, color, align)


def arrow(slide, x1, y1, x2, y2, color="sec", width=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.name = "Arrow"
    colour(c.line.color, color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    return c


def slide_number(slide, x, y, color="mut"):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(0.6), Inches(0.3))
    box.name = "Slide number"
    tf = box.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    fld = etree.SubElement(p._p, qn("a:fld"))
    fld.set("id", "{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}")
    fld.set("type", "slidenum")
    rpr = etree.SubElement(fld, qn("a:rPr"))
    rpr.set("lang", "en-GB")
    rpr.set("sz", "1000")
    fill_el = etree.SubElement(rpr, qn("a:solidFill"))
    etree.SubElement(fill_el, qn("a:schemeClr")).set("val", "accent5")
    etree.SubElement(fld, qn("a:t")).text = "‹#›"


def new_slide(kicker, title, subtitle=None, question=None, dark=False, notes="", source=True):
    slide = prs.slides.add_slide(LAYOUT)
    ink, soft = ("bg1", "bg2") if dark else ("tx1", "sec")
    if dark:
        slide.background.fill.solid()
        colour(slide.background.fill.fore_color, "tx2")
    text(slide, 0.5, 0.32, 5.5, 0.3, kicker.upper(), size=12, bold=True, color="bg2" if dark else "pri", name="Section label")
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.5), Inches(0.62), Inches(12.33), Inches(0.62)
    tf = t.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.TOP
    fill_paragraphs(tf, title, 28, True, ink)
    if subtitle:
        text(slide, 0.5, 1.27, 12.33, 0.6, subtitle, size=14, color=soft, name="Subtitle")
    if not dark and source:
        text(slide, 0.5, 6.93, 11.4, 0.4, SOURCE, size=10, color="mut", name="Source line")
    slide_number(slide, 12.23, 6.98, "bg2" if dark else "mut")
    if question is not None:
        x = 12.83
        for i in reversed(range(3)):
            on = i == question
            w = 0.5 + 0.082 * len(QUESTIONS[i])
            x -= w
            pill = card(slide, x, 0.27, w, 0.34, fill="pri" if on else "bg2", name=f"Question {i + 1} marker")
            label_shape(pill, f"{i + 1}  {QUESTIONS[i]}", size=10, bold=on, color="bg1" if on else "mut")
            x -= 0.1
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


# ---------------------------------------------------------------- charts
def style_chart(chart, legend=True, title=None, size=11):
    chart.font.size = Pt(size)
    colour(chart.font.color, "sec")
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
    chart.has_title = title is not None
    if title:
        chart.chart_title.text_frame.text = title
        run = chart.chart_title.text_frame.paragraphs[0].runs[0]
        run.font.size = Pt(13)
        run.font.bold = True
        colour(run.font.color, "tx1")
        chart.chart_title.include_in_layout = False


def style_axes(chart, number_format="General", vmin=None, vmax=None, major=None, skip=None, value_title=None, category_title=None):
    va, ca = chart.value_axis, chart.category_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = GRID
    va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background()
    va.tick_labels.number_format = number_format
    va.tick_labels.number_format_is_linked = False
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    if major is not None:
        va.major_unit = major
    ca.has_major_gridlines = False
    ca.format.line.color.rgb = GRID
    ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    if value_title:
        va.has_title = True
        va.axis_title.text_frame.text = value_title
        run = va.axis_title.text_frame.paragraphs[0].runs[0]
        run.font.size = Pt(11)
        run.font.bold = False
        colour(run.font.color, "sec")
    if category_title:
        ca.has_title = True
        ca.axis_title.text_frame.text = category_title
        run = ca.axis_title.text_frame.paragraphs[0].runs[0]
        run.font.size = Pt(11)
        run.font.bold = False
        colour(run.font.color, "sec")
    if skip:
        el = ca._element
        node = etree.Element(qn("c:tickLblSkip"))
        node.set("val", str(skip))
        mark = etree.Element(qn("c:tickMarkSkip"))
        mark.set("val", str(skip))
        anchor = el.find(qn("c:noMultiLvlLbl"))
        if anchor is not None:
            anchor.addprevious(node)
            anchor.addprevious(mark)
        else:
            el.append(node)
            el.append(mark)


def line_chart(slide, x, y, w, h, cats, series, name, title=None, legend=True, **axes):
    """`series` is a list of (label, values, colour key, width in pt, dashed)."""
    data = CategoryChartData()
    data.categories = cats
    for label, values, *_ in series:
        data.add_series(label, values)
    frame = slide.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(x), Inches(y), Inches(w), Inches(h), data)
    frame.name = name
    chart = frame.chart
    style_chart(chart, legend=legend, title=title)
    style_axes(chart, **axes)
    for s, (_, _, key, width, dashed) in zip(chart.plots[0].series, series):
        s.smooth = False
        s.marker.style = XL_MARKER_STYLE.NONE
        colour(s.format.line.color, key)
        s.format.line.width = Pt(width)
        if dashed:
            s.format.line.dash_style = MSO_LINE.DASH
    return chart


def bar_chart(slide, x, y, w, h, cats, series, name, horizontal=True, title=None, legend=True, label_format='0.0',
              custom_labels=None, point_colors=None, gap=60, **axes):
    """`series` is a list of (label, values, colour key)."""
    data = CategoryChartData()
    data.categories = cats
    for label, values, _ in series:
        data.add_series(label, values)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    frame = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), data)
    frame.name = name
    chart = frame.chart
    style_chart(chart, legend=legend, title=title)
    style_axes(chart, **axes)
    plot = chart.plots[0]
    plot.gap_width = gap
    plot.overlap = -10
    plot.has_data_labels = True
    plot.data_labels.number_format = label_format
    plot.data_labels.number_format_is_linked = False
    plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    plot.data_labels.font.size = Pt(11)
    colour(plot.data_labels.font.color, "tx1")
    for si, (s, (_, _, key)) in enumerate(zip(plot.series, series)):
        s.format.fill.solid()
        colour(s.format.fill.fore_color, key)
        s.invert_if_negative = False
        if point_colors:
            for pi, pkey in enumerate(point_colors):
                pt = s.points[pi]
                pt.format.fill.solid()
                colour(pt.format.fill.fore_color, pkey)
        if custom_labels:
            for pi, label in enumerate(custom_labels[si]):
                tf = s.points[pi].data_label.text_frame
                tf.text = label
                run = tf.paragraphs[0].runs[0]
                run.font.size = Pt(11)
                colour(run.font.color, "tx1")
                s.points[pi].data_label.position = XL_LABEL_POSITION.OUTSIDE_END
    return chart


PENDING_LINKS: list[tuple[object, int]] = []  # (shape, 1-based slide number), resolved after all slides exist


def link(shapes, target: int) -> None:
    for item in shapes:
        PENDING_LINKS.append((item, target))


def table(slide, x, y, w, col_widths, header, rows, name, size=12, row_h=0.5):
    """A native table with explicit fills, so it follows the theme and not a built-in table style."""
    frame = slide.shapes.add_table(len(rows) + 1, len(header), Inches(x), Inches(y), Inches(w), Inches(row_h * (len(rows) + 1)))
    frame.name = name
    tbl = frame.table
    total = sum(col_widths)
    for ci, cw in enumerate(col_widths):
        tbl.columns[ci].width = Inches(w * cw / total)
    for ri, cells in enumerate([header, *rows]):
        tbl.rows[ri].height = Inches(row_h)
        for ci, value in enumerate(cells):
            cell = tbl.cell(ri, ci)
            cell.fill.solid()
            colour(cell.fill.fore_color, "pri" if ri == 0 else ("bg1" if ri % 2 else "bg2"))
            cell.margin_left = cell.margin_right = Inches(0.1)
            cell.margin_top = cell.margin_bottom = Inches(0.05)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            fill_paragraphs(tf, value, size, ri == 0 or ci == 0, "bg1" if ri == 0 else "tx1")
    return tbl


def pct(v: float, d: int = 0) -> str:
    return f"{v:.{d}f}%"


# ================================================================ title page
# The title as registered for the talk. The slide and the file's properties both use it.
TALK_TITLE = "When is electricity clean, and does shifting load pay off in CO₂?"


def s00():
    f = D["fan"]
    s = new_slide("Probabilistic forecasting and MLOps", TALK_TITLE, dark=True, notes=(
        "Good morning. Electricity is not equally clean at every hour, and that is the opportunity this project is about. "
        "I forecast, a day ahead and hour by hour, how much CO2 a kilowatt-hour will carry in Germany-Luxembourg and in France, "
        "with a 90% interval, and I run it as a tested pipeline on Databricks. The curve on the right is one real forecast."))
    title = s.shapes.title
    # Two lines at this size; the box ends just above the subtitle at 3.75.
    title.top, title.width, title.height = Inches(2.3), Inches(6.6), Inches(1.3)
    title.text_frame.paragraphs[0].runs[0].font.size = Pt(32)
    text(s, 0.5, 3.75, 6.3, 1.1, "A 24-hour forecast of grid carbon intensity for Germany-Luxembourg and France, "
         "with an interval you can check.", size=20, color="bg2", name="Subtitle")
    text(s, 0.5, 5.55, 6.3, 1.2, [
        [("Dr. Adewole Olagoke", {"bold": True, "color": "bg1", "size": 16})],
        "Project walkthrough" if INTERVIEW else "11 December 2026",
        "github.com/aolagoke-dsae/grid-carbon-intensity-mlops",
    ], size=13, color="bg2", spacing=3, name="Author, date and repository")
    chart = line_chart(s, 7.3, 1.6, 5.53, 4.3, f["hours"], [
        ("Upper bound (95%)", f["p95"], "bg2", 1.0, False),
        ("Median forecast", f["p50"], "de", 4.0, False),
        ("Lower bound (5%)", f["p05"], "bg2", 1.0, False),
        ("What happened", f["realized"], "bg1", 2.0, True),
    ], name="One real forecast (decorative)", legend=False, number_format="0", vmin=0)
    chart.value_axis.has_major_gridlines = False
    chart.value_axis.visible = False
    chart.category_axis.visible = False
    text(s, 7.3, 5.95, 5.53, 0.6, "One real forecast: DE-LU, issued 16 September 2025. Median (bold), 90% interval (thin) "
         "and what happened (dashed).", size=11, color="bg2", align="r", name="Chart caption")


# ================================================================ 1 the three questions
def s01():
    f = D["fan"]
    s = new_slide("Overview", "Three questions this talk answers",
                  "Each is answered with data the model never saw during development.", notes=(
        "The talk answers three questions, and I will come back to them at the end. First, can we forecast carbon intensity a "
        "day ahead better than simply assuming no change? Second, can we trust the interval: does a 90% interval really contain "
        "90% of what happens? Third, does acting on the forecast pay: does a load that follows it actually emit less? "
        "The chart is one real forecast from the held-out period, with the cleanest three hours it would have recommended."))
    details = ["A day ahead, hour by hour, better than assuming no change?",
               "Does the 90% interval really contain 90% of what happens?",
               "Does a load that follows the forecast emit less CO\u2082?"]
    for i, (q, d) in enumerate(zip(QUESTIONS, details)):
        y = 2.2 + 1.45 * i
        badge(s, 0.5, y, 0.7, str(i + 1), size=24)
        text(s, 1.45, y - 0.05, 4.9, 0.45, q, size=22, bold=True, name=f"Question {i + 1}")
        text(s, 1.45, y + 0.42, 4.9, 0.6, d, size=14, color="sec", name=f"Question {i + 1} detail")
    line_chart(s, 6.8, 2.0, 6.03, 4.75, f["hours"], [
        ("Upper bound (95%)", f["p95"], "de", 1.0, False),
        ("Median forecast", f["p50"], "de", 3.0, False),
        ("Lower bound (5%)", f["p05"], "de", 1.0, False),
        ("What happened", f["realized"], "tx1", 2.0, True),
    ], name="One real forecast", title="One real forecast: DE-LU, 16 Sep 2025",
        number_format="0", vmin=0, skip=6, value_title="gCO\u2082eq/kWh")


# ================================================================ 2 problem
def s02():
    p = D["problem"]
    s = new_slide("Motivation", f"In Germany the same kWh emits {p['ratio']['DE-LU']:.1f} times as much CO₂ at the wrong hour",
                  "Flexible loads (vehicle fleets, heat pumps, cold stores, batch computing) can choose their hours, if told a day ahead.", notes=(
        "These are average days from two years of data. In Germany-Luxembourg in summer, solar pushes midday intensity to "
        f"less than half the night-time level. On a typical day the dirtiest hour emits about {p['ratio']['DE-LU']:.1f} times as much as the cleanest. "
        "France is nuclear-dominated: it is more than ten times cleaner on average and its daily swing is small. "
        "So the opportunity is mainly in Germany. Note that each chart has its own scale."))
    for i, z in enumerate(("DE-LU", "FR")):
        line_chart(s, 0.5 + 4.2 * i, 2.05, 4.05, 4.2, p["hours"], [
            ("Summer (Jun–Aug)", p["profiles"][z]["summer"], ZL[z], 3.0, False),
            ("Winter (Dec–Feb)", p["profiles"][z]["winter"], ZL[z], 2.25, True),
        ], name=f"Average day {z}", title=f"{z}: carbon intensity", number_format="0", vmin=0, skip=6,
            category_title="Hour of day (UTC)")
    text(s, 0.5, 6.28, 8.4, 0.6, [[("Carbon intensity: ", {"bold": True, "color": "tx1"}),
                                   ("grams of CO₂-equivalent emitted for each kWh of electricity generated (gCO₂eq/kWh). "
                                    "DE-LU is the Germany-Luxembourg market zone, FR is France. Average day, 2024–2025; each grid on its own scale.", {})]], size=12, color="sec", name="Definition of carbon intensity")
    stats = [
        (f"{p['ratio']['DE-LU']:.1f}×", "dirtiest against cleanest hour of a typical day in DE-LU", f"FR: {p['ratio']['FR']:.1f}×", "de"),
        (pct(p["below"]["DE-LU"]), "how far the cleanest hour sits below the day's average in DE-LU", f"FR: {pct(p['below']['FR'])}", "de"),
        (f"{p['mean']['DE-LU']:.0f} vs {p['mean']['FR']:.0f}", "mean gCO₂eq/kWh, DE-LU against FR, 2024–2025", "", "tx1"),
    ]
    for i, (big, label, note, key) in enumerate(stats):
        y = 2.05 + 1.6 * i
        card(s, 9.1, y, 3.73, 1.5, name=f"Stat card {i + 1}")
        text(s, 9.3, y + 0.1, 3.3, 0.6, big, size=32, bold=True, color=key, name=f"Stat {i + 1}")
        text(s, 9.3, y + 0.72, 3.35, 0.5, label, size=12, name=f"Stat {i + 1} label")
        if note:
            text(s, 9.3, y + 1.17, 3.3, 0.25, note, size=11, color="sec", name=f"Stat {i + 1} note")


# ================================================================ 3 lineage
def s03():
    ln = D["lineage"]
    s = new_slide("Data introduction: source" if INTERVIEW else "Backup: data source", "From ENTSO-E to forecast: one hash chain across two repositories",
                  "The upstream project pulls and derives the data. This project starts from its export, pinned by hash, "
                  "so any forecast traces back to a raw response.", notes=(
        "The data is ENTSO-E's generation by fuel type. A separate, public upstream project pulls it through the API, keeps "
        "every raw response unchanged with its SHA-256, and derives carbon intensity in DuckDB and dbt. Only two Parquet files "
        "cross into this project. The hourly snapshot built from them is pinned by hash: if the file changes, the pipeline "
        "refuses to run. Follow the chain at the bottom: any forecast can be traced back to the raw response it came from. "
        "DuckDB, the API token, the raw responses and the emission-factor uncertainty all stay upstream by design."))
    steps = [
        ("Pull", "up", "ENTSO-E API: actual generation by fuel type (A75), DE-LU and FR"),
        ("Land", "up", "Raw responses kept unchanged; SHA-256 of each in a manifest"),
        ("Transform", "up", "DuckDB + dbt. Carbon intensity = generation × emission factors"),
        ("Export", "up", "Two Parquet marts: 4.2 million generation rows, 246,310 carbon-intensity rows"),
        ("Snapshot", "pri", f"One row per zone and hour, 2024–2025: {ln['snapshot_rows']:,} rows; {ln['partial_hours']} incomplete hours flagged"),
        ("Ingest (G1)", "pri", "Snapshot hash is pinned: a changed file stops the run"),
        ("Use", "pri", "Features, forecast, monitoring; same file locally and on Databricks"),
    ]
    w, gap, bgap, x0, y0, h = 1.6, 0.12, 0.33, 0.5, 2.6, 1.85
    xs, x = [], x0
    for i in range(7):
        xs.append(x)
        x += w + (bgap if i == 3 else gap)
    card(s, xs[0] - 0.12, y0 - 0.5, xs[3] + w - xs[0] + 0.24, h + 0.64, fill="bg2", name="Upstream repository region")
    card(s, xs[4] - 0.12, y0 - 0.5, xs[6] + w - xs[4] + 0.24, h + 0.64, fill="bg2", name="This project region")
    text(s, xs[0], y0 - 0.42, 6.5, 0.3, [[("UPSTREAM REPOSITORY", {"bold": True, "color": "up"}),
                                           ("   grid-carbon-intensity · 2021–2025 · commit 7caaee3", {"color": "sec"})]],
         size=11, name="Upstream label")
    text(s, xs[4], y0 - 0.42, 5.0, 0.3, [[("THIS PROJECT", {"bold": True, "color": "pri"}),
                                           ("   grid-carbon-intensity-mlops · 2024–2025", {"color": "sec"})]],
         size=11, name="Project label")
    for i, ((title, key, body), bx) in enumerate(zip(steps, xs)):
        card(s, bx, y0, w, h, fill="bg1", line=key, name=f"Step {i + 1} box")
        badge(s, bx + 0.1, y0 + 0.12, 0.34, str(i + 1), fill=key, size=12)
        text(s, bx + 0.5, y0 + 0.12, w - 0.55, 0.34, title, size=14, bold=True, anchor="middle", name=f"Step {i + 1} title")
        text(s, bx + 0.1, y0 + 0.62, w - 0.2, h - 0.7, body, size=11, name=f"Step {i + 1} text")
        if i < 6:
            arrow(s, bx + w + 0.01, y0 + h / 2, xs[i + 1] - 0.01, y0 + h / 2, color="tx1", width=2.25 if i == 3 else 1.25)
    mid = (xs[3] + w + xs[4]) / 2
    text(s, mid - 1.6, y0 - 0.86, 3.2, 0.28, "Only data crosses: two Parquet files", size=12, bold=True, align="c", name="Boundary label")

    yc = y0 + h + 0.55
    chain = {1: "SHA-256 of each raw response", 3: "source_sha256 on every row", 4: "upstream_payload_sha256 per hour",
             5: "snapshot SHA-256 pinned", 6: "source_file_sha256 on every row"}
    line = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(xs[1] + w / 2), Inches(yc), Inches(xs[6] + w / 2), Inches(yc))
    line.name = "Hash chain line"
    colour(line.line.color, "tx1")
    line.line.width = Pt(2)
    text(s, 0.5, yc - 0.17, 1.6, 0.34, "The hash chain", size=13, bold=True, anchor="middle", name="Hash chain label")
    for idx, label in chain.items():
        cx = xs[idx] + w / 2
        dot = shape(s, MSO_SHAPE.OVAL, cx - 0.08, yc - 0.08, 0.16, 0.16, fill="bg1", line="tx1", name="Hash chain point", line_width=2)
        del dot
        text(s, cx - 0.8, yc + 0.16, 1.6, 0.45, label, size=10, align="c", name="Hash chain item")
    text(s, 0.5, 6.2, 12.33, 0.3, [[("Stays upstream, by design:  ", {"bold": True}),
                                     ("the API token · the raw responses · DuckDB and dbt · the emission-factor uncertainty · "
                                      "prices, load and flows", {"color": "sec"})]], size=12, name="Stays upstream")


# ================================================================ 4 method
def s04():
    m = D["method"]
    s = new_slide("ML model", "Predict the change from now, then calibrate on last month's errors",
                  "Four blocks in strict time order. The last one was opened once, after the analysis plan was committed.",
                  question=0, notes=(
        "The data is cut into four blocks in time order, so nothing from the future leaks backwards. The model is trained on "
        "the first block. The second supplies the first forecast errors for calibration. All tuning used the third. The fourth "
        "was opened exactly once, under a plan committed beforehand. The model predicts the change from the current intensity, "
        "so 'no change' is its natural baseline. The interval is then corrected with conformal calibration: every week, the "
        f"offsets are recomputed from the last {WINDOW} days of real errors, separately for each zone and lead time. A new model only "
        "replaces the champion if it passes the gate."))
    roles = ["fits the model", "first forecast errors", "tuning and gate", "opened once"]
    total = sum(b["rows"] for b in m["splits"])
    x = 0.5
    for i, (b, role) in enumerate(zip(m["splits"], roles)):
        w = 12.33 * b["rows"] / total
        last = i == 3
        bar = shape(s, MSO_SHAPE.RECTANGLE, x, 2.0, w - 0.05, 0.5, fill="pri" if last else "bg2", name=f"{b['name']} block")
        label_shape(bar, b["name"], size=13, bold=True, color="bg1" if last else "tx1", align="l")
        when = f"from {b['start']}" if not last else f"{b['start'][:6]} to {b['end']}"
        text(s, x + 0.03, 2.56, max(w - 0.1, 1.75), 0.75, [f"{b['rows']:,} rows", when, role], size=11, color="sec",
             name=f"{b['name']} details")
        x += w
    steps = [
        ("Features", f"Only what is known at the moment of forecasting, plus the calendar of the target hour. {m['feature_rows']:,} rows."),
        ("Model", "A tree-based model (LightGBM) predicts a low, a middle and a high value for the change from the current carbon intensity."),
        ("Calibrate", f"The interval is widened or narrowed using the last {m['window_days']} days of real errors, per zone and lead time, every {m['step_days']} days."),
        ("Gate", f"Promote only if coverage is {m['band'][0]}–{m['band'][1]}% in each zone, it beats “no change”, and it is no worse than the model in use."),
    ]
    w, gap, y0, h = 2.86, 0.297, 3.75, 2.35
    for i, (title, body) in enumerate(steps):
        bx = 0.5 + i * (w + gap)
        card(s, bx, y0, w, h, fill="bg1", line="pri", name=f"Method step {i + 1}")
        badge(s, bx + 0.18, y0 + 0.18, 0.44, str(i + 1), size=15)
        text(s, bx + 0.75, y0 + 0.18, w - 0.9, 0.44, title, size=18, bold=True, anchor="middle", name=f"Method step {i + 1} title")
        text(s, bx + 0.2, y0 + 0.85, w - 0.4, h - 1.0, body, size=14, name=f"Method step {i + 1} text")
        if i < 3:
            arrow(s, bx + w + 0.03, y0 + h / 2, bx + w + gap - 0.03, y0 + h / 2, color="tx1")
    text(s, 0.5, 6.3, 12.33, 0.3, [[("Lead time: ", {"bold": True, "color": "tx1"}), ("how many hours ahead a forecast looks, from 1 to 24. ", {}),
                                    ("Low, middle, high: ", {"bold": True, "color": "tx1"}),
                                    ("the 5%, 50% and 95% points; the middle is the median forecast. ", {}),
                                    (f"Model: {m['trees']} trees, frozen before the held-out test.", {})]], size=12, color="sec",
         name="Model note")


# ================================================================ 5 trust
def s05():
    t = D["trust"]
    s = new_slide("Error analysis: calibration",
                  f"On unseen data the 90% interval covered {t['coverage']['DE-LU']:.1f}% (DE-LU) and {t['coverage']['FR']:.1f}% (FR)",
                  "Held-out test, 13 September to 31 December 2025, replayed week by week exactly as it would have run.", question=1, notes=(
        "This is the central result. The interval is supposed to contain 90% of outcomes; on data the model never saw it "
        f"contained {t['coverage']['DE-LU']:.1f}% in Germany-Luxembourg and {t['coverage']['FR']:.1f}% in France. "
        f"The 95% intervals on those figures are {t['ci']['DE-LU'][0]} to {t['ci']['DE-LU'][1]} and {t['ci']['FR'][0]} to {t['ci']['FR'][1]}. "
        "On the left, coverage by lead time: Germany holds at every lead time, France drifts down beyond 12 hours. "
        "On the right, coverage week by week: single weeks leave the band often, because a week of hourly forecasts is strongly "
        "correlated. The average holds; one week is noisy. I name the weak spots rather than hide them."))
    n = len(t["lead_hours"])
    line_chart(s, 0.5, 2.0, 5.9, 3.55, [str(h) for h in t["lead_hours"]], [
        ("DE-LU", t["lead"]["DE-LU"], "de", 2.5, False),
        ("FR", t["lead"]["FR"], "fr", 2.5, False),
        ("Target 90%", [0.90] * n, "sec", 1.0, True),
    ], name="Coverage by lead time", title="Coverage by lead time (hours ahead)", number_format="0%", vmin=0.80, vmax=0.95, major=0.05, skip=3)
    nw = len(t["weeks"])
    line_chart(s, 6.93, 2.0, 5.9, 3.55, t["weeks"], [
        ("DE-LU", t["weekly"]["DE-LU"], "de", 2.5, False),
        ("FR", t["weekly"]["FR"], "fr", 2.5, False),
        ("Lower limit 87%", [0.87] * nw, "mut", 1.0, True),
        ("Upper limit 96%", [0.96] * nw, "mut", 1.0, True),
    ], name="Coverage week by week", title="Coverage week by week", number_format="0%", vmin=0.75, vmax=1.0, major=0.05, skip=4)
    notes = [
        ("Where it holds", "Every lead time in DE-LU; 1 to 12 hours in FR."),
        ("Where it does not", f"FR at 13–24 hours: {t['fr_day_ahead']}%. DE-LU with the most renewables: {t['de_top_renewables']}%."),
        ("Week to week", f"Outside the band in {t['weeks_outside']['DE-LU']} of {t['n_weeks']} weeks (DE-LU) and {t['weeks_outside']['FR']} (FR). One week is noisy."),
    ]
    for i, (head, body) in enumerate(notes):
        bx = 0.5 + i * 4.177
        card(s, bx, 5.7, 3.977, 1.08, name=f"Note card {i + 1}")
        text(s, bx + 0.18, 5.78, 3.6, 0.3, head, size=13, bold=True, name=f"Note {i + 1} title")
        text(s, bx + 0.18, 6.08, 3.65, 0.65, body, size=12, name=f"Note {i + 1} text")


# ================================================================ 6 hypotheses
def s06():
    h = D["hypotheses"]
    s = new_slide("Hypotheses: outcome", "Three hypotheses, written down before the test: all three falsified",
                  "Each was evaluated once, under an analysis plan committed to the repository before the held-out data was opened.",
                  question=1, notes=(
        "I registered three hypotheses before looking at the held-out data, and all three failed. "
        f"H1: with the most renewables Germany is indeed cleaner, by about {h['h1_cleaner_by']:.0f} grams, but the interval was narrower, not wider. "
        f"H2: Germany was never cleaner than France at midday, on {h['h2_days_cleaner']} of {h['h2_days']} qualifying days; the closest gap was {h['h2_closest']} grams. "
        "H3: I expected the calendar to overtake the current state of the grid a day ahead. The current state stays several "
        "times more important at every lead time; its lead shrinks but never reverses. "
        "I report them as registered. A project that only confirms its own expectations has usually not tested them."))
    cards = [
        ("H1", "High renewables", "In DE-LU, hours with the most renewables are cleaner and have a wider interval.",
         f"Cleaner: yes, by {h['h1_cleaner_by']:.0f} g CO₂/kWh. Wider: no, the interval was {h['h1_narrower_by']:.0f} narrower."),
        ("H2", "Midday parity", "On sunny, windy days DE-LU is cleaner than FR around midday.",
         f"DE-LU was cleaner on {h['h2_days_cleaner']} of {h['h2_days']} such days. Closest gap: {h['h2_closest']} g CO₂/kWh."),
        ("H3", "What drives the forecast", "A day ahead, the calendar matters more than the current state of the grid.",
         "The current state dominates at every lead time. Its lead shrinks but never reverses."),
    ]
    w, gap, y0, hh = 3.977, 0.2, 2.0, 4.78
    for i, (code, name, claim, found) in enumerate(cards):
        bx = 0.5 + i * (w + gap)
        card(s, bx, y0, w, hh, fill="bg1", line="mut", name=f"{code} card")
        text(s, bx + 0.2, y0 + 0.12, w - 0.4, 0.4, [[(code + "  ", {"color": "pri", "size": 20}), (name, {"size": 16})]], size=16, bold=True,
             anchor="middle", name=f"{code} heading")
        text(s, bx + 0.2, y0 + 0.6, w - 0.4, 0.55, claim, size=12, color="sec", name=f"{code} claim")
        text(s, bx + 0.2, y0 + 3.55, w - 0.4, 0.6, found, size=12, name=f"{code} finding")
        pill = card(s, bx + 0.2, y0 + 4.25, 1.3, 0.36, fill="tx1", name=f"{code} verdict")
        label_shape(pill, "FALSIFIED", size=11, bold=True, color="bg1")
    cy, ch = y0 + 1.2, 2.3
    bar_chart(s, 0.5 + 0.1, cy, w - 0.2, ch, ["Fewest renewables", "Most renewables"],
              [("Interval width, gCO₂eq/kWh", [h["h1_width_bottom"], h["h1_width_top"]], "de")], name="H1 chart", horizontal=False,
              legend=False, title="Width of the 90% interval", label_format="0", number_format="0", vmin=0, vmax=320, major=80, gap=90,
              value_title="g CO₂ per kWh")
    bar_chart(s, 0.5 + (w + gap) + 0.1, cy, w - 0.2, ch, ["DE-LU", "FR"],
              [("Midday mean, gCO₂eq/kWh", [h["h2_de"], h["h2_fr"]], "de")], name="H2 chart", horizontal=False, legend=False,
              title="Midday carbon intensity", label_format="0", number_format="0", vmin=0, vmax=240, major=60, gap=90, value_title="g CO₂ per kWh",
              point_colors=["de", "fr"])
    line_chart(s, 0.5 + 2 * (w + gap) + 0.1, cy, w - 0.2, ch, ["1–6 h ahead", "13–24 h ahead"], [
        ("DE-LU", h["h3"]["DE-LU"], "de", 2.5, False),
        ("FR", h["h3"]["FR"], "fr", 2.5, False),
        ("Reversal below 1", [1, 1], "tx1", 1.0, True),
    ], name="H3 chart", title="Current state vs calendar (ratio)", number_format="0", vmin=0, vmax=8, major=2)


# ================================================================ 7 value
def s07():
    v = D["value"]
    names = {"yesterday": "Repeat yesterday", "cautious": "Forecast, upper bound", "median": "Forecast, median", "hindsight": "Hindsight (best possible)"}
    s = new_slide("Results: business value", "Following the forecast avoided more CO₂ and backfired a third as often",
                  f"Each day one 3-hour window is chosen at 00:00 UTC and compared with running at an arbitrary time. "
                  f"Held-out test, {v['n_days']} days.", question=2, notes=(
        "Does it pay to act on the forecast? Each day I pick one three-hour window and compare what it emitted with the day's "
        "average. Intensity in grams per kilowatt-hour is the same number as kilograms per megawatt-hour, so the difference is "
        f"CO2 avoided per megawatt-hour moved. A load of one megawatt-hour a day in Germany would have avoided {v['tonnes']['DE-LU'][2]} tonnes "
        f"over the {v['n_days']} days, {v['captured']['DE-LU']:.0f}% of what perfect hindsight allows. In France the grid is already "
        f"clean, so timing is worth only {v['tonnes']['FR'][2]} tonnes. The stronger argument is on the right: the forecast "
        f"backfired on {v['bad']['DE-LU'][2]:.0f}% of days against {v['bad']['DE-LU'][0]:.0f}% for the rule of thumb. "
        "Two caveats: choosing by the upper bound did not help, and I do not scale this to a year or to euros."))
    order = list(reversed(v["order"]))
    idx = [v["order"].index(o) for o in order]
    bar_chart(s, 0.5, 2.0, 6.0, 4.05, [names[o] for o in order], [
        ("FR", [v["tonnes"]["FR"][i] for i in idx], "fr"),
        ("DE-LU", [v["tonnes"]["DE-LU"][i] for i in idx], "de"),
    ], name="What it was worth", title=f"Tonnes CO₂ avoided, 1 MWh per day over {v['n_days']} days", label_format='0.0" t"',
        number_format="0", vmin=0, vmax=10, major=2)
    order2 = list(reversed(v["order"][:3]))
    idx2 = [v["order"].index(o) for o in order2]
    labels = [[f"{v['bad'][z][i]:.0f}% (worst {v['worst'][z][i]:.0f}%)" for i in idx2] for z in ("FR", "DE-LU")]
    bar_chart(s, 6.83, 2.0, 6.0, 4.05, [names[o] for o in order2], [
        ("FR", [v["bad"]["FR"][i] / 100 for i in idx2], "fr"),
        ("DE-LU", [v["bad"]["DE-LU"][i] / 100 for i in idx2], "de"),
    ], name="How often it backfired", title="Share of days dirtier than average (worst day)",
        custom_labels=labels, number_format="0%", vmin=0, vmax=0.5, major=0.1)
    card(s, 0.5, 6.15, 12.33, 0.65, name="Takeaway card")
    text(s, 0.7, 6.15, 11.95, 0.65, [[
        (f"DE-LU: {v['tonnes']['DE-LU'][2]} t avoided, {v['captured']['DE-LU']:.0f}% of hindsight. ", {"bold": True}),
        (f"FR: {v['tonnes']['FR'][2]} t, the grid is already clean. Choosing by the upper bound did not help. Not scaled to a year; no euros.", {})]],
        size=13, anchor="middle", name="Takeaway")


# ================================================================ 8 mlops
def s08():
    m = D["mlops"]
    s = new_slide("MLOps on Databricks" if INTERVIEW else "Backup: MLOps on Databricks", "Every claim is enforced by a gate, a test or a deliberate breakage",
                  "Two pipeline jobs, deployed as one Databricks Asset Bundle on serverless compute; models and tables governed in Unity Catalog.",
                  notes=(
        "Two pipeline jobs run on Databricks. The training job ingests, builds features, trains a challenger and puts it through the "
        "gate. The inference job recalibrates the champion on the latest errors, then scores the next 24 hours, computes "
        "monitoring and writes the dashboard extracts. A third, small job checks the question-answering agent. Three gates guard the pipeline: the data file's hash is pinned, no "
        "feature may come from after the moment of forecasting, and a model is promoted only if it passes the coverage, skill "
        f"and champion checks. There are {m['tests']} tests. To know the tests themselves are any good, a script breaks the code "
        f"on purpose, one line at a time, {m['breakages']} ways, and every one of those breakages is caught."))

    def node(x, y, w, label, name):
        b = card(s, x, y, w, 0.6, fill="bg1", line="pri", name=name)
        label_shape(b, label, size=13)

    text(s, 0.5, 2.0, 7.4, 0.32, [[("Training job  ", {"bold": True, "size": 16}), ("run when the model changes", {"color": "sec", "size": 12})]],
         size=16, name="Training job label")
    xs = [0.5, 2.4, 4.3, 6.2]
    for x, t in zip(xs, ["Ingest", "Build features", "Train challenger", "Gate"]):
        node(x, 2.45, 1.6, t, f"Training: {t}")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(s, a + 1.62, 2.75, b - 0.02, 2.75, color="tx1", width=1.25)
    for x, t in zip([0.5, 2.4, 6.2], ["G1: file hash pinned", "G2: nothing from after the forecast moment", "G3: coverage, skill, champion"]):
        text(s, x, 3.1, 1.75, 0.5, t, size=11, color="sec", name="Gate note")

    text(s, 0.5, 3.85, 7.4, 0.32, [[("Inference job  ", {"bold": True, "size": 16}), ("on a schedule (paused in the dev target)", {"color": "sec", "size": 12})]],
         size=16, name="Inference job label")
    node(0.5, 4.95, 1.9, "Recalibrate champion", "Inference: recalibrate")
    for k, t in enumerate(["Score the next 24 hours", "Monitor coverage and drift", "Write dashboard extracts"]):
        y = 4.3 + 0.72 * k
        node(3.4, y, 3.0, t, f"Inference: {t}")
        arrow(s, 2.42, 5.25, 3.38, y + 0.3, color="tx1", width=1.25)
    text(s, 0.5, 6.48, 7.6, 0.3, f"Verified on the workspace: every task has succeeded; the champion is model version {m['champion_version']}.",
         size=12, color="sec", name="Workspace note")

    for i, (big, label) in enumerate([(str(m["tests"]), "automated tests, about 13 seconds"),
                                      (f"{m['breakages']} of {m['breakages']}", "deliberate breakages caught")]):
        bx = 8.4 + i * 2.265
        card(s, bx, 2.0, 2.165, 1.55, name=f"Stat card {i + 1}")
        text(s, bx + 0.18, 2.1, 1.85, 0.6, big, size=30, bold=True, color="pri", name=f"Stat {i + 1}")
        text(s, bx + 0.18, 2.78, 1.85, 0.7, label, size=12, name=f"Stat {i + 1} label")
    card(s, 8.4, 3.7, 4.43, 2.6, name="Breakage explanation card")
    text(s, 8.6, 3.82, 4.05, 0.32, "What a “deliberate breakage” is", size=14, bold=True, name="Breakage heading")
    text(s, 8.6, 4.2, 4.05, 2.5, [
        "A script breaks the code on purpose, one line at a time, and reruns the tests. For example:",
        "– let a feature look one hour ahead",
        "– count coverage on one bound only",
        "– promote a model that failed the gate",
        "A breakage the tests miss is a gap in the tests.",
    ], size=12, spacing=4, name="Breakage text")
    back_link(s)


# ================================================================ 9 limits
def s09():
    lim, t = D["limits"], D["trust"]
    s = new_slide("Backup: monitoring and limits", "What monitoring shows, and what is not solved",
                  "The same weekly series that tell whether the model is healthy also show where it is weakest.", notes=(
        "Monitoring tells me where the model is weak, and I would rather say it than have it found. On the left: skill against "
        f"'no change' fades from summer to winter, with weeks as low as {lim['lowest_skill']['DE-LU']:.0f}% and {lim['lowest_skill']['FR']:.0f}%. "
        "The model is not yet retrained on a schedule. On the right: the value of following the forecast in Germany. In summer "
        "the rule of thumb does just as well, because the solar pattern repeats. From October the forecast pulls ahead. "
        "Four limits are listed at the bottom; the first is the one I would fix next."))
    line_chart(s, 0.5, 2.0, 5.9, 3.3, lim["weeks"], [
        ("DE-LU", lim["skill"]["DE-LU"], "de", 2.5, False),
        ("FR", lim["skill"]["FR"], "fr", 2.5, False),
    ], name="Skill week by week", title="Skill against “no change”, week by week", number_format="0%", vmin=-0.6, vmax=0.8, major=0.2, skip=4)
    line_chart(s, 6.93, 2.0, 5.9, 3.3, lim["value_weeks"], [
        ("Forecast", lim["value"]["median"], "de", 2.5, False),
        ("Repeat yesterday", lim["value"]["yesterday"], "mut", 2.0, True),
    ], name="Value week by week", title="DE-LU: kg CO₂ avoided per MWh moved", number_format="0", vmin=-50, vmax=250, major=50, skip=4)
    items = [
        ("Skill fades towards winter", f"Lowest weeks: {lim['lowest_skill']['DE-LU']:.0f}% (DE-LU), {lim['lowest_skill']['FR']:.0f}% (FR). No retraining schedule yet."),
        ("The upper bound picks no better", f"By the upper bound it backfired on {lim['cautious_bad']:.0f}% of days; by the median, {lim['median_bad']:.0f}%."),
        ("France a day ahead", f"Coverage at 13–24 hours is {t['fr_day_ahead']}%, below the accepted band."),
        ("Two platforms, two fits", f"Trained on Databricks, forecasts differ by about {F['platform_diff_pct']}% in DE-LU. Conclusions agree."),
    ]
    w, gap = 2.93, 0.203
    for i, (head, body) in enumerate(items):
        bx = 0.5 + i * (w + gap)
        card(s, bx, 5.45, w, 1.33, name=f"Limit card {i + 1}")
        text(s, bx + 0.15, 5.52, w - 0.3, 0.32, head, size=13, bold=True, name=f"Limit {i + 1} title")
        text(s, bx + 0.15, 5.86, w - 0.3, 0.9, body, size=12, name=f"Limit {i + 1} text")
    back_link(s)


# ================================================================ 10 summary
def s10():
    t, v, sm = D["trust"], D["value"], D["summary"]
    s = new_slide("Summary and next steps", "Three questions, three answers", dark=True, notes=(
        "To close. Can we forecast it? Yes: the median forecast's error is "
        f"{sm['skill']['DE-LU']:.0f}% lower than 'no change' in Germany and {sm['skill']['FR']:.0f}% lower in France. "
        "Can we trust the interval? Mostly: it covers close to the 90% it promises, and the weak spots are named. "
        f"Does acting on it pay? In Germany, yes: {v['tonnes']['DE-LU'][2]} tonnes of CO2 avoided per daily megawatt-hour over the test period, "
        "and it backfires a third as often as the rule of thumb. Next come a retraining schedule, the two dashboards, and "
        "live data for the question-answering agent, which already runs on Databricks. Thank you."))
    text(s, 0.5, 1.27, 12.33, 0.4, "All figures from the held-out test, 13 September to 31 December 2025.", size=14, color="bg2", name="Subtitle")
    answers = [
        (QUESTIONS[0], "Yes", f"Median error {sm['skill']['DE-LU']:.0f}% lower than “no change” in DE-LU and {sm['skill']['FR']:.0f}% lower in FR. The edge narrows in winter."),
        (QUESTIONS[1], "Mostly", f"{t['coverage']['DE-LU']:.1f}% and {t['coverage']['FR']:.1f}% covered against a target of 90%. Weak spots: France a day ahead, very high renewables."),
        (QUESTIONS[2], "In Germany", f"{v['tonnes']['DE-LU'][2]} t CO₂ avoided per daily MWh in DE-LU, {v['tonnes']['FR'][2]} t in FR. It backfires a third as often as the rule of thumb."),
    ]
    w, gap, y0, h = 3.977, 0.2, 2.0, 3.15
    for i, (q, a, body) in enumerate(answers):
        bx = 0.5 + i * (w + gap)
        card(s, bx, y0, w, h, fill="bg1", name=f"Answer card {i + 1}")
        badge(s, bx + 0.22, y0 + 0.22, 0.46, str(i + 1), size=16)
        text(s, bx + 0.82, y0 + 0.22, w - 1.0, 0.46, q, size=16, bold=True, anchor="middle", name=f"Answer {i + 1} question")
        text(s, bx + 0.22, y0 + 0.9, w - 0.44, 0.85, a, size=40, bold=True, color="pri", anchor="middle", name=f"Answer {i + 1}")
        text(s, bx + 0.22, y0 + 1.95, w - 0.44, 1.5, body, size=14, name=f"Answer {i + 1} detail")
    text(s, 0.5, 5.55, 12.33, 0.9, [
        [("Next  ", {"bold": True, "color": "bg1"}), ("a retraining schedule · the two dashboards · live data for the question-answering agent", {})],
        [("Code  ", {"bold": True, "color": "bg1"}), ("github.com/aolagoke-dsae/grid-carbon-intensity-mlops · upstream: github.com/aolagoke-dsae/grid-carbon-intensity", {})],
        [(SOURCE, {"size": 10})],
    ], size=13, color="bg2", spacing=5, name="Next steps and source")


# ================================================================ closing page
def s11():
    s = new_slide("Questions", "Thank you. Questions?", dark=True, notes=(
        "Thank you. I am happy to take questions. If it helps: the code, the analysis plan committed before the test, the "
        "full results including the three falsified hypotheses, and the register of every issue found in review are all in the "
        "public repository. In slide-show mode, each of the four cards jumps back to that part of the talk."))
    title = s.shapes.title
    title.top, title.width, title.height = Inches(1.9), Inches(12.33), Inches(1.2)
    title.text_frame.paragraphs[0].runs[0].font.size = Pt(54)
    text(s, 0.5, 3.2, 11.5, 0.6, "Each card below is a link back to that part of the talk.", size=20, color="bg2",
         name="Invitation")
    topics = [("Data", "ENTSO-E to snapshot, pinned by hash", NUM["s03"] if INTERVIEW else NUM["s_data"]), ("Method", "quantiles, conformal calibration, the gate", NUM["s04"]),
              ("Results", "coverage, hypotheses, CO\u2082 avoided", NUM["s05"]), ("Engineering", "tests, breakages, Databricks jobs", NUM["s08"])]
    w, gap = 2.93, 0.203
    for i, (head, body, target) in enumerate(topics):
        bx = 0.5 + i * (w + gap)
        parts = [
            card(s, bx, 4.2, w, 1.25, fill="bg1", name=f"Topic card {i + 1} (link to slide {target})"),
            text(s, bx + 0.18, 4.28, w - 0.36, 0.36, [[(head, {}), (f"   slide {target} ›", {"size": 12, "bold": False, "color": "sec"})]],
                 size=16, bold=True, color="pri", name=f"Topic {i + 1}"),
            text(s, bx + 0.18, 4.68, w - 0.36, 0.7, body, size=13, name=f"Topic {i + 1} detail"),
        ]
        link(parts, target)  # the text boxes sit on top of the card, so all three carry the link
    more = text(s, 0.5, 5.62, 6.0, 0.3, "More detail: backup slides \u203a", size=13, bold=True, color="bg1", name="Link to backup slides")
    link([more], NUM["b00"])
    text(s, 0.5, 6.1, 12.33, 1.0, [
        [("Dr. Adewole Olagoke", {"bold": True, "color": "bg1", "size": 15})],
        "github.com/aolagoke-dsae/grid-carbon-intensity-mlops  \u00b7  upstream data: github.com/aolagoke-dsae/grid-carbon-intensity",
        [(SOURCE, {"size": 10})],
    ], size=13, color="bg2", spacing=4, name="Contact and source")


PROFILE_URL = "https://github.com/aolagoke-dsae"


# ================================================================ template: about me
def s_about():
    s = new_slide("About me", "Dr. Adewole Olagoke",
                  "A quantitative ecologist moving into data science and analytics.", source=False, notes=(
        "A word about me. My doctorate is in quantitative ecology and ecological modelling, and I have spent more than ten years "
        "on statistical modelling, time series and uncertainty analysis. I coordinated data harmonisation and quality assurance "
        "across an EU consortium of 19 partners in 10 countries. I am now looking for data science, analytics engineering and "
        "environmental data roles in Germany, from January 2027. This project is the kind of work I want to do: a model whose "
        "uncertainty is checked, inside a pipeline that is tested."))
    photo_file = Path("dashboards/design/profile_photo.jpg")
    if photo_file.exists():
        pic = s.shapes.add_picture(str(photo_file), Inches(0.75), Inches(1.95), Inches(2.1), Inches(2.1))
        pic.name = "Profile photo"
        geom = pic._element.spPr.find(qn("a:prstGeom"))
        geom.set("prst", "ellipse")  # a round frame; the picture itself stays square and uncropped in the file
        colour(pic.line.color, "mut")
        pic.line.width = Pt(1.25)
    else:
        photo = shape(s, MSO_SHAPE.OVAL, 0.75, 1.95, 2.1, 2.1, fill="bg2", line="mut", name="Photo placeholder (replace with a picture)")
        label_shape(photo, "Photo", size=14, color="mut")
    under_photo = "Data scientist" if INTERVIEW else "Data Analytics track"
    text(s, 0.5, 4.15, 2.6, 0.8, [[(under_photo, {"bold": True, "size": 15, "color": "pri"})], "Germany, from January 2027"],
         size=12, color="sec", align="c", spacing=2, name="Role or track, place and start date")
    # A code the audience can photograph: it opens the profile page, which carries this slide's content.
    qr_file = OUT.parent / "profile_qr.png"
    segno.make(PROFILE_URL, error="m").save(str(qr_file), scale=20, border=4, dark="#000000", light="#FFFFFF")
    qr = s.shapes.add_picture(str(qr_file), Inches(1.0), Inches(5.0), Inches(1.6), Inches(1.6))
    qr.name = "QR code to the profile page"
    qr.click_action.hyperlink.address = PROFILE_URL
    text(s, 0.5, 6.6, 2.6, 0.3, "Scan to read more about me", size=11, color="sec", align="c", name="QR caption")
    facts = [
        ("Background", "Doctorate in quantitative ecology and ecological modelling; over ten years of statistical modelling, time series and uncertainty analysis."),
        ("Experience", "Coordinated data harmonisation and quality assurance across an EU consortium of 19 partners in 10 countries."),
        ("Now", "Building tested, documented data products with Python, SQL, dbt, DuckDB and Databricks."),
        ("Looking for", "Data science, analytics engineering and environmental data roles in Germany."),
    ]
    for i, (head, body) in enumerate(facts):
        y = 2.0 + 1.12 * i
        card(s, 3.6, y, 9.23, 1.0, name=f"About card {i + 1}")
        text(s, 3.8, y, 1.7, 1.0, head, size=14, bold=True, color="pri", anchor="middle", name=f"About {i + 1} label")
        text(s, 5.5, y, 7.15, 1.0, body, size=14, anchor="middle", name=f"About {i + 1} text")
    text(s, 3.6, 6.5, 9.23, 0.3, "github.com/aolagoke-dsae  ·  linkedin.com/in/adewole-olagoke", size=12, color="sec", name="Links")


# ================================================================ template: stakeholder
def s_stake():
    s = new_slide("Stakeholder", "Who needs this: an operator with a load that can wait",
                  "Someone who decides today when tomorrow's electricity will be used.", notes=(
        "The stakeholder is whoever schedules a flexible load: a fleet manager choosing when vehicles charge, a facility "
        "manager running heat pumps or a cold store, or an engineer scheduling batch computing. Their decision is made once a "
        "day, for the next day: which hours to run. They need three things from me, and those are the three questions of this "
        "talk: a forecast that is better than assuming nothing changes, an interval they can rely on, and evidence that "
        "following it actually reduces emissions."))
    card(s, 0.5, 2.0, 6.06, 1.75, name="Who card")
    text(s, 0.7, 2.1, 5.7, 0.34, "Who", size=14, bold=True, color="pri", name="Who label")
    text(s, 0.7, 2.5, 5.7, 1.2, ["Fleet managers: when vehicles charge", "Facility managers: heat pumps, cold stores",
                                 "Platform engineers: batch computing"], size=14, spacing=3, name="Who text")
    card(s, 6.77, 2.0, 6.06, 1.75, name="Decision card")
    text(s, 6.97, 2.1, 5.7, 0.34, "Their decision", size=14, bold=True, color="pri", name="Decision label")
    text(s, 6.97, 2.5, 5.7, 1.2, ["Made once a day, for the next 24 hours:", "which hours should a load of a few hours run,",
                                  "and how far can the forecast be trusted?"], size=14, spacing=3, name="Decision text")
    text(s, 0.5, 4.0, 12.33, 0.34, "What they need answered: the three questions of this talk", size=14, bold=True, name="Questions heading")
    details = ["A day ahead, hour by hour, better than assuming no change?",
               "Does the 90% interval really contain 90% of what happens?",
               "Does a load that follows the forecast emit less CO₂?"]
    w, gap = 3.977, 0.2
    for i, (q, d) in enumerate(zip(QUESTIONS, details)):
        bx = 0.5 + i * (w + gap)
        card(s, bx, 4.5, w, 2.25, fill="bg1", line="pri", name=f"Question card {i + 1}")
        badge(s, bx + 0.2, 4.7, 0.56, str(i + 1), size=20)
        text(s, bx + 0.92, 4.7, w - 1.1, 0.56, q, size=18, bold=True, anchor="middle", name=f"Question {i + 1}")
        text(s, bx + 0.2, 5.5, w - 0.4, 1.1, d, size=14, color="sec", name=f"Question {i + 1} detail")


# ================================================================ template: data introduction
def s_data():
    t = D["template"]
    f = t["facts"]
    s = new_slide("Data introduction", "Two grids, two years, one row per hour",
                  f"What the model learns from. Exploration used data before {t['eda_until']} only, so the held-out period stayed unseen.", notes=(
        "The dataset is small and simple. Two bidding zones, Germany-Luxembourg and France, two years, one row per zone and "
        f"hour: {f['rows']:,} rows. Each row holds the generation by source and the carbon intensity derived from it. The chart "
        f"shows why the two grids behave so differently: France is about {t['mix']['FR'][3]:.0f}% nuclear, Germany-Luxembourg is more than half "
        f"renewable but still about {t['mix']['DE-LU'][4]:.0f}% fossil. That mix is what drives everything that follows."))
    table(s, 0.5, 2.0, 5.6, [1.7, 3.9], ["", "The dataset"], [
        ["Zones", "Germany-Luxembourg (DE-LU) and France (FR)"],
        ["Period", f"{f['start']} to {f['end']}"],
        ["Resolution", "Hourly, UTC"],
        ["Rows", f"{f['rows']:,} (one per zone and hour)"],
        ["Each row", "Generation by source (MWh), shares of wind, solar, nuclear and fossil, carbon intensity"],
        ["Target", "Carbon intensity of generation, gCO₂eq/kWh"],
    ], name="Dataset facts", size=12, row_h=0.6)
    src = text(s, 0.5, 6.3, 5.6, 0.5, [[("Where it comes from: ", {"bold": True, "color": "tx1"}),
                                        ("ENTSO-E, pulled and derived in a separate project and pinned here by hash. ", {}),
                                        (f"Details on slide {NUM['s03']} \u203a", {"bold": True, "color": "pri"})]],
               size=12, color="sec", name=f"Data source (link to slide {NUM['s03']})")
    link([src], NUM["s03"])
    data = CategoryChartData()
    data.categories = ["FR", "DE-LU"]
    for i, name in enumerate(t["mix_names"]):
        data.add_series(name, [t["mix"]["FR"][i] / 100, t["mix"]["DE-LU"][i] / 100])
    frame = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED, Inches(6.4), Inches(2.0), Inches(6.43), Inches(3.1), data)
    frame.name = "Generation mix"
    chart = frame.chart
    style_chart(chart, title="Where the electricity comes from (share of generation)")
    style_axes(chart, number_format="0%", vmin=0, vmax=1.0, major=0.25)
    chart.plots[0].gap_width = 60
    chart.plots[0].overlap = 100
    for srs, rgb in zip(chart.plots[0].series, ["2B5C96", "E3A008", "6BAF92", "8C6BB1", "57606A"]):
        srs.format.fill.solid()
        srs.format.fill.fore_color.rgb = RGBColor.from_string(rgb)
    stats = [(f"{t['mix']['FR'][3]:.0f}%", "of French generation is nuclear", "fr"),
             (f"{t['mix']['DE-LU'][4]:.0f}%", "of DE-LU generation is fossil", "de")]
    for i, (big, label, key) in enumerate(stats):
        bx = 6.4 + i * 3.315
        card(s, bx, 5.3, 3.115, 1.45, name=f"Stat card {i + 1}")
        text(s, bx + 0.18, 5.38, 2.8, 0.6, big, size=30, bold=True, color=key, name=f"Stat {i + 1}")
        text(s, bx + 0.18, 6.05, 2.8, 0.6, label, size=13, name=f"Stat {i + 1} label")


# ================================================================ template: EDA framing the hypotheses
def s_eda():
    t, pr = D["template"], D["problem"]
    s = new_slide("Exploring the data", "What the data suggested: three hypotheses, fixed before modelling",
                  f"Each plot uses data before {t['eda_until']} only. Each hypothesis was written down, then tested once on the held-out period.", notes=(
        "Three things stood out in the exploration, and each became a hypothesis. First, in Germany intensity falls steeply as "
        "the renewable share rises, so I expected high-renewable hours to be cleaner but harder to forecast. Second, in summer "
        "German midday intensity drops sharply, so I asked whether Germany could beat France around midday on good days; the "
        "plot already shows how far apart they are. Third, intensity is strongly autocorrelated and has a clear 24-hour cycle, "
        "so I expected the calendar to matter more than the current state a day ahead. Hold these in mind: I come back to all "
        "three."))
    w, gap, y0 = 3.977, 0.2, 2.0
    obs = [
        ("H1", "In Germany-Luxembourg (DE-LU) carbon intensity falls steeply as the share of renewables rises.",
         "Hours with the most renewables are cleaner and have a wider interval."),
        ("H2", "In summer DE-LU carbon intensity drops sharply at midday; FR stays low all day.",
         "On sunny, windy days DE-LU is cleaner than FR around midday."),
        ("H3", "Carbon intensity resembles the hour before, and the same hour a day earlier.",
         "A day ahead, the calendar matters more than the current state of the grid."),
    ]
    for i, (code, seen, hypo) in enumerate(obs):
        bx = 0.5 + i * (w + gap)
        card(s, bx, y0, w, 4.78, fill="bg1", line="mut", name=f"{code} exploration card")
        text(s, bx + 0.2, y0 + 2.9, w - 0.4, 0.6, [[("Seen: ", {"bold": True}), (seen, {})]], size=12, name=f"{code} observation")
        card(s, bx + 0.15, y0 + 3.6, w - 0.3, 1.05, name=f"{code} hypothesis card")
        text(s, bx + 0.3, y0 + 3.6, w - 0.6, 1.05, [[(code + "  ", {"bold": True, "color": "pri", "size": 16}), (hypo, {})]], size=13,
             anchor="middle", name=f"{code} hypothesis")
    line_chart(s, 0.6, y0 + 0.1, w - 0.2, 2.75, [f"{v:.0f}%" for v in t["decile_share"]],
               [("DE-LU", t["decile_intensity"], "de", 2.5, False)], name="Carbon intensity by renewable share", legend=False,
               title="Carbon intensity in DE-LU", number_format="0", vmin=0, skip=3,
               value_title="g CO₂ per kWh", category_title="Share of renewables in generation")
    line_chart(s, 0.6 + (w + gap), y0 + 0.1, w - 0.2, 2.75, pr["hours"], [
        ("DE-LU", t["summer_day"]["DE-LU"], "de", 2.5, False), ("FR", t["summer_day"]["FR"], "fr", 2.5, False),
    ], name="Summer day", title="Carbon intensity, summer day", number_format="0", vmin=0, skip=6,
        value_title="g CO₂ per kWh", category_title="Hour of day (UTC)")
    line_chart(s, 0.6 + 2 * (w + gap), y0 + 0.1, w - 0.2, 2.75, [str(v) for v in t["acf_lags"]], [
        ("DE-LU", t["acf"]["DE-LU"], "de", 2.5, False), ("FR", t["acf"]["FR"], "fr", 2.5, False),
    ], name="Autocorrelation", title="How alike are two hours?", number_format="0.00", vmin=0, vmax=1.0, major=0.25, skip=12,
        value_title="Correlation of carbon intensity", category_title="Hours apart")


# ================================================================ template: baseline
def s_base():
    t, sm = D["template"], D["summary"]
    s = new_slide("Baseline", f"Baseline: “no change”. The model's error is {sm['skill']['DE-LU']:.0f}% and {sm['skill']['FR']:.0f}% lower",
                  "Held-out test. Error: how far the median forecast misses on average (pinball loss, half the mean absolute error), in g CO₂ per kWh.",
                  question=0, notes=(
        "The baseline is persistence: tomorrow's intensity at each hour equals the intensity now. It is a strong baseline at "
        "short lead times, because intensity changes slowly. The table gives the error of the baseline and of the model on the "
        "held-out test. The model wins in every zone and every lead-time band. The improvement is largest a few hours ahead. "
        "A promotion gate in the pipeline enforces exactly this: a model that does not beat the baseline in every zone and band "
        "cannot become the champion."))
    rows = [[z, band.replace("All lead times", "All, 1–24 hours"), f"{b:.2f}", f"{m:.2f}", f"{imp:.0f}% lower"]
            for z, band, b, m, imp in t["baseline"]]
    table(s, 0.5, 2.0, 8.0, [1.3, 2.2, 1.9, 1.6, 1.7], ["Zone", "Hours ahead", "Baseline error", "Model error", "Model against baseline"],
          rows, name="Baseline against model", size=13, row_h=0.5)
    points = [
        ("The baseline", "Persistence: every hour of tomorrow has the carbon intensity of right now."),
        ("Why it is hard to beat", "Carbon intensity changes slowly, so “the same as now” is a good guess for the next few hours."),
        ("Enforced, not just reported", "A model that fails to beat it in any zone or lead-time band cannot be promoted."),
    ]
    for i, (head, body) in enumerate(points):
        y = 2.0 + 1.52 * i
        card(s, 8.8, y, 4.03, 1.4, name=f"Point card {i + 1}")
        text(s, 8.98, y + 0.1, 3.7, 0.32, head, size=14, bold=True, name=f"Point {i + 1} title")
        text(s, 8.98, y + 0.46, 3.7, 0.9, body, size=12, name=f"Point {i + 1} text")


# ================================================================ template: model comparison
def s_comp():
    t = D["template"]
    s = new_slide("Model comparison", f"{F['configurations']} configurations compared; the choice was made on validation only",
                  "5 model sizes × 2 targets × 4 calibration windows. The held-out test was used once, for the chosen model.",
                  question=0, notes=(
        f"I compared {F['configurations']} configurations: five model sizes, two ways of defining the target, and four lengths of calibration "
        "window. The rules for choosing were stated in advance, and everything was compared on the validation block, never on "
        f"the test. Predicting the change beat predicting the level for every model size. A {WINDOW}-day calibration window gave "
        f"coverage closest to 90% for every size. Among sizes, bigger models are slightly more accurate, but I kept {TREES} trees "
        "because the whole pipeline has to run locally in under 15 seconds. That is why the final results table has one model: "
        "the comparison happened before the test was opened."))
    rows = [list(r_) for r_ in t["sizes"]]
    for r_ in rows:
        if r_[0].startswith("200"):
            r_[0] = f"{TREES} trees (chosen)"
    table(s, 0.5, 2.0, 7.4, [2.3, 1.2, 1.2, 1.35, 1.35, 1.1],
          ["Model size", "Coverage DE-LU", "Coverage FR", "Error DE-LU", "Error FR", "Fit time"], rows,
          name="Model sizes on validation", size=12, row_h=0.58)
    text(s, 0.5, 5.6, 7.4, 0.9, "Error is pinball loss: how far the forecast misses on average; lower is better. Validation block (July to September), so "
         "not comparable with the held-out table. "
         f"Every one of the {F['configurations']} configurations beat the baseline in every zone and lead-time band.", size=12, color="sec", name="Table note")
    decisions = [
        ("Target", "Predict the change from now", "Lower error than predicting the level, for all 5 sizes"),
        ("Calibration window", f"{WINDOW} days", "Coverage closest to 90% in the worse zone, for all 5 sizes"),
        ("Model size", f"{TREES} trees", "Lowest error that keeps the local run under 15 seconds"),
    ]
    for i, (what, chosen, why) in enumerate(decisions):
        y = 2.0 + 1.52 * i
        card(s, 8.2, y, 4.63, 1.4, name=f"Decision card {i + 1}")
        text(s, 8.38, y + 0.1, 4.3, 0.3, what, size=12, bold=True, color="sec", name=f"Decision {i + 1} label")
        text(s, 8.38, y + 0.4, 4.3, 0.4, chosen, size=17, bold=True, color="pri", name=f"Decision {i + 1} choice")
        text(s, 8.38, y + 0.86, 4.3, 0.5, why, size=12, name=f"Decision {i + 1} reason")


# ================================================================ template: recommendations
def s_rec():
    v, b, t, tp = D["value"], D["backup"], D["trust"], D["template"]
    a = b["avoided"]["DE-LU"]["median"]
    s = new_slide("Recommendations", "Four recommendations to an operator of flexible load",
                  "Each follows from a result on the held-out test.", notes=(
        "What should the operator actually do? Four things. Choose the window by the median forecast: it backfired least. "
        "Start with loads on the German grid, where the swing is large; in France timing is worth little. Keep the window as "
        "short as the process allows, because short windows fit inside the cleanest hours. And plan on the interval a day ahead "
        "in Germany, but allow extra margin in France, where the interval is slightly too narrow beyond 12 hours."))
    recs = [
        ("Choose the window by the median forecast",
         f"It was dirtier than the day's average on {v['bad']['DE-LU'][2]:.0f}% of days, against {v['bad']['DE-LU'][1]:.0f}% when choosing by the upper bound and {v['bad']['DE-LU'][0]:.0f}% when repeating yesterday."),
        ("Start with loads on the German grid",
         f"{v['tonnes']['DE-LU'][2]} t CO₂ avoided per daily MWh in DE-LU against {v['tonnes']['FR'][2]} t in FR over {v['n_days']} days."),
        ("Keep the window as short as the process allows",
         f"{a[0]:.0f} kg CO₂ avoided per MWh with a {b['window_lengths'][0]}-hour window, {a[-1]:.0f} kg with a {b['window_lengths'][-1]}-hour window."),
        ("Rely on the interval in DE-LU; add margin in FR a day ahead",
         f"At 13–24 hours the 90% interval covered {tp['de_day_ahead']}% in DE-LU and {t['fr_day_ahead']}% in FR."),
    ]
    w, h, gx, gy = 6.06, 2.25, 0.21, 0.2
    for i, (head, evidence) in enumerate(recs):
        rw, col = divmod(i, 2)
        bx, by = 0.5 + col * (w + gx), 2.0 + rw * (h + gy)
        card(s, bx, by, w, h, fill="bg1", line="pri", name=f"Recommendation card {i + 1}")
        badge(s, bx + 0.2, by + 0.2, 0.5, str(i + 1), size=17)
        text(s, bx + 0.88, by + 0.2, w - 1.05, 0.75, head, size=17, bold=True, anchor="middle", name=f"Recommendation {i + 1}")
        text(s, bx + 0.2, by + 1.15, w - 0.4, 1.0, [[("Evidence: ", {"bold": True, "color": "sec"}), (evidence, {})]], size=13,
             name=f"Recommendation {i + 1} evidence")


# ================================================================ interview: what the project demonstrates
def s_skills():
    m, t, b = D["mlops"], D["trust"], D["backup"]
    s = new_slide("Overview", "What this project demonstrates",
                  "One project, six skills. Each row names the evidence and where to find it in the repository.", notes=(
        "Before the detail, here is what I would like you to take from this project. It is a forecasting model, but the "
        "modelling is perhaps a third of the work. The rest is what makes a model trustworthy in practice: a data lineage that "
        "can be traced, an evaluation fixed before the results were seen, a pipeline that runs on Databricks with a promotion "
        "gate, tests that I have tried to break on purpose, and results stated in terms a stakeholder can act on. I will go "
        "through each, and I am happy to stop at any of them."))
    table(s, 0.5, 2.0, 12.33, [2.5, 6.3, 3.5], ["Skill", "Evidence in this project", "Where to look"], [
        ["Probabilistic modelling", f"Quantile forecasts with a calibrated 90% interval; coverage of {t['coverage']['DE-LU']:.1f}% and {t['coverage']['FR']:.1f}% on unseen data",
         "src/carbon_mlops/models"],
        ["Experimental rigour", "Analysis plan committed before the test; three hypotheses falsified and reported as registered",
         "docs/ANALYSIS_PLAN_H1_H3.md"],
        ["Data engineering", "Lineage pinned by hash from the ENTSO-E API through dbt and DuckDB to Delta tables",
         "ingestion, upstream repository"],
        ["MLOps", "Three Databricks jobs, a model registry with a promotion gate, weekly recalibration, monitoring",
         "databricks.yml, jobs.py"],
        ["Testing", f"{m['tests']} automated tests; {m['breakages']} deliberate breakages, all caught; checks on every push",
         "tests, scripts/mutation_audit.py"],
        ["Communication", f"Value stated in tonnes of CO₂, four recommendations, and {len(b['review_open'])} open findings named, not hidden",
         "README.md, docs/REVIEW_ISSUES.md"],
    ], name="Skills and evidence", size=13, row_h=0.66)


# ================================================================ interview: lessons
def s_lessons():
    if INTERVIEW:
        title, notes = "What I would do differently next time", (
            "Four things I would do differently. First, I would seal the test set technically from day one. Here it was scored "
            "during the review against an earlier model; I disclosed that, but it should not have been possible. Second, I would "
            "write the breakage audit before trusting the tests: the first time I ran it, four of twenty-eight deliberate breakages "
            "passed unnoticed. Third, I would recompute every headline number independently before reporting it: an early coverage "
            "figure for France did not reproduce. Fourth, I would look for better inputs before a bigger model: tripling the model "
            "size bought about three percent, while published day-ahead schedules were never tried.")
    else:
        title, notes = "What I learned, and would do differently", (
            "Four lessons, briefly. One: the test set was scored during my review before the final model, so next time I would "
            "lock it away by construction. Two: my first tests missed four of twenty-eight deliberate breakages, so I would break "
            "the tests on purpose from the start. Three: an early coverage figure did not reproduce, so every headline number gets "
            "a second, independent computation. Four: a model three times the size gained about three percent, so I would look for "
            "better inputs before a bigger model. About forty seconds: one sentence each, then move on.")
    s = new_slide("Reflection", title,
                  "Four lessons from the review of this project, each with what happened and what I would change.", notes=notes)
    lessons = [
        ("Seal the test set by construction", "The held-out period was scored several times against an earlier model during the review.",
         "Keep it where the training code cannot read it until a plan is committed."),
        ("Break the tests before trusting them", "The first audit caught 24 of 28 deliberate breakages; four passed unnoticed.",
         "Write the breakage audit alongside the first tests, not after the model."),
        ("Recompute every headline number", f"An early coverage figure for FR, {F['fr_first_reported']:.0f}%, came out at {F['fr_reproduced']:.0f}% when reproduced.",
         "A second, independent computation before any figure is reported."),
        ("Better inputs before a bigger model", f"{MULTIPLE} the trees cut the error by about {F['largest_error_cut_pct']:.0f}%; day-ahead schedules were never tried.",
         "Spend the effort on forward-looking features first."),
    ]
    w, h, gx, gy = 6.06, 2.25, 0.21, 0.2
    for i, (head, happened, change) in enumerate(lessons):
        rw, col = divmod(i, 2)
        bx, by = 0.5 + col * (w + gx), 2.0 + rw * (h + gy)
        card(s, bx, by, w, h, fill="bg1", line="pri", name=f"Lesson card {i + 1}")
        badge(s, bx + 0.2, by + 0.2, 0.5, str(i + 1), size=17)
        text(s, bx + 0.88, by + 0.2, w - 1.05, 0.5, head, size=17, bold=True, anchor="middle", name=f"Lesson {i + 1}")
        text(s, bx + 0.2, by + 0.9, w - 0.4, 1.25, [
            [("What happened: ", {"bold": True, "color": "sec"}), (happened, {})],
            [("Next time: ", {"bold": True, "color": "pri"}), (change, {})],
        ], size=13, spacing=5, name=f"Lesson {i + 1} detail")


# ================================================================ backup slides
NUM: dict[str, int] = {}  # function name -> slide number, filled in before the slides are built
BACKUP_QUESTIONS = {
    "b01": "How accurate is it at each lead time?",
    "b02": "Where does the interval fail?",
    "b03": "Why recalibrate every week?",
    "b04": "Does the value depend on the window length?",
    "b05": "How do you know nothing leaks from the future?",
    "b06": "Why these design choices?",
    "b07": "What could be wrong with these results?",
    "b08": "What did the review find?",
    "s08": "How is it run and tested on Databricks?",
    "s09": "What does monitoring show?",
    "b09": "What would production need?",
    "b10": "What do the terms mean?",
    "s03": "Where does the data come from?",
    "s_data": "What is in the data?",
    "s_eda": "What did the exploration show?",
    "s_comp": "Which models were compared?",
    "s_rec": "What should an operator do?",
}
BACKUP_Q: list[tuple[str, str]] = []  # (question, slide function name), filled from the appendix order


def back_link(s):
    if any(sh.name == "Link to backup index" for sh in s.shapes) or not IN_APPENDIX:
        return
    # On the footer line, between the source text and the slide number. Higher up it sat on
    # top of the bottom row of cards on four slides.
    b = text(s, 10.75, 6.92, 1.35, 0.26, "‹ Backup index", size=11, color="pri", align="r", name="Link to backup index")
    link([b], NUM["b00"])


def b00():
    s = new_slide("Backup", "Backup: questions you might ask",
                  "Not part of the talk. Each card links to a slide that answers the question.", notes=(
        "Backup slides for questions. Each card is a link in slide-show mode. Every backup slide has a link back here."))
    cols = 2 if len(BACKUP_Q) <= 12 else 3
    per_col = -(-len(BACKUP_Q) // cols)
    gx, gy = (0.43, 0.1) if cols == 2 else (0.2, 0.12)
    w = (12.33 - gx * (cols - 1)) / cols
    h = min(0.82, (4.75 - gy * (per_col - 1)) / per_col)
    for i, (question, name) in enumerate(BACKUP_Q):
        target = NUM[name]
        col, rw = divmod(i, per_col)
        bx, by = 0.5 + col * (w + gx), 2.0 + rw * (h + gy)
        c = card(s, bx, by, w, h, name=f"Backup card {i + 1} (link to slide {target})")
        n = badge(s, bx + 0.18, by + (h - 0.5) / 2, 0.5, str(target), size=13)
        t = text(s, bx + 0.85, by, w - 1.0, h, question, size=15 if cols == 2 else 13, bold=True, anchor="middle", name=f"Backup question {i + 1}")
        link([c, n, t], target)


def b01():
    b, t = D["backup"], D["trust"]
    s = new_slide("Backup: accuracy", "The forecast beats “no change” at almost every lead time",
                  "Share of the error of a no-change forecast that the median forecast removes, held-out test. Above zero is better.", notes=(
        "Skill here is one minus the ratio of the median forecast's pinball loss to that of a no-change forecast. In Germany it "
        "is positive at every lead time and largest a few hours ahead. In France it is positive from two hours ahead, but at one "
        "hour ahead the model loses to no-change: French intensity barely moves in an hour, so 'the same as now' is very hard to "
        "beat. That is logged as an open finding."))
    line_chart(s, 0.5, 2.0, 8.2, 4.5, [str(h) for h in t["lead_hours"]], [
        ("DE-LU", b["skill_by_lead"]["DE-LU"], "de", 2.5, False),
        ("FR", b["skill_by_lead"]["FR"], "fr", 2.5, False),
        ("No better than no-change", [0] * 24, "tx1", 1.0, True),
    ], name="Skill by lead time", title="Skill against “no change” by lead time (hours ahead)", number_format="0%", skip=3)
    fr1, de_best = b["skill_by_lead"]["FR"][0], max(b["skill_by_lead"]["DE-LU"])
    for i, (big, label, key) in enumerate([
        (pct(100 * de_best), "best lead time in DE-LU", "de"),
        (pct(100 * fr1), "FR at 1 hour ahead: worse than no-change", "fr"),
    ]):
        y = 2.0 + 1.75 * i
        card(s, 9.0, y, 3.83, 1.6, name=f"Stat card {i + 1}")
        text(s, 9.2, y + 0.12, 3.4, 0.6, big, size=32, bold=True, color=key, name=f"Stat {i + 1}")
        text(s, 9.2, y + 0.8, 3.4, 0.7, label, size=13, name=f"Stat {i + 1} label")
    text(s, 9.0, 5.55, 3.83, 0.95, "In FR intensity barely moves within an hour, so “the same as now” is hard to beat. Logged as an open finding.",
         size=12, color="sec", name="Explanation")
    back_link(s)


def b02():
    b = D["backup"]
    s = new_slide("Backup: calibration", "Coverage is uneven across renewable regimes and across months",
                  "Held-out test. Target 90%, accepted band 87–96%. Quartiles are of the renewable share at the moment of forecasting.", notes=(
        "The overall coverage hides variation. In Germany the interval under-covers when renewables are lowest and when they are "
        "highest, and over-covers in between: the calibration is per zone and lead time, not per regime. France is weakest in its "
        "second quartile. By month, October is weak in both zones, and December in France. A natural next step is to condition the calibration on the renewable regime."))
    q = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    bar_chart(s, 0.5, 2.0, 6.0, 4.45, q, [("DE-LU", b["quartile"]["DE-LU"], "de"), ("FR", b["quartile"]["FR"], "fr")],
              name="Coverage by renewable quartile", horizontal=False, title="Coverage by renewable share quartile", label_format="0%",
              number_format="0%", vmin=0, vmax=1.0, major=0.2, gap=70)
    bar_chart(s, 6.83, 2.0, 6.0, 4.45, b["month_labels"], [("DE-LU", b["month"]["DE-LU"], "de"), ("FR", b["month"]["FR"], "fr")],
              name="Coverage by month", horizontal=False, title="Coverage by month", label_format="0%",
              number_format="0%", vmin=0, vmax=1.0, major=0.2, gap=70)
    back_link(s)


def b03():
    b, t = D["backup"], D["trust"]
    s = new_slide("Backup: method", "Offsets fixed once did not carry over; weekly recalibration did",
                  "Coverage of the 90% interval on the held-out period, earlier model with fixed offsets against the final model.", notes=(
        "Conformal prediction guarantees coverage only if calibration and future errors are exchangeable. Hourly grid data is "
        "autocorrelated and seasonal, so that does not hold, and the guarantee cannot be assumed: coverage has to be checked. "
        f"With offsets computed once, France covered only {b['static']['FR']:.0f}% on the test period. Recomputing the offsets every week from the last "
        f"{WINDOW} days of real errors brought it to {t['coverage']['FR']:.1f}%. The two bars are different model versions, so this is a before-and-after, "
        "not a controlled experiment."))
    bar_chart(s, 0.5, 2.0, 6.4, 4.45, ["DE-LU", "FR"], [
        ("Fixed offsets (earlier model)", [b["static"]["DE-LU"] / 100, b["static"]["FR"] / 100], "mut"),
        ("Recalibrated weekly (final model)", [t["coverage"]["DE-LU"] / 100, t["coverage"]["FR"] / 100], "pri"),
    ], name="Fixed against weekly calibration", horizontal=False, title="Coverage on the held-out period", label_format="0.0%",
        number_format="0%", vmin=0, vmax=1.0, major=0.2, gap=80)
    points = [
        ("The guarantee does not apply", "Conformal coverage is guaranteed only for exchangeable data. Hourly grid data is autocorrelated and seasonal."),
        ("So coverage is measured, not assumed", "Every forecast in the test is calibrated only on errors observed before it."),
        ("Not a controlled comparison", "The two bars are different model versions: a before-and-after from the review."),
    ]
    for i, (head, body) in enumerate(points):
        y = 2.0 + 1.5 * i
        card(s, 7.2, y, 5.63, 1.36, name=f"Point card {i + 1}")
        text(s, 7.4, y + 0.1, 5.25, 0.32, head, size=14, bold=True, name=f"Point {i + 1} title")
        text(s, 7.4, y + 0.47, 5.25, 0.85, body, size=12, name=f"Point {i + 1} text")
    back_link(s)


def b04():
    b = D["backup"]
    cats = [f"{n} h" for n in b["window_lengths"]]
    s = new_slide("Backup: business value", "Shorter windows avoid more; the forecast leads at every length",
                  "Held-out test, DE-LU. A longer window is easier to hit but includes dirtier hours.", notes=(
        "The main slide uses a three-hour window. The result is not specific to that choice. The shorter the window, the more "
        "CO2 is avoided per megawatt-hour, because a short window can sit entirely in the cleanest hours. The forecast beats the "
        "rule of thumb at every length. At six hours the forecast backfires about as often as it does at three for the rule of "
        "thumb, so long loads gain least."))
    a = b["avoided"]["DE-LU"]
    line_chart(s, 0.5, 2.0, 6.0, 4.45, cats, [
        ("Hindsight (best possible)", a["hindsight"], "tx1", 1.5, True),
        ("Forecast, median", a["median"], "de", 3.0, False),
        ("Repeat yesterday", a["yesterday"], "mut", 2.0, True),
    ], name="Avoided by window length", title="kg CO₂ avoided per MWh moved, by window length", number_format="0", vmin=0)
    bad = b["bad_by_window"]["DE-LU"]
    bar_chart(s, 6.83, 2.0, 6.0, 4.45, cats, [
        ("Repeat yesterday", [v / 100 for v in bad["yesterday"]], "mut"),
        ("Forecast, median", [v / 100 for v in bad["median"]], "de"),
    ], name="Bad days by window length", horizontal=False, title="Share of days dirtier than the day's average", label_format="0%",
        number_format="0%", vmin=0, vmax=0.4, major=0.1, gap=70)
    back_link(s)


def b05():
    m = D["mlops"]
    s = new_slide("Backup: leakage", "Three independent guards against using the future",
                  "A forecast issued at time t may use nothing observed after t. Each guard below is enforced by code, not by convention.", notes=(
        "Leakage is the most common way a forecasting project fools itself. Three guards. First, an audit on every feature row "
        "checks that the newest feature timestamp is not after the forecast origin and that the target is exactly origin plus "
        "lead time. Second, the four blocks are purged: a row belongs to a block only if its target also falls inside it, so no "
        "target hour is shared. Third, the breakage audit deliberately shifts a feature one hour into the future and confirms a "
        "test fails."))
    y = 2.15
    shape(s, MSO_SHAPE.RECTANGLE, 0.5, y + 0.32, 12.33, 0.05, fill="mut", name="Time axis")
    marks = [(2.0, "features: lags and rolling statistics", "sec"), (6.3, "t: the forecast is issued", "pri"), (10.3, "t + lead time: the target hour", "de")]
    for x, label, key in marks:
        shape(s, MSO_SHAPE.OVAL, x - 0.12, y + 0.225, 0.24, 0.24, fill=key, name="Time marker")
        text(s, x - 1.8, y + 0.58, 3.6, 0.3, label, size=13, bold=key != "sec", align="c", name="Time label")
    text(s, 0.5, y - 0.05, 5.6, 0.3, "everything the model may see", size=12, color="sec", align="r", name="Left of origin")
    text(s, 6.6, y - 0.05, 5.6, 0.3, "nothing from here may be used", size=12, color="sec", name="Right of origin")
    guards = [
        ("1", "Audit on every row", "The newest feature timestamp must not be after t, and the target must be exactly t plus the lead time. Gate G2."),
        ("2", "Purged blocks", "A row belongs to a block only if its target hour does too, so no target hour is shared between train, calibration, validation and test."),
        ("3", "Deliberate breakage", f"A script shifts a feature one hour into the future and checks that a test fails. One of {m['breakages']} such breakages."),
    ]
    w, gap = 3.977, 0.2
    for i, (num, head, body) in enumerate(guards):
        bx = 0.5 + i * (w + gap)
        card(s, bx, 3.75, w, 2.6, fill="bg1", line="pri", name=f"Guard card {i + 1}")
        badge(s, bx + 0.2, 3.93, 0.46, num, size=16)
        text(s, bx + 0.8, 3.93, w - 1.0, 0.46, head, size=17, bold=True, anchor="middle", name=f"Guard {i + 1} title")
        text(s, bx + 0.2, 4.6, w - 0.4, 1.65, body, size=14, name=f"Guard {i + 1} text")
    back_link(s)


def b06():
    s = new_slide("Backup: design", "Each design choice has an alternative that was considered",
                  "“Measured”: decided on the validation block. “Review”: found in the pre-freeze review. “Judgement”: an engineering call.", notes=(
        "Be ready to defend each choice. The target formulation and the recalibration schedule were chosen on the validation "
        "block, never on the test. LightGBM is a judgement: the dataset is small and tabular, the quantile objective is built in, "
        "and the whole test suite has to run in seconds. A deep model would need far more data and tuning for an uncertain gain. "
        "The pinned snapshot trades liveness for reproducibility."))
    table(s, 0.5, 2.0, 12.33, [2.6, 3.0, 5.4, 1.3], ["Choice", "Alternative", "Why", "Basis"], [
        ["Predict the change from now", "Predict the level directly", "“No change” becomes the zero prediction, so the model only has to learn the departure from it", "measured"],
        [f"Weekly recalibration, last {WINDOW} days", "Offsets computed once", "Fixed offsets under-covered in FR when the season changed", "measured"],
        ["Offsets per zone and lead time", "One offset per zone", "Uncertainty grows with lead time and differs by a factor of ten between the grids", "measured"],
        ["LightGBM quantile regression", "Neural or statistical models", "Small tabular data, built-in quantile objective, trains in seconds, easy to test", "judgement"],
        ["Pinned snapshot of the data", "Live API calls", "Every result is reproducible and needs no token; the cost is that the data is not live", "judgement"],
        ["Gate on coverage in each zone", "Gate on pooled coverage", "A pooled figure let a good zone hide a failing one", "review"],
    ], name="Design choices", size=12, row_h=0.62)
    back_link(s)


def b07():
    s = new_slide("Backup: validity", "Six reasons to read these results with care",
                  "Stated in the repository alongside the results, not discovered afterwards.", notes=(
        "These are the threats to validity I would raise myself. The most important is the first: the held-out test was not "
        "blind. During the review it had been scored several times against an earlier model, and that is what prompted the move "
        "to weekly recalibration. The final model's settings were then chosen on the validation block only, and the hypothesis "
        "evaluation ran once under a committed plan, but the test is not untouched and I say so."))
    items = [
        ("The test was not blind", "It was scored several times against an earlier model during the review. Final settings were then chosen on validation only."),
        ("One period, four months", "September to December 2025. Summer behaviour is seen only in validation."),
        ("Two grids", "Germany-Luxembourg and France. Nothing is claimed for other zones."),
        ("Emission factors are fixed", "The interval covers forecast error only. Uncertainty in the life-cycle factors is not included, and for FR it is larger."),
        ("Generation, not consumption", "Intensity is of electricity generated in the zone. Imports and exports are not accounted for."),
        ("No prices, no schedules", "Day-ahead prices, load and published generation schedules are not used as features, and value is not in euros."),
    ]
    w, h, gx, gy = 3.977, 2.1, 0.2, 0.2
    for i, (head, body) in enumerate(items):
        rw, col = divmod(i, 3)
        bx, by = 0.5 + col * (w + gx), 2.0 + rw * (h + gy)
        card(s, bx, by, w, h, name=f"Validity card {i + 1}")
        text(s, bx + 0.2, by + 0.15, w - 0.4, 0.4, head, size=15, bold=True, name=f"Validity {i + 1} title")
        text(s, bx + 0.2, by + 0.65, w - 0.4, 1.35, body, size=13, name=f"Validity {i + 1} text")
    back_link(s)


def b08():
    b, m = D["backup"], D["mlops"]
    s = new_slide("Backup: review", "An adversarial review before the freeze changed the results",
                  "Every issue is logged in the repository with its status. Three examples of what it found:", notes=(
        "Before freezing the code I had the project reviewed adversarially and logged every issue. Three examples. The coverage "
        f"figures first reported could not be reproduced; the reproduced figure for France was {F['fr_reproduced']:.0f}%, not {F['fr_first_reported']:.0f}%. A quantile was "
        "computed one rank too high, which made intervals slightly too wide. And the first version of the test suite let several "
        "deliberate breakages through, which is how the breakage audit came to exist. The open items are findings about the "
        "model, not unfixed bugs."))
    for i, (big, label) in enumerate([(str(b["review_items"]), "items logged in the register"),
                                      (str(len(b["review_open"])), "open findings, each stated in the docs"),
                                      (f"{m['breakages']} of {m['breakages']}", "deliberate breakages now caught")]):
        bx = 0.5 + i * 1.98
        card(s, bx, 2.0, 1.85, 1.75, name=f"Stat card {i + 1}")
        text(s, bx + 0.14, 2.1, 1.6, 0.6, big, size=26, bold=True, color="pri", name=f"Stat {i + 1}")
        text(s, bx + 0.14, 2.78, 1.6, 0.9, label, size=12, name=f"Stat {i + 1} label")
    text(s, 0.5, 4.0, 5.8, 0.3, "Open findings", size=14, bold=True, name="Open findings heading")
    text(s, 0.5, 4.35, 5.8, 2.1, [
        "FR a day ahead under-covers; FR at one hour loses to no-change",
        "Coverage is uneven across renewable regimes",
        "Skill fades towards winter; no retraining schedule",
        "Choosing by the upper bound does not help",
        "Fits differ slightly between platforms; drift thresholds too sensitive",
    ], size=12, spacing=5, name="Open findings list")
    examples = [
        ("Coverage could not be reproduced", f"The first reported figure for FR was {F['fr_first_reported']:.0f}%. Reproduced on the same data: {F['fr_reproduced']:.0f}%. This led to weekly recalibration."),
        ("A quantile one rank too high", "The calibration quantile used an order statistic one rank above the intended one. Fixed with an explicit rank."),
        ("Tests that tested too little", "Several deliberate breakages passed the first test suite. Each survivor became a new test."),
    ]
    for i, (head, body) in enumerate(examples):
        y = 2.0 + 1.5 * i
        card(s, 6.75, y, 6.08, 1.36, fill="bg1", line="mut", name=f"Example card {i + 1}")
        text(s, 6.95, y + 0.1, 5.7, 0.32, head, size=14, bold=True, name=f"Example {i + 1} title")
        text(s, 6.95, y + 0.47, 5.7, 0.85, body, size=12, name=f"Example {i + 1} text")
    back_link(s)


def b09():
    s = new_slide("Backup: next steps", "From a verified pipeline to a production service: five gaps",
                  "What exists is reproducible and tested. These are the pieces a live service would add, in the order I would build them.", notes=(
        "If asked what it would take to run this for real. First a retraining schedule, because monitoring already shows skill "
        "fading towards winter. Second live ingestion, replacing the pinned snapshot with scheduled pulls that are hashed the "
        "same way. Third, published day-ahead schedules and prices as features, which the snapshot does not carry. Fourth, alerts "
        "wired to the monitoring table. Fifth, the question-answering agent. It is built and tested: a language model only chooses "
        "which tested tool to call, and every number in an answer comes from that tool. It runs on Databricks against the "
        "forecast tables, with a hosted language model choosing the tool and each call traced. What remains is live data."))
    table(s, 0.5, 2.0, 12.33, [0.6, 3.2, 5.2, 3.3], ["", "Gap", "What it would take", "Why it comes in this order"], [
        ["1", "Retraining schedule", "Retrain monthly on a rolling window; the gate already decides promotion", "Monitoring shows skill fading towards winter"],
        ["2", "Live ingestion", "Scheduled API pulls, each response hashed as upstream does", "Removes the fixed end date of 31 December 2025"],
        ["3", "Forward-looking features", "Day-ahead generation schedules, load forecasts and prices", "The likeliest source of better day-ahead skill, especially in FR"],
        ["4", "Alerting", "Rules on the monitoring table: runs of weeks outside the band", "The numbers exist; nobody is told yet"],
        ["5", "Question-answering agent in service", "Runs on Databricks with a language model choosing the tool, each call traced; point it at live forecast tables", "Useful only once the tables are live"],
    ], name="Production gaps", size=12, row_h=0.68)
    back_link(s)


def b10():
    s = new_slide("Backup: terms", "The terms used in this talk",
                  "For readers of the deck who were not in the room.", notes="A glossary for anyone reading the deck without the talk.")
    table(s, 0.5, 2.0, 12.33, [2.6, 9.7], ["Term", "Meaning here"], [
        ["DE-LU, FR", "Electricity market zones: Germany with Luxembourg, and France."],
        ["Carbon intensity", "Grams of CO₂-equivalent emitted per kWh generated in a zone. The same number is kg per MWh."],
        ["Lead time", "How many hours ahead a forecast looks, from 1 to 24."],
        ["Quantile forecast", "The 5%, 50% and 95% points of what the model expects. The 50% point is the median forecast."],
        ["90% interval", "The range from the 5% to the 95% point. It should contain the outcome nine times in ten."],
        ["Coverage", "The share of outcomes that actually fell inside the interval. The target is 90%."],
        ["Conformal calibration", "Widening or narrowing the interval by offsets computed from recent real errors."],
        ["“No change” (persistence)", "The baseline forecast: carbon intensity stays at its current value."],
        ["Pinball loss", "How far a forecast misses on average. For the median forecast it is half the mean absolute error."],
        ["Skill", "The share of the baseline's error that the forecast removes. Zero means no better than the baseline."],
        ["Drift (PSI)", "A measure of how far the recent distribution of an input has moved from a reference period."],
    ], name="Glossary", size=12, row_h=0.375)
    back_link(s)


if INTERVIEW:
    MAIN = [s00, s_about, s_skills, s02, s_stake, s03, s04, s_base, s05, s06, s07, s08, s_lessons, s10, s11]
    APPENDIX = [b00, s_data, s_eda, s_comp, b01, b02, b03, b04, s_rec, b05, b06, b07, b08, s09, b09, b10]
else:
    MAIN = [s00, s_about, s02, s_stake, s_data, s_eda, s04, s_base, s_comp, s05, s06, s07, s_rec, s_lessons, s10, s11]
    APPENDIX = [b00, s03, b01, b02, b03, b04, b05, b06, b07, b08, s08, s09, b09, b10]
ORDER = MAIN + APPENDIX
NUM.update({fn.__name__: i + 1 for i, fn in enumerate(ORDER)})
BACKUP_Q.extend((BACKUP_QUESTIONS[fn.__name__], fn.__name__) for fn in APPENDIX[1:])
patch_theme()
IN_APPENDIX = False
for build in ORDER:
    IN_APPENDIX = build in APPENDIX[1:]
    build()
    back_link(prs.slides[len(prs.slides) - 1])  # every backup slide gets a way back, whichever deck it is in
assert len(prs.slides) == len(ORDER)
for item, target in PENDING_LINKS:
    item.click_action.target_slide = prs.slides[target - 1]
prs.core_properties.title = TALK_TITLE.replace("₂", "2")
prs.core_properties.author = "Dr. Adewole Olagoke"
prs.core_properties.subject = "Probabilistic forecasting of grid carbon intensity for DE-LU and FR"
_ = copy  # kept for local experiments with duplicated shapes
prs.save(OUT)
print("written", OUT)
