#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


def inline_markup(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def table_block(lines: list[str], style: ParagraphStyle) -> Table:
    rows: list[list[Paragraph]] = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        rows.append([Paragraph(inline_markup(c), style) for c in cells])
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def build_story(markdown: str):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=8.5, leading=11, spaceAfter=5)
    small = ParagraphStyle("Small", parent=body, fontSize=7, leading=8)
    code = ParagraphStyle("Code", parent=styles["Code"], fontName="Courier", fontSize=7, leading=8)
    story = []
    lines = markdown.splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), code))
                story.append(Spacer(1, 0.12 * cm))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue
        if not line.strip():
            story.append(Spacer(1, 0.08 * cm))
            i += 1
            continue
        if line.lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            story.append(table_block(block, small))
            story.append(Spacer(1, 0.15 * cm))
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), styles["Heading2"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), styles["Heading3"]))
        elif line.startswith("- "):
            story.append(Paragraph("&#8226; " + inline_markup(line[2:]), body))
        elif re.match(r"^\d+\. ", line):
            story.append(Paragraph(inline_markup(line), body))
        else:
            story.append(Paragraph(inline_markup(line), body))
        i += 1
    return story


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_md")
    parser.add_argument("output_pdf")
    args = parser.parse_args()
    src = Path(args.input_md)
    dst = Path(args.output_pdf)
    dst.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(dst),
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title=src.stem,
    )
    doc.build(build_story(src.read_text(encoding="utf-8")))
    print(dst)


if __name__ == "__main__":
    main()
