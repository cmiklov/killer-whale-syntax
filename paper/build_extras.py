#!/usr/bin/env python3
"""
Build PDF versions of the executive summary and open day briefing.

Usage:
  python build_extras.py          # builds both PDFs
"""

import os
import markdown
from weasyprint import HTML

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS = {
    "EXECUTIVE_SUMMARY.md": "EXECUTIVE_SUMMARY.pdf",
}

CSS = """
@page {
    size: A4;
    margin: 2cm 2.5cm;
    @bottom-center {
        content: counter(page);
        font-family: "Times New Roman", Times, serif;
        font-size: 9pt;
        color: #666;
    }
}

body {
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 11pt;
    line-height: 1.5;
    max-width: 42em;
    margin: 0 auto;
    padding: 0;
    color: #1a1a1a;
}

h1 {
    font-size: 18pt;
    text-align: center;
    margin-bottom: 0.2em;
}

h1 + h3 {
    text-align: center;
    font-weight: normal;
    font-style: italic;
    font-size: 12pt;
    margin-top: 0;
    margin-bottom: 0.5em;
}

/* Author block centering */
h3 + p, h3 + p + p {
    text-align: center;
}

h2 {
    font-size: 14pt;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    border-bottom: none;
}

h3 {
    font-size: 12pt;
    margin-top: 1.2em;
    margin-bottom: 0.4em;
}

p {
    text-align: justify;
    margin-bottom: 0.8em;
    text-indent: 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 9pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #999;
    padding: 3px 6px;
    text-align: left;
}

th {
    background: #f0f0f0;
    font-weight: bold;
}

tr:nth-child(even) {
    background: #fafafa;
}

code {
    font-family: "Courier New", monospace;
    font-size: 10pt;
    background: #f5f5f5;
    padding: 1px 3px;
}

em {
    font-style: italic;
}

strong {
    font-weight: bold;
}

ol, ul {
    margin-bottom: 0.8em;
}

li {
    margin-bottom: 0.3em;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 2em 0;
}

a {
    color: #1a1a1a;
    text-decoration: none;
}

/* Keep tables together */
table {
    page-break-inside: avoid;
}

/* Horizontal rule as subtle separator */
hr + p, hr + h2 {
    margin-top: 0.5em;
}
"""


def build_doc(md_file, pdf_file):
    md_path = os.path.join(PAPER_DIR, md_file)
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found")
        return False

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    md = markdown.Markdown(extensions=["tables", "smarty"])
    body_html = md.convert(content)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{md_file.replace('.md', '').replace('_', ' ').title()}</title>
    <style>{CSS}</style>
</head>
<body>
{body_html}
</body>
</html>"""

    pdf_path = os.path.join(PAPER_DIR, pdf_file)
    HTML(string=html).write_pdf(pdf_path)
    print(f"Written: {pdf_path}")
    return True


def main():
    for md_file, pdf_file in DOCS.items():
        build_doc(md_file, pdf_file)


if __name__ == "__main__":
    main()
