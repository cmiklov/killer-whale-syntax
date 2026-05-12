#!/usr/bin/env python3
"""
Build the orca paper from markdown sections into formatted output.

Usage:
  python build_paper.py              # outputs PAPER.html
  python build_paper.py --pdf        # outputs PAPER.pdf (requires weasyprint)
  python build_paper.py --print      # outputs PAPER_PRINT.pdf (compact, no TOC links, page numbers)
"""

import sys
import os
import markdown

SECTIONS = [
    "00_abstract.md",
    "01_introduction.md",
    "02_background.md",
    "03_data.md",
    "04_methods.md",
    "05_results.md",
    "06_discussion.md",
    "07_conservation.md",
    "08_limitations.md",
    "09_conclusion.md",
    "10_references.md",
]

CSS = """
@page {
    size: A4;
    margin: 2.5cm;
}

body {
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 12pt;
    line-height: 1.6;
    max-width: 42em;
    margin: 0 auto;
    padding: 2em;
    color: #1a1a1a;
}

h1 {
    font-size: 18pt;
    text-align: center;
    margin-bottom: 0.2em;
    page-break-before: avoid;
}

h1 + h3 {
    text-align: center;
    font-weight: normal;
    font-style: italic;
    font-size: 12pt;
    margin-top: 0;
    margin-bottom: 0.5em;
}

h3 + p > strong:first-child {
    /* Author line */
}

/* Author block centering */
h3 + p, h3 + p + p, h3 + p + p + p {
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
    font-size: 10pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #999;
    padding: 4px 8px;
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

/* Section numbers */
h1 {
    counter-reset: h2counter;
}

/* Page breaks before major sections */
h1:not(:first-of-type) {
    page-break-before: always;
}

/* Abstract styling */
h2:first-of-type + p {
    font-size: 11pt;
}

/* Title page */
.title-page {
    page-break-after: always;
    text-align: center;
    padding-top: 20%;
}

.title-page h1 {
    font-size: 22pt;
    margin-bottom: 0.3em;
}

.title-page h3 {
    font-size: 13pt;
    font-weight: normal;
    font-style: italic;
    margin-bottom: 2em;
}

.title-page p {
    text-align: center;
    font-size: 12pt;
}

/* Table of contents */
nav.toc {
    page-break-after: always;
    margin-top: 4em;
}

nav.toc h2 {
    text-align: center;
    font-size: 16pt;
}

nav.toc ol {
    list-style: none;
    padding-left: 0;
    font-size: 12pt;
    line-height: 2.2;
}

nav.toc li {
    border-bottom: none;
    overflow: hidden;
}

nav.toc li a {
    display: block;
    overflow: hidden;
}

nav.toc a {
    text-decoration: none;
    color: #1a1a1a;
}

/* References */
h1:last-of-type ~ p {
    text-indent: -2em;
    padding-left: 2em;
    font-size: 10pt;
    line-height: 1.4;
}
"""

def build():
    paper_dir = os.path.dirname(os.path.abspath(__file__))

    # Concatenate all sections
    full_md = []
    for section in SECTIONS:
        path = os.path.join(paper_dir, section)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                full_md.append(f.read())
        else:
            print(f"Warning: {section} not found, skipping")

    combined = "\n\n---\n\n".join(full_md)

    # Convert to HTML
    md = markdown.Markdown(extensions=["tables", "smarty"])
    body_html = md.convert(combined)

    # Build table of contents
    toc = """<nav class="toc">
<h2>Contents</h2>
<ol>
<li><a href="#abstract">Abstract</a></li>
<li><a href="#1-introduction">1. Introduction</a></li>
<li><a href="#2-background-and-related-work">2. Background and Related Work</a></li>
<li><a href="#3-data">3. Data</a></li>
<li><a href="#4-methods">4. Methods</a></li>
<li><a href="#5-results">5. Results</a></li>
<li><a href="#6-discussion">6. Discussion</a></li>
<li><a href="#7-conservation-applications">7. Conservation Applications</a></li>
<li><a href="#8-limitations">8. Limitations</a></li>
<li><a href="#9-conclusion">9. Conclusion</a></li>
<li><a href="#references">References</a></li>
</ol>
</nav>"""

    # Add IDs to headings for TOC links
    import re
    def slugify(text):
        text = re.sub(r'<[^>]+>', '', text)
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[\s]+', '-', text)
        return text

    def add_ids(html_text):
        def replacer(m):
            tag = m.group(1)
            attrs = m.group(2) or ''
            content = m.group(3)
            slug = slugify(content)
            return f'<{tag}{attrs} id="{slug}">{content}</{tag}>'
        return re.sub(r'<(h[12])((?:\s[^>]*)?)>(.*?)</\1>', replacer, html_text)

    body_html = add_ids(body_html)

    # Extract title block (everything before the first <h2>) for title page
    title_split = body_html.split('<h2', 1)
    if len(title_split) == 2:
        title_block = title_split[0]
        rest_html = '<h2' + title_split[1]
    else:
        title_block = ''
        rest_html = body_html

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topological Syntax in Killer Whale Communication</title>
    <style>{CSS}</style>
</head>
<body>
<div class="title-page">
{title_block}
</div>
{toc}
{rest_html}
</body>
</html>"""

    # Write HTML
    html_path = os.path.join(paper_dir, "PAPER.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Written: {html_path}")

    # PDF if requested
    if "--pdf" in sys.argv or "--print" in sys.argv:
        try:
            from weasyprint import HTML
            pdf_path = os.path.join(paper_dir, "PAPER.pdf")
            HTML(string=html).write_pdf(pdf_path)
            print(f"Written: {pdf_path}")
        except Exception as e:
            print(f"PDF generation failed: {e}")
            print("HTML file is still available.")

    # Print-ready version
    if "--print" in sys.argv:
        print_css = CSS + """
/* Print overrides */
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
    font-size: 11pt;
    line-height: 1.5;
    padding: 0;
}

nav.toc {
    page-break-after: always;
}

nav.toc a {
    color: #1a1a1a;
    text-decoration: none;
}

nav.toc a::after {
    content: target-counter(attr(href), page);
    float: right;
    font-variant-numeric: tabular-nums;
}

/* Remove link styling for print */
a {
    color: #1a1a1a;
    text-decoration: none;
}

/* Tighter tables for print */
th, td {
    padding: 3px 6px;
    font-size: 9pt;
}

/* Smaller references for print */
h1:last-of-type ~ p {
    font-size: 9pt;
    line-height: 1.3;
}

/* Section breaks */
h1:not(:first-of-type) {
    page-break-before: always;
}

/* Keep tables together */
table {
    page-break-inside: avoid;
}
"""
        print_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Topological Syntax in Killer Whale Communication</title>
    <style>{print_css}</style>
</head>
<body>
<div class="title-page">
{title_block}
</div>
{toc}
{rest_html}
</body>
</html>"""

        # Write the print HTML so an external renderer (e.g. Chrome headless
        # --print-to-pdf) can be used when weasyprint's GTK dependencies
        # aren't available on this system.
        print_html_path = os.path.join(paper_dir, "PAPER_PRINT.html")
        with open(print_html_path, "w", encoding="utf-8") as f:
            f.write(print_html)
        print(f"Written: {print_html_path}")

        # Attempt weasyprint PDF generation; fall back gracefully if GTK
        # libraries are not installed.
        try:
            from weasyprint import HTML as WHTML
            print_path = os.path.join(paper_dir, "PAPER_PRINT.pdf")
            WHTML(string=print_html).write_pdf(print_path)
            print(f"Written: {print_path}")
        except Exception as e:
            print(f"Print PDF generation via weasyprint failed: {e}")
            print(f"  -> use Chrome/Edge to render PAPER_PRINT.html to PDF.")

if __name__ == "__main__":
    build()
