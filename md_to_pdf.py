"""
将 project_report.md 转换为 PDF（通过 Chrome Headless 打印）
"""
import os
import subprocess
import tempfile

import markdown


def md_to_html(md_path, html_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # 简单处理：把 $$...$$ 包裹为 <div class="math"> 以便打印时保留
    import re
    md_text = re.sub(r'\$\$(.+?)\$\$', r'<div class="math">\1</div>', md_text, flags=re.DOTALL)
    md_text = re.sub(r'\$(.+?)\$', r'<span class="math-inline">\1</span>', md_text)

    # 转换 markdown -> html
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
    body = md.convert(md_text)

    # 构建完整 HTML，嵌入 CSS
    css = """
    @page { size: A4; margin: 2.5cm 2cm; }
    body {
        font-family: "Microsoft YaHei", "SimSun", "PingFang SC", sans-serif;
        font-size: 12pt;
        line-height: 1.8;
        color: #333;
    }
    h1 {
        font-size: 22pt;
        text-align: center;
        margin-top: 2cm;
        margin-bottom: 0.8cm;
        font-weight: bold;
    }
    h2 {
        font-size: 16pt;
        margin-top: 1.2cm;
        margin-bottom: 0.5cm;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 4px;
        font-weight: bold;
    }
    h3 {
        font-size: 14pt;
        margin-top: 0.8cm;
        margin-bottom: 0.4cm;
        font-weight: bold;
    }
    p {
        text-indent: 2em;
        margin: 0.4em 0;
    }
    p:has(img) {
        text-indent: 0;
        text-align: center;
    }
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 0.5cm auto;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 0.6cm auto;
        font-size: 11pt;
    }
    th, td {
        border: 1px solid #ccc;
        padding: 6px 10px;
        text-align: center;
    }
    th {
        background-color: #f5f5f5;
        font-weight: bold;
    }
    code {
        font-family: Consolas, "Courier New", monospace;
        background: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 10.5pt;
    }
    pre {
        background: #f4f4f4;
        padding: 12px;
        border-radius: 4px;
        overflow-x: auto;
        font-size: 10pt;
        line-height: 1.5;
    }
    pre code {
        background: transparent;
        padding: 0;
    }
    blockquote {
        border-left: 4px solid #2c3e50;
        margin: 0.5cm 0;
        padding-left: 1em;
        color: #555;
    }
    ul, ol {
        margin: 0.4cm 0 0.4cm 2em;
    }
    li {
        margin: 0.2em 0;
    }
    .math {
        text-align: center;
        font-family: "Times New Roman", serif;
        font-style: italic;
        margin: 0.6cm 0;
        font-size: 13pt;
    }
    .math-inline {
        font-family: "Times New Roman", serif;
        font-style: italic;
    }
    strong {
        font-weight: bold;
    }
    hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 1cm 0;
    }
    """

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>基于手势识别的无接触智慧电梯控制器设计与实现</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


def html_to_pdf(html_path, pdf_path):
    # 尝试 Chrome，备选 Edge
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]

    browser = None
    for p in chrome_paths + edge_paths:
        if os.path.exists(p):
            browser = p
            break

    if not browser:
        raise RuntimeError("未找到 Chrome 或 Edge 浏览器，无法生成 PDF。")

    html_abs = os.path.abspath(html_path)
    pdf_abs = os.path.abspath(pdf_path)

    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf=" + pdf_abs,
        "--print-to-pdf-no-header",
        "file:///" + html_abs.replace("\\", "/"),
    ]

    print(f"使用浏览器: {browser}")
    print(f"正在生成 PDF: {pdf_abs}")
    subprocess.run(cmd, check=True)
    print("PDF 生成成功！")


def main():
    md_path = "project_report.md"
    pdf_path = "project_report.pdf"

    # 使用临时目录存放中间 HTML
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "report.html")

        print("Step 1: Markdown -> HTML ...")
        md_to_html(md_path, html_path)

        print("Step 2: HTML -> PDF (via Chrome Headless) ...")
        html_to_pdf(html_path, pdf_path)

    print(f"\n最终 PDF 已保存: {os.path.abspath(pdf_path)}")


if __name__ == "__main__":
    main()
