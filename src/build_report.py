# -*- coding: utf-8 -*-
"""把 chart_data.json 注入报告模板，生成可直接打开/发布的单文件 HTML。"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
tpl = (BASE / "report" / "eta_report.template.html").read_text(encoding="utf-8")
data = json.loads((BASE / "outputs" / "chart_data.json").read_text(encoding="utf-8"))

# 压缩 JSON，避免模板里出现 </script> 之类的破窗字符
payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = tpl.replace("__CHART_DATA__", payload)

out = BASE / "report" / "eta_report.html"
out.write_text(html, encoding="utf-8")
print(f"written {out}  ({out.stat().st_size/1024:.0f} KB)")
