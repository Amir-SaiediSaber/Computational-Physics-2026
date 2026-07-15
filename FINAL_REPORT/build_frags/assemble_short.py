"""Assemble the condensed (<10 pp) aanda.tex from the short fragments.

The full-length manuscript remains reproducible via assemble.py; this script
builds the course-format version (single group report under 10 pages).
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
orig = open(os.path.join(ROOT, "aanda_original_backup.tex")).read().split("\n")


def rng(a, b):  # 1-indexed inclusive
    return "\n".join(orig[a - 1:b])


def frag(name):
    return open(os.path.join(HERE, name)).read()


parts = []
# Preamble + \begin{document}
parts.append(rng(1, 32))
# Title block + abstract
parts.append(frag("frag_titleblock.tex"))
parts.append("%-------------------------------------------------------------------")
# 1 Introduction
parts.append(frag("frag_intro_short.tex"))
# 2 Data and shared evaluation protocol
parts.append(frag("frag_data_short.tex"))
# 3 Classical baseline
parts.append(frag("frag_classical_short.tex"))
# 4 Mask R-CNN
parts.append(frag("frag_maskrcnn_short.tex"))
# 5 SmallUNet
parts.append(frag("frag_unet_short.tex"))
# 6 YOLOv8
parts.append(frag("frag_yolo_short.tex"))
parts.append("\\FloatBarrier")
# 7 Head-to-head comparison
parts.append(frag("frag_comparison_short.tex"))
# 8 South pole application
parts.append(frag("frag_southpole_short.tex"))
# 9 Conclusions + acknowledgements
parts.append(frag("frag_conclusions_short.tex"))
# Bibliography: original items + additions, sorted by author label then year
parts.append("\\FloatBarrier")
bib_src = rng(1469, 1507) + "\n" + frag("frag_bib_extra.tex")
entries = ["\\bibitem" + chunk.strip() for chunk in bib_src.split("\\bibitem")[1:]]


def bib_sort_key(entry):
    m = re.match(r"\\bibitem\[([^(]*)\((\d{4})\)", entry)
    return (m.group(1).strip().lower(), m.group(2)) if m else ("", "")


entries.sort(key=bib_sort_key)
parts.append("\\begin{thebibliography}{}")
parts.append("\n\n".join(entries))
parts.append("\\end{thebibliography}")
# Appendices (supplementary figures, tables, and derivations; outside the
# 10-page main body)
parts.append(frag("frag_appendix.tex"))
parts.append("")
parts.append("\\end{document}")

out = "\n\n".join(parts) + "\n"
with open(os.path.join(ROOT, "aanda.tex"), "w") as f:
    f.write(out)
print("assembled:", out.count("\n"), "lines")
