"""Assemble the restructured aanda.tex from the original + new fragments."""
import os

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
# Title block + abstract (new)
parts.append(frag("frag_titleblock.tex"))
parts.append("%-------------------------------------------------------------------")
# 1 Introduction (new)
parts.append(frag("frag_intro.tex"))
# 2 Data and shared evaluation protocol (new; absorbs old dataset subsection)
parts.append(frag("frag_data.tex"))
# 3 Classical baseline (new)
parts.append(frag("frag_classical.tex"))
# 4 Mask R-CNN section (original lines 92-368) + label
maskrcnn = rng(92, 368)
maskrcnn = maskrcnn.replace(
    "\\section{Instance segmentation of lunar surface features with a custom Mask R-CNN}",
    "\\section{Instance segmentation with a custom Mask R-CNN}\n\\label{sec:maskrcnn}")
parts.append(maskrcnn)
# 5 SmallUNet section: header+motivation (369-375), bridge, then 445-756, 816-873
unet_head = rng(369, 375).replace(
    "\\section{Lunar Surface Feature Segmentation}",
    "\\section{Semantic segmentation with a compact U-Net (SmallUNet)}")
parts.append(unet_head)
parts.append(
    "The seven target classes, the tile format, and the label characteristics "
    "are those of Sect.~\\ref{sec:mr_dataset}; because multiple classes can "
    "coexist in a single pixel (e.g.\\ a pit on the rim of a crater), the task "
    "is posed as independent binary classification per class rather than a "
    "single-label softmax problem.\n")
parts.append(rng(445, 756))
parts.append(rng(816, 873))
# 6 YOLO section: new head, original body 888-1176, ensemble subsection, barrier
parts.append(frag("frag_yolo_head.tex"))
yolo_body = rng(888, 1176)
yolo_body = yolo_body.replace("\\subsubsection{", "\\subsection{")
yolo_body = yolo_body.replace(
    "(the right) Full dataset: recall grows slowly to $\\sim$0.05, \n    reflecting the dominance of the impact crater class. (The left) Reduced dataset: \n    recall grows steadily to $\\sim$0.043, indicating substantially \n    improved detection capability across the remaining six classes.",
    "Right: full dataset --- recall grows slowly to $\\sim$0.05,\n    reflecting the dominance of the impact crater class. Left: reduced\n    dataset --- recall grows steadily to $\\sim$0.043, now spread across the\n    six rare classes rather than concentrated on craters.")
parts.append(yolo_body)
parts.append(frag("frag_yolo_ensemble.tex"))
parts.append("\\FloatBarrier")
# 7 Head-to-head comparison (new)
parts.append(frag("frag_comparison.tex"))
# 8 South pole application (new; absorbs old south-pole subsection + Giuseppe pipeline)
parts.append(frag("frag_southpole.tex"))
# 9 Conclusions + acknowledgements (new)
parts.append(frag("frag_conclusions.tex"))
# Bibliography: original items + additions
parts.append("\\FloatBarrier")
parts.append("\\begin{thebibliography}{}")
parts.append(rng(1469, 1507))
parts.append(frag("frag_bib_extra.tex"))
parts.append("\\end{thebibliography}")
parts.append("")
parts.append("\\end{document}")

out = "\n\n".join(parts) + "\n"
with open(os.path.join(ROOT, "aanda.tex"), "w") as f:
    f.write(out)
print("assembled:", out.count("\n"), "lines")
