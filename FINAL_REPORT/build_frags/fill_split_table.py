"""Fill the ZZ placeholders in frag_comparison.tex from pair_results.csv,
then reassemble aanda.tex."""
import csv
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.expanduser(
    "~/lcp_retake_work/Computational-Physics-2026/leakage_exp/pair_results.csv")

rows = {r["run"]: r for r in csv.DictReader(open(CSV))}

MAP = {
    "ZZCRR": ("impact_crater_random__own_val", "map50"),
    "ZZCRRP": ("impact_crater_random__own_val", "precision"),
    "ZZCRRR": ("impact_crater_random__own_val", "recall"),
    "ZZCRS": ("impact_crater_spatial__own_val", "map50"),
    "ZZCRSP": ("impact_crater_spatial__own_val", "precision"),
    "ZZCRSR": ("impact_crater_spatial__own_val", "recall"),
    "ZZCRX": ("impact_crater_random__spatial_val", "map50"),
    "ZZCRXP": ("impact_crater_random__spatial_val", "precision"),
    "ZZCRXR": ("impact_crater_random__spatial_val", "recall"),
    "ZZWRR": ("wrinkle_ridge_random__own_val", "map50"),
    "ZZWRRP": ("wrinkle_ridge_random__own_val", "precision"),
    "ZZWRRR": ("wrinkle_ridge_random__own_val", "recall"),
    "ZZWRS": ("wrinkle_ridge_spatial__own_val", "map50"),
    "ZZWRSP": ("wrinkle_ridge_spatial__own_val", "precision"),
    "ZZWRSR": ("wrinkle_ridge_spatial__own_val", "recall"),
    "ZZWRX": ("wrinkle_ridge_random__spatial_val", "map50"),
    "ZZWRXP": ("wrinkle_ridge_random__spatial_val", "precision"),
    "ZZWRXR": ("wrinkle_ridge_random__spatial_val", "recall"),
}

frag_path = os.path.join(HERE, "frag_comparison.tex")
tex = open(frag_path).read()
tex = tex.replace("ZZEPOCH", "40")
# longest tokens first so ZZCRRP is replaced before ZZCRR
for tok in sorted(MAP, key=len, reverse=True):
    run, field = MAP[tok]
    tex = tex.replace(tok, f"{float(rows[run][field]):.3f}")

disc = open(os.path.join(HERE, "frag_split_discussion.tex")).read()
tex = tex.replace("ZZSPLITDISCUSSION", disc)

open(frag_path, "w").write(tex)
print("placeholders remaining:", tex.count("ZZ"))
subprocess.run(["python3", os.path.join(HERE, "assemble.py")], check=True)
