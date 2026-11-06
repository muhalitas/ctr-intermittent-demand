#!/usr/bin/env python3
"""Syntetos-Boylan demand classification — definitive computation.

Standard formulas (Syntetos, Boylan, & Croston, 2005):
  ADI  = total_periods / number_of_periods_with_demand
       = average inter-arrival period in time-units
  CV²  = ( σ(non-zero demand sizes) / μ(non-zero demand sizes) )²
       = squared coefficient of variation of non-zero demand sizes only

Classification cut-offs (Syntetos & Boylan, 2005):
  Smooth        : ADI ≤ 1.32  AND  CV² ≤ 0.49
  Erratic       : ADI ≤ 1.32  AND  CV² >  0.49
  Intermittent  : ADI >  1.32  AND  CV² ≤ 0.49
  Lumpy         : ADI >  1.32  AND  CV² >  0.49

Input  : Veri/filtered_data.csv (URUNID, Date, Satis Adet)
Output : sb_classification_definitive.csv (per-SKU)
         + counts per category printed to stdout.
"""
import pandas as pd
import numpy as np
import os

INPUT  = "/sessions/loving-clever-volta/mnt/My Drive/doktora/Veri/filtered_data.csv"
OUTPUT = "/sessions/loving-clever-volta/mnt/outputs/sb_classification_definitive.csv"

print(f"Loading {INPUT} ...")
df = pd.read_csv(INPUT, usecols=["URUNID", "Date", "Satis Adet"])
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["URUNID", "Date"])

print(f"Total rows         : {len(df):,}")
print(f"Unique SKUs        : {df['URUNID'].nunique():,}")
print(f"Date range         : {df['Date'].min().date()}  to  {df['Date'].max().date()}")
print()

# ------- Per-SKU classification -------
records = []
for pid, g in df.groupby("URUNID"):
    y = g["Satis Adet"].values.astype(float)
    n_periods = len(y)                    # T
    n_demand  = int((y > 0).sum())        # nd
    if n_demand < 2:                      # not enough demand events to classify
        continue
    nz = y[y > 0]                          # non-zero demand sizes
    adi = n_periods / n_demand
    if nz.mean() > 0:
        cv2 = (nz.std(ddof=1) / nz.mean()) ** 2     # ddof=1 → sample CV² (Syntetos/Boylan convention)
    else:
        cv2 = 0.0
    if   adi <= 1.32 and cv2 <= 0.49: cat = "Smooth"
    elif adi <= 1.32 and cv2 >  0.49: cat = "Erratic"
    elif adi >  1.32 and cv2 <= 0.49: cat = "Intermittent"
    else:                              cat = "Lumpy"
    records.append({
        "URUNID":    pid,
        "T":         n_periods,
        "n_demand":  n_demand,
        "ADI":       round(adi, 4),
        "CV2":       round(cv2, 4),
        "sparsity":  round((y == 0).mean(), 4),
        "Class":     cat,
    })
res = pd.DataFrame(records)
res.to_csv(OUTPUT, index=False)

# ------- Summary -------
print("Syntetos-Boylan classification (definitive)")
print("-" * 60)
print(f"  Total SKUs evaluated          : {len(res):,}")
print(f"  Excluded (n_demand<2)         : {df['URUNID'].nunique() - len(res):,}")
print()
counts = res["Class"].value_counts().reindex(
    ["Smooth", "Erratic", "Intermittent", "Lumpy"], fill_value=0
)
for cat in ["Smooth", "Erratic", "Intermittent", "Lumpy"]:
    n = int(counts[cat])
    pct = 100.0 * n / len(res) if len(res) else 0
    print(f"  {cat:14s} : {n:>5,d}  ({pct:5.1f}%)")
print(f"  {'TOTAL':14s} : {len(res):>5,d}")
print()

# ------- Distribution diagnostics -------
print("ADI distribution:")
print(res["ADI"].describe().round(2).to_string())
print()
print("CV² distribution:")
print(res["CV2"].describe().round(4).to_string())
print()
print("Sparsity (zero-rate) distribution:")
print(res["sparsity"].describe().round(4).to_string())
print()

# ------- Subset that the paper analyzes -------
intermittent_only = res[res["Class"] == "Intermittent"]
lumpy_only        = res[res["Class"] == "Lumpy"]
sparse_pop        = res[(res["ADI"] > 1.32)]
print(f"Strict 'Intermittent' subset (paper-correct)        : {len(intermittent_only):,}")
print(f"Strict 'Lumpy' subset                                : {len(lumpy_only):,}")
print(f"Combined sparse / slow-moving (ADI>1.32, any CV²)    : {len(sparse_pop):,}")
print()
print(f"Per-SKU classification written to: {OUTPUT}")
