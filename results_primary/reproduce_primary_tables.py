"""Recompute the primary-catalogue median tables from the released per-SKU files."""
import pandas as pd

m = pd.read_csv("per_sku_metrics.csv.gz")
c = pd.read_csv("sku_characteristics.csv.gz")

print("Table 2 - median metrics (clipped):")
print(m.groupby("model")[["MAE", "sMAPE", "MASE", "RMSSE"]].median().round(3).to_string(), "\n")

print("Table 6 - median MAE before clipping:")
print(m.groupby("model")["MAE_unclipped"].median().round(3).to_string(), "\n")

x = m.merge(c[["sku", "sparsity_band_table3"]], on="sku")
print("Table 3 - median MAE by sparsity band:")
print(x.pivot_table(index="sparsity_band_table3", columns="model", values="MAE", aggfunc="median").round(3).to_string(), "\n")

w = m.pivot(index="sku", columns="model", values="MAE")
print("Table 4 - paired per-SKU MAE (CTR better / worse / tie):")
for comp in ["FNN", "RNN", "LSTM", "SES", "Croston", "SBA"]:
    d = w[comp] - w["CTR"]
    print(f"  CTR vs {comp:8s} {int((d > 0).sum()):5d} / {int((d < 0).sum()):5d} / {int((d == 0).sum()):5d}")
