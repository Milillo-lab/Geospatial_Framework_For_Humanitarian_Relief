## D) Relief Calculation

# =============================== D0) Imports ===============================
import os
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

# (Used only if we need to aggregate from buildings)
import geopandas as gpd
import yaml

# ============== D1) Inputs, Outputs, and Constants  =============
OUT_BASE    = r"./Outputs"
C_OUT_DIR   = os.path.join(OUT_BASE, "C")
IMAGES_DIR  = os.path.join(OUT_BASE, "Images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# Fallback shapefile from Section C (buildings with Volume + Pop_Est + Damaged)
C_BLD_POP_SHP = os.path.join(C_OUT_DIR, "Evaluated_Buildings_withVol_Pop.shp")

# WHO / Sphere-like planning constants (with 10% buffer)
GUIDELINES = require("GUIDELINES", dict)

# Planning durations
DUR_AFFECTED   = require("DUR_AFFECTED", int)   # days
DUR_UNAFFECTED = require("DUR_UNAFFECTED", int) # days

# =================== D2) Get population by damage class ======================
# We need totals for Damaged==yes and Damaged==no.

def _load_buildings_memory_or_c():
    # Prefer in-memory 'b' from Section C, else read C outputs shapefile
    if "b" in globals() and b is not None and not b.empty:
        return b
    if os.path.exists(C_BLD_POP_SHP):
        return gpd.read_file(C_BLD_POP_SHP)
    raise FileNotFoundError("No buildings with Pop_Est available in memory or at Outputs_Final\\C.")

b_src = _load_buildings_memory_or_c()
if "Damaged" not in b_src.columns or "Pop_Est" not in b_src.columns:
    raise ValueError("Buildings data must include 'Damaged' and 'Pop_Est' columns.")

# Normalize labels and compute class totals
b_src["Damaged"] = b_src["Damaged"].astype(str).str.strip().str.lower()
affected_pop   = int(pd.to_numeric(b_src.loc[b_src["Damaged"] == "yes", "Pop_Est"], errors="coerce").fillna(0).sum())
unaffected_pop = int(pd.to_numeric(b_src.loc[b_src["Damaged"] == "no",  "Pop_Est"], errors="coerce").fillna(0).sum())
total_pop      = affected_pop + unaffected_pop

print(f" Population — Affected (90d): {affected_pop:,} | Unaffected (14d): {unaffected_pop:,} | Total: {total_pop:,}")

# ========================== D3) Supply calculators ===========================
def calc_water_liters(pop, days):
    return pop * GUIDELINES["water_l_per_person_per_day"] * days * GUIDELINES["buffer_factor"]

def calc_food_kcal(pop, days):
    return pop * GUIDELINES["food_kcal_per_person_per_day"] * days * GUIDELINES["buffer_factor"]

def calc_shelter_area(pop):
    # Shelter is capacity-like (no duration multiplier), buffer not usually applied
    return pop * GUIDELINES["shelter_m2_per_person"]

def calc_iehk_kits(pop, days):
    if pop <= 0:
        return 0
    kits_fraction = (pop / GUIDELINES["iehk_pop_served"]) * (days / GUIDELINES["iehk_duration_days"])
    return math.ceil(kits_fraction)

# =================== D4) Compute items for both conditions =====================
records = []

# Affected (90d)
aw_liters  = calc_water_liters(affected_pop, DUR_AFFECTED)
af_kcal    = calc_food_kcal(affected_pop, DUR_AFFECTED)
as_m2      = calc_shelter_area(affected_pop)
ai_kits    = calc_iehk_kits(affected_pop, DUR_AFFECTED)

records.append({
    "Class": "Affected (90d)", "Days": DUR_AFFECTED, "Population": affected_pop,
    "Water_L": aw_liters, "Water_m3": aw_liters/1000.0,
    "Food_kcal": af_kcal, "Food_Mkcal": af_kcal/1e6,
    "Shelter_m2": as_m2, "IEHK_kits": ai_kits
})

# Unaffected (14d)
uw_liters  = calc_water_liters(unaffected_pop, DUR_UNAFFECTED)
uf_kcal    = calc_food_kcal(unaffected_pop, DUR_UNAFFECTED)
us_m2      = calc_shelter_area(unaffected_pop)
ui_kits    = calc_iehk_kits(unaffected_pop, DUR_UNAFFECTED)

records.append({
    "Class": "Unaffected (14d)", "Days": DUR_UNAFFECTED, "Population": unaffected_pop,
    "Water_L": uw_liters, "Water_m3": uw_liters/1000.0,
    "Food_kcal": uf_kcal, "Food_Mkcal": uf_kcal/1e6,
    "Shelter_m2": us_m2, "IEHK_kits": ui_kits
})

df_supply = pd.DataFrame.from_records(records)

# Totals for final summary
totals = {
    "Population_total": total_pop,
    "Water_L_total":   df_supply["Water_L"].sum(),
    "Water_m3_total":  df_supply["Water_m3"].sum(),
    "Food_kcal_total": df_supply["Food_kcal"].sum(),
    "Food_Mkcal_total":df_supply["Food_Mkcal"].sum(),
    "Shelter_m2_total":df_supply["Shelter_m2"].sum(),
    "IEHK_kits_total": int(df_supply["IEHK_kits"].sum()),  # already ceil per class
}

# ========================= D5) All Bar Charts ===============================
# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------
def _series_with_total(df, colname):
    vals = {
        "Affected (90d)":   float(df.loc[df["Class"] == "Affected (90d)", colname].iloc[0]) if (df["Class"] == "Affected (90d)").any() else 0.0,
        "Unaffected (14d)": float(df.loc[df["Class"] == "Unaffected (14d)", colname].iloc[0]) if (df["Class"] == "Unaffected (14d)").any() else 0.0,
    }
    vals["Total"] = vals["Affected (90d)"] + vals["Unaffected (14d)"]
    labels = ["Affected (90d)", "Unaffected (14d)", "Total"]
    values = [vals[l] for l in labels]
    return labels, values


def _bar_and_save(labels, values, title, ylabel, fname, value_fmt="{:,.0f}"):
    COLORS = {
        "Affected (90d)": "#d73027",   # red
        "Unaffected (14d)": "#9e9e9e", # gray
        "Damaged": "#d73027",
        "Undamaged": "#9e9e9e",
        "Total": "#2e2e2e",            # dark
    }
    bar_colors = [COLORS.get(l, "#9e9e9e") for l in labels]

    plt.figure(figsize=(6, 4))  # small inline view
    bars = plt.bar(labels, values, edgecolor="black", color=bar_colors)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h, value_fmt.format(h),
                 ha="center", va="bottom", fontsize=9)
    plt.title(title, fontsize=12)
    plt.ylabel(ylabel)
    plt.tight_layout()

    out_png = Path(IMAGES_DIR) / fname
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    try:
        display(Image(filename=str(out_png), width=300))
    except Exception:
        pass
    print(f" Saved: {out_png}")


# ---------------------------------------------------------------------------
#  1) Water (m³)
# ---------------------------------------------------------------------------
_labels, _vals = _series_with_total(df_supply, "Water_m3")
_bar_and_save(_labels, _vals, "Water Required (m³)", "Total Water (m³)",
              "D_Water_required_m3_from_memory.png", "{:,.0f}")

# ---------------------------------------------------------------------------
#  2) Food (Million kcal)
# ---------------------------------------------------------------------------
_labels, _vals = _series_with_total(df_supply, "Food_Mkcal")
_bar_and_save(_labels, _vals, "Food Required (Million kcal)", "Total Food (Million kcal)",
              "D_Food_required_MillionKcal_from_memory.png", "{:,.0f}")

# ---------------------------------------------------------------------------
#  3) Shelter (m²)
# ---------------------------------------------------------------------------
_labels, _vals = _series_with_total(df_supply, "Shelter_m2")
_bar_and_save(_labels, _vals, "Shelter Area Needed (m²)", "Total Shelter (m²)",
              "D_Shelter_area_required_m2_from_memory.png", "{:,.0f}")

# ---------------------------------------------------------------------------
#  4) IEHK kits
# ---------------------------------------------------------------------------
_labels, _vals = _series_with_total(df_supply, "IEHK_kits")
_bar_and_save(_labels, _vals, "Medical IEHK Kits Required", "Equivalent IEHK Kits",
              "D_IEHK_kits_required_from_memory.png", "{:,.0f}")

# ---------------------------------------------------------------------------
#  5) Buildings: Damaged vs Undamaged + Total
# ---------------------------------------------------------------------------
_b = b_src if ("b_src" in globals() and b_src is not None) else _load_buildings_memory_or_c()
_dam = int((_b["Damaged"].astype(str).str.lower() == "yes").sum())
_und = int((_b["Damaged"].astype(str).str.lower() == "no").sum())
labels_bld = ["Damaged", "Undamaged", "Total"]
vals_bld   = [_dam, _und, _dam + _und]

_bar_and_save(labels_bld, vals_bld,
              "Buildings: Damaged vs Undamaged",
              "Number of Buildings",
              "D_Buildings_Damaged_vs_Undamaged.png", "{:,.0f}")

# ---------------------------------------------------------------------------
#  6) Population: Affected vs Unaffected + Total
# ---------------------------------------------------------------------------
labels_pop = ["Affected (90d)", "Unaffected (14d)", "Total"]
vals_pop   = [affected_pop, unaffected_pop, total_pop]

_bar_and_save(labels_pop, vals_pop,
              "Population: Affected vs Unaffected",
              "People",
              "D_Population_Affected_vs_Unaffected.png", "{:,.0f}")

# ========================= D6) Summary =========================
import pandas as pd
from pathlib import Path

# --- Helper formatter ---
def _fmt(x, unit=""):
    if isinstance(x, (int, float)):
        return f"{x:,.0f}{unit}"
    return str(x)

print("\n===== HUMANITARIAN SUPPLY SUMMARY  =====")

# Prepare list for export table
summary_rows = []

# Per horizon (Affected / Unaffected)
for _, row in df_supply.iterrows():
    cls  = row["Class"]
    days = row["Days"]
    pop  = int(row["Population"])
    summary_rows.append({
        "Category": cls,
        "People": pop,
        "Duration_days": days,
        "Water_L": round(row["Water_L"], 2),
        "Water_m3": round(row["Water_m3"], 2),
        "Food_kcal": round(row["Food_kcal"], 2),
        "Food_Mkcal": round(row["Food_Mkcal"], 2),
        "Shelter_m2": round(row["Shelter_m2"], 2),
        "IEHK_kits": int(row["IEHK_kits"]),
    })

    print(f"\n— {cls} —")
    print(f"  People:            {_fmt(pop)}")
    print(f"  Duration:          {days} days")
    print(f"  Water:             {_fmt(row['Water_L'], ' L')}  ({_fmt(row['Water_m3'])} m³)")
    print(f"  Food:              {_fmt(row['Food_kcal'])} kcal  ({_fmt(row['Food_Mkcal'])} million kcal)")
    print(f"  Shelter:           {_fmt(row['Shelter_m2'])} m² (min {_fmt(GUIDELINES['shelter_m2_per_person'])} m²/person)")
    print(f"  IEHK Kits (ceil):  {_fmt(row['IEHK_kits'])}")

# Totals (for console + CSV)
summary_rows.append({
    "Category": "TOTAL",
    "People": int(totals["Population_total"]),
    "Duration_days": "Mixed (90+14)",
    "Water_L": round(totals["Water_L_total"], 2),
    "Water_m3": round(totals["Water_m3_total"], 2),
    "Food_kcal": round(totals["Food_kcal_total"], 2),
    "Food_Mkcal": round(totals["Food_Mkcal_total"], 2),
    "Shelter_m2": round(totals["Shelter_m2_total"], 2),
    "IEHK_kits": int(totals["IEHK_kits_total"]),
})

print("\n— TOTALS (Affected 90d + Unaffected 14d) —")
print(f"  Total People:      {_fmt(totals['Population_total'])}")
print(f"  Total Water:       {_fmt(totals['Water_L_total'], ' L')}  ({_fmt(totals['Water_m3_total'])} m³)")
print(f"  Total Food:        {_fmt(totals['Food_kcal_total'])} kcal  ({_fmt(totals['Food_Mkcal_total'])} million kcal)")
print(f"  Total Shelter:     {_fmt(totals['Shelter_m2_total'])} m²")
print(f"  Total IEHK Kits:   {_fmt(totals['IEHK_kits_total'])}")

print("\nAssumptions: 20 L p⁻¹ d⁻¹ water, 2,100 kcal p⁻¹ d⁻¹ food, 4 m² p⁻¹ shelter; "
      "IEHK = 10 000 people per 90 days; 10 % buffer applied to water & food.")
print("Durations: Affected = 90 days, Unaffected = 14 days.")
print("================================================================")

# --- Save summary to CSV (under Outputs_Final\D) ---
D_OUT_DIR = Path(OUT_BASE) / "D"
D_OUT_DIR.mkdir(parents=True, exist_ok=True)
out_summary_csv = D_OUT_DIR / "D_Supply_Summary_Table.csv"

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(out_summary_csv, index=False)

print(f" Summary table saved → {out_summary_csv}")
display(df_summary)
