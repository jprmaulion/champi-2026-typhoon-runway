"""
champi_utils.py: the plumbing behind the Champi notebooks.

The notebook calls a handful of short functions from here so its own cells
stay readable. Everything is plain numpy, pandas, scipy and matplotlib.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Polygon, Patch
from matplotlib.path import Path as MPath
from matplotlib.transforms import Affine2D
from scipy.optimize import curve_fit

# ---------------------------------------------------------------- constants
KT_TO_KMH = 1.852
NM_TO_KM = 1.852
EARTH_R_KM = 6371.0

# Saffir-Simpson style categories on 1-minute winds in knots (JTWC practice)
CAT_EDGES = [0, 34, 64, 83, 96, 113, 137, 999]
CAT_NAMES = ["Depression", "Tropical storm", "Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"]
CAT_SHORT = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"]
SUPER_TYPHOON_KT = 130  # JTWC super typhoon threshold
RI_KT_PER_24H = 30      # rapid intensification threshold

# Emanuel (2000): average intensification and decay rates, m/s per day
EMANUEL_RISE_MS_DAY, EMANUEL_DECAY_MS_DAY = 12.0, 8.0

# Colors: ordinal blue ramp for categories, categorical slots for model lines
CAT_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]  # ColorBrewer Blues 7
MODEL_COLORS = {"Logistic": "#2a78d6", "Gompertz": "#eb6834", "Richards": "#1baf7a",
                "Two-stage logistic": "#6250d6", "Whole-life curve": "#e34948"}
SIZE_COLORS = {"roci": "#52514e", "r34": "#2a78d6", "rmw": "#eb6834"}
INK, MUTED, GRID, SURFACE, LAND, COAST = "#0b0b0b", "#52514e", "#e6e4de", "#fcfcfb", "#ebe8e1", "#b9b5ac"

# A coarse outline of Taiwan's coast, used only if the coastline file is missing
TAIWAN_COAST = np.array([
    (121.53, 25.30), (121.75, 25.15), (122.00, 25.00), (121.85, 24.60), (121.60, 24.00),
    (121.15, 22.75), (120.85, 21.90), (120.60, 22.35), (120.27, 22.60), (120.10, 23.00),
    (120.15, 23.50), (120.35, 24.10), (120.50, 24.30), (120.90, 24.85), (121.05, 25.05),
    (121.40, 25.20)])

# Philippine Area of Responsibility (PAGASA): (lon, lat) corners
PAR_BOX = np.array([(115, 5), (115, 15), (120, 21), (120, 25), (135, 25), (135, 5), (115, 5)])

PLACES = {"Taipei": (121.56, 25.04), "Manila": (120.98, 14.60), "Guam": (144.75, 13.45),
          "Okinawa": (127.70, 26.30), "Majuro": (171.38, 7.09)}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
})

QUADRANTS = ["ne", "se", "sw", "nw"]


# ---------------------------------------------------------------- loading
def load_track(path="data/champi_2026_track.csv") -> pd.DataFrame:
    """Read the trimmed track file and add time in days since the first position.
    Storm-size columns (nautical miles in the file) are added in km when present."""
    df = pd.read_csv(path, parse_dates=["iso_time"])
    df = df.sort_values("iso_time").reset_index(drop=True)
    df["lon"] = df["lon"] % 360
    df["days"] = (df["iso_time"] - df["iso_time"].iloc[0]).dt.total_seconds() / 86400
    df["wind_kmh"] = df["wind_kt"] * KT_TO_KMH
    df["category"] = df["wind_kt"].apply(category_of)
    if "roci_nm" in df:
        df["roci_km"] = df["roci_nm"] * NM_TO_KM
        df["rmw_km"] = df["rmw_nm"] * NM_TO_KM
        for q in QUADRANTS:
            df[f"r34_{q}_km"] = df[f"r34_{q}_nm"].where(df[f"r34_{q}_nm"] > 0) * NM_TO_KM
        quads = df[[f"r34_{q}_km" for q in QUADRANTS]]
        df["r34_mean_km"] = quads.mean(axis=1)
        df["r34_min_km"], df["r34_max_km"] = quads.min(axis=1), quads.max(axis=1)
    return df


def has_size(df) -> bool:
    return "roci_km" in df and df["roci_km"].notna().any()


def load_land(path="data/land_wpac.geojson"):
    """Return a list of coastline rings, or None if the file is missing."""
    p = Path(path)
    if not p.exists():
        return None
    rings = []
    for feat in json.loads(p.read_text())["features"]:
        for poly in feat["geometry"]["coordinates"]:
            ring = np.array(poly[0], dtype=float)
            # move only the far-western longitudes (Pacific islands past the
            # date line) to 180-360; Europe and Africa stay put so big
            # continents are not stretched across the whole map
            ring[:, 0] = np.where(ring[:, 0] < -100, ring[:, 0] + 360, ring[:, 0])
            rings.append(ring)
    return rings


def taiwan_outline(land=None):
    """Taiwan's coastline as (lon, lat) points: from the coastline file when available."""
    if land:
        cands = [r for r in land if 119.5 < r[:, 0].mean() < 122.5 and 21.5 < r[:, 1].mean() < 25.5]
        if cands:
            return max(cands, key=len)
    return TAIWAN_COAST


def category_of(kt: float) -> int:
    return int(np.digitize(kt, CAT_EDGES) - 1)


def data_source_note(df) -> str:
    src = str(df["source"].iloc[0])
    if "bdeck" in src:
        return ("JTWC working best track (b-deck), provisional: 6-hourly positions with storm size. "
                "Agencies finalize the best track months later.")
    if "rainviewer" in src:
        return "Provisional track (6-hourly positions relayed by RainViewer)."
    if "PROVISIONAL" in src.upper():
        return "IBTrACS provisional track: agencies have not finalized Champi's best track yet."
    return "IBTrACS best track."


# ---------------------------------------------------------------- geometry
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Compass bearing from point 1 to point 2 (0 = north, 90 = east)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(x, y)) % 360


def destination(lat, lon, bearing, dist_km):
    """Point reached from (lat, lon) going dist_km along a compass bearing."""
    lat1, lon1, b = np.radians(lat), np.radians(lon), np.radians(bearing)
    d = dist_km / EARTH_R_KM
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(b))
    lon2 = lon1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)


def quadrant_of(bearing):
    return QUADRANTS[int(bearing % 360 // 90)]


def runway_numbers(df, land=None) -> pd.Series:
    """The headline facts about Champi's journey, in one table."""
    step_km = haversine_km(df.lat.values[:-1], df.lon.values[:-1], df.lat.values[1:], df.lon.values[1:])
    step_h = np.diff(df["days"].values) * 24
    hours_at = lambda mask: float(np.sum(step_h[mask[1:]]))  # hours spent in steps ending at that level
    w = df["wind_kt"].values
    i_peak = int(np.argmax(w))
    first_ty = int(np.argmax(w >= 64)) if (w >= 64).any() else None
    coast = taiwan_outline(land)
    d_tw = np.array([haversine_km(la, lo, coast[:, 1], coast[:, 0]).min() for la, lo in zip(df.lat, df.lon)])
    i_tw = int(np.argmin(d_tw))
    change24 = pd.Series(w, index=df["iso_time"]).diff(4)  # 4 six-hour steps = 24 h
    return pd.Series({
        "First position (UTC)": df["iso_time"].iloc[0],
        "First position (lat, lon)": f"{df.lat.iloc[0]:.1f}N, {df.lon.iloc[0]:.1f}E",
        "Last position (UTC)": df["iso_time"].iloc[-1],
        "Days tracked": round(df["days"].iloc[-1], 2),
        "Track length (km)": round(float(step_km.sum())),
        "Straight-line distance (km)": round(float(haversine_km(df.lat.iloc[0], df.lon.iloc[0],
                                                                 df.lat.iloc[-1], df.lon.iloc[-1]))),
        "Peak wind (kt)": int(w[i_peak]),
        "Peak wind (km/h)": round(w[i_peak] * KT_TO_KMH),
        "Lowest pressure (hPa)": int(df["pres_hpa"].min()) if df["pres_hpa"].notna().any() else np.nan,
        "Peak time (UTC)": df["iso_time"].iloc[i_peak],
        "Days from genesis to peak": round(df["days"].iloc[i_peak], 2),
        "Days from typhoon strength to peak": (round(df["days"].iloc[i_peak] - df["days"].iloc[first_ty], 2)
                                                if first_ty is not None else np.nan),
        "Days at typhoon strength or more (64+ kt)": round(hours_at(w >= 64) / 24, 2),
        "Days at major strength (96+ kt)": round(hours_at(w >= 96) / 24, 2),
        "Days at super typhoon strength (130+ kt)": round(hours_at(w >= SUPER_TYPHOON_KT) / 24, 2),
        "Largest 24 h strengthening (kt)": int(change24.max()),
        "Largest 24 h strengthening ends (UTC)": change24.idxmax(),
        "Rapid intensification (30+ kt in 24 h)": bool(change24.max() >= RI_KT_PER_24H),
        "Largest 24 h weakening (kt)": int(-change24.min()),
        "Largest 24 h weakening ends (UTC)": change24.idxmin(),
        "Closest approach to Taiwan coast (km)": round(float(d_tw[i_tw])),
        "Closest approach time (UTC)": df["iso_time"].iloc[i_tw],
    })


# ---------------------------------------------------------------- growth models
# All rise models start at the first observed wind L and level off at a ceiling K.
def logistic(t, K, r, t0, L):
    return L + (K - L) / (1 + np.exp(-r * (t - t0)))


def gompertz(t, K, r, t0, L):
    return L + (K - L) * np.exp(-np.exp(-r * (t - t0)))


def richards(t, K, r, t0, nu, L):
    return L + (K - L) / (1 + nu * np.exp(-r * (t - t0))) ** (1 / nu)


def two_stage(t, K1, r1, t1, D, r2, t2, L):
    """Two logistic surges stacked (bi-logistic): a first climb to a plateau K1,
    then a second climb of D knots, so the final ceiling is K1 + D."""
    return L + (K1 - L) / (1 + np.exp(-r1 * (t - t1))) + D / (1 + np.exp(-r2 * (t - t2)))


def whole_life(t, A, tp, a, L):
    """Skewed bell, the shape hydrologists use for a flood wave: rises to a peak
    of L + A at time tp; the shape number a sets how sharp the peak is."""
    x = np.clip(t / tp, 1e-9, None)
    return L + A * x ** a * np.exp(a * (1 - x))


def aicc(rss, n, k):
    """Akaike information criterion with small-sample correction. Lower is better.
    k counts fitted parameters plus one for the noise level."""
    k = k + 1
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def _max_slope(f, t):
    tt = np.linspace(t.min(), t.max(), 2000)
    return float(np.max(np.gradient(f(tt), tt)))


def _best_fit(g, t, v, starts, bounds):
    """Least squares from several starting guesses; keeps the best."""
    best = None
    for p0 in starts:
        try:
            p, _ = curve_fit(g, t, v, p0=p0, bounds=bounds, maxfev=100000)
        except RuntimeError:
            continue
        rss = float(np.sum((v - g(t, *p)) ** 2))
        if best is None or rss < best[1]:
            best = (p, rss)
    return best


def fit_rise(df):
    """Fit four growth curves to the strengthening phase (genesis to peak):
    three single S-curves and one two-stage curve."""
    i_peak = int(np.argmax(df["wind_kt"].values))
    t = df["days"].values[: i_peak + 1]
    v = df["wind_kt"].values[: i_peak + 1].astype(float)
    L, top, t_end = v[0], v.max(), t[-1]
    lo3, hi3 = [top * 0.8, 0.05, 0], [250, 20, t_end]
    specs = {
        "Logistic": (logistic, [[top, 1.0, t_end / 2], [top, 3.0, t_end * 0.4]], (lo3, hi3)),
        "Gompertz": (gompertz, [[top, 1.0, t_end / 2], [top, 3.0, t_end * 0.4]], (lo3, hi3)),
        "Richards": (richards, [[top, 1.0, t_end / 2, 1.0], [top, 3.0, t_end * 0.4, 3.0]],
                     ([top * 0.8, 0.05, 0, 0.02], [250, 20, t_end, 20])),
        "Two-stage logistic": (two_stage,
                               [[top * f1, r1, t_end * m1, top * (1 - f1), r2, t_end * m2]
                                for f1 in (0.75, 0.85) for r1 in (1.5, 3.0) for m1 in (0.35, 0.45)
                                for r2 in (3.0, 10.0) for m2 in (0.8, 0.9)],
                               ([L + 10, 0.05, 0, 0, 0.05, 0], [top, 30, t_end, 100, 30, t_end + 0.5])),
    }
    rows, curves = [], {}
    for name, (f, starts, bounds) in specs.items():
        g = lambda tt, *p, f=f: f(tt, *p, L)
        p, rss = _best_fit(g, t, v, starts, bounds)
        fitted = lambda tt, p=p, g=g: g(tt, *p)
        curves[name] = fitted
        two = name == "Two-stage logistic"
        rows.append({"model": name, "parameters": len(p),
                     "ceiling_kt": round(p[0] + p[3], 1) if two else round(p[0], 1),
                     "first_plateau_kt": round(p[0], 1) if two else np.nan,
                     "growth_rate_per_day": round(p[1], 3), "midpoint_day": round(p[2], 2),
                     "shape_nu": round(p[3], 3) if name == "Richards" else np.nan,
                     "second_surge_day": round(p[5], 2) if two else np.nan,
                     "second_surge_rate_per_day": round(p[4], 2) if two else np.nan,
                     "fastest_growth_kt_per_day": round(_max_slope(fitted, t), 1),
                     "rmse_kt": round(np.sqrt(rss / len(t)), 2), "AICc": round(aicc(rss, len(t), len(p)), 2)})
    table = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)
    table["delta_AICc"] = (table["AICc"] - table["AICc"].min()).round(2)
    table["akaike_weight"] = (np.exp(-table["delta_AICc"] / 2) / np.exp(-table["delta_AICc"] / 2).sum()).round(3)
    return table, curves, (t, v)


def growth_robustness(df):
    """How much does the two-stage result depend on individual positions? Refits the
    contest after dropping each position in the strengthening phase, and after ending
    the window 6 and 12 hours before the peak. Returns the two-stage lead (AICc points
    over the best single S-curve) for each check; positive = two-stage still wins."""
    import warnings
    warnings.filterwarnings("ignore")

    def lead(d):
        d = d.reset_index(drop=True).copy()
        d["days"] = (d.iso_time - d.iso_time.iloc[0]).dt.total_seconds() / 86400
        t, _, _ = fit_rise(d)
        single = t[t.model != "Two-stage logistic"].iloc[0]
        two = t[t.model == "Two-stage logistic"].iloc[0]
        return round(float(single.AICc - two.AICc), 1)

    ip = int(df.wind_kt.idxmax())
    rise = df.iloc[: ip + 1]
    rows = [{"check": "All positions, genesis to peak (baseline)", "lead_AICc": lead(rise)}]
    drops = [(j, lead(rise.drop(index=j))) for j in range(1, ip + 1)]
    early = [g for j, g in drops if j < ip - 2]
    rows.append({"check": f"Drop any one of the first {len(early)} positions", "lead_AICc": f"{min(early)} to {max(early)}"})
    for j in range(ip - 2, ip + 1):
        r = df.loc[j]
        rows.append({"check": f"Drop the {r.wind_kt} kt position ({r.iso_time:%d %b %H:%M} UTC)",
                     "lead_AICc": dict(drops)[j]})
    for h, k in [(12, 2)]:  # ending 6 h early is the same as dropping the peak position, shown above
        rows.append({"check": f"End the window {h} h before the peak", "lead_AICc": lead(df.iloc[: ip + 1 - k])})
    out = pd.DataFrame(rows)

    def verdict(v):
        if isinstance(v, str):
            lo = float(v.split(" to ")[0])
            return "decisive" if lo >= 10 else ("clear" if lo >= 4 else "weak")
        return "decisive" if v >= 10 else "clear" if v >= 4 else "meaningful" if v >= 2 else "single curve wins" if v < 0 else "tie"
    out["verdict"] = out["lead_AICc"].apply(verdict)
    return out


def residual_lag1(curves, data, model):
    """Correlation between neighboring residuals: near 0 means independent errors."""
    t, v = data
    r = v - curves[model](t)
    return float(np.corrcoef(r[:-1], r[1:])[0, 1])


def fit_whole_life(df):
    """Fit one skewed bell to Champi's entire life."""
    t, v = df["days"].values, df["wind_kt"].values.astype(float)
    L, i_peak = v[0], int(np.argmax(v))
    p, _ = curve_fit(lambda tt, A, tp, a: whole_life(tt, A, tp, a, L), t, v,
                     p0=[v.max() - L, t[i_peak], 5.0],
                     bounds=([10, 0.5, 0.5], [250, t[-1] + 5, 200]), maxfev=50000)
    fitted = lambda tt: whole_life(tt, *p, L)
    rss = float(np.sum((v - fitted(t)) ** 2))
    return {"peak_kt": round(L + p[0], 1), "peak_day": round(p[1], 2), "shape_a": round(p[2], 2),
            "rmse_kt": round(np.sqrt(rss / len(t)), 2), "curve": fitted}


def _rise_rate(df):
    t, v = df["days"].values, df["wind_kt"].values.astype(float)
    i_peak, i_ty = int(np.argmax(v)), int(np.argmax(v >= 64))
    return (v[i_peak] - v[i_ty]) / (t[i_peak] - t[i_ty]) * 0.514444


def rise_and_decay(df):
    """Average strengthening rate (typhoon strength to peak) and weakening rate
    (peak to last position), in m/s per day so they line up with Emanuel (2000)."""
    t, v = df["days"].values, df["wind_kt"].values.astype(float)
    i_peak = int(np.argmax(v))
    rise = _rise_rate(df)
    decay = (v[i_peak] - v[-1]) / (t[-1] - t[i_peak]) * 0.514444
    ends = decay_by_endpoint(df)
    return pd.Series({
        "Rise rate, typhoon strength to peak (m/s per day)": round(rise, 1),
        "Decay rate, peak to last position (m/s per day)": round(decay, 1),
        "Decay / rise (Champi, to last position)": round(decay / rise, 2),
        "Decay / rise, lowest over end points": round(ends["decay_over_rise"].min(), 2),
        "Decay / rise, highest over end points": round(ends["decay_over_rise"].max(), 2),
        "Decay / rise (Emanuel 2000 average)": round(EMANUEL_DECAY_MS_DAY / EMANUEL_RISE_MS_DAY, 2),
        "Wind lost since peak (kt)": int(v[i_peak] - v[-1]),
        "Wind at last position (kt)": int(v[-1]),
    })


def decay_by_endpoint(df, min_hours=24):
    """Decay-to-rise ratio if the decay window ended at each position after the peak
    (windows shorter than min_hours are skipped as too noisy)."""
    t, v = df["days"].values, df["wind_kt"].values.astype(float)
    i_peak = int(np.argmax(v))
    rise = _rise_rate(df)
    rows = []
    for j in range(i_peak + 1, len(v)):
        hours = (t[j] - t[i_peak]) * 24
        if hours < min_hours:
            continue
        decay = (v[i_peak] - v[j]) / (t[j] - t[i_peak]) * 0.514444
        rows.append({"end_time": df["iso_time"].iloc[j], "wind_kt": int(v[j]), "hours_after_peak": hours,
                     "decay_ms_per_day": round(decay, 1), "decay_over_rise": round(decay / rise, 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- storm size
def size_numbers(df) -> pd.Series:
    """Headline facts about Champi's size (needs the size columns)."""
    i_peak = int(df.wind_kt.idxmax())
    g = df.dropna(subset=["r34_mean_km"])
    i_big = int(g["r34_mean_km"].idxmax())
    i_rmw = int(df["rmw_km"].idxmin())
    return pd.Series({
        "First gale-force radius reported (UTC)": g["iso_time"].iloc[0],
        "Gale zone at first report, average diameter (km)": round(2 * g["r34_mean_km"].iloc[0]),
        "Gale zone at peak wind, average diameter (km)": round(2 * df.loc[i_peak, "r34_mean_km"]),
        "Gale zone at its largest, average diameter (km)": round(2 * df.loc[i_big, "r34_mean_km"]),
        "Gale zone largest at (UTC)": df.loc[i_big, "iso_time"],
        "Wind when gale zone was largest (kt)": int(df.loc[i_big, "wind_kt"]),
        "Widest single gale radius (km)": round(df["r34_max_km"].max()),
        "Outer edge (ROCI) diameter, range (km)": f"{round(2 * df['roci_km'].min()):,} to {round(2 * df['roci_km'].max()):,}",
        "Eyewall radius (RMW) at first position (km)": round(df["rmw_km"].iloc[0]),
        "Tightest eyewall radius (km)": round(df.loc[i_rmw, "rmw_km"]),
        "Tightest eyewall first reached (UTC)": df.loc[i_rmw, "iso_time"],
    })


def taiwan_clock(df, land=None):
    """For each position: distance from Champi's center to Taiwan's nearest coast, and
    the gale-force radius in the quadrant facing Taiwan. When the radius is at
    least the distance, gale-force winds reach Taiwan's coast (by JTWC's radii)."""
    coast = taiwan_outline(land)
    rows = []
    for _, r in df.iterrows():
        d = haversine_km(r.lat, r.lon, coast[:, 1], coast[:, 0])
        k = int(np.argmin(d))
        b = float(bearing_deg(r.lat, r.lon, coast[k, 1], coast[k, 0]))
        q = quadrant_of(b)
        reach = r.get(f"r34_{q}_km", np.nan)
        rows.append({"iso_time": r.iso_time, "taiwan_time": r.iso_time + pd.Timedelta(hours=8),
                     "distance_km": round(float(d[k])), "bearing_to_taiwan": round(b), "quadrant": q.upper(),
                     "gale_radius_toward_taiwan_km": round(reach) if pd.notna(reach) else np.nan,
                     "gale_reaches_taiwan": bool(pd.notna(reach) and reach >= d[k]), "wind_kt": int(r.wind_kt)})
    return pd.DataFrame(rows)


def gale_window(clock):
    """First and last position with gale-force winds reaching Taiwan, and the hours in between."""
    hit = clock[clock["gale_reaches_taiwan"]]
    if hit.empty:
        return None
    first, last = hit["iso_time"].iloc[0], hit["iso_time"].iloc[-1]
    return {"first_utc": first, "last_utc": last, "first_taiwan_time": first + pd.Timedelta(hours=8),
            "last_taiwan_time": last + pd.Timedelta(hours=8),
            "hours_first_to_last_position": (last - first).total_seconds() / 3600,
            "closest_km": int(clock["distance_km"].min())}


def par_window(df):
    """First and last position inside the Philippine Area of Responsibility."""
    inside = MPath(PAR_BOX).contains_points(np.column_stack([df.lon, df.lat]))
    if not inside.any():
        return None
    t = df["iso_time"][inside]
    return {"first_position_inside_utc": t.min(), "last_position_inside_utc": t.max(), "positions_inside": int(inside.sum())}


def peak_vs_arrival(df, clock):
    """Champi at its strongest vs Champi at its closest to Taiwan."""
    ip, it = int(df.wind_kt.idxmax()), int(clock.distance_km.idxmin())
    rows = []
    for lab, i in [("Strongest (peak wind)", ip), ("Closest to Taiwan", it)]:
        r = df.loc[i]
        rows.append({"moment": lab, "time_utc": r.iso_time, "taiwan_time": r.iso_time + pd.Timedelta(hours=8),
                     "category": CAT_NAMES[r.category], "wind_kt": int(r.wind_kt), "wind_kmh": round(r.wind_kmh),
                     "pressure_hpa": int(r.pres_hpa),
                     "gale_zone_diameter_km": round(2 * r.r34_mean_km) if pd.notna(r.get("r34_mean_km", np.nan)) else np.nan,
                     "distance_to_taiwan_coast_km": int(clock.loc[i, "distance_km"])})
    return pd.DataFrame(rows).set_index("moment")


def plot_peak_vs_arrival(df, clock, axes=None):
    """Strength and size over time, marking the peak, the closest approach to
    Taiwan, the stretch inside PAR and the hours of gale-force winds on Taiwan."""
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, height_ratios=[1.4, 1])
    ax1, ax2 = axes
    pv = peak_vs_arrival(df, clock)
    par = par_window(df)
    hit = clock[clock["gale_reaches_taiwan"]]
    for a in (ax1, ax2):
        if par:
            a.axvspan(par["first_position_inside_utc"], par["last_position_inside_utc"], color=MUTED, alpha=0.07, lw=0)
        if not hit.empty:
            a.axvspan(hit.iso_time.iloc[0], hit.iso_time.iloc[-1], color=SIZE_COLORS["r34"], alpha=0.12, lw=0)
    for edge, name in zip(CAT_EDGES[1:-1], CAT_NAMES[1:]):
        ax1.axhline(edge, color=GRID, lw=1, zorder=0)
        ax1.text(df.iso_time.iloc[0], edge + 1.5, name, fontsize=7.5, color=MUTED, va="bottom")
    ax1.plot(df.iso_time, df.wind_kt, color=INK, lw=2, zorder=3)
    ax1.set_ylim(0, 185)
    ax1.set_ylabel("Wind (knots)")
    if par:
        ax1.text(par["first_position_inside_utc"], 178, " inside PAR", fontsize=8, color=MUTED, va="top")
    if not hit.empty:
        ax1.text(hit.iso_time.iloc[0], 166, " gale-force winds\n on Taiwan's coast", fontsize=8,
                 color=SIZE_COLORS["r34"], va="top")
    p, c = pv.iloc[0], pv.iloc[1]
    ax1.scatter(p.time_utc, p.wind_kt, marker="*", s=220, color=CAT_COLORS[-1], ec=SURFACE, lw=1, zorder=5)
    ax1.annotate(f"Strongest: {p.category}, {p.wind_kt} kt, {p.pressure_hpa} hPa\n"
                 f"open sea, {p.distance_to_taiwan_coast_km:,} km from Taiwan", (p.time_utc, p.wind_kt),
                 xytext=(-2.6, 0), textcoords=("offset fontsize"), fontsize=8.5, color=INK, ha="right", va="center",
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax1.scatter(c.time_utc, c.wind_kt, s=70, color=CAT_COLORS[3], ec=INK, lw=1, zorder=5)
    ax1.annotate(f"Closest to Taiwan: {c.category}, {c.wind_kt} kt, {c.pressure_hpa} hPa\n"
                 f"{c.distance_to_taiwan_coast_km} km off the coast", (c.time_utc, c.wind_kt),
                 xytext=(c.time_utc - pd.Timedelta(hours=60), 40), fontsize=8.5, color=INK, ha="left",
                 arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax1.set_title("Strongest is not when it hits", loc="left")
    ax1.grid(axis="x", visible=False)
    ax2.plot(df.iso_time, 2 * df.r34_mean_km, color=SIZE_COLORS["r34"], lw=2.2)
    for r, mk, sz in [(p, "*", 220), (c, "o", 70)]:
        y = r.gale_zone_diameter_km
        ax2.scatter(r.time_utc, y, marker=mk, s=sz, color=SIZE_COLORS["r34"], ec=INK if mk == "o" else SURFACE, lw=1, zorder=5)
        ax2.annotate(f"{y:,.0f} km", (r.time_utc, y), xytext=(6, -14), textcoords="offset points", fontsize=8.5, color=INK)
    ax2.set_ylim(0, 1200)
    ax2.set_ylabel("Gale-force zone,\naverage diameter (km)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.set_xlabel("Date (UTC)")
    ax2.grid(axis="x", visible=False)
    return axes


def near_taiwan_numbers(df, clock, within_km=300):
    """Peak wind vs the strongest wind while Champi was within `within_km` of Taiwan's coast."""
    ip = int(df.wind_kt.idxmax())
    near = clock[clock.distance_km <= within_km]
    jn = int(near.wind_kt.idxmax())
    jc = int(clock.distance_km.idxmin())
    row = lambda i: {"time_utc": df.iso_time[i], "taiwan_time": df.iso_time[i] + pd.Timedelta(hours=8),
                     "wind_kt": int(df.wind_kt[i]), "category": CAT_NAMES[df.category[i]],
                     "pressure_hpa": int(df.pres_hpa[i]), "distance_km": int(clock.distance_km[i])}
    return pd.DataFrame({"Peak (anywhere)": row(ip), f"Strongest within {within_km} km of Taiwan": row(jn),
                         "Closest to Taiwan": row(jc)}).T


def plot_wind_vs_distance(df, clock, ax=None, within_km=300,
                          title="The closer Champi got to Taiwan, the weaker it was"):
    """Wind at every position against distance to Taiwan's coast (log scale, Taiwan on the right)."""
    ax = ax or plt.subplots(figsize=(10, 4.8))[1]
    d, w = clock.distance_km.values, df.wind_kt.values
    ax.axvspan(within_km, 30, color=SIZE_COLORS["r34"], alpha=0.10, lw=0)
    for edge, name in zip(CAT_EDGES[1:-1], CAT_NAMES[1:]):
        ax.axhline(edge, color=GRID, lw=1, zorder=0)
        ax.text(6800, edge + 1.5, "TS" if name == "Tropical storm" else name, fontsize=7.5, color=MUTED, va="bottom")
    ax.plot(d, w, color=MUTED, lw=0.8, alpha=0.6, zorder=2)
    ax.scatter(d, w, s=34, c=[CAT_COLORS[c] for c in df.category], ec=INK, lw=0.4, zorder=3)
    nt = near_taiwan_numbers(df, clock, within_km)
    p, n, c = nt.iloc[0], nt.iloc[1], nt.iloc[2]
    ax.annotate(f"Peak: {p.wind_kt} kt ({p.category})\n{p.distance_km:,} km from Taiwan",
                (p.distance_km, p.wind_kt), xytext=(p.distance_km * 1.9, p.wind_kt + 10), fontsize=8.5,
                color=INK, ha="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.annotate(f"Strongest within {within_km} km:\n{n.wind_kt} kt ({n.category}), {n.distance_km} km",
                (n.distance_km, n.wind_kt), xytext=(n.distance_km * 1.25, n.wind_kt + 38), fontsize=8.5,
                color=INK, ha="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.annotate(f"Closest: {c.wind_kt} kt ({c.category}),\n{c.distance_km} km off the coast",
                (c.distance_km, c.wind_kt), xytext=(c.distance_km * 1.05, c.wind_kt - 42), fontsize=8.5,
                color=INK, ha="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.text(within_km * 0.95, 178, f"within {within_km} km\nof Taiwan", fontsize=8, color=SIZE_COLORS["r34"],
            ha="right", va="top")
    ax.set_xscale("log")
    ax.set_xlim(7000, 30)
    ticks = [50, 100, 200, 500, 1000, 2000, 4000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 185)
    ax.set_xlabel("Distance from Champi's center to Taiwan's coast (km, log scale; Taiwan to the right)")
    ax.set_ylabel("Wind (knots)")
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left")
    return ax


def _distance_field(coast, extent, step=0.04):
    lon = np.arange(extent[0], extent[1] + step, step)
    lat = np.arange(extent[2], extent[3] + step, step)
    LO, LA = np.meshgrid(lon, lat)
    # densify the coastline so distances are smooth
    seg = []
    for a, b in zip(coast, np.roll(coast, -1, axis=0)):
        n = max(2, int(np.hypot(*(b - a)) / 0.02))
        seg.append(np.linspace(a, b, n, endpoint=False))
    pts = np.vstack(seg)
    D = np.full(LO.shape, np.inf)
    for k in range(0, len(pts), 200):
        chunk = pts[k:k + 200]
        dd = haversine_km(LA[..., None], LO[..., None], chunk[:, 1], chunk[:, 0]).min(axis=-1)
        D = np.minimum(D, dd)
    return LO, LA, D


def present_category_legend(ax, sub, loc="upper right", fontsize=7.8, title=None, compact=False):
    """Highlighted key listing only the categories the storm held in `sub`, strongest first,
    each with its wind range and the winds the storm actually had in that category."""
    cats = sorted(sub.category.unique(), reverse=True)
    handles = []
    for c in cats:
        w = sorted(sub.wind_kt[sub.category == c].astype(int).unique())
        seen = f"{w[0]} kt" if len(w) == 1 else f"{w[0]} to {w[-1]} kt"
        lo, hi = CAT_EDGES[c], CAT_EDGES[c + 1] - 1
        rng = f"{lo} kt or more" if c == len(CAT_NAMES) - 1 else f"{lo} to {hi} kt"
        lab = CAT_SHORT[c] if compact else f"{CAT_NAMES[c]} ({rng}): {seen}"
        handles.append(plt.Line2D([], [], color=CAT_COLORS[c], lw=3, marker="o", ms=6, mec=INK, mew=0.5, label=lab))
    leg = ax.legend(handles=handles, loc=loc, fontsize=fontsize, frameon=True, ncol=len(cats) if compact else 1,
                    handlelength=1.4 if not compact else 1.0, columnspacing=0.6, borderpad=0.6 if not compact else 0.3,
                    title=title or ("Strength near Taiwan, strongest first\n(category: range · Champi's winds)"
                                    if not compact else "Category near Taiwan"),
                    title_fontsize=fontsize)
    leg.get_title().set_weight("bold")
    leg.get_frame().set_facecolor("#eef4fb")
    leg.get_frame().set_edgecolor(INK)
    leg.get_frame().set_linewidth(1.1)
    return leg


def plot_taiwan_zoom(df, clock, land, ax=None, start="2026-10-22 00:00", rings=(100, 200, 300),
                     title="Champi's last two days near Taiwan", marker="dot", symbol_alpha=0.45, symbol_size=560,
                     compact=False, highlight="ring", legend_mode="present"):
    """Zoomed map of northeast Taiwan: distance rings from the coast and each position
    labeled with Taiwan time and wind. All markers are the same size and solid. The first
    position, the closest approach and the last position get a black ring and bold labels
    with the category (highlight="fade" restores the older faded look). legend_mode="present"
    lists only the categories Champi held near Taiwan, with its winds in each;
    legend_mode="category" shows the full seven-step scale."""
    ax = ax or plt.subplots(figsize=(7.5, 7))[1]
    extent = (118.5, 127.5, 19.5, 29.5)
    _base_map(ax, land, extent=extent, par=False)
    ax.set_xticks(range(119, 128, 2)); ax.set_xticklabels([f"{x}°E" for x in range(119, 128, 2)])
    ax.set_yticks(range(20, 30, 2)); ax.set_yticklabels([f"{y}°N" for y in range(20, 30, 2)])
    LO, LA, D = _distance_field(taiwan_outline(land), extent)
    cs = ax.contour(LO, LA, D, levels=list(rings), colors=[SIZE_COLORS["r34"]], linewidths=0.9,
                    linestyles="dashed", zorder=2)
    ax.clabel(cs, fmt=lambda v: f"{v:.0f} km", fontsize=7.5, inline=True)
    sub = df[df.iso_time >= start]
    first = sub.index[0]
    prev = df.loc[[first - 1]] if first > 0 else sub.iloc[:0]
    seg = pd.concat([prev, sub])
    sym = marker == "symbol"
    for j in range(1, len(seg)):
        ax.plot(seg.lon.values[j - 1:j + 1], seg.lat.values[j - 1:j + 1], color=CAT_COLORS[seg.category.iloc[j]],
                lw=1.4 if sym else 2.6, alpha=0.7 if sym else 1.0, solid_capstyle="round", zorder=4)
    ic = int(clock.distance_km.idxmin())
    key_fixes = {sub.index[0]: "first position shown", ic: "closest to Taiwan", sub.index[-1]: "last position"}
    pad = 15 if sym else 10
    offsets = {"2026-10-23 12:00": (6, 11), "2026-10-23 18:00": (-pad - 2, -16), "2026-10-24 00:00": (pad, -4),
               "2026-10-23 06:00": (pad, -2), "2026-10-24 06:00": (pad, 0), "2026-10-24 12:00": (pad, 0),
               "2026-10-22 00:00": (-pad, -10) if sym else (-pad, -8)}
    ring = highlight == "ring"
    for i, r in sub.iterrows():
        tw = r.iso_time + pd.Timedelta(hours=8)
        hi = i in key_fixes
        if sym:
            _draw_symbol(ax, r.lon, r.lat, CAT_COLORS[r.category], angle_deg=0, hollow=r.wind_kt < 64,
                         size=symbol_size, zorder=6 if hi else 5,
                         alpha=1.0 if (hi or ring) else symbol_alpha)
        else:
            ax.scatter(r.lon, r.lat, s=46, color=CAT_COLORS[r.category], ec=INK,
                       lw=1.4 if (hi and not ring) else 0.5, zorder=6 if hi else 5)
        if hi and ring:
            ax.scatter(r.lon, r.lat, s=(symbol_size if sym else 46) * 1.9, facecolor="none", edgecolor=INK,
                       lw=1.5, zorder=6.5)
        key = f"{r.iso_time:%Y-%m-%d %H:%M}"
        dx, dy = offsets.get(key, (pad, -3))
        text = f"{tw:%d %b %H:%M} · {r.wind_kt} kt"
        if hi:
            text += f" · {CAT_NAMES[r.category]}\n({key_fixes[i]})"
        ax.annotate(text, (r.lon, r.lat), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.3 if hi else 7.8, color=INK, weight="bold" if hi else "normal",
                    ha="left" if dx > 0 else "right", va="center" if not hi else ("top" if dy < 0 else "center"),
                    zorder=7)
    ring_handle = plt.Line2D([], [], marker="o", ls="", ms=12, mfc="none", mec=INK, mew=1.5,
                             label="First, closest and last position (bold label with category)")
    dist_handle = plt.Line2D([], [], color=SIZE_COLORS["r34"], lw=1, ls="--", label="Distance from Taiwan's coast")
    if sym:
        sp = storm_symbol_path()
        extra = [dist_handle,
                 plt.Line2D([], [], marker=sp, ls="", ms=15, mfc=CAT_COLORS[4], mec=SURFACE,
                            alpha=1.0 if ring else symbol_alpha, label="Storm symbol at each 6-hourly position"),
                 ring_handle if ring else
                 plt.Line2D([], [], marker=sp, ls="", ms=15, mfc=CAT_COLORS[4], mec=SURFACE,
                            label="First, closest and last position: solid, bold label with category"),
                 plt.Line2D([], [], marker="o", ls="", ms=0, label="Open center = below typhoon strength (64 kt)")]
    else:
        extra = [dist_handle,
                 plt.Line2D([], [], marker="o", ls="", ms=6, mfc=CAT_COLORS[4], mec=INK, mew=0.5,
                            label="6-hourly position: Taiwan time · wind"),
                 ring_handle if ring else
                 plt.Line2D([], [], marker="o", ls="", ms=6, mfc=CAT_COLORS[3], mec=INK, mew=1.4,
                            label="First, closest and last position: bold label with category")]
    if compact:  # small panels: one legend only
        if legend_mode == "present":
            present_category_legend(ax, sub, loc="upper left", fontsize=6.5, compact=True)
        else:
            leg = ax.legend(handles=[plt.Line2D([], [], color=c, lw=3, label=n) for c, n in zip(CAT_COLORS, CAT_SHORT)],
                            ncol=4, loc="upper left", fontsize=6.5, handlelength=0.8, columnspacing=0.5,
                            title="Category (1-min wind)", title_fontsize=6.5, frameon=True)
            leg.get_frame().set_facecolor(SURFACE)
            leg.get_frame().set_edgecolor(GRID)
        ax.set_title(title, loc="left")
        return ax
    if legend_mode == "present":
        leg = present_category_legend(ax, sub, loc="lower left")
    else:
        leg = ax.legend(handles=[plt.Line2D([], [], color=c, lw=3, label=n) for c, n in zip(CAT_COLORS, CAT_SHORT)],
                        ncol=4, loc="upper right", fontsize=6.5, handlelength=0.8, columnspacing=0.55,
                        title="Category (1-min wind)", title_fontsize=6.5)
        leg.get_frame().set_facecolor(SURFACE)
    ax.add_artist(leg)
    leg2 = ax.legend(handles=extra, loc="upper left", fontsize=7.5, frameon=True)
    leg2.get_frame().set_facecolor(SURFACE)
    leg2.get_frame().set_edgecolor(GRID)
    ax.set_title(title, loc="left")
    return ax


# ---------------------------------------------------------------- map helpers
def _base_map(ax, land, extent=(116, 170, 4, 30), par=False):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    if land:
        for ring in land:
            ax.add_patch(Polygon(ring, closed=True, fc=LAND, ec=COAST, lw=0.6, zorder=1))
    if par:
        ax.plot(PAR_BOX[:, 0], PAR_BOX[:, 1], color=MUTED, lw=0.9, ls=(0, (5, 4)), zorder=2)
        ax.text(134.6, 5.5, "Philippine Area of\nResponsibility (PAR)", fontsize=7, color=MUTED,
                ha="right", va="bottom", zorder=2)
    xt = range(int(np.ceil(extent[0] / 10) * 10), int(extent[1]) + 1, 10)
    ax.set_xticks(list(xt))
    ax.set_xticklabels([f"{x}°E" for x in xt])
    yt = range(int(np.ceil(extent[2] / 5) * 5), int(extent[3]) + 1, 5)
    ax.set_yticks(list(yt))
    ax.set_yticklabels([f"{y}°N" for y in yt])
    for s in ax.spines.values():
        s.set_visible(False)
    for name, (x, y) in PLACES.items():
        if extent[0] <= x <= extent[1] and extent[2] <= y <= extent[3]:
            ax.plot(x, y, "o", ms=3, color=MUTED, zorder=3)
            ax.annotate(name, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8, color=MUTED)


def _track_segments(ax, df, upto=None, lw=2.4, alpha=1.0):
    n = len(df) if upto is None else upto + 1
    for j in range(1, n):
        ax.plot(df.lon.values[j - 1:j + 1], df.lat.values[j - 1:j + 1], alpha=alpha,
                color=CAT_COLORS[df.category.iloc[j]], lw=lw, solid_capstyle="round", zorder=4)


def _category_legend(ax, loc="upper right", extra=None):
    """Compact category key: thin swatches and small type, so it does not compete with the map."""
    handles = [plt.Line2D([], [], color=c, lw=3, label=n) for c, n in zip(CAT_COLORS, CAT_SHORT)]
    leg = ax.legend(handles=handles, ncol=7, loc=loc, fontsize=6.3, handlelength=0.8, handletextpad=0.3,
                    columnspacing=0.55, borderpad=0.3, title="Category (1-min wind)", title_fontsize=6.3)
    if extra:
        ax.add_artist(leg)
        ax.legend(handles=extra, loc="upper right", bbox_to_anchor=(1.0, 0.90), fontsize=7.5, handlelength=1.6)


def gale_polygon(lat, lon, radii_km, step=5):
    """Gale-force zone as a polygon: each quadrant drawn at its own radius."""
    pts = []
    for qi, q in enumerate(QUADRANTS):
        rad = radii_km[q]
        if not np.isfinite(rad) or rad <= 0:
            pts.append((lon, lat))
            continue
        for b in np.arange(qi * 90, qi * 90 + 90 + 1e-9, step):
            la, lo = destination(lat, lon, b, rad)
            pts.append((lo, la))
    return np.array(pts)


def circle_polygon(lat, lon, radius_km, step=5):
    b = np.arange(0, 360 + step, step)
    la, lo = destination(lat, lon, b, radius_km)
    return np.column_stack([lo, la])


def _row_radii(r):
    return {q: r.get(f"r34_{q}_km", np.nan) for q in QUADRANTS}


def plot_track_map(df, land, ax=None, title="Champi's path, colored by strength", show_size_at=None):
    ax = ax or plt.subplots(figsize=(10, 5.6))[1]
    _base_map(ax, land, extent=(114, 166, 4, 32), par=True)
    if show_size_at is not None and has_size(df):
        for i in show_size_at:
            r = df.loc[i]
            poly = gale_polygon(r.lat, r.lon, _row_radii(r))
            ax.add_patch(Polygon(poly, closed=True, fc=CAT_COLORS[r.category], ec=CAT_COLORS[r.category],
                                 alpha=0.14, lw=1, zorder=2))
    _track_segments(ax, df)
    i_peak = int(df.wind_kt.idxmax())
    ax.scatter(df.lon.iloc[0], df.lat.iloc[0], s=60, color=CAT_COLORS[0], ec=INK, lw=0.8, zorder=5)
    ax.annotate(f"genesis, {df.iso_time.iloc[0]:%d %b}", (df.lon.iloc[0], df.lat.iloc[0]), xytext=(-10, -14),
                textcoords="offset points", fontsize=8, color=INK, ha="right")
    ax.scatter(df.lon.iloc[i_peak], df.lat.iloc[i_peak], s=110, marker="*", color=CAT_COLORS[-1],
               ec=SURFACE, lw=1, zorder=6)
    ax.annotate(f"peak {df.wind_kt.iloc[i_peak]} kt", (df.lon.iloc[i_peak], df.lat.iloc[i_peak]),
                xytext=(6, 8), textcoords="offset points", fontsize=8, color=INK)
    extra = None
    if show_size_at is not None and has_size(df):
        extra = [Patch(fc=CAT_COLORS[3], alpha=0.3, label="Gale-force zone (34 kt+) at marked times")]
    _category_legend(ax, extra=extra)
    ax.set_title(title, loc="left")
    return ax


def plot_intensity(df, ax=None, title="Champi's strength over time"):
    ax = ax or plt.subplots(figsize=(10, 4.6))[1]
    for edge, name in zip(CAT_EDGES[1:-1], CAT_NAMES[1:]):
        ax.axhline(edge, color=GRID, lw=1, zorder=0)
        ax.text(df.iso_time.iloc[0], edge + 1.5, name, fontsize=7.5, color=MUTED, va="bottom")
    ax.axhline(SUPER_TYPHOON_KT, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.text(df.iso_time.iloc[-1], SUPER_TYPHOON_KT - 1.5, "super typhoon (JTWC)", fontsize=7.5,
            color=MUTED, ha="right", va="top")
    ax.plot(df.iso_time, df.wind_kt, color=INK, lw=2, zorder=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.scatter(df.iso_time, df.wind_kt, s=14, color=[CAT_COLORS[c] for c in df.category], zorder=4,
               ec=SURFACE, lw=0.5)
    ax.set_ylabel("1-minute sustained wind (knots)")
    ax.set_ylim(0, max(175, df.wind_kt.max() + 10))
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left")
    return ax


def plot_rise_fits(df, table, curves, data, ax=None, title="Four growth curves on the strengthening phase"):
    ax = ax or plt.subplots(figsize=(10, 4.8))[1]
    t, v = data
    tt = np.linspace(0, t.max(), 400)
    ax.scatter(t, v, s=22, color=INK, zorder=5, label="Observed (6-hourly)")
    w = table.set_index("model")["akaike_weight"]
    for name in table["model"]:
        two = name == "Two-stage logistic"
        ax.plot(tt, curves[name](tt), lw=2.6 if two else 1.6, color=MODEL_COLORS[name], zorder=4 if two else 3,
                ls="-" if two else (0, (5, 2)), label=f"{name}  (evidence weight {w[name]:.2f})")
    ax.set_xlabel(f"Days since genesis ({df.iso_time.iloc[0]:%d %b %H:%M} UTC)")
    ax.set_ylabel("Wind (knots)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(title, loc="left")
    return ax


def plot_residuals(table, curves, data, ax=None):
    """Misses of the best single S-curve next to the misses of the two-stage curve."""
    ax = ax or plt.subplots(figsize=(10, 2.8))[1]
    t, v = data
    ax.axhline(0, color=MUTED, lw=0.8)
    single = [m for m in table["model"] if m != "Two-stage logistic"][0]
    ax.bar(t - 0.04, v - curves[single](t), width=0.08, color=MODEL_COLORS[single], label=f"Best single S-curve ({single})")
    ax.bar(t + 0.04, v - curves["Two-stage logistic"](t), width=0.08, color=MODEL_COLORS["Two-stage logistic"],
           label="Two-stage logistic")
    ax.set_xlabel("Days since genesis")
    ax.set_ylabel("Observed minus fit (kt)")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_title("What each curve misses", loc="left")
    return ax


def plot_whole_life(df, life, ax=None, title="One curve for the whole life"):
    ax = ax or plt.subplots(figsize=(10, 4.4))[1]
    tt = np.linspace(0, df.days.max(), 500)
    ax.scatter(df.days, df.wind_kt, s=20, color=INK, zorder=4, label="Observed")
    ax.plot(tt, life["curve"](tt), lw=2, color=MODEL_COLORS["Whole-life curve"], zorder=3,
            label=f"Skewed bell (peak {life['peak_kt']:.0f} kt on day {life['peak_day']:.1f})")
    ax.set_xlabel("Days since genesis")
    ax.set_ylabel("Wind (knots)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(title, loc="left")
    return ax


def plot_decay_endpoints(ends, ax=None, title="Decay-to-rise ratio, depending on where the decay window ends"):
    ax = ax or plt.subplots(figsize=(10, 3.2))[1]
    emanuel = EMANUEL_DECAY_MS_DAY / EMANUEL_RISE_MS_DAY
    ax.axhline(emanuel, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(ends.end_time.iloc[0], emanuel - 0.04, f"Emanuel (2000) average, {emanuel:.2f}", fontsize=8,
            color=MUTED, va="top")
    ax.axhline(1, color=GRID, lw=1)
    ax.plot(ends.end_time, ends.decay_over_rise, color=MODEL_COLORS["Whole-life curve"], lw=2, marker="o", ms=4)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_ylabel("Decay / rise")
    ax.set_ylim(0, max(2.0, ends.decay_over_rise.max() + 0.2))
    ax.set_xlabel("End of decay window (UTC)")
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left")
    return ax


def plot_size(df, axes=None):
    """Wind (top) and storm size (bottom) on one time axis."""
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, height_ratios=[1, 1.5])
    ax1, ax2 = axes
    plot_intensity(df, ax=ax1, title="Strength and size over time")
    ax1.set_xlabel("")
    ax2.plot(df.iso_time, 2 * df.roci_km, color=SIZE_COLORS["roci"], lw=1.6, ls=(0, (5, 3)),
             label="Whole storm: outer closed isobar (ROCI), diameter")
    ax2.fill_between(df.iso_time, 2 * df.r34_min_km, 2 * df.r34_max_km, color=SIZE_COLORS["r34"], alpha=0.18, lw=0,
                     label="Gale-force zone: narrowest to widest quadrant, as diameter")
    ax2.plot(df.iso_time, 2 * df.r34_mean_km, color=SIZE_COLORS["r34"], lw=2.2,
             label="Gale-force zone (34 kt+): average diameter")
    ax2.plot(df.iso_time, 2 * df.rmw_km, color=SIZE_COLORS["rmw"], lw=2,
             label="Eyewall: diameter of maximum winds (2 × RMW)")
    i_peak = int(df.wind_kt.idxmax())
    for a in (ax1, ax2):
        a.axvline(df.iso_time.iloc[i_peak], color=MUTED, lw=0.8, ls=":")
    ax2.set_ylim(0, 1500)
    ax2.text(df.iso_time.iloc[i_peak], 1450, " peak wind", fontsize=8, color=MUTED, va="top")
    ax2.set_ylabel("Diameter (km)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    # legend below the chart so it never sits on the lines
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8, frameon=False)
    ax2.grid(axis="x", visible=False)
    return axes


def plot_taiwan_clock(clock, ax=None, start=None, title="When did gale-force winds reach Taiwan?"):
    ax = ax or plt.subplots(figsize=(10, 4.2))[1]
    c = clock if start is None else clock[clock.iso_time >= start]
    x = c["taiwan_time"]
    ax.plot(x, c["distance_km"], color=INK, lw=2, marker="o", ms=3.5, label="Distance from center to Taiwan's coast")
    ax.plot(x, c["gale_radius_toward_taiwan_km"], color=SIZE_COLORS["r34"], lw=2, marker="o", ms=3.5,
            label="Gale-force radius on the side facing Taiwan")
    hit = c["gale_reaches_taiwan"].values
    ax.fill_between(x, 0, 1, where=hit, transform=ax.get_xaxis_transform(), color=SIZE_COLORS["r34"],
                    alpha=0.12, lw=0, step="mid", label="Gale-force winds reach the coast")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_xlabel("Taiwan time (UTC+8)")
    ax.set_ylabel("km")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left")
    return ax


# ---------------------------------------------------------------- the storm symbol
def storm_symbol_path():
    """A two-armed spiral, the standard weather-map sign for a tropical cyclone
    (the style PAGASA uses on its track maps). Unit size, centered at 0."""
    s = np.linspace(0, 1, 30)
    ang = np.radians(15 + 150 * s)
    rad = 0.32 + 0.68 * s
    width = 0.24 * (1 - s) ** 0.8 + 0.02
    cx, cy = rad * np.cos(ang), rad * np.sin(ang)
    nx, ny = np.cos(ang), np.sin(ang)  # outward normal (approximate)
    left = np.column_stack([cx + nx * width / 2, cy + ny * width / 2])
    right = np.column_stack([cx - nx * width / 2, cy - ny * width / 2])
    arm = np.vstack([left, right[::-1], left[:1]])
    arm2 = -arm
    circ = MPath.circle((0, 0), 0.42)
    verts, codes = [], []
    for poly in (arm, arm2):
        verts += poly.tolist()
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(poly) - 2) + [MPath.CLOSEPOLY]
    verts += circ.vertices.tolist()
    codes += circ.codes.tolist()
    return MPath(verts, codes)


def _draw_symbol(ax, lon, lat, color, angle_deg, hollow, size=620, zorder=8, alpha=1.0):
    path = storm_symbol_path().transformed(Affine2D().rotate_deg(angle_deg))
    light = color in CAT_COLORS[:2]  # the two palest steps get a grey outline so they never vanish
    ax.scatter(lon, lat, s=size, marker=path, color=color, ec=MUTED if light else SURFACE, lw=0.7 if light else 1.0,
               zorder=zorder, alpha=alpha)
    if hollow:  # below typhoon strength: open center, as on weather maps
        ax.scatter(lon, lat, s=size * 0.07, color=SURFACE, zorder=zorder + 1, lw=0)


# ---------------------------------------------------------------- pressure sticks
STICK_DEG_PER_HPA = 0.045  # stick length in degrees of latitude per hPa below the storm's edge
STICK_EDGE_HPA = 1010      # edge pressure to use when the track has no outer-isobar pressure


def _deficit(row):
    edge = row.get("poci_hpa", np.nan)
    edge = STICK_EDGE_HPA if pd.isna(edge) else edge
    return max(float(edge) - float(row["pres_hpa"]), 0.0) if pd.notna(row["pres_hpa"]) else np.nan


def _stick(ax, lon, lat, deficit, **kw):
    h = deficit * STICK_DEG_PER_HPA / 2
    ax.plot([lon, lon], [lat - h, lat + h], solid_capstyle="butt", **kw)
    return lat + h, lat - h


def _pressure_sticks(ax, df, i, fr):
    """One vertical stick per past position (longer = lower central pressure), the current
    one in the category color, and the lifetime low drawn bold once it has happened."""
    i_low = int(df["pres_hpa"].idxmin())
    for j in range(i + 1):
        if j == i_low and i >= i_low:
            continue
        r = df.iloc[j]
        d = _deficit(r)
        if np.isfinite(d):
            _stick(ax, r.lon, r.lat, d, color=MUTED, lw=1.1, alpha=0.6, zorder=6)
    if i >= i_low:  # the lifetime low, bold, labeled for the rest of the animation
        r = df.iloc[i_low]
        top, _ = _stick(ax, r.lon, r.lat, _deficit(r), color=INK, lw=3, zorder=7)
        ax.annotate(f"Lowest of its life: {int(r.pres_hpa)} hPa\n{r.iso_time:%d %b %H:%M} UTC", (r.lon, top),
                    xytext=(r.lon + 3.0, top + 2.6), fontsize=8.5, color=INK, weight="bold",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.9), zorder=9)
    else:  # before the low: a light running label
        jm = int(df["pres_hpa"].iloc[: i + 1].idxmin())
        r = df.iloc[jm]
        top = r.lat + _deficit(r) * STICK_DEG_PER_HPA / 2
        right_side = r.lon < 150  # near the map's right edge, put the label on the left
        ax.annotate(f"Lowest so far: {int(r.pres_hpa)} hPa", (r.lon, top),
                    xytext=(r.lon + (2.0 if right_side else -2.0), top + 2.2), ha="left" if right_side else "right",
                    fontsize=8, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), zorder=9,
                    annotation_clip=False)
    cat = int(df.category.iloc[i])
    d = _deficit({"poci_hpa": fr.get("poci_hpa", np.nan), "pres_hpa": fr.get("pres_hpa", np.nan)})
    if np.isfinite(d):
        _stick(ax, fr.lon, fr.lat, d, color=CAT_COLORS[cat], lw=3, zorder=7.5)
    kx, ky = 146.5, 4.2  # scale key in open ocean, lower right
    ax.plot([kx, kx], [ky, ky + 50 * STICK_DEG_PER_HPA], color=MUTED, lw=1.6)
    ax.text(kx + 0.6, ky + 25 * STICK_DEG_PER_HPA,
            "Pressure stick: longer = lower central pressure\n(this one = 50 hPa below the storm's edge)",
            fontsize=7.5, color=MUTED, va="center")


# ---------------------------------------------------------------- animation
def _interp_track(df, per_step=2):
    """Insert frames between 6-hourly positions (straight-line interpolation) for smooth motion.
    Category and labels always come from the latest real position."""
    cols = ["lat", "lon", "wind_kt", "pres_hpa", "poci_hpa", "roci_km", "rmw_km"] + [f"r34_{q}_km" for q in QUADRANTS]
    cols = [c for c in cols if c in df]
    out = []
    for i in range(len(df)):
        r = df.iloc[i]
        out.append({**{c: r[c] for c in cols}, "iso_time": r.iso_time, "step": i, "real": True})
        if i + 1 < len(df) and per_step > 1:
            n = df.iloc[i + 1]
            for k in range(1, per_step):
                f = k / per_step
                row = {c: (r[c] + f * (n[c] - r[c])) if pd.notna(r[c]) and pd.notna(n[c]) else np.nan for c in cols}
                out.append({**row, "iso_time": r.iso_time + f * (n.iso_time - r.iso_time), "step": i, "real": False})
    return pd.DataFrame(out)


def make_gif(df, land, path="figures/champi_track.gif", per_step=2, ms_per_frame=110, hold_ms=3000, dpi=80,
             marker="symbol", show_size=True, strips=True, pressure_sticks=False):
    """The storm, one 6-hourly position at a time. Defaults: spiral symbol colored by category, gale-force
    zone (34 kt+, each quadrant at its own radius) and outer edge (ROCI) drawn to
    scale, with wind and pressure strips underneath.
    marker="dot" draws a plain dot instead of the symbol; show_size=False drops the
    size shapes; strips=False leaves the map alone. Returns the file path."""
    from PIL import Image
    frames_df = _interp_track(df, per_step)
    size = has_size(df) and show_size
    extent = (114, 166, 3, 34)
    frames = []
    for k, fr in frames_df.iterrows():
        i = int(fr["step"])
        real = df.iloc[i]
        cat = int(real.category)
        if strips:
            fig = plt.figure(figsize=(9, 7.4), dpi=dpi)
            gs = fig.add_gridspec(3, 1, height_ratios=[4.6, 1, 1], hspace=0.12)
            ax = fig.add_subplot(gs[0])
        else:
            fig, ax = plt.subplots(figsize=(9, 5.6), dpi=dpi)
        _base_map(ax, land, extent=extent, par=True)
        _track_segments(ax, df, upto=i, lw=2)
        if not fr["real"] and i + 1 < len(df):  # extend the line to the in-between position
            ax.plot([real.lon, fr.lon], [real.lat, fr.lat], color=CAT_COLORS[int(df.category.iloc[i + 1])],
                    lw=2, solid_capstyle="round", zorder=4)
        if size and pd.notna(fr.get("roci_km", np.nan)):
            ax.add_patch(Polygon(circle_polygon(fr.lat, fr.lon, fr.roci_km), closed=True, fc=MUTED, alpha=0.06,
                                 ec=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2))
            radii = {q: fr.get(f"r34_{q}_km", np.nan) for q in QUADRANTS}
            if any(np.isfinite(v) and v > 0 for v in radii.values()):
                ax.add_patch(Polygon(gale_polygon(fr.lat, fr.lon, radii), closed=True, fc=CAT_COLORS[cat],
                                     alpha=0.30, ec=CAT_COLORS[cat], lw=1, zorder=3))
        if pressure_sticks:
            _pressure_sticks(ax, df, i, fr)
        if marker == "dot":
            ax.scatter(fr.lon, fr.lat, s=160, color=CAT_COLORS[cat], ec=SURFACE, lw=2, zorder=8)
        else:
            _draw_symbol(ax, fr.lon, fr.lat, CAT_COLORS[cat], angle_deg=k * 24, hollow=real.wind_kt < 64)
        extra = [Patch(fc=CAT_COLORS[4], alpha=0.35, label="Gale-force winds (34 kt+), to scale"),
                 plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3)), label="Outer edge of the storm (ROCI)")] if size else None
        _category_legend(ax, extra=extra)
        pres = f" · {int(real.pres_hpa)} hPa" if pd.notna(real.get("pres_hpa", np.nan)) else ""
        gale = ""
        if size and pd.notna(real.get("r34_mean_km", np.nan)):
            gale = f" · gale zone ~{2 * real.r34_mean_km:,.0f} km across"
        ax.set_title(f"Typhoon Champi (2026)   {real.iso_time:%d %b %H:%M} UTC\n"
                     f"{CAT_NAMES[cat]} · {real.wind_kt} kt ({real.wind_kmh:.0f} km/h){pres}{gale}",
                     loc="left", fontsize=10.5)
        for row, (col, lab, inv) in enumerate([("wind_kt", "Wind (kt)", False), ("pres_hpa", "Pressure (hPa)", True)]
                                              if strips else []):
            a = fig.add_subplot(gs[row + 1])
            a.plot(df.iso_time, df[col], color=GRID, lw=1.6)
            a.plot(df.iso_time[: i + 1], df[col][: i + 1], color=INK, lw=1.8)
            a.axvline(fr.iso_time, color=CAT_COLORS[cat], lw=1.5)
            if pd.notna(real[col]):
                a.scatter(real.iso_time, real[col], s=22, color=CAT_COLORS[cat], zorder=5)
            a.set_ylabel(lab, fontsize=8)
            a.tick_params(labelsize=7.5)
            a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            a.grid(axis="x", visible=False)
            if inv:
                a.invert_yaxis()
            if row == 0:
                a.set_xticklabels([])
        fig.text(0.125, 0.012, "Positions every 6 h; in-between frames are interpolated. Source: JTWC working best track.",
                 fontsize=7, color=MUTED)
        if not strips:  # fixed layout: every frame identical, nothing can push the map around
            fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.885)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=96))
    durations = [ms_per_frame] * len(frames)
    durations[-1] = hold_ms
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    return path


# ---------------------------------------------------------------- one-page summary
def _key_numbers_panel(ax, df, table, runway, sizes, gale, near, title="Key numbers"):
    ax.axis("off")
    gap = table.delta_AICc[table.model != "Two-stage logistic"].min()
    lines = [("Track, genesis to exit past Taiwan", f"{runway['Track length (km)']:,} km in {runway['Days tracked']:.0f} days"),
             ("Peak", f"{runway['Peak wind (kt)']} kt ({runway['Peak wind (km/h)']} km/h), {runway['Lowest pressure (hPa)']} hPa"),
             ("Fastest 24 h gain / loss", f"+{runway['Largest 24 h strengthening (kt)']} kt / "
                                          f"-{runway['Largest 24 h weakening (kt)']} kt")]
    if sizes is not None:
        lines.append(("Largest gale-force zone",
                      f"{sizes['Gale zone at its largest, average diameter (km)']:,} km across, a day after the peak"))
    if near is not None:
        lines.append(("Strongest within 300 km of Taiwan", f"{near.iloc[1].wind_kt} kt ({near.iloc[1].category})"))
    if gale is not None:
        lines.append(("Gale winds on Taiwan (Taiwan time)",
                      f"{gale['first_taiwan_time']:%d %b %H:%M} to {gale['last_taiwan_time']:%d %b %H:%M}"))
    lines.append(("Best growth law", f"{table['model'].iloc[0]} (+{gap:.0f} AICc ahead)"))
    ax.set_title(title, loc="left")
    for k, (lab, val) in enumerate(lines):
        y = 0.95 - k * 0.125
        ax.text(0, y, lab, fontsize=9, color=MUTED, transform=ax.transAxes)
        ax.text(0, y - 0.05, val, fontsize=10.5, color=INK, weight="bold", transform=ax.transAxes)


def _size_panel(fig, spec, df, title):
    sg = spec.subgridspec(2, 1, height_ratios=[1, 1.2], hspace=0.1)
    a1, a2 = fig.add_subplot(sg[0]), fig.add_subplot(sg[1], sharex=None)
    i_peak, ib = int(df.wind_kt.idxmax()), int(df.r34_mean_km.idxmax())
    a1.plot(df.iso_time, df.wind_kt, color=INK, lw=1.8)
    a1.set_ylabel("Wind (kt)")
    a1.set_title(title, loc="left")
    a2.plot(df.iso_time, 2 * df.r34_mean_km, color=SIZE_COLORS["r34"], lw=2)
    a2.set_ylabel("Gale zone (km)")
    for a in (a1, a2):
        a.axvline(df.iso_time[i_peak], color=MUTED, lw=0.8, ls=":")
        a.grid(axis="x", visible=False)
        a.set_xlim(df.iso_time.iloc[0], df.iso_time.iloc[-1])
        a.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    a1.set_xticklabels([])
    a1.annotate("peak wind", (df.iso_time[i_peak], df.wind_kt[i_peak]), xytext=(-50, -6), textcoords="offset points",
                fontsize=8, color=INK)
    a2.scatter(df.iso_time[ib], 2 * df.r34_mean_km[ib], s=30, color=SIZE_COLORS["r34"], ec=INK, zorder=5)
    a2.annotate(f"widest: {2 * df.r34_mean_km[ib]:,.0f} km", (df.iso_time[ib], 2 * df.r34_mean_km[ib]),
                xytext=(-82, -24), textcoords="offset points", fontsize=8, color=INK)


def graphical_abstract(df, land, table, curves, data, runway, rates, sizes=None, gale=None,
                       path="figures/graphical_abstract.png"):
    """One-page summary in three acts: where (map and key numbers), then
    1. it grew in two surges, 2. its winds peaked but it kept getting bigger, 3. it reached
    Taiwan as a Cat 2, not a Cat 5."""
    size = has_size(df)
    clock = taiwan_clock(df, land)
    near = near_taiwan_numbers(df, clock) if size else None
    fig = plt.figure(figsize=(18, 12.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.25, wspace=0.22)
    marks = [int(df.wind_kt.idxmax()), int(clock.distance_km.idxmin())] if size else None
    plot_track_map(df, land, ax=fig.add_subplot(gs[0, :2]),
                   title="Where: born near the Marshall Islands, peaked at sea, brushed Taiwan", show_size_at=marks)
    _key_numbers_panel(fig.add_subplot(gs[0, 2]), df, table, runway, sizes, gale, near)
    ax_g = fig.add_subplot(gs[1, 0])
    plot_rise_fits(df, table, curves, data, ax=ax_g, title="1. It grew in two surges")
    ax_g.legend(loc="upper left", fontsize=7.5)
    gap = table.delta_AICc[table.model != "Two-stage logistic"].min()
    ax_g.text(0.98, 0.04, f"Two surges beat one S-curve\nby {gap:.0f}+ AICc points (decisive)", transform=ax_g.transAxes,
              ha="right", va="bottom", fontsize=8.5, color=MODEL_COLORS["Two-stage logistic"], weight="bold")
    if size:
        _size_panel(fig, gs[1, 1], df, "2. Its winds peaked, but it kept getting bigger")
        plot_taiwan_zoom(df, clock, land, ax=fig.add_subplot(gs[1, 2]), compact=True,
                         title="3. It reached Taiwan as a Cat 2, not a Cat 5")
    fig.suptitle("Does a Typhoon Grow Like a Population? Growth-Curve Models of Typhoon Champi (2026)",
                 x=0.04, ha="left", fontsize=15, weight="bold", color=INK)
    fig.text(0.04, 0.012, data_source_note(df) + "  Winds: JTWC 1-minute sustained, knots. "
             "Panel 3 labels: Taiwan time and wind; bold = first position shown, closest approach, last position.",
             fontsize=8, color=MUTED)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
