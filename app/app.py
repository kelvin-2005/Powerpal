# --- path shim so we can import from project root packages (core/*) ---
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# std / third-party imports
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from core.model import SoftmaxShareModel, CLASS_ORDER
# local modules
from core.config import (
    DEFAULT_TARIFF_AED_PER_KWH, CO2_KG_PER_KWH,
    DEFAULT_DIRICHLET_LAMBDA, BOUNDS
)
from core.features import Profile, build_feature_vector
from core.model import SoftmaxShareModel, CLASS_ORDER
from core.calibrator import load_lambda, dirichlet_mean_ci
from core.savings import bill_to_kwh, ac_savings, led_savings
from core.online_learner import fit_or_update
from core.bandit import recommend

# --- Paths ---
BASE = Path(__file__).resolve().parents[1]
DATA_SYN = BASE / "data" / "synthetic"
USER_DIR = BASE / "data" / "user"
USER_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = DATA_SYN / "softmax_model.json"
STATE_PATH = USER_DIR / "model_state.json"
ACTIONS_PATH = USER_DIR / "actions.csv"
HISTORY_PATH = USER_DIR / "history.csv"

# --- Streamlit page setup ---
st.set_page_config(page_title="Powerpal", page_icon="⚡", layout="wide")
st.title("⚡ Powerpal — SDG7")
st.caption("Supervised softmax • Dirichlet-calibrated uncertainty • Contextual bandit • Optional online personalization")

# ---------------- helpers ----------------
CATEGORY_COLORS = {
    "ac": "#1f77b4",        # blue
    "lighting": "#f2c744",  # yellow
    "appliances": "#2ca02c",
    "other": "#8884d8",
}

def append_action(action_type: str, delta, est_kwh_saved: float, est_aed_saved: float):
    """Append a committed action to data/user/actions.csv."""
    row = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "action_type": action_type,
        "delta": delta,
        "est_kwh_saved": round(float(est_kwh_saved), 2),
        "est_aed_saved": round(float(est_aed_saved), 2),
        "source": "model"
    }
    if ACTIONS_PATH.exists():
        df = pd.read_csv(ACTIONS_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(ACTIONS_PATH, index=False)

def build_export_assets(fig, df_break, month_aed, month_kwh, total_aed, total_kwh, co2_kg) -> tuple[bytes, bytes]:
    """
    Returns (png_bytes, pdf_bytes) for the current summary.
    Export-only tweaks: white background, fixed colorway, readable text.
    """
    export_fig = go.Figure(fig)
    export_fig.update_layout(
        template=None,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="black",
        legend_font_color="black",
        legend_bgcolor="white",
    )
    export_fig.update_traces(
        textfont_color="black",
        marker=dict(line=dict(color="white", width=1))
    )
    png_bytes: bytes = export_fig.to_image(format="png", scale=2)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, H - 2.2*cm, "Smart Energy Advisor — Summary")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, H - 2.8*cm, "AI-driven split (Dirichlet) • Actionable savings")

    y = H - 4.2*cm
    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Key Metrics")
    c.setFont("Helvetica", 11); y -= 0.8*cm
    c.drawString(2*cm, y, f"Savings this month: {month_aed:.0f} AED  |  {month_kwh:.0f} kWh"); y -= 0.6*cm
    c.drawString(2*cm, y, f"Cumulative savings: {total_aed:.0f} AED  |  {total_kwh:.0f} kWh"); y -= 0.6*cm
    c.drawString(2*cm, y, f"CO₂ avoided: {co2_kg:.0f} kg")

    y -= 1.1*cm
    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Energy Breakdown (this month)")
    y -= 0.7*cm; c.setFont("Helvetica", 10)
    c.drawString(2*cm, y, "Category"); c.drawString(7*cm, y, "kWh (mean)"); c.drawString(11*cm, y, "AED")
    c.line(2*cm, y-0.1*cm, 18.5*cm, y-0.1*cm); y -= 0.6*cm
    for _, r in df_break.round(1).iterrows():
        c.drawString(2*cm, y, str(r["category"]).title())
        c.drawString(7*cm, y, f'{r["kwh"]:.1f}')
        c.drawString(11*cm, y, f'{r["aed"]:.1f}')
        y -= 0.5*cm
        if y < 6*cm:
            break

    img = ImageReader(io.BytesIO(png_bytes))
    c.drawImage(img, 9*cm, 4.2*cm, width=9.5*cm, height=9.5*cm, preserveAspectRatio=True, mask='auto')

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(2*cm, 1.5*cm, "Softmax shares • Dirichlet credible intervals • Contextual bandit recommendations")

    c.showPage()
    c.save()
    pdf_bytes: bytes = buf.getvalue()
    buf.close()
    return png_bytes, pdf_bytes

def persona_level(n: int) -> str:
    if n >= 12: return "High"
    if n >= 6:  return "Medium"
    if n >= 3:  return "Low"
    return "Very low"

def persona_emoji(level: str) -> str:
    return {"High":"🟢", "Medium":"🟡", "Low":"🟠", "Very low":"🔴"}.get(level, "🔴")

# ------------------- Load model + calibration -------------------
from core.model import SoftmaxShareModel, CLASS_ORDER
# Prefer GBM if present
try:
    from core.model_gbm import GBMShareModel
except Exception:
    GBMShareModel = None

GBM_MODEL_PATH = DATA_SYN / "softmax_model_lgbm.json"

if GBM_MODEL_PATH.exists() and GBMShareModel is not None:
    model = GBMShareModel().load(str(GBM_MODEL_PATH))
    st.caption("Model: LightGBM (centered logits)")
elif MODEL_PATH.exists():
    model = SoftmaxShareModel()
    model.load(str(MODEL_PATH))
    st.caption("Model: Ridge (centered logits)")
else:
    st.warning("No model found. Run scripts/train_lgbm.py (preferred) or scripts/train_model.py.")
    st.stop()
# λ from JSON (and default fallback)
lmb = load_lambda(str(STATE_PATH), DEFAULT_DIRICHLET_LAMBDA)

# τ (temperature) from JSON if present
tau = 1.0
cal_meta_cov = None
try:
    with open(str(STATE_PATH), "r") as _f:
        _state = json.load(_f)
        tau = float(_state.get("temperature_tau", 1.0))
        cal_meta_cov = _state.get("calibration_meta", {}).get("achieved_coverage", None)
except Exception:
    pass

# ------------------------ Sidebar inputs ------------------------
st.sidebar.header("Inputs")

bill = st.sidebar.number_input(
    "Monthly bill (AED)",
    min_value=BOUNDS.min_bill, max_value=BOUNDS.max_bill, value=400.0, step=10.0,
    help="Your total monthly electricity bill in AED. Used to estimate total kWh and savings potential."
)

tariff = st.sidebar.number_input(
    "Tariff (AED/kWh)",
    min_value=BOUNDS.min_tariff, max_value=BOUNDS.max_tariff, value=DEFAULT_TARIFF_AED_PER_KWH, step=0.01,
    help="Price per kWh charged by your utility (DEWA/SEWA/FEWA/AADC). This converts bills to energy (kWh)."
)

home_type = st.sidebar.selectbox(
    "Home type", ["apartment", "villa"], index=0,
    help="Dwelling type. Villas usually have more AC & lighting demand than apartments."
)

size = st.sidebar.selectbox(
    "Home size", ["S","M","L"], index=1,
    help="Approximate floor area category. Larger homes consume more AC & lighting energy."
)

occupants = st.sidebar.number_input(
    "Occupants",
    min_value=BOUNDS.min_occupants, max_value=BOUNDS.max_occupants, value=3, step=1,
    help="Number of people living in the home. Affects lighting and appliances load."
)

setpoint_cur = st.sidebar.number_input(
    "AC setpoint (current °C)",
    min_value=BOUNDS.min_setpoint, max_value=BOUNDS.max_setpoint, value=24, step=1,
    help="Your thermostat’s current temperature setting. Lower values mean higher AC consumption."
)

setpoint_target = st.sidebar.number_input(
    "AC setpoint (target °C)",
    min_value=BOUNDS.min_setpoint, max_value=BOUNDS.max_setpoint, value=25, step=1,
    help="The temperature you’re willing to move to. Raising setpoint saves ~4% AC energy per °C."
)

led_pct = st.sidebar.slider(
    "Current LED share (%)",
    min_value=BOUNDS.min_led_pct, max_value=BOUNDS.max_led_pct, value=50, step=5,
    help="Approximate % of your light bulbs already LED. Higher % means fewer savings left."
)

target_led_pct = st.sidebar.slider(
    "Target LED share (%)",
    min_value=max(led_pct, BOUNDS.min_led_pct), max_value=100, value=100, step=5,
    help="Where you want to reach with LED lighting. Difference from current → target drives savings."
)

# --------------------- Pro mode (experimental) ---------------------
with st.sidebar.expander("⚙️ Pro mode (experimental)", expanded=False):
    st.caption("Nudges Lighting/Appliances shares based on simple occupancy proxies (hrs/day).")
    occ_living = st.slider("Living room occupancy (hrs/day)", 0, 16, 6)
    occ_bed    = st.slider("Bedrooms occupancy (hrs/day)",    0, 16, 8)
    occ_kitchen= st.slider("Kitchen use (hrs/day)",           0, 12, 2)
    occ_other  = st.slider("Other spaces (hrs/day)",          0, 16, 4)
    # Convert to gentle multipliers around a baseline (8 hrs/day nominal)
    # Lighting tied to (living + bedrooms + other); Appliances tied to kitchen + occupants.
    base_occ = 8.0
    light_mult = (0.7 + 0.3 * ((occ_living + occ_bed + occ_other) / max(base_occ*3, 1e-6)))
    apps_mult  = (0.8 + 0.2 * ((occ_kitchen / max(2.0, 1e-6)) * (0.6 + 0.4 * (occupants/4.0))))
    # Clamp to sensible band
    light_mult = float(np.clip(light_mult, 0.7, 1.3))
    apps_mult  = float(np.clip(apps_mult, 0.7, 1.3))
    pro_enabled = st.checkbox("Enable Pro mode nudges", value=False)
    if pro_enabled:
        st.caption(f"Applying nudges: Lighting ×{light_mult:.2f}, Appliances ×{apps_mult:.2f}")

# UAE seasonal hint (May–Oct)
if pd.Timestamp.now().month in [5,6,7,8,9,10]:
    st.info("🌞 UAE summer months: AC usually dominates usage; AC-related actions can have outsized impact now.")

# ---------------- Optional: Personalize with history ------------
with st.expander("Optional: Personalize with past months (online learning)"):
    df_hist = None
    hist_n = 0
    if HISTORY_PATH.exists():
        df_hist = pd.read_csv(HISTORY_PATH)
        hist_n = len(df_hist)
        st.dataframe(df_hist.tail(5), use_container_width=True)

    colh1, colh2, colh3 = st.columns([1,1,2])
    with colh1:
        add_hist = st.button("Add current month to history")
    with colh2:
        clear_hist = st.button("Clear history")

    st.markdown("**Bulk import past months (CSV)**")
    up = st.file_uploader(
        "Upload CSV with columns: month,bill_aed,tariff,home_type,size,occupants,setpoint,led_pct",
        type=["csv"]
    )
    if up is not None:
        try:
            new_hist = pd.read_csv(up)
            required_cols = {"month","bill_aed","tariff","home_type","size","occupants","setpoint","led_pct"}
            if not required_cols.issubset(set(new_hist.columns)):
                st.error(f"CSV must include columns: {', '.join(sorted(required_cols))}")
            else:
                new_hist["month"] = pd.to_datetime(new_hist["month"], format="%Y-%m").dt.strftime("%Y-%m")
                if HISTORY_PATH.exists():
                    cur = pd.read_csv(HISTORY_PATH)
                    merged = pd.concat([cur, new_hist], ignore_index=True)
                else:
                    merged = new_hist.copy()
                merged = merged.drop_duplicates(subset=["month"]).sort_values("month")
                merged.to_csv(HISTORY_PATH, index=False)
                st.success(f"Imported {len(new_hist)} rows. History now has {len(merged)} month(s).")
                st.rerun()
        except Exception as e:
            st.error(f"Import failed: {e}")

    st.markdown("**Or generate 12-month UAE-style seasonal placeholders (demo)**")
    if st.button("Generate 12-month seasonal placeholders"):
        base_tariff = float(tariff)
        months = pd.date_range(end=pd.Timestamp.today().replace(day=1), periods=12, freq="MS")
        rows = []
        for m in months:
            hot = m.month in [5,6,7,8,9,10]
            bill_guess = float(bill) * (1.20 if hot else 0.85)
            rows.append({
                "month": m.strftime("%Y-%m"),
                "bill_aed": round(bill_guess, 0),
                "tariff": base_tariff,
                "home_type": home_type,
                "size": size,
                "occupants": occupants,
                "setpoint": 24 if hot else 23,
                "led_pct": led_pct
            })
        demo_df = pd.DataFrame(rows)
        if HISTORY_PATH.exists():
            cur = pd.read_csv(HISTORY_PATH)
            merged = pd.concat([cur, demo_df], ignore_index=True)
        else:
            merged = demo_df
        merged = merged.drop_duplicates(subset=["month"]).sort_values("month")
        merged.to_csv(HISTORY_PATH, index=False)
        st.success(f"Added {len(demo_df)} demo months. Replace with real bills when available.")
        st.rerun()

    if add_hist:
        row = {
            "month": pd.Timestamp.today().strftime("%Y-%m"),
            "bill_aed": bill, "tariff": tariff,
            "home_type": home_type, "size": size, "occupants": occupants,
            "setpoint": setpoint_cur, "led_pct": led_pct
        }
        if df_hist is None:
            df_hist = pd.DataFrame([row])
        else:
            df_hist = pd.concat([df_hist, pd.DataFrame([row])], ignore_index=True)
        df_hist.to_csv(HISTORY_PATH, index=False)
        st.rerun()

    if clear_hist:
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
        st.rerun()

# Personalization badge
hist_n = len(pd.read_csv(HISTORY_PATH)) if HISTORY_PATH.exists() else 0
level = persona_level(hist_n)
st.caption(f"Personalization level: {persona_emoji(level)} **{level}** (based on {hist_n} month(s) of history)")

# Uncertainty badge
badge = f"Uncertainty calibration: λ={lmb:.0f}"
if abs(tau - 1.0) > 1e-12:
    badge += f" • τ={tau:.2f}"
if cal_meta_cov is not None:
    badge += f" • achieved 95% coverage: {cal_meta_cov:.3f}"
st.caption(badge)

# ---------------------- Predict + calibrate ----------------------
profile = Profile(
    bill_aed=bill, tariff=tariff, home_type=home_type, size=size,
    occupants=occupants, setpoint=setpoint_cur, led_pct=led_pct
)

# NEW: context extras for the model
month_now = int(pd.Timestamp.now().month)
cdd_proxy_now = 8.0 if month_now in [5,6,7,8,9,10] else 2.5
x = build_feature_vector(profile, extra={"month": month_now, "cdd_proxy": cdd_proxy_now})

shares = model.predict_shares(x)
# ---- Optional residual corrector (tiny MLP) to nudge logits ----
try:
    from joblib import load as _joblib_load
    rc_path = DATA_SYN / "residual_corrector.joblib"
    if rc_path.exists():
        mlp = _joblib_load(rc_path)
        # base logits from current shares
        import numpy as _np
        eps=1e-6
        p_arr = _np.array([shares[c] for c in CLASS_ORDER], dtype=float)
        z = _np.log(p_arr + eps); z = z - z.mean()
        r = mlp.predict(x.reshape(1, -1))[0]  # residual logits
        z_corr = z + r
        z_corr = z_corr - z_corr.mean()
        p_corr = _np.exp(z_corr); p_corr = p_corr / p_corr.sum()
        shares = {c: float(p_corr[i]) for i, c in enumerate(CLASS_ORDER)}
except Exception:
    # fail-safe: ignore corrector if anything goes wrong
    pass

# Apply temperature scaling to means if τ != 1
if abs(tau - 1.0) > 1e-12:
    p = np.array([shares[c] for c in CLASS_ORDER], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p ** tau
    p = p / p.sum()
    shares = {c: float(p[i]) for i, c in enumerate(CLASS_ORDER)}

# Apply PRO MODE nudges (gentle, then renormalize)
if 'pro_enabled' in locals() and pro_enabled:
    shares_mod = shares.copy()
    shares_mod["lighting"]   = shares_mod.get("lighting", 0.0)   * float(light_mult)
    shares_mod["appliances"] = shares_mod.get("appliances", 0.0) * float(apps_mult)
    # keep AC/Other constant share mass then renormalize all
    p = np.array([shares_mod[c] for c in CLASS_ORDER], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    shares = {c: float(p[i]) for i, c in enumerate(CLASS_ORDER)}

# Dirichlet-calibrated mean + CI
calib = dirichlet_mean_ci(shares, lmb, ci=0.95)

kwh_total = bill_to_kwh(bill, tariff)
rows = []
for c in CLASS_ORDER:
    mean, lo, hi = calib[c]
    rows.append({
        "category": c,
        "share": mean,
        "kwh": mean * kwh_total,
        "kwh_lo": lo * kwh_total,
        "kwh_hi": hi * kwh_total,
        "aed": mean * kwh_total * tariff
    })
df_break = pd.DataFrame(rows).sort_values("kwh", ascending=False)

# ------------------ Learn elasticities & compute savings --------
if HISTORY_PATH.exists() and hist_n > 0:
    df_hist_load = pd.read_csv(HISTORY_PATH)
    params = fit_or_update(df_hist_load)
    beta_ac = params["beta_ac_per_c"]
    led_eff = params["led_efficacy"]
    personalized = True
else:
    beta_ac = 0.04  # 4%/°C rule of thumb
    led_eff = 0.70
    personalized = False

# AC savings (from current -> target setpoint)
kwh_ac = float(df_break.loc[df_break["category"]=="ac","kwh"].values[0]) if "ac" in df_break["category"].values else 0.0
delta_c = max(0, setpoint_target - setpoint_cur)
kwh_save_ac = ac_savings(kwh_ac, delta_c, elasticity=beta_ac)
aed_save_ac = kwh_save_ac * tariff
ac_rel_pct  = (kwh_save_ac / max(kwh_ac, 1e-9)) * 100.0 if kwh_ac > 0 else 0.0

# LED savings (from current LED% -> target LED%)
kwh_light = float(df_break.loc[df_break["category"]=="lighting","kwh"].values[0]) if "lighting" in df_break["category"].values else 0.0
kwh_save_led = led_savings(kwh_light, led_pct, target_led_pct=target_led_pct, efficacy=led_eff)
aed_save_led = kwh_save_led * tariff
led_rel_pct  = (kwh_save_led / max(kwh_light, 1e-9)) * 100.0 if kwh_light > 0 else 0.0

# ------------- Derive "After" breakdown from actions (dynamic) -------------
factor_ac = max(0.0, (kwh_ac - kwh_save_ac) / max(kwh_ac, 1e-9)) if kwh_ac > 0 else 1.0
factor_led = max(0.0, (kwh_light - kwh_save_led) / max(kwh_light, 1e-9)) if kwh_light > 0 else 1.0

df_after = df_break.copy()
df_after["kwh_after"] = df_after["kwh"]
df_after.loc[df_after["category"] == "ac", "kwh_after"] *= factor_ac
df_after.loc[df_after["category"] == "lighting", "kwh_after"] *= factor_led

before_total_kwh = float(df_break["kwh"].sum())
after_total_kwh = float(df_after["kwh_after"].sum())

df_after["share_after"] = df_after["kwh_after"] / max(after_total_kwh, 1e-9)
df_after["aed_after"] = df_after["kwh_after"] * float(tariff)

# Scale credible intervals proportionally (simple/conservative)
scale_map = {"ac": factor_ac, "lighting": factor_led}
df_after["kwh_lo_after"] = df_after.apply(lambda r: r["kwh_lo"] * scale_map.get(r["category"], 1.0), axis=1)
df_after["kwh_hi_after"] = df_after.apply(lambda r: r["kwh_hi"] * scale_map.get(r["category"], 1.0), axis=1)

# Prebuild tables for Before/After (used also by export)
table_before = df_break[["category","kwh","kwh_lo","kwh_hi","aed"]].round(2).rename(columns={
    "category":"Category","kwh":"kWh (mean)","kwh_lo":"kWh (low)","kwh_hi":"kWh (high)","aed":"AED"
})
table_after = df_after[["category","kwh_after","kwh_lo_after","kwh_hi_after","aed_after"]].round(2).rename(columns={
    "category":"Category","kwh_after":"kWh (mean)","kwh_lo_after":"kWh (low)","kwh_hi_after":"kWh (high)","aed_after":"AED"
})

# Totals in AED
before_total_aed = float(df_break["aed"].sum())
after_total_aed  = float(df_after["aed_after"].sum())

# --------------------------- Top KPIs ----------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Estimated bill (input)", f"{bill:.0f} AED")
top_cat = df_break.iloc[0]["category"].title() if not df_break.empty else "-"
k2.metric("Top driver", top_cat)
placeholder_saving = st.empty()  # will fill below

# --------------------------- Layout -----------------------------
col1, col2 = st.columns([1.1, 1])

with col1:
    st.subheader("Where your bill goes (with confidence)")

    # SINGLE PIE: toggle morphs between Before and After
    view_after = st.toggle("Apply selected changes (show After view)", value=False)

    if view_after:
        # Build delta labels vs BEFORE
# Build delta labels vs BEFORE  — FIXED to avoid duplicate 'share_after'
        right_before = df_break[["category", "share"]].rename(columns={"share": "share_before"})
        left_after = df_after.drop(columns=["share"], errors="ignore")  # drop left 'share' to prevent clash
        df_compare = left_after.merge(right_before, on="category", how="left")

        df_compare = df_compare.sort_values("kwh_after", ascending=False)
        df_compare["delta_share"] = df_compare["share_after"] - df_compare["share_before"]

        labels = [
            f"{r['category'].title()} {r['share_after']*100:.1f}% ({r['delta_share']*100:+.1f}%)"
            for _, r in df_compare.iterrows()
        ]

        pie = px.pie(
            df_compare, values="kwh_after", names="category", hole=0.4,
            color="category", color_discrete_map=CATEGORY_COLORS,
            title=f"After (what-if) — {after_total_kwh:.0f} kWh / {after_total_aed:.0f} AED"
        )
        pie.update_traces(text=labels, textinfo="text")
        st.plotly_chart(pie, use_container_width=True)
        st.caption("ℹ️ **After** applies your AC/LED changes and renormalizes the shares. Labels show Δ vs. Before.")
    else:
        df_view = df_break
        pie = px.pie(
            df_view, values="kwh", names="category", hole=0.4,
            color="category", color_discrete_map=CATEGORY_COLORS,
            title=f"Before — {before_total_kwh:.0f} kWh / {bill:.0f} AED"
        )
        pie.update_traces(textinfo="percent+label")
        st.plotly_chart(pie, use_container_width=True)
        st.caption("ℹ️ **Before** reflects your current inputs (bill, tariff, home type/size, occupants, setpoint, LED%).")

    with st.expander("What am I looking at?"):
        st.write(
            "- The pie shows your monthly energy split.\n"
            "- Toggle **Apply selected changes** to see the *what-if* effect of your AC and LED choices.\n"
            "- Credible intervals use Dirichlet calibration (λ) with optional temperature scaling (τ)."
        )

    # Credible-interval table follows the current view
    if view_after:
        st.caption("95% credible intervals (kWh) — After (scaled from Dirichlet means)")
        st.dataframe(table_after, use_container_width=True)
    else:
        st.caption("95% credible intervals (kWh) — Before (Dirichlet-calibrated)")
        st.dataframe(table_before, use_container_width=True)

with col2:
    st.subheader("What you can do now")

    conf_note = "" if hist_n >= 6 else " _(low personalization — estimates may vary)_"

    # --- Action cards with micro-explanations ---
    st.markdown(
        f"**❄️ Raise AC setpoint by +{delta_c}°C** → "
        f"**~{kwh_save_ac:.0f} kWh / {aed_save_ac:.0f} AED** this month "
        f"(_~{ac_rel_pct:.1f}% of your AC load_){conf_note}"
        + (" _(personalized)_ " if personalized and hist_n >= 6 else "")
    )
    st.caption("Why: each +1°C typically trims compressor runtime ~3–5% (we’re using your learned elasticity if available).")

    st.markdown(
        f"**💡 Move LEDs from {led_pct:.0f}% → {target_led_pct:.0f}%** → "
        f"**~{kwh_save_led:.0f} kWh / {aed_save_led:.0f} AED** this month "
        f"(_~{led_rel_pct:.1f}% of Lighting_){conf_note}"
        + (" _(personalized)_ " if personalized and hist_n >= 6 else "")
    )
    st.caption("Why: higher LED share cuts lumen-for-watt; we estimate remaining halogen/CFL replaced at your stated target.")

    # Update KPI #3
    placeholder_saving.metric("Potential savings (this setup)", f"{(aed_save_ac + aed_save_led):.0f} AED/mo")

    # Bandit recommend (your policy)
    actions = [
        {"name": f"Raise AC by +{delta_c} °C", "type": "setpoint", "delta": delta_c},
        {"name": f"Increase LEDs to {target_led_pct:.0f}%", "type": "led", "delta": max(0, target_led_pct - led_pct)},
    ]
    expected_rewards = [aed_save_ac, aed_save_led]
    choice = recommend(actions, expected_rewards, eps=0.1)
    st.success(f"AI Recommendation: **{actions[choice]['name']}** (est. **{expected_rewards[choice]:.0f} AED** saved)")

    # Commit actions
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Commit AC action", help="Log that you plan to raise your setpoint."):
            append_action("setpoint", delta_c, kwh_save_ac, aed_save_ac)
            st.rerun()
    with c2:
        if st.button("✅ Commit LED action", help="Log that you plan to increase LED share to the target."):
            append_action("led", max(0, target_led_pct - led_pct), kwh_save_led, aed_save_led)
            st.rerun()

# ------------------ Before vs After (What-If) -------------------
st.markdown("---")
st.subheader("Estimated bill after your selected changes")

bar_df = pd.DataFrame({
    "View": ["Before","After"],
    "AED":  [before_total_aed, after_total_aed]
})
bar = px.bar(bar_df, x="View", y="AED", text="AED", title="Before vs After (AED/month)")
bar.update_traces(texttemplate="AED %{y:,.0f}", textposition="outside")
bar.update_layout(yaxis_title="AED / month", xaxis_title="", yaxis_tickformat=",")
st.plotly_chart(bar, use_container_width=True)

k1b, k2b, k3b = st.columns(3)
k1b.metric("Before", f"{before_total_aed:,.0f} AED")
k2b.metric("After (What-If)", f"{after_total_aed:,.0f} AED", delta=f"-{(before_total_aed-after_total_aed):,.0f} AED")
k3b.metric("Est. kWh saved", f"{((aed_save_ac + aed_save_led)/max(tariff,1e-6)):,.0f} kWh")

with st.expander("How we computed this"):
    st.write(
        "- **AC**: ~4% energy reduction per +1°C (personalized if your history is present).\n"
        "- **LEDs**: Savings from current LED share → target LED share using your lighting kWh and efficacy.\n"
        "- Shares are renormalized so the pie always sums to 100%."
    )

# ------------------------- Dashboard ----------------------------
st.markdown("---")
st.subheader("📊 Savings dashboard")

if ACTIONS_PATH.exists():
    df_act = pd.read_csv(ACTIONS_PATH)

    month = pd.Timestamp.now().strftime("%Y-%m")
    this_month = df_act[df_act["date"].str.startswith(month)]
    total_month_aed = this_month["est_aed_saved"].sum() if not this_month.empty else 0.0
    total_month_kwh = this_month["est_kwh_saved"].sum() if not this_month.empty else 0.0
    total_all_aed = df_act["est_aed_saved"].sum()
    total_all_kwh = df_act["est_kwh_saved"].sum()
    co2_avoided = total_all_kwh * CO2_KG_PER_KWH

    c1d, c2d, c3d = st.columns(3)
    c1d.metric("Savings this month (AED)", f"{total_month_aed:.0f}")
    c2d.metric("Cumulative savings (AED)", f"{total_all_aed:.0f}")
    c3d.metric("CO₂ avoided (kg)", f"{co2_avoided:.0f}")

    df_act["date"] = pd.to_datetime(df_act["date"], errors="coerce").dt.floor("D")
    for col in ["est_aed_saved", "est_kwh_saved"]:
        df_act[col] = pd.to_numeric(df_act[col], errors="coerce").fillna(0.0)

    daily = (
        df_act.groupby("date")[["est_aed_saved", "est_kwh_saved"]]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    if daily.empty or (daily[["est_aed_saved","est_kwh_saved"]].sum().sum() == 0):
        st.info("No savings trend to show yet — commit some actions on different days to build a history.")
    else:
        ymax = float(daily[["est_aed_saved","est_kwh_saved"]].max().max())
        fig = px.line(
            daily, x="date", y=["est_aed_saved","est_kwh_saved"],
            labels={"value":"Amount","variable":"Metric","date":"Date"},
            title="Savings over time — AED & kWh",
            markers=True,
        )
        fig.update_yaxes(range=[0, max(1.0, ymax * 1.2)])
        fig.update_layout(legend_title_text="Metric")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Recent action log")
    st.dataframe(df_act.tail(10), use_container_width=True)

    st.markdown("### Export")
    try:
        apply_after_flag = "Apply selected changes (show After view)" in st.session_state and st.session_state["Apply selected changes (show After view)"]
        current_table = table_after if apply_after_flag else table_before
        current_table_export = current_table.rename(columns={"Category":"category","kWh (mean)":"kwh","AED":"aed"})[["category","kwh","aed"]]
        current_pie = pie

        png_bytes, pdf_bytes = build_export_assets(
            current_pie, current_table_export,
            total_month_aed, total_month_kwh,
            total_all_aed, total_all_kwh,
            co2_avoided
        )
        colx, coly = st.columns(2)
        with colx:
            st.download_button(
                label="⬇️ Download Pie (PNG)",
                data=png_bytes,
                file_name="smart_energy_summary_pie.png",
                mime="image/png",
                use_container_width=True
            )
        with coly:
            st.download_button(
                label="⬇️ Download 1-page Summary (PDF)",
                data=pdf_bytes,
                file_name="smart_energy_summary.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    except Exception as e:
        st.info(f"Export not available yet — {e}\nTip: install 'kaleido' for Plotly image export (pip install -U kaleido).")

else:
    st.info("No committed actions yet. Use the buttons above to log your first saving.")
