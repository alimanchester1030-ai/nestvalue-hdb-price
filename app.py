"""
NestValue — HDB Resale Price Estimator
Streamlit deployment for the MLDP (CAI2C08) project.
Proof-of-concept scoped to 4 towns: Ang Mo Kio, Clementi, Woodlands, Serangoon.
"""

import datetime
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="NestValue | HDB Resale Price Estimator",
    page_icon="🏠",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main { background-color: #F7F8FA; }
    .hero {
        background: linear-gradient(135deg, #0F5C4F 0%, #14806A 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
    }
    .hero h1 { margin-bottom: 4px; font-size: 28px; }
    .hero p { margin: 0; opacity: 0.9; font-size: 15px; }
    .price-card {
        background: white;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #E8EAED;
        margin-top: 18px;
    }
    .price-card .label { color: #6B7280; font-size: 14px; letter-spacing: 0.5px; text-transform: uppercase; }
    .price-card .value { color: #0F5C4F; font-size: 42px; font-weight: 700; margin: 6px 0; }
    .price-card .sub { color: #6B7280; font-size: 13px; }
    .stButton>button {
        background-color: #0F5C4F;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #14806A; color: white; }
    .footnote { color: #9CA3AF; font-size: 12px; margin-top: 30px; text-align: center; }
    .scope-note { background: #FFF8E6; border: 1px solid #F0DDA0; border-radius: 10px; padding: 10px 14px; font-size: 13px; color: #7A5C00; margin-bottom: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🏠 NestValue</h1>
        <p>Instant, data-driven HDB resale price estimates — powered by a machine learning model
        trained on real Singapore resale transactions (2017 onwards).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="scope-note">
    📍 <b>Proof-of-concept scope:</b> this version covers 4 towns — Ang Mo Kio, Clementi, Woodlands,
    and Serangoon — chosen to represent different regions and a mix of mature and newer estates.
    A production version would extend the same approach to all towns.
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Load model & lookup table
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("hdb_price_model.pkl")
    town_psm_median = joblib.load("town_psm_median.pkl")
    return model, town_psm_median


try:
    model, town_psm_median = load_artifacts()
    load_error = None
except Exception as e:
    model, town_psm_median = None, None
    load_error = str(e)

if load_error:
    st.error(
        "⚠️ The prediction model could not be loaded. This is usually caused by a "
        "scikit-learn version mismatch between training and deployment. Please make sure "
        "`requirements.txt` pins the exact scikit-learn version used to train the model, "
        "and that `hdb_price_model.pkl` / `town_psm_median.pkl` are in the same folder as this app."
    )
    st.caption(f"Technical details: {load_error}")
    st.stop()

TOWNS = sorted(town_psm_median.index.tolist())
FLAT_TYPES = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
FLAT_MODELS = [
    '2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved', 'Improved-Maisonette',
    'Maisonette', 'Model A', 'Model A-Maisonette', 'Model A2', 'Multi Generation',
    'New Generation', 'Premium Apartment', 'Premium Apartment Loft', 'Premium Maisonette',
    'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2'
]

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
st.subheader("Tell us about the flat")

col1, col2 = st.columns(2)
with col1:
    default_town_idx = TOWNS.index("CLEMENTI") if "CLEMENTI" in TOWNS else 0
    town = st.selectbox("Town", TOWNS, index=default_town_idx)
    flat_type = st.selectbox("Flat Type", FLAT_TYPES, index=FLAT_TYPES.index("4 ROOM"))
    flat_model = st.selectbox("Flat Model", FLAT_MODELS, index=FLAT_MODELS.index("Model A"))

with col2:
    floor_area_sqm = st.slider("Floor Area (sqm)", min_value=31, max_value=250, value=95, step=1)
    storey_mid = st.slider("Storey (approx.)", min_value=1, max_value=50, value=8, step=1)
    lease_commence_date = st.slider(
        "Lease Commencement Year", min_value=1966, max_value=2022, value=1995, step=1
    )

current_year = datetime.date.today().year
remaining_lease_years = max(0.0, 99 - (current_year - lease_commence_date))
flat_age = current_year - lease_commence_date

with st.expander("ℹ️ Derived values used by the model"):
    st.write(f"- Estimated remaining lease: **{remaining_lease_years:.0f} years**")
    st.write(f"- Flat age: **{flat_age} years**")
    st.write(f"- Transaction year assumed: **{current_year}**")

st.markdown("---")
predict_clicked = st.button("🔍 Estimate Resale Price")


# ----------------------------------------------------------------------
# Validation + Prediction helpers
# ----------------------------------------------------------------------
def validate_inputs():
    errors = []
    if floor_area_sqm < 20 or floor_area_sqm > 400:
        errors.append("Floor area looks unrealistic — please check the value.")
    if lease_commence_date > current_year:
        errors.append("Lease commencement year cannot be in the future.")
    if remaining_lease_years <= 0:
        errors.append("This flat's 99-year lease has already expired based on the commencement year given.")
    return errors


def build_input_row(selected_town):
    row = pd.DataFrame([{
        "town": selected_town,
        "flat_type": flat_type,
        "floor_area_sqm": float(floor_area_sqm),
        "flat_model": flat_model,
        "lease_commence_date": lease_commence_date,
        "remaining_lease_years": remaining_lease_years,
        "storey_mid": float(storey_mid),
        "transaction_year": current_year,
        "flat_age": flat_age,
    }])
    row["town_psm_median"] = row["town"].map(town_psm_median)
    if row["town_psm_median"].isna().any():
        row["town_psm_median"] = town_psm_median.median()
    return row


if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if predict_clicked:
    errors = validate_inputs()
    if errors:
        for e in errors:
            st.warning(f"⚠️ {e}")
    else:
        try:
            new_flat = build_input_row(town)
            prediction = model.predict(new_flat)[0]

            # Also predict the same flat spec across all 4 towns, for comparison
            town_predictions = {}
            for t in TOWNS:
                row = build_input_row(t)
                town_predictions[t] = model.predict(row)[0]

            st.session_state.last_prediction = {
                "price": prediction,
                "town": town,
                "flat_type": flat_type,
                "floor_area_sqm": floor_area_sqm,
                "town_predictions": town_predictions,
            }
        except Exception as e:
            st.error("⚠️ Something went wrong while generating the estimate. Please try different inputs.")
            st.caption(f"Technical details: {e}")

# ----------------------------------------------------------------------
# Display result
# ----------------------------------------------------------------------
if st.session_state.last_prediction:
    pred = st.session_state.last_prediction
    st.markdown(
        f"""
        <div class="price-card">
            <div class="label">Estimated Resale Price</div>
            <div class="value">S$ {pred['price']:,.0f}</div>
            <div class="sub">{pred['flat_type']} · {pred['town'].title()} · {pred['floor_area_sqm']} sqm</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    price_per_sqm = pred["price"] / pred["floor_area_sqm"]
    st.caption(f"≈ S$ {price_per_sqm:,.0f} per sqm — for comparison against similar listings.")

    # --- Town comparison chart ---
    st.markdown("#### How does this flat's price compare across towns?")
    st.caption("Same flat type, floor area, storey, and lease — only the town changes.")

    town_preds = pred["town_predictions"]
    towns_sorted = sorted(town_preds.items(), key=lambda x: x[1])
    labels = [t.title() for t, _ in towns_sorted]
    values = [v for _, v in towns_sorted]
    colors = ["#0F5C4F" if t.upper() == pred["town"] else "#B5C9C4" for t, _ in towns_sorted]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("Estimated Price (SGD)")
    ax.bar_label(bars, labels=[f"${v:,.0f}" for v in values], padding=3, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

else:
    st.info("👆 Fill in the flat details above and click **Estimate Resale Price** to see a prediction.")

st.markdown(
    """
    <div class="footnote">
    NestValue is a student project prototype for CAI2C08 (Machine Learning for Developers), scoped to
    4 towns as a proof-of-concept. Estimates are based on a Random Forest model trained on HDB resale
    transactions (Jan 2017 onwards, data.gov.sg) and should not be relied on for actual property transactions.
    </div>
    """,
    unsafe_allow_html=True,
)
