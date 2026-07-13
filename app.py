
import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Breast Cancer Diagnostic System",page_icon="🩺",layout="wide")

MODEL_PATH="breast_cancer_model.pkl"   
SCALER_PATH="scaler.pkl"               

FEATURES=[
"radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
"compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
"radius_se","texture_se","perimeter_se","area_se","smoothness_se",
"compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
"radius_worst","texture_worst","perimeter_worst","area_worst",
"smoothness_worst","compactness_worst","concavity_worst",
"concave_points_worst","symmetry_worst","fractal_dimension_worst"
]

@st.cache_resource
def load():
    model=joblib.load(MODEL_PATH)
    scaler=None
    try:
        scaler=joblib.load(SCALER_PATH)
    except:
        pass
    return model,scaler

model,scaler=load()

st.markdown("""
<style>
.block-container{max-width:1200px;padding-top:1.2rem}
.card{background:#f8fbfc;padding:18px;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.08)}
</style>
""",unsafe_allow_html=True)

st.title("🩺 Breast Cancer Wisconsin Diagnostic")
st.caption("Logistic Regression Prediction Dashboard")
st.warning("Educational use only. Not for clinical diagnosis.")

DEFAULT_VALUES = {
    "radius_mean": 13.37,
    "texture_mean": 18.84,
    "perimeter_mean": 86.24,
    "area_mean": 551.10,
    "smoothness_mean": 0.09587,
    "compactness_mean": 0.09263,
    "concavity_mean": 0.06154,
    "concave_points_mean": 0.03350,
    "symmetry_mean": 0.17920,
    "fractal_dimension_mean": 0.06154,

    "radius_se": 0.32420,
    "texture_se": 1.10800,
    "perimeter_se": 2.28750,
    "area_se": 24.53000,
    "smoothness_se": 0.00638,

    "compactness_se": 0.02045,
    "concavity_se": 0.02589,
    "concave_points_se": 0.01093,
    "symmetry_se": 0.01873,
    "fractal_dimension_se": 0.00319,

    "radius_worst": 14.97,
    "texture_worst": 25.41,
    "perimeter_worst": 97.66,
    "area_worst": 686.50,
    "smoothness_worst": 0.13130,

    "compactness_worst": 0.21190,
    "concavity_worst": 0.22670,
    "concave_points_worst": 0.09993,
    "symmetry_worst": 0.28220,
    "fractal_dimension_worst": 0.08004
}
st.subheader("Quick Assessment")




radius_mean = st.number_input(
    "Average Radius",
    value=DEFAULT_VALUES["radius_mean"],
    help="Average distance from the center of the cell nucleus."
)

perimeter_mean = st.number_input(
    "Average Perimeter",
    value=DEFAULT_VALUES["perimeter_mean"],
    help="Average perimeter of the cell nucleus."
)

area_mean = st.number_input(
    "Average Area",
    value=DEFAULT_VALUES["area_mean"],
    help="Average area of the cell nucleus."
)
 
concavity_mean = st.number_input(
    "Average Concavity",
    value=DEFAULT_VALUES["concavity_mean"],
    help="Measures how concave the nucleus boundary is."
)

concave_points_mean = st.number_input(
    "Average Concave Points",
    value=DEFAULT_VALUES["concave_points_mean"],
    help="Number of concave portions of the nucleus boundary."
)
    
st.info(
    "Quick Assessment Mode: The five most influential measurements are shown first. "
    "The remaining diagnostic measurements are pre-filled with median values from the "
    "training dataset and can be adjusted if available."
)
with st.expander("Advanced Measurements (Uses Training Dataset Medians by Default)"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        texture_mean = st.number_input(
        "Texture Mean",
        value=DEFAULT_VALUES["texture_mean"]
    )

        smoothness_mean = st.number_input(
        "Smoothness Mean",
        value=DEFAULT_VALUES["smoothness_mean"]
    )

        compactness_mean = st.number_input(
        "Compactness Mean",
        value=DEFAULT_VALUES["compactness_mean"]
    )

        symmetry_mean = st.number_input(
        "Symmetry Mean",
        value=DEFAULT_VALUES["symmetry_mean"]
    )

        fractal_dimension_mean = st.number_input(
        "Fractal Dimension Mean",
        value=DEFAULT_VALUES["fractal_dimension_mean"]
    )
        radius_se = st.number_input(
            "Radius Standard Error",
            value=DEFAULT_VALUES["radius_se"]
        )

        texture_se = st.number_input(
            "Texture Standard Error",
            value=DEFAULT_VALUES["texture_se"]
        )

        perimeter_se = st.number_input(
            "Perimeter Standard Error",
            value=DEFAULT_VALUES["perimeter_se"]
        )

        area_se = st.number_input(
            "Area Standard Error",
            value=DEFAULT_VALUES["area_se"]
        )

        smoothness_se = st.number_input(
            "Smoothness Standard Error",
            value=DEFAULT_VALUES["smoothness_se"]
        )

        compactness_se = st.number_input(
            "Compactness Standard Error",
            value=DEFAULT_VALUES["compactness_se"]
        )

        concavity_se = st.number_input(
            "Concavity Standard Error",
            value=DEFAULT_VALUES["concavity_se"]
        )

        concave_points_se = st.number_input(
            "Concave Points Standard Error",
            value=DEFAULT_VALUES["concave_points_se"]
        )
    with adv_col2:
        symmetry_se = st.number_input(
            "Symmetry Standard Error",
            value=DEFAULT_VALUES["symmetry_se"]
        )

        fractal_dimension_se = st.number_input(
            "Fractal Dimension Standard Error",
            value=DEFAULT_VALUES["fractal_dimension_se"]
        )
    

        radius_worst = st.number_input(
            "Radius Worst",
            value=DEFAULT_VALUES["radius_worst"]
        )

        texture_worst = st.number_input(
            "Texture Worst",
            value=DEFAULT_VALUES["texture_worst"]
        )

        perimeter_worst = st.number_input(
            "Perimeter Worst",
            value=DEFAULT_VALUES["perimeter_worst"]
        )

        area_worst = st.number_input(
            "Area Worst",
            value=DEFAULT_VALUES["area_worst"]
        )

        smoothness_worst = st.number_input(
            "Smoothness Worst",
            value=DEFAULT_VALUES["smoothness_worst"]
        )

        compactness_worst = st.number_input(
            "Compactness Worst",
            value=DEFAULT_VALUES["compactness_worst"]
        )

        concavity_worst = st.number_input(
            "Concavity Worst",
            value=DEFAULT_VALUES["concavity_worst"]
        )

        concave_points_worst = st.number_input(
            "Concave Points Worst",
            value=DEFAULT_VALUES["concave_points_worst"]
        )

        symmetry_worst = st.number_input(
            "Symmetry Worst",
            value=DEFAULT_VALUES["symmetry_worst"]
        )

        fractal_dimension_worst = st.number_input(
            "Fractal Dimension Worst",
            value=DEFAULT_VALUES["fractal_dimension_worst"]
        )
vals = [
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    compactness_mean,
    concavity_mean,
    concave_points_mean,
    symmetry_mean,
    fractal_dimension_mean,

    radius_se,
    texture_se,
    perimeter_se,
    area_se,
    smoothness_se,

    compactness_se,
    concavity_se,
    concave_points_se,
    symmetry_se,
    fractal_dimension_se,

    radius_worst,
    texture_worst,
    perimeter_worst,
    area_worst,
    smoothness_worst,

    compactness_worst,
    concavity_worst,
    concave_points_worst,
    symmetry_worst,
    fractal_dimension_worst
]

if st.button("Predict",use_container_width=True):
    X=np.array(vals).reshape(1,-1)
    if scaler is not None:
        X=scaler.transform(X)
    pred=model.predict(X)[0]
    try:
        prob=model.predict_proba(X)[0][1]
    except:
        prob=None

    c1,c2=st.columns([2,1])
    with c1:
        if pred == 1:
            st.error("⚠️ High Risk of Malignancy")

            st.markdown("""
### AI Assessment

The model predicts that the tumor is **more likely to be Malignant**.

**What this means:**
- The entered measurements show characteristics commonly associated with malignant tumors.
- This is **not a confirmed diagnosis**.
- Further medical evaluation, such as imaging or a biopsy, is recommended.

""")

else:
    st.success("✅ Low Risk of Malignancy")

    st.markdown("""
### AI Assessment

The model predicts that the tumor is **more likely to be Benign**.

**What this means:**
- The entered measurements are more consistent with benign tumors.
- This prediction is based on patterns learned from the Breast Cancer Wisconsin Diagnostic dataset.
- Continue following your healthcare provider's recommendations.

""")
    
