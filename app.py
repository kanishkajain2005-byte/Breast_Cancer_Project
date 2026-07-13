
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

vals=[]
cols=st.columns(2)
with st.container():
    st.markdown('<div class="card">',unsafe_allow_html=True)
    for i,f in enumerate(FEATURES):
        with cols[i%2]:
            vals.append(st.number_input(f.replace("_"," ").title(),value=0.0,key=f))
    st.markdown("</div>",unsafe_allow_html=True)

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
        if pred==1:
            st.error("Prediction: Malignant")
        else:
            st.success("Prediction: Benign")
        if prob is not None:
            st.progress(float(prob))
            st.metric("Malignant Probability",f"{prob*100:.2f}%")
    with c2:
        st.subheader("Model")
        st.write("- Logistic Regression")
        st.write("- Dataset: WDBC")

with st.expander("About"):
    st.write("Uses all 30 diagnostic features from the Breast Cancer Wisconsin (Diagnostic) dataset.")
