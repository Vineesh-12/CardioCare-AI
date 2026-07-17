import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from lime.lime_tabular import LimeTabularExplainer
from typing import Tuple, Dict, Any, List

# --- MAPPINGS & CONSTANTS ---
FEATURE_NAMES_MAPPING = {
    "age": "Age (Years)",
    "sex": "Biological Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Cholesterol (mg/dL)",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG Results",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "slope": "Slope of Peak Exercise ST Segment",
    "ca": "Number of Major Vessels Colored",
    "thal": "Thalassemia Indicator"
}

SEX_MAPPING = {"Female": 0, "Male": 1}
CP_MAPPING = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
FBS_MAPPING = {"Lower than 120 mg/dl": 0, "Higher than 120 mg/dl": 1}
RESTECG_MAPPING = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
EXANG_MAPPING = {"No": 0, "Yes": 1}
SLOPE_MAPPING = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
THAL_MAPPING = {"Unknown": 0, "Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3}

SCALER_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
MODEL_FEATURES = ['sex', 'cp', 'trestbps', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
MODEL_FEATURE_INDICES = [SCALER_FEATURES.index(f) for f in MODEL_FEATURES]

# --- CONFIG ---
def setup_page():
    st.set_page_config(page_title="CardioCare | Advanced Clinical AI", layout="wide", page_icon="🩺")
    st.markdown("""
    <style>
    :root { --primary: #0ea5e9; --bg-color: #0f172a; --card-bg: #1e293b; --text-color: #f8fafc; }
    body { background-color: var(--bg-color); color: var(--text-color); }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3, h4 { color: #38bdf8 !important; }
    div[data-testid="stSidebar"] { background-color: var(--card-bg); border-right: 1px solid #334155; }
    .stTextArea textarea { background-color: #0f172a; color: #f8fafc; border-color: #334155; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC ---
@st.cache_resource
def load_assets() -> Tuple[Any, Any, List[str], np.ndarray]:
    model = pickle.load(open("heart_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    features = pickle.load(open("features.pkl", "rb"))
    X_train_scaled = pickle.load(open("X_train.pkl", "rb"))
    return model, scaler, features, X_train_scaled

def unscale_x_train(X_train_scaled, scaler):
    N = X_train_scaled.shape[0]
    X_13 = np.zeros((N, 13))
    X_13[:, MODEL_FEATURE_INDICES] = X_train_scaled
    X_13_unscaled = scaler.inverse_transform(X_13)
    return X_13_unscaled[:, MODEL_FEATURE_INDICES]

def get_predict_fn(model, scaler):
    def predict_fn(X_unscaled_10):
        # Handles 1D or 2D arrays
        is_1d = False
        if len(X_unscaled_10.shape) == 1:
            X_unscaled_10 = X_unscaled_10.reshape(1, -1)
            is_1d = True
            
        N = X_unscaled_10.shape[0]
        X_13 = np.zeros((N, 13))
        X_13[:, MODEL_FEATURE_INDICES] = X_unscaled_10
        X_13_scaled = scaler.transform(X_13)
        X_10_scaled = X_13_scaled[:, MODEL_FEATURE_INDICES]
        
        prob = model.predict_proba(X_10_scaled)
        return prob[0] if is_1d else prob
    return predict_fn

@st.cache_resource
def get_explainer(_predict_fn, X_train_unscaled):
    # For SHAP, wrap to output just the positive class probability
    def shap_predict(x): return _predict_fn(x)[:, 1]
    return shap.Explainer(shap_predict, X_train_unscaled)

# --- UI COMPONENTS ---
def get_sidebar_inputs() -> Dict[str, Any]:
    st.sidebar.title("🩺 Patient Vitals")
    st.sidebar.markdown("Enter patient metrics for analysis.")
    
    age = st.sidebar.slider("Age (Years)", 20, 100, 50)
    sex_str = st.sidebar.selectbox("Biological Sex", list(SEX_MAPPING.keys()))
    cp_str = st.sidebar.selectbox("Chest Pain Type", list(CP_MAPPING.keys()))
    trestbps = st.sidebar.slider("Resting BP (mm Hg)", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 600, 200)
    fbs_str = st.sidebar.selectbox("Fasting Blood Sugar", list(FBS_MAPPING.keys()))
    restecg_str = st.sidebar.selectbox("Resting ECG", list(RESTECG_MAPPING.keys()))
    thalach = st.sidebar.slider("Maximum Heart Rate", 60, 220, 150)
    exang_str = st.sidebar.selectbox("Exercise Induced Angina", list(EXANG_MAPPING.keys()))
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope_str = st.sidebar.selectbox("ST Segment Slope", list(SLOPE_MAPPING.keys()))
    ca = st.sidebar.slider("Major Vessels Colored", 0, 4, 0)
    thal_str = st.sidebar.selectbox("Thalassemia", list(THAL_MAPPING.keys()), index=2)

    return {
        "age": age, "sex": SEX_MAPPING[sex_str], "cp": CP_MAPPING[cp_str],
        "trestbps": trestbps, "chol": chol, "fbs": FBS_MAPPING[fbs_str],
        "restecg": RESTECG_MAPPING[restecg_str], "thalach": thalach,
        "exang": EXANG_MAPPING[exang_str], "oldpeak": oldpeak,
        "slope": SLOPE_MAPPING[slope_str], "ca": ca, "thal": THAL_MAPPING[thal_str],
        "raw_strings": {
            "Sex": sex_str, "Chest Pain": cp_str, "Fasting Blood Sugar": fbs_str,
            "Rest ECG": restecg_str, "Exercise Angina": exang_str, 
            "Slope": slope_str, "Thal": thal_str
        }
    }

def draw_gauge(prob: float):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Probability (%)", 'font': {'size': 20, 'color': 'white'}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': 'white'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "#10b981"},
                {'range': [30, 60], 'color': "#f59e0b"},
                {'range': [60, 100], 'color': "#ef4444"}],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

def generate_ehr_note(inputs, prob, top_features, friendly_names):
    risk_cat = "Low" if prob < 0.3 else "Moderate" if prob < 0.6 else "High" if prob < 0.8 else "Very High"
    note = f"Patient is a {inputs['age']} yr old {inputs['raw_strings']['Sex']} presenting with {inputs['raw_strings']['Chest Pain']}. "
    note += f"Vitals include Resting BP {inputs['trestbps']} mmHg, Max HR {inputs['thalach']} bpm, and Cholesterol {inputs['chol']} mg/dL. "
    note += f"\n\nCardioCare AI Assessment indicates a {risk_cat} Risk ({prob*100:.1f}%) of heart disease. "
    note += f"Primary factors increasing risk identified as: "
    note += ", ".join([friendly_names[f] for f in top_features if f in friendly_names]) + ". "
    note += "\n\nPlan: Consider lifestyle interventions and monitor closely based on identified risk drivers."
    return note

def main():
    setup_page()
    model, scaler, features, X_train_scaled = load_assets()
    
    # 1. Unscale X_train for accurate interpretability
    X_train_unscaled = unscale_x_train(X_train_scaled, scaler)
    predict_fn = get_predict_fn(model, scaler)
    explainer = get_explainer(predict_fn, X_train_unscaled)
    
    # 2. Extract inputs & ensure they map EXACTLY to what model expects
    inputs = get_sidebar_inputs()
    
    # Create the 10-feature unscaled array for prediction/explanations
    input_unscaled_10 = np.array([[
        inputs[f] for f in MODEL_FEATURES
    ]])
    
    prob = predict_fn(input_unscaled_10)[0][1]
    
    # --- DASHBOARD UI ---
    st.title("🏥 CardioCare Clinical Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Patient Age", f"{inputs['age']} yrs")
    with col2: st.metric("Resting BP", f"{inputs['trestbps']} mmHg")
    with col3: st.metric("Cholesterol", f"{inputs['chol']} mg/dL")
    with col4: st.metric("Max Heart Rate", f"{inputs['thalach']} bpm")
        
    st.divider()
    
    colA, colB = st.columns([1, 1])
    with colA:
        draw_gauge(prob)
        
    with colB:
        st.markdown("### 🔬 Clinical 'What-If' Simulator")
        st.markdown("Adjust targets below to simulate the effect of medical interventions on the patient's risk.")
        
        sim_trestbps = st.slider("Simulate Lower BP", 80, 200, inputs['trestbps'])
        sim_chol = st.slider("Simulate Lower Cholesterol", 100, 600, inputs['chol']) # Model might not use chol, let's see!
        sim_thalach = st.slider("Simulate Improved Max HR", 60, 220, inputs['thalach'])
        
        # Build simulated 10-feature array
        sim_inputs = inputs.copy()
        sim_inputs['trestbps'] = sim_trestbps
        sim_inputs['chol'] = sim_chol
        sim_inputs['thalach'] = sim_thalach
        
        sim_array = np.array([[sim_inputs[f] for f in MODEL_FEATURES]])
        sim_prob = predict_fn(sim_array)[0][1]
        
        delta = sim_prob - prob
        if delta < 0:
            st.success(f"**Intervention Effect:** Risk reduced by {abs(delta)*100:.1f}% ➔ New Risk: {sim_prob*100:.1f}%")
        elif delta > 0:
            st.warning(f"**Intervention Effect:** Risk increased by {delta*100:.1f}% ➔ New Risk: {sim_prob*100:.1f}%")
        else:
            st.info("No significant change in risk detected.")

    st.divider()
    st.subheader("🧠 Explainable AI & Clinical Notes")
    
    tab1, tab2, tab3 = st.tabs(["📝 Auto-Generated EHR Note", "📊 SHAP (Global)", "📈 LIME (Local)"])
    
    # Prepare friendly names for explanations
    friendly_features = [FEATURE_NAMES_MAPPING.get(f, f) for f in MODEL_FEATURES]
    
    with tab2:
        st.markdown("#### Exact Feature Contributions")
        shap_values = explainer(input_unscaled_10)
        fig, ax = plt.subplots(figsize=(10, 4))
        shap_values.feature_names = friendly_features
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        
        # Get top features for the EHR note
        shap_vals = shap_values.values[0]
        imp_df = pd.DataFrame({"feature": MODEL_FEATURES, "impact": shap_vals})
        top_risk_drivers = imp_df[imp_df['impact'] > 0].sort_values(by="impact", ascending=False)['feature'].tolist()[:3]

    with tab1:
        st.markdown("Ready-to-copy summary for patient record systems:")
        ehr_text = generate_ehr_note(inputs, prob, top_risk_drivers, FEATURE_NAMES_MAPPING)
        st.text_area("Clinical Note", ehr_text, height=200)
        
    with tab3:
        st.markdown("#### Patient-Specific Threshold Analysis")
        lime_exp = LimeTabularExplainer(
            training_data=X_train_unscaled,
            feature_names=friendly_features,
            class_names=['Normal', 'Heart Disease'],
            mode='classification'
        )
        exp = lime_exp.explain_instance(input_unscaled_10[0], predict_fn)
        
        lime_data = []
        for feat_rule, contrib in exp.as_list():
            lime_data.append([
                feat_rule,
                "Increased Risk" if contrib > 0 else "Reduced Risk",
                f"{contrib:.3f}"
            ])
            
        lime_df = pd.DataFrame(lime_data, columns=["Condition / Threshold", "Impact", "Weight"])
        st.dataframe(lime_df, use_container_width=True)

if __name__ == "__main__":
    main()
