# 🩺 CardioCare AI Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/) <!-- Note: You can replace this link with your actual Streamlit URL later -->
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CardioCare is an advanced, clinical-grade decision support web application designed to predict the probability of heart disease in patients based on their medical vitals. 

Beyond simple predictions, CardioCare integrates **Explainable AI (XAI)** to provide clinicians with transparent, actionable insights into *why* the AI made a specific assessment, ensuring trust and interpretability in medical environments.

## 🚀 Key Features

- **High-Accuracy Ensemble ML:** Utilizes a Stacking Classifier (combining Random Forest, KNN, Decision Trees, and Logistic Regression) to deliver highly accurate predictions.
- **True-Value Explainable AI (XAI):** Implements custom unscaled LIME and SHAP algorithms to generate exact, real-world clinical thresholds (e.g., `Resting Blood Pressure > 130 mmHg`).
- **🔬 "What-If" Clinical Simulator:** Allows doctors to dynamically adjust patient vitals (e.g., simulating a 20-point drop in blood pressure) to instantly calculate the reduction in disease risk.
- **📝 Auto-Generated EHR Notes:** Synthesizes patient vitals, risk scores, and top risk drivers into a professional clinical note ready to be copied into Electronic Health Record (EHR) systems.
- **Professional UI/UX:** Built with Streamlit and Plotly, featuring a custom dark healthcare aesthetic and interactive gauge charts.

## 🛠️ Technology Stack

- **Frontend/UI:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
- **Explainable AI:** [SHAP](https://shap.readthedocs.io/), [LIME](https://github.com/marcotcr/lime)
- **Data Visualization:** [Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/)

## 💻 Local Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vineesh-12/CardioCare-AI.git
   cd CardioCare-AI
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv .venv
   # On Windows use: .venv\Scripts\activate
   # On macOS/Linux use: source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```
   The dashboard will automatically open in your default web browser at `http://localhost:8501`.

## 📁 Repository Structure

```text
CardioCare-AI/
├── app.py                 # Main Streamlit application script
├── heart_model.pkl        # Serialized Stacking Classifier ML model
├── scaler.pkl             # Serialized MinMaxScaler for data normalization
├── features.pkl           # Saved list of feature column names
├── X_train.pkl            # Training data background for LIME/SHAP explainers
├── requirements.txt       # Python package dependencies
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
