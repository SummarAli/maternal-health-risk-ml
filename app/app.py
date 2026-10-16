"""
Maternal Health Risk Predictor
A Streamlit web application for predicting maternal health risk levels
Based on Random Forest classification with 6 physiological features
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .risk-high {
        color: #D32F2F;
        font-size: 28px;
        font-weight: bold;
    }
    .risk-mid {
        color: #F57C00;
        font-size: 28px;
        font-weight: bold;
    }
    .risk-low {
        color: #388E3C;
        font-size: 28px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the trained Random Forest model"""
    try:
        with open('maternal_rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Make sure 'maternal_rf_model.pkl' is in the same folder as app.py")
        return None

# Label encoding used in training (alphabetical): 0 = high, 1 = low, 2 = mid
RISK_LABELS = {
    0: 'high risk',
    1: 'low risk', 
    2: 'mid risk'
}

# Feature information
FEATURE_INFO = {
    'Age': {'min': 10, 'max': 70, 'default': 25, 'unit': 'years'},
    'SystolicBP': {'min': 70, 'max': 160, 'default': 120, 'unit': 'mmHg'},
    'DiastolicBP': {'min': 49, 'max': 100, 'default': 80, 'unit': 'mmHg'},
    'BS': {'min': 6.0, 'max': 19.0, 'default': 8.0, 'unit': 'mmol/L', 'step': 0.1},
    'BodyTemp': {'min': 98.0, 'max': 103.0, 'default': 98.6, 'unit': '°F', 'step': 0.1},
    'HeartRate': {'min': 40, 'max': 90, 'default': 70, 'unit': 'bpm'}
}

FEATURE_NAMES = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
FEATURE_LABELS = ['Age', 'Systolic BP', 'Diastolic BP', 'Blood Sugar', 'Body Temperature', 'Heart Rate']

# Normal ranges for reference
NORMAL_RANGES = {
    'Age': '18-40 (optimal childbearing age)',
    'SystolicBP': '90-120 mmHg (normal)',
    'DiastolicBP': '60-80 mmHg (normal)',
    'BS': '4.0-6.0 mmol/L (normal fasting)',
    'BodyTemp': '97.0-99.0 °F (normal)',
    'HeartRate': '60-100 bpm (normal resting)'
}

def get_clinical_guidance(risk_level, confidence):
    """Provide clinical guidance based on risk level"""
    if risk_level == 'high risk':
        return {
            'color': 'error',
            'icon': '🚨',
            'title': 'HIGH RISK',
            'message': 'Immediate medical attention recommended',
            'actions': [
                'Contact your healthcare provider immediately',
                'Monitor blood pressure and blood sugar closely',
                'Do not delay seeking medical care',
                'Keep a record of all vital signs',
                'Inform healthcare provider of any symptoms'
            ]
        }
    elif risk_level == 'mid risk':
        return {
            'color': 'warning',
            'icon': '⚠️',
            'title': 'MEDIUM RISK',
            'message': 'Consult healthcare provider soon',
            'actions': [
                'Schedule an appointment with your healthcare provider',
                'Continue regular prenatal check-ups',
                'Monitor blood pressure and blood sugar regularly',
                'Maintain healthy diet and exercise habits',
                'Report any new symptoms to your doctor'
            ]
        }
    else:  # low risk
        return {
            'color': 'success',
            'icon': '✅',
            'title': 'LOW RISK',
            'message': 'Continue routine monitoring',
            'actions': [
                'Attend all scheduled prenatal appointments',
                'Maintain healthy lifestyle',
                'Monitor for any unusual symptoms',
                'Continue current care plan',
                'Stay informed about maternal health'
            ]
        }

def plot_feature_importance(model):
    """Plot the model's overall feature importances"""
    # Get the base estimator from CalibratedClassifierCV
    if hasattr(model, 'calibrated_classifiers_'):
        # It's a CalibratedClassifierCV, get the first calibrated classifier
        base_model = model.calibrated_classifiers_[0].estimator
    else:
        base_model = model
    
    if hasattr(base_model, 'feature_importances_'):
        importances = base_model.feature_importances_
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bars
        bars = ax.barh(FEATURE_LABELS, importances, color='#1f77b4', alpha=0.7)
        
        # Customize plot
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title('Overall Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            ax.text(imp, bar.get_y() + bar.get_height()/2, 
                   f' {imp:.3f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return fig
    return None

def main():
    # Title and description
    st.title("🏥 Maternal Health Risk Predictor")
    st.markdown("---")
    
    # Sidebar for information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This tool uses machine learning to assess maternal health risk 
        based on six physiological indicators.
        """)
        
        st.header("⚠️ Important Disclaimer")
        st.warning("""
        **This tool is for educational and research purposes only.**
        
        It is NOT a substitute for professional medical advice, 
        diagnosis, or treatment.
        
        Always consult a qualified healthcare provider for medical decisions.
        """)
        
        st.header("📊 Model Performance")
        st.info("""
        - **Accuracy:** 87%
        - **High-Risk F1-Score:** 0.95
        - **Model:** Random Forest
        - **Dataset:** 1,014 cases
        """)
        
        st.header("📚 How to Use")
        st.write("""
        1. Input patient vital signs using the sliders
        2. Click 'Predict Risk Level'
        3. Review the prediction and guidance
        4. Consult healthcare provider for decisions
        """)
    
    # Main content
    st.header("Patient Information")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Input fields
    inputs = {}
    
    with col1:
        st.subheader("Basic & Cardiovascular")
        
        inputs['Age'] = st.slider(
            'Age (years)',
            min_value=FEATURE_INFO['Age']['min'],
            max_value=FEATURE_INFO['Age']['max'],
            value=FEATURE_INFO['Age']['default'],
            help=f"Normal range: {NORMAL_RANGES['Age']}"
        )
        
        inputs['SystolicBP'] = st.slider(
            'Systolic Blood Pressure (mmHg)',
            min_value=FEATURE_INFO['SystolicBP']['min'],
            max_value=FEATURE_INFO['SystolicBP']['max'],
            value=FEATURE_INFO['SystolicBP']['default'],
            help=f"Normal range: {NORMAL_RANGES['SystolicBP']}"
        )
        
        inputs['DiastolicBP'] = st.slider(
            'Diastolic Blood Pressure (mmHg)',
            min_value=FEATURE_INFO['DiastolicBP']['min'],
            max_value=FEATURE_INFO['DiastolicBP']['max'],
            value=FEATURE_INFO['DiastolicBP']['default'],
            help=f"Normal range: {NORMAL_RANGES['DiastolicBP']}"
        )
    
    with col2:
        st.subheader("Metabolic & Vital Signs")
        
        inputs['BS'] = st.slider(
            'Blood Sugar (mmol/L)',
            min_value=FEATURE_INFO['BS']['min'],
            max_value=FEATURE_INFO['BS']['max'],
            value=FEATURE_INFO['BS']['default'],
            step=FEATURE_INFO['BS']['step'],
            help=f"Normal range: {NORMAL_RANGES['BS']}"
        )
        
        inputs['BodyTemp'] = st.slider(
            'Body Temperature (°F)',
            min_value=FEATURE_INFO['BodyTemp']['min'],
            max_value=FEATURE_INFO['BodyTemp']['max'],
            value=FEATURE_INFO['BodyTemp']['default'],
            step=FEATURE_INFO['BodyTemp']['step'],
            help=f"Normal range: {NORMAL_RANGES['BodyTemp']}"
        )
        
        inputs['HeartRate'] = st.slider(
            'Heart Rate (bpm)',
            min_value=FEATURE_INFO['HeartRate']['min'],
            max_value=FEATURE_INFO['HeartRate']['max'],
            value=FEATURE_INFO['HeartRate']['default'],
            help=f"Normal range: {NORMAL_RANGES['HeartRate']}"
        )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("Predict Risk Level", type="primary", use_container_width=True):
        # Prepare features
        features = np.array([[
            inputs['Age'],
            inputs['SystolicBP'],
            inputs['DiastolicBP'],
            inputs['BS'],
            inputs['BodyTemp'],
            inputs['HeartRate']
        ]])
        
        # Predict and map the encoded class back to its label
        prediction_encoded = model.predict(features)[0]
        prediction = RISK_LABELS[prediction_encoded]
        
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100
        
        # Get guidance
        guidance = get_clinical_guidance(prediction, confidence)
        
        # Display results
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        # Risk level with styling
        if prediction == 'high risk':
            st.markdown(f'<p class="risk-high">{guidance["icon"]} {guidance["title"]}</p>', unsafe_allow_html=True)
        elif prediction == 'mid risk':
            st.markdown(f'<p class="risk-mid">{guidance["icon"]} {guidance["title"]}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="risk-low">{guidance["icon"]} {guidance["title"]}</p>', unsafe_allow_html=True)
        
        # Confidence
        st.metric("Model Confidence", f"{confidence:.1f}%")
        
        # Probability distribution (same order as the label encoding)
        st.subheader("📈 Probability Distribution")
        prob_data = pd.DataFrame({
            'Risk Level': ['High Risk', 'Low Risk', 'Mid Risk'],
            'Probability': probabilities * 100
        })
        
        fig_prob, ax_prob = plt.subplots(figsize=(10, 4))
        colors = ['#D32F2F', '#388E3C', '#F57C00']
        bars = ax_prob.bar(prob_data['Risk Level'], prob_data['Probability'], color=colors, alpha=0.7)
        ax_prob.set_ylabel('Probability (%)', fontweight='bold')
        ax_prob.set_title('Risk Level Probabilities', fontweight='bold', fontsize=14)
        ax_prob.grid(axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for bar, prob in zip(bars, prob_data['Probability']):
            height = bar.get_height()
            ax_prob.text(bar.get_x() + bar.get_width()/2., height,
                        f'{prob:.1f}%',
                        ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig_prob)
        
        # Clinical guidance
        st.markdown("---")
        st.subheader(f"{guidance['icon']} Clinical Guidance")
        
        if guidance['color'] == 'error':
            st.error(guidance['message'])
        elif guidance['color'] == 'warning':
            st.warning(guidance['message'])
        else:
            st.success(guidance['message'])
        
        st.write("**Recommended Actions:**")
        for action in guidance['actions']:
            st.write(f"• {action}")
        
        # Feature importance
        st.markdown("---")
        st.subheader("Which Vital Signs Matter Most to the Model?")
        
        fig_importance = plot_feature_importance(model)
        if fig_importance:
            st.pyplot(fig_importance)
            st.caption("Overall feature importance across the training data (not specific to this individual prediction)")
        
        # Input summary
        st.markdown("---")
        st.subheader("📋 Input Summary")
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        
        with col_sum1:
            st.metric("Age", f"{inputs['Age']} years")
            st.metric("Systolic BP", f"{inputs['SystolicBP']} mmHg")
        
        with col_sum2:
            st.metric("Diastolic BP", f"{inputs['DiastolicBP']} mmHg")
            st.metric("Blood Sugar", f"{inputs['BS']} mmol/L")
        
        with col_sum3:
            st.metric("Body Temp", f"{inputs['BodyTemp']} °F")
            st.metric("Heart Rate", f"{inputs['HeartRate']} bpm")
    
    # Footer
    st.markdown("---")
    st.caption("""
    **Developed by:** Summar Ali | **Model:** Random Forest (87% accuracy) | 
    **Dataset:** 1,014 maternal health cases | **Version:** 1.0
    
    *This tool is for educational purposes only and should not replace professional medical advice.*
    """)

if __name__ == "__main__":
    main()
