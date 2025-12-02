import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path('gb_model.pkl')
SEX_ENCODER_PATH = Path('sex_encoder.pkl')
SEVERITY_ENCODER_PATH = Path('severity_encoder.pkl')

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        sex_encoder = joblib.load(SEX_ENCODER_PATH)
        severity_encoder = joblib.load(SEVERITY_ENCODER_PATH)
        return model, sex_encoder, severity_encoder
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e.filename}")
        st.info("Run the training notebook to generate `gb_model.pkl`, `sex_encoder.pkl`, `severity_encoder.pkl`.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        st.stop()


def main():
    st.set_page_config(page_title="Appendicitis Severity Predictor", layout="wide")
    st.title("🏥 Appendicitis Severity Prediction")
    st.markdown("Predicts severity (Complicated vs Uncomplicated) using the trained model.")

    model, sex_encoder, severity_encoder = load_assets()

    with st.sidebar:
        st.header("Patient Clinical Data")
        age = st.slider("Age (Years)", min_value=1, max_value=120, value=30, step=1)
        sex = st.selectbox("Sex", options=['Male', 'Female'])
        wbc_count = st.number_input("WBC Count (x10⁹/L)", min_value=1.0, max_value=40.0, value=12.0, step=0.1)
        crp = st.number_input("CRP Level (mg/L)", min_value=0.1, max_value=100.0, value=15.0, step=0.1)
        appendix_on_us = st.selectbox("Appendix on Ultrasound", options=['Yes', 'No'])
        neutrophil_percentage = st.number_input("Neutrophil Percentage (%)", min_value=10.0, max_value=99.0, value=75.0, step=0.1)
        alvarado_score = st.number_input("Alvarado Score", min_value=1, max_value=10, value=2, step=1)
        pediatric_appendicitis_score = st.number_input("Pediatric Appendicitis Score", min_value=1, max_value=10, value=2, step=1)
        predict_button = st.button("Predict Severity")

    st.subheader("Prediction Results")

    if predict_button:
        # Encode sex using saved encoder
        try:
            sex_encoded = int(sex_encoder.transform([sex])[0])
        except Exception as e:
            st.error(f"Error encoding `sex`: {e}")
            return

        # Map appendix_on_us: LabelEncoder used in training would map 'No'->0 'Yes'->1 (alphabetical),
        # so use same mapping here
        appendix_val = 1 if appendix_on_us == 'Yes' else 0

        # Build input DataFrame with same feature names used during training
        input_df = pd.DataFrame({
            'age': [age],
            'sex_encoded': [sex_encoded],
            'wbc_count': [wbc_count],
            'crp': [crp],
            'appendix_on_us': [appendix_val],
            'neutrophil_percentage': [neutrophil_percentage],
            'alvarado_score': [alvarado_score],
            'pediatric_appendicitis_score': [pediatric_appendicitis_score]
        })

        try:
            pred_encoded = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]
            pred_label = severity_encoder.inverse_transform([pred_encoded])[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

        st.markdown(f"**Predicted Severity:** {pred_label}")
        if str(pred_label).lower() == 'complicated':
            st.warning("Prediction suggests a Complicated case. Seek urgent clinical care.")
        else:
            st.success("Prediction suggests an Uncomplicated case.")

        proba_df = pd.DataFrame({
            'Severity': severity_encoder.classes_,
            'Probability': pred_proba
        }).sort_values(by='Probability', ascending=False)
        proba_df['Probability'] = proba_df['Probability'].apply(lambda x: f"{x*100:.1f}%")
        st.table(proba_df)


if __name__ == '__main__':
    main()
