import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path('gb_model.pkl')
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path


MODEL_PATH = Path('knn_model.pkl')
SEX_ENCODER_PATH = Path('sex_encoder.pkl')
SEVERITY_ENCODER_PATH = Path('severity_encoder.pkl')
SCALER_PATH = Path('scaler.pkl')
DATA_CSV = Path('Appendicitis Project Assets') / 'appendicitis_data.csv'


@st.cache_resource
def load_assets():
    """Load model, encoders and scaler. Build appendix_on_us encoder from CSV if available."""
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file {MODEL_PATH} not found. Run the training notebook first.")
        st.stop()

    try:
        sex_encoder = joblib.load(SEX_ENCODER_PATH)
    except Exception:
        sex_encoder = None

    try:
        severity_encoder = joblib.load(SEVERITY_ENCODER_PATH)
    except Exception:
        severity_encoder = None

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        scaler = None

    # Build appendix_on_us encoder by fitting a LabelEncoder on the CSV column if available
    appendix_encoder = None
    if DATA_CSV.exists():
        try:
            df = pd.read_csv(DATA_CSV)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            df = df.loc[:, ~df.columns.str.contains('^unnamed')]
            if 'paedriatic_appendicitis_score' in df.columns:
                df = df.rename(columns={'paedriatic_appendicitis_score': 'pediatric_appendicitis_score'})
            if 'appendix_on_us' in df.columns:
                appendix_encoder = LabelEncoder()
                appendix_encoder.fit(df['appendix_on_us'].astype(str))
        except Exception:
            appendix_encoder = None

    return model, sex_encoder, severity_encoder, scaler, appendix_encoder


def safe_transform(le, val):
    """Try several casings for LabelEncoder.transform to avoid unseen label errors."""
    if le is None:
        raise ValueError('LabelEncoder is None')
    for candidate in (val, str(val).lower(), str(val).title(), str(val).upper()):
        try:
            return int(le.transform([candidate])[0])
        except Exception:
            continue
    # as a last resort, try raw string
    return int(le.transform([str(val)])[0])


def main():
    st.set_page_config(page_title="Appendicitis Severity Predictor", layout="wide")
    st.title("🏥 Appendicitis Severity Prediction")

    model, sex_encoder, severity_encoder, scaler, appendix_encoder = load_assets()

    with st.sidebar:
        st.header("Patient Clinical Data")
        age = st.slider("Age (Years)", 1, 120, 30)
        sex = st.selectbox("Sex", options=["Male", "Female"]) 
        wbc_count = st.number_input("WBC Count (x10⁹/L)", min_value=1.0, max_value=40.0, value=12.0, step=0.1)
        crp = st.number_input("CRP Level (mg/L)", min_value=0.1, max_value=100.0, value=15.0, step=0.1)
        appendix_on_us = st.selectbox("Appendix on Ultrasound", options=["Yes", "No"]) 
        neutrophil_percentage = st.number_input("Neutrophil Percentage (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
        alvarado_score = st.number_input("Alvarado Score", min_value=0.0, max_value=10.0, value=2.0, step=1.0)
        pediatric_appendicitis_score = st.number_input("Pediatric Appendicitis Score", min_value=0.0, max_value=10.0, value=2.0, step=1.0)
        predict_button = st.button("Predict Severity")

    st.subheader("Prediction Results")

    if predict_button:
        # Encode sex
        try:
            sex_encoded = safe_transform(sex_encoder, sex)
        except Exception:
            # fallback: map common values
            sex_encoded = 1 if str(sex).lower() in ['male', 'm'] else 0

        # Encode appendix_on_us using encoder if available otherwise fallback
        try:
            if appendix_encoder is not None:
                appendix_encoded = safe_transform(appendix_encoder, appendix_on_us)
            else:
                appendix_encoded = 1 if str(appendix_on_us).lower() in ['yes', 'y', 'true', '1'] else 0
        except Exception:
            appendix_encoded = 1 if str(appendix_on_us).lower() in ['yes', 'y', 'true', '1'] else 0

        # Build feature array in the exact training order used for KNN
        X = np.array([[
            float(age),
            float(sex_encoded),
            float(wbc_count),
            float(crp),
            float(appendix_encoded),
            float(neutrophil_percentage),
            float(alvarado_score),
            float(pediatric_appendicitis_score)
        ]])

        # Apply scaler if available. If the scaler was fitted with feature names,
        # construct a DataFrame with those names so transform aligns correctly.
        if scaler is not None:
            try:
                if hasattr(scaler, 'feature_names_in_'):
                    cols = list(scaler.feature_names_in_)
                    mapping = {
                        'age': age,
                        'sex_encoded': sex_encoded,
                        'wbc_count': wbc_count,
                        'crp': crp,
                        'appendix_on_us': appendix_encoded,
                        'neutrophil_percentage': neutrophil_percentage,
                        'alvarado_score': alvarado_score,
                        'pediatric_appendicitis_score': pediatric_appendicitis_score
                    }
                    row = [float(mapping.get(c.lower(), 0.0)) for c in cols]
                    # Construct a DataFrame with the same column names the scaler was fitted with
                    df_row = pd.DataFrame([row], columns=cols)
                    Xs = scaler.transform(df_row)
                else:
                    Xs = scaler.transform(X)
                # If the scaler produced a different number of features than the model expects,
                # fall back to the unscaled input to avoid dimension mismatch.
                try:
                    expected = getattr(model, 'n_features_in_', None)
                    if expected is not None and Xs.shape[1] != expected:
                        Xs = X
                except Exception:
                    pass
                X = Xs
            except Exception:
                # If scaling fails, continue with unscaled X
                pass

        # Predict
        prediction_encoded = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None

        predicted_severity = None
        if severity_encoder is not None:
            try:
                predicted_severity = severity_encoder.inverse_transform([prediction_encoded])[0]
            except Exception:
                predicted_severity = str(prediction_encoded)
        else:
            predicted_severity = str(prediction_encoded)

        st.markdown(f"**Predicted Severity:** {predicted_severity}")

        if prediction_proba is not None:
            # Normalize prediction_proba into a 1D numpy array representing class probabilities
            probs = np.asarray(prediction_proba)
            if probs.ndim > 1:
                # If shape is (1, n_classes) take the first row; otherwise try to ravel
                if probs.shape[0] == 1:
                    probs = probs[0]
                else:
                    probs = probs.ravel()

            # Determine class labels to align with probabilities
            if severity_encoder is not None:
                classes = list(severity_encoder.classes_)
            elif hasattr(model, 'classes_'):
                classes = list(model.classes_)
            else:
                # fallback to numeric class indices
                classes = list(range(probs.size))

            # Align lengths: try reshape, pad or trim as last resort to avoid DataFrame length mismatch
            if probs.size != len(classes):
                try:
                    if probs.size == len(classes):
                        probs = probs.reshape(len(classes))
                    elif probs.size % len(classes) == 0:
                        probs = probs.reshape(-1, len(classes))[0]
                    else:
                        # pad with zeros or trim to match class count
                        if probs.size < len(classes):
                            probs = np.pad(probs, (0, len(classes) - probs.size), constant_values=0.0)
                        else:
                            probs = probs[:len(classes)]
                except Exception:
                    probs = np.asarray(probs).ravel()[:len(classes)]

            proba_df = pd.DataFrame({'Class': classes, 'Probability': probs}).sort_values(by='Probability', ascending=False)
            proba_df['Probability'] = proba_df['Probability'].apply(lambda x: f"{float(x) * 100:.1f}%")
            st.dataframe(proba_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
