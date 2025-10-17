import joblib
import numpy as np

# Load fitted preprocessor and model
preprocessor = joblib.load("models/preprocessor.pkl")  # If saved during training
best_model = joblib.load("models/RandomForest.pkl")

def predict_anemia(Gender, Hemoglobin, MCH, MCHC, MCV):
    try:
        X = np.array([[Gender, Hemoglobin, MCH, MCHC, MCV]], dtype=float)
        X_proc = preprocessor.transform(X)
        pred = best_model.predict(X_proc)[0]
        txt = "Anemic" if int(pred) == 1 else "Normal"
        return txt
    except Exception as e:
        return f"Error: {e}"

import gradio as gr

iface = gr.Interface(
    fn=predict_anemia,
    inputs=[
        gr.Number(label="Gender (numeric code)"),
        gr.Number(label="Hemoglobin (g/dL)"),
        gr.Number(label="MCH (pg)"),
        gr.Number(label="MCHC (g/dL)"),
        gr.Number(label="MCV (fL)")
    ],
    outputs=gr.Text(label="Prediction: Anemia Status"),
    title="Anemia Sense Predictor",
    description="Predicts 'Anemic' or 'Normal' using clinical parameters and pre-trained model."
)

if __name__ == "__main__":
    iface.launch()
