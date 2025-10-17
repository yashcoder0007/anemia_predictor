import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

preprocessor = Pipeline([
    # Define your SimpleImputer, StandardScaler here exactly as you did when training!
    # For production, it's best to save the fitted preprocessor as a joblib file after training and then load it here:
    # joblib.load('preprocessor.pkl')
    # But for this demo, example code is given. Adjust as needed.
])

best_model = joblib.load("models/RandomForest.pkl")  # <-- Update to the best model path

def predict_anemia(Gender, Hemoglobin, MCH, MCHC, MCV):
    try:
        # Order of features must match your model training!
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
