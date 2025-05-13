import shap
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# === Cargar modelo y tokenizer entrenado ===
model_path = "modelo_funcionalidad"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# === Pipeline de clasificación ===
clf_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# === Explicador SHAP ===
explainer = shap.Explainer(clf_pipeline)
df = pd.read_csv("cartas.csv").dropna(subset=["oracle_text"])
ejemplos = df["oracle_text"].sample(3, random_state=42).tolist()

# === Visualizar SHAP ===
shap_values = explainer(ejemplos)
shap.plots.text(shap_values[0])
