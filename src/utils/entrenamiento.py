import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
from datasets import Dataset

import sys
sys.path.append('../')

# === Heurística para generar etiquetas funcional / no funcional ===
palabras_funcionales = [
    "draw", "destroy", "exile", "counter", "flying", "haste", "indestructible",
    "double strike", "trample", "lifelink", "scry", "tutor", "reanimate", "token", "search"
]

def limpiar_oracle(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

def heuristica_funcional(texto):
    texto = limpiar_oracle(texto)
    return "funcional" if any(p in texto for p in palabras_funcionales) else "no funcional"

# === Cargar CSV y etiquetar ===
df = pd.read_csv("../data/dataCardsclean.csv")
df = df[df["oracle_text"].notna()]
df["label_text"] = df["oracle_text"].apply(heuristica_funcional)

# === Codificar etiquetas ===
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label_text"])  # funcional=1, no funcional=0

# === Separar datos ===
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["oracle_text"], df["label"], test_size=0.2, random_state=42
)

# === Tokenización ===
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

# === Dataset en formato HuggingFace ===
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': list(train_labels)
})
val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'labels': list(val_labels)
})

# === Modelo ===
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# === Entrenamiento ===
args = TrainingArguments(
    output_dir="./resultados",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

trainer.train()
trainer.save_model("modelo_funcionalidad")
tokenizer.save_pretrained("modelo_funcionalidad")
