from pathlib import Path

# Sample Gradio-based app.py based on your notebook logic
app_py_content = '''\
import pandas as pd
import numpy as np
import gradio as gr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load and prepare data
url = "https://drive.google.com/uc?id=1zIk9JOdJEu9YF7Xuv2C8f2Q8ySfG3nHd"
df = pd.read_csv(url)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

# Clean numeric columns
def clean_numeric(col):
    return (
        col.astype(str)
        .str.replace(r'[^\\d.]', '', regex=True)
        .replace('', np.nan)
        .astype(float)
    )

for c in df.columns:
    df[c] = clean_numeric(df[c])

df = df.dropna()

# Binary classification target
df['plastic_contribution'] = df["M[E] (metric tons year -1)"].apply(lambda x: 0 if x > 6008 else 1)

X = df.drop(columns=["M[E] (metric tons year -1)", "plastic_contribution"])
y = df["plastic_contribution"]

# Scale and train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# Feature names
input_labels = list(X.columns)

def predict(*features):
    input_array = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    return "Low Contributor" if prediction == 1 else "High Contributor"

inputs = [gr.Number(label=label) for label in input_labels]

gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Label(),
    title="River Plastic Pollution Classifier",
    description="Enter river metrics to predict if it's a high or low plastic contributor."
).launch()
'''

# Save the app.py file
app_path = Path("/mnt/data/app.py")
app_path.write_text(app_py_content)
app_path
