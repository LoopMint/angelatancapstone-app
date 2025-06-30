import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import tempfile

# Load and prepare dataset
file_path = 'QSRanking.csv'

@st.cache_data
def load_data():
    df = pd.read_csv(file_path, encoding='latin1')
    if 'RANK_2024' in df.columns:
        df = df.drop(columns=['RANK_2024'])
    df['RANK_2025'] = pd.to_numeric(df['RANK_2025'], errors='coerce').fillna(1402)
    df['Top100'] = np.where(df['RANK_2025'] <= 100, 1, 0)
    feature_cols = ['Academic_Reputation_Score', 'Employer_Reputation_Score',
                    'Citations_per_Faculty_Score', 'Faculty_Student_Score',
                    'International_Faculty_Score']
    df = df.dropna(subset=feature_cols + ['Institution_Name', 'Top100'])
    return df, feature_cols

df, feature_cols = load_data()
X = df[feature_cols].values
y_class = df['Top100'].values
y_reg = df['RANK_2025'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train classification model
X_train, X_val, y_train_c, y_val_c = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)
model_class = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model_class.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_class.fit(X_train, y_train_c, epochs=20, validation_data=(X_val, y_val_c), verbose=0)

# Train regression model
_, _, y_train_r, y_val_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
model_reg = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model_reg.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model_reg.fit(X_train, y_train_r, epochs=20, validation_data=(X_val, y_val_r), verbose=0)

# Siamese model
def build_siamese_model(input_dim):
    base = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu')
    ])
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))
    encoded_a = base(input_a)
    encoded_b = base(input_b)
    diff = layers.Subtract()([encoded_a, encoded_b])
    out = layers.Dense(1, activation='sigmoid')(diff)
    return Model([input_a, input_b], out)

model_rank = build_siamese_model(X.shape[1])
model_rank.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def create_pairs(X, y_rank):
    pairs_a, pairs_b, labels = [], [], []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            pairs_a.append(X[i])
            pairs_b.append(X[j])
            labels.append(1 if y_rank[i] < y_rank[j] else 0)
    return np.array(pairs_a), np.array(pairs_b), np.array(labels)

X_a, X_b, y_rank_pair = create_pairs(X_scaled, y_reg)
model_rank.fit([X_a, X_b], y_rank_pair, epochs=5, batch_size=256, verbose=0)

# Streamlit UI
st.title("🌟 QS Ranking AI Dashboard")
st.write("Data source: QS World University Rankings 2025")
st.dataframe(df)

st.header("🔍 University Prediction")
univ_input = st.text_input("Enter university name:")
model_type = st.radio("Select model type:", ["Classification", "Regression"])
if st.button("Predict"):
    matched = df[df['Institution_Name'].str.lower().str.contains(univ_input.lower())]
    if matched.empty:
        st.error("University not found.")
    else:
        inst = matched.iloc[0]
        X_input = scaler.transform(inst[feature_cols].values.reshape(1, -1))
        if model_type == "Classification":
            prob = model_class.predict(X_input)[0][0] * 100
            expected = "HIGH" if inst['RANK_2025'] <= 100 else "LOW"
            st.success(f"🔢 RANK_2025: {inst['RANK_2025']} → Expected: {expected}")
            st.info(f"📈 Predicted Top 100 Probability: {prob:.1f}%")
        else:
            pred_rank = model_reg.predict(X_input)[0][0]
            percentage = (pred_rank / 1402) * 100
            st.success(f"🔢 Actual RANK_2025: {inst['RANK_2025']}")
            st.info(f"📈 Predicted RANK_2025: {pred_rank:.1f} ({percentage:.1f}%)")

st.header("🤝 Pairwise Ranking")
col1, col2 = st.columns(2)
univ_a = col1.text_input("University A")
univ_b = col2.text_input("University B")

if st.button("Compare and Generate PDF"):
    u1 = df[df['Institution_Name'].str.lower().str.contains(univ_a.lower())]
    u2 = df[df['Institution_Name'].str.lower().str.contains(univ_b.lower())]
    if u1.empty or u2.empty:
        st.error("One or both universities not found.")
    else:
        X1 = scaler.transform(u1.iloc[0][feature_cols].values.reshape(1, -1))
        X2 = scaler.transform(u2.iloc[0][feature_cols].values.reshape(1, -1))
        prob = model_rank.predict([X1, X2])[0][0] * 100
        better = u1.iloc[0]['Institution_Name'] if prob >= 50 else u2.iloc[0]['Institution_Name']

        vals1 = u1.iloc[0][feature_cols].values
        vals2 = u2.iloc[0][feature_cols].values
        x = np.arange(len(feature_cols))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(x - 0.2, vals1, height=0.4, label=u1.iloc[0]['Institution_Name'])
        ax.barh(x + 0.2, vals2, height=0.4, label=u2.iloc[0]['Institution_Name'])
        ax.set_yticks(x)
        ax.set_yticklabels(feature_cols)
        ax.set_title("Feature Comparison")
        ax.legend()
        st.pyplot(fig)

        # Save PDF
        os.makedirs("reports", exist_ok=True)
        pdf_path = "reports/ranking_report.pdf"
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        text = c.beginText(40, 750)
        text.textLine("Pairwise Ranking Prediction")
        text.textLine(f"Predicted better ranked: {better}")
        text.textLine(f"Probability University A better: {prob:.1f}%")
        c.drawText(text)
        img_buf = BytesIO()
        fig.savefig(img_buf, format='png')
        img_buf.seek(0)
        tmp_img_path = "reports/tmp_chart.png"
        with open(tmp_img_path, 'wb') as f:
            f.write(img_buf.read())
        c.drawImage(tmp_img_path, 40, 300, width=500, preserveAspectRatio=True)
        c.save()
        os.remove(tmp_img_path)
        st.success("✅ PDF saved.")
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name="ranking_report.pdf")

st.header("🏆 Top 10 Universities")
top10 = df.nsmallest(10, 'RANK_2025')
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top10['Institution_Name'], top10['RANK_2025'], color='green')
ax.set_xlabel("RANK_2025")
ax.set_title("Top 10 Universities by RANK_2025")
ax.invert_yaxis()
st.pyplot(fig)
