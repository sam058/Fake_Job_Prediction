import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.title("📊 Model Analysis & Comparison")

# Load models
lr = pickle.load(open("lr_model.pkl", "rb"))
nb = pickle.load(open("nb_model.pkl", "rb"))
rf = pickle.load(open("rf_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load dataset 
df = pd.read_csv("fake_job_postings.csv")
df.fillna('', inplace=True)

df['text'] = df['title'] + " " + df['description'] + " " + df['requirements'] + " " + df['company_profile']

X = df['text']
y = df['fraudulent']

# Vectorize
X_vec = vectorizer.transform(X)

# -------------------------------
# 📊 Accuracy Comparison
# -------------------------------
st.subheader("Model Accuracy Comparison")

accuracies = [
    (lr.score(X_vec, y)),
    (nb.score(X_vec, y)),
    (rf.score(X_vec, y))
]

models = ["Logistic Regression", "Naive Bayes", "Random Forest"]

fig, ax = plt.subplots()
ax.bar(models, accuracies)
ax.set_title("Accuracy Comparison")
ax.set_ylim(0.9, 1.0)

st.pyplot(fig)

# -------------------------------
# 📊 Confusion Matrix (RF)
# -------------------------------
st.subheader("Confusion Matrix - Random Forest")

y_pred = rf.predict(X_vec)

cm = confusion_matrix(y, y_pred)
fig2, ax2 = plt.subplots()

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", ax=ax2)

st.pyplot(fig2)

# -------------------------------
# 📊 Class Distribution
# -------------------------------
st.subheader("Class Distribution")

fig3, ax3 = plt.subplots()
df['fraudulent'].value_counts().plot(kind='bar', ax=ax3)
ax3.set_title("Real vs Fake Jobs")

st.pyplot(fig3)