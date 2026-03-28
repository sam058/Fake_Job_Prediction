import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load models
lr = pickle.load(open("lr_model.pkl", "rb"))
nb = pickle.load(open("nb_model.pkl", "rb"))
rf = pickle.load(open("rf_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Prediction", "Analysis"])

# -------------------------------
# 🟢 PAGE 1: PREDICTION
# -------------------------------
if page == "Prediction":
    st.title("💼 Fake Job Posting Detector")

    model_choice = st.selectbox(
        "Select Model",
        ("Logistic Regression", "Naive Bayes", "Random Forest")
    )

    title = st.text_input("Job Title")
    description = st.text_area("Job Description")
    requirements = st.text_area("Requirements")
    company = st.text_area("Company Profile")

    input_text = title + " " + description + " " + requirements + " " + company

    if st.button("Predict"):
        if input_text.strip() == "":
            st.warning("Please enter job details.")
        else:
            data = vectorizer.transform([input_text])

            if model_choice == "Logistic Regression":
                prediction = lr.predict(data)
            elif model_choice == "Naive Bayes":
                prediction = nb.predict(data)
            else:
                prediction = rf.predict(data)

            if prediction[0] == 1:
                st.error("⚠️ Fake Job Posting")
            else:
                st.success("✅ Legitimate Job Posting")

# -------------------------------
# 🔵 PAGE 2: ANALYSIS
# -------------------------------
elif page == "Analysis":
    st.title("📊 Model Analysis")

    df = pd.read_csv("fake_job_postings.csv")
    df.fillna('', inplace=True)

    df['text'] = df['title'] + " " + df['description'] + " " + df['requirements'] + " " + df['company_profile']

    X = df['text']
    y = df['fraudulent']

    X_vec = vectorizer.transform(X)

    # Accuracy comparison
    st.subheader("Model Accuracy Comparison")

    accuracies = [
        lr.score(X_vec, y),
        nb.score(X_vec, y),
        rf.score(X_vec, y)
    ]

    models = ["Logistic Regression", "Naive Bayes", "Random Forest"]

    fig, ax = plt.subplots()
    ax.bar(models, accuracies)
    ax.set_ylim(0.9, 1.0)
    ax.set_title("Accuracy Comparison")
    st.pyplot(fig)

    # Confusion Matrix (RF)
    st.subheader("Confusion Matrices")


    y_pred = rf.predict(X_vec)
    cm = confusion_matrix(y, y_pred)

    model_dict = {
        "Logistic Regression": lr,
        "Naive Bayes": nb,
        "Random Forest": rf
    }
    for name, model in model_dict.items():
        st.write(f"### {name}")

        y_pred = model.predict(X_vec)
        cm = confusion_matrix(y, y_pred)

        fig_cm, ax_cm = plt.subplots()
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues", ax=ax_cm)

        st.pyplot(fig_cm)

   