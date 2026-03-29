# 💼 Fake Job Posting Detection System

A Machine Learning + NLP based web application that detects whether a job posting is **real or fraudulent**.

---

## 🚀 Project Overview

Online job platforms often contain fraudulent job postings that can mislead or scam users.  
This project aims to build a classification system that identifies such fake job listings using textual data.

The system uses Natural Language Processing (NLP) techniques and Machine Learning models to analyze job descriptions and predict their authenticity.

---

## 🧠 Features

- 🔍 Detect fake vs real job postings
- 🧾 Uses multiple text inputs:
  - Job Title
  - Description
  - Requirements
  - Company Profile
- 🤖 Multiple ML models:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
- 📊 Model comparison & confusion matrices
- 🌐 Interactive web app using Streamlit

---

## 📊 Dataset

- Source: Kaggle Fake Job Postings Dataset  
- Size: ~17,000 job postings  
- Target Variable:
  - `0` → Real Job  
  - `1` → Fake Job  

---

## ⚙️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  
- Pickle  

---

## 🧪 Machine Learning Pipeline

1. **Data Preprocessing**
   - Handling missing values
   - Combining text features

2. **Feature Engineering**
   - Merging multiple text columns into one

3. **Vectorization**
   - TF-IDF (Term Frequency - Inverse Document Frequency)

4. **Model Training**
   - Logistic Regression  
   - Naive Bayes  
   - Random Forest  

5. **Evaluation**
   - Accuracy  
   - Confusion Matrix  

---

## 📈 Results

- Random Forest achieved the highest accuracy among all models  
- The system effectively identifies fraudulent job postings based on textual patterns  

---

## 🖥️ Web Application

Built using Streamlit with two main sections:

### 🔮 Prediction
- Input job details
- Select model
- Get real-time prediction (Fake / Legitimate)

### 📊 Analysis
- Model accuracy comparison
- Confusion matrices for all models

---

## ▶️ How to Run

1. Clone the repository:
   
git clone https://github.com/sam058/Fake_Job_Prediction.git
cd Fake_Job_Prediction

2. Install dependencies:

pip install -r requirements.txt

3. Run the Streamlit app:

streamlit run app.py


---

## 💡 Future Improvements

- Use Deep Learning models (LSTM, BERT)
- Add explainability (SHAP)
- Improve preprocessing (lemmatization, stopword removal)
- Real-time job scraping

---

## 🎯 Use Case

This system can help:
- Job seekers avoid scams  
- Platforms flag suspicious listings  
- Improve trust in job portals  

---


## ⭐ Acknowledgements

- Kaggle for dataset  
- Scikit-learn documentation  
- Streamlit for UI framework  

