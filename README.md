# 🧠 Autism Spectrum Disorder (ASD) Prediction using Deep Neural Network

## 📘 Project Overview
This project aims to assist in the **early detection of Autism Spectrum Disorder (ASD)** using a **Deep Neural Network (DNN)** model built with **Keras and TensorFlow**.  
By analyzing behavioral and demographic data, the model predicts the likelihood of ASD, helping support faster screening and decision-making.  

This tool is designed for **educational and research purposes only** — it is **not** a medical diagnostic system.

---

## 🎯 Objective
To develop a reliable deep learning model that predicts whether an individual shows signs of Autism Spectrum Disorder based on screening questionnaire data.

---

## 🧩 Key Features
- 🔍 **ASD Detection** using a Deep Neural Network  
- 🧮 **Data Preprocessing** with encoding and scaling  
- 📊 **Performance Evaluation** (Accuracy, Precision, Recall, F1-Score)  
- 🌐 **Flask Web Application** for interactive predictions  
- 💾 Model saved as `.h5` and scaler as `.pkl`  
- 📈 **Visualization** of training performance  

---

## 🏗️ System Architecture
Dataset (CSV)
│
▼
Data Preprocessing → Feature Encoding → Scaling
│
▼
Deep Neural Network (Sequential Model)
│
▼
Model Training & Evaluation
│
▼
Save Model (.h5) + Scaler (.pkl)
│
▼
Flask Web App Interface → User Prediction Output

yaml
Copy code

---

## 🧰 Tech Stack
| Component | Technology Used |
|------------|----------------|
| **Programming Language** | Python |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Web Framework** | Flask |
| **Libraries** | NumPy, Pandas, Scikit-learn, Matplotlib |
| **IDE** | VS Code |
| **Dataset Source** | UCI / Kaggle (ASD Screening Dataset) |

---

## 📂 Project Structure
ASD-Prediction-Using-DNN/
│
├── static/ # CSS, JS, images
├── templates/ # HTML templates (index.html, result.html)
├── dataset/
│ └── asd_screening.csv
│
├── model/
│ ├── asd_model.h5
│ └── scaler.pkl
│
├── app.py # Flask main app
├── model_train.py # DNN model training script
├── requirements.txt # Project dependencies
└── README.md # Project documentation

yaml
Copy code

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/ASD-Prediction-Using-DNN.git
cd ASD-Prediction-Using-DNN
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run Model Training (Optional)
bash
Copy code
python model_train.py
4️⃣ Launch Flask Web App
bash
Copy code
python app.py
5️⃣ Access in Browser
cpp
Copy code
http://127.0.0.1:5000/
🧠 Model Details
Model Type: Sequential Deep Neural Network

Input Layer: Number of input features (based on dataset)

Hidden Layers: Dense layers with ReLU activation

Output Layer: Sigmoid activation (binary classification)

Loss Function: Binary Crossentropy

Optimizer: Adam

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

Example architecture:

python
Copy code
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
📊 Evaluation Metrics
Metric	Description
Accuracy	Overall correctness of predictions
Precision	Correct positive predictions / All positive predictions
Recall	Correct positive predictions / All actual positives
F1-Score	Harmonic mean of Precision & Recall

Example Results:

Metric	Score
Accuracy	94%
Precision	92%
Recall	91%
F1-Score	91%

💡 Future Enhancements
🔹 Implement Explainable AI (XAI) with SHAP or LIME for transparency

🔹 Compare DNN with SVM, Random Forest, and XGBoost models

🔹 Integrate real-time API data collection for wider screening

🔹 Create a mobile-friendly UI with Streamlit or Flutter

🔹 Deploy the app on AWS / Render / Heroku

⚠️ Ethical Note
This project is developed for educational and research use.
It should not be used for clinical or diagnostic purposes.
Always consult a certified medical professional for ASD evaluation.



📜 License
This project is open-source and available under the MIT License.

pgsql
Copy code

---

✅ **Tips before uploading:**
1. Add a `requirements.txt`:
   ```bash
   pip freeze > requirements.txt
Include a .gitignore:

markdown
Copy code
venv/
__pycache__/
*.pkl
*.h5
Add screenshots of your UI (optional):

markdown
Copy code
screenshots/
    homepage.png
    result_page.png
And reference them in your README like:

markdown
Copy code
![Homepage](screenshots/homepage.png
