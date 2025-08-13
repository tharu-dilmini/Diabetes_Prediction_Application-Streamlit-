# Diabetes_Prediction_Application-Streamlit-
📌Diabetes Prediction Streamlit App

This project demonstrates a complete end-to-end Machine Learning workflow using the Pima Indians Diabetes dataset. It includes data exploration, preprocessing, model training, model evaluation, and deployment via Streamlit Cloud.

📌 Project Overview

The objective is to train a machine learning model that predicts the likelihood of diabetes based on patient health metrics. The trained model is integrated into a Streamlit web application that allows users to explore the dataset, visualize data, and make predictions.

📌Key Features:

Interactive Data Explorer to browse and filter the dataset
Visualizations including histograms, scatter plots, and correlation heatmaps
Prediction form to input patient data and get instant results
Model performance metrics with confusion matrix and classification report
Fully deployable to Streamlit Cloud

your-project/
app.py                   # Streamlit web app
model_training.py        # Model training and saving script
model.pkl                # Trained ML model
scaler.pkl               # Fitted StandardScaler
requirements.txt         # Python dependencies
data/
   └── diabetes.csv         # Dataset
notebooks/
   └── model_training_notebook.ipynb  # (Optional) EDA & training in Jupyter
README.md                # Documentation

📌Dataset

Source: Pima Indians Diabetes Dataset
Rows: 768
Columns: 9 (8 features + 1 target Outcome)
Target: Outcome (0 = No diabetes, 1 = Diabetes)
Common preprocessing: Replace medically impossible zeros in certain columns with NaN and impute medians.

📌Features:

Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age

📌Installation & Running Locally

1.Clone the repository: 

git clone https://github.com/your-username/diabetes-prediction-streamlit.git
cd diabetes-prediction-streamlit

2.Create virtual environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3.Install dependencies

pip install -r requirements.txt

4.Prepare dataset Place diabetes.csv inside the data/ folder.

5.Train the model

python model_training.py

6.Run Streamlit app

Run Streamlit app

🌐Deployment to Streamlit Cloud

Push your repository to GitHub.
Go to Streamlit Cloud.
Connect your GitHub account and select the repository.
Set Main file path to app.py.
Deploy.

📌Ensure:

requirements.txt is complete
model.pkl and scaler.pkl are included in the repo
data/diabetes.csv is in the repo or app downloads it automatically

📈 Model Training Details

Models used:

Logistic Regression
Random Forest Classifier
Support Vector Classifier
Evaluation Metrics:
Accuracy
Confusion Matrix
Classification Report

Best Model: Selected based on highest test set accuracy.

💡Technologies Used

Python 3.9+
Streamlit
scikit-learn
pandas
numpy
seaborn, matplotlib, plotly

📜 License

This project is for educational purposes.

✍️ Details

Name: W.M.T.Dilmini
GitHub: https://github.com/tharu-dilmini/Diabetes_Prediction_Application
