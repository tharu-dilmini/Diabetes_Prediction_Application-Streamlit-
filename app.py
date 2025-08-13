import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# --------------------
# App configuration
# --------------------
st.set_page_config(page_title='Diabetes Prediction', layout='wide')

# --------------------
# Load Data Function
# --------------------
@st.cache_data
def load_data():
    file_path = "data/diabetes.csv"

    # If file not found locally, try downloading from GitHub (example raw file URL)
    if not os.path.exists(file_path):
        st.warning("Dataset not found locally. Downloading...")
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"  # Replace with your dataset URL
        df = pd.read_csv(url)
        os.makedirs("data", exist_ok=True)
        df.to_csv(file_path, index=False)
        return df

    return pd.read_csv(file_path)

# --------------------
# Load Model & Scaler
# --------------------
@st.cache_resource
def load_model_and_scaler():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Load data and model
df = load_data()
model, scaler = load_model_and_scaler()

# --------------------
# Sidebar Navigation
# --------------------
st.sidebar.title('Navigation')
page = st.sidebar.radio(
    'Go to',
    ['Home', 'Data Explorer', 'Visualizations', 'Model Prediction', 'Model Performance']
)

# --------------------
# Home Page
# --------------------
if page == 'Home':
    st.title('Diabetes Prediction App')
    st.write('This app uses a trained machine learning model to predict diabetes (Outcome) from medical features.')
    st.markdown('**Dataset:** Pima Indians Diabetes')
    st.write('---')

    col1, col2, col3 = st.columns(3)
    col1.metric('Rows', df.shape[0])
    col2.metric('Columns', df.shape[1])
    col3.metric('Positive Cases', int(df['Outcome'].sum()))

# --------------------
# Data Explorer
# --------------------
elif page == 'Data Explorer':
    st.header('Data Explorer')
    st.write(df.head())

    if st.checkbox('Show data types'):
        st.write(df.dtypes)
    if st.checkbox('Show descriptive stats'):
        st.write(df.describe())

    st.subheader('Interactive filtering')
    col = st.selectbox('Select a column to filter', df.columns.tolist())
    if df[col].dtype != 'object':
        min_val, max_val = float(df[col].min()), float(df[col].max())
        selected = st.slider('Select range', min_val, max_val, (min_val, max_val))
        filtered = df[(df[col] >= selected[0]) & (df[col] <= selected[1])]
        st.write(filtered)
    else:
        val = st.selectbox('Select value', df[col].unique())
        st.write(df[df[col] == val])

# --------------------
# Visualizations
# --------------------
elif page == 'Visualizations':
    st.header('Visualizations')

    st.subheader('Animated Histogram')
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    col_hist = st.selectbox('Choose numeric column for histogram', numeric_cols, key='hist_col')
    fig_hist = px.histogram(df, x=col_hist, nbins=30, title=f'Histogram of {col_hist}', 
                            animation_frame='Outcome', color='Outcome')
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader('Correlation Heatmap')
    if st.button('Show correlation heatmap'):
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.subheader('Animated Scatter Plot')
    x_axis = st.selectbox('X axis for scatter', numeric_cols, index=0, key='scatter_x')
    y_axis = st.selectbox('Y axis for scatter', numeric_cols, index=1, key='scatter_y')
    fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color='Outcome', 
                            title=f'{y_axis} vs {x_axis}', animation_frame='Age',
                            hover_data=['Pregnancies', 'Glucose'])
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader('Animated Box Plot')
    col_box = st.selectbox('Choose numeric column for box plot', numeric_cols, key='box_col')
    # Bin Age into groups for animation
    df['AgeGroup'] = pd.cut(df['Age'], bins=5, labels=['Young', 'Young Adult', 'Adult', 'Middle Aged', 'Senior'])
    fig_box = px.box(df, x='AgeGroup', y=col_box, color='Outcome', 
                     title=f'Box Plot of {col_box} by Age Group', animation_frame='AgeGroup')
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader('Animated Line Plot')
    col_line = st.selectbox('Choose numeric column for line plot', numeric_cols, key='line_col')
    # Aggregate data by Age and Outcome
    line_data = df.groupby(['Age', 'Outcome'])[col_line].mean().reset_index()
    fig_line = px.line(line_data, x='Age', y=col_line, color='Outcome', 
                       title=f'Trend of {col_line} over Age', animation_frame='Outcome')
    st.plotly_chart(fig_line, use_container_width=True)

# --------------------
# Model Prediction
# --------------------
elif page == 'Model Prediction':
    st.header('Make a prediction')
    st.write('Enter feature values to get a prediction from the saved model.')

    with st.form('prediction_form'):
        pregnancies = st.number_input('Pregnancies', 0, 20, int(df['Pregnancies'].median()))
        glucose = st.number_input('Glucose', 0.0, 300.0, float(df['Glucose'].median()))
        bp = st.number_input('BloodPressure', 0.0, 200.0, float(df['BloodPressure'].median()))
        skin = st.number_input('SkinThickness', 0.0, 100.0, float(df['SkinThickness'].median()))
        insulin = st.number_input('Insulin', 0.0, 900.0, float(df['Insulin'].median()))
        bmi = st.number_input('BMI', 0.0, 100.0, float(df['BMI'].median()))
        dpf = st.number_input('DiabetesPedigreeFunction', 0.0, 5.0, float(df['DiabetesPedigreeFunction'].median()))
        age = st.number_input('Age', 0, 120, int(df['Age'].median()))
        submitted = st.form_submit_button('Predict')

    if submitted:
        input_df = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1] if hasattr(model, 'predict_proba') else None

        st.subheader('Prediction result')
        st.write('**Prediction (1=Diabetes, 0=No Diabetes):**', int(pred))
        if proba is not None:
            st.write('**Probability of Diabetes:**', round(proba, 3))

# --------------------
# Model Performance
# --------------------
elif page == 'Model Performance':
    st.header('Model Performance')

    data = df.copy()
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for c in cols_with_zero:
        data[c] = data[c].replace(0, np.nan)
    data[cols_with_zero] = data[cols_with_zero].fillna(data[cols_with_zero].median())

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    st.write('Accuracy:', acc)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.text(classification_report(y_test, preds))
