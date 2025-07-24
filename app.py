import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load("random_forest_model.pkl")

# Load dataset for scaler fit
df = pd.read_csv("data/Life Expectancy Data.csv")
df.drop(["Country", "Year"], axis=1, inplace=True)
df.dropna(inplace=True)
df = pd.get_dummies(df, columns=["Status"], drop_first=True)

# Fit scaler on original features
X = df.drop("Life expectancy ", axis=1)
scaler = StandardScaler()
scaler.fit(X)
feature_names = X.columns.tolist()

# Streamlit UI
st.set_page_config(page_title="Life Expectancy Predictor", layout="wide")

st.title("🌍 Life Expectancy Predictor")
st.markdown("**Based on WHO & UN indicators (SDG 3 - Good Health & Well-being)**")

with st.sidebar:
    st.header("Input Features")
    st.info("👉 Modify these values to simulate different health scenarios.")

    def user_inputs():
        inputs = {
            'Adult Mortality': st.slider("Adult Mortality", 0.0, 1000.0, 200.0),
            'infant deaths': st.slider("Infant Deaths", 0.0, 1000.0, 10.0),
            'Alcohol': st.slider("Alcohol Consumption (L)", 0.0, 20.0, 5.0),
            'percentage expenditure': st.slider("Health Expenditure (%)", 0.0, 5000.0, 100.0),
            'Hepatitis B': st.slider("Hepatitis B Vaccination (%)", 0.0, 100.0, 80.0),
            'Measles': st.slider("Measles Cases", 0.0, 100000.0, 1000.0),
            'BMI': st.slider("BMI", 0.0, 100.0, 25.0),
            'under-five deaths': st.slider("Under-Five Deaths", 0.0, 1000.0, 10.0),
            'Polio': st.slider("Polio Vaccination (%)", 0.0, 100.0, 80.0),
            'Total expenditure': st.slider("Total Expenditure (%)", 0.0, 20.0, 5.0),
            'Diphtheria': st.slider("Diphtheria Vaccination (%)", 0.0, 100.0, 80.0),
            'HIV/AIDS': st.slider("HIV/AIDS (%)", 0.0, 100.0, 0.1),
            'GDP': st.slider("GDP (USD)", 0.0, 100000.0, 10000.0),
            'Population': st.number_input("Population", 0.0, 1e10, 1e6),
            'thinness 1-19 years': st.slider("Thinness (1-19 yrs)", 0.0, 100.0, 5.0),
            'thinness 5-9 years': st.slider("Thinness (5-9 yrs)", 0.0, 100.0, 5.0),
            'Income composition of resources': st.slider("Income Composition", 0.0, 1.0, 0.5),
            'Schooling': st.slider("Average Schooling Years", 0.0, 20.0, 10.0),
            'Status_Developed': st.selectbox("Development Status", options=[0, 1], format_func=lambda x: "Developing" if x == 0 else "Developed")
        }
        return pd.DataFrame([inputs])

    input_df = user_inputs()

# Ensure all columns match
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0.0  # add missing dummy columns

# Reorder columns to match training order
input_df = input_df[feature_names]

# Predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

# Output
st.success("✅ Model is ready for prediction!")
st.subheader("📈 Predicted Life Expectancy:")
st.metric(label="Estimated Life Expectancy", value=f"{prediction[0]:.2f} years")

# Footer
st.markdown("---")
st.caption("Built as part of SDG 3 project using WHO data. Model: Random Forest")
