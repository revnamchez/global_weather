import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Global Extreme Weather Dashboard", layout="centered")
st.title("Global Extreme Weather Dashboard")
st.markdown("*An MSc Project*") 
st.write("Enter regional metrics below to calculate predicted economic losses.")

# 2. Safely Load Model Artifacts
@st.cache_resource
def load_pipeline():
    artifacts = joblib.load('global_extreme_weather_gb_model.joblib')
    return artifacts['model'], artifacts['disaster_encoder'], artifacts['features_list']

try:
    model, encoder, features_list = load_pipeline()
    
    # 3. Create Interactive Layout Columns
    st.header("⚙️ Predictive Input Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        disaster_choice = st.selectbox("Disaster Profile Type", options=['Flood', 'Drought', 'Tropical Storm'])
        duration = st.slider("Disaster Duration (Days)", min_value=1, max_value=180, value=10)
        deaths = st.number_input("Recorded Casualties (Direct Deaths)", min_value=0, max_value=5000, value=50)
        
    with col2:
        gdp_per_cap = st.number_input("Country GDP Per Capita ($ USD)", min_value=200, max_value=100000, value=3500)
        population = st.number_input("Total Country Population", min_value=100000, max_value=1500000000, value=30000000)
        hdi_proxy = st.slider("HDI Income Proxy Score", min_value=0.1, max_value=1.0, value=0.55, step=0.01)

    st.markdown("---")

    # 4. Trigger Model Estimation on Button Click
    if st.button("🚀 Calculate Predicted Economic Damage"):
        # Compile engineered interaction ratios matching the model's structure
        deaths_per_capita = deaths / (population + 1)
        gdp_density_index = gdp_per_cap * hdi_proxy
        impact_duration_ratio = deaths * duration
        encoded_disaster = encoder.transform([disaster_choice])[0]
        
        # Build the exact evaluation row
        input_data = pd.DataFrame([{
            'duration_days': duration,
            'total_deaths': deaths,
            'gdp_per_capita_usd': gdp_per_cap,
            'country_population': population,
            'hdi_income_proxy': hdi_proxy,
            'disaster_type_encoded': encoded_disaster,
            'deaths_per_capita': deaths_per_capita,
            'gdp_density_index': gdp_density_index,
            'impact_duration_ratio': impact_duration_ratio
        }])
        
        # Reorder columns to guarantee compliance with the training dataset
        input_data = input_data[features_list]
        
        # Run prediction and invert natural log transform
        log_pred = model.predict(input_data)[0]
        true_usd = np.expm1(log_pred)
        
        # Display formatted financial output
        st.success(f"### Predicted Financial Loss: ${true_usd:,.2f} USD")
        
        # Model performance evaluation disclaimer context
        st.info("ℹ️ *Model Evaluation Framework Context: This model accounts for 43.2% of global macroeconomic variance.*")

except Exception as e:
    st.error(f"Error loading model environment: {e}")
    st.warning("Please ensure you ran 'python retrain.py' successfully in this folder first!")
