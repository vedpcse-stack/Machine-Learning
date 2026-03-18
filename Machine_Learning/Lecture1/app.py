import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_HEART.pkl')
cols = joblib.load('cols.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Heart Disease Prediction BY Ved')
st.markdown('provide following details')

age = st.slider( 'age' , 18 , 100 , 40 )
sex = st.selectbox( 'sex' , ['male' , 'female']  )
chest_pain_type = st.selectbox( 'chest Pain Type' , ['typical angina' , 'atypical angina' , 'non-anginal pain' , 'asymptomatic']  )
resting_blood_pressure = st.number_input( 'resting blood pressure' , 90 , 200 , 120 )
cholesterol = st.number_input( 'cholesterol' , 100 , 600 , 200 )
fasting_blood_sugar = st.selectbox( 'fasting blood sugar > 120 mg/dl' , [0 , 1]  )
resting_ecg_results = st.selectbox( 'resting electrocardiographic results' , ['normal' , 'ST-T wave abnormality' , 'left ventricular hypertrophy']  )
max_heart_rate_achieved = st.number_input( 'maximum heart rate achieved' , 60 , 202 , 100 )
exercise_induced_angina = st.selectbox( 'exercise induced angina' , ['Yes' , 'No']  )
oldpeak = st.number_input( 'oldpeak' , 0.0 , 10.0 , 2.0 )
st_slope = st.selectbox( 'ST slope' , ['upsloping' , 'flat' , 'downsloping']  ) 

if st.button('Predict'):

    sex = 1 if sex == 'male' else 0
    chest_pain_type = 0 if chest_pain_type == 'typical angina' else 1 if chest_pain_type == 'atypical angina' else chest_pain_type == 'non-anginal   pain'
    exercise_induced_angina = 1 if exercise_induced_angina == 'Yes' else 0
    resting_ecg_results = 0 if resting_ecg_results == 'normal' else 1 if resting_ecg_results == 'ST-T wave abnormality' else 2
    st_slope = 0 if st_slope == 'upsloping' else 1 if st_slope == 'flat' else 2
    fasting_blood_sugar = 1 if fasting_blood_sugar == 1 else 0
    

    raw_input = {
        'Age':age,
        'Sex_Msex':sex,
        'chest_pain_type':chest_pain_type,
        'resting_blood_pressure':resting_blood_pressure,
        'Cholesterol':cholesterol,
        'FastingBS':fasting_blood_sugar,
        'RestingECG_Normal':resting_ecg_results,
        'MaxHR':max_heart_rate_achieved,
        'ExerciseAngina_Y':exercise_induced_angina,
        'Oldpeak':oldpeak,
        'ST_Slope_Flat':st_slope,
    }

    input_df = pd.DataFrame([raw_input])
    
    for col in cols :
        if col not in input_df.columns :
            input_df[col] = 0
    
    input_df = input_df[cols]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    if prediction == 1 :
        st.error(' ⚠️ Heart Disease Detected')
    else :
        st.success(" ✅ Low risk of Heart Disease ")