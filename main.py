import numpy as np
import pickle
import streamlit as st
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "trained_model.sav")
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

def main():
    # App title
    st.title("Diabetes Prediction Web App")
    st.write("Enter the details below to check if a person is diabetic or not.")

    # Get user input
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=140, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=100)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

    # Button for prediction
    if st.button("Predict"):
        # Prepare the input data as numpy array
        input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Make prediction
        prediction = loaded_model.predict(input_data_reshaped)

        # Show result
        if prediction[0] == 0:
            st.success("The person is **not diabetic**.")
        else:
            st.error("The person is **diabetic**.")

if __name__ == '__main__':
    main()
