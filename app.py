from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    pregnancies = float(request.form['Pregnancies'])
    glucose = float(request.form['Glucose'])
    blood_pressure = float(request.form['BloodPressure'])
    skin_thickness = float(request.form['SkinThickness'])
    insulin = float(request.form['Insulin'])
    bmi = float(request.form['BMI'])
    dpf = float(request.form['DiabetesPedigreeFunction'])
    age = float(request.form['Age'])
    
    if bmi < 18.5:
        Weight_Status_Healthy_Weight = 0	
        Weight_Status_Obesity = 0	
        Weight_Status_Overweight = 0	
        Weight_Status_Underweight = 1
        
    elif 18.5 <= bmi <= 24.9:
        Weight_Status_Healthy_Weight = 1	
        Weight_Status_Obesity = 0	
        Weight_Status_Overweight = 0	
        Weight_Status_Underweight = 0
       
    elif 25.0 <= bmi <= 29.9:
        Weight_Status_Healthy_Weight = 0	
        Weight_Status_Obesity = 0	
        Weight_Status_Overweight = 1	
        Weight_Status_Underweight = 0
        
    elif bmi >= 30.0:
        Weight_Status_Healthy_Weight = 0	
        Weight_Status_Obesity = 1	
        Weight_Status_Overweight = 0	
        Weight_Status_Underweight = 0
        
   
    if 18 <= age <= 65:
        Age_Category_Adult = 1
        Age_Category_Old = 0
        
    elif age >= 66:
        Age_Category_Adult = 0
        Age_Category_Old = 1        

    # Create an array for the model input
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age,
                          Weight_Status_Healthy_Weight, Weight_Status_Obesity, Weight_Status_Overweight,
                          Weight_Status_Underweight, Age_Category_Adult, Age_Category_Old]])
    
    # Make the prediction
    prediction = model.predict(features)[0]

    # Return the result
    return render_template('index.html', prediction_text=f'Diabetes Prediction: {"Positive" if prediction == 1 else "Negative"}')

if __name__ == '__main__':
    app.run(debug=True)
