import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

# Load data
df = pd.read_csv(r"c:\Users\Naman jain\Downloads\Hear_Disease_Pridiction - DataSet.csv")
df = df.drop(columns=['date', 'id'])

# Encode categorical variables
label_encoders = {}
for column in ['country', 'occupation']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df.drop(['disease'], axis=1)
y = df['disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

def predict_disease():
    try:
        age = int(age_entry.get())
        gender = int(gender_var.get())  # 1 = Female, 2 = Male
        height = int(height_entry.get())
        weight = float(weight_entry.get())
        ap_hi = int(ap_hi_entry.get())
        ap_lo = int(ap_lo_entry.get())
        cholesterol = int(cholesterol_var.get())  # 1 = Normal, 2 = Above Normal, 3 = High
        gluc = int(gluc_var.get())  # 1 = Normal, 2 = Above Normal, 3 = High
        smoke = int(smoke_var.get())  # 0 = No, 1 = Yes
        alco = int(alco_var.get())  # 0 = No, 1 = Yes
        active = int(active_var.get())  # 0 = No, 1 = Yes
        country = country_entry.get()
        occupation = occupation_entry.get()

        country_encoded = label_encoders['country'].transform([country])[0] if country in label_encoders['country'].classes_ else 0
        occupation_encoded = label_encoders['occupation'].transform([occupation])[0] if occupation in label_encoders['occupation'].classes_ else 0

        user_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, country_encoded, occupation_encoded]])
        prediction = decision_tree.predict(user_data)

        result = "You may have heart disease. Consult a doctor." if prediction[0] == 1 else "No heart disease detected."
        messagebox.showinfo("Prediction Result", result)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid input values.")

# GUI setup
root = tk.Tk()
root.title("Heart Disease Prediction")
root.geometry("400x600")

tk.Label(root, text="Enter Your Details", font=("Arial", 14)).pack()

age_entry = tk.Entry(root)
tk.Label(root, text="Age:").pack()
age_entry.pack()

gender_var = tk.StringVar(value="1")
tk.Label(root, text="Gender:").pack()
tk.Radiobutton(root, text="Female", variable=gender_var, value="1").pack()
tk.Radiobutton(root, text="Male", variable=gender_var, value="2").pack()

height_entry = tk.Entry(root)
tk.Label(root, text="Height (cm):").pack()
height_entry.pack()

weight_entry = tk.Entry(root)
tk.Label(root, text="Weight (kg):").pack()
weight_entry.pack()

ap_hi_entry = tk.Entry(root)
tk.Label(root, text="Systolic BP:").pack()
ap_hi_entry.pack()

ap_lo_entry = tk.Entry(root)
tk.Label(root, text="Diastolic BP:").pack()
ap_lo_entry.pack()

cholesterol_var = tk.StringVar(value="1")
tk.Label(root, text="Cholesterol Level:").pack()
tk.Radiobutton(root, text="Normal", variable=cholesterol_var, value="1").pack()
tk.Radiobutton(root, text="Above Normal", variable=cholesterol_var, value="2").pack()
tk.Radiobutton(root, text="High", variable=cholesterol_var, value="3").pack()

gluc_var = tk.StringVar(value="1")
tk.Label(root, text="Glucose Level:").pack()
tk.Radiobutton(root, text="Normal", variable=gluc_var, value="1").pack()
tk.Radiobutton(root, text="Above Normal", variable=gluc_var, value="2").pack()
tk.Radiobutton(root, text="High", variable=gluc_var, value="3").pack()

smoke_var = tk.StringVar(value="0")
tk.Label(root, text="Do you smoke?").pack()
tk.Radiobutton(root, text="No", variable=smoke_var, value="0").pack()
tk.Radiobutton(root, text="Yes", variable=smoke_var, value="1").pack()

alco_var = tk.StringVar(value="0")
tk.Label(root, text="Do you consume alcohol?").pack()
tk.Radiobutton(root, text="No", variable=alco_var, value="0").pack()
tk.Radiobutton(root, text="Yes", variable=alco_var, value="1").pack()

active_var = tk.StringVar(value="0")
tk.Label(root, text="Are you physically active?").pack()
tk.Radiobutton(root, text="No", variable=active_var, value="0").pack()
tk.Radiobutton(root, text="Yes", variable=active_var, value="1").pack()

country_entry = tk.Entry(root)
tk.Label(root, text="Country:").pack()
country_entry.pack()

occupation_entry = tk.Entry(root)
tk.Label(root, text="Occupation:").pack()
occupation_entry.pack()

tk.Button(root, text="Predict Disease", command=predict_disease).pack()

root.mainloop()
