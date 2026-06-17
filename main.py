from tkinter import *
import tkinter
from tkinter import filedialog
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import os
import matplotlib.pyplot as plt
import joblib
from PIL import Image, ImageTk

mse_list = []
mae_list = []
r2_list = []

model_folder = "model"

def uploadDataset(): 
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, filename + ' Loaded\n')
    dataset = pd.read_csv(filename)
    text.insert(END, str(dataset.head()) + "\n\n")

def Preprocess_Dataset():
    global dataset, X, y

    text.delete('1.0', END)

    dataset = dataset.drop_duplicates()
    dataset = dataset.dropna()

    text.insert(END, str(dataset.isnull().sum()) + "\n\n")

    # ===== Graphs =====
    year_salary = dataset.groupby('work_year')['salary_in_usd'].mean().sort_index()

    plt.figure(figsize=(8,5))
    plt.plot(year_salary.index, year_salary.values, marker='o')
    plt.title("Salary Trend Over Years")
    plt.grid(True)
    plt.show()

    plt.scatter(dataset['salary_in_usd'], dataset['work_year'])
    plt.title("Scatter Plot")
    plt.show()

    plt.hist(dataset['salary_in_usd'])
    plt.title("Salary Distribution")
    plt.show()

    avg_salary = dataset.groupby('experience_level')['salary_in_usd'].mean()
    plt.bar(avg_salary.index, avg_salary.values)
    plt.title("Experience vs Salary")
    plt.show()

    dataset.boxplot(column='salary_in_usd', by='company_size')
    plt.suptitle("")
    plt.show()

    dataset['employment_type'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title("Employment Type Distribution")
    plt.show()

    # ===== Encoding =====
    encoders = {}
    for col in dataset.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])
        encoders[col] = le

    joblib.dump(encoders, "model/label_encoders.pkl")

    y = dataset['salary_in_usd']
    X = dataset.drop(['salary_in_usd'], axis=1)

def Train_Test_Splitting():
    global X, y
    global x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    text.delete('1.0', END)
    text.insert(END, "Total records: " + str(X.shape[0]) + "\n\n")
    text.insert(END, "Training records: " + str(x_train.shape[0]) + "\n\n")
    text.insert(END, "Testing records: " + str(x_test.shape[0]) + "\n\n")

def Calculate_Metrics(algorithm, predict, y_test):

    mse = mean_squared_error(y_test, predict)
    mae = mean_absolute_error(y_test, predict)
    r2 = r2_score(y_test, predict) * 100

    mse_list.append(mse)
    mae_list.append(mae)
    r2_list.append(r2)

    text.insert(END, algorithm + " MSE : " + str(mse) + "\n")
    text.insert(END, algorithm + " MAE : " + str(mae) + "\n")
    text.insert(END, algorithm + " R2 Score : " + str(r2) + "\n\n")

    plt.figure(figsize=(6,6))
    plt.scatter(predict, y_test)
    plt.plot([np.min(y_test), np.max(y_test)],
             [np.min(y_test), np.max(y_test)], '--')
    plt.title(algorithm + " Regression Plot")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ===== EXISTING MODELS (3 MODELS) =====
def existing_classifier():
    global x_train, y_train, x_test, y_test
    text.delete('1.0', END)

    # KNN
    knn_path = os.path.join(model_folder, "KNN.joblib")
    if os.path.exists(knn_path):
        knn = joblib.load(knn_path)
    else:
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(x_train, y_train)
        joblib.dump(knn, knn_path)

    pred_knn = knn.predict(x_test)
    Calculate_Metrics("KNN Regressor", pred_knn, y_test)

    # Linear Regression
    linear_path = os.path.join(model_folder, "LinearRegression.joblib")
    if os.path.exists(linear_path):
        linear = joblib.load(linear_path)
    else:
        linear = LinearRegression()
        linear.fit(x_train, y_train)
        joblib.dump(linear, linear_path)

    pred_lr = linear.predict(x_test)
    Calculate_Metrics("Linear Regression", pred_lr, y_test)

    # Huber Regression
    huber_path = os.path.join(model_folder, "HuberRegression.joblib")
    if os.path.exists(huber_path):
        huber = joblib.load(huber_path)
    else:
        huber = HuberRegressor()
        huber.fit(x_train, y_train)
        joblib.dump(huber, huber_path)

    pred_hr = huber.predict(x_test)
    Calculate_Metrics("Huber Regression", pred_hr, y_test)

# ===== PROPOSED MODEL =====
def proposed_classifier():
    global x_train, y_train, x_test, y_test
    text.delete('1.0', END)

    hgb_path = os.path.join(model_folder, "HistGradientBoosting.joblib")
    if os.path.exists(hgb_path):
        hgb = joblib.load(hgb_path)
    else:
        hgb = HistGradientBoostingRegressor(random_state=42)
        hgb.fit(x_train, y_train)
        joblib.dump(hgb, hgb_path)

    pred_hgb = hgb.predict(x_test)
    Calculate_Metrics("Hist Gradient Boosting", pred_hgb, y_test)

def Prediction():
    text.delete('1.0', END)

    filename = filedialog.askopenfilename(initialdir="Dataset")
    test = pd.read_csv(filename)

    encoders = joblib.load("model/label_encoders.pkl")

    for col in test.select_dtypes(include=['object']).columns:
        if col in encoders:
            test[col] = encoders[col].transform(test[col])

    model = joblib.load("model/HistGradientBoosting.joblib")
    pred = model.predict(test)

    text.insert(END, "Row-wise Predictions:\n\n")

    for i in range(len(test)):
        row_data = test.iloc[i].to_dict()
        prediction = int(round(pred[i]))

        # Convert row into horizontal string
        row_string = " | ".join([f"{key}: {value}" for key, value in row_data.items()])
        
        text.insert(END, f"Row {i+1} -> {row_string} | Predicted Salary: {prediction}\n\n")

def graph():
    df = pd.DataFrame([
        ['KNN', 'MSE', mse_list[0]],
        ['KNN', 'MAE', mae_list[0]],
        ['KNN', 'R2', r2_list[0]],
        ['Linear', 'MSE', mse_list[1]],
        ['Linear', 'MAE', mae_list[1]],
        ['Linear', 'R2', r2_list[1]],
        ['Huber', 'MSE', mse_list[2]],
        ['Huber', 'MAE', mae_list[2]],
        ['Huber', 'R2', r2_list[2]],
        ['HGB', 'MSE', mse_list[3]],
        ['HGB', 'MAE', mae_list[3]],
        ['HGB', 'R2', r2_list[3]],
    ], columns=['Model','Metric','Value'])

    pivot = df.pivot(index='Model', columns='Metric', values='Value')
    pivot.plot(kind='bar')

    plt.title("Model Comparison")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def close():
    main.destroy()

# ===== GUI =====
main = Tk()

bg_image = Image.open("bg.png")
bg_image = bg_image.resize((main.winfo_screenwidth(), main.winfo_screenheight()), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = Label(main, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)


screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

font = ('times', 15, 'bold')
title = Label(main, text='Data Driven Modeling of Salary Dynamics Across Job Roles In 2024 using Predictive Machine Learning ')
title.config(bg='black', fg='thistle1')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0,y=5)

ff = ('Times New Roman', 12, 'bold')

btn_width = 20
btn_height = 1

Button(main, text="Dataset", command=uploadDataset, width=btn_width, height=btn_height, font=ff).place(x=20,y=100)
Button(main, text="Preprocessing", command=Preprocess_Dataset, width=btn_width, height=btn_height, font=ff).place(x=20,y=160)
Button(main, text="Train Test Splitting", command=Train_Test_Splitting, width=btn_width, height=btn_height, font=ff).place(x=20,y=220)
Button(main, text="Existing Classifier", command=existing_classifier, width=btn_width, height=btn_height, font=ff).place(x=20,y=280)
Button(main, text="Proposed Classifier", command=proposed_classifier, width=btn_width, height=btn_height, font=ff).place(x=20,y=340)
Button(main, text="Prediction", command=Prediction, width=btn_width, height=btn_height, font=ff).place(x=20,y=400)
Button(main, text="Comparison Graph", command=graph, width=btn_width, height=btn_height, font=ff).place(x=20,y=460)
Button(main, text="Exit", command=close, width=btn_width, height=btn_height, font=ff).place(x=20,y=520)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=60)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330,y=100)
text.config(font=font1)

main.config(bg='DarkSlateGray1')
bg_label.lower()
main.mainloop()