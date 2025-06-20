# 🧠 Student Health Classifier

A simple machine learning project to predict a student's mental health based on lifestyle habits such as sleep, study time, social activity, exercise, and stress levels. This project uses a Random Forest Classifier to make predictions and takes inputs interactively from the user.

---

## 📌 Features

- Predicts mental health level (e.g., "Healthy", "Moderate", "Stressed") based on:
  - Daily sleep hours
  - Daily study hours
  - Daily social hours
  - Weekly exercise days
  - Current stress level (0–10)
- Uses a `RandomForestClassifier` from scikit-learn
- Automatic model training from a dataset (`dummy_dataset.csv`)
- CLI-based menu for user interaction
- Preprocessing includes missing value handling using `SimpleImputer`

---

## 🚀 How It Works

1. Loads and preprocesses data from `dummy_dataset.csv`
2. Trains a Random Forest Classifier
3. Asks the user for 5 lifestyle-related inputs
4. Predicts the student's mental health category
5. Allows repeating predictions via menu

---

## 🧪 Sample Run

=== Student Health Prediction Menu ===

    Predict Mental Health

    Exit
    Enter your choice: 1
    How many hours do you sleep daily? 6
    How many hours do you study daily?: 4
    How many hours do you spend getting involved socially daily?: 2
    How many days do you exercise in a week?: 3
    What is your current stress level on a scale of 0–10: 6
    🧠 Your predicted mental health condition is: Moderate


---

## 📁 Project Structure
'''
.
├── main.py # Main program file
├── dummy_dataset.csv # Dataset for training the model
└── README.md # This file

'''
---

## 🛠 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn

Install dependencies using:

```bash
pip install -r requirements.txt

Or manually:

pip install pandas numpy scikit-learn