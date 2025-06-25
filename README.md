# 🧠 Student Mental Health Predictor

This project is a machine learning-based tool that predicts a student's mental health status based on lifestyle habits such as sleep, study time, exercise frequency, and emotional states. The model is trained using a Random Forest Classifier and the application is built with Gradio for an interactive web interface.

## 🚀 Features

- Predict mental health status based on:
  - Daily sleep hours
  - Daily study hours
  - Weekly social and exercise activity
  - Weekly stress, happiness, and anxiety levels
- Interactive sliders for easy input
- Result displayed as both text and GIF-based emotion
- Built with Gradio + Scikit-learn

## 🧰 Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

📁 Project Structure
```
.
├── Data/
│   ├── new_data.csv
│   └── Images/
│       ├── happy.gif
│       ├── apathy.gif
│       ├── anxious.gif
│       ├── irritated.gif
│       └── sad.gif
├── main.py
├── requirements.txt
└── README.md
```

🏁 How to Run

python main.py

A Gradio interface will launch in your browser for you to interact with the model.
🧠 Model

    Algorithm: Random Forest Classifier

    Input: Numeric lifestyle and emotion scores

    Output: Predicted mental health label + mood GIF

📸 Example Output
Input Values	Output Prediction
Sleep: 7 hrs, Study: 5 hrs, ...	"happy" + 😊 GIF
📜 License

This project is licensed under the MIT License.


---

### ✅ `requirements.txt`

```txt
gradio
pandas
numpy
scikit-learn