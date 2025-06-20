import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class StudentHealthClassifier:
    def __init__(self):
        self.data="dummy_dataset.csv"
        self.train_model()

    def getuserinfo(self):
        sleep_hr=int(input("How many hours do you sleep daily?"))
        study_hr=int(input("How many hours do you study daily?: "))
        social_hr=int(input("How many hours do you spend getting involved socially daily?: "))
        exc_days=int(input("How many days do you exercise in a week?: "))
        str_levs=int(input("What is your current stress level on a scale of 0-10: "))
        return sleep_hr,study_hr,social_hr,exc_days,str_levs
    
    def preprocess_data(self):
        df=pd.read_csv("dummy_dataset.csv")
        X=df.iloc[:,1:-1].values
        Y=df.iloc[:,-1].values
        imp=SimpleImputer(missing_values=np.nan,strategy="mean")
        X=imp.fit_transform(X)
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
        return X_train,X_test,Y_train,Y_test
    
    def train_model(self):
        X_train,X_test,Y_train,Y_test=self.preprocess_data()
        self.clf=RandomForestClassifier()
        self.clf.fit(X_train,Y_train)
        #y_pred=clf.predict(X_test) 
        
        # print(confusion_matrix(Y_test,y_pred))
        # print(accuracy_score(Y_test,y_pred)) -> 0.727

    def pred(self):
        pred_data=[self.getuserinfo()]
        pred=self.clf.predict(pred_data)
        print(f"Your predicted mental health comes out to be: {pred[0]}")

if __name__ == "__main__":
    shc = StudentHealthClassifier()

    while True:
        print("\n=== Student Health Prediction Menu ===")
        print("1. Predict Mental Health")
        print("2. Exit")

        choice = input("Enter your choice: ")

        match choice:
            case "1":
                shc.pred()
            case "2":
                print(" Exiting program. Take care!")
                break
            case _:
                print(" Invalid choice. Please select 1 or 2.")


        



