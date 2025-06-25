import gradio as gr
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class StudentHealthClassifier:
    def __init__(self):
        self.data="Data/new_data.csv"
        self.train_model()
        self.Img_path={"happy":"Data/Images/happy.gif",
                   "apathy":"Data/Images/apathy.gif",
                   "anxious":"Data/Images/anxious.gif",
                   "irritable":"Data/Images/irritated.gif",
                   "sad":"Data/Images/sad.gif"
        }

    # def getuserinfo(self):
    #     sleep_hr=int(input("How many hours do you sleep daily?"))
    #     study_hr=int(input("How many hours do you study daily?: "))
    #     social_hr=int(input("How many hours do you spend getting involved socially daily?: "))
    #     exc_days=int(input("How many days do you exercise in a week?: "))
    #     str_levs=int(input("What is your current stress level on a scale of 0-10: "))
    #     hap_levs=int(input("How happy were you this week on a scale of 0-10"))
    #     anx_level=int(input("How anxious did you feel throughout this week on a scale of 0-10"))
    #     return sleep_hr,study_hr,social_hr,exc_days,str_levs,hap_levs,anx_level

    def preprocess_data(self):
        df=pd.read_csv(self.data)
        df=df.drop(["mental_health_status"],axis=1)
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
        # print(accuracy_score(Y_test,y_pred)) -> 0.9071729957805907

    # def pred(self):
    #     pred_data=[self.getuserinfo()]
        

    def predictor(self,sleep_hr,study_hr,social_hr,exc_days,str_levs,hap_levs,anx_level):
     input_data=[[sleep_hr,study_hr,social_hr,exc_days,str_levs,hap_levs,anx_level]]
     pred=self.clf.predict(input_data)
     emoji=self.Img_path[pred[0]]
     res=f"Your predicted mental health comes out to be {pred[0]}"
     return res ,emoji

def display(obj:StudentHealthClassifier):
    interface=gr.Interface(fn=obj.predictor,
                 inputs=[
                     gr.Slider(0, 12, value=7, label="Average sleep per day (hours)"),
                     gr.Slider(0, 12, value=5, label="Average study time per day (hours)"),
                     gr.Slider(0, 7, value=3, label="Number of socially active days this week"),
                     gr.Slider(0, 7, value=3, label="Number of exercise days this week"),
                     gr.Slider(0, 10, value=3, label="Overall stress level this week"),
                     gr.Slider(0, 10, value=7, label="Overall happiness level this week"),
                     gr.Slider(0, 10, value=2, label="Overall anxiety level this week"),
                 ],
                 outputs=[
                     gr.Textbox(label="Predicted Result"),
                     gr.Image(type="filepath",label="Mood")
                          ],
                 title="Student mental health predictor",
                 description="Adjust the sliders to input your lifestyle habits and get a prediction."
    )
    interface.launch()

if __name__ == "__main__":
    shc=StudentHealthClassifier()
    display(shc)