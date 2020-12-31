import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
data=pd.read_csv('C:\\Users\\jubyj\\Desktop\\data science\\salary2.csv')
x=data.drop(['Salary'],axis=1)
y=data['Salary']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
m=regressor.fit(x_train,y_train)
pickle.dump(regressor,open('model.pkl','wb'))