# -*- coding: utf-8 -*-
import pandas as pd
df=pd.read_csv("C:/Users/acer/credit score rating for pharmaceuticlal/clustered_data.csv")
#####data preprocessing
df.dtypes
#dropping the unnecessary columns
df.drop("retailer_names",axis=1,inplace=True)
df.drop("R",axis=1,inplace=True)
df.drop("F",axis=1,inplace=True)
df.drop("M",axis=1,inplace=True)

df.drop("RFMScore",axis=1,inplace=True)
df.drop("RFMGroup",axis=1,inplace=True)
df.drop("Color",axis=1,inplace=True)
df.drop("Cluster",axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df["RFM_Loyalty_Level"]=lb.fit_transform(df["RFM_Loyalty_Level"])
#0=bronze,1=gold,2=platinum,3=silver
predictors=df.iloc[:,0:3]
target=df.iloc[:,3]
# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=2)
from sklearn.ensemble import AdaBoostClassifier as ac
from sklearn import metrics
classifier=ac(base_estimator=None,learning_rate=1.0,n_estimators=50,random_state=2)

classifier.fit(x_train,y_train)
#predicting a new result
y_pred=classifier.predict(x_test)
## accuracy score
from sklearn import metrics
r_square=metrics.r2_score(y_test, y_pred)
print(r_square)
mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
print(mean_squared_log_error)
#save the model to the disk
import pickle
filename="model.pkl"
pickle.dump(classifier,open(filename,"wb"))
model=pickle.load(open("model.pkl","rb"))
