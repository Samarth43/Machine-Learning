import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as met


phoneFile = pd.read_csv("https://raw.githubusercontent.com/omairaasim/machine_learning/master/project_14_naive_bayes/iphone_purchase_records.csv")
features = phoneFile.iloc[: , 0:3].values
label = phoneFile.iloc[: , 3].values

le = LabelEncoder()
features[: , 0] = le.fit_transform(features[: , 0])

train_features , test_features , train_label , test_label = train_test_split(features,label,test_size=0.25,random_state=0)

model = KNeighborsClassifier()
model.fit(train_features,train_label)

pred_label = model.predict(test_features)

cm = met.confusion_matrix(test_label,pred_label)
print(cm)
accuracy = met.accuracy_score(test_label,pred_label)
print("Accuracy : ",accuracy)
precision = met.precision_score(test_label,pred_label)
print("Precision : ",precision)
recall = met.recall_score(test_label,pred_label)
print("Recall : ",recall)



