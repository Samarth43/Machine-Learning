import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as met

#Loading Dataset
data = pd.read_csv("https://raw.githubusercontent.com/omairaasim/machine_learning/master/project_14_naive_bayes/iphone_purchase_records.csv")
features = data.iloc[: , :3].values
label = data.iloc[: ,3].values

# Converting Gender into Number
le = LabelEncoder()
features[: ,0] = le.fit_transform(features[: ,0])
#print(features)

train_features, test_features, train_label, test_label = train_test_split(features,label,test_size=0.25,random_state=0)

model = GaussianNB()
model.fit(train_features,train_label)

pred_label = model.predict(test_features)

print("Confusion Matrix : ",met.confusion_matrix(test_label,pred_label))
print("Accuracy : ",met.accuracy_score(test_label,pred_label))
print("Precision : ",met.precision_score(test_label,pred_label))
print("Recall : ",met.recall_score(test_label,pred_label))
