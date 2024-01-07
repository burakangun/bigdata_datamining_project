import pandas as pd
import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/metin_siniflandirma.csv", index_col=0)
df["Text_2"] = df["Text"].apply(preprocessing.pre_processing)
df["Text_2"] = df["Text_2"].apply(preprocessing.remove_space)
df["Text_3"] = [' '.join(wrd for wrd in x)for x in df["Text_2"]]

df_index = df[df["Text_2"].str[0].isnull()].index
df = df.drop(df_index)
df = df.reset_index()
del df["index"]
# to see the first 5 row of each column
print(df.head())

# to see the labels and how many elements they contain
df.groupby("Label").size()
####################################################
from sklearn.model_selection import train_test_split

msg_train,msg_test,label_train,label_test =train_test_split(df["Text_3"].tolist(), df["Label"].tolist(), test_size=0.2, random_state=42)
print(len(msg_test))
print(len(msg_train))
print(len(label_train))
print(len(label_test))

####################################################
df_test = pd.DataFrame({"Text":msg_test, "Label":label_test})
df_test.groupby("Label").size()
####################################################

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

svm = Pipeline([('vect', TfidfVectorizer()), ('svm', LinearSVC())])
svm.fit(msg_train,label_train)
y_pred_class=svm.predict(msg_test)

####################################################

from sklearn.metrics import f1_score,accuracy_score

print("svm accuracy score:", accuracy_score(label_test,y_pred_class))
print("svm accuracy score:", f1_score(label_test,y_pred_class,average="weighted"))

len(label_test)

import matplotlib.pyplot as plt
cm = confusion_matrix(label_test,y_pred_class, labels=svm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
disp.plot()

# Let's test a sentence that we created.
msg_test_2=["bugün iyi hissetmiyorum."]
y_pred_class = svm.predict(msg_test_2)
print("y-pred : ", y_pred_class)