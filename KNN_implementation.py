import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from imblearn.over_sampling import RandomOverSampler

cols = ["fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist",
        "class"]
magic = pd.read_csv("D:\C_Downloads\Machine Learning\yoututbecourse\magic+gamma+telescope\magic04.data",names=cols)
print(magic.head())
magic['class'] = (magic["class"] == 'g').astype(int)
print(magic.head(10))
# for label in cols[:-1]:
#     plt.hist(magic[magic['class']==1][label],color='blue',label='gamma',alpha=0.7, density=True)
#     plt.hist(magic[magic['class']==0][label],color='red',label='hadron',alpha=0.7, density=True)
#     plt.title(label)
#     plt.ylabel("Prob")
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()
train,val,test = np.split(magic.sample(frac=1),[int(0.6*len(magic)),int(0.8*len(magic))])

def scale_data(df, oversample=False):
    x = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)
    data = np.hstack((x,np.reshape(y,(-1,1))))
    return data, x, y

train, train_x, train_y = scale_data(train, oversample=True)    
val, val_x, val_y = scale_data(val, oversample=False)    
test, test_x, test_y = scale_data(test, oversample=False)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(train_x, train_y)

pred_y = knn_model.predict(test_x)

print(pred_y)

print(classification_report(test_y,pred_y), "KNN")

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(train_x, train_y)
pred_y = nb.predict(test_x)

print(classification_report(test_y,pred_y), "NB")

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_x,train_y)
pred_y = lr.predict(test_x)

print(classification_report(test_y,pred_y), "LR")

