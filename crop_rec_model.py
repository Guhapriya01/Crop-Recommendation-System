import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import numpy as np
import seaborn as sns

# %matplotlib inline

df=pd.read_csv("Crop_recommendation.csv")
df.head(3)

df.info()

"""# Visualization"""

def box_plot(df, ft):
    df.boxplot(column=[ft])
    plt.grid(False)
    plt.show()

def hist_plot(df, ft):
    df.hist(column=[ft],bins=50)
    plt.grid(False)
    plt.show()

cols= df.columns
for x in cols:
    if(x!="label"):
        box_plot(df,x)

outliers=['p','k','temperature','rainfall','ph','humidity']

def outlier(df, ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR=Q3 - Q1
    lower_bound = Q1 -(1.5 * IQR) 
    upper_bound = Q3 +(1.5 * IQR)
    ls = df.index[(df[ft]<lower_bound)|(df[ft]>upper_bound)]
    return ls

index_list = []
for features in outliers:
    index_list.extend(outlier(df, features))

def remove(df, ls):         # Removing outliers from the dataset
    ls =sorted(set(ls))
    df=df.drop(ls)
    return df

df_cleaned = remove(df, index_list)

cols= df.columns
for x in cols:
    if(x!="label"):
        hist_plot(df,x)

corr=df.corr()
sns.heatmap(corr)

df.head(2)

sns.pairplot(df,hue="label")

"""# Transformations and Feature Selection


"""

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cc='label_encoded'
df[cc]=le.fit_transform(df["label"])
df.tail(3)

df["label_encoded"].unique()

len(df["label"].unique())

data=df.copy()
df.drop("label",axis="columns",inplace=True)
df.head(2)

X=df.drop("label_encoded",axis="columns")
y=df["label_encoded"]

for i in range(22):
    print(i,len(y[y==i]))

discrete_features = X.dtypes == int

from sklearn.feature_selection import mutual_info_classif
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)

feature_sel = [feature for feature in X.columns if mi_scores[feature]>1]

feature_sel

"""# Training and Testing"""

#X=X[feature_sel]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

models = [LinearSVC(max_iter=10000),
          SVC(kernel='rbf'), 
          KNeighborsClassifier(), 
          LogisticRegression(),
          RandomForestClassifier(max_depth=30, max_features='sqrt',
                                 min_samples_leaf=5,min_samples_split=10,n_estimators=900),
          DecisionTreeClassifier(criterion='entropy',max_depth=20,max_features='auto',
                                 min_samples_split=4), 
          GaussianNB()]
names = ['Linear SVC','SVC', 'KNearestNeighbors','LogisticRegression','RandomForestClassifier',
         'DecisionTree', 'GaussianNB']
acc = []

for model in range(len(models)):
    clf = models[model]
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc.append(accuracy_score( y_test,pred))
    print(names[model])
    print(classification_report(y_test,pred))

models = {'Algorithm': names, 'Accuracy': acc}
models_df = pd.DataFrame(models)
models_df

clf=RandomForestClassifier()

parameters = {'max_depth' : [int(x) for x in np.linspace(5,30,num=6)]
              , 'criterion' : ['gini', 'entropy']
              , 'max_features' : ['auto', 'sqrt', 'log2']
              , 'min_samples_split' : [1.0,2,4,6]
             }

n_estimators=[int(x) for x in np.linspace(100,1200,num=12)]
max_features=['auto','sqrt']
max_depth=[int(x) for x in np.linspace(5,30,num=6)]
min_samples_leaf=[2,5,10,15,100]
min_samples_split=[1.0,2,5,10]

grid2 = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(grid2)

grid=RandomizedSearchCV(estimator=clf,param_distributions=grid2,scoring='neg_mean_squared_error',cv=5,verbose=2,
                        n_jobs=1,n_iter=10,random_state=0)

grid.fit(X_train,y_train)

grid.best_params_

grid.best_score_

predictions=grid.predict(X_test)

print(classification_report(y_test,predictions))

import pickle
with open('crop_rec_model.pkl', 'wb') as file:
    pickle.dump(grid, file)



