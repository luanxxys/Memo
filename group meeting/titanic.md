# Titanic

### Data set

![titanic_data](./images/titanic_data.PNG)

### Source code

```python
## Data Exploration
# import module and train/test set
import numpy as np
import pandas as pd
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
# combine the train set and predict set to simplify the step
full=pd.concat([train,test],ignore_index=True)

## Feature Engineering
# fill the missing 'Age' according to their 'Title'
full['Title']=full['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
nn = {'Capt':'Rareman', 'Col':'Rareman','Don':'Rareman','Dona':'Rarewoman',
    'Dr':'Rareman','Jonkheer':'Rareman','Lady':'Rarewoman','Major':'Rareman',
    'Master':'Master','Miss':'Miss','Mlle':'Rarewoman','Mme':'Rarewoman',
    'Mr':'Mr','Mrs':'Mrs','Ms':'Rarewoman','Rev':'Mr','Sir':'Rareman',
    'the Countess':'Rarewoman'}

full.Title=full.Title.map(nn)

# assign the female 'Dr' to 'Rarewoman'
full.loc[full.PassengerId==797,'Title']='Rarewoman'
full.Age.fillna(999,inplace=True)
def girl(aa):
    if (aa.Age!=999)&(aa.Title=='Miss')&(aa.Age<=14):
        return 'Girl'
    elif (aa.Age==999)&(aa.Title=='Miss')&(aa.Parch!=0):
        return 'Girl'
    else:
        return aa.Title

full['Title']=full.apply(girl,axis=1)

Tit=['Mr','Miss','Mrs','Master','Girl','Rareman','Rarewoman']
for i in Tit:
    full.loc[(full.Age==999)&(full.Title==i),'Age']=full.loc[full.Title==i,'Age'].median()

# select the features to use - Sex, Age, Pclass
# setting the selected features together makes steps fewer
predictors=['Pclass', 'Sex', 'Age']

# convert categorical variables to numerical variables
full_dummies=pd.get_dummies(full[predictors])

# scaling - Age
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(full_dummies['Age'].values.reshape(-1, 1))
full_dummies['Age'] = age_scale_param.transform(full_dummies['Age'].values.reshape(-1, 1))

# separate the processed set into train_set, train_target, predict_set
X=full_dummies[:891]
y=full.Survived[:891]
test_X=full_dummies[891:]

## Model Building
# Logistic Regression model -0.7889 -0.7946
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C= 0.1)

# Random Forest model -0.7957 -0.8115
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=3, n_estimators=500)

# SVM model - 0.7946 -0.8136
from sklearn.svm import SVC, LinearSVC
svc = SVC(C=0.88, gamma=0.8, probability=True)

# KNN model -0.8081 -worse
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

# decision tree -0.8081 -worse
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=3)

# GradientBoosting model -0.8047 -0.8081
from sklearn.ensemble import GradientBoostingClassifier
gb =GradientBoostingClassifier(random_state=3, learning_rate=0.003, max_depth=20, n_estimators=500)

# XgBoost model -0.8069 -0.8249
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=500,max_depth=6,learning_rate=0.03)

# adaboost -0.7845 -0.7900
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(random_state=3,n_estimators=500,learning_rate=0.01)

## Ensemble Learning
# use VotingClassifier to ensemble the previous training classifier
from sklearn.ensemble import VotingClassifier
eclf_soft=VotingClassifier(estimators=[('LR',lr),('RF',rf),('GDBT',gb),('SVM',svc),('KNN',knn),('D_Tree',dt),('ADAB',ada)],voting='soft')

# add weights
eclfW_soft=VotingClassifier(estimators=[('LR',lr),('RF',rf),('GDBT',gb),('SVM',svc),('KNN',knn),('D_Tree',dt),('ADAB',ada)],voting='soft',weights=[1,1,1,1,1,1,1])

eclfW_soft.fit(X, y)

## Predict
# use the final model to predict the given test set file,
# make the specific format of the data stored in csv
prediction = eclfW_soft.predict(test_X)
result = pd.DataFrame({'PassengerId': test['PassengerId'].as_matrix(), 'Survived': prediction.astype(np.int32)})
result.to_csv("./ensemble_result.csv", index=False)
```

### Procedure

![Titanic](./images/Titanic.PNG)
