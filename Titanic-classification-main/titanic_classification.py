

# Reading 
import pandas as pd 
train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')   # without Survived

train.head()

"""### Data Info
- Survived: 	0 = No, 1 = Yes  
- pclass: 	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd  	
- sibsp:	# of siblings / spouses aboard the Titanic  	
- parch:	# of parents / children aboard the Titanic  	
- ticket:	Ticket number	
- cabin:	Cabin number	
- embarked:	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton  
"""

test.head()

train.shape

test.shape

train.info()

test.info()

train.isnull().sum()

test.isnull().sum()

"""**Age** and **Cabin** have many NULL values

# Visualization
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

def chart(col):
    survived=train[train['Survived']==1][col].value_counts()
    dead=train[train['Survived']==0][col].value_counts()
    df=pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',figsize=(10,5))

chart('Sex')

"""* Women more survivied than Men"""

chart('Pclass')

"""* class 1 more survived than other classes
* class 3 more dead than other classes

# ===========================
# Pre-processing

* Many algorithms in machine learning require a numerical representation of objects, since such representations facilitate processing and statistical analysis.

we try use the name column to get any info by using title of each person
"""

train.head()

train_test_data = [train,test]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train['Title'].value_counts()

test['Title'].value_counts()

"""* So, we must mapping this values 

Mr : 0  
Miss : 1  
Mrs: 2  
Others: 3
"""

mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3,
           "Mlle": 3,"Countess": 3,"Ms": 3, "Lady": 3, "Jonkheer": 3,
           "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(mapping)

chart('Title')

"""* **Mr** is highly dead 
* **Miss** is highly Survived 

"""

# delete unnecessary feature from dataset
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

train.head()

test.head()

"""# ==================================
* Mapping of Sex col 
  * male: 0
  * female: 1
"""

import numpy as np
train['Sex']=np.where(train['Sex']=='male',0,1)
test['Sex']=np.where(test['Sex']=='male',0,1)

train.head()

test.head()

chart('Sex')

"""# ===================

* Missing values in **Age** column

**filling missing values of age by madian of ages for each title**
"""

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"),inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"),inplace=True)

train.Age.isnull().sum()

test.Age.isnull().sum()

"""* No null values in Age """

sns.kdeplot(x='Age',hue='Survived',data=train, shade=True).set(xlim=(0, train['Age'].max()))
plt.show()

"""* the most ages dead and Survived is between 20 and 40 years

# ==================================

### Embarked
 * filling missing values
"""

train.isnull().sum()

chart('Embarked')

"""* Most Passenges are from S , so will fill any null value with "S"
"""

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train.isnull().sum()

test.isnull().sum()

# mapping for Embarked 
Map={"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(Map)

"""### Fare
 * filling missing values
"""

train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"),inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"),inplace=True)

sns.kdeplot(x='Fare',hue='Survived',data=train, shade=True)
plt.show()

"""* The most Passengers are dead , they have low Fare

# =================================
"""

train.Cabin.value_counts()

for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

train.Cabin.value_counts()

test.Cabin.value_counts()

chart('Cabin')

"""* Most Passengers in cabin C"""

mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(mapping)

# Filling
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

train.isnull().sum()

test.isnull().sum()

"""**No Null values in our data**

# ======================

#### AllFamilySize
"""

train["AllFamilySize"] = train["SibSp"] + train["Parch"] + 1
test["AllFamilySize"] = test["SibSp"] + test["Parch"] + 1

sns.kdeplot(x='AllFamilySize',hue='Survived',data=train, shade=True)
plt.show()

# drop some features
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)

train.head()

test.head()

X = train.drop('Survived', axis=1)
y = train['Survived']

X.shape, y.shape

X.head()

y.head()

"""* when we look to ranges of min and max , they are big , so we will apply feature scalling """

# Feature Scalling For Training Data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X = sc_X.transform(X)

X=pd.DataFrame(X)
X

# Splitting data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

X_train.head()

"""# Modeling

# 1- RandomForestClassifier
"""

from sklearn.ensemble import RandomForestClassifier
Cl_RF=RandomForestClassifier()
Cl_RF=Cl_RF.fit(X_train,y_train)

# Prediction 
y_pred_RF=Cl_RF.predict(X_test)
y_pred_RF

# Accuracy 
print('Accuracy of RandomForestClassifier model is ',Cl_RF.score(X_test,y_test)*100,'%')

"""# 2- KN Neighbors"""

# KN Neighbors
from sklearn.neighbors import KNeighborsClassifier  
KNN_model=KNeighborsClassifier(n_neighbors=58)
KNN_model.fit(X_train,y_train)

# Prediction 
y_pred_KNN=KNN_model.predict(X_test)
y_pred_KNN

# Accuracy 
print('Accuracy of KNN_Model model is ',KNN_model.score(X_test,y_test)*100,'%')

"""# 3- Decision Tree"""

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
DT_model=DecisionTreeClassifier(criterion='entropy', random_state=0)
DT_model.fit(X_train,y_train)

# Prediction 
y_pred_DT=DT_model.predict(X_test)
y_pred_DT

# Accuracy 
print('Accuracy of Decision Tree model is ',DT_model.score(X_test,y_test)*100,'%')

"""# 4- Support Vector Machine SVM """

# SVM
from sklearn.svm import SVC  
SVM_model=SVC(kernel='rbf')
SVM_model.fit(X_train,y_train)

# Prediction 
y_pred_SVM=SVM_model.predict(X_test)
y_pred_SVM

# Accuracy 
print('Accuracy of SVM model is ',SVM_model.score(X_test,y_test)*100,'%')

"""# 5-Naive Bayes"""

# Naive_Bayes
from sklearn.naive_bayes import GaussianNB
NB_model=GaussianNB()
NB_model.fit(X_train,y_train)

# Prediction 
y_pred_NB=NB_model.predict(X_test)
y_pred_NB

# Accuracy 
print('Accuracy of Naive Bayes model is ',NB_model.score(X_test,y_test)*100,'%')

print('Accuracy of KNN_Model model is ',KNN_model.score(X_test,y_test)*100,'%')
print('Accuracy of SVM model is ',SVM_model.score(X_test,y_test)*100,'%')
print('Accuracy of Decision Tree model is ',DT_model.score(X_test,y_test)*100,'%')
print('Accuracy of Naive Bayes model is ',NB_model.score(X_test,y_test)*100,'%')
print('Accuracy of Random Forest model is ',Cl_RF.score(X_test,y_test)*100,'%')

"""# The High model accuracy is : 
             Random Forest model model is (85%) , SVM model is  (80.7%) and KNN_Model model is (81.6%)

# ==========================================================

# Testing Data
"""

test.head()

Test=test.drop('PassengerId',axis=1)
Test.head()

# Feature Scalling For Testing Data
from sklearn.preprocessing import StandardScaler
sc_test = StandardScaler()
test_data= sc_X.fit_transform(Test)
test_data= sc_X.transform(Test)

test_data=pd.DataFrame(test_data)

test_data.head()

"""# 1- Test data whit Random Forest """

y_pred_RF =Cl_RF.predict(test_data)
y_pred_RF

Y_pred_RF=pd.DataFrame(y_pred_RF)
test['Survived(RF)']=Y_pred_RF
test.head()

"""# ====================================================

# 2- Test data whit SVM
"""

y_pred_SVM =SVM_model.predict(test_data)
y_pred_SVM

Y_predSVM=pd.DataFrame(y_pred_SVM)
test['Survived(SVM)']=Y_predSVM
test.head(50)

"""# ============================================

# 3- Test data whit KNN
"""

y_pred_SVM =SVM_model.predict(test_data)
y_pred_SVM

Y_predKNN=pd.DataFrame(y_pred_KNN)
test['Survived(KNN)']=Y_predKNN
test.head(50)

# Saving The Result of our model
test.to_csv('Result.csv', index=False)
