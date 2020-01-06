# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:19:54 2019

@author: mayong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import  LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

# 加载数据
raw_train_data = pd.read_csv('D:/数据挖掘项目练习/Titanic/train.txt',sep='\t')
raw_test_data = pd.read_csv('D:/数据挖掘项目练习/Titanic/test.txt',sep='\t')

train_x = raw_train_data.drop('Survived ',axis=1)
train_y = pd.DataFrame({'PassengerID ':raw_train_data['PassengerId '],'Survived ':raw_train_data['Survived ']})

all_data = pd.concat([train_x,raw_test_data],axis=0)
all_data_del=all_data.drop('Name ',axis=1)

# 绘图查看特征分布情况
fig = plt.figure(figsize=(10,10))
plt.subplot2grid((2,3),(0,0))

#生存人数统计
raw_train_data['Survived '].value_counts().plot(kind='bar')
plt.title('1=Survived')
plt.ylabel('People')

# 客舱等级人数分布
plt.subplot2grid((2,3),(0,1))
raw_train_data['Pclass '].value_counts().plot(kind='bar')
plt.title('PassagerLevel')
plt.ylabel('People')

# 年龄与生存的分布
plt.subplot2grid((2,3),(0,2))
plt.scatter(raw_train_data['Survived '],raw_train_data['Age '])
plt.grid(b=True,which='major',axis='y')
plt.title('Age-Survived')

# 各个年龄存活人数分布情况
plt.subplot2grid((2,3),(1,0), colspan=2)
raw_train_data['Age '][raw_train_data['Pclass ']==1].plot(kind='kde')
raw_train_data['Age '][raw_train_data['Pclass ']==2].plot(kind='kde')
raw_train_data['Age '][raw_train_data['Pclass ']==3].plot(kind='kde')
plt.xlabel(u"age")
plt.ylabel(u"density")
plt.title(u"Age distribution")
plt.legend((u'1st', u'2ed',u'3rd'),loc='best')

# 各个登船口岸人数的分布情况
plt.subplot2grid((2,3),(1,2))
raw_train_data['Embarked'].value_counts().plot(kind='bar')
plt.title(u"Number of people boarding at each boarding port")
plt.ylabel(u"Number of People") 
plt.show()

# 客舱等级与生存是否有关系
fig = plt.Figure(figsize=(5,5))
Survived0 = raw_train_data['Pclass '][raw_train_data['Survived ']==0].value_counts()
Survived1 = raw_train_data['Pclass '][raw_train_data['Survived ']==1].value_counts()
df = pd.DataFrame({'Survived':Survived1,'Unsurvived':Survived0})
df.plot(kind='bar',stacked=True)
plt.title('Rescue by passenger class')
plt.xlabel("Passenger Class") 
plt.ylabel("Number of People") 
plt.show()

# 性别与生存是否有关系
fig = plt.Figure(figsize=(5,5))
Survived0_sex = raw_train_data['Sex '][raw_train_data['Survived ']==0].value_counts()
Survived1_sex = raw_train_data['Sex '][raw_train_data['Survived ']==1].value_counts()
df_sex = pd.DataFrame({'Survived':Survived1_sex,'Unservived':Survived0_sex})
df.plot(kind='bar',stacked=True)
plt.title('Rescue by passenger sex ')
plt.xlabel('Passenger sex')
plt.ylabel('Number of people')
plt.show()

# 不同客舱不同性别的生存情况
fig = plt.Figure(figsize=(20,20))
plt.subplot2grid((2,3),(0,0))
raw_train_data['Survived '][raw_train_data['Pclass ']==1][raw_train_data['Sex ']=='female '].value_counts().plot(kind='bar')
plt.title('Pclass=1 sex = female')
plt.ylabel('Number of people')
plt.xlabel('1=survived 0=unservived')

plt.subplot2grid((2,3),(0,1))
raw_train_data['Survived '][raw_train_data['Pclass ']==1][raw_train_data['Sex ']=='male '].value_counts().plot(kind='bar')
plt.title('Pclass=1 sex = male')
plt.ylabel('Number of people')
plt.xlabel('1=survived 0=unservived')

plt.subplot2grid((2,3),(0,2))
raw_train_data['Survived '][raw_train_data['Pclass ']==2][raw_train_data['Sex ']=='female '].value_counts().plot(kind='bar')
plt.title('Pclass=2 sex = female')
plt.ylabel('Number of people')
plt.xlabel('1=survived 0=unservived')

plt.subplot2grid((2,3),(1,0))
raw_train_data['Survived '][raw_train_data['Pclass ']==2][raw_train_data['Sex ']=='male '].value_counts().plot(kind='bar')
plt.title('Pclass=2 sex = male')
plt.ylabel('Number of people')
plt.xlabel('1=survived 0=unservived')

plt.subplot2grid((2,3),(1,1))
raw_train_data['Survived '][raw_train_data['Pclass ']==3][raw_train_data['Sex ']=='female '].value_counts().plot(kind='bar')
plt.title('Pclass=3 sex = female')
plt.ylabel('Number of people')
plt.xlabel('1=survived 0=unservived')

plt.subplot2grid((2,3),(1,2))
raw_train_data['Survived '][raw_train_data['Pclass ']==3][raw_train_data['Sex ']=='male '].value_counts().plot(kind='bar')
plt.title('Pclass=3 sex = female')
plt.ylabel('Number of people')
plt.xlabel('1=survived 0=unservived')
plt.show()

#登陆港口与获救与否分布情况
fig = plt.Figure(figsize=(5,5))
Survived0_embarked = raw_train_data['Embarked'][raw_train_data['Survived ']==0].value_counts()
Survived1_embarked = raw_train_data['Embarked'][raw_train_data['Survived ']==1].value_counts()
df_embarked = pd.DataFrame({'Survived':Survived1_embarked,'Unsurvived':Survived0_embarked})
plt.plot(kind='bar')
plt.show()

# 有船舱号和没船舱号与获救与否的关系
fig = plt.Figure(figsize=(5,5))
Survived_cabin = raw_train_data['Survived '][pd.notnull(raw_train_data['Cabin '])].value_counts()
Survived_Nocabin = raw_train_data['Survived '][pd.isnull(raw_train_data['Cabin '])].value_counts()
df_cabin = pd.DataFrame({'Hascabin':Survived_cabin,'NoCabin':Survived_Nocabin})
df_cabin.plot(kind='bar')
plt.show()

# 特征处理
# 因为原始的数据中，很多特征里面的内容是不完全的，需要进一步处理才行，比如有的特征缺失某个值，有的特征非常的长尾等等。
# age数据处理
#age数据我们理解为是一个category类型的数据，并且其中包含有一些nan值。
#对于nan值的处理方法比较多，这里选择填充的办法进行处理。
#但是，这里的age虽然表示category的特征，但本身是数值型的，取均值填充什么的不太好，那么我们通过拟合的方式选择：
#1、将数据按照有年龄（不是nan）和没有年龄（是nan）分成两部分A，B。
#2、对于A部分，我们将年龄作为label，其他值作为样本，训练一个model出来。这里选择使用randomforest。
#3、对于B部分，我们将其认为是测试数据，利用训练好的模型，取计算每个样本的年龄。
#4、这里，我们把它看看做是一个regressor问题。

#这里我们把训练数据和测试数据一起处理了
know_age = all_data_del[pd.notnull(all_data_del['Age '])]
unknow_age = all_data_del[pd.isnull(all_data_del['Age '])]

# 把这行删了，因为这行有fare为nan
know_age = know_age.drop(152)

age_train_y = know_age['Age ']
age_train_x=know_age[['Fare ', 'Parch ', 'SibSp ', 'Pclass ']]


# 构建模型开始训练，预测年龄，应用随机分类回归算法进行预测
model = RandomForestRegressor(random_state=0,n_estimators=200,n_jobs=1)
model.fit(age_train_x,age_train_y)
predicted_age = model.predict(unknow_age[['Fare ', 'Parch ', 'SibSp ', 'Pclass ']])
all_data_nonull=all_data_del.copy()
all_data_nonull.loc[all_data_del['Age '].isnull(),'Age '] = predicted_age

# 把年龄做成分段的形式
result={}
step=10
for i in range(10,80,step):
    age_results =train_y.loc[all_data_nonull[(all_data_nonull['Age ']<=i) & (all_data_nonull['Age ']>i-step)]['PassengerId ']].describe()
    age_temp=age_results.loc[age_results.index=='mean','Survived ']
    result[i]=age_temp.values[0]

fig = plt.Figure(figsize=(5,5))
plt.plot(list(result.keys()),list(result.values()))
plt.show()

'''从上面这个图可以得出的结论有：
1. 如果年龄在0~10岁的话，获救概率在0.6以上
2. 10~20和20~30岁获救的概率差不多，在0.3~0.35之间
3. 30~60岁之间，获救概率在0.4左右，
4. 在60岁以上，获救概率又下来了，只有0.3不到。
因此，我们把年龄再分为几段：
 1：0~10   child
 2：10~30  young man
 3: 30~60  man
 4: 60~    elder
'''
all_data_age=all_data_nonull.copy()
all_data_age.loc[all_data_nonull['Age '] <=10,'Age '] = 'child'
all_data_age.loc[(all_data_nonull['Age '] >10)&(all_data_nonull['Age ']<=30),'Age '] = 'young'
all_data_age.loc[(all_data_nonull['Age '] >30)&(all_data_nonull['Age ']<=60),'Age '] = 'man'
all_data_age.loc[all_data_nonull['Age '] >60,'Age '] = 'elder'

all_data_nonull=all_data_age

# 对category特征做one-hot编码
# 我们首先需要对cabin做一个简单处理
# 我们对有记录的cabin写为yes，没有的写为no

all_data_nonull.loc[all_data_del['Cabin '].notnull(),'Cabin '] = 'yes'
all_data_nonull.loc[all_data_del['Cabin '].isnull(),'Cabin '] = 'no'

# 使用get_dummies进行one-hot编码
dummies_Cabin = pd.get_dummies(all_data_nonull['Cabin '],prefix='Cabin ')
dummies_Embarked = pd.get_dummies(all_data_nonull['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(all_data_nonull['Sex '],prefix='Sex ')
dummies_plcass = pd.get_dummies(all_data_nonull['Pclass '],prefix='Pclass ')
dummies_age = pd.get_dummies(all_data_nonull['Age '],prefix='Age ')

# 拼接数据
concat_data=pd.concat([all_data_nonull,dummies_Cabin,dummies_Embarked,dummies_plcass,dummies_Sex,dummies_age],axis=1)
concat_data.drop(['Age ','Pclass ','Sex ','Ticket ','Cabin ','Embarked'],axis=1,inplace=True)






# 对numerical数据调用sklearn的包做归一化
scaler = preprocessing.StandardScaler()
#age_scaler_parm = scaler.fit(concat_data['Age '].values.reshape(-1,1))
#concat_data['Age '] = scaler.fit_transform(concat_data['Age '].values.reshape(-1,1),age_scaler_parm)

concat_data.loc[concat_data['Fare '].isnull(),'Fare '] = concat_data['Fare '].mean()
fare_scaler_parm = scaler.fit(concat_data['Fare '].values.reshape(-1,1))
concat_data['Fare ']=scaler.fit_transform(concat_data['Fare '].values.reshape(-1,1),fare_scaler_parm)

#应用逻辑回归算法构建模型
train_data_set_x=concat_data.iloc[:raw_train_data.shape[0]]
test_data_set_x=concat_data.iloc[raw_train_data.shape[0]:]
X_train, X_test, y_train, y_test = train_test_split(train_data_set_x,train_y['Survived '] , test_size=0.2,random_state=1)

LR_model = LogisticRegressionCV(Cs=10,cv=10,scoring='accuracy')
LR_model.fit(X_train,y_train)
LR_predicted_result = LR_model.predict(X_test)
print('LR prediction result')
print ('查准率、查全率、F1值：')
print (classification_report(y_test, LR_predicted_result, target_names=None))

# 应用支持向量机算法构建模型
c_list=np.linspace(3,5,10)
parameters={'kernel':('linear', 'poly', 'rbf', 'sigmoid'),'C':c_list}
SVC_model = SVC(gamma='auto')
clf=GridSearchCV(SVC_model,parameters,scoring='accuracy')
clf.fit(X_train,y_train)
print(clf.best_score_)
best_svm = clf.best_estimator_
best_svm.fit(X_train,y_train)
SVM_predicted_result = best_svm.predict(X_test)
print('SVM prediction result')
print ('查准率、查全率、F1值：')
print (classification_report(y_test, SVM_predicted_result, target_names=None))

#应用随机森林算法构建模型
n_estimator_list=np.linspace(10,100,10).astype(int)
max_depth_list=np.linspace(1,20,20).astype(int)
max_features_list=('auto','sqrt','log2',None)
parameter_rf={'n_estimators':n_estimator_list,'max_features':max_features_list,'max_depth':max_depth_list}
model_rf=RandomForestClassifier()
clf_rf=GridSearchCV(model_rf,parameter_rf,scoring='accuracy')
clf_rf.fit(X_train,y_train)
print(clf_rf.best_score_)
best_rf = clf_rf.best_estimator_
RF_predicted_result = best_rf.predict(X_test)
print('RF prediction result')
print ('查准率、查全率、F1值：')
print (classification_report(y_test, SVM_predicted_result, target_names=None))

#模型融合
model_vc=VotingClassifier(estimators=[('lg',LR_model),('svn',best_svm),('rf',best_rf)],voting='hard')
model_vc.fit(X_train,y_train)
predicted_final = model_vc.predict(X_test)
print('VC prediction result')
print ('查准率、查全率、F1值：')
print (classification_report(y_test, SVM_predicted_result, target_names=None))